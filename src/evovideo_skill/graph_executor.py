from __future__ import annotations

import re
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from evovideo_skill.evaluators import EvaluatorSuite
from evovideo_skill.graph_skill import GraphEdge, GraphNode, ToolPathGraph, VideoGenerationState
from evovideo_skill.models import VideoArtifact, VideoPlan, VideoTask
from evovideo_skill.tools import ToolExecutionContext, ToolRegistry
from evovideo_skill.weighted_tool_graph import H3_LOCAL_EVALUATION_PROTOCOL, h3_local_seed_applied


class GraphExecutionError(RuntimeError):
    pass


class GraphToolExecutionError(GraphExecutionError):
    """An execution failure attributed to one concrete tool node."""

    def __init__(
        self,
        tool_name: str,
        node_id: str,
        cause: Exception,
        executed_tools: list[str] | None = None,
    ):
        self.tool_name = tool_name
        self.node_id = node_id
        self.cause_type = type(cause).__name__
        self.executed_tools = list(executed_tools or [])
        super().__init__(
            f"tool {tool_name!r} failed at node {node_id!r}: "
            f"{self.cause_type}: {cause}"
        )


def _positive_duration(value: Any) -> float | None:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    return duration if duration > 0 else None


def _primary_video_duration(media: dict[str, Any]) -> tuple[float, str]:
    """Prefer video timing over container timing, which can include an AAC tail."""
    for stream in media.get("streams", []):
        if stream.get("codec_type") != "video" or stream.get("disposition", {}).get("attached_pic"):
            continue
        duration = _positive_duration(stream.get("duration"))
        if duration is not None:
            return duration, "video_stream"
        duration_ts = _positive_duration(stream.get("duration_ts"))
        try:
            time_base = float(Fraction(str(stream.get("time_base"))))
        except (TypeError, ValueError, ZeroDivisionError):
            time_base = 0.0
        if duration_ts is not None and time_base > 0:
            return duration_ts * time_base, "video_stream_timestamps"
    duration = _positive_duration(media.get("format", {}).get("duration"))
    if duration is None:
        raise GraphExecutionError("H3 output has no positive video or container duration")
    return duration, "container"


def _h3_duration_tolerance(target_seconds: float) -> float:
    # Native diffusion frame counts and audio muxing quantize nominal durations.
    return max(0.5, min(0.75, float(target_seconds) * 0.05))


@dataclass
class GraphExecutionResult:
    artifact: VideoArtifact
    state: VideoGenerationState
    node_artifacts: dict[str, VideoArtifact]
    executed_tools: list[str]


class ArtifactPassingGraphExecutor:
    """Execute a tool graph as a DAG and pass artifacts along graph edges."""

    def __init__(self, tools: ToolRegistry, evaluators: EvaluatorSuite, artifact_augmenter: Any | None = None):
        self.tools = tools
        self.evaluators = evaluators
        self.artifact_augmenter = artifact_augmenter

    def execute(self, task: VideoTask, plan: VideoPlan, graph: ToolPathGraph,
                node_cache: Any | None = None, observer: Any | None = None) -> GraphExecutionResult:
        self.validate_graph(graph)
        ordered = self._topological_order(graph)
        state = VideoGenerationState(
            task.task_id,
            task.prompt,
            graph.graph_id,
            budget_used=graph.estimated_cost(),
        )
        artifacts: dict[str, VideoArtifact] = {}
        executed: list[str] = []
        latest: VideoArtifact | None = None
        incoming = self._incoming_edges(graph)
        node_evidence: dict[str, dict[str, Any]] = {}

        for node in ordered:
            edges = incoming.get(node.node_id, [])
            active_edges = [edge for edge in edges if self._condition_passes(edge, graph, state)]
            if edges and not active_edges:
                state.record(node, node.node_type, "skipped", {"reason": "incoming_conditions_false"})
                if observer is not None:
                    observer.node(node.node_id, "skipped", reason="incoming_conditions_false")
                continue
            inputs = {
                edge.source: artifacts[edge.source]
                for edge in active_edges
                if edge.source in artifacts
            }
            if node.node_type == "trigger":
                state.record(node, "trigger", "completed")
                if observer is not None:
                    observer.node(node.node_id, "completed")
                continue
            if node.node_type == "verifier":
                if observer is not None:
                    observer.node(node.node_id, "running")
                target = next(reversed(inputs.values()), latest)
                if target is None:
                    state.record(node, "verifier", "skipped", {"reason": "no_input_artifact"})
                    if observer is not None:
                        observer.node(node.node_id, "skipped", reason="no_input_artifact")
                    continue
                if self.artifact_augmenter is not None and "vlm_evaluation" not in target.metadata:
                    target = self.artifact_augmenter.augment(task, target)
                artifacts[node.node_id] = target
                metric = self._metric(node.name, task, target)
                state.verifier_scores[node.name] = metric.score
                state.record(
                    node,
                    "verifier",
                    "passed" if metric.passed else "failed",
                    {"score": metric.score, "threshold": metric.threshold, "evidence": metric.evidence},
                )
                if not metric.passed:
                    state.failures.append(metric.name)
                if observer is not None:
                    observer.node(node.node_id, "completed", metric={"name": metric.name,
                        "score": metric.score, "passed": metric.passed})
                continue
            if node.node_type != "tool":
                state.record(node, node.node_type, "skipped", {"reason": "unsupported_node_type"})
                if observer is not None:
                    observer.node(node.node_id, "skipped", reason="unsupported_node_type")
                continue
            if not self.tools.has(node.name):
                raise GraphExecutionError(
                    f"graph {graph.skill_name} requires unavailable tool {node.name!r} at node {node.node_id!r}"
                )

            context = ToolExecutionContext(
                node_id=node.node_id,
                node_config=dict(node.config),
                input_artifacts=inputs,
                all_artifacts=dict(artifacts),
                state=state,
            )
            state.record(node, "tool", "running", {"input_artifact_ids": [a.artifact_id for a in inputs.values()]})
            if observer is not None:
                observer.node(node.node_id, "running", input_nodes=list(inputs),
                              input_artifact_ids=[a.artifact_id for a in inputs.values()])
            started = time.monotonic()
            cached = None
            if node_cache is not None:
                cached = node_cache.lookup(task, plan, node, inputs, self.tools.spec(node.name))
                node_cache.before_node(task, node, cached is not None)
            try:
                spec = self.tools.spec(node.name)
                self._validate_runtime_inputs(node, spec, inputs)
                if node.name == "failed_segment_localizer" and self.artifact_augmenter is not None and cached is None:
                    for source_id, source in list(inputs.items()):
                        if "vlm_evaluation" not in source.metadata:
                            verified = self.artifact_augmenter.augment(task, source)
                            inputs[source_id] = artifacts[source_id] = verified
                            context.all_artifacts[source_id] = verified
                artifact = cached if cached is not None else self.tools.get(node.name).run_with_context(task, plan, context)
                artifact.metadata.setdefault("artifact_type", spec.output_type)
                artifact.metadata.setdefault("tool_spec", spec.to_dict())
                if "artifact_contract" not in artifact.metadata:
                    from evovideo_skill.artifact_contracts import output_contract

                    artifact.metadata["artifact_contract"] = output_contract(spec).to_dict()
                if (
                    inputs
                    and spec.consumes_upstream
                    and spec.backend != "builtin"
                    and artifact.metadata.get("upstream_conditioning_consumed") is not True
                ):
                    raise GraphExecutionError(
                        f"tool {node.name!r} declared consumes_upstream=True but its real adapter "
                        "did not confirm upstream conditioning"
                    )
                if node_cache is not None and cached is None:
                    node_cache.store(task, plan, node, inputs, spec, artifact)
            except GraphToolExecutionError:
                raise
            except Exception as exc:
                state.record(
                    node,
                    "tool",
                    "failed",
                    {"error_type": type(exc).__name__, "error": str(exc)[-2000:]},
                )
                raise GraphToolExecutionError(
                    node.name,
                    node.node_id,
                    exc,
                    executed_tools=executed,
                ) from exc
            artifacts[node.node_id] = artifact
            node_evidence[node.node_id] = {
                "tool": node.name,
                "config": dict(node.config),
                "artifact_id": artifact.artifact_id,
                "input_nodes": list(inputs),
                "input_artifact_ids": [a.artifact_id for a in inputs.values()],
                "output_contract": artifact.metadata.get("artifact_contract"),
                "wall_seconds": time.monotonic() - started,
                "cache_hit": cached is not None,
                "observations": {
                    key: artifact.metadata[key]
                    for key in ("local_video_path", "local_image_path", "local_audio_path",
                                "sampled_frame_paths", "duration_seconds", "has_audio",
                                "upstream_conditioning_consumed", "h3_mode", "h3_request_hash")
                    if key in artifact.metadata
                },
                "semantic_postcondition": "unknown",
            }
            if observer is not None:
                from evovideo_skill.conditioning_workspace import artifact_view
                node_evidence[node.node_id]["artifact_state"] = artifact_view(artifact)
                observer.completed(node.node_id, artifact, node_evidence[node.node_id])
            latest = artifact
            executed.append(node.name)
            state.artifacts.append(artifact.artifact_id)
            state.record(node, "tool", "completed", {"artifact_id": artifact.artifact_id})

        if latest is None:
            raise GraphExecutionError(f"graph {graph.skill_name} produced no artifact")
        latest = self._select_output_artifact(graph, artifacts, latest)
        h3_backends = {self.tools.spec(name).backend for name in executed} & {"minimax-h3", "local-h3"}
        if any(artifact.metadata.get("provider") == "local-h3" for artifact in artifacts.values()):
            h3_backends.add("local-h3")
        if h3_backends:
            from evovideo_skill.h3_api import probe_media

            path = latest.metadata.get("local_video_path")
            if not path:
                raise GraphExecutionError("H3 path must end in a materialized video")
            media = probe_media(path)
            actual_duration, duration_source = _primary_video_duration(media)
            duration_tolerance = _h3_duration_tolerance(task.duration_seconds)
            if abs(actual_duration - task.duration_seconds) > duration_tolerance:
                raise GraphExecutionError(
                    f"H3 output duration {actual_duration:.3f}s does not match the task's "
                    f"{task.duration_seconds}s within {duration_tolerance:.3f}s "
                    f"({duration_source}); ensure per-call durations sum to the final path duration"
                )
            has_audio = any(stream.get("codec_type") == "audio" for stream in media.get("streams", []))
            if task.metadata.get("h3_audio_criteria") and not has_audio:
                raise GraphExecutionError("H3 task requires audio evidence but the final path dropped its soundtrack")
            latest.metadata.update(
                duration_seconds=actual_duration,
                duration_source=duration_source,
                duration_tolerance_seconds=duration_tolerance,
                has_audio=has_audio,
            )
            if "minimax-h3" in h3_backends:
                latest.metadata.update(provider_seed_control=False, generation_seed_applied=False)
            else:
                label = task.metadata.get("replicate_label", task.metadata.get("evaluation_seed", task.metadata.get("generation_seed")))
                generated = [artifact for node_id, artifact in artifacts.items()
                             if graph.node(node_id).node_type == "tool"
                             and (self.tools.spec(graph.node(node_id).name).backend == "local-h3"
                                  or artifact.metadata.get("provider") == "local-h3")
                             and self.tools.spec(graph.node(node_id).name).output_type == "video"]
                controlled = bool(generated) and all(h3_local_seed_applied(item.metadata, label) for item in generated)
                latest.metadata.update(provider="local-h3", provider_seed_control=True,
                                       generation_seed_applied=controlled, replicate_label=label,
                                       evaluation_protocol=H3_LOCAL_EVALUATION_PROTOCOL)
                if controlled:
                    latest.metadata["generation_seed"] = label
        latest.tool_chain = list(executed)
        latest.metadata.update(
            {
                "tool_path_graph_id": graph.graph_id,
                "tool_path_skill": graph.skill_name,
                "executed_tool_nodes": list(executed),
                "artifact_lineage": {
                    node_id: {
                        "artifact_id": artifact.artifact_id,
                        "producer": graph.node(node_id).name,
                    }
                    for node_id, artifact in artifacts.items()
                },
                "state_graph": state.to_dict(),
                "node_evidence": node_evidence,
            }
        )
        if observer is not None:
            observer.generated(latest)
        return GraphExecutionResult(latest, state, artifacts, executed)

    def validate_graph(self, graph: ToolPathGraph) -> None:
        """Validate availability, types, topology, and final video semantics."""
        self._validate_tool_contracts(graph)
        self._topological_order(graph)
        incoming = self._incoming_edges(graph)
        outgoing = self._outgoing_edges(graph)
        for node in graph.nodes:
            if node.node_type != "tool":
                continue
            spec = self.tools.spec(node.name)
            producer_inputs = [
                edge for edge in incoming.get(node.node_id, [])
                if graph.node(edge.source).node_type in {"tool", "verifier"}
            ]
            if spec.consumes_upstream and not producer_inputs:
                raise GraphExecutionError(
                    f"tool {node.name!r} consumes upstream artifacts but node {node.node_id!r} "
                    "has no tool/verifier producer"
                )
        terminal_tools = [
            node for node in graph.nodes
            if node.node_type == "tool" and not outgoing.get(node.node_id)
        ]
        video_tools = [
            node for node in graph.nodes
            if node.node_type == "tool" and self.tools.spec(node.name).output_type == "video"
        ]
        if video_tools and not any(self.tools.spec(node.name).output_type == "video" for node in terminal_tools):
            raise GraphExecutionError(
                f"graph {graph.skill_name} produces video but has no terminal video-producing tool; "
                "analysis-only branches cannot be scored as edited video"
            )

    def _validate_tool_contracts(self, graph: ToolPathGraph) -> None:
        for node in graph.nodes:
            if node.node_type == "tool" and node.name not in self.tools.available_names():
                raise GraphExecutionError(
                    f"graph {graph.skill_name} requires unavailable or unverified tool {node.name!r}"
                )
        incoming = self._incoming_edges(graph)
        for edge in graph.edges:
            source = graph.node(edge.source)
            target = graph.node(edge.target)
            if target.node_type != "tool" or source.node_type not in {"tool", "verifier"}:
                continue
            producers = self._artifact_producers(graph, source, incoming, set())
            if not producers:
                raise GraphExecutionError(
                    f"invalid artifact edge {edge.source}->{edge.target}: verifier has no upstream tool artifact"
                )
            for producer in producers:
                produced_contract = (
                    producer.config.get("target_contract")
                    if producer.config.get("contract_alignment")
                    else None
                )
                check = self.tools.connection_contract_check(producer.name, target.name, produced_contract)
                compatible, reason = check.compatible, check.reason
                if not compatible:
                    raise GraphExecutionError(
                        f"invalid artifact edge {edge.source}->{edge.target} "
                        f"(forwarded from {producer.node_id}): {reason}"
                    )

    def _artifact_producers(
        self,
        graph: ToolPathGraph,
        node: GraphNode,
        incoming: dict[str, list[GraphEdge]],
        visiting: set[str],
    ) -> list[GraphNode]:
        """Resolve the real typed producer through transparent verifier nodes."""
        if node.node_type == "tool":
            return [node]
        if node.node_type != "verifier" or node.node_id in visiting:
            return []
        visiting = {*visiting, node.node_id}
        producers: list[GraphNode] = []
        for upstream in incoming.get(node.node_id, []):
            producers.extend(
                self._artifact_producers(graph, graph.node(upstream.source), incoming, visiting)
            )
        return producers

    @staticmethod
    def _validate_runtime_inputs(node: GraphNode, spec: Any, inputs: dict[str, VideoArtifact]) -> None:
        required = set(spec.input_types)
        if not inputs:
            if spec.consumes_upstream:
                raise GraphExecutionError(
                    f"tool {node.name!r} requires an upstream artifact but received none"
                )
            return
        if not required:
            raise GraphExecutionError(
                f"source tool {node.name!r} received upstream artifacts but its adapter declares no artifact inputs"
            )
        if "any" in required:
            return
        actual = {str(artifact.metadata.get("artifact_type", "video")) for artifact in inputs.values()}
        if not actual.intersection(required) and not ({"identity_reference", "keyframes"} & actual and "image" in required):
            raise GraphExecutionError(
                f"tool {node.name!r} requires artifact types {sorted(required)}, received {sorted(actual)}"
            )

    def _metric(self, name: str, task: VideoTask, artifact: VideoArtifact):
        report = self.evaluators.evaluate(task, artifact)
        for metric in report.metrics:
            if metric.name == name:
                return metric
        raise GraphExecutionError(f"no evaluator registered for verifier node {name!r}")

    @staticmethod
    def _incoming_edges(graph: ToolPathGraph) -> dict[str, list[GraphEdge]]:
        incoming: dict[str, list[GraphEdge]] = {node.node_id: [] for node in graph.nodes}
        for edge in graph.edges:
            incoming[edge.target].append(edge)
        return incoming

    @staticmethod
    def _outgoing_edges(graph: ToolPathGraph) -> dict[str, list[GraphEdge]]:
        outgoing: dict[str, list[GraphEdge]] = {node.node_id: [] for node in graph.nodes}
        for edge in graph.edges:
            outgoing[edge.source].append(edge)
        return outgoing

    def _select_output_artifact(
        self,
        graph: ToolPathGraph,
        artifacts: dict[str, VideoArtifact],
        latest: VideoArtifact,
    ) -> VideoArtifact:
        outgoing = self._outgoing_edges(graph)
        terminal = [
            artifacts[node.node_id]
            for node in graph.nodes
            if node.node_id in artifacts and not outgoing.get(node.node_id)
        ]
        terminal_videos = [
            artifact for artifact in terminal
            if artifact.metadata.get("artifact_type") == "video"
        ]
        if terminal_videos:
            return terminal_videos[-1]
        produced_videos = [
            artifact for artifact in artifacts.values()
            if artifact.metadata.get("artifact_type") == "video"
        ]
        if produced_videos and any(
            node.node_type == "tool"
            and not outgoing.get(node.node_id)
            and self.tools.spec(node.name).output_type == "video"
            for node in graph.nodes
        ):
            # A conditional repair sink may be skipped when its verifier passes.
            return produced_videos[-1]
        has_video = any(
            artifact.metadata.get("artifact_type") == "video"
            for artifact in artifacts.values()
        )
        if has_video:
            raise GraphExecutionError(
                f"graph {graph.skill_name} ended in a non-video artifact; add a real editor/generator sink"
            )
        return latest

    def _condition_passes(self, edge: GraphEdge, graph: ToolPathGraph, state: VideoGenerationState) -> bool:
        condition = (edge.condition or "always").strip().lower()
        if condition in {"", "always", "identity_reference_ready", "motif_sequence"}:
            return True
        match = re.fullmatch(r"([a-z0-9_]+)\s*(<=|>=|<|>)\s*(threshold|[0-9.]+)", condition)
        if not match:
            return True
        key, operator, raw_threshold = match.groups()
        aliases = {
            "action_score": "prompt_action_alignment",
            "identity_score": "identity_consistency",
            "background_score": "background_preservation",
        }
        score = state.verifier_scores.get(aliases.get(key, key))
        if score is None:
            return False
        source = graph.node(edge.source)
        threshold = float(source.config.get("threshold", 0.5)) if raw_threshold == "threshold" else float(raw_threshold)
        return {
            "<": score < threshold,
            "<=": score <= threshold,
            ">": score > threshold,
            ">=": score >= threshold,
        }[operator]

    @staticmethod
    def _topological_order(graph: ToolPathGraph) -> list[GraphNode]:
        nodes = {node.node_id: node for node in graph.nodes}
        if len(nodes) != len(graph.nodes):
            raise GraphExecutionError(f"graph {graph.skill_name} contains duplicate node ids")
        indegree = {node_id: 0 for node_id in nodes}
        outgoing: dict[str, list[str]] = {node_id: [] for node_id in nodes}
        for edge in graph.edges:
            if edge.source not in nodes or edge.target not in nodes:
                raise GraphExecutionError(
                    f"graph {graph.skill_name} has dangling edge {edge.edge_id}: {edge.source}->{edge.target}"
                )
            outgoing[edge.source].append(edge.target)
            indegree[edge.target] += 1
        order_index = {node.node_id: index for index, node in enumerate(graph.nodes)}
        ready = sorted((node_id for node_id, degree in indegree.items() if degree == 0), key=order_index.get)
        ordered: list[GraphNode] = []
        while ready:
            node_id = ready.pop(0)
            ordered.append(nodes[node_id])
            for target in outgoing[node_id]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
                    ready.sort(key=order_index.get)
        if len(ordered) != len(nodes):
            raise GraphExecutionError(f"graph {graph.skill_name} contains a cycle")
        return ordered
