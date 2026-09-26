from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from evovideo_skill.diagnosis import FailureDiagnoser
from evovideo_skill.h3_graph_contracts import validate_h3_reference_flow, validate_h3_composition_duration
from evovideo_skill.evaluators import EvaluatorSuite
from evovideo_skill.graph_skill import (
    BoundedGraphEdit,
    ExperienceRecord,
    GraphEdge,
    GraphNode,
    GraphSkillMemory,
    ToolPathGraph,
    VideoGenerationState,
    graph_score_mean,
    graph_score_stability,
)
from evovideo_skill.graph_executor import ArtifactPassingGraphExecutor
from evovideo_skill.models import EvaluationReport, FailureReport, FailureType, SkillValidationReport, TaskMode, VideoArtifact, VideoPlan, VideoTask
from evovideo_skill.planning import Planner
from evovideo_skill.reward_router import TaskConditionedRewardRouter, TaskReward
from evovideo_skill.skill_memory import SkillMemory
from evovideo_skill.tools import ToolRegistry
from evovideo_skill.tool_onboarding import CapabilityRequest
from evovideo_skill.vlm_evaluator import VLMEvidenceAugmenter
from evovideo_skill.weighted_tool_graph import (
    H3_COMPARISON_PROTOCOL, H3_LOCAL_COMPARISON_PROTOCOL, H3_LOCAL_EVALUATION_PROTOCOL,
    graph_behavior_fingerprint, h3_local_active, h3_local_seed_applied, observed_tool_edges,
)


H3_GENERATORS = {"h3_t2va", "h3_fl2va", "h3_ref2va"}


def h3_registry_active(available_tools: set[str]) -> bool:
    """The legacy mock alias alone must never switch a Wan registry to H3."""
    return bool(H3_GENERATORS & available_tools)


def h3_run_should_stop(error: Exception) -> bool:
    """Operational ledger stops must not become tool-quality evidence."""
    cause_types = {type(error).__name__, getattr(error, "cause_type", "")}
    message = str(error).lower()
    return bool(cause_types & {"H3BudgetExceeded", "H3SubmissionUnknown", "H3PollingInterrupted", "VerifierEvidenceUnavailable"}) or (
        "h3 api call budget exhausted" in message or "submission_unknown" in message
        or "h3 submission outcome unknown" in message
    )


def validate_h3_node_configs(graph: ToolPathGraph, available_tools: set[str], *, local: bool = False) -> None:
    """Check native configuration without guessing references or parent ordering.

    Materialized reference contents and API-specific limits remain runtime checks.
    """
    if not h3_registry_active(available_tools):
        return
    seed_error = ("local-h3 seeds are fixed by task metadata, not node config" if local
                  else "H3 API does not support seeds")
    roles = {"image": {"reference_image", "first_frame", "last_frame"},
             "video": {"reference_video"}, "audio": {"reference_audio"}}

    def ordered_ids(value: Any, label: str, *, allow_repeats: bool = False) -> list[str]:
        if not isinstance(value, list) or not value or any(
            not isinstance(item, str) or not item.strip() for item in value
        ) or (not allow_repeats and len(set(value)) != len(value)):
            raise ValueError(f"{label} must be a nonempty ordered list" + (" of unique ids" if not allow_repeats else " of ids"))
        return value

    def number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    for node in graph.nodes:
        if node.node_type != "tool":
            continue
        config = node.config
        label = f"{node.node_id} ({node.name})"
        if node.name == "h3_reference_pack" and "semantic_role" in config:
            # Only a single explicit binding has an unambiguous destination.
            bindings = config.get("bindings")
            semantic = config["semantic_role"]
            if (not isinstance(semantic, str) or not semantic.strip()
                    or not isinstance(bindings, list) or len(bindings) != 1
                    or not isinstance(bindings[0], dict)
                    or bindings[0].get("semantic_role", semantic) != semantic):
                raise ValueError(f"{label}: move semantic_role into each explicit binding; ambiguous pack config")
            node.config = config = {k: v for k, v in config.items() if k != "semantic_role"}
            config["bindings"] = [{**bindings[0], "semantic_role": semantic}]
            repairs = graph.stats.setdefault("automatic_repairs", [])
            note = f"{node.node_id}: moved semantic_role into its single explicit binding"
            if note not in repairs:
                repairs.append(note)
        common = {"cost", "backend", "provenance", "shared_node"}
        allowed_config = {
            "mock_text_to_video": {"prompt", "duration_seconds", "shot_index", "reference_ids", "ratio"},
            "h3_t2va": {"prompt", "duration_seconds", "shot_index", "reference_ids", "ratio"},
            "h3_fl2va": {"prompt", "duration_seconds", "shot_index", "reference_ids", "ratio"},
            "h3_ref2va": {"prompt", "duration_seconds", "shot_index", "reference_ids", "ratio"},
            "h3_reference_bank": set(),
            "h3_reference_select": {"reference_ids", "roles", "semantic_roles"},
            "h3_reference_pack": {"bindings"},
            "h3_frame_extract": {"position", "time_seconds", "role"},
            "h3_audio_extract": {"start_seconds", "end_seconds"},
            "h3_reference_trim": {"reference_id", "start_seconds", "end_seconds"},
            "h3_av_concat": {"source_nodes"},
        }
        if node.name in allowed_config:
            if node.name in H3_GENERATORS | {"mock_text_to_video"}:
                allowed_config[node.name] |= {"conditioning_strategy", "prompt_task_hashes"}
            unsupported = set(config) - allowed_config[node.name] - common
            if unsupported:
                if unsupported & {"seed", "generation_seed", "random_seed"}:
                    raise ValueError(f"{label}: {seed_error}")
                raise ValueError(f"{label}: unsupported H3 config fields {sorted(unsupported)}")
        parents = {edge.source for edge in graph.edges if edge.target == node.node_id
                   and graph.node(edge.source).node_type in {"tool", "verifier"}}
        if node.name == "bridge_materialize_image":
            pending, ancestors = list(parents), set()
            while pending:
                source = pending.pop()
                if source in ancestors:
                    continue
                ancestors.add(source)
                pending.extend(edge.source for edge in graph.edges if edge.target == source)
            tools = {graph.node(source).name for source in ancestors if graph.node(source).node_type == "tool"}
            if tools <= {"temporal_decomposer", "keyframe_generator", "character_sheet_generator"}:
                raise ValueError(f"{label}: symbolic-only keyframes cannot supply a physical H3 anchor; "
                                 "generate a real draft and use h3_frame_extract, or use real task references")
        if node.name in H3_GENERATORS | {"mock_text_to_video"}:
            if any(key in config for key in ("seed", "generation_seed", "random_seed")):
                raise ValueError(f"{label}: {seed_error}")
            duration = config.get("duration_seconds", 4)
            if type(duration) is not int or not 4 <= duration <= 15:
                raise ValueError(f"{label}: duration_seconds must be an integer within 4..15")
            if "shot_index" in config and (type(config["shot_index"]) is not int or config["shot_index"] < 0):
                raise ValueError(f"{label}: shot_index must be a nonnegative integer")
            if "prompt" in config and not isinstance(config["prompt"], str):
                raise ValueError(f"{label}: prompt must be a string")
            from evovideo_skill.h3_prompt_binding import STRATEGIES

            if "conditioning_strategy" in config and config["conditioning_strategy"] not in STRATEGIES:
                raise ValueError(f"{label}: unknown conditioning_strategy")
            if "prompt_task_hashes" in config:
                ordered_ids(config["prompt_task_hashes"], f"{label}.prompt_task_hashes")
            if "reference_ids" in config:
                ordered_ids(config["reference_ids"], f"{label}.reference_ids")
            if node.name in {"h3_fl2va", "h3_ref2va"} and not parents:
                raise ValueError(f"{label}: explicit reference artifact input required")
        elif node.name == "h3_reference_select":
            ids = ordered_ids(config.get("reference_ids"), f"{label}.reference_ids")
            overrides = config.get("roles", {})
            if not isinstance(overrides, dict) or any(
                key not in ids or value not in set().union(*roles.values())
                for key, value in overrides.items()
            ):
                raise ValueError(f"{label}: roles must map selected ids to API roles")
            semantic_roles = config.get("semantic_roles", {})
            if not isinstance(semantic_roles, dict) or any(
                key not in ids or not isinstance(value, str) or not value.strip()
                for key, value in semantic_roles.items()
            ):
                raise ValueError(f"{label}: semantic_roles must map selected ids to descriptions")
        elif node.name == "h3_frame_extract":
            if config.get("position") not in {"first", "last", "time"}:
                raise ValueError(f"{label}: position must be first, last, or time")
            if config.get("position") == "time" and (
                not number(config.get("time_seconds")) or not 0 <= config["time_seconds"] < float("inf")
            ):
                raise ValueError(f"{label}: time_seconds must be finite and nonnegative")
            if config.get("position") != "time" and "time_seconds" in config:
                raise ValueError(f"{label}: time_seconds is only valid for position=time")
            if "role" in config and config["role"] not in roles["image"]:
                raise ValueError(f"{label}: invalid extracted image role")
        elif node.name == "h3_reference_pack":
            bindings = config.get("bindings")
            if not isinstance(bindings, list) or not bindings:
                raise ValueError(f"{label}: explicit ordered bindings required")
            for binding in bindings:
                if not isinstance(binding, dict) or binding.get("source") not in parents:
                    raise ValueError(f"{label}: every binding.source must be an explicit parent node id")
                if binding.get("role") not in roles.get(binding.get("kind"), set()):
                    raise ValueError(f"{label}: binding kind and API role are incompatible")
                producer = graph.node(binding["source"]).name
                native_kind = ({"h3_frame_extract": "image", "h3_audio_extract": "audio",
                    "h3_av_concat": "video", **{name: "video" for name in H3_GENERATORS}}).get(producer)
                if native_kind is not None and binding["kind"] != native_kind:
                    raise ValueError(f"{label}: binding kind {binding['kind']} does not match producer {producer} ({native_kind})")
        elif node.name == "h3_reference_trim":
            start, end = config.get("start_seconds"), config.get("end_seconds")
            if not isinstance(config.get("reference_id"), str) or not config["reference_id"].strip():
                raise ValueError(f"{label}: reference_id required")
            if not number(start) or not number(end) or not 0 <= start < end < float("inf"):
                raise ValueError(f"{label}: require 0 <= start_seconds < end_seconds")
        elif node.name == "h3_av_concat":
            sources = ordered_ids(config.get("source_nodes"), f"{label}.source_nodes", allow_repeats=True)
            if set(sources) != parents:
                raise ValueError(f"{label}: source_nodes must order all explicit video parents")
        if node.name == "h3_reference_bank" and parents:
            raise ValueError(f"{label}: task reference bank accepts no artifact parents")
        if node.name in {"h3_reference_select", "h3_frame_extract", "h3_audio_extract", "h3_reference_trim"} and len(parents) != 1:
            raise ValueError(f"{label}: exactly one explicit artifact parent required")

    validate_h3_reference_flow(graph)


@dataclass
class GraphPathRollout:
    task: VideoTask
    graph: ToolPathGraph
    plan: VideoPlan
    artifact: VideoArtifact
    evaluation: EvaluationReport
    failure: FailureReport | None
    state: VideoGenerationState
    reward: TaskReward | None = None

    @property
    def score(self) -> float:
        return self.reward.score if self.reward is not None else self.evaluation.score


@dataclass
class GraphPathCandidate:
    graph: ToolPathGraph
    edits: list[BoundedGraphEdit]
    reason: str
    parent_failure_types: list[str]


@dataclass
class GraphPathValidationReport:
    skill_name: str
    validation_task_ids: list[str]
    baseline_score: float
    candidate_score: float
    quality_gain: float
    estimated_cost: float
    stability: float
    accepted: bool
    tool_path: list[str]
    edits: list[dict[str, object]]
    evidence: list[str] = field(default_factory=list)
    baseline_pass_rate: float = 0.0
    candidate_pass_rate: float = 0.0
    passed_validation_task_ids: list[str] = field(default_factory=list)


@dataclass
class GraphEvolutionReport:
    baseline_rollouts: list[GraphPathRollout]
    candidates: list[GraphPathCandidate]
    validation_reports: list[GraphPathValidationReport]
    selected_graphs: list[ToolPathGraph]
    memory_dir: str


class GraphToolPathEvolver:
    """Verifier-gated graph-structured tool-path evolution for video generation."""

    def __init__(
        self,
        skill_memory: SkillMemory,
        graph_memory: GraphSkillMemory,
        tools: ToolRegistry | None = None,
        planner: Planner | None = None,
        evaluators: EvaluatorSuite | None = None,
        diagnoser: FailureDiagnoser | None = None,
        vlm_augmenter: VLMEvidenceAugmenter | None = None,
        min_quality_gain: float = 0.02,
        mutation_proposer: Any | None = None,
        template_mutations_enabled: bool = True,
        weighted_tool_graph: Any | None = None,
        historical_path_composer: Any | None = None,
        reward_router: TaskConditionedRewardRouter | None = None,
    ):
        self.skill_memory = skill_memory
        self.graph_memory = graph_memory
        self.tools = tools or ToolRegistry.with_mock_tools()
        self.planner = planner or Planner()
        self.evaluators = evaluators or EvaluatorSuite()
        self.diagnoser = diagnoser or FailureDiagnoser()
        self.vlm_augmenter = vlm_augmenter
        self.reward_router = reward_router or TaskConditionedRewardRouter()
        self.min_quality_gain = min_quality_gain
        self.mutation_proposer = mutation_proposer
        self.template_mutations_enabled = template_mutations_enabled
        self.weighted_tool_graph = weighted_tool_graph
        self.historical_path_composer = historical_path_composer
        self.last_mutation_error: str | None = None
        self.last_candidate_audit: list[dict[str, Any]] = []
        self._candidate_rejection_cache: dict[tuple[Any, ...], str] = {}
        self.executor = ArtifactPassingGraphExecutor(self.tools, self.evaluators, self.vlm_augmenter)

    def evolve(self, tasks: list[VideoTask], max_candidates: int = 8) -> GraphEvolutionReport:
        baseline = self.baseline_graph()
        baseline_rollouts = [self.rollout(task, baseline) for task in tasks]
        candidates = self.propose_candidates(baseline_rollouts)[:max_candidates]
        validation_reports = [self.validate_candidate(candidate, tasks) for candidate in candidates]
        selected_reports = self.select_pareto_frontier(validation_reports)
        selected_names = {report.skill_name for report in selected_reports}
        selected_graphs: list[ToolPathGraph] = []
        for candidate in candidates:
            report = next((item for item in validation_reports if item.skill_name == candidate.graph.skill_name and item.accepted), None)
            if report is None:
                continue
            candidate.graph.stats.update(
                {
                    "baseline_score": report.baseline_score,
                    "candidate_score": report.candidate_score,
                    "quality_gain": report.quality_gain,
                    "estimated_cost": report.estimated_cost,
                    "stability": report.stability,
                    "pass_rate": report.candidate_pass_rate,
                    "baseline_pass_rate": report.baseline_pass_rate,
                    "validation_task_ids": sorted(set(report.validation_task_ids)),
                    "passed_validation_task_ids": report.passed_validation_task_ids,
                    "accepted": report.accepted,
                    "pareto_selected": report.skill_name in selected_names,
                    "source_failure_types": [item.removeprefix("failure:") for item in report.evidence if item.startswith("failure:")],
                }
            )
            self.graph_memory.upsert_graph(candidate.graph)
            if report.skill_name not in selected_names:
                continue
            skill = candidate.graph.to_skill_card()
            skill.validation = SkillValidationReport(
                skill_name=skill.skill_name,
                validation_task_ids=report.validation_task_ids,
                baseline_score=report.baseline_score,
                candidate_score=report.candidate_score,
                baseline_pass_rate=report.baseline_pass_rate,
                candidate_pass_rate=report.candidate_pass_rate,
                quality_gain=report.quality_gain,
                estimated_cost=report.estimated_cost,
                accepted=report.accepted,
                evidence=report.evidence,
            )
            self.skill_memory.upsert(skill)
            selected_graphs.append(candidate.graph)

        for rollout in baseline_rollouts:
            failure_types = [item.value for item in rollout.failure.failure_types] if rollout.failure else []
            self.graph_memory.append_experience(
                ExperienceRecord(
                    task_id=rollout.task.task_id,
                    prompt=rollout.task.prompt,
                    failure_types=failure_types,
                    graph_id=rollout.graph.graph_id,
                    tool_path=rollout.graph.executable_tool_names(self.tools.available_names()),
                    score=rollout.score,
                    cost=rollout.graph.estimated_cost(),
                    accepted=False,
                    evidence=[metric.name for metric in rollout.evaluation.failed_metrics],
                )
            )
        for report in selected_reports:
            for task in tasks:
                self.graph_memory.append_experience(
                    ExperienceRecord(
                        task_id=task.task_id,
                        prompt=task.prompt,
                        failure_types=[item for item in report.evidence if item.startswith("failure:")],
                        graph_id=report.skill_name,
                        tool_path=report.tool_path,
                        score=report.candidate_score,
                        cost=report.estimated_cost,
                        accepted=True,
                        evidence=report.evidence,
                    )
                )

        return GraphEvolutionReport(
            baseline_rollouts=baseline_rollouts,
            candidates=candidates,
            validation_reports=validation_reports,
            selected_graphs=selected_graphs,
            memory_dir=str(self.graph_memory.memory_dir),
        )

    def rollout(self, task: VideoTask, graph: ToolPathGraph, *, node_cache=None, observer=None) -> GraphPathRollout:
        validate_h3_node_configs(graph, self.tools.available_names(), local=h3_local_active(self.tools))
        if h3_registry_active(self.tools.available_names()):
            validate_h3_composition_duration(graph, task)
        plan = self.planner.plan(task, [graph.skill_name])
        executable = graph.executable_tool_names(self.tools.available_names())
        if executable:
            plan.tool_chain = executable
        execution_options = {}
        if node_cache is not None:
            execution_options["node_cache"] = node_cache
        if observer is not None:
            execution_options["observer"] = observer
        execution = self.executor.execute(task, plan, graph, **execution_options)
        artifact = execution.artifact
        if h3_registry_active(self.tools.available_names()) and (
            set(artifact.tool_chain) & (H3_GENERATORS | {"mock_text_to_video"})
        ):
            label = (task.metadata or {}).get("replicate_label", (task.metadata or {}).get("evaluation_seed", (task.metadata or {}).get("generation_seed")))
            local = h3_local_active(self.tools) or artifact.metadata.get("provider") == "local-h3"
            artifact.metadata["generation_seed_applied"] = h3_local_seed_applied(artifact.metadata, label) if local else False
            if not local:
                artifact.metadata["provider_seed_control"] = False
            artifact.metadata["replicate_label"] = label
            artifact.metadata["evaluation_protocol"] = H3_LOCAL_EVALUATION_PROTOCOL if local else "unseeded_h3_api"
        state = execution.state
        if h3_registry_active(self.tools.available_names()):
            artifact.metadata["h3_observed_tool_edges"] = observed_tool_edges(graph, state, execution.node_artifacts)
        artifact.metadata.update(
            {
                "tool_path_nodes": [node.name for node in graph.nodes],
                "tool_path_edges": [edge.edge_id for edge in graph.edges],
            }
        )
        if self.vlm_augmenter is not None and "vlm_evaluation" not in artifact.metadata:
            artifact = self.vlm_augmenter.augment(task, artifact)
        evaluation = self.evaluators.evaluate(task, artifact)
        reward = self.reward_router.evaluate(task, artifact, evaluation)
        artifact.metadata["task_reward"] = reward.to_dict()
        state.verifier_scores.update(
            {f"reward:{name}": score for name, score in reward.components.items()}
        )
        for metric in evaluation.metrics:
            state.verifier_scores[metric.name] = metric.score
            if not metric.passed:
                state.failures.append(metric.name)
        failure = self.diagnoser.diagnose(task, artifact, evaluation)
        evaluation_seed = (task.metadata or {}).get("replicate_label", (task.metadata or {}).get("evaluation_seed", (task.metadata or {}).get("generation_seed")))
        seed_requested = evaluation_seed is not None
        seed_controlled = artifact.metadata.get(
            "generation_seed_applied",
            all(self.tools.spec(name).backend == "builtin" for name in artifact.tool_chain),
        )
        if (
            self.weighted_tool_graph is not None
            and not graph.stats.get("defer_weighted_credit", False)
        ):
            self.weighted_tool_graph.record_rollout(
                task=task,
                graph_id=graph.graph_id,
                tools=list(artifact.tool_chain),
                quality=reward.score,
                cost=graph.estimated_cost(),
                success=evaluation.passed,
                event_id=f"{task.task_id}:{graph.graph_id}:{artifact.artifact_id}",
                seed_controlled=seed_controlled if seed_requested else None,
                arena_variant=graph.stats.get("tool_arena_variant"),
                evaluation_seed=evaluation_seed,
                **({
                    "graph_behavior_fingerprint": graph_behavior_fingerprint(graph),
                    "actual_edges": artifact.metadata.get("h3_observed_tool_edges", []),
                    "comparison_protocol": (H3_LOCAL_COMPARISON_PROTOCOL
                                            if artifact.metadata.get("evaluation_protocol") == H3_LOCAL_EVALUATION_PROTOCOL
                                            else H3_COMPARISON_PROTOCOL),
                } if h3_registry_active(self.tools.available_names()) else {}),
            )
        return GraphPathRollout(task, graph, plan, artifact, evaluation, failure, state, reward)

    def propose_candidates(self, baseline_rollouts: list[GraphPathRollout]) -> list[GraphPathCandidate]:
        self.last_candidate_audit = []
        candidates: list[GraphPathCandidate] = []
        if self.mutation_proposer is not None:
            task_conditioned: dict[str, Any] = {}
            wanted_graph_ids = {rollout.graph.graph_id for rollout in baseline_rollouts}
            if self.weighted_tool_graph is not None:
                for rollout in baseline_rollouts:
                    if rollout.failure is None:
                        continue
                    context = self.weighted_tool_graph.search_context(
                        rollout.task,
                        tool_registry=self.tools,
                    )
                    task_conditioned[rollout.task.task_id] = context
                    for path in context.get("top_paths", []):
                        wanted_graph_ids.update(path.get("graph_ids", []))
            historical = sorted(
                self.graph_memory.list_graphs(),
                key=lambda graph: (
                    graph.graph_id in wanted_graph_ids,
                    float(graph.stats.get("quality_gain", 0.0)),
                    float(graph.stats.get("stability", 0.0)),
                ),
                reverse=True,
            )[:8]
            search_context: dict[str, Any] = {
                "task_conditioned": task_conditioned,
                "historical_graphs": [graph.to_dict() for graph in historical],
            }
            try:
                candidates.extend(
                    self.mutation_proposer.propose(
                        baseline_rollouts,
                        self.tools.available_names(),
                        search_context=search_context,
                    )
                )
                self.last_mutation_error = None
            except Exception as exc:
                if h3_registry_active(self.tools.available_names()) and h3_run_should_stop(exc):
                    raise
                self.last_mutation_error = str(exc)
                self.last_candidate_audit.append(
                    {
                        "stage": "llm_mutation",
                        "status": "rejected",
                        "rejection_reason": str(exc),
                    }
                )
                print(f"LLM graph mutation skipped: {exc}")
            finally:
                for error in getattr(self.mutation_proposer, "last_candidate_errors", []):
                    self.last_candidate_audit.append(
                        {"stage": "llm_candidate_parse", "status": "rejected", **error}
                    )
                for audit in getattr(self.mutation_proposer, "last_invention_audits", []):
                    self.last_candidate_audit.append(dict(audit))
                for result in getattr(self.mutation_proposer, "last_onboarding_results", []):
                    self.last_candidate_audit.append(
                        {"stage": "tool_onboarding", **result}
                    )
        if self.historical_path_composer is not None and not h3_registry_active(self.tools.available_names()):
            candidates.extend(self.historical_path_composer.propose(baseline_rollouts))
        if not self.template_mutations_enabled or h3_registry_active(self.tools.available_names()):
            return self._prepare_executable_candidates(candidates, baseline_rollouts)
        for rollout in baseline_rollouts:
            if rollout.failure is None:
                continue
            failure_types = set(rollout.failure.failure_types)
            if failure_types & {FailureType.IDENTITY_DRIFT, FailureType.CLOTHING_COLOR_DRIFT}:
                candidates.append(self._candidate_localized_identity_repair(rollout.failure))
                candidates.append(self._candidate_identity_keyframe_i2v(rollout.failure))
                if self._looks_multishot(rollout.task):
                    candidates.append(self._candidate_multishot_character_graph(rollout.failure))
            if FailureType.MOTION_MISMATCH in failure_types:
                candidates.append(self._candidate_temporal_plan_t2v(rollout.failure))
                candidates.append(self._candidate_temporal_keyframe_graph(rollout.failure))
                candidates.append(self._candidate_segment_repair_graph(rollout.failure))
            if failure_types & {FailureType.STYLE_DRIFT, FailureType.TEMPORAL_FLICKER} or "anime" in rollout.task.prompt.lower():
                candidates.append(self._candidate_style_transfer_graph(rollout.failure))
            if FailureType.EDITING_LEAKAGE in failure_types or rollout.task.mode == TaskMode.EDITING:
                candidates.append(self._candidate_region_edit_graph(rollout.failure))
        return self._prepare_executable_candidates(candidates, baseline_rollouts)

    def _prepare_executable_candidates(
        self,
        candidates: list[GraphPathCandidate],
        baseline_rollouts: list[GraphPathRollout] | None = None,
    ) -> list[GraphPathCandidate]:
        if baseline_rollouts is not None:
            reference_task_ids = sorted({
                rollout.task.task_id
                for rollout in baseline_rollouts
                if rollout.task.reference_video
            })
            for candidate in candidates:
                candidate.graph.stats.pop("near_miss_repair", None)
                if h3_registry_active(self.tools.available_names()):
                    from evovideo_skill.h3_prompt_binding import task_prompt_key

                    hashes = sorted({task_prompt_key(r.task) for r in baseline_rollouts})
                    for node in candidate.graph.nodes:
                        if node.name in H3_GENERATORS | {"mock_text_to_video"} and node.config.get("prompt"):
                            # Bind free-form scene details to the task that supplied them.
                            node.config.setdefault("prompt_task_hashes", hashes)
                        if node.name in H3_GENERATORS | {"mock_text_to_video"} and "shot_index" not in node.config:
                            # A whole-video terminal inherits each task's duration.
                            # Segment generators feeding a compositor retain theirs.
                            pending = [edge.target for edge in candidate.graph.edges if edge.source == node.node_id]
                            seen = set()
                            has_downstream_tool = False
                            while pending:
                                target = pending.pop()
                                if target in seen:
                                    continue
                                seen.add(target)
                                if candidate.graph.node(target).node_type == "tool":
                                    has_downstream_tool = True
                                    break
                                pending.extend(edge.target for edge in candidate.graph.edges if edge.source == target)
                            if not has_downstream_tool and "duration_seconds" in node.config:
                                old_duration = node.config.pop("duration_seconds")
                                candidate.graph.stats.setdefault("automatic_repairs", []).append(
                                    f"{node.node_id}: terminal duration {old_duration} replaced by current task duration"
                                )
                candidate.graph.stats["planning_context"] = {
                    "reference_video_task_ids": reference_task_ids,
                    "source_video_available": bool(reference_task_ids),
                }
        executable = self._executable_candidates(self._dedupe_candidates(candidates))
        expanded = self._expand_tool_arena_variants(executable)
        if len(expanded) == len(executable):
            return executable
        return self._executable_candidates(self._dedupe_candidates(expanded))

    def _expand_tool_arena_variants(
        self,
        candidates: list[GraphPathCandidate],
    ) -> list[GraphPathCandidate]:
        """Create matched path arms by substituting one same-capability tool at a time."""
        if os.environ.get("OPEN_WORLD_TOOL_ARENA", "0") != "1":
            return candidates
        max_variants = max(1, int(os.environ.get("OPEN_WORLD_ARENA_MAX_GRAPH_VARIANTS", "8")))
        expanded: list[GraphPathCandidate] = []
        added = 0
        for candidate in candidates:
            expanded.append(candidate)
            for node in candidate.graph.nodes:
                if node.node_type != "tool" or not self.tools.has(node.name):
                    continue
                current = self.tools.spec(node.name)
                variants = [
                    spec for spec in self.tools.find_by_capability(current.capability)
                    if spec.name != node.name
                    and spec.verified
                    and spec.backend != "builtin"
                    and "tool-arena" in str(spec.provenance)
                ]
                if not variants or (
                    current.backend == "builtin" and "tool-arena" not in str(current.provenance)
                ):
                    continue
                family_id = f"{candidate.graph.graph_id}:{node.node_id}:{current.capability}"
                candidate.graph.stats.setdefault("tool_arena_variant", {
                    "family_id": family_id,
                    "capability": current.capability,
                    "node_id": node.node_id,
                    "selected_tool": node.name,
                    "compared_tools": [node.name, *[item.name for item in variants]],
                    "comparison_stage": "same_path_tool_backend",
                })
                for variant in variants:
                    if added >= max_variants:
                        break
                    clone = ToolPathGraph.from_dict(candidate.graph.to_dict())
                    replacement = clone.node(node.node_id)
                    previous_name = replacement.name
                    replacement.name = variant.name
                    replacement.config["cost"] = float(variant.estimated_cost)
                    suffix = re.sub(r"[^a-z0-9]+", "_", variant.name.lower()).strip("_")
                    clone.graph_id = f"{candidate.graph.graph_id}__arena_{suffix}"
                    clone.skill_name = clone.graph_id
                    clone.stats["tool_arena_variant"] = {
                        "family_id": family_id,
                        "capability": current.capability,
                        "node_id": node.node_id,
                        "selected_tool": variant.name,
                        "replaced_tool": previous_name,
                        "compared_tools": [node.name, *[item.name for item in variants]],
                        "comparison_stage": "same_path_tool_backend",
                    }
                    edit = BoundedGraphEdit(
                        "replace_node",
                        target=node.node_id,
                        payload={
                            "node_id": node.node_id,
                            "node_type": "tool",
                            "name": variant.name,
                            "config": dict(replacement.config),
                        },
                        reason=(
                            f"tool-arena substitution: compare {variant.name} against "
                            f"{previous_name} under the same graph"
                            + (" with matched local H3 task/seed replicates" if h3_local_active(self.tools)
                               else " with unseeded H3 API samples" if h3_registry_active(self.tools.available_names())
                               else " and paired seeds")
                        ),
                    )
                    expanded.append(
                        GraphPathCandidate(
                            clone,
                            [*candidate.edits, edit],
                            f"{candidate.reason} Tool arena arm uses {variant.name}.",
                            list(candidate.parent_failure_types),
                        )
                    )
                    added += 1
                if added >= max_variants:
                    break
        return expanded

    def _executable_candidates(self, candidates: list[GraphPathCandidate]) -> list[GraphPathCandidate]:
        """Keep only paths whose tool capabilities are real and artifact-compatible."""
        executable: list[GraphPathCandidate] = []
        for candidate in candidates:
            available = self.tools.available_names()
            base_audit = {
                "stage": "candidate_preflight",
                "graph_id": candidate.graph.graph_id,
                "proposal_source": candidate.graph.stats.get("proposal_source", "template_or_composition"),
                "reason": candidate.reason,
                "tools": candidate.graph.tool_names(),
            }
            original_cache_key = self._candidate_cache_key(candidate, available)
            cached_reason = self._candidate_rejection_cache.get(original_cache_key)
            if cached_reason is not None:
                self.last_candidate_audit.append(
                    {
                        **base_audit,
                        "status": "skipped",
                        "rejection_reason": "cached_rejection",
                        "cached_rejection_reason": cached_reason,
                    }
                )
                continue
            missing = sorted(name for name in candidate.graph.tool_names() if name not in available)
            onboarding_results = self._onboard_candidate_tools(candidate, missing)
            for result in onboarding_results:
                self.last_candidate_audit.append({"stage": "tool_onboarding", **result})
            available = self.tools.available_names()
            missing = sorted(name for name in candidate.graph.tool_names() if name not in available)
            if missing:
                self._candidate_rejection_cache[original_cache_key] = "unavailable_tools"
                self.last_candidate_audit.append(
                    {
                        **base_audit,
                        "status": "rejected",
                        "rejection_reason": "unavailable_tools",
                        "missing_tools": missing,
                        "available_tools": sorted(available),
                    }
                )
                continue
            try:
                validate_h3_node_configs(candidate.graph, available, local=h3_local_active(self.tools))
            except (TypeError, ValueError) as exc:
                self._cache_candidate_rejection(candidate, available, str(exc))
                self.last_candidate_audit.append(
                    {**base_audit, "status": "rejected", "rejection_reason": str(exc)}
                )
                continue
            repairs = self._repair_artifact_edges(candidate)
            contract_repairs = self._repair_contract_edges(candidate)
            repairs.extend(contract_repairs)
            if repairs:
                base_audit["auto_repairs"] = repairs
                base_audit["tools"] = candidate.graph.tool_names()
                repaired_cache_key = self._candidate_cache_key(candidate, available)
                cached_reason = self._candidate_rejection_cache.get(repaired_cache_key)
                if cached_reason is not None:
                    self.last_candidate_audit.append(
                        {
                            **base_audit,
                            "status": "skipped",
                            "rejection_reason": "cached_rejection",
                            "cached_rejection_reason": cached_reason,
                        }
                    )
                    continue
            compatible = True
            incompatibility = ""
            for edge in candidate.graph.edges:
                source = candidate.graph.node(edge.source)
                target = candidate.graph.node(edge.target)
                if source.node_type != "tool" or target.node_type != "tool":
                    continue
                produced_contract = (
                    source.config.get("target_contract")
                    if source.config.get("contract_alignment")
                    else None
                )
                check = self.tools.connection_contract_check(source.name, target.name, produced_contract)
                valid, reason = check.compatible, check.reason
                if not valid:
                    compatible = False
                    incompatibility = f"{edge.source}->{edge.target}: {reason}"
                    break
            if not compatible:
                self._cache_candidate_rejection(candidate, available, incompatibility)
                self.last_candidate_audit.append(
                    {**base_audit, "status": "rejected", "rejection_reason": incompatibility}
                )
                continue
            try:
                self.executor.validate_graph(candidate.graph)
            except Exception as exc:
                if h3_registry_active(self.tools.available_names()) and h3_run_should_stop(exc):
                    raise
                candidate.graph.stats["rejection_reason"] = str(exc)
                self._cache_candidate_rejection(candidate, available, str(exc))
                self.last_candidate_audit.append(
                    {**base_audit, "status": "rejected", "rejection_reason": str(exc)}
                )
                continue
            self.last_candidate_audit.append({**base_audit, "status": "executable"})
            executable.append(candidate)
        return executable

    def _onboard_candidate_tools(
        self,
        candidate: GraphPathCandidate,
        missing: list[str],
    ) -> list[dict[str, Any]]:
        manager = getattr(self.mutation_proposer, "onboarding_manager", None)
        if manager is None or not missing:
            return []
        requests = []
        for name in missing:
            spec = self.tools.suggested_spec(name)
            required_input_types = self._candidate_required_input_types(candidate.graph, name)
            requests.append(
                CapabilityRequest(
                    capability=spec.capability,
                    suggested_tool_name=name,
                    required_input_types=required_input_types or list(spec.input_types),
                    reason=f"Candidate graph {candidate.graph.graph_id} requires missing tool {name}.",
                )
            )
        results = manager.onboard_requests(requests)
        replacements: dict[str, str] = {}
        for name, result in zip(missing, results):
            if result.tool_name and self.tools.has(result.tool_name):
                replacements[name] = result.tool_name
        if replacements:
            for node in candidate.graph.nodes:
                if node.node_type == "tool" and node.name in replacements:
                    node.name = replacements[node.name]
            candidate.graph.stats.setdefault("onboarded_tool_replacements", {}).update(replacements)
        return [result.to_dict() for result in results]

    def _candidate_required_input_types(self, graph: ToolPathGraph, tool_name: str) -> list[str]:
        """Describe the artifacts that this specific graph will feed to a missing tool."""
        incoming: dict[str, list[GraphEdge]] = {node.node_id: [] for node in graph.nodes}
        for edge in graph.edges:
            if edge.target in incoming:
                incoming[edge.target].append(edge)
        required: list[str] = []
        for target in graph.nodes:
            if target.node_type != "tool" or target.name != tool_name:
                continue
            for edge in incoming.get(target.node_id, []):
                source = graph.node(edge.source)
                producer = (
                    source
                    if source.node_type == "tool"
                    else self._verifier_artifact_producer(graph, source, incoming)
                )
                if producer is None or not self.tools.has(producer.name):
                    continue
                artifact_type = str(self.tools.spec(producer.name).output_type)
                if artifact_type and artifact_type not in required:
                    required.append(artifact_type)
        return required

    def _repair_artifact_edges(self, candidate: GraphPathCandidate) -> list[dict[str, str]]:
        graph = candidate.graph
        incoming = {node.node_id: [] for node in graph.nodes}
        for edge in graph.edges:
            incoming[edge.target].append(edge)
        repairs: list[dict[str, str]] = []
        existing_ids = {edge.edge_id for edge in graph.edges}
        for target_index, target in enumerate(graph.nodes):
            if target.node_type != "tool":
                continue
            if h3_registry_active(self.tools.available_names()) and target.name.startswith("h3_"):
                # H3 bindings and reference order are explicit planner decisions.
                continue
            spec = self.tools.spec(target.name)
            producers = [
                edge for edge in incoming[target.node_id]
                if graph.node(edge.source).node_type in {"tool", "verifier"}
            ]
            if not spec.consumes_upstream or producers:
                continue
            source = None
            for candidate_source in reversed(graph.nodes[:target_index]):
                if candidate_source.node_type != "tool":
                    continue
                contract_check = self.tools.connection_contract_check(candidate_source.name, target.name)
                if contract_check.compatible or contract_check.bridges:
                    source = candidate_source
                    break
            if (
                source is None
                and self.tools.has("task_reference_video")
                and graph.stats.get("planning_context", {}).get("source_video_available")
            ):
                source = next(
                    (
                        node for node in graph.nodes
                        if node.node_type == "tool" and node.name == "task_reference_video"
                    ),
                    None,
                )
                if source is None:
                    node_id = "tool_task_reference_video"
                    suffix = 2
                    existing_node_ids = {node.node_id for node in graph.nodes}
                    while node_id in existing_node_ids:
                        node_id = f"tool_task_reference_video_{suffix}"
                        suffix += 1
                    proposed = GraphNode(
                        node_id,
                        "tool",
                        "task_reference_video",
                        {"cost": float(self.tools.spec("task_reference_video").estimated_cost)},
                    )
                    check = self.tools.connection_contract_check(proposed.name, target.name)
                    if check.compatible or check.bridges:
                        source = proposed
                        graph.nodes.append(source)
                        incoming[source.node_id] = []
                        candidate.edits.append(
                            BoundedGraphEdit(
                                "add_node",
                                payload={
                                    "node_id": source.node_id,
                                    "node_type": "tool",
                                    "name": source.name,
                                    "config": dict(source.config),
                                },
                                reason="materialize the task reference video as the editing graph root",
                            )
                        )
            if source is None:
                continue
            edge_id = f"auto_artifact_{source.node_id}_{target.node_id}"
            suffix = 2
            while edge_id in existing_ids:
                edge_id = f"auto_artifact_{source.node_id}_{target.node_id}_{suffix}"
                suffix += 1
            edge = GraphEdge(edge_id, source.node_id, target.node_id)
            graph.edges.append(edge)
            incoming[target.node_id].append(edge)
            existing_ids.add(edge_id)
            reason = f"auto-connect {source.name} output to upstream consumer {target.name}"
            candidate.edits.append(
                BoundedGraphEdit("add_edge", payload={"edge_id": edge_id, "source": source.node_id, "target": target.node_id}, reason=reason)
            )
            repairs.append({
                "kind": "task_reference_video_root" if source.name == "task_reference_video" else "artifact_edge",
                "edge_id": edge_id,
                "source": source.node_id,
                "target": target.node_id,
            })
        if repairs:
            graph.stats.setdefault("automatic_graph_repairs", []).extend(repairs)
        return repairs

    def _repair_contract_edges(self, candidate: GraphPathCandidate) -> list[dict[str, Any]]:
        """Insert executable bridge nodes when repository artifact contracts are convertible."""
        graph = candidate.graph
        repairs: list[dict[str, Any]] = []
        existing_nodes = {node.node_id for node in graph.nodes}
        existing_edges = {edge.edge_id for edge in graph.edges}
        replacements: list[tuple[GraphEdge, list[GraphEdge], list[GraphNode]]] = []
        incoming = {node.node_id: [] for node in graph.nodes}
        for edge in graph.edges:
            incoming[edge.target].append(edge)
        for edge in list(graph.edges):
            source = graph.node(edge.source)
            target = graph.node(edge.target)
            if target.node_type != "tool" or source.node_type not in {"tool", "verifier"}:
                continue
            if h3_registry_active(self.tools.available_names()) and target.name.startswith("h3_"):
                continue
            producer = source if source.node_type == "tool" else self._verifier_artifact_producer(graph, source, incoming)
            if producer is None:
                continue
            produced_contract = (
                producer.config.get("target_contract")
                if producer.config.get("contract_alignment")
                else None
            )
            check = self.tools.connection_contract_check(producer.name, target.name, produced_contract)
            if check.compatible or not check.bridges:
                continue
            previous = edge.source
            new_nodes: list[GraphNode] = []
            new_edges: list[GraphEdge] = []
            for bridge_index, (bridge_name, bridge_config) in enumerate(check.bridges, start=1):
                base_node_id = f"bridge_{edge.source}_{edge.target}_{bridge_index}"
                node_id = base_node_id
                suffix = 2
                while node_id in existing_nodes:
                    node_id = f"{base_node_id}_{suffix}"
                    suffix += 1
                existing_nodes.add(node_id)
                config = dict(bridge_config)
                config["cost"] = float(self.tools.spec(bridge_name).estimated_cost)
                config["contract_alignment"] = True
                node = GraphNode(node_id, "tool", bridge_name, config)
                new_nodes.append(node)
                edge_id = f"contract_{previous}_{node_id}"
                while edge_id in existing_edges:
                    edge_id += "_2"
                existing_edges.add(edge_id)
                new_edges.append(
                    GraphEdge(edge_id, previous, node_id, edge.condition if bridge_index == 1 else "always")
                )
                previous = node_id
            final_edge_id = f"contract_{previous}_{edge.target}"
            while final_edge_id in existing_edges:
                final_edge_id += "_2"
            existing_edges.add(final_edge_id)
            new_edges.append(GraphEdge(final_edge_id, previous, edge.target, "always"))
            replacements.append((edge, new_edges, new_nodes))
            repairs.append(
                {
                    "kind": "artifact_contract_bridge",
                    "replaced_edge": edge.edge_id,
                    "producer": producer.name,
                    "consumer": target.name,
                    "reason": check.reason,
                    "bridges": [node.name for node in new_nodes],
                    "target_contract": check.target.to_dict() if check.target else None,
                }
            )
        for old_edge, new_edges, new_nodes in replacements:
            graph.edges = [item for item in graph.edges if item.edge_id != old_edge.edge_id]
            graph.nodes.extend(new_nodes)
            graph.edges.extend(new_edges)
            for node in new_nodes:
                candidate.edits.append(
                    BoundedGraphEdit(
                        "add_node",
                        payload={
                            "node_id": node.node_id,
                            "node_type": node.node_type,
                            "name": node.name,
                            "config": node.config,
                        },
                        reason="automatically align repository artifact contracts",
                    )
                )
        if repairs:
            graph.stats.setdefault("artifact_contract_repairs", []).extend(repairs)
        return repairs

    @staticmethod
    def _verifier_artifact_producer(
        graph: ToolPathGraph,
        verifier: GraphNode,
        incoming: dict[str, list[GraphEdge]],
    ) -> GraphNode | None:
        queue = [edge.source for edge in incoming.get(verifier.node_id, [])]
        visited: set[str] = set()
        while queue:
            node_id = queue.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            node = graph.node(node_id)
            if node.node_type == "tool":
                return node
            if node.node_type == "verifier":
                queue.extend(edge.source for edge in incoming.get(node_id, []))
        return None

    @staticmethod
    def _candidate_signature(candidate: GraphPathCandidate) -> tuple[Any, ...]:
        return (
            candidate.graph.graph_id,
            tuple((node.node_id, node.node_type, node.name) for node in candidate.graph.nodes),
            tuple((edge.source, edge.target, edge.condition) for edge in candidate.graph.edges),
        )

    def _candidate_cache_key(
        self,
        candidate: GraphPathCandidate,
        available: set[str],
    ) -> tuple[Any, ...]:
        return (*self._candidate_signature(candidate), tuple(sorted(available)))

    def _cache_candidate_rejection(
        self,
        candidate: GraphPathCandidate,
        available: set[str],
        reason: str,
    ) -> None:
        self._candidate_rejection_cache[self._candidate_cache_key(candidate, available)] = reason

    def validate_candidate(self, candidate: GraphPathCandidate, tasks: list[VideoTask]) -> GraphPathValidationReport:
        validation_tasks = self._validation_tasks(tasks, candidate)
        baseline_graph = self.baseline_graph()
        baseline_rollouts = [self.rollout(task, baseline_graph) for task in validation_tasks]
        candidate_rollouts = [self.rollout(task, candidate.graph) for task in validation_tasks]
        baseline_scores = [rollout.score for rollout in baseline_rollouts]
        candidate_scores = [rollout.score for rollout in candidate_rollouts]
        baseline_score = graph_score_mean(baseline_scores)
        candidate_score = graph_score_mean(candidate_scores)
        gain = candidate_score - baseline_score
        stability = graph_score_stability(candidate_scores)
        baseline_pass_rate = sum(1 for rollout in baseline_rollouts if rollout.evaluation.passed) / max(1, len(baseline_rollouts))
        candidate_pass_rate = sum(1 for rollout in candidate_rollouts if rollout.evaluation.passed) / max(1, len(candidate_rollouts))
        paired_gains = [candidate.score - baseline.score for baseline, candidate in zip(baseline_rollouts, candidate_rollouts)]
        improved_fraction = sum(value > 0.0 for value in paired_gains) / max(1, len(paired_gains))
        accepted = (
            gain >= self.min_quality_gain
            and candidate_score >= baseline_score
            and candidate_pass_rate >= baseline_pass_rate
            and improved_fraction >= 0.5
        )
        evidence = [f"failure:{item}" for item in candidate.parent_failure_types]
        evidence.extend(edit.reason for edit in candidate.edits if edit.reason)
        return GraphPathValidationReport(
            skill_name=candidate.graph.skill_name,
            validation_task_ids=[task.task_id for task in validation_tasks],
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            quality_gain=gain,
            estimated_cost=candidate.graph.estimated_cost(),
            stability=stability,
            accepted=accepted,
            tool_path=candidate.graph.executable_tool_names(self.tools.available_names()),
            edits=[edit.__dict__ for edit in candidate.edits],
            evidence=evidence,
            baseline_pass_rate=baseline_pass_rate,
            candidate_pass_rate=candidate_pass_rate,
            passed_validation_task_ids=sorted({
                rollout.task.task_id
                for rollout in candidate_rollouts
                if rollout.evaluation.passed
            }),
        )

    @staticmethod
    def select_pareto_frontier(reports: list[GraphPathValidationReport]) -> list[GraphPathValidationReport]:
        accepted = [report for report in reports if report.accepted]
        frontier = []
        for report in accepted:
            dominated = False
            for other in accepted:
                if other is report:
                    continue
                no_worse = (
                    other.quality_gain >= report.quality_gain
                    and other.stability >= report.stability
                )
                strictly_better = (
                    other.quality_gain > report.quality_gain
                    or other.stability > report.stability
                )
                if no_worse and strictly_better:
                    dominated = True
                    break
            if not dominated:
                frontier.append(report)
        return sorted(frontier, key=lambda item: (item.quality_gain, item.stability), reverse=True)

    def baseline_graph(self=None) -> ToolPathGraph:
        # Preserve the legacy class-level factory as well as instance calls.
        h3_active = self is not None and h3_registry_active(self.tools.available_names())
        return ToolPathGraph(
            graph_id="baseline_t2v_graph",
            skill_name="baseline_t2v_graph",
            description=(
                "Baseline H3 direct native conditioning: task frame roles use fl2va, "
                "other task references use ref2va, and no references use t2va. Long tasks "
                "independently generate declared h3_shots then concatenate them."
                if h3_active else "Baseline single-call text-to-video path."
            ),
            triggers=["generation"],
            nodes=[
                GraphNode("trigger_generation", "trigger", "generation", {"cost": 0.0}),
                GraphNode("tool_t2v", "tool", "mock_text_to_video", {"cost": 1.0}),
            ],
            edges=[GraphEdge("e_trigger_t2v", "trigger_generation", "tool_t2v")],
            validators=["identity_consistency", "clothing_color_consistency", "prompt_action_alignment"],
        )

    def _candidate_identity_keyframe_i2v(self, failure: FailureReport) -> GraphPathCandidate:
        graph = self.baseline_graph()
        graph.graph_id = "identity_reference_i2v_graph"
        graph.skill_name = "identity_reference_i2v_graph"
        graph.description = "Extract or synthesize identity keyframes, then use I2V to stabilize subject identity and clothing."
        i2v_name = self._preferred_capability_tool(
            "image_conditioned_video_generation",
            fallback="mock_image_to_video",
            preferred_inputs={"image"},
        )
        edits = [
            BoundedGraphEdit(
                "add_node",
                payload={"node_id": "tool_identity_frame", "node_type": "tool", "name": "extract_reference_identity_frame", "config": {"cost": 0.35}},
                reason="add an identity reference frame extraction node",
            ),
            BoundedGraphEdit(
                "add_node",
                payload={"node_id": "tool_i2v", "node_type": "tool", "name": i2v_name, "config": {"cost": 1.5}},
                reason="regenerate from the extracted reference with I2V",
            ),
            BoundedGraphEdit("add_validator", payload={"validator": "identity_consistency"}, reason="verify identity before committing the path"),
        ]
        for edit in edits:
            graph = graph.apply_edit(edit)
        graph.edges.extend(
            [
                GraphEdge("e_t2v_identity", "tool_t2v", "tool_identity_frame"),
                GraphEdge("e_identity_i2v", "tool_identity_frame", "tool_i2v", "identity_reference_ready"),
            ]
        )
        graph.triggers.extend(["identity_drift", "clothing_color_drift", "same character"])
        graph.fallbacks.append("multi_shot_character_graph")
        return GraphPathCandidate(graph, edits, "Use reference-frame I2V for identity or clothing drift.", [item.value for item in failure.failure_types])

    def _candidate_multishot_character_graph(self, failure: FailureReport) -> GraphPathCandidate:
        graph = self.baseline_graph()
        graph.graph_id = "multi_shot_character_graph"
        graph.skill_name = "multi_shot_character_graph"
        graph.description = "Build a character-sheet and generate multiple shots through I2V before stitching."
        multi_i2v_name = self._preferred_capability_tool(
            "multi_shot_identity_conditioned_generation",
            fallback=self._preferred_capability_tool(
                "image_conditioned_video_generation",
                fallback="mock_multi_shot_i2v",
                preferred_inputs={"image"},
            ),
            preferred_inputs={"image"},
        )
        edits = [
            BoundedGraphEdit("add_node", payload={"node_id": "tool_shot_plan", "node_type": "tool", "name": "scene_splitter", "config": {"cost": 0.35}}, reason="split long prompt into shot-level states"),
            BoundedGraphEdit("add_node", payload={"node_id": "tool_character_sheet", "node_type": "tool", "name": "character_sheet_generator", "config": {"cost": 0.55}}, reason="add persistent character sheet"),
            BoundedGraphEdit("add_node", payload={"node_id": "tool_multi_i2v", "node_type": "tool", "name": multi_i2v_name, "config": {"cost": 2.25}}, reason="refine the draft through a character-conditioned multi-shot generator"),
        ]
        for edit in edits:
            graph = graph.apply_edit(edit)
        graph.edges.extend(
            [
                GraphEdge("e_draft_shot_plan", "tool_t2v", "tool_shot_plan"),
                GraphEdge("e_shot_character", "tool_shot_plan", "tool_character_sheet"),
                GraphEdge("e_character_multi_i2v", "tool_character_sheet", "tool_multi_i2v"),
            ]
        )
        graph.triggers.extend(["multi_shot", "cross_shot_identity", "same character"])
        graph.validators.extend(["identity_consistency", "clothing_color_consistency"])
        graph.fallbacks.append("identity_reference_i2v_graph")
        return GraphPathCandidate(graph, edits, "Use character-sheet multi-shot I2V for cross-shot identity drift.", [item.value for item in failure.failure_types])

    def _candidate_temporal_plan_t2v(self, failure: FailureReport) -> GraphPathCandidate:
        """Cheap causal repair that conditions T2V on an explicit ordered plan."""
        graph = self.baseline_graph()
        graph.graph_id = "temporal_plan_conditioned_t2v_graph"
        graph.skill_name = graph.graph_id
        graph.description = (
            "Decompose the requested actions before generation and compile the ordered "
            "temporal states into the real T2V prompt."
        )
        edits = [
            BoundedGraphEdit(
                "add_node",
                payload={
                    "node_id": "tool_temporal_plan",
                    "node_type": "tool",
                    "name": "temporal_decomposer",
                    "config": {"cost": 0.2},
                },
                reason="make every requested action an explicit ordered generation stage",
            ),
            BoundedGraphEdit(
                "add_edge",
                payload={
                    "edge_id": "e_trigger_temporal_plan",
                    "source": "trigger_generation",
                    "target": "tool_temporal_plan",
                    "condition": "always",
                },
                reason="build the temporal plan before video generation",
            ),
            BoundedGraphEdit(
                "add_edge",
                payload={
                    "edge_id": "e_temporal_plan_t2v",
                    "source": "tool_temporal_plan",
                    "target": "tool_t2v",
                    "condition": "always",
                    "config": {"binding": "prompt"},
                },
                reason="compile the ordered plan into the Wan generation prompt",
            ),
            BoundedGraphEdit(
                "add_validator",
                payload={"validator": "prompt_action_alignment"},
                reason="validate the ordered action sequence",
            ),
        ]
        for edit in edits:
            graph = graph.apply_edit(edit)
        graph.triggers.extend([
            "motion_mismatch",
            "action_order",
            "state_transition_accuracy",
            "prompt_omission",
        ])
        graph.stats["mechanism_family"] = "temporal_prompt_conditioning"
        return GraphPathCandidate(
            graph,
            edits,
            "Condition Wan T2V on an explicit ordered temporal plan before spending budget on pixel editing.",
            [item.value for item in failure.failure_types],
        )

    def _candidate_temporal_keyframe_graph(self, failure: FailureReport) -> GraphPathCandidate:
        graph = self.baseline_graph()
        graph.graph_id = "temporal_keyframe_i2v_graph"
        graph.skill_name = "temporal_keyframe_i2v_graph"
        graph.description = "Decompose ordered actions into keyframes and condition generation on those temporal anchors."
        i2v_name = self._preferred_capability_tool(
            "image_conditioned_video_generation",
            fallback="mock_image_to_video",
            preferred_inputs={"image"},
        )
        edits = [
            BoundedGraphEdit("add_node", payload={"node_id": "tool_temporal_decomposer", "node_type": "tool", "name": "temporal_decomposer", "config": {"cost": 0.4}}, reason="add explicit temporal decomposition"),
            BoundedGraphEdit("add_node", payload={"node_id": "tool_keyframe_generator", "node_type": "tool", "name": "keyframe_generator", "config": {"cost": 0.6}}, reason="add action keyframe generation"),
            BoundedGraphEdit("add_node", payload={"node_id": "tool_action_i2v", "node_type": "tool", "name": i2v_name, "config": {"cost": 1.5}}, reason="refine the draft by conditioning motion on materialized keyframes"),
        ]
        for edit in edits:
            graph = graph.apply_edit(edit)
        graph.edges.extend(
            [
                GraphEdge("e_temporal_keyframe", "tool_temporal_decomposer", "tool_keyframe_generator"),
                GraphEdge("e_draft_keyframe", "tool_t2v", "tool_keyframe_generator"),
                GraphEdge("e_keyframe_i2v", "tool_keyframe_generator", "tool_action_i2v"),
            ]
        )
        graph.triggers.extend(["motion_mismatch", "action_order", "multi_action"])
        graph.validators.append("prompt_action_alignment")
        graph.fallbacks.append("segment_repair_graph")
        return GraphPathCandidate(graph, edits, "Use bounded temporal decomposition and keyframe-conditioned I2V.", [item.value for item in failure.failure_types])

    def _candidate_segment_repair_graph(self, failure: FailureReport) -> GraphPathCandidate:
        return self._candidate_localized_draft_repair(
            failure,
            graph_id="localized_temporal_segment_repair_graph",
            verifier="prompt_action_alignment",
            triggers=["motion_mismatch", "failed_segment", "action_order", "repair"],
            reason=(
                "Localize the failed Wan draft span, regenerate it from boundary pixels, "
                "and splice only that span back into the untouched draft."
            ),
        )

    def _candidate_localized_identity_repair(
        self,
        failure: FailureReport,
    ) -> GraphPathCandidate:
        return self._candidate_localized_draft_repair(
            failure,
            graph_id="localized_identity_segment_repair_graph",
            verifier="identity_consistency",
            triggers=["identity_drift", "clothing_color_drift", "same character", "repair"],
            reason=(
                "Repair only the identity-drift span using boundary-conditioned I2V and "
                "retain all healthy Wan frames outside it."
            ),
        )

    def _candidate_localized_draft_repair(
        self,
        failure: FailureReport,
        *,
        graph_id: str,
        verifier: str,
        triggers: list[str],
        reason: str,
    ) -> GraphPathCandidate:
        graph = self.baseline_graph()
        graph.graph_id = graph_id
        graph.skill_name = graph_id
        graph.description = (
            "Generate one Wan draft, localize the failed temporal span, extract its boundary "
            "pixels, regenerate only that span, and splice it back while retaining healthy frames."
        )
        repair_name = self._preferred_capability_tool(
            "image_conditioned_video_generation",
            fallback="mock_image_to_video",
            preferred_inputs={"image"},
        )
        edits = [
            BoundedGraphEdit("add_node", payload={"node_id": "verifier_draft_failure", "node_type": "verifier", "name": verifier, "config": {"threshold": 0.95, "cost": 0.2}}, reason="obtain VLM evidence on the Wan draft before changing pixels"),
            BoundedGraphEdit("add_node", payload={"node_id": "tool_failure_localizer", "node_type": "tool", "name": "failed_segment_localizer", "config": {"cost": 0.15}}, reason="convert verifier evidence into the smallest failed temporal span"),
            BoundedGraphEdit("add_node", payload={"node_id": "tool_boundary_frames", "node_type": "tool", "name": "boundary_frame_extractor", "config": {"cost": 0.15}}, reason="materialize healthy pixels immediately around the failed span"),
            BoundedGraphEdit("add_node", payload={"node_id": "tool_local_repair", "node_type": "tool", "name": repair_name, "config": {"cost": 1.5}}, reason="regenerate the failed span from real Wan boundary pixels"),
            BoundedGraphEdit("add_node", payload={"node_id": "tool_segment_stitch", "node_type": "tool", "name": "segment_stitcher", "config": {"cost": 0.2}}, reason="replace only the failed interval and retain healthy Wan frames"),
            BoundedGraphEdit("add_edge", payload={"edge_id": "e_draft_verify", "source": "tool_t2v", "target": "verifier_draft_failure", "condition": "always"}, reason="verify the original Wan draft"),
            BoundedGraphEdit("add_edge", payload={"edge_id": "e_verify_localize", "source": "verifier_draft_failure", "target": "tool_failure_localizer", "condition": "always"}, reason="localize the observed failure"),
            BoundedGraphEdit("add_edge", payload={"edge_id": "e_draft_boundary", "source": "verifier_draft_failure", "target": "tool_boundary_frames", "condition": "always"}, reason="provide original Wan pixels to boundary extraction"),
            BoundedGraphEdit("add_edge", payload={"edge_id": "e_localize_boundary", "source": "tool_failure_localizer", "target": "tool_boundary_frames", "condition": "always"}, reason="select boundary positions from the failure plan"),
            BoundedGraphEdit("add_edge", payload={"edge_id": "e_boundary_repair", "source": "tool_boundary_frames", "target": "tool_local_repair", "condition": "always"}, reason="condition local regeneration on a materialized boundary frame"),
            BoundedGraphEdit("add_edge", payload={"edge_id": "e_draft_stitch", "source": "verifier_draft_failure", "target": "tool_segment_stitch", "condition": "always"}, reason="retain the complete original draft as the composition base"),
            BoundedGraphEdit("add_edge", payload={"edge_id": "e_localize_stitch", "source": "tool_failure_localizer", "target": "tool_segment_stitch", "condition": "always"}, reason="tell the stitcher exactly which interval may change"),
            BoundedGraphEdit("add_edge", payload={"edge_id": "e_repair_stitch", "source": "tool_local_repair", "target": "tool_segment_stitch", "condition": "always"}, reason="splice the generated repair into the failed interval"),
        ]
        for edit in edits:
            graph = graph.apply_edit(edit)
        graph.triggers.extend(triggers)
        graph.validators.append(verifier)
        graph.stats["repair_composition"] = {
            "draft_generator": "mock_text_to_video",
            "verifier": verifier,
            "localizer": "failed_segment_localizer",
            "boundary_conditioner": "boundary_frame_extractor",
            "repair_generator": repair_name,
            "composer": "segment_stitcher",
            "preserve_healthy_content": True,
            "replace_whole_video": False,
        }
        graph.stats["mechanism_family"] = "localized_boundary_conditioned_segment_repair"
        graph.stats["healthy_content_preservation"] = True
        return GraphPathCandidate(
            graph,
            edits,
            reason,
            [item.value for item in failure.failure_types],
        )

    def _preferred_capability_tool(
        self,
        capability: str,
        *,
        fallback: str,
        preferred_inputs: set[str] | None = None,
    ) -> str:
        preferred_inputs = preferred_inputs or set()
        matches = self.tools.find_by_capability(capability)
        if not matches:
            return fallback
        ranked = sorted(
            matches,
            key=lambda spec: (
                spec.backend != "mcp",
                preferred_inputs.issubset(set(spec.input_types)),
                -float(spec.estimated_cost),
            ),
            reverse=True,
        )
        return str(ranked[0].name)

    def _candidate_style_transfer_graph(self, failure: FailureReport) -> GraphPathCandidate:
        graph = self.baseline_graph()
        graph.graph_id = "style_transfer_v2v_graph"
        graph.skill_name = "style_transfer_v2v_graph"
        graph.description = (
            "Load the task source video, apply a verified source-preserving V2V "
            "style transfer, then stabilize the result with temporal deflicker."
        )
        style_name = self._preferred_capability_tool(
            "video_style_transfer",
            fallback="mock_video_style_transfer",
            preferred_inputs={"video"},
        )
        deflicker_name = self._preferred_capability_tool(
            "temporal_deflickering",
            fallback="temporal_deflicker",
            preferred_inputs={"video"},
        )
        edits = [
            BoundedGraphEdit(
                "replace_node",
                target="tool_t2v",
                payload={
                    "node_id": "tool_t2v",
                    "node_type": "tool",
                    "name": "task_reference_video",
                    "config": {"cost": 0.05},
                },
                reason="use the benchmark source video instead of regenerating it with T2V",
            ),
            BoundedGraphEdit("add_node", payload={"node_id": "tool_v2v_style", "node_type": "tool", "name": style_name, "config": {"cost": 2.0}}, reason="apply verified V2V style transfer to the source video"),
            BoundedGraphEdit("add_node", payload={"node_id": "tool_deflicker", "node_type": "tool", "name": deflicker_name, "config": {"cost": 0.45}}, reason="stabilize the stylized pixels with a verified deflicker runtime"),
        ]
        for edit in edits:
            graph = graph.apply_edit(edit)
        graph.edges.extend(
            [
                GraphEdge("e_source_style", "tool_t2v", "tool_v2v_style"),
                GraphEdge("e_style_deflicker", "tool_v2v_style", "tool_deflicker"),
            ]
        )
        graph.triggers.extend(["style_drift", "temporal_flicker", "anime", "style transfer"])
        graph.validators.extend(["vbench_dimension_alignment", "background_preservation"])
        return GraphPathCandidate(graph, edits, "Use V2V style transfer plus deflicker for style or flicker failures.", [item.value for item in failure.failure_types])

    def _candidate_region_edit_graph(self, failure: FailureReport) -> GraphPathCandidate:
        graph = self.baseline_graph()
        graph.graph_id = "region_constrained_edit_graph"
        graph.skill_name = "region_constrained_edit_graph"
        region_matches = self.tools.find_by_capability("region_video_editing")
        if region_matches:
            editor_name = self._preferred_capability_tool(
                "region_video_editing",
                fallback="mock_region_video_editor",
                preferred_inputs={"video"},
            )
            graph.description = (
                "Load the task source video, track the target, and apply a verified "
                "region-aware editor while preserving non-target content."
            )
        else:
            editor_name = self._preferred_capability_tool(
                "global_video_editing",
                fallback="mock_global_video_editor",
                preferred_inputs={"video"},
            )
            graph.description = (
                "Load the task source video and apply the strongest verified source-video "
                "editor with explicit non-target preservation constraints."
            )
        edits = [
            BoundedGraphEdit(
                "replace_node",
                target="tool_t2v",
                payload={
                    "node_id": "tool_t2v",
                    "node_type": "tool",
                    "name": "task_reference_video",
                    "config": {"cost": 0.05},
                },
                reason="edit the benchmark source video instead of creating an unrelated T2V clip",
            ),
            BoundedGraphEdit("add_node", payload={"node_id": "tool_region_edit", "node_type": "tool", "name": editor_name, "config": {"cost": 1.6}}, reason="apply the best verified source-video editor"),
            BoundedGraphEdit("add_validator", payload={"validator": "background_preservation"}, reason="verify non-target preservation"),
        ]
        if region_matches:
            edits.insert(
                1,
                BoundedGraphEdit("add_node", payload={"node_id": "tool_object_tracker", "node_type": "tool", "name": "object_tracker", "config": {"cost": 0.45}}, reason="track the editable target region"),
            )
        for edit in edits:
            graph = graph.apply_edit(edit)
        if region_matches:
            graph.edges.extend(
                [
                    GraphEdge("e_source_track", "tool_t2v", "tool_object_tracker"),
                    GraphEdge("e_source_region_edit", "tool_t2v", "tool_region_edit"),
                    GraphEdge("e_track_region_edit", "tool_object_tracker", "tool_region_edit"),
                ]
            )
        else:
            graph.edges.append(
                GraphEdge("e_source_global_edit", "tool_t2v", "tool_region_edit")
            )
            graph.stats["region_edit_fallback"] = "source_preserving_global_video_editing"
        graph.triggers.extend(["editing_leakage", "unchanged", "only target"])
        graph.validators.extend(["background_preservation", "target_edit_success"])
        return GraphPathCandidate(
            graph,
            edits,
            "Use source-conditioned editing for edit leakage and non-target preservation.",
            [item.value for item in failure.failure_types],
        )

    def _validation_tasks(self, tasks: list[VideoTask], candidate: GraphPathCandidate) -> list[VideoTask]:
        validation_tasks = list(tasks)
        parent = set(candidate.parent_failure_types)
        if FailureType.IDENTITY_DRIFT.value in parent or FailureType.CLOTHING_COLOR_DRIFT.value in parent:
            validation_tasks.append(
                VideoTask(
                    "graph-heldout-identity-detective",
                    "The same detective in a red scarf walks through three scenes, turns, and waves while keeping the same face and scarf.",
                )
            )
        if FailureType.MOTION_MISMATCH.value in parent:
            validation_tasks.append(
                VideoTask(
                    "graph-heldout-motion-chef",
                    "A chef places a tomato on a board, slices it, pushes it into a pan, and stirs in the correct order.",
                )
            )
        if FailureType.EDITING_LEAKAGE.value in parent:
            validation_tasks.append(
                VideoTask(
                    "graph-heldout-region-edit",
                    "Change only the red car to blue while keeping every person, building, and background region unchanged.",
                    mode=TaskMode.EDITING,
                )
            )
        deduped = []
        seen = set()
        for task in validation_tasks:
            if task.task_id in seen:
                continue
            seen.add(task.task_id)
            deduped.append(task)
        return deduped

    @staticmethod
    def _looks_multishot(task: VideoTask) -> bool:
        text = task.prompt.lower()
        return any(token in text for token in ["scene", "shot", "story", "then", "across", "three"])

    @staticmethod
    def _dedupe_candidates(candidates: list[GraphPathCandidate]) -> list[GraphPathCandidate]:
        seen = set()
        deduped = []
        for candidate in candidates:
            payload = candidate.graph.to_dict()
            for key in ("graph_id", "skill_name", "description", "created_at", "updated_at", "stats"):
                payload.pop(key, None)
            signature = str(payload)
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(candidate)
        return deduped

    @staticmethod
    def _incoming_condition(graph: ToolPathGraph, node_id: str) -> str:
        conditions = [edge.condition for edge in graph.edges if edge.target == node_id]
        return ",".join(conditions) if conditions else "entry"
