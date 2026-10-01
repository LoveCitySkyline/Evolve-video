"""Controls shared by the three graph-evolution research questions."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

from evovideo_skill.graph_evolver import H3_GENERATORS, GraphPathRollout, validate_h3_node_configs
from evovideo_skill.graph_skill import GraphEdge, GraphNode, ToolPathGraph
from evovideo_skill.research_subgraphs import stable_hash
from evovideo_skill.weighted_tool_graph import h3_local_active, task_class


ARMS = {
    "prompt": ("prompt", "none", "local"),
    "composition": ("composition", "whole", "local"),
    "graph_none": ("graph", "none", "local"),
    "graph_whole": ("graph", "whole", "local"),
    "graph_subgraph": ("graph", "subgraph", "local"),
    "graph_scalar": ("graph", "subgraph", "scalar"),
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def append_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        stream.flush()


def applicability(task) -> str:
    # Exact duration/shot shape prevents accidental reuse of a six-second program
    # as an eighteen-second program. Physical input IDs never become shared state.
    return json.dumps([task_class(task), task.mode.value, task.duration_seconds,
                       [s.get("duration_seconds") for s in task.metadata.get("h3_shots", [])],
                       bool(task.reference_video), bool(task.metadata.get("h3_audio_criteria"))])


def feedback_view(rollout: GraphPathRollout, mode: str) -> dict:
    if mode not in {"scalar", "local"}:
        raise ValueError("unknown feedback mode")
    result = {"score": rollout.score}
    if mode == "scalar":
        return result
    metadata = rollout.artifact.metadata
    vlm = metadata.get("vlm_evaluation", {})
    if not isinstance(vlm, dict):
        vlm = {}
    result.update(
        metrics={m.name: {"score": m.score, "evidence": m.evidence}
                 for m in rollout.evaluation.active_metrics},
        criterion_evidence=deepcopy(vlm.get("criterion_evidence", {})),
        failed_segments=deepcopy(vlm.get("failed_segments", [])),
        nodes=deepcopy(metadata.get("node_evidence", {})),
        evidence_status="observed" if vlm.get("criterion_evidence") else "unknown",
        attribution="Temporal localization is verifier evidence, not proof that a node caused the failure.",
    )
    return result


def graph_payload(graph: ToolPathGraph) -> dict:
    """Exclude training statistics and free-form descriptions from scalar arms."""
    return {"nodes": [asdict(n) for n in graph.nodes], "edges": [asdict(e) for e in graph.edges]}


def composition_template(baseline: ToolPathGraph) -> ToolPathGraph:
    graph = ToolPathGraph.from_dict(deepcopy(baseline.to_dict()))
    graph.graph_id = graph.skill_name = "fixed_temporal_composition"
    graph.nodes.insert(1, GraphNode("fixed_plan", "tool", "temporal_decomposer", {}))
    graph.edges = [GraphEdge("composition_0", "trigger_generation", "fixed_plan"),
                   GraphEdge("composition_1", "fixed_plan", "tool_t2v")]
    return graph


def decode_graph(value: dict) -> ToolPathGraph:
    if not isinstance(value, dict) or set(value) != {"nodes", "edges"}:
        raise ValueError("graph must contain only nodes and edges; reward/validators/task edits are forbidden")
    nodes, edges = value["nodes"], value["edges"]
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("nodes/edges must be arrays")
    for node in nodes:
        if set(node) - {"node_id", "node_type", "name", "config"}:
            raise ValueError("unsupported node fields")
    graph = ToolPathGraph("candidate", "candidate", "Research candidate", [],
                          [GraphNode.from_dict(n) for n in nodes], [GraphEdge.from_dict(e) for e in edges])
    graph.graph_id = graph.skill_name = "research_" + stable_hash(value)[:16]
    return graph


def validate_candidate(graph: ToolPathGraph, parent: ToolPathGraph, executor,
                       search: str, max_nodes: int, max_edits: int) -> None:
    if not graph.nodes or len(graph.nodes) > max_nodes:
        raise ValueError("node budget exceeded or empty graph")
    ids = [n.node_id for n in graph.nodes]
    if len(set(ids)) != len(ids) or len({e.edge_id for e in graph.edges}) != len(graph.edges):
        raise ValueError("duplicate node/edge IDs")
    if any(n.node_type not in {"tool", "trigger"} for n in graph.nodes):
        raise ValueError("research arms use a fixed external verifier, no mutable verifier nodes")
    if any(e.source not in ids or e.target not in ids or e.condition != "always" for e in graph.edges):
        raise ValueError("research candidates must be closed, unconditional executable DAGs")
    if any(e.config for e in graph.edges):
        raise ValueError("edge config is not part of this research search space")
    old = {n.node_id: asdict(n) for n in parent.nodes}
    new = {n.node_id: asdict(n) for n in graph.nodes}
    edits = sum(old.get(k) != new.get(k) for k in old.keys() | new.keys())
    old_edges = {json.dumps(asdict(e), sort_keys=True) for e in parent.edges}
    new_edges = {json.dumps(asdict(e), sort_keys=True) for e in graph.edges}
    edits += len(old_edges ^ new_edges)
    if edits > max_edits:
        raise ValueError("bounded edit budget exceeded")
    if search in {"prompt", "composition"}:
        stripped = deepcopy(graph_payload(graph))
        target = deepcopy(graph_payload(parent))
        for payload in (stripped, target):
            for n in payload["nodes"]:
                n["config"].pop("prompt", None)
        if stripped != target:
            raise ValueError("fixed-topology arm may only change per-node prompt instructions")
        for node in graph.nodes:
            if "prompt" in node.config and executor.tools.spec(node.name).output_type != "video":
                raise ValueError("prompt instructions must be consumed by the generator")
    for node in graph.nodes:
        if "prompt" in node.config and (not isinstance(node.config["prompt"], str)
                                       or len(node.config["prompt"]) > 8000):
            raise ValueError("stage prompt must be a string of at most 8000 characters")
    executor.validate_graph(graph)
    video_sinks = [n for n in graph.nodes if n.node_type == "tool"
                   and not any(e.source == n.node_id for e in graph.edges)]
    if len(video_sinks) != 1 or executor.tools.spec(video_sinks[0].name).output_type != "video":
        sinks = [{"node_id": n.node_id, "tool": n.name,
                  "output_type": executor.tools.spec(n.name).output_type} for n in video_sinks]
        raise ValueError("research graph must have exactly one terminal video output; "
                         "all other nodes must feed it. Actual terminal tool nodes: " + json.dumps(sinks))
    # Every node must contribute to that output; no free dangling computations.
    reachable = {video_sinks[0].node_id}
    for _ in graph.nodes:
        reachable |= {e.source for e in graph.edges if e.target in reachable}
    if reachable != set(ids):
        raise ValueError("disconnected graph nodes")
    validate_h3_node_configs(graph, executor.tools.available_names(), local=h3_local_active(executor.tools))


def generation_credits(task, graph: ToolPathGraph) -> tuple[int, float]:
    """Conservative native-call/seconds reservations, not claimed GPU measurements."""
    calls, seconds = 0, 0.0
    for node in graph.nodes:
        if node.node_type != "tool" or node.name not in H3_GENERATORS | {"mock_text_to_video"}:
            continue
        duration = node.config.get("duration_seconds", task.duration_seconds)
        if "shot_index" in node.config:
            index = node.config["shot_index"]
            shots = task.metadata.get("h3_shots", [])
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(shots):
                raise ValueError("invalid shot index for this task")
            duration = node.config.get("duration_seconds", shots[index]["duration_seconds"])
        count = 1
        if node.name == "mock_text_to_video" and (duration > 15 or task.metadata.get("story_contract")) and "shot_index" not in node.config:
            shots = task.metadata.get("h3_shots", [])
            if not shots or sum(s["duration_seconds"] for s in shots) != duration:
                raise ValueError("long direct baseline needs declared shot durations")
            count = len(shots)
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("invalid generation duration")
        calls += count
        seconds += duration
    return calls, seconds


class ResearchBudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetLedger:
    max_calls: int
    max_seconds: float
    reserved_calls: int = 0
    reserved_seconds: float = 0.0

    def reserve(self, calls: int, seconds: float) -> None:
        if self.reserved_calls + calls > self.max_calls or self.reserved_seconds + seconds > self.max_seconds:
            raise ResearchBudgetExceeded("arm generation budget exhausted before executing the candidate")
        self.reserved_calls += calls
        self.reserved_seconds += seconds


def validate_splits(dataset) -> None:
    groups = [dataset.train, dataset.validation, dataset.test]
    if any(not group for group in groups):
        raise ValueError("research requires nonempty, disjoint train/validation/test splits")
    seen_ids, seen_groups = set(), set()
    for group in groups:
        ids = {t.task_id for t in group}
        scenarios = {str(t.metadata.get("scenario_group") or t.metadata.get("scenario_id") or t.task_id)
                     for t in group}
        if len(ids) != len(group) or ids & seen_ids or scenarios & seen_groups:
            raise ValueError("task or scenario leakage across research splits")
        seen_ids |= ids
        seen_groups |= scenarios
