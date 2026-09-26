from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from evovideo_skill.models import SkillCard, utc_now


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    name: str
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: "GraphNode | dict[str, Any]") -> "GraphNode":
        if isinstance(value, cls):
            return value
        payload = dict(value)
        nested = payload.pop("node", None)
        if isinstance(nested, dict):
            for key, item in nested.items():
                payload.setdefault(key, item)
        allowed = {"node_id", "node_type", "name", "config"}
        payload = {key: item for key, item in payload.items() if key in allowed}
        payload.setdefault("config", {})
        return cls(**payload)


@dataclass
class GraphEdge:
    edge_id: str
    source: str
    target: str
    condition: str = "always"
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: "GraphEdge | dict[str, Any]") -> "GraphEdge":
        if isinstance(value, cls):
            return value
        payload = dict(value)
        nested = payload.pop("edge", None)
        if isinstance(nested, dict):
            for key, item in nested.items():
                payload.setdefault(key, item)
        allowed = {"edge_id", "source", "target", "condition", "config"}
        payload = {key: item for key, item in payload.items() if key in allowed}
        payload.setdefault("condition", "always")
        payload.setdefault("config", {})
        return cls(**payload)


@dataclass
class BoundedGraphEdit:
    op: str
    target: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class ToolPathGraph:
    graph_id: str
    skill_name: str
    description: str
    triggers: list[str]
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    validators: list[str] = field(default_factory=list)
    fallbacks: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    stats: dict[str, Any] = field(default_factory=dict)

    def node(self, node_id: str) -> GraphNode:
        for item in self.nodes:
            if item.node_id == node_id:
                return item
        raise KeyError(f"node not found: {node_id}")

    def tool_names(self) -> list[str]:
        return [node.name for node in self.nodes if node.node_type == "tool"]

    def executable_tool_names(self, available: set[str] | None = None) -> list[str]:
        names = self.tool_names()
        if available is not None:
            names = [name for name in names if name in available]
        return names

    def estimated_cost(self) -> float:
        node_cost = sum(float(node.config.get("cost", 0.25)) for node in self.nodes)
        edge_cost = 0.05 * len(self.edges)
        return node_cost + edge_cost

    def apply_edit(self, edit: BoundedGraphEdit) -> "ToolPathGraph":
        graph = ToolPathGraph.from_dict(self.to_dict())
        graph.updated_at = utc_now()
        if edit.op == "add_node":
            graph.nodes.append(GraphNode.from_dict(edit.payload))
        elif edit.op == "delete_node":
            graph.nodes = [node for node in graph.nodes if node.node_id != edit.target]
            graph.edges = [edge for edge in graph.edges if edge.source != edit.target and edge.target != edit.target]
        elif edit.op == "replace_node":
            replacement = GraphNode.from_dict(edit.payload)
            graph.nodes = [replacement if node.node_id == edit.target else node for node in graph.nodes]
            for edge in graph.edges:
                if edge.source == edit.target:
                    edge.source = replacement.node_id
                if edge.target == edit.target:
                    edge.target = replacement.node_id
        elif edit.op == "add_edge":
            graph.edges.append(GraphEdge.from_dict(edit.payload))
        elif edit.op == "delete_edge":
            graph.edges = [edge for edge in graph.edges if edge.edge_id != edit.target]
        elif edit.op == "tighten_trigger":
            condition = str(edit.payload.get("trigger", "")).strip()
            if condition and condition not in graph.triggers:
                graph.triggers.append(condition)
        elif edit.op == "add_validator":
            validator = str(edit.payload.get("validator") or edit.target or "").strip()
            if validator and validator not in graph.validators:
                graph.validators.append(validator)
        elif edit.op == "add_fallback":
            fallback = str(edit.payload.get("fallback", "")).strip()
            if fallback and fallback not in graph.fallbacks:
                graph.fallbacks.append(fallback)
        elif edit.op == "change_threshold":
            target = str(edit.target or edit.payload.get("validator") or "").strip()
            matching = next((node for node in graph.nodes if node.node_id == target), None)
            if matching is not None:
                matching.config["threshold"] = edit.payload["threshold"]
            elif target:
                if target not in graph.validators:
                    graph.validators.append(target)
                graph.stats.setdefault("validator_thresholds", {})[target] = float(
                    edit.payload["threshold"]
                )
            else:
                raise ValueError("change_threshold requires a node or validator target")
        else:
            raise ValueError(f"unsupported bounded graph edit op: {edit.op}")
        return graph

    def to_skill_card(self) -> SkillCard:
        tools = self.executable_tool_names()
        return SkillCard(
            skill_name=self.skill_name,
            version="graph-1.0",
            description=self.description,
            triggers=self.triggers,
            failure_conditions=list(self.triggers),
            inputs=["video_task", "tool_path_graph", "state"],
            procedure=[
                "retrieve the graph-structured tool path for matching triggers",
                "execute graph nodes with state checkpoints and artifact caching",
                "run verifier nodes and follow repair branches when thresholds fail",
            ],
            tools=tools,
            evaluators=self.validators,
            fallbacks=self.fallbacks,
            anti_patterns=[
                "do not append expensive tools without held-out validation",
                "do not rewrite the whole skill when a bounded node or edge edit is sufficient",
            ],
            memory={"success_cases": [], "failure_cases": []},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolPathGraph":
        payload = dict(data)
        payload["nodes"] = [GraphNode.from_dict(node) for node in payload.get("nodes", [])]
        payload["edges"] = [GraphEdge.from_dict(edge) for edge in payload.get("edges", [])]
        return cls(**payload)


@dataclass
class GraphExecutionEvent:
    node_id: str
    node_name: str
    event_type: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoGenerationState:
    task_id: str
    prompt: str
    active_graph_id: str
    artifacts: list[str] = field(default_factory=list)
    verifier_scores: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    budget_used: float = 0.0
    events: list[GraphExecutionEvent] = field(default_factory=list)

    def record(self, node: GraphNode, event_type: str, status: str, metadata: dict[str, Any] | None = None) -> None:
        self.events.append(GraphExecutionEvent(node.node_id, node.name, event_type, status, metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperienceRecord:
    task_id: str
    prompt: str
    failure_types: list[str]
    graph_id: str
    tool_path: list[str]
    score: float
    cost: float
    accepted: bool
    evidence: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GraphSkillMemory:
    def __init__(self, memory_dir: str | Path):
        self.memory_dir = Path(memory_dir)
        self.graph_dir = self.memory_dir / "graph_skills"
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.experience_path = self.graph_dir / "experience_graph.json"

    def upsert_graph(self, graph: ToolPathGraph) -> None:
        graph.updated_at = utc_now()
        path = self.graph_dir / f"{graph.skill_name}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(graph.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    def list_graphs(self) -> list[ToolPathGraph]:
        graphs = []
        for path in sorted(self.graph_dir.glob("*.json")):
            if path == self.experience_path:
                continue
            with path.open("r", encoding="utf-8") as handle:
                graphs.append(ToolPathGraph.from_dict(json.load(handle)))
        return graphs

    def append_experience(self, record: ExperienceRecord) -> None:
        records = self.list_experiences()
        records.append(record)
        with self.experience_path.open("w", encoding="utf-8") as handle:
            json.dump([item.to_dict() for item in records], handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    def list_experiences(self) -> list[ExperienceRecord]:
        if not self.experience_path.exists():
            return []
        with self.experience_path.open("r", encoding="utf-8") as handle:
            return [ExperienceRecord(**item) for item in json.load(handle)]

    def retrieve_procedural_paths(self, prompt: str, failure_types: list[str], limit: int = 5) -> list[ExperienceRecord]:
        prompt_terms = set(prompt.lower().split())
        wanted = set(failure_types)

        def rank(record: ExperienceRecord) -> tuple[float, float]:
            overlap = len(prompt_terms & set(record.prompt.lower().split()))
            failure_overlap = len(wanted & set(record.failure_types))
            return (float(record.accepted) + failure_overlap + overlap * 0.01, record.score - record.cost * 0.01)

        return sorted(self.list_experiences(), key=rank, reverse=True)[:limit]


def graph_score_mean(scores: list[float]) -> float:
    return mean(scores) if scores else 0.0


def graph_score_stability(scores: list[float]) -> float:
    if len(scores) <= 1:
        return 1.0
    return max(0.0, 1.0 - pstdev(scores))
