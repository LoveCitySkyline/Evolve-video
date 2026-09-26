from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from evovideo_skill.graph_skill import GraphSkillMemory, ToolPathGraph
from evovideo_skill.models import FailureType, VideoTask
from evovideo_skill.weighted_tool_graph import task_class


@dataclass
class RoutedToolPath:
    graph: ToolPathGraph
    score: float
    reason: str
    expanded_tools: list[str]
    estimated_cost: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph": self.graph.to_dict(),
            "score": self.score,
            "reason": self.reason,
            "expanded_tools": self.expanded_tools,
            "estimated_cost": self.estimated_cost,
            "metadata": self.metadata,
        }


class FoundationAwareGraphRouter:
    """Route unseen tasks through learned foundation and task-level graph skills."""

    def __init__(self, graph_memory: GraphSkillMemory, cost_weight: float = 0.08, foundation_bonus: float = 0.2):
        self.graph_memory = graph_memory
        self.cost_weight = cost_weight
        self.foundation_bonus = foundation_bonus

    def route(
        self,
        task: VideoTask,
        failure_types: list[FailureType] | list[str] | None = None,
        top_k: int = 3,
    ) -> list[RoutedToolPath]:
        failures = {item.value if isinstance(item, FailureType) else str(item) for item in (failure_types or [])}
        prompt_terms = set(task.prompt.lower().replace("-", " ").split())
        category = task_class(task)
        routes: list[RoutedToolPath] = []
        for graph in self.graph_memory.list_graphs():
            if graph.skill_name == "baseline_t2v_graph":
                continue
            score, reasons = self._score_graph(graph, prompt_terms, failures, category)
            if score <= 0:
                continue
            routes.append(
                RoutedToolPath(
                    graph=graph,
                    score=score,
                    reason="; ".join(reasons),
                    expanded_tools=graph.tool_names(),
                    estimated_cost=graph.estimated_cost(),
                    metadata={
                        "foundation_skill": bool(graph.stats.get("foundation_skill")),
                        "quality_gain": graph.stats.get("quality_gain", 0.0),
                        "stability": graph.stats.get("stability", 1.0),
                        "utility": graph.stats.get("utility", 0.0),
                    },
                )
            )
        return sorted(routes, key=lambda item: (item.score, -item.estimated_cost), reverse=True)[:top_k]

    def _score_graph(self, graph: ToolPathGraph, prompt_terms: set[str], failures: set[str], category: str) -> tuple[float, list[str]]:
        reasons: list[str] = []
        triggers = [trigger.lower() for trigger in graph.triggers]
        trigger_terms = set(" ".join(triggers).replace("-", " ").split())
        trigger_overlap = len(prompt_terms & trigger_terms)
        if trigger_overlap:
            reasons.append(f"trigger_overlap={trigger_overlap}")
        failure_overlap = len(failures & set(graph.triggers)) + len(failures & set(graph.stats.get("source_failure_types", [])))
        if failure_overlap:
            reasons.append(f"failure_overlap={failure_overlap}")
        capability_overlap = self._capability_overlap(graph, prompt_terms)
        if capability_overlap:
            reasons.append(f"capability_overlap={capability_overlap}")
        quality_gain = float(graph.stats.get("quality_gain", graph.stats.get("avg_gain", 0.0)))
        stability = float(graph.stats.get("stability", 1.0))
        utility = float(graph.stats.get("utility", 0.0))
        foundation_bonus = self.foundation_bonus if graph.stats.get("foundation_skill") else 0.0
        category_match = 1.0 if graph.stats.get("task_class") == category else 0.0
        path_ucb = float(graph.stats.get("path_ucb", 0.0))
        score = (
            trigger_overlap * 0.4
            + failure_overlap * 0.8
            + capability_overlap * 0.6
            + quality_gain * 2.0
            + stability * 0.2
            + utility
            + foundation_bonus
            + category_match
            + 0.2 * path_ucb
            - self.cost_weight * graph.estimated_cost()
        )
        if graph.stats.get("foundation_skill"):
            reasons.append("foundation_skill")
        if quality_gain:
            reasons.append(f"quality_gain={quality_gain:.3f}")
        if category_match:
            reasons.append(f"task_class={category}")
        if not reasons and graph.stats.get("foundation_skill"):
            reasons.append("default_foundation_candidate")
        return score, reasons

    @staticmethod
    def _capability_overlap(graph: ToolPathGraph, prompt_terms: set[str]) -> int:
        tools = " ".join(graph.tool_names()).lower()
        overlap = 0
        if prompt_terms & {"identity", "character", "same", "face", "clothing", "coat"} and any(token in tools for token in ["i2v", "image_to_video", "character"]):
            overlap += 1
        if prompt_terms & {"anime", "style", "cartoon"} and "style" in tools:
            overlap += 1
        if prompt_terms & {"action", "motion", "order", "then", "segment"} and any(token in tools for token in ["keyframe", "segment", "image_to_video"]):
            overlap += 1
        if prompt_terms & {"only", "region", "target", "background", "unchanged"} and any(token in tools for token in ["region", "tracker", "editor"]):
            overlap += 1
        if prompt_terms & {"scene", "shot", "story", "multi"} and any(token in tools for token in ["scene", "multi_shot", "character_sheet"]):
            overlap += 1
        return overlap
