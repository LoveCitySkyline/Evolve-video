from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from evovideo_skill.graph_composition import GraphCompositionError, merge_tool_paths
from evovideo_skill.graph_skill import GraphSkillMemory, ToolPathGraph
from evovideo_skill.weighted_tool_graph import H3_COMPARISON_PROTOCOL, WeightedToolGraphMemory, graph_behavior_fingerprint
from evovideo_skill.tools import ToolRegistry


@dataclass
class OnlineFoundationConfig:
    min_support: int = 2
    min_advantage: float = 0.0
    min_stability: float = 0.8
    max_skills_per_category: int = 2
    max_tools_per_skill: int = 6


class TaskConditionedFoundationPromoter:
    """Propose reusable task-class paths from online node/edge/path credit."""

    def __init__(
        self,
        memory: WeightedToolGraphMemory,
        config: OnlineFoundationConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        graph_memory: GraphSkillMemory | None = None,
    ):
        self.memory = memory
        self.config = config or OnlineFoundationConfig()
        self.tool_registry = tool_registry
        self.graph_memory = graph_memory
        self.last_deferrals: list[dict[str, Any]] = []

    def propose(self) -> list[ToolPathGraph]:
        proposals: list[ToolPathGraph] = []
        self.last_deferrals = []
        originals = sorted(self.graph_memory.list_graphs(), key=lambda graph: graph.graph_id) if self.graph_memory is not None else []
        for category in sorted(self.memory.categories):
            reconstructable = []
            validated_sources: dict[str, ToolPathGraph] = {}
            for path in self.memory.top_paths(category, limit=30):
                tools = path["tools"]
                native_h3 = (
                    path.get("comparison_protocol") == H3_COMPARISON_PROTOCOL
                    or any(name.startswith("h3_") for name in tools)
                    or ("mock_text_to_video" in tools and self.tool_registry is not None
                        and bool({"h3_t2va", "h3_fl2va", "h3_ref2va"} & self.tool_registry.available_names()))
                )
                if native_h3:
                    source = next((graph for graph in originals
                                   if graph.graph_id in path["graph_ids"]
                                   and graph.stats.get("accepted") is True
                                   and isinstance(graph.stats.get("validation_seeds"), list)
                                   and len({seed for seed in graph.stats["validation_seeds"] if type(seed) is int}) >= 3
                                   and (path.get("graph_behavior_fingerprint") is None
                                        or path["graph_behavior_fingerprint"] == graph_behavior_fingerprint(graph))), None)
                    if source is not None and len(source.tool_names()) <= self.config.max_tools_per_skill:
                        validated_sources[path["path_id"]] = source
                        reconstructable.append(path)
                        continue
                    reason = "h3_native_foundation_requires_original_validated_graph" if source is None else "h3_native_foundation_exceeds_tool_limit"
                elif path.get("graph_behavior_fingerprint") is not None:
                    reason = "behavior_specific_foundation_requires_original_validated_graph"
                else:
                    reconstructable.append(path)
                    continue
                if reason:
                    self.last_deferrals.append({
                        "status": "deferred", "task_class": category, "path_id": path["path_id"],
                        "source_graph_ids": list(path["graph_ids"]),
                        "reason": reason,
                        "detail": "Require an accepted original graph with at least three validation replicates within the tool limit; never reconstruct or truncate node configs, repeated calls, or ordered reference bindings.",
                    })
            eligible = [
                path
                for path in reconstructable
                if path["uses"] >= self.config.min_support
                and path["mean_advantage"] >= self.config.min_advantage
                and path["stability"] >= self.config.min_stability
                and path["success_rate"] > 0.0
                and path.get("execution_error_rate", 0.0) == 0.0
                and (path["path_id"] in validated_sources or path.get("seed_control_rate") in {None, 1.0})
            ]
            eligible.sort(
                key=lambda item: (
                    item["mean_advantage"],
                    item["stability"],
                    item["uses"],
                ),
                reverse=True,
            )
            for path in eligible[: self.config.max_skills_per_category]:
                source = validated_sources.get(path["path_id"])
                tools = source.tool_names() if source is not None else list(path["tools"])[: self.config.max_tools_per_skill]
                if not tools:
                    continue
                digest = graph_behavior_fingerprint(source)[:12] if source is not None else hashlib.sha1("::".join(tools).encode("utf-8")).hexdigest()[:8]
                name = f"foundation_{_safe(category)}_{digest}"
                if source is not None:
                    graph = ToolPathGraph.from_dict(source.to_dict())
                    graph.graph_id = graph.skill_name = name
                    graph.stats.update({
                        "foundation_source_graph_id": source.graph_id,
                        "foundation_source_validation_seeds": list(source.stats["validation_seeds"]),
                        "comparison_protocol": H3_COMPARISON_PROTOCOL,
                    })
                else:
                    try:
                        graph = merge_tool_paths(
                            [tools],
                            name=name,
                            triggers=[category, "foundation_skill"],
                            cost_by_tool=self._cost_by_tool(tools, float(path["mean_cost"])),
                        )
                    except GraphCompositionError:
                        continue
                    graph.description = f"Task-conditioned foundation path for {category}: " + " -> ".join(tools)
                important = {
                    item["tool"]: item
                    for item in self.memory.important_nodes(category, limit=50)
                    if item["tool"] in tools
                }
                core_tools = [
                    tool
                    for tool in tools
                    if tool in important
                    and important[tool]["uses"] >= self.config.min_support
                    and important[tool]["mean_advantage"] >= self.config.min_advantage
                ]
                graph.stats.update(
                    {
                        "foundation_skill": True,
                        "online_foundation": True,
                        "task_class": category,
                        "support": path["uses"],
                        "source_graph_ids": path["graph_ids"],
                        "candidate_score": path["mean_quality"],
                        "quality_gain": path["mean_advantage"],
                        "stability": path["stability"],
                        "estimated_cost": path["mean_cost"],
                        "utility": path["mean_reward"],
                        "path_ucb": path["ucb"],
                        "seed_control_rate": path.get("seed_control_rate"),
                        "execution_error_rate": path.get("execution_error_rate", 0.0),
                        "node_credit": important,
                        "foundation_core_tools": core_tools,
                        "accepted": False,
                    }
                )
                proposals.append(graph)
        return proposals

    def _cost_by_tool(self, tools: list[str], path_cost: float) -> dict[str, float]:
        if self.tool_registry is not None:
            return {
                tool: float(self.tool_registry.spec(tool).estimated_cost)
                for tool in tools
                if self.tool_registry.has(tool)
            }
        edge_cost = 0.05 * len(tools)
        per_tool = max(0.0, path_cost - edge_cost) / max(1, len(tools))
        return {tool: per_tool for tool in tools}


def _safe(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "general"
