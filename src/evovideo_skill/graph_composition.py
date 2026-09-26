from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from evovideo_skill.graph_evolver import GraphPathCandidate, GraphPathRollout
from evovideo_skill.graph_skill import BoundedGraphEdit, GraphEdge, GraphNode, ToolPathGraph
from evovideo_skill.weighted_tool_graph import WeightedToolGraphMemory, task_class


class GraphCompositionError(ValueError):
    pass


def merge_tool_paths(
    paths: list[list[str]],
    name: str,
    triggers: list[str] | None = None,
    cost_by_tool: dict[str, float] | None = None,
) -> ToolPathGraph:
    clean_paths = [list(dict.fromkeys(tool for tool in path if tool)) for path in paths]
    clean_paths = [path for path in clean_paths if path]
    if not clean_paths:
        raise GraphCompositionError("cannot merge empty tool paths")
    tools = list(dict.fromkeys(tool for path in clean_paths for tool in path))
    node_id = {tool: f"tool_{_safe(tool)}" for tool in tools}
    nodes = [GraphNode("trigger_task", "trigger", "task_start", {"cost": 0.0})]
    nodes.extend(
        GraphNode(
            node_id[tool],
            "tool",
            tool,
            {"cost": float((cost_by_tool or {}).get(tool, 0.25)), "shared_node": True},
        )
        for tool in tools
    )
    edge_pairs: list[tuple[str, str]] = []
    seen_tools: set[str] = set()
    previous_terminal: str | None = None
    for path in clean_paths:
        overlap = seen_tools & set(path)
        if previous_terminal is not None and not overlap:
            edge_pairs.append((node_id[previous_terminal], node_id[path[0]]))
        else:
            edge_pairs.append(("trigger_task", node_id[path[0]]))
        edge_pairs.extend((node_id[source], node_id[target]) for source, target in zip(path, path[1:]))
        seen_tools.update(path)
        previous_terminal = path[-1]
    edge_pairs = list(dict.fromkeys(edge_pairs))
    edges = [
        GraphEdge(f"edge_{index:03d}", source, target, "always", {"merged_path_support": 1})
        for index, (source, target) in enumerate(edge_pairs)
    ]
    graph = ToolPathGraph(
        graph_id=name,
        skill_name=name,
        description="Merged historical tool paths with canonical shared tool nodes.",
        triggers=list(dict.fromkeys(triggers or [])),
        nodes=nodes,
        edges=edges,
        stats={"composition_source": "historical_path_merge", "merged_paths": clean_paths},
    )
    _validate_acyclic(graph)
    return graph


@dataclass
class HistoricalPathComposer:
    weighted_graph: WeightedToolGraphMemory
    max_candidates: int = 2
    tool_registry: object | None = None

    def propose(self, rollouts: list[GraphPathRollout]) -> list[GraphPathCandidate]:
        candidates: list[GraphPathCandidate] = []
        for rollout in rollouts:
            if rollout.failure is None:
                continue
            current = rollout.artifact.tool_chain or rollout.graph.tool_names()
            category = task_class(rollout.task)
            for donor in self.weighted_graph.top_paths(category, limit=self.max_candidates + 2):
                donor_tools = list(donor.get("tools", []))
                if not donor_tools or donor_tools == current or set(donor_tools) <= set(current):
                    continue
                digest = hashlib.sha1("::".join(current + donor_tools).encode("utf-8")).hexdigest()[:8]
                name = f"merged_{category}_{digest}"
                try:
                    cost_by_tool = self._tool_costs(rollout.graph, donor_tools)
                    graph = merge_tool_paths(
                        [current, donor_tools],
                        name=name,
                        triggers=[*rollout.graph.triggers, category, *[item.value for item in rollout.failure.failure_types]],
                        cost_by_tool=cost_by_tool,
                    )
                except GraphCompositionError:
                    continue
                graph.stats.update(
                    {
                        "donor_path_id": donor.get("path_id"),
                        "donor_ucb": donor.get("ucb", 0.0),
                        "donor_mean_advantage": donor.get("mean_advantage", 0.0),
                    }
                )
                graph.fallbacks.append(str(donor.get("path_id", "historical_path")))
                edits = [
                    BoundedGraphEdit(
                        "add_fallback",
                        payload={"fallback": str(donor.get("path_id", "historical_path"))},
                        reason="compose the failed parent with a high-UCB historical path",
                    )
                ]
                candidates.append(
                    GraphPathCandidate(
                        graph,
                        edits,
                        "Merge the current failed path with a high-value historical path while sharing common tools.",
                        [item.value for item in rollout.failure.failure_types],
                    )
                )
                if len(candidates) >= self.max_candidates:
                    return candidates
        return candidates

    def _tool_costs(self, source_graph: ToolPathGraph, donor_tools: list[str]) -> dict[str, float]:
        costs = {
            node.name: float(node.config.get("cost", 0.25))
            for node in source_graph.nodes
            if node.node_type == "tool"
        }
        registry = self.tool_registry
        if registry is not None:
            for tool in donor_tools:
                if tool in costs or not registry.has(tool):
                    continue
                costs[tool] = float(registry.spec(tool).estimated_cost)
        return costs


def _validate_acyclic(graph: ToolPathGraph) -> None:
    nodes = {node.node_id for node in graph.nodes}
    indegree = {node: 0 for node in nodes}
    outgoing = {node: [] for node in nodes}
    for edge in graph.edges:
        if edge.source not in nodes or edge.target not in nodes:
            raise GraphCompositionError(f"dangling merged edge {edge.source}->{edge.target}")
        if edge.source == edge.target:
            raise GraphCompositionError("shared-node merge produced a self cycle")
        indegree[edge.target] += 1
        outgoing[edge.source].append(edge.target)
    ready = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for target in outgoing[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if visited != len(nodes):
        raise GraphCompositionError("historical paths have conflicting tool order and cannot form a DAG")


def _safe(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "tool"
