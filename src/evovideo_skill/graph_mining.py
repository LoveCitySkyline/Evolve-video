from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import combinations
from statistics import mean, pstdev
from typing import Any

from evovideo_skill.graph_algorithms import betweenness_centrality, pagerank
from evovideo_skill.graph_skill import ExperienceRecord, ToolPathGraph


@dataclass
class WeightedToolEdge:
    source: str
    target: str
    weight: float
    support: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GlobalToolGraph:
    node_weights: dict[str, float]
    edge_weights: list[WeightedToolEdge]
    graph_count: int
    accepted_experience_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolMotif:
    motif_id: str
    tools: list[str]
    source_graph_ids: list[str]
    support: int
    avg_gain: float
    avg_score: float
    stability: float
    estimated_cost: float
    utility: float
    mdl_gain: float
    evidence: list[str] = field(default_factory=list)
    validation_task_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GraphMotifMiner:
    """Mine reusable high-utility tool motifs from accepted graph paths.

    The implementation is intentionally dependency-free. It treats each
    accepted graph skill as a short ordered tool path and mines contiguous
    subsequences, which is the common case for video tool chains. For richer
    DAGs, edge evidence is still included in the global weighted graph.
    """

    def __init__(
        self,
        min_support: int = 2,
        min_utility: float = 0.05,
        max_motif_size: int = 4,
        cost_weight: float = 0.08,
        mdl_weight: float = 0.02,
    ):
        self.min_support = min_support
        self.min_utility = min_utility
        self.max_motif_size = max_motif_size
        self.cost_weight = cost_weight
        self.mdl_weight = mdl_weight

    def build_global_tool_graph(self, graphs: list[ToolPathGraph], experiences: list[ExperienceRecord]) -> GlobalToolGraph:
        accepted_scores = self._accepted_scores(experiences)
        node_weights: Counter[str] = Counter()
        edge_weights: Counter[tuple[str, str]] = Counter()
        edge_support: Counter[tuple[str, str]] = Counter()
        for graph in graphs:
            weight = self._graph_weight(graph, accepted_scores)
            tools = self._tool_path(graph)
            for tool in tools:
                node_weights[tool] += weight
            for source, target in self._ordered_edges(graph, tools):
                edge_weights[(source, target)] += weight
                edge_support[(source, target)] += 1
        edges = [
            WeightedToolEdge(source, target, weight, edge_support[(source, target)])
            for (source, target), weight in sorted(edge_weights.items(), key=lambda item: (-item[1], item[0]))
        ]
        return GlobalToolGraph(dict(node_weights), edges, len(graphs), sum(1 for item in experiences if item.accepted))

    def mine(self, graphs: list[ToolPathGraph], experiences: list[ExperienceRecord]) -> list[ToolMotif]:
        accepted_graphs = self.evidence_backed_graphs(graphs, experiences)
        accepted_scores = self._accepted_scores(experiences)
        source_by_motif: dict[tuple[str, ...], set[str]] = defaultdict(set)
        gains_by_motif: dict[tuple[str, ...], list[float]] = defaultdict(list)
        scores_by_motif: dict[tuple[str, ...], list[float]] = defaultdict(list)
        costs_by_motif: dict[tuple[str, ...], list[float]] = defaultdict(list)
        tasks_by_motif: dict[tuple[str, ...], set[str]] = defaultdict(set)

        for graph in accepted_graphs:
            tools = self._tool_path(graph)
            gain = float(graph.stats.get("quality_gain", 0.0))
            score = float(graph.stats.get("candidate_score", accepted_scores.get(graph.skill_name, 0.0)))
            for motif in self._subsequences(tools):
                source_by_motif[motif].add(graph.skill_name)
                passed_tasks = graph.stats.get("passed_validation_task_ids", [])
                tasks_by_motif[motif].update(str(item) for item in passed_tasks)
                gains_by_motif[motif].append(gain)
                scores_by_motif[motif].append(score)
                costs_by_motif[motif].append(self._motif_cost(graph, motif))

        motifs: list[ToolMotif] = []
        for tools, source_graphs in source_by_motif.items():
            validation_tasks = tasks_by_motif[tools]
            support = len(validation_tasks)
            if support < self.min_support:
                continue
            avg_gain = mean(gains_by_motif[tools]) if gains_by_motif[tools] else 0.0
            avg_score = mean(scores_by_motif[tools]) if scores_by_motif[tools] else 0.0
            stability = self._stability(scores_by_motif[tools])
            cost = mean(costs_by_motif[tools]) if costs_by_motif[tools] else float(len(tools))
            mdl_gain = self._mdl_gain(len(tools), support)
            utility = support * avg_gain
            if utility < self.min_utility:
                continue
            motif_id = self._motif_id(tools)
            motifs.append(
                ToolMotif(
                    motif_id=motif_id,
                    tools=list(tools),
                    source_graph_ids=sorted(source_graphs),
                    support=support,
                    avg_gain=avg_gain,
                    avg_score=avg_score,
                    stability=stability,
                    estimated_cost=cost,
                    utility=utility,
                    mdl_gain=mdl_gain,
                    evidence=[f"support={support}", f"sources={','.join(sorted(source_graphs))}"],
                    validation_task_ids=sorted(validation_tasks),
                )
            )
        return sorted(
            motifs,
            key=lambda item: (item.utility, item.support, item.stability, item.mdl_gain),
            reverse=True,
        )

    @staticmethod
    def evidence_backed_graphs(
        graphs: list[ToolPathGraph],
        experiences: list[ExperienceRecord],
    ) -> list[ToolPathGraph]:
        accepted_graph_ids = {item.graph_id for item in experiences if item.accepted}
        return [
            graph
            for graph in graphs
            if graph.stats.get("accepted") is True
            and graph.skill_name in accepted_graph_ids
            and float(graph.stats.get("pass_rate", 0.0)) > 0.0
            and bool(graph.stats.get("passed_validation_task_ids"))
        ]

    def centrality(self, global_graph: GlobalToolGraph) -> dict[str, dict[str, float]]:
        outgoing: Counter[str] = Counter()
        incoming: Counter[str] = Counter()
        for edge in global_graph.edge_weights:
            outgoing[edge.source] += edge.weight
            incoming[edge.target] += edge.weight
        nodes = set(global_graph.node_weights) | set(outgoing) | set(incoming)
        rank = pagerank(global_graph)
        between = betweenness_centrality(global_graph)
        return {
            node: {
                "weighted_degree": float(global_graph.node_weights.get(node, 0.0) + outgoing[node] + incoming[node]),
                "pagerank": float(rank.get(node, 0.0)),
                "betweenness": float(between.get(node, 0.0)),
            }
            for node in sorted(nodes)
        }

    def _subsequences(self, tools: list[str]) -> list[tuple[str, ...]]:
        motifs: set[tuple[str, ...]] = set()
        upper = min(self.max_motif_size, len(tools))
        for size in range(1, upper + 1):
            for start in range(0, len(tools) - size + 1):
                motifs.add(tuple(tools[start : start + size]))
        if len(tools) <= self.max_motif_size:
            for size in range(2, len(tools) + 1):
                for combo in combinations(tools, size):
                    motifs.add(tuple(combo))
        return sorted(motifs)

    @staticmethod
    def _tool_path(graph: ToolPathGraph) -> list[str]:
        tools = graph.tool_names()
        if tools:
            return tools
        return graph.executable_tool_names()

    @staticmethod
    def _ordered_edges(graph: ToolPathGraph, tools: list[str]) -> list[tuple[str, str]]:
        node_name_by_id = {node.node_id: node.name for node in graph.nodes if node.node_type == "tool"}
        edges = []
        for edge in graph.edges:
            source = node_name_by_id.get(edge.source)
            target = node_name_by_id.get(edge.target)
            if source and target:
                edges.append((source, target))
        if edges:
            return edges
        return list(zip(tools, tools[1:]))

    @staticmethod
    def _accepted_scores(experiences: list[ExperienceRecord]) -> dict[str, float]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for item in experiences:
            if item.accepted:
                grouped[item.graph_id].append(item.score)
        return {graph_id: mean(scores) for graph_id, scores in grouped.items() if scores}

    @staticmethod
    def _graph_weight(graph: ToolPathGraph, accepted_scores: dict[str, float]) -> float:
        score = float(graph.stats.get("candidate_score", accepted_scores.get(graph.skill_name, 1.0)))
        gain = float(graph.stats.get("quality_gain", 0.0))
        stability = float(graph.stats.get("stability", 1.0))
        return max(0.01, score + gain) * max(0.01, stability)

    @staticmethod
    def _motif_cost(graph: ToolPathGraph, motif: tuple[str, ...]) -> float:
        costs = [float(node.config.get("cost", 0.25)) for node in graph.nodes if node.name in motif]
        return sum(costs) if costs else float(len(motif))

    @staticmethod
    def _stability(scores: list[float]) -> float:
        if len(scores) <= 1:
            return 1.0
        return max(0.0, 1.0 - pstdev(scores))

    @staticmethod
    def _mdl_gain(motif_size: int, support: int) -> float:
        if support <= 1:
            return 0.0
        original = motif_size * support
        compressed = motif_size + support
        return float(original - compressed)

    @staticmethod
    def _motif_id(tools: tuple[str, ...]) -> str:
        clean = [tool.replace("mock_", "").replace("_", "-") for tool in tools]
        return "motif_" + "__".join(clean)
