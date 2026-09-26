from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evovideo_skill.evolution_data import task_category
from evovideo_skill.graph_algorithms import betweenness_centrality, pagerank
from evovideo_skill.graph_skill import ToolPathGraph
from evovideo_skill.models import VideoTask, utc_now


H3_COMPARISON_PROTOCOL = "matched task/replicate, not matched random seed"
H3_LOCAL_COMPARISON_PROTOCOL = "matched_task_seed_replicates"
H3_LOCAL_EVALUATION_PROTOCOL = "h3_local_seeded_replicates_v1"


def h3_local_active(tool_registry: Any = None, runtime_settings: Any = None) -> bool:
    available = getattr(tool_registry, "available_names", lambda: set())()
    spec = getattr(tool_registry, "spec", None)
    if callable(spec) and any(spec(name).backend == "local-h3" for name in available):
        return True
    if isinstance(runtime_settings, dict):
        return any(runtime_settings.get(key) == "local-h3" for key in ("provider", "backend")) or any(
            h3_local_active(runtime_settings=runtime_settings.get(key)) for key in ("runtime", "settings")
        )
    return any(getattr(runtime_settings, key, None) == "local-h3" for key in ("provider", "backend"))


def h3_local_seed_applied(metadata: dict[str, Any], label: int | None) -> bool:
    """Seed capability alone is not evidence that this replicate used its seed."""
    return (
        type(label) is int
        and metadata.get("provider") == "local-h3"
        and metadata.get("provider_seed_control") is True
        and type(metadata.get("generation_seed")) is int
        and metadata["generation_seed"] == label
        and metadata.get("generation_seed_applied") is not False
    )


def graph_behavior_fingerprint(graph: ToolPathGraph) -> str:
    """Execution identity shared with the rollout cache, excluding bookkeeping."""
    occurrences: dict[tuple[str, str], int] = {}
    labels: dict[str, str] = {}
    nodes = []
    for node in graph.nodes:
        canonical_name = "__trigger__" if node.node_type == "trigger" else node.name
        key = (node.node_type, canonical_name)
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        label = f"{node.node_type}:{canonical_name}:{occurrence}"
        labels[node.node_id] = label
        semantic_config = {} if node.node_type == "trigger" else {
            key: value for key, value in node.config.items()
            if key not in {"cost", "backend", "provenance"}
        }
        nodes.append((label, semantic_config))
    edges = sorted(
        (labels.get(edge.source, edge.source), labels.get(edge.target, edge.target),
         edge.condition, edge.config)
        for edge in graph.edges
    )
    payload = {"nodes": sorted(nodes, key=lambda item: item[0]), "edges": edges}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def observed_tool_edges(graph: ToolPathGraph, state: Any, artifacts: dict[str, Any]) -> list[tuple[str, str]]:
    """Only credit declared tool edges whose artifacts were actually consumed.

    Verifier forwarding is deliberately omitted rather than inferred as a new edge.
    """
    inputs = {
        event.node_id: set(event.metadata.get("input_artifact_ids", []))
        for event in state.events if event.event_type == "tool" and event.status == "running"
    }
    nodes = {node.node_id: node for node in graph.nodes if node.node_type == "tool"}
    return [
        (nodes[edge.source].name, nodes[edge.target].name)
        for edge in graph.edges
        if edge.source in nodes and edge.target in nodes
        and edge.source in artifacts and edge.target in artifacts
        and artifacts[edge.source].artifact_id in inputs.get(edge.target, set())
    ]


def normalize_task_class(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "general"


def task_class(task: VideoTask) -> str:
    metadata = task.metadata or {}
    explicit = (
        metadata.get("task_class")
        or metadata.get("task_family")
        or metadata.get("category")
        or metadata.get("vbench_dimension")
    )
    if explicit:
        return normalize_task_class(str(explicit))
    failures = metadata.get("expected_failure_modes") or []
    if failures:
        return normalize_task_class(str(failures[0]))
    text = task.prompt.lower()
    rules = (
        ("multi_shot_identity", ("same character", "same face", "shot", "scene", "story")),
        ("ordered_motion", ("correct order", "then", "sequence", "motion", "action")),
        ("style_transfer", ("anime", "cartoon", "style transfer", "stylize")),
        ("regional_editing", ("change only", "unchanged", "region", "replace only")),
        ("long_video", ("long video", "continuation", "multiple scenes")),
    )
    for category, terms in rules:
        if any(term in text for term in terms):
            return category
    return normalize_task_class(task_category(task))


@dataclass
class CreditStats:
    uses: int = 0
    successes: int = 0
    total_quality: float = 0.0
    total_reward: float = 0.0
    total_advantage: float = 0.0
    total_cost: float = 0.0
    reward_square_sum: float = 0.0
    controlled_uses: int = 0
    uncontrolled_uses: int = 0
    execution_errors: int = 0

    def update(
        self,
        quality: float,
        reward: float,
        advantage: float,
        cost: float,
        success: bool,
        seed_controlled: bool | None = None,
        execution_error: bool = False,
    ) -> None:
        self.uses += 1
        self.successes += int(success)
        self.total_quality += quality
        self.total_reward += reward
        self.total_advantage += advantage
        self.total_cost += cost
        self.reward_square_sum += reward * reward
        self.controlled_uses += int(seed_controlled is True)
        self.uncontrolled_uses += int(seed_controlled is False)
        self.execution_errors += int(execution_error)

    @property
    def mean_quality(self) -> float:
        return self.total_quality / max(1, self.uses)

    @property
    def mean_reward(self) -> float:
        return self.total_reward / max(1, self.uses)

    @property
    def mean_advantage(self) -> float:
        return self.total_advantage / max(1, self.uses)

    @property
    def mean_cost(self) -> float:
        return self.total_cost / max(1, self.uses)

    @property
    def success_rate(self) -> float:
        return self.successes / max(1, self.uses)

    @property
    def seed_control_rate(self) -> float | None:
        measured = self.controlled_uses + self.uncontrolled_uses
        return self.controlled_uses / measured if measured else None

    @property
    def execution_error_rate(self) -> float:
        return self.execution_errors / max(1, self.uses)

    @property
    def stability(self) -> float:
        if self.uses <= 1:
            return 1.0
        variance = max(0.0, self.reward_square_sum / self.uses - self.mean_reward**2)
        return max(0.0, 1.0 - math.sqrt(variance))

    def ucb(self, total_uses: int, exploration_weight: float) -> float:
        exploration = exploration_weight * math.sqrt(math.log(max(2, total_uses)) / max(1, self.uses))
        return self.mean_reward + exploration

    def to_summary(self, total_uses: int, exploration_weight: float) -> dict[str, Any]:
        return {
            **asdict(self),
            "mean_quality": self.mean_quality,
            "mean_reward": self.mean_reward,
            "mean_advantage": self.mean_advantage,
            "mean_cost": self.mean_cost,
            "success_rate": self.success_rate,
            "seed_control_rate": self.seed_control_rate,
            "execution_error_rate": self.execution_error_rate,
            "stability": self.stability,
            "ucb": self.ucb(total_uses, exploration_weight),
        }


@dataclass
class PathCreditStats(CreditStats):
    tools: list[str] = field(default_factory=list)
    graph_ids: list[str] = field(default_factory=list)
    graph_behavior_fingerprint: str | None = None
    comparison_protocol: str | None = None

    def record_graph(self, graph_id: str) -> None:
        if graph_id not in self.graph_ids:
            self.graph_ids.append(graph_id)


@dataclass
class TaskClassGraph:
    nodes: dict[str, CreditStats] = field(default_factory=dict)
    edges: dict[str, CreditStats] = field(default_factory=dict)
    paths: dict[str, PathCreditStats] = field(default_factory=dict)
    tool_arenas: dict[str, dict[str, CreditStats]] = field(default_factory=dict)
    arena_families: dict[str, dict[str, Any]] = field(default_factory=dict)
    baseline_quality_total: float = 0.0
    baseline_quality_count: int = 0
    baseline_quality_by_context: dict[str, float] = field(default_factory=dict)
    total_rollouts: int = 0

    @property
    def baseline_quality(self) -> float:
        return self.baseline_quality_total / max(1, self.baseline_quality_count)


@dataclass
class _WeightedEdgeView:
    source: str
    target: str
    weight: float


@dataclass
class _WeightedGraphView:
    node_weights: dict[str, float]
    edge_weights: list[_WeightedEdgeView]


class WeightedToolGraphMemory:
    """Persistent task-conditioned quality-gain graph over paths and tools."""

    def __init__(
        self,
        path: str | Path,
        cost_weight: float = 0.03,
        exploration_weight: float = 0.15,
    ):
        self.path = Path(path)
        self.cost_weight = cost_weight
        self.exploration_weight = exploration_weight
        self.categories: dict[str, TaskClassGraph] = {}
        self.event_ids: set[str] = set()
        self._load()

    def record_rollout(
        self,
        task: VideoTask,
        graph_id: str,
        tools: list[str],
        quality: float,
        cost: float,
        success: bool,
        event_id: str | None = None,
        seed_controlled: bool | None = None,
        execution_error: bool = False,
        failed_tool: str | None = None,
        arena_variant: dict[str, Any] | None = None,
        evaluation_seed: int | None = None,
        positive_credit_allowed: bool = True,
        graph_behavior_fingerprint: str | None = None,
        actual_edges: list[tuple[str, str]] | None = None,
        comparison_protocol: str | None = None,
    ) -> bool:
        clean_tools = [tool for tool in tools if tool]
        if not clean_tools:
            return False
        category = task_class(task)
        event_id = event_id or self._event_id(task.task_id, graph_id, clean_tools, quality)
        if graph_behavior_fingerprint is not None:
            event_id = f"{event_id}:behavior={graph_behavior_fingerprint}:replicate={evaluation_seed}"
        if event_id in self.event_ids:
            return False
        self.event_ids.add(event_id)
        graph = self.categories.setdefault(category, TaskClassGraph())
        graph.total_rollouts += 1
        is_baseline = graph_id.startswith("baseline_t2v_graph") or graph_id == "base"
        context_key = self.context_key(task.task_id, evaluation_seed)
        if is_baseline:
            graph.baseline_quality_total += quality
            graph.baseline_quality_count += 1
            graph.baseline_quality_by_context[context_key] = quality
        paired_baseline = graph.baseline_quality_by_context.get(context_key)
        # Candidate credit is matched by task and evaluation label (a replicate,
        # not a controllable provider seed, for cloud H3). Using
        # a category mean can make a regressing path look positive merely
        # because it was evaluated on an easier task.
        raw_advantage = (
            0.0
            if is_baseline or paired_baseline is None
            else quality - paired_baseline
        )
        advantage = (
            raw_advantage
            if positive_credit_allowed or raw_advantage <= 0.0
            else 0.0
        )
        reward = advantage

        path_key = self.path_key(clean_tools, graph_behavior_fingerprint)
        path_stats = graph.paths.setdefault(path_key, PathCreditStats(
            tools=list(clean_tools), graph_behavior_fingerprint=graph_behavior_fingerprint,
        ))
        if comparison_protocol is None:
            if task.metadata.get("evaluation_protocol") == H3_LOCAL_EVALUATION_PROTOCOL:
                comparison_protocol = H3_LOCAL_COMPARISON_PROTOCOL
            elif any(name.startswith("h3_") for name in clean_tools):
                comparison_protocol = H3_COMPARISON_PROTOCOL
        if comparison_protocol is not None:
            path_stats.comparison_protocol = comparison_protocol
        path_stats.record_graph(graph_id)
        path_stats.update(quality, reward, advantage, cost, success, seed_controlled, execution_error)

        unique_tools = list(dict.fromkeys(clean_tools))
        attributed_tools = unique_tools
        if execution_error and failed_tool in unique_tools:
            # The path is unusable, but completed upstream tools did not produce
            # a quality observation and must not inherit the downstream crash.
            attributed_tools = [str(failed_tool)]
        node_credit = advantage / max(1, len(attributed_tools))
        node_cost = cost / max(1, len(attributed_tools))
        for tool in attributed_tools:
            stats = graph.nodes.setdefault(tool, CreditStats())
            stats.update(
                quality, node_credit, node_credit, node_cost, success, seed_controlled, execution_error
            )

        transitions = (
            list(zip(clean_tools, clean_tools[1:])) if actual_edges is None else
            [(source, target) for source, target in actual_edges
             if source in unique_tools and target in unique_tools]
        )
        attributed_transitions = transitions
        if execution_error and failed_tool in unique_tools:
            attributed_transitions = [
                (source, target) for source, target in transitions if target == failed_tool
            ]
        edge_credit = advantage / max(1, len(attributed_transitions))
        edge_cost = cost / max(1, len(attributed_transitions))
        for source, target in attributed_transitions:
            stats = graph.edges.setdefault(self.edge_key(source, target), CreditStats())
            stats.update(
                quality, edge_credit, edge_credit, edge_cost, success, seed_controlled, execution_error
            )
        if arena_variant:
            capability = str(arena_variant.get("capability") or "").strip()
            selected_tool = str(arena_variant.get("selected_tool") or "").strip()
            family_id = str(arena_variant.get("family_id") or "").strip()
            arena_observed = not execution_error or not failed_tool or selected_tool == failed_tool
            if capability and selected_tool and family_id and arena_observed:
                arena_stats = graph.tool_arenas.setdefault(capability, {}).setdefault(
                    selected_tool, CreditStats()
                )
                arena_stats.update(
                    quality,
                    advantage,
                    advantage,
                    cost,
                    success,
                    seed_controlled,
                    execution_error,
                )
                family = graph.arena_families.setdefault(
                    family_id,
                    {
                        "capability": capability,
                        "node_id": arena_variant.get("node_id"),
                        "compared_tools": list(arena_variant.get("compared_tools", [])),
                        "observations": 0,
                    },
                )
                family["observations"] = int(family.get("observations", 0)) + 1
        self.save()
        return True

    def path_prior(
        self, category: str, tools: list[str], graph_behavior_fingerprint: str | None = None,
    ) -> float:
        graph = self.categories.get(normalize_task_class(category))
        if graph is None:
            return 0.0
        exact = graph.paths.get(self.path_key(tools, graph_behavior_fingerprint))
        if exact is not None:
            return exact.ucb(graph.total_rollouts, self.exploration_weight)
        node_stats = [graph.nodes[tool] for tool in tools if tool in graph.nodes]
        composed = (
            sum(item.ucb(graph.total_rollouts, self.exploration_weight) for item in node_stats) / len(node_stats)
            if node_stats else 0.0
        )
        optimistic = max(
            (item.ucb(graph.total_rollouts, self.exploration_weight) for item in graph.paths.values()),
            default=graph.baseline_quality,
        ) + self.exploration_weight
        return max(composed + self.exploration_weight, optimistic)

    def top_paths(self, category: str, limit: int = 5) -> list[dict[str, Any]]:
        graph = self.categories.get(normalize_task_class(category))
        if graph is None:
            return []
        ranked = sorted(
            graph.paths.items(),
            key=lambda item: (
                item[1].ucb(graph.total_rollouts, self.exploration_weight),
                item[1].mean_advantage,
                item[1].success_rate,
            ),
            reverse=True,
        )
        return [
            {
                "path_id": path_id,
                "tools": stats.tools,
                "graph_ids": stats.graph_ids,
                **stats.to_summary(graph.total_rollouts, self.exploration_weight),
            }
            for path_id, stats in ranked[:limit]
        ]

    def important_nodes(self, category: str, limit: int = 10) -> list[dict[str, Any]]:
        graph = self.categories.get(normalize_task_class(category))
        if graph is None:
            return []
        centrality = self.centrality(category)
        ranked = sorted(
            graph.nodes.items(),
            key=lambda item: (
                item[1].mean_advantage * item[1].stability
                + 0.2 * centrality.get(item[0], {}).get("pagerank", 0.0)
                + 0.1 * centrality.get(item[0], {}).get("betweenness", 0.0),
                item[1].uses,
                item[1].mean_reward,
            ),
            reverse=True,
        )
        return [
            {
                "tool": tool,
                **stats.to_summary(graph.total_rollouts, self.exploration_weight),
                **centrality.get(tool, {}),
            }
            for tool, stats in ranked[:limit]
        ]

    def centrality(self, category: str) -> dict[str, dict[str, float]]:
        graph = self.categories.get(normalize_task_class(category))
        if graph is None:
            return {}
        node_weights = {
            tool: max(0.01, stats.mean_reward) * stats.stability * math.log1p(stats.uses)
            for tool, stats in graph.nodes.items()
        }
        edges = []
        weighted_degree = {tool: node_weights.get(tool, 0.0) for tool in graph.nodes}
        for edge_key, stats in graph.edges.items():
            source, target = edge_key.split("->", 1)
            weight = max(0.01, stats.mean_reward) * stats.stability * math.log1p(stats.uses)
            edges.append(_WeightedEdgeView(source, target, weight))
            weighted_degree[source] = weighted_degree.get(source, 0.0) + weight
            weighted_degree[target] = weighted_degree.get(target, 0.0) + weight
        view = _WeightedGraphView(node_weights, edges)
        rank = pagerank(view)
        between = betweenness_centrality(view, directed=True)
        return {
            tool: {
                "weighted_degree": weighted_degree.get(tool, 0.0),
                "pagerank": rank.get(tool, 0.0),
                "betweenness": between.get(tool, 0.0),
            }
            for tool in graph.nodes
        }

    def important_edges(self, category: str, limit: int = 15) -> list[dict[str, Any]]:
        graph = self.categories.get(normalize_task_class(category))
        if graph is None:
            return []
        ranked = sorted(
            graph.edges.items(),
            key=lambda item: (
                item[1].mean_advantage * item[1].stability,
                item[1].uses,
            ),
            reverse=True,
        )
        return [
            {
                "source": edge.split("->", 1)[0],
                "target": edge.split("->", 1)[1],
                **stats.to_summary(graph.total_rollouts, self.exploration_weight),
            }
            for edge, stats in ranked[:limit]
        ]

    def tool_arenas(
        self,
        category: str,
        tool_registry: Any | None,
    ) -> dict[str, Any]:
        """Rank interchangeable implementations separately from complete paths."""
        if tool_registry is None:
            return {}
        graph = self.categories.get(normalize_task_class(category))
        manifests = tool_registry.manifests()
        grouped: dict[str, list[dict[str, Any]]] = {}
        for manifest in manifests:
            provenance = str(manifest.get("provenance", ""))
            if manifest.get("backend") == "builtin" or "tool-arena" not in provenance:
                continue
            grouped.setdefault(str(manifest.get("capability", "unknown")), []).append(manifest)
        arenas: dict[str, Any] = {}
        for capability, specs in sorted(grouped.items()):
            if len(specs) < 2:
                continue
            rankings = []
            for spec in specs:
                tool = str(spec["name"])
                stats = (
                    graph.tool_arenas.get(capability, {}).get(tool)
                    if graph is not None
                    else None
                )
                summary = (
                    stats.to_summary(graph.total_rollouts, self.exploration_weight)
                    if stats is not None and graph is not None
                    else {
                        "uses": 0,
                        "mean_quality": 0.0,
                        "mean_reward": 0.0,
                        "mean_advantage": 0.0,
                        "mean_cost": float(spec.get("estimated_cost", 0.0)),
                        "success_rate": 0.0,
                        "seed_control_rate": None,
                        "execution_error_rate": 0.0,
                        "stability": 0.0,
                        "ucb": self.exploration_weight,
                    }
                )
                utility = float(summary["mean_reward"])
                rankings.append({
                    "tool": tool,
                    "backend": spec.get("backend"),
                    "provenance": spec.get("provenance"),
                    "model": spec.get("model"),
                    "evaluated": int(summary["uses"]) > 0,
                    "arena_utility": utility,
                    **summary,
                })
            rankings.sort(
                key=lambda item: (
                    bool(item["evaluated"]),
                    float(item["arena_utility"]),
                    float(item["ucb"]),
                ),
                reverse=True,
            )
            evaluated = [item for item in rankings if item["evaluated"]]
            expected_tools = {
                str(tool)
                for details in (graph.arena_families if graph else {}).values()
                if details.get("capability") == capability
                for tool in details.get("compared_tools", [])
            }
            evaluated_tools = {str(item["tool"]) for item in evaluated}
            complete = bool(expected_tools) and expected_tools.issubset(evaluated_tools)
            winner = None
            if (
                len(evaluated) >= 2
                and int(evaluated[0].get("successes", 0)) > 0
                and float(evaluated[0]["arena_utility"]) > 0.0
            ):
                winner = evaluated[0]["tool"]
            arenas[capability] = {
                "winner": winner,
                "selection_status": "complete" if complete else "provisional",
                "pending_tools": sorted(expected_tools - evaluated_tools),
                "comparison_protocol": (
                    self._h3_comparison_protocol(category, tool_registry) if self._has_h3_observations(category, tool_registry)
                    else "same graph, task, prompt, verifier, and paired generation seeds"
                ),
                "families": [
                    {"family_id": family_id, **details}
                    for family_id, details in sorted((graph.arena_families if graph else {}).items())
                    if details.get("capability") == capability
                ],
                "rankings": rankings,
            }
        return arenas

    def search_context(
        self,
        task: VideoTask,
        top_k: int = 5,
        tool_registry: Any | None = None,
    ) -> dict[str, Any]:
        category = task_class(task)
        graph = self.categories.get(category)
        return {
            "task_class": category,
            "total_rollouts": graph.total_rollouts if graph else 0,
            "baseline_quality": graph.baseline_quality if graph else 0.0,
            "top_paths": self.top_paths(category, top_k),
            "important_nodes": self.important_nodes(category, top_k * 2),
            "important_edges": self.important_edges(category, top_k * 3),
            "tool_arenas": self.tool_arenas(category, tool_registry),
            "credit_assignment": (
                "Observed quality delta against matched task/seed replicates for local H3. "
                "Shared node/edge credit divides that observed delta; seed control does not guarantee statistical gain."
                if self._h3_comparison_protocol(category, tool_registry) == H3_LOCAL_COMPARISON_PROTOCOL else
                "Observed quality delta against the same task/replicate baseline; H3 uses independent provider "
                "randomness, not matched random seeds. Shared node/edge credit divides that observed delta."
                if self._has_h3_observations(category, tool_registry) else
                "quality gain relative to the same task and generation seed baseline; path reward receives the full gain, "
                "while node and edge credit divide that gain across the executed path"
            ),
            "reward_objective": "task_conditioned_reward",
        }

    def _h3_comparison_protocol(self, category: str, tool_registry: Any | None = None) -> str:
        if h3_local_active(tool_registry):
            return H3_LOCAL_COMPARISON_PROTOCOL
        available = getattr(tool_registry, "available_names", lambda: set())()
        spec = getattr(tool_registry, "spec", None)
        if callable(spec) and any(spec(name).backend == "minimax-h3" for name in available):
            return H3_COMPARISON_PROTOCOL
        graph = self.categories.get(category)
        protocols = {path.comparison_protocol for path in graph.paths.values()} if graph else set()
        return H3_LOCAL_COMPARISON_PROTOCOL if protocols == {H3_LOCAL_COMPARISON_PROTOCOL} else H3_COMPARISON_PROTOCOL

    def _has_h3_observations(self, category: str, tool_registry: Any | None = None) -> bool:
        graph = self.categories.get(category)
        if graph is not None and any(
            path.comparison_protocol in {H3_COMPARISON_PROTOCOL, H3_LOCAL_COMPARISON_PROTOCOL}
            or any(name.startswith("h3_") for name in path.tools)
            for path in graph.paths.values()
        ):
            return True
        available = getattr(tool_registry, "available_names", lambda: set())()
        return bool({"h3_t2va", "h3_fl2va", "h3_ref2va"} & available)

    def categories_summary(self, tool_registry: Any | None = None) -> dict[str, Any]:
        return {
            category: {
                "total_rollouts": graph.total_rollouts,
                "baseline_quality": graph.baseline_quality,
                "top_paths": self.top_paths(category),
                "important_nodes": self.important_nodes(category),
                "important_edges": self.important_edges(category),
                "tool_arenas": self.tool_arenas(category, tool_registry),
            }
            for category, graph in sorted(self.categories.items())
        }

    def save(self) -> None:
        payload = {
            "version": 5,
            "credit_policy": "validation-gated-positive-advantage",
            "reward_objective": "task_conditioned_reward",
            "cost_weight": self.cost_weight,
            "exploration_weight": self.exploration_weight,
            "event_ids": sorted(self.event_ids),
            "categories": {
                category: {
                    "nodes": {name: asdict(stats) for name, stats in graph.nodes.items()},
                    "edges": {name: asdict(stats) for name, stats in graph.edges.items()},
                    "paths": {name: asdict(stats) for name, stats in graph.paths.items()},
                    "tool_arenas": {
                        capability: {name: asdict(stats) for name, stats in tools.items()}
                        for capability, tools in graph.tool_arenas.items()
                    },
                    "arena_families": graph.arena_families,
                    "baseline_quality_total": graph.baseline_quality_total,
                    "baseline_quality_count": graph.baseline_quality_count,
                    "baseline_quality_by_context": graph.baseline_quality_by_context,
                    "total_rollouts": graph.total_rollouts,
                }
                for category, graph in self.categories.items()
            },
            "summary": self.categories_summary(),
            "updated_at": utc_now(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        temporary.replace(self.path)

    def reset(self) -> None:
        self.categories.clear()
        self.event_ids.clear()
        if self.path.exists():
            self.path.unlink()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.event_ids = set(payload.get("event_ids", []))
        for category, raw in payload.get("categories", {}).items():
            self.categories[category] = TaskClassGraph(
                nodes={name: CreditStats(**stats) for name, stats in raw.get("nodes", {}).items()},
                edges={name: CreditStats(**stats) for name, stats in raw.get("edges", {}).items()},
                paths={name: PathCreditStats(**stats) for name, stats in raw.get("paths", {}).items()},
                tool_arenas={
                    capability: {name: CreditStats(**stats) for name, stats in tools.items()}
                    for capability, tools in raw.get("tool_arenas", {}).items()
                },
                arena_families={
                    str(name): dict(details)
                    for name, details in raw.get("arena_families", {}).items()
                },
                baseline_quality_total=float(raw.get("baseline_quality_total", 0.0)),
                baseline_quality_count=int(raw.get("baseline_quality_count", 0)),
                baseline_quality_by_context={
                    str(name): float(value)
                    for name, value in raw.get("baseline_quality_by_context", {}).items()
                },
                total_rollouts=int(raw.get("total_rollouts", 0)),
            )
        if int(payload.get("version", 1)) < 2:
            self._migrate_quality_gain_rewards()

    def _migrate_quality_gain_rewards(self) -> None:
        """Rebase legacy cost-aware totals onto their stored quality advantages."""
        for graph in self.categories.values():
            collections = [graph.nodes, graph.edges, graph.paths]
            collections.extend(graph.tool_arenas.values())
            for collection in collections:
                for stats in collection.values():
                    stats.total_reward = stats.total_advantage
                    mean_reward = stats.total_reward / max(1, stats.uses)
                    stats.reward_square_sum = mean_reward * mean_reward * stats.uses

    @staticmethod
    def path_key(tools: list[str], graph_behavior_fingerprint: str | None = None) -> str:
        # A DAG may have multiple topological tool orders, all of one behavior.
        return f"behavior:{graph_behavior_fingerprint}" if graph_behavior_fingerprint is not None else " -> ".join(tools)

    @staticmethod
    def edge_key(source: str, target: str) -> str:
        return f"{source}->{target}"

    @staticmethod
    def context_key(task_id: str, evaluation_seed: int | None) -> str:
        return f"{task_id}:seed={evaluation_seed if evaluation_seed is not None else 'none'}"

    @staticmethod
    def _event_id(task_id: str, graph_id: str, tools: list[str], quality: float) -> str:
        payload = json.dumps([task_id, graph_id, tools, round(quality, 6)], separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
