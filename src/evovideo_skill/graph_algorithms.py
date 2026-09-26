from __future__ import annotations

import heapq
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from math import inf
from typing import Any, Protocol

from evovideo_skill.graph_skill import ToolPathGraph


class GraphLike(Protocol):
    node_weights: dict[str, float]
    edge_weights: list[Any]


@dataclass
class ShortestToolPath:
    source: str
    target: str
    path: list[str]
    cost: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def directed_adjacency(global_graph: GraphLike, inverse_weight: bool = True) -> dict[str, list[tuple[str, float]]]:
    nodes = set(global_graph.node_weights)
    adjacency: dict[str, list[tuple[str, float]]] = {node: [] for node in nodes}
    for edge in global_graph.edge_weights:
        nodes.update([edge.source, edge.target])
        weight = edge.weight if edge.weight > 0 else 1.0
        cost = 1.0 / weight if inverse_weight else weight
        adjacency.setdefault(edge.source, []).append((edge.target, cost))
        adjacency.setdefault(edge.target, adjacency.get(edge.target, []))
    return adjacency


def undirected_adjacency(global_graph: GraphLike, inverse_weight: bool = True) -> dict[str, list[tuple[str, float]]]:
    adjacency = directed_adjacency(global_graph, inverse_weight=inverse_weight)
    for edge in global_graph.edge_weights:
        weight = edge.weight if edge.weight > 0 else 1.0
        cost = 1.0 / weight if inverse_weight else weight
        adjacency.setdefault(edge.target, []).append((edge.source, cost))
    return adjacency


def pagerank(global_graph: GraphLike, damping: float = 0.85, iterations: int = 50, tolerance: float = 1e-8) -> dict[str, float]:
    adjacency = directed_adjacency(global_graph, inverse_weight=False)
    nodes = sorted(adjacency)
    if not nodes:
        return {}
    n = len(nodes)
    rank = {node: 1.0 / n for node in nodes}
    out_weight = {node: sum(weight for _, weight in adjacency[node]) for node in nodes}
    for _ in range(iterations):
        updated = {node: (1.0 - damping) / n for node in nodes}
        dangling = sum(rank[node] for node in nodes if out_weight[node] == 0.0)
        for node in nodes:
            updated[node] += damping * dangling / n
            if out_weight[node] == 0.0:
                continue
            for neighbor, weight in adjacency[node]:
                updated[neighbor] += damping * rank[node] * weight / out_weight[node]
        delta = sum(abs(updated[node] - rank[node]) for node in nodes)
        rank = updated
        if delta < tolerance:
            break
    return rank


def betweenness_centrality(global_graph: GraphLike, directed: bool = False) -> dict[str, float]:
    adjacency = directed_adjacency(global_graph, inverse_weight=False) if directed else undirected_adjacency(global_graph, inverse_weight=False)
    nodes = sorted(adjacency)
    centrality = {node: 0.0 for node in nodes}
    for source in nodes:
        stack: list[str] = []
        predecessors: dict[str, list[str]] = {node: [] for node in nodes}
        sigma = {node: 0.0 for node in nodes}
        sigma[source] = 1.0
        distance = {node: -1 for node in nodes}
        distance[source] = 0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            stack.append(node)
            for neighbor, _ in adjacency[node]:
                if distance[neighbor] < 0:
                    queue.append(neighbor)
                    distance[neighbor] = distance[node] + 1
                if distance[neighbor] == distance[node] + 1:
                    sigma[neighbor] += sigma[node]
                    predecessors[neighbor].append(node)
        dependency = {node: 0.0 for node in nodes}
        while stack:
            node = stack.pop()
            for predecessor in predecessors[node]:
                if sigma[node] > 0:
                    dependency[predecessor] += (sigma[predecessor] / sigma[node]) * (1.0 + dependency[node])
            if node != source:
                centrality[node] += dependency[node]
    scale = 1.0
    if len(nodes) > 2:
        scale = 1.0 / ((len(nodes) - 1) * (len(nodes) - 2))
        if not directed:
            scale *= 2.0
    return {node: score * scale for node, score in centrality.items()}


def dijkstra_shortest_path(global_graph: GraphLike, source: str, target: str) -> ShortestToolPath | None:
    adjacency = directed_adjacency(global_graph, inverse_weight=True)
    if source not in adjacency or target not in adjacency:
        return None
    heap = [(0.0, source, [source])]
    best = {source: 0.0}
    while heap:
        cost, node, path = heapq.heappop(heap)
        if node == target:
            return ShortestToolPath(source, target, path, cost)
        if cost > best.get(node, inf):
            continue
        for neighbor, edge_cost in adjacency[node]:
            if neighbor in path:
                continue
            new_cost = cost + edge_cost
            if new_cost < best.get(neighbor, inf):
                best[neighbor] = new_cost
                heapq.heappush(heap, (new_cost, neighbor, path + [neighbor]))
    return None


def k_shortest_simple_paths(global_graph: GraphLike, source: str, target: str, k: int = 3) -> list[ShortestToolPath]:
    adjacency = directed_adjacency(global_graph, inverse_weight=True)
    if source not in adjacency or target not in adjacency:
        return []
    heap = [(0.0, source, [source])]
    results: list[ShortestToolPath] = []
    seen_paths: set[tuple[str, ...]] = set()
    while heap and len(results) < k:
        cost, node, path = heapq.heappop(heap)
        if node == target:
            key = tuple(path)
            if key not in seen_paths:
                seen_paths.add(key)
                results.append(ShortestToolPath(source, target, path, cost))
            continue
        for neighbor, edge_cost in adjacency.get(node, []):
            if neighbor in path:
                continue
            heapq.heappush(heap, (cost + edge_cost, neighbor, path + [neighbor]))
    return results


def louvain_style_communities(global_graph: GraphLike, max_iter: int = 20) -> dict[str, int]:
    adjacency = undirected_adjacency(global_graph, inverse_weight=False)
    nodes = sorted(adjacency)
    communities = {node: idx for idx, node in enumerate(nodes)}
    for _ in range(max_iter):
        changed = False
        for node in nodes:
            scores: Counter[int] = Counter()
            for neighbor, weight in adjacency[node]:
                scores[communities[neighbor]] += weight
            if not scores:
                continue
            best_community, _ = max(scores.items(), key=lambda item: (item[1], -item[0]))
            if best_community != communities[node]:
                communities[node] = best_community
                changed = True
        if not changed:
            break
    normalized = {}
    remap: dict[int, int] = {}
    for node in nodes:
        community = communities[node]
        if community not in remap:
            remap[community] = len(remap)
        normalized[node] = remap[community]
    return normalized


def wl_feature_vector(graph: ToolPathGraph, iterations: int = 2) -> Counter[str]:
    labels = {node.node_id: f"{node.node_type}:{node.name}" for node in graph.nodes}
    neighbors: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        neighbors[edge.source].add(edge.target)
        neighbors[edge.target].add(edge.source)
    features: Counter[str] = Counter(labels.values())
    for _ in range(iterations):
        updated = {}
        for node_id, label in labels.items():
            neighborhood = "|".join(sorted(labels[item] for item in neighbors[node_id] if item in labels))
            updated[node_id] = f"{label}<-{neighborhood}"
        labels = updated
        features.update(labels.values())
    return features


def wl_kernel_similarity(left: ToolPathGraph, right: ToolPathGraph, iterations: int = 2) -> float:
    left_features = wl_feature_vector(left, iterations)
    right_features = wl_feature_vector(right, iterations)
    dot = sum(left_features[key] * right_features.get(key, 0) for key in left_features)
    left_norm = sum(value * value for value in left_features.values()) ** 0.5
    right_norm = sum(value * value for value in right_features.values()) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def approximate_graph_edit_distance(left: ToolPathGraph, right: ToolPathGraph) -> float:
    left_nodes = Counter((node.node_type, node.name) for node in left.nodes)
    right_nodes = Counter((node.node_type, node.name) for node in right.nodes)
    left_edges = Counter((left.node(edge.source).name, left.node(edge.target).name) for edge in left.edges if _has_node(left, edge.source) and _has_node(left, edge.target))
    right_edges = Counter((right.node(edge.source).name, right.node(edge.target).name) for edge in right.edges if _has_node(right, edge.source) and _has_node(right, edge.target))
    node_distance = _multiset_l1(left_nodes, right_nodes)
    edge_distance = _multiset_l1(left_edges, right_edges)
    return float(node_distance + edge_distance)


def _multiset_l1(left: Counter[Any], right: Counter[Any]) -> int:
    keys = set(left) | set(right)
    return sum(abs(left.get(key, 0) - right.get(key, 0)) for key in keys)


def _has_node(graph: ToolPathGraph, node_id: str) -> bool:
    return any(node.node_id == node_id for node in graph.nodes)
