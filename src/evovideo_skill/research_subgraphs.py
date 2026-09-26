"""Bounded, exact executable suffix mining with explicit artifact ports.

This is not gSpan: fragments have at most four nodes and one external producer.
Canonical labeling enumerates permutations, preserving configuration and edges.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from itertools import combinations, permutations
import json
from typing import Any

from evovideo_skill.artifact_contracts import output_contract
from evovideo_skill.graph_skill import GraphEdge, GraphNode, ToolPathGraph
from evovideo_skill.tools import ToolRegistry


def stable_hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def remap_config(config: dict, mapping: dict[str, str]) -> dict:
    result = deepcopy(config)
    if "source_nodes" in result:
        result["source_nodes"] = [mapping.get(x, x) for x in result["source_nodes"]]
    for binding in result.get("bindings", []):
        if isinstance(binding, dict):
            for key in ("source", "source_node"):
                if key in binding:
                    binding[key] = mapping.get(binding[key], binding[key])
    return result


@dataclass
class ExecutableFragment:
    fragment_id: str
    applicability: str
    nodes: list[dict]
    edges: list[dict]
    input_contract: dict
    output_contract: dict
    observations: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class SubgraphLibrary:
    def __init__(self, tools: ToolRegistry, max_nodes: int = 4, max_enumerations: int = 2000):
        if not 1 <= max_nodes <= 4 or max_enumerations <= 0:
            raise ValueError("exact suffix mining needs 1..4 nodes and a positive enumeration cap")
        self.tools = tools
        self.max_nodes = max_nodes
        self.max_enumerations = max_enumerations
        self.fragments: dict[str, ExecutableFragment] = {}

    def mine(self, graph: ToolPathGraph, applicability: str, observation: dict) -> list[str]:
        tools = [n for n in graph.nodes if n.node_type == "tool"]
        found = []
        explored = 0
        for size in range(1, min(self.max_nodes, len(tools)) + 1):
            for subset in combinations(tools, size):
                explored += 1
                if explored > self.max_enumerations:
                    return found
                ids = {n.node_id for n in subset}
                # A reusable suffix must not silently drop side outputs or control flow.
                if any(e.source in ids and e.target not in ids for e in graph.edges):
                    continue
                boundary = [e for e in graph.edges if e.target in ids and e.source not in ids]
                if len({e.source for e in boundary}) != 1:
                    continue
                parent = graph.node(boundary[0].source)
                if parent.node_type != "tool":
                    continue
                edges = [e for e in graph.edges if e.target in ids]
                if any(e.condition != "always" for e in edges):
                    continue
                sinks = [n for n in subset if not any(e.source == n.node_id for e in edges)]
                if len(sinks) != 1 or self.tools.spec(sinks[0].name).output_type != "video":
                    continue
                reachable = {e.target for e in boundary}
                for _ in subset:
                    reachable |= {e.target for e in edges if e.source in reachable}
                if reachable != ids:
                    continue
                # Literal reference IDs/shot indices bind a procedure to a donor task.
                if any(set(n.config) & {"reference_ids", "id", "shot_index", "ids"} for n in subset):
                    continue
                produced = output_contract(self.tools.spec(parent.name)).to_dict()
                output = output_contract(self.tools.spec(sinks[0].name)).to_dict()
                encodings = []
                for ordering in permutations(subset):
                    mapping = {n.node_id: f"n{i}" for i, n in enumerate(ordering)}
                    mapping[parent.node_id] = "$input"
                    nodes = [{"node_id": mapping[n.node_id], "node_type": "tool", "name": n.name,
                              "config": remap_config(n.config, mapping)} for n in ordering]
                    links = sorted([{"source": mapping[e.source], "target": mapping[e.target],
                                     "condition": e.condition, "config": deepcopy(e.config)} for e in edges],
                                   key=lambda e: (e["source"], e["target"], json.dumps(e, sort_keys=True)))
                    encodings.append(json.dumps({"nodes": nodes, "edges": links}, sort_keys=True))
                canonical = json.loads(min(encodings))
                key = stable_hash([applicability, canonical, produced, output])[:24]
                fragment = self.fragments.setdefault(key, ExecutableFragment(
                    key, applicability, canonical["nodes"], canonical["edges"], produced, output))
                if observation not in fragment.observations:
                    fragment.observations.append(deepcopy(observation))
                found.append(key)
        return list(dict.fromkeys(found))

    def retrieve(self, applicability: str, limit: int = 8) -> list[dict]:
        matches = [f for f in self.fragments.values() if f.applicability == applicability]
        # Support is a retrieval prior, not a causal node reward.
        matches.sort(key=lambda f: (-len({o["task_id"] for o in f.observations}), f.fragment_id))
        return [f.to_dict() for f in matches[:limit]]

    def append(self, graph: ToolPathGraph, fragment_id: str, source_node: str,
               applicability: str) -> ToolPathGraph:
        fragment = self.fragments[fragment_id]
        if fragment.applicability != applicability:
            raise ValueError("fragment applicability mismatch")
        source = graph.node(source_node)
        if source.node_type != "tool" or any(e.source == source_node for e in graph.edges):
            raise ValueError("fragment input must be a terminal tool node")
        if output_contract(self.tools.spec(source.name)).to_dict() != fragment.input_contract:
            raise ValueError("fragment input contract mismatch; no implicit bridge is allowed")
        result = ToolPathGraph.from_dict(deepcopy(graph.to_dict()))
        prefix = f"reuse_{fragment_id[:8]}_"
        while any(n.node_id.startswith(prefix) for n in result.nodes):
            prefix += "r_"
        mapping = {n["node_id"]: prefix + n["node_id"] for n in fragment.nodes}
        mapping["$input"] = source_node
        for node in fragment.nodes:
            result.nodes.append(GraphNode(mapping[node["node_id"]], "tool", node["name"],
                                          remap_config(node["config"], mapping)))
        for i, edge in enumerate(fragment.edges):
            result.edges.append(GraphEdge(prefix + str(i), mapping[edge["source"]], mapping[edge["target"]],
                                          edge["condition"], deepcopy(edge["config"])))
        result.graph_id = result.skill_name = "reuse_" + stable_hash(result.to_dict())[:16]
        result.stats = {"reused_fragment_id": fragment_id}
        return result
