"""Program-derived structural witnesses for portable strategy instantiation.

A template describes the measured target topology, fixed controls and late-bound
slots. It is deliberately conservative: structural equivalence is not semantic
proof, and alternative topologies require a separately measured strategy.
"""
from copy import deepcopy

from evovideo_skill.research_subgraphs import stable_hash

SLOTS = {"prompt", "reference_id", "reference_ids", "semantic_role", "semantic_roles",
         "shot_index", "duration_seconds", "time_seconds", "start_seconds", "end_seconds"}


def structural_template(graph):
    ids = {n.node_id: f"n{i}" for i, n in enumerate(graph.nodes)}
    def clean(value, key=""):
        if key in SLOTS:
            return {"slot": key}
        if key == "roles" and isinstance(value, dict):
            return {"reference_role_slots": sorted(value.values())}
        if key in {"source", "source_node", "source_nodes"}:
            return [ids[x] for x in value] if isinstance(value, list) else ids[value]
        if isinstance(value, dict):
            return {k: clean(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    return {"version": 1, "nodes": [{"id": ids[n.node_id], "type": n.node_type,
        "tool": n.name, "config": clean(n.config)} for n in graph.nodes],
        "edges": [{"source": ids[e.source], "target": ids[e.target], "condition": e.condition,
                   "config": clean(e.config)} for e in graph.edges]}


def measured_contract(before, after):
    if before is None:
        return None
    target = structural_template(after)
    return {"version": 1, "target": target, "target_hash": stable_hash(target),
            "origin": "program_derived_from_measured_graph", "requires_change": True}


def verify_instantiation(parent, candidate, entries, used_ids):
    from evovideo_skill.research_protocol import graph_payload
    known = {e["strategy_id"]: e for e in entries}
    if len(used_ids) != len(set(used_ids)):
        raise ValueError("duplicate strategy IDs")
    if used_ids and graph_payload(parent) == graph_payload(candidate):
        raise ValueError("claimed strategy did not change the graph")
    target = structural_template(candidate)
    for key in used_ids:
        contract = known[key].get("structural_contract")
        if (not isinstance(contract, dict) or contract.get("version") != 1
                or stable_hash(contract.get("target")) != contract.get("target_hash")):
            raise ValueError("strategy has no verified structural contract; relearn legacy memory")
        if target != contract["target"]:
            raise ValueError("candidate does not instantiate the measured strategy topology and controls")
    return {"verified_strategy_ids": list(used_ids), "target_hash": stable_hash(target)}
