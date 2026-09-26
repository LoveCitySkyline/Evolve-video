"""Controlled, executable two-factor conditioning experiments.

Factors are disjoint edits relative to one common anchor. The joint graph is
compiled, never independently hallucinated. Effects are local, not universal.
"""
from copy import deepcopy
import math
import statistics

from evovideo_skill.conditioning_memory import metric_vector, paired_effect, validate_strategy
from evovideo_skill.research_protocol import decode_graph, graph_payload, validate_candidate


MISSING = object()
CONDITION_FIELDS = {"reference_ids", "roles", "semantic_roles", "bindings", "position",
                    "time_seconds", "role", "reference_id", "start_seconds", "end_seconds",
                    "source_nodes", "shot_index", "duration_seconds"}


def changes(parent, child):
    """Field-addressed edits; lists such as bindings are atomic ordered values."""
    old, new = graph_payload(parent), graph_payload(child)
    result = {}
    for kind, identifier in (("nodes", "node_id"), ("edges", "edge_id")):
        left = {x[identifier]: x for x in old[kind]}
        right = {x[identifier]: x for x in new[kind]}
        for key in sorted(left.keys() | right.keys()):
            if key not in left or key not in right:
                result[(kind, key)] = deepcopy(right[key]) if key in right else MISSING
            elif kind == "edges":
                if left[key] != right[key]:
                    result[(kind, key)] = deepcopy(right[key])
            else:
                for field in ("name", "node_type"):
                    if left[key][field] != right[key][field]:
                        result[(kind, key, field)] = right[key][field]
                a, b = left[key]["config"], right[key]["config"]
                for field in sorted(a.keys() | b.keys()):
                    if a.get(field, MISSING) != b.get(field, MISSING):
                        result[(kind, key, "config", field)] = deepcopy(b[field]) if field in b else MISSING
    return result


def merge_factors(anchor, a, b):
    da, db = changes(anchor, a), changes(anchor, b)
    if not da or not db:
        raise ValueError("both conditioning factors must make a nonempty change")
    for x in da:
        for y in db:
            if x[:min(len(x), len(y))] == y[:min(len(x), len(y))]:
                raise ValueError("overlapping factor edits; move shared edits into the anchor")
    payload = graph_payload(anchor)
    tables = {kind: {x[key]: x for x in payload[kind]}
              for kind, key in (("nodes", "node_id"), ("edges", "edge_id"))}
    for path, value in {**da, **db}.items():
        target = tables[path[0]]
        for part in path[1:-1]:
            target = target[part]
        if value is MISSING:
            target.pop(path[-1], None)
        else:
            target[path[-1]] = deepcopy(value)
    return decode_graph({kind: list(table.values()) for kind, table in tables.items()})


def condition_only(parent, child):
    """Keep text instructions fixed so a factorial is not prompt rewriting."""
    for path, value in changes(parent, child).items():
        if path[0] != "nodes":
            continue
        if len(path) == 2 and value is not MISSING and "prompt" in value.get("config", {}):
            raise ValueError("new conditioning nodes must use the unchanged task/shot prompt")
        if len(path) == 4 and path[-1] not in CONDITION_FIELDS:
            raise ValueError(f"non-conditioning field changed: {path[-1]}")
        if len(path) == 3 and path[-1] == "node_type":
            raise ValueError("conditioning edits cannot change node types")


def decode_experiment(raw, task, parent, executor, config):
    if not isinstance(raw, dict) or set(raw) != {"anchor", "a", "b", "joint_strategy"}:
        raise ValueError("experiment requires exactly anchor, a, b, joint_strategy")
    anchor = decode_graph(raw["anchor"])
    graphs, strategies = {"anchor": anchor}, {}
    for label in ("a", "b"):
        item = raw[label]
        if not isinstance(item, dict) or set(item) != {"graph", "strategy"}:
            raise ValueError("each factor needs graph and strategy")
        graphs[label] = decode_graph(item["graph"])
        strategies[label] = validate_strategy(item["strategy"], task)
    graphs["joint"] = merge_factors(anchor, graphs["a"], graphs["b"])
    strategies["joint"] = validate_strategy(raw["joint_strategy"], task)
    for label, graph in graphs.items():
        base = parent if label == "anchor" else anchor
        condition_only(base, graph)
        validate_candidate(graph, base, executor, "graph", config["max_nodes"], config["max_edits"])
    return graphs, strategies


def estimate(values):
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return {"mean": mean, "standard_error": se, "replicates": len(values),
            "sign_agreement": max(sum(v > 0 for v in values), sum(v < 0 for v in values)) / len(values)}


def interaction_effect(cells):
    if set(cells) != {"anchor", "a", "b", "joint"}:
        raise ValueError("interaction requires all four observed cells")
    anchor = cells["anchor"]
    marginal = {k: paired_effect(anchor, cells[k]) for k in ("a", "b", "joint")}
    if len({r["seed"] for r in anchor}) != len(anchor):
        raise ValueError("duplicate interaction seeds")
    dims = set(metric_vector(anchor[0]))
    for records in cells.values():
        for record in records:
            if set(metric_vector(record)) != dims:
                raise ValueError("all factorial cells must have identical measured dimensions")
    values, metrics, repeats = [], {k: [] for k in sorted(dims)}, []
    for i, record in enumerate(anchor):
        row = {k: v[i] for k, v in cells.items()}
        effect = row["joint"]["score"] - row["a"]["score"] - row["b"]["score"] + record["score"]
        if not math.isfinite(effect):
            raise ValueError("nonfinite interaction score")
        values.append(effect)
        vectors = {k: metric_vector(v) for k, v in row.items()}
        for k in dims:
            metrics[k].append(vectors["joint"][k] - vectors["a"][k] - vectors["b"][k] + vectors["anchor"][k])
        repeats.append({"seed": record["seed"], "interaction": effect,
                        "evaluations": {k: v["evaluation_id"] for k, v in row.items()}})
    return {"quality": estimate(values), "metrics": {k: estimate(v) for k, v in metrics.items()},
            "marginal_effects": marginal, "replicates": repeats,
            "attribution": "local paired-seed interaction relative to a common anchor; not causal identification",
            "uncertainty": "standard errors describe seed variation, not verifier bias or cross-task generalization"}


def effect_supported(effect, config):
    if effect["gain"] <= config["selection_min_gain"]:
        return False
    if any(v < -config["max_metric_regression"] for v in effect["metric_deltas"].values()):
        return False
    pairs = effect["pairs"]
    if sum(p["delta"] > 0 for p in pairs) / len(pairs) < config.get("min_positive_seed_fraction", 0):
        return False
    se = (effect.get("gain_std") or 0) / math.sqrt(len(pairs))
    return effect["gain"] - config.get("gain_se_multiplier", 0) * se > 0


def connection_manifest(graph):
    return {"nodes": [{"id": n.node_id, "tool": n.name,
                       "inputs": [e.source for e in graph.edges if e.target == n.node_id],
                       "conditioning": {k: deepcopy(v) for k, v in n.config.items() if k in CONDITION_FIELDS}}
                      for n in graph.nodes if n.node_type == "tool"],
            "note": "Declared dependency and binding; semantic role does not guarantee model disentanglement."}
