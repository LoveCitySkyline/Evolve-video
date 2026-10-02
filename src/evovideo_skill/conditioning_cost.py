"""Auditable generation-cost proxies, kept separate from quality and GPU time."""
from copy import deepcopy
import math
import statistics

from evovideo_skill.graph_skill import ToolPathGraph
from evovideo_skill.research_protocol import generation_credits

DEFAULTS = {"enabled": False, "call_weight": .02, "second_weight": .02,
            "experiment_weight": .01, "max_quality_drop": 0.,
            "min_net_gain": .01, "min_validation_net_gain": .02}


def cost_options(config):
    raw = config.get("cost_objective", {})
    if not isinstance(raw, dict) or set(raw) - DEFAULTS.keys():
        raise ValueError("invalid cost_objective settings")
    opts = {**DEFAULTS, **raw}
    if type(opts["enabled"]) is not bool:
        raise ValueError("cost_objective.enabled must be boolean")
    for key in DEFAULTS.keys() - {"enabled"}:
        value = opts[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid cost_objective.{key}")
    if opts["max_quality_drop"] > 1:
        raise ValueError("max_quality_drop must be in [0,1]")
    if opts["enabled"] and opts["call_weight"] + opts["second_weight"] <= 0:
        raise ValueError("enabled cost objective needs a positive deployment cost weight")
    return opts


def profile(task, graph, options):
    """Cold full-graph cost; never make a long deployment look free due to cache."""
    calls, seconds = generation_credits(task, graph)
    # A task-fixed reference, independent of the candidate and cache state.
    # Long fixture/custom tasks need not have an executable direct baseline.
    shots = task.metadata.get("h3_shots", [])
    base_seconds = task.duration_seconds
    base_calls = (len(shots) if shots and (base_seconds > 15 or task.metadata.get("story_contract"))
                  else max(1, math.ceil(base_seconds / 15)))
    nodes = []
    for node in graph.nodes:
        c, s = generation_credits(task, ToolPathGraph("node", "", "", [], [node], []))
        if c:
            nodes.append({"node_id": node.node_id, "tool": node.name,
                          "calls": c, "generated_seconds": s})
    return {"calls": calls, "generated_seconds": seconds,
            "normalized_calls": calls / base_calls, "normalized_seconds": seconds / base_seconds,
            "baseline_calls": base_calls, "baseline_generated_seconds": base_seconds,
            "normalization": "declared baseline shots, otherwise minimum 15s chunks; task duration",
            "nodes": nodes, "objective": deepcopy(options),
            "basis": "cold_native_calls_and_generated_video_seconds",
            "qualification": "Proxy only; excludes GPU time, queue, planner, verifier and media processing costs."}


def penalty(cost, options):
    return (options["call_weight"] * cost["normalized_calls"] +
            options["second_weight"] * cost["normalized_seconds"])


def delta(before, after):
    return {key: after[key] - before[key] for key in
            ("calls", "generated_seconds", "normalized_calls", "normalized_seconds")}


def annotate_effect(effect, before, after):
    available = ["generation_cost" in r for r in [*before, *after]]
    if not any(available):
        return effect
    if not all(available):
        raise ValueError("mixed missing cost evidence; cannot assume zero cost")
    options = before[0]["generation_cost"]["objective"]
    if any(r["generation_cost"]["objective"] != options for r in [*before, *after]):
        raise ValueError("cost objective changed between paired observations")
    rows = []
    for a, b in zip(before, after):
        ac, bc = a["generation_cost"], b["generation_cost"]
        if (ac["baseline_calls"], ac["baseline_generated_seconds"]) != (bc["baseline_calls"], bc["baseline_generated_seconds"]):
            raise ValueError("cost normalization changed between paired observations")
        d = delta(ac, bc)
        dq = b["score"] - a["score"]
        dc = penalty(bc, options) - penalty(ac, options)
        rows.append({"seed": a["seed"], **d, "quality_gain": dq,
                     "cost_penalty_delta": dc, "net_gain": dq - dc})
    effect["cost_effect"] = {"objective": deepcopy(options), "pairs": rows,
        **{key: statistics.mean(r[key] for r in rows) for key in rows[0] if key != "seed"},
        "net_gain_std": statistics.stdev(r["net_gain"] for r in rows) if len(rows) > 1 else None,
        "basis": "cold_graph_cost_not_cache_discounted"}
    return effect


def selection_gain(effect, config):
    from evovideo_skill.conditioning_bargaining import options
    if options(config)['enabled']:
        if 'bargaining' not in effect:
            raise ValueError('bargaining selection requires complete evidence')
        return effect['bargaining']['gain']
    if cost_options(config)["enabled"]:
        if "cost_effect" not in effect:
            raise ValueError("cost-aware selection requires complete cost evidence")
        return effect["cost_effect"]["net_gain"]
    return effect["gain"]


def pareto_points(cells, options):
    """Keep the original vector: scalarization must not conceal a metric tradeoff."""
    points = []
    for label, records in cells.items():
        if not records or any("generation_cost" not in r for r in records):
            continue
        q = statistics.mean(r["score"] for r in records)
        cost = {k: statistics.mean(r["generation_cost"][k] for r in records)
                for k in ("calls", "generated_seconds", "normalized_calls", "normalized_seconds")}
        points.append({"cell": label, "quality": q, **cost, "cost_penalty": penalty(cost, options),
                       "utility": q - penalty(cost, options)})
    for p in points:
        p["dominated_by"] = [o["cell"] for o in points if
            o["quality"] >= p["quality"] and o["calls"] <= p["calls"] and
            o["generated_seconds"] <= p["generated_seconds"] and
            (o["quality"] > p["quality"] or o["calls"] < p["calls"] or o["generated_seconds"] < p["generated_seconds"])]
        p["pareto"] = not p["dominated_by"]
    return points
