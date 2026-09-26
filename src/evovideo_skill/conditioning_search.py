"""Task-scoped signed interaction graphs and bounded active factorial selection.

The estimator is an empirical, task-balanced table, not a trained quality model.
Only measured train interventions update it. Unseen pairs retain prior uncertainty.
"""
from copy import deepcopy
from itertools import combinations
import math
import re
import statistics

from evovideo_skill.conditioning_interactions import condition_only, changes, merge_factors
from evovideo_skill.conditioning_memory import metric_vector, task_state, validate_strategy
from evovideo_skill.conditioning_repair import check_local_candidate
from evovideo_skill.graph_skill import ToolPathGraph
from evovideo_skill.research_protocol import decode_graph, graph_payload, generation_credits, validate_candidate
from evovideo_skill.research_subgraphs import stable_hash


DEFAULTS = {
    "enabled": False, "pool_size": 4, "exploration_beta": .5, "prior_std": .1,
    "min_uncertainty": .01, "local_repair": True, "max_separator_size": 3,
    "max_separator_candidates": 2000, "boundary_limit": 4,
    "preservation_threshold": .85, "preservation_tolerance": .03,
}


def search_options(config):
    raw = config.get("active_graph_search", {})
    if not isinstance(raw, dict) or set(raw) - DEFAULTS.keys():
        raise ValueError("invalid active_graph_search settings")
    options = {**DEFAULTS, **raw}
    for key in ("enabled", "local_repair"):
        if type(options[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    for key, low, high in (("pool_size", 2, 8), ("max_separator_size", 1, 4),
                           ("max_separator_candidates", 1, 10000), ("boundary_limit", 1, 16)):
        if type(options[key]) is not int or not low <= options[key] <= high:
            raise ValueError(f"{key} must be in [{low}, {high}]")
    for key in ("exploration_beta", "prior_std", "min_uncertainty",
                "preservation_threshold", "preservation_tolerance"):
        if type(options[key]) not in (int, float) or not math.isfinite(options[key]) or options[key] < 0:
            raise ValueError(f"invalid {key}")
    if not 0 < options["min_uncertainty"] <= options["prior_std"] <= 1:
        raise ValueError("require 0 < min_uncertainty <= prior_std <= 1")
    if options["preservation_threshold"] > 1 or options["preservation_tolerance"] > 1:
        raise ValueError("preservation settings must be in [0,1]")
    if options["enabled"] and config["search_mode"] != "factorial":
        raise ValueError("active graph search requires search_mode=factorial")
    return options


def context_view(task, records):
    state = task_state(task)
    return {k: deepcopy(state[k]) for k in ("family", "mode", "requires_audio", "duration_seconds",
                                           "shot_count", "reference_counts")} | {
        "criteria": sorted(metric_vector(records[0])) if records else [],
        "failed_criteria": sorted({name for r in records for segment in r.get("feedback", {}).get("failed_segments", [])
                                   if isinstance(segment, dict) for name in segment.get("failed_criteria", []) if isinstance(name, str)}),
        "shot_durations": [s.get("duration_seconds") for s in task.metadata.get("h3_shots", [])]}


def graph_sketch(task, graph):
    """Conservative node-order normalization; no claim of general isomorphism."""
    ids = {n.node_id: f"n{i}" for i, n in enumerate(graph.nodes)}
    refs = {r.get("id"): f"reference:{i}:{r.get('kind')}:{r.get('role')}:{r.get('semantic_role')}"
            for i, r in enumerate(task.metadata.get("h3_references", []))}
    refs.setdefault("source-video", "source-video" if task.reference_video else "unavailable-source-video")

    def clean(value, key=""):
        if key == "prompt":
            return "$fixed_task_prompt"
        if key in {"reference_id", "reference_ids"}:
            if isinstance(value, list):
                return [clean(v, "reference_id") for v in value]
            return refs.get(value, "unknown_reference:" + stable_hash(value)[:12])
        if key in {"source", "source_node", "source_nodes"}:
            if isinstance(value, list):
                return [clean(v, "source") for v in value]
            return ids.get(value, "unknown_source:" + stable_hash(value)[:12])
        if isinstance(value, dict):
            return {k: clean(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v, key) for v in value]
        if isinstance(value, str) and ("/" in value or task.task_id in value):
            return "opaque:" + stable_hash(value)[:12]
        return value
    return {"nodes": [{"node_id": ids[n.node_id], "tool": n.name, "type": n.node_type,
                        "config": clean(n.config)} for n in graph.nodes],
            "edges": [{"source": ids[e.source], "target": ids[e.target],
                       "condition": e.condition, "config": clean(e.config)} for e in graph.edges]}


def factor_descriptor(task, anchor, graph, context):
    # Both endpoints are retained: the same field edit on a different scaffold is
    # a different intervention, even if the planner reuses the same strategy name.
    value = {"context": context, "before": graph_sketch(task, anchor), "after": graph_sketch(task, graph)}
    return {"factor_id": stable_hash(value)[:24], **value}


class SignedInteractionGraph:
    def __init__(self, data=None):
        self.data = deepcopy(data or {"version": "signed-conditioning-v1", "nodes": {}, "edges": {}})

    def observe(self, task, descriptors, interaction, event_id):
        if not event_id.startswith("factorial/"):
            raise ValueError("signed evidence can only be updated by training factorials")
        for label in ("a", "b"):
            descriptor = descriptors[label]
            node = self.data["nodes"].setdefault(descriptor["factor_id"], {**descriptor, "observations": {}})
            effect = interaction["marginal_effects"][label]
            node["observations"].setdefault(event_id, {"task_id": task.task_id,
                "values": [p["delta"] for p in effect["pairs"]], "metrics": effect["metric_deltas"]})
        keys = sorted(descriptors[k]["factor_id"] for k in ("a", "b"))
        edge_id = stable_hash(keys)[:24]
        edge = self.data["edges"].setdefault(edge_id, {"factors": keys, "observations": {}})
        edge["observations"].setdefault(event_id, {"task_id": task.task_id,
            "values": [p["interaction"] for p in interaction["replicates"]],
            "metrics": {k: v["mean"] for k, v in interaction["metrics"].items()}})

    @staticmethod
    def estimate(entry, options):
        rows = list((entry or {}).get("observations", {}).values())
        if not rows:
            return {"mean": 0., "uncertainty": options["prior_std"], "task_support": 0,
                    "experiment_count": 0, "metric_means": {}}
        by_task = {}
        for row in rows:
            by_task.setdefault(row["task_id"], []).append(row)
        means = [statistics.mean(statistics.mean(r["values"]) for r in group) for group in by_task.values()]
        # Repeated adaptive experiments on one task do not create independent tasks.
        between = statistics.stdev(means) / math.sqrt(len(means)) if len(means) > 1 else 0.
        within = max(statistics.stdev(r["values"]) / math.sqrt(len(r["values"]))
                     if len(r["values"]) > 1 else options["prior_std"] for r in rows)
        dims = sorted({k for r in rows for k in r["metrics"]})
        metrics = {k: statistics.mean(statistics.mean(r["metrics"][k] for r in group if k in r["metrics"])
                     for group in by_task.values() if any(k in r["metrics"] for r in group)) for k in dims}
        return {"mean": statistics.mean(means), "uncertainty": max(options["min_uncertainty"],
                    options["prior_std"] / math.sqrt(len(means) + 1), between, within),
                "task_support": len(means), "experiment_count": len(rows), "metric_means": metrics}

    def predict(self, descriptors, options):
        keys = sorted(d["factor_id"] for d in descriptors)
        parts = [self.estimate(self.data["nodes"].get(k), options) for k in keys]
        parts.append(self.estimate(self.data["edges"].get(stable_hash(keys)[:24]), options))
        mean = sum(p["mean"] for p in parts)
        uncertainty = sum(p["uncertainty"] for p in parts)
        return {"predicted_gain_from_anchor": mean, "uncertainty": uncertainty,
                "acquisition": mean + options["exploration_beta"] * uncertainty,
                "terms": parts, "qualification": "Empirical acquisition heuristic, not a calibrated confidence bound."}

    def view(self, options, context=None):
        nodes = [{"factor_id": key, "descriptor": {k: v for k, v in entry.items() if k != "observations"},
                  **self.estimate(entry, options)} for key, entry in sorted(self.data["nodes"].items())
                 if context is None or entry["context"] == context]
        ids = {n["factor_id"] for n in nodes}
        edges = [{"edge_id": key, "factors": entry["factors"], **self.estimate(entry, options)}
                 for key, entry in sorted(self.data["edges"].items()) if set(entry["factors"]) <= ids]
        return {"nodes": nodes, "edges": edges, "version": self.data["version"],
                "scope": "train-only task-balanced empirical evidence"}


def decode_pool(raw, task, parent, executor, config, frontier):
    options = search_options(config)
    if not isinstance(raw, dict) or set(raw) != {"anchor", "factors"}:
        raise ValueError("active proposal needs exactly anchor and factors")
    if not isinstance(raw["factors"], list) or not 2 <= len(raw["factors"]) <= options["pool_size"]:
        raise ValueError("factor pool outside configured bounds")
    anchor = decode_graph(raw["anchor"])
    condition_only(parent, anchor)
    validate_candidate(anchor, parent, executor, "graph", config["max_nodes"], config["max_edits"])
    check_local_candidate(parent, anchor, frontier)
    factors, errors, seen, signatures = [], [], set(), set()
    for index, item in enumerate(raw["factors"]):
        try:
            if not isinstance(item, dict) or set(item) != {"id", "graph", "strategy"}:
                raise ValueError("factor needs id, graph, strategy")
            key = item["id"]
            if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", key) or key in seen:
                raise ValueError("factor IDs must be unique snake_case strings")
            seen.add(key)
            graph = decode_graph(item["graph"])
            if not changes(anchor, graph):
                raise ValueError("empty conditioning factor")
            signature = stable_hash(graph_payload(graph))
            if signature in signatures:
                raise ValueError("duplicate factor graph")
            signatures.add(signature)
            strategy = validate_strategy(item["strategy"], task)
            condition_only(anchor, graph)
            validate_candidate(graph, anchor, executor, "graph", config["max_nodes"], config["max_edits"])
            local = check_local_candidate(parent, graph, frontier)
            factors.append({"id": key, "graph": graph, "strategy": strategy, "repair": local})
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            errors.append({"factor_index": index, "error": str(exc)})
    return anchor, factors, errors


def select_pair(task, parent, anchor, factors, records, evidence, executor, config, frontier, remaining):
    options, context = search_options(config), context_view(task, records)
    ranked, rejected, combinations_by_id = [], [], {}
    for a, b in combinations(factors, 2):
        pair_id = "+".join(sorted([a["id"], b["id"]]))
        try:
            joint = merge_factors(anchor, a["graph"], b["graph"])
            condition_only(anchor, joint)
            validate_candidate(joint, anchor, executor, "graph", config["max_nodes"], config["max_edits"])
            repair = check_local_candidate(parent, joint, frontier)
            graphs = {"anchor": anchor, "a": a["graph"], "b": b["graph"], "joint": joint}
            # Conservative full-cell reservations, excluding an already evaluated parent.
            # Runtime hash-checked replay may reduce this bound further.
            calls, seconds, seen = 0, 0., {stable_hash(graph_payload(parent))}
            for graph in graphs.values():
                key = stable_hash(graph_payload(graph))
                if key not in seen:
                    c, s = generation_credits(task, graph)
                    calls += c * len(config["evaluation_seeds"])
                    seconds += s * len(config["evaluation_seeds"])
                    seen.add(key)
            if calls > remaining["calls"] or seconds > remaining["seconds"]:
                raise ValueError("insufficient budget for conservative complete factorial reservation")
            descriptors = {label: factor_descriptor(task, anchor, graphs[label], context) for label in ("a", "b")}
            prediction = evidence.predict(list(descriptors.values()), options)
            affected = set(repair["impact"]["affected_nodes"])
            affected_calls, affected_seconds = generation_credits(task,
                ToolPathGraph("affected", "affected", "", [], [n for n in joint.nodes if n.node_id in affected], []))
            other_outputs = {span["node_id"] for span in frontier.get("output_spans", [])} - set(frontier.get("targets", []))
            joint_strategy = {"name": "joint_" + stable_hash([descriptors[k]["factor_id"] for k in ("a", "b")])[:16],
                "instruction": "Combine two compatible interventions: " + a["strategy"]["instruction"][:800] + " / " + b["strategy"]["instruction"][:800],
                "hypothesis": "Joint conditioning may improve quality or expose a conflict; use the measured interaction evidence.",
                "risks": list(dict.fromkeys(a["strategy"]["risks"] + b["strategy"]["risks"]))[:8],
                "required_references": {k: max(a["strategy"]["required_references"].get(k, 0), b["strategy"]["required_references"].get(k, 0))
                    for k in a["strategy"]["required_references"].keys() | b["strategy"]["required_references"].keys()}}
            # Two factors can each require one image but consume different images.
            # Preserve the union's minimum asset requirements in the portable strategy.
            bound_refs = set()
            def collect(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key == "reference_ids" and isinstance(item, list):
                            bound_refs.update(x for x in item if isinstance(x, str))
                        elif key == "reference_id" and isinstance(item, str):
                            bound_refs.add(item)
                        else:
                            collect(item)
                elif isinstance(value, list):
                    for item in value:
                        collect(item)
            collect(graph_payload(joint))
            refs = {r.get("id"): r.get("kind") for r in task.metadata.get("h3_references", [])}
            for kind in ("image", "video", "audio"):
                count = sum(refs.get(key) == kind for key in bound_refs)
                if count:
                    joint_strategy["required_references"][kind] = max(count, joint_strategy["required_references"].get(kind, 0))
            strategies = {"a": a["strategy"], "b": b["strategy"], "joint": validate_strategy(joint_strategy, task)}
            ranked.append({"pair_id": pair_id, "factor_ids": [a["id"], b["id"]], **prediction,
                           "cold_budget_bound": {"calls": calls, "seconds": seconds}, "repair": repair,
                           "joint_affected_budget": {"calls": affected_calls, "seconds": affected_seconds},
                           "other_outputs_affected": sorted(affected & other_outputs)})
            combinations_by_id[pair_id] = (graphs, strategies, descriptors)
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            rejected.append({"pair_id": pair_id, "error": str(exc)})
    ranked.sort(key=lambda r: (-r["acquisition"], len(r["other_outputs_affected"]),
                              r["joint_affected_budget"]["seconds"], r["cold_budget_bound"]["seconds"], r["pair_id"]))
    selected = ranked[0]["pair_id"] if ranked else None
    return (combinations_by_id.get(selected), {"selected_pair": selected, "ranked_pairs": ranked,
        "rejected_pairs": rejected, "context": context,
        "evidence_hash": stable_hash(evidence.data), "candidate_scope": "bounded legal pairs from one LLM pool"})
