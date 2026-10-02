"""Empirical conditioning strategies, not a trained predictor or causal model."""
from copy import deepcopy
import math
import re
import statistics

from evovideo_skill.research_subgraphs import stable_hash
from evovideo_skill.conditioning_cost import annotate_effect


def task_state(task, observations=()):
    refs = task.metadata.get("h3_references", [])
    counts = {kind: sum(r.get("kind") == kind for r in refs) for kind in ("image", "video", "audio")}
    counts["video"] = max(counts["video"], int(bool(task.reference_video)))
    return {
        "family": str(task.metadata.get("task_family") or task.metadata.get("category") or "general"),
        "mode": task.mode.value,
        "duration_seconds": task.duration_seconds,
        "shot_count": len(task.metadata.get("h3_shots", [])) or 1,
        "reference_counts": counts,
        "requires_audio": bool(task.metadata.get("h3_audio_criteria")),
        "requirements": deepcopy(task.metadata.get("evaluation", {})),
        "observations": list(observations),
        "unobserved_defects": "unknown; do not infer a measured failure from the task text",
    }


def task_payload(task):
    # Only this task's inputs and public rubric enter a planner request.
    return {"prompt": task.prompt, "mode": task.mode.value, "duration_seconds": task.duration_seconds,
            "reference_video": task.reference_video,
            "metadata": {k: deepcopy(task.metadata[k]) for k in (
                "h3_references", "h3_shots", "h3_global_constraints", "h3_audio_criteria",
                "evaluation", "constraints", "temporal_steps", "story_contract") if k in task.metadata}}


def validate_strategy(value, task):
    required = {"name", "instruction", "hypothesis", "risks", "required_references"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("strategy needs exactly name, instruction, hypothesis, risks, required_references")
    for key in ("name", "instruction", "hypothesis"):
        if not isinstance(value[key], str) or not 1 <= len(value[key].strip()) <= 2000:
            raise ValueError(f"invalid strategy {key}")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", value["name"]):
        raise ValueError("strategy name must be a short snake_case identifier")
    if not isinstance(value["risks"], list) or len(value["risks"]) > 8 or any(
            not isinstance(x, str) or len(x) > 1000 for x in value["risks"]):
        raise ValueError("risks must be a short string list")
    refs = value["required_references"]
    available = task_state(task)["reference_counts"]
    if not isinstance(refs, dict) or set(refs) - set(available) or any(
            type(n) is not int or not 0 <= n <= available[k] for k, n in refs.items()):
        raise ValueError("strategy requires unavailable task references")
    # Reject obvious donor material. Semantic generality still needs auditing.
    text = " ".join([value["instruction"], value["hypothesis"], *value["risks"]])
    forbidden = [task.prompt, task.task_id, task.reference_video]
    forbidden += [r.get("uri") for r in task.metadata.get("h3_references", [])]
    if any(x and len(str(x)) > 4 and re.search(r"(?<!\w)" + re.escape(str(x)) + r"(?!\w)", text)
           for x in forbidden) or re.search(r"https?://|file://", text):
        raise ValueError("strategy prose contains donor prompt, ID or asset URI")
    return deepcopy(value)


def portable_recipe(graph):
    """A typed structural sketch for LLM adaptation, never directly executable."""
    mapping = {n.node_id: f"n{i}" for i, n in enumerate(graph.nodes)}
    allowed = {"position", "role", "kind", "mode", "source", "source_node", "source_nodes",
               "bindings", "shot_index", "duration_seconds", "time_seconds", "start_seconds", "end_seconds"}
    def clean(value, key=""):
        if key == "shot_index":
            return "$target_shot_index"
        if key.endswith("seconds"):
            return "$target_" + key
        if key in {"source", "source_node"}:
            return mapping.get(value, "$target_source")
        if key == "source_nodes":
            return [mapping.get(x, "$target_source") for x in value]
        if isinstance(value, dict):
            return {k: clean(v, k) for k, v in value.items() if k in allowed}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    return {"executable": False, "nodes": [{"node_id": mapping[n.node_id], "node_type": n.node_type,
             "name": n.name, "config": clean(n.config),
             "adapt_fields": sorted(set(n.config) - allowed)} for n in graph.nodes],
            "edges": [{"source": mapping[e.source], "target": mapping[e.target]} for e in graph.edges]}


def metric_vector(record):
    metrics = {k: v["score"] for k, v in record.get("feedback", {}).get("metrics", {}).items()}
    metrics.update(record.get("reward", {}).get("components", {}))
    metrics.update(record.get("criterion_scores", {}))
    return {k: float(v) for k, v in metrics.items() if type(v) in (int, float) and math.isfinite(v)}


def paired_effect(before, after):
    if len(before) != len(after) or not before:
        raise ValueError("an intervention needs nonempty matched repeats")
    pairs = []
    for a, b in zip(before, after):
        if (a["task_id"], a["seed"]) != (b["task_id"], b["seed"]):
            raise ValueError("unpaired task/seed observations")
        av, bv = metric_vector(a), metric_vector(b)
        if set(av) != set(bv):
            raise ValueError("paired evaluations have different observed metrics")
        pairs.append({"seed": a["seed"], "delta": b["score"] - a["score"],
                      "metric_deltas": {k: bv[k] - av[k] for k in av.keys() & bv.keys()},
                      "before_evaluation": a["evaluation_id"], "after_evaluation": b["evaluation_id"],
                      "reused_nodes": b.get("reused_nodes", [])})
    dimensions = set.intersection(*(set(p["metric_deltas"]) for p in pairs))
    result = {"gain": statistics.mean(p["delta"] for p in pairs), "pairs": pairs,
            "before_metric_means": {k: statistics.mean(metric_vector(r)[k] for r in before) for k in dimensions},
            "after_metric_means": {k: statistics.mean(metric_vector(r)[k] for r in after) for k in dimensions},
            "metric_deltas": {k: statistics.mean(p["metric_deltas"][k] for p in pairs) for k in sorted(dimensions)},
            "gain_std": statistics.stdev(p["delta"] for p in pairs) if len(pairs) > 1 else None,
            "attribution": "paired local graph intervention, not proof of a universal causal effect"}
    return annotate_effect(result, before, after)


class StrategyMemory:
    def __init__(self, entries=None, frozen=False):
        self.entries = deepcopy(entries or {})
        self.frozen = frozen

    def observe(self, task, proposal, graph, effect, event_id, before_graph=None):
        if self.frozen:
            raise RuntimeError("frozen strategy memory cannot receive test observations")
        strategy = validate_strategy(proposal, task)
        state = task_state(task)
        recipe = portable_recipe(graph)
        scope = {k: state[k] for k in ("family", "mode", "requires_audio")}
        before_recipe = portable_recipe(before_graph) if before_graph is not None else None
        from evovideo_skill.strategy_contracts import measured_contract
        contract = measured_contract(before_graph, graph)
        key = stable_hash([strategy["name"], scope, strategy["required_references"], before_recipe, recipe, contract])[:24]
        entry = self.entries.setdefault(key, {"strategy_id": key, "strategy": strategy,
            "structural_contract": contract,
            "scope": scope, "recipe": recipe, "before_recipe": before_recipe, "observations": []})
        if not any(o["event_id"] == event_id for o in entry["observations"]):
            entry["observations"].append({"event_id": event_id, "task_id": task.task_id,
                "state": {k: state[k] for k in ("duration_seconds", "shot_count", "reference_counts")},
                "effect": deepcopy(effect)})
        return key

    @staticmethod
    def matches(entry, state):
        return (all(state[k] == v for k, v in entry["scope"].items()) and
                all(state["reference_counts"].get(k, 0) >= n
                    for k, n in entry["strategy"]["required_references"].items()))

    @staticmethod
    def view(entry):
        effects = [o["effect"] for o in entry["observations"]]
        by_task = {}
        for observation in entry["observations"]:
            by_task.setdefault(observation["task_id"], []).append(observation["effect"]["gain"])
        gains = [statistics.mean(v) for v in by_task.values()]
        selection_by_task = {}
        for observation in entry["observations"]:
            e = observation["effect"]
            c = e.get("cost_effect", {})
            selection_by_task.setdefault(observation["task_id"], []).append(
                c["net_gain"] if c.get("objective", {}).get("enabled") else e["gain"])
        selection_mean = statistics.mean(statistics.mean(v) for v in selection_by_task.values())
        dims = sorted({k for e in effects for k in e["metric_deltas"]})
        deltas = {k: statistics.mean(e["metric_deltas"][k] for e in effects if k in e["metric_deltas"]) for k in dims}
        return {k: deepcopy(entry[k]) for k in ("strategy_id", "strategy", "scope", "recipe", "before_recipe")} | {
            "structural_contract": deepcopy(entry.get("structural_contract")),
            "evidence": {"task_support": len(gains), "experiment_count": len(effects),
                "mean_train_gain": statistics.mean(gains), "metric_deltas": deltas,
                "mean_selection_gain": selection_mean,
                "selection_status": "positive" if selection_mean > 0 else "nonpositive",
                "cost_effects": [deepcopy(e["cost_effect"]) for e in effects if "cost_effect" in e][-8:],
                "observed_conditions": [o["state"] for o in entry["observations"]],
                "contextual_effects": [{"state": o["state"],
                    "before_metric_means": o["effect"].get("before_metric_means", {}),
                    "gain": o["effect"]["gain"], "metric_deltas": o["effect"]["metric_deltas"]}
                    for o in entry["observations"]][-8:],
                "conditioning_interactions": [deepcopy(e["interaction"]) for e in effects
                                               if "interaction" in e][-8:],
                "observed_risks": [k for k, v in deltas.items() if v < 0],
                "status": "mixed" if min(gains) < 0 < max(gains) else
                          "negative" if statistics.mean(gains) <= 0 else "promising",
                "note": "training evidence; prose hypotheses and risks are not verified facts"}}

    def retrieve(self, task, limit=4, admitted=None):
        matches = [self.view(e) for k, e in self.entries.items()
                   if self.matches(e, task_state(task)) and (admitted is None or k in admitted)]
        matches.sort(key=lambda e: (-e["evidence"]["task_support"], -e["evidence"]["mean_selection_gain"], e["strategy_id"]))
        # Include one counterexample strategy rather than silently discarding all failures.
        positive = [e for e in matches if e["evidence"]["mean_selection_gain"] > 0]
        negative = [e for e in matches if e["evidence"]["mean_selection_gain"] <= 0]
        return (positive[:max(0, limit - bool(negative))] + negative[:1])[:limit] if negative else positive[:limit]

    def snapshot(self):
        return deepcopy(self.entries)
