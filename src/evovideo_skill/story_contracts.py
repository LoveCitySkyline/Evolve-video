"""Declarative story requirements and evidence-based acceptance; never a truth oracle.

Facts use user-defined stable keys (e.g. key.owner). Desired transitions are checked
before generation. Observations remain separate, with unknown distinct from false.
"""
from copy import deepcopy
import math


def _facts(value, label):
    if not isinstance(value, dict) or any(not isinstance(k, str) or not k or
            not isinstance(v, (str, int, float, bool)) or
            isinstance(v, float) and not math.isfinite(v) for k, v in value.items()):
        raise ValueError(f"{label} must map fact names to finite scalar values")
    return value


def prepare_story_task(task):
    """Validate and compile an optional story contract into the frozen public rubric."""
    contract = task.metadata.get("story_contract")
    if contract is None:
        return task
    if not isinstance(contract, dict) or contract.get("version") != 1:
        raise ValueError("story_contract requires version=1")
    shots = task.metadata.get("h3_shots", [])
    rules = contract.get("shots", [])
    if not shots or len(rules) != len(shots):
        raise ValueError("story contract must cover every declared shot")
    if any(type(s.get("duration_seconds")) is not int or not 4 <= s["duration_seconds"] <= 15 for s in shots):
        raise ValueError("story shots must have integer durations in 4..15 seconds")
    if sum(s["duration_seconds"] for s in shots) != task.duration_seconds:
        raise ValueError("story shot durations must cover the complete task")
    desired = dict(_facts(contract.get("initial_state", {}), "initial_state"))
    rubric = task.metadata.setdefault("evaluation", {})
    generated, event_ids, offset = {}, set(), 0
    for index, (shot, rule) in enumerate(zip(shots, rules)):
        if not isinstance(rule, dict) or rule.get("shot_index") != index:
            raise ValueError("story shots must be ordered with unique contiguous shot_index")
        pre = _facts(rule.get("preconditions", {}), "preconditions")
        post = _facts(rule.get("postconditions", {}), "postconditions")
        invariants = _facts(rule.get("invariants", {}), "invariants")
        for key, value in {**pre, **invariants}.items():
            if key not in desired or desired[key] != value:
                raise ValueError(f"shot {index}: contradictory or unestablished precondition {key}")
        if any(k in pre and pre[k] != v or k in post and post[k] != v for k, v in invariants.items()):
            raise ValueError("shot transition contradicts an invariant")
        checks = [(f"{kind}.{key}", f"{kind}: {key} must equal {value!r}")
                  for kind, facts in (("pre", pre), ("post", post), ("invariant", invariants))
                  for key, value in facts.items()]
        events = rule.get("events", [])
        if not isinstance(events, list) or not events:
            raise ValueError("every story shot needs an explicit event")
        for event in events:
            if (not isinstance(event, dict) or not isinstance(event.get("id"), str) or not event["id"]
                    or event["id"] in event_ids or not isinstance(event.get("description"), str)
                    or not event["description"].strip()):
                raise ValueError("story events need unique IDs and nonempty descriptions")
            event_ids.add(event["id"])
            checks.append(("event." + event["id"], event["description"]))
        threshold = rule.get("threshold", .9)
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("story threshold must be in (0,1]")
        end = offset + shot["duration_seconds"]
        for suffix, description in checks:
            key = f"story.s{index}.{suffix}"
            generated[key] = {"description": f"Shot {index}, {offset}..{end}s: {description}. "
                "Judge visible evidence in this shot only. Pre is its beginning, post its end; "
                "an invariant must hold throughout. Occlusion is unknown, not proof.",
                "threshold": threshold, "mandatory": True, "weight": 1.,
                "story_shot_index": index, "aggregation": "mean"}
        desired.update(post)
        offset = end
    final = _facts(contract.get("final_state", {}), "final_state")
    if any(desired.get(k) != v for k, v in final.items()):
        raise ValueError("declared final state is not established by story transitions")
    for key, value in generated.items():
        if key in rubric and rubric[key] != value:
            raise ValueError(f"story criterion collision: {key}")
    if any(k.startswith("story.") and k not in generated for k in rubric):
        raise ValueError("undeclared criterion in reserved story namespace")
    rubric.update(generated)
    return task


def validate_story_graph(task, graph):
    """Require ordered, complete shot coverage on the actual output ancestry."""
    if not task.metadata.get("story_contract"):
        return
    nodes = {n.node_id: n for n in graph.nodes}
    sinks = [n for n in graph.nodes if n.node_type == "tool" and not any(e.source == n.node_id for e in graph.edges)]
    if len(sinks) != 1:
        raise ValueError("story graph needs a single terminal output")
    count = len(task.metadata["h3_shots"])
    def sequence(node, visited):
        if node.node_id in visited:
            raise ValueError("cyclic story output")
        if node.name == "h3_av_concat":
            return [i for source in node.config.get("source_nodes", [])
                    for i in sequence(nodes[source], visited | {node.node_id})]
        if node.name == "mock_text_to_video" and "shot_index" not in node.config:
            return list(range(count))  # The native baseline adapter explicitly expands h3_shots.
        if node.name in {"mock_text_to_video", "h3_t2va", "h3_fl2va", "h3_ref2va"}:
            index = node.config.get("shot_index")
            if type(index) is not int or not 0 <= index < count:
                raise ValueError("each story output generator must bind a declared shot_index")
            if node.config.get("duration_seconds", task.metadata["h3_shots"][index]["duration_seconds"]) != task.metadata["h3_shots"][index]["duration_seconds"]:
                raise ValueError("story output cannot alter a declared shot duration")
            return [index]
        raise ValueError("unsupported story output time transformation")
    if sequence(sinks[0], set()) != list(range(count)):
        raise ValueError("story output must cover every shot once in declared order")


def acceptance_report(task, artifact):
    """Keep feasibility separate from scalar quality. Missing evidence stays unknown."""
    meta = artifact.metadata
    vlm = meta.get("vlm_evaluation", {})
    scores = dict(vlm.get("criterion_scores", {}))
    scores.update(meta.get("audio_evaluation", {}).get("criterion_scores", {}))
    from evovideo_skill.reward_router import TaskConditionedRewardRouter
    scores.update(TaskConditionedRewardRouter._task_metric_scores(artifact))
    checks = {}
    for name, rule in task.metadata.get("evaluation", {}).items():
        if not isinstance(rule, dict) or rule.get("mandatory") is not True:
            continue
        threshold = rule.get("threshold", .9)
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("mandatory threshold must be in (0,1]")
        score = scores.get(name)
        evidence = vlm.get("criterion_evidence", {}).get(name)
        window = None
        if name.startswith("story."):
            index = rule["story_shot_index"]
            durations = [s["duration_seconds"] for s in task.metadata["h3_shots"]]
            window = {"start_seconds": sum(durations[:index]), "end_seconds": sum(durations[:index+1])}
            observed_windows = vlm.get("verification_metadata", {}).get("windows", [])
            window_matches = (index < len(observed_windows) and all(
                observed_windows[index].get(k) == v for k, v in window.items()))
            rows = vlm.get("criterion_observations", {}).get(name, [])
            values, texts = [], []
            for row in rows:
                matches = [s for s in row.get("segments", []) if s.get("segment_id") == rule["story_shot_index"]]
                if (row.get("status") != "observed" or len(matches) != 1
                        or matches[0].get("status") != "observed"
                        or not matches[0].get("evidence")
                        or type(matches[0].get("score")) not in (int, float)
                        or not math.isfinite(matches[0]["score"]) or not 0 <= matches[0]["score"] <= 1):
                    values = []
                    break
                values.append(matches[0]["score"])
                texts.append(matches[0]["evidence"])
            score = min(values) if values and window_matches else None
            evidence = texts
        valid = type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1
        status = "unknown" if not valid else "passed" if score >= threshold else "failed"
        checks[name] = {"status": status, "score": score if valid else None,
                        "threshold": threshold, "evidence": evidence, "window": window}
    if not checks:
        status = "not_applicable"
    else:
        status = "failed" if any(c["status"] == "failed" for c in checks.values()) else (
            "unknown" if any(c["status"] == "unknown" for c in checks.values()) else "passed")
    timeline = []
    for rule in task.metadata.get("story_contract", {}).get("shots", []):
        i = rule["shot_index"]
        timeline.append({"shot_index": i, "desired_postconditions": deepcopy(rule.get("postconditions", {})),
            "observed_postconditions": {k: {"status": checks[f"story.s{i}.post.{k}"]["status"],
                "value": v if checks[f"story.s{i}.post.{k}"]["status"] == "passed" else None,
                "evidence": checks[f"story.s{i}.post.{k}"]["evidence"]}
                for k, v in rule.get("postconditions", {}).items()}})
    return {"status": status, "checks": checks, "story_timeline": timeline,
            "qualification": "Observed contract satisfaction under the configured verifier, not ground truth."}


def preserves_mandatory(before, after):
    previous = before.get("acceptance", {}).get("checks", {})
    current = after.get("acceptance", {}).get("checks", {})
    return all(current.get(k, {}).get("status") == "passed"
               for k, v in previous.items() if v["status"] == "passed")
