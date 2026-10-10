"""Source-backed action obligations. No NLP-inferred objects or state values.

Conservative clause splitting exposes conjunctions to independent judging; it is
not a semantic parser. Full source text remains authoritative in every check.
"""
from copy import deepcopy
import re

VERSION = "source-backed-obligations-v1"
# Split only recognizable clause starts, never arbitrary noun/adjective lists.
VERBS = ("adds|carries|closes|continues|cuts|displays|drops|emerges|fixes|holds|"
         "inserts|lifts|loads|lowers|opens|passes|places|presses|pulls|pushes|"
         "puts|raises|releases|removes|rests|returns|rotates|runs|sets|shows|"
         "slides|stamps|stays|stops|takes|threads|tightens|turns|walks|withdraws|"
         "hangs|packs|secures|spreads|stands|twists|unfolds|wraps")
SPLIT = re.compile(r"\s+(and then|and|then|while|without|before|after)\s+(?="
                   r"[AB]\b|the\b|it\b|its\b|" + VERBS + r"\b|[a-z]+ing\b)")
POLICY = ("Follow every declared action and qualifier, including its actor, object, "
          "direction, contents, order and explicit retention conditions. Correct end "
          "positions alone do not establish that the required actions occurred. "
          "Do not reverse actions, introduce unexplained object appearance, or reset "
          "state at a cut. Allow only changes specified by the story. Hidden state is "
          "not direct visual evidence; do not force hidden objects into view.")


def state_flow_requirement(contract, index):
    prior = [e["description"] for r in contract["shots"][:index] for e in r["events"]]
    current = [e["description"] for e in contract["shots"][index]["events"]]
    return ("State continuity within this shot and across its entry boundary. Original setup "
        f"(historical context, not a reset target): {contract['semantics']['initial_setup']!r}. "
        f"Earlier required events (do not replay): {prior!r}. Current events: {current!r}. "
        "Track the same relevant objects, their explicitly established contents, custody and "
        "physical condition. Preserve an established property unless a declared action changes "
        "it; respect legitimate transfers, transformations, additions and removals. A carried "
        "container must not silently refill or empty, a repaired object must not reset, and "
        "occlusion does not authorize an identity swap. Check only properties supported by the "
        "quoted story, never invent a count, hidden content, exact distance or grip. Visible "
        "contradictions fail even if the final location is correct. A gap between samples does "
        "not prove disappearance or spontaneous creation; unresolved evidence stays unknown. "
        "An earlier failed action does not prove a later action succeeded: judge the candidate's "
        "visible state against the required narrative, and explain the mismatch locally.")


def obligations(text):
    """Lossless spans: connectors stay attached to their clauses (incl. negation)."""
    # Long prose includes examples, conditional evidence rules and negative
    # instructions: preserve it as one requirement instead of guessing its logic.
    if re.search(r"[.;:]|\b(if|unless|not|never|no)\b", text, re.I):
        return [{"start": 0, "end": len(text), "quote": text, "relation": "whole"}]
    matches = list(SPLIT.finditer(text))
    starts = [0] + [m.start(1) for m in matches]
    result = []
    for i, start in enumerate(starts):
        end = starts[i+1] if i+1 < len(starts) else len(text)
        while end > start and text[end-1].isspace():
            end -= 1
        result.append({"start": start, "end": end, "quote": text[start:end],
                       "relation": matches[i-1].group(1) if i else "whole"})
    return result


def attach_semantics(task, setup):
    contract = task["metadata"]["story_contract"]
    contract["semantics"] = {"version": VERSION, "initial_setup": setup,
                              "inference_policy": "explicit_source_only"}
    for rule in contract["shots"]:
        for event in rule["events"]:
            event["obligations"] = obligations(event["description"])


def validate_semantics(contract):
    semantics = contract.get("semantics")
    if semantics is None:
        if any("obligations" in e for r in contract["shots"] for e in r["events"]):
            raise ValueError("event obligations require a versioned semantics contract")
        return False
    if (not isinstance(semantics, dict)
            or set(semantics) != {"version", "initial_setup", "inference_policy"}
            or semantics["version"] != VERSION
            or semantics["inference_policy"] != "explicit_source_only"
            or not isinstance(semantics["initial_setup"], str)
            or not semantics["initial_setup"].strip()):
        raise ValueError("invalid story semantics contract")
    for rule in contract["shots"]:
        for event in rule["events"]:
            if event.get("obligations") != obligations(event["description"]):
                raise ValueError("story obligation spans must match the complete original event source")
    return True


def combine_obligations(task, observations):
    """Conjoin actual judgments in the host, never synthesize visual evidence."""
    contract = task.metadata.get("story_contract", {})
    if not contract.get("semantics"):
        return {}
    audit = {}
    for rule in contract["shots"]:
        index = rule["shot_index"]
        for event in rule["events"]:
            if len(event["obligations"]) <= 1:
                continue
            parent = f"story.s{index}.event.{event['id']}"
            children = [f"story.s{index}.obligation.{event['id']}.{i}"
                        for i in range(len(event["obligations"]))]
            names = [parent, *children]
            rows = [observations.get(name, []) for name in names]
            if not rows[0] or any(len(r) != len(rows[0]) for r in rows):
                raise ValueError("missing source-backed obligation observations")
            audit[parent] = {"components": names, "original_parent": deepcopy(rows[0])}
            for repeat in range(len(rows[0])):
                components = [r[repeat] for r in rows]
                combined = deepcopy(components[0])
                target = next(s for s in combined["segments"] if s["segment_id"] == index)
                segments = [next(s for s in c["segments"] if s["segment_id"] == index)
                            for c in components]
                known = all(c["status"] == "observed" and not c.get('fact_conflicts') for c in components + segments)
                score = min(c["score"] for c in components + segments) if known else None
                status = "observed" if known else "unobserved"
                evidence = "Host conjunction of independent source requirements: " + " | ".join(
                    f"{name}: {c['evidence']}" for name, c in zip(names, components))
                combined.update(status=status, score=score, evidence=evidence,
                    observation_basis="insufficient_evidence" if not known else (
                        "visible_match" if score == 1 else "visible_mismatch"),
                    scope_issues=sorted({s for c in components for s in c.get("scope_issues", [])}))
                target.update(status=status, score=score, evidence=evidence)
                # Raw component assessments remain in the audit; the conjunction
                # must not retain an incompatible model assessment from its parent.
                for row in (combined, target):
                    row.pop("assessment", None)
                    row["aggregation_source"] = "host_source_obligation_conjunction"
                observations[parent][repeat] = combined
    return audit


def event_checks(event, enabled):
    """Keep one total event weight; every component is independently mandatory."""
    source = event["description"]
    if enabled and event.get("obligations") != obligations(source):
        raise ValueError("story obligation spans must match the complete original event source")
    units = event.get("obligations", []) if enabled else []
    split = len(units) > 1
    weight = 1. / (1 + len(units)) if split else 1.
    description = source
    if enabled:
        description += (" Evaluate ALL requirements in this event, including explicit "
            "direction, contents and temporal relations. Use the weakest supported "
            "component, not their average. Correct endpoints do not prove the action; "
            "a visible reverse action fails, while an unresolvable sampling gap is unknown.")
    result = [("event." + event["id"], description, weight)]
    if split:
        for i, unit in enumerate(units):
            result.append((f"obligation.{event['id']}.{i}",
                f"Required source clause: {unit['quote']!r}. Complete original event: {source!r}. "
                "Judge this clause in that context, preserving its subject, objects, "
                "negation and connective. 'And' alone does not assert strict order; "
                "'before/after/then' and 'while' retain their explicit temporal meanings. "
                "Do not require an unstated grip, distance or extra action. Other clauses "
                "cannot compensate for this clause failing. Missing evidence stays unknown.", weight))
    return result


def render_shots(task):
    """Render generation from the SAME validated contract used by the rubric."""
    contract = task.metadata["story_contract"]
    if not contract.get("semantics"):
        return
    def facts(values):
        return "; ".join(f"{k}: {v}" for k, v in values.items())
    for i, (shot, rule) in enumerate(zip(task.metadata["h3_shots"], contract["shots"])):
        # Camera is a presentation choice, not a semantic scoring requirement.
        prompt = f"Shot {i}. "
        if i == 0:
            prompt += "Initial scene: " + contract["semantics"]["initial_setup"] + " "
        else:
            prior = [e["description"] for r in contract["shots"][:i] for e in r["events"]]
            prompt += ("Narrative history already completed, not actions to replay: " +
                       " ".join(prior) + " Carry forward the resulting contents, custody and "
                       "physical condition except where this shot explicitly changes them. ")
        prompt += "Begin with " + facts(rule["preconditions"]) + ". "
        for event in rule["events"]:
            prompt += "Required event: " + event["description"] + ". "
            if len(event["obligations"]) > 1:
                prompt += "All clauses must be satisfied in context: " + " / ".join(
                    u["quote"] for u in event["obligations"]) + ". "
        prompt += "End with " + facts(rule["postconditions"]) + ". "
        if rule.get("invariants"):
            prompt += "Throughout preserve " + facts(rule["invariants"]) + ". "
        prompt += POLICY
        shot["prompt"] = prompt


def semantic_audit(tasks):
    rows = []
    for raw in tasks:
        contract = raw["metadata"]["story_contract"]
        validate_semantics(contract)
        events = [e for r in contract["shots"] for e in r["events"]]
        keys = set(contract["initial_state"])
        for rule in contract["shots"]:
            keys.update(rule["postconditions"])
        pending = []
        for event in events:
            if len(event.get("obligations", [])) == 1 and re.search(
                    r"\b(and|while|without|before|after|then)\b", event["description"]):
                pending.append({"event_id": event["id"], "source": event["description"],
                                "reason": "compound_kept_whole_no_safe_automatic_split"})
        rows.append({"task_id": raw["task_id"], "split": raw["metadata"]["split"],
            "family": raw["metadata"]["task_family"], "fact_keys": sorted(keys),
            "events": len(events), "obligations": sum(len(e.get("obligations", [])) for e in events),
            "split_events": sum(len(e.get("obligations", [])) > 1 for e in events),
            "review_items": pending,
            "state_model_review_required": len(keys) == 1,
            "semantic_human_validation": False})
    return {"version": VERSION, "tasks": len(rows),
        "split_events": sum(r["split_events"] for r in rows),
        "review_items": sum(len(r["review_items"]) for r in rows),
        "single_fact_tasks": sum(r["state_model_review_required"] for r in rows),
        "qualification": "All original event text remains mandatory. Clause spans are structural, "
            "not a verified semantic parse. Single-fact models may omit secondary state; "
            "source-backed clauses cover explicit actions but do not invent missing facts. "
            "Audit the listed cases before claiming a semantically validated benchmark.",
        "records": rows}
