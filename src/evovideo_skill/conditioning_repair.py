"""Bounded vertex separators for condition repair; reachability is not causality."""
from itertools import combinations
import math
import statistics

from evovideo_skill.conditioning_memory import metric_vector
from evovideo_skill.conditioning_workspace import repair_impact
from evovideo_skill.graph_skill import ToolPathGraph
from evovideo_skill.research_protocol import generation_credits, H3_GENERATORS


GENERATORS = H3_GENERATORS | {"mock_text_to_video"}


def adjacency(graph):
    children = {n.node_id: set() for n in graph.nodes}
    parents = {n.node_id: set() for n in graph.nodes}
    for edge in graph.edges:
        children[edge.source].add(edge.target)
        parents[edge.target].add(edge.source)
    return children, parents


def closure(starts, links, blocked=()):
    seen, pending = set(), set(starts) - set(blocked)
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        pending.update(links.get(node, set()) - seen - set(blocked))
    return seen


def dominators(graph):
    """DAG dominators with an implicit common root (including multiple roots)."""
    _, parents = adjacency(graph)
    result, pending = {}, set(parents)
    while pending:
        ready = sorted(n for n in pending if parents[n] <= result.keys())
        if not ready:
            raise ValueError("repair analysis requires a DAG")
        for node in ready:
            result[node] = {node} | (set.intersection(*(result[p] for p in parents[node]))
                                     if parents[node] else set())
            pending.remove(node)
    return {k: sorted(v) for k, v in result.items()}


def output_spans(task, graph):
    """Only explicit concat/generation timelines are mapped, never guessed from IDs."""
    children, _ = adjacency(graph)
    nodes = {n.node_id: n for n in graph.nodes}
    sinks = [n for n in graph.nodes if not children[n.node_id] and n.node_type == "tool"]
    if len(sinks) != 1:
        return []

    def visit(node_id, stack):
        if node_id in stack:
            raise ValueError("cyclic output timeline")
        node = nodes[node_id]
        if node.name == "h3_av_concat":
            sources = node.config.get("source_nodes", [])
            if not sources:
                raise ValueError("unknown concat order")
            return [item for source in sources for item in visit(source, stack | {node_id})]
        if node.name not in GENERATORS:
            raise ValueError("unknown output time transformation")
        _, duration = generation_credits(task, ToolPathGraph("span", "span", "", [], [node], []))
        return [(node_id, duration)]

    try:
        sequence = visit(sinks[0].node_id, set())
    except (KeyError, ValueError, TypeError):
        return []
    if abs(sum(d for _, d in sequence) - task.duration_seconds) > .05:
        return []
    spans, offset = [], 0.0
    for node_id, duration in sequence:
        spans.append({"node_id": node_id, "start_seconds": offset, "end_seconds": offset + duration})
        offset += duration
    return spans


def segment_scores(record):
    vlm = record.get("artifact", {}).get("metadata", {}).get("vlm_evaluation", {})
    windows = vlm.get("verification_metadata", {}).get("windows", [])
    result = {}
    for criterion, rows in vlm.get("criterion_observations", {}).items():
        by_window = {}
        for row in rows:
            for segment in row.get("segments", []):
                index = segment.get("segment_id")
                value = segment.get("score")
                if (segment.get("status") != "observed" or type(index) is not int or
                        not 0 <= index < len(windows) or type(value) not in (int, float) or
                        not math.isfinite(value)):
                    continue
                span = windows[index]
                key = (criterion, float(span["start_seconds"]), float(span["end_seconds"]))
                by_window.setdefault(key, []).append(value)
        result.update({k: statistics.mean(v) for k, v in by_window.items()})
    return result


def repair_frontier(task, graph, records, options):
    children, parents = adjacency(graph)
    spans, failed = output_spans(task, graph), []
    for record in records:
        for segment in record.get("feedback", {}).get("failed_segments", []):
            if not isinstance(segment, dict) or not segment.get("failed_criteria"):
                continue
            a, b = segment.get("start_ratio"), segment.get("end_ratio")
            if (type(a) in (int, float) and type(b) in (int, float) and
                    math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= 1):
                failed.append((a * task.duration_seconds, b * task.duration_seconds))
    targets = {s["node_id"] for s in spans if any(
        s["start_seconds"] < b and s["end_seconds"] > a for a, b in failed)}
    base = {"targets": sorted(targets), "output_spans": spans, "boundaries": [],
            "qualification": "Structural intervention candidates, not causal fault attribution.",
            "dominators": dominators(graph),
            "protected_evidence": [{"seed": r.get("seed"),
                "criteria": {k: v for k, v in metric_vector(r).items() if v >= options["preservation_threshold"]},
                "windows": [{"criterion": k[0], "start_seconds": k[1], "end_seconds": k[2], "score": v}
                    for k, v in segment_scores(r).items() if v >= options["preservation_threshold"]]}
                for r in records]}
    if not targets:
        return {**base, "status": "unlocalized", "reason": "No observed defect with a supported output timeline."}
    ancestors = closure(targets, parents)
    tools = {n.node_id for n in graph.nodes if n.node_type == "tool"}
    candidates = sorted(ancestors & tools)
    # Tool roots of the induced dependency graph are suspected influence sources.
    sources = {n for n in candidates if not (closure(parents[n], parents) & tools & ancestors)}
    unaffected = {s["node_id"] for s in spans} - targets
    examined = 0
    boundaries = []
    for size in range(1, min(options["max_separator_size"], len(candidates)) + 1):
        for subset in combinations(candidates, size):
            if examined >= options["max_separator_candidates"]:
                break
            examined += 1
            if closure(sources, children, subset) & targets:
                continue
            affected = closure(subset, children)
            subgraph = ToolPathGraph("repair", "repair", "", [],
                                     [n for n in graph.nodes if n.node_id in affected], [])
            calls, seconds = generation_credits(task, subgraph)
            boundary = {"nodes": list(subset), "affected_nodes": sorted(affected),
                "preserved_nodes": sorted(tools - affected), "generation_calls": calls,
                "generated_seconds": seconds, "other_output_nodes_affected": sorted(unaffected & affected)}
            boundaries.append(boundary)
        if examined >= options["max_separator_candidates"]:
            break
    boundaries.sort(key=lambda b: (len(b["other_output_nodes_affected"]), b["generated_seconds"],
                                   b["generation_calls"], len(b["nodes"]), b["nodes"]))
    return {**base, "status": "localized", "sources": sorted(sources),
            "boundaries": boundaries[:options["boundary_limit"]],
            "examined_subsets": examined, "bounded_search": True,
            "selection_objective": "lexicographic: preserve other outputs, then union regeneration duration/calls"}


def check_local_candidate(parent, child, frontier):
    """New nodes are allowed; edits to existing nodes must fit one frontier."""
    impact = repair_impact(parent, child)
    if frontier["status"] != "localized":
        return {"status": "global_exploration", "impact": impact}
    old = {n.node_id for n in parent.nodes}
    changed = set(impact["changed_nodes"]) & old
    affected = set(impact["affected_nodes"]) & old
    new_nodes = {n.node_id for n in child.nodes} - old
    attachments = {e.source for e in child.edges if e.source in old and e.target in new_nodes}
    for boundary in frontier["boundaries"]:
        allowed = set(boundary["affected_nodes"])
        if (changed <= allowed and affected <= allowed and
                (changed or attachments & allowed or not impact["changed_nodes"])):
            return {"status": "within_boundary", "boundary": boundary["nodes"], "impact": impact}
    raise ValueError("candidate modifies nodes outside every permitted repair boundary")


def preservation_report(before, after, threshold, tolerance, *, paired=True):
    """Check paired satisfied criteria/windows. Missing protected evidence rejects."""
    if len(before) != len(after) or not before:
        raise ValueError("preservation requires paired records")
    from evovideo_skill.story_contracts import preserves_mandatory
    violations, protected = [], 0
    for left, right in zip(before, after):
        if left["task_id"] != right["task_id"] or paired and left["seed"] != right["seed"]:
            raise ValueError("unpaired preservation records")
        if not preserves_mandatory(left, right):
            violations.append({"seed": left["seed"], "scope": "mandatory", "reason": "passed constraint regressed"})
        for scope, a, b in (("criterion", metric_vector(left), metric_vector(right)),
                            ("segment", segment_scores(left), segment_scores(right))):
            for key, value in a.items():
                if value < threshold:
                    continue
                protected += 1
                if key not in b or b[key] < value - tolerance - 1e-12:
                    violations.append({"seed": left["seed"], "scope": scope, "key": key,
                                       "before": value, "after": b.get(key)})
    return {"passed": not violations, "protected_observations": protected,
            "violations": violations, "threshold": threshold, "max_regression": tolerance,
            "qualification": "Only observed criteria/windows are protected; no inferred per-node approval."}
