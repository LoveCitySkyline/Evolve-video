"""Task-local project evidence, separate from portable conditioning strategies.

Inspired by JarvisHub's project state and checked-generation contracts. This is
an observation layer, not an alternate executor, quality oracle or cache.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import uuid
import time

from evovideo_skill.research_protocol import append_json, graph_payload, write_json


VERSION = "conditioning-workspace-v1"
REFERENCE_FIELDS = ("id", "kind", "uri", "role", "semantic_role", "duration_seconds")


def reference_view(references):
    return [{key: deepcopy(ref[key]) for key in REFERENCE_FIELDS if key in ref}
            for ref in references if isinstance(ref, dict)]


def artifact_view(artifact):
    meta = artifact.metadata
    return {"artifact_id": artifact.artifact_id, "artifact_type": meta.get("artifact_type", "unknown"),
        "media": {key: deepcopy(meta[key]) for key in (
            "local_video_path", "local_image_path", "local_audio_path", "reference_image",
            "duration_seconds", "has_audio") if key in meta},
        "references": reference_view(meta.get("h3_references", [])),
        "submitted_conditioning": reference_view(meta.get("h3_conditioning", [])),
        "request_hash": meta.get("h3_request_hash"),
        "adapter_reports_consumption": meta.get("upstream_conditioning_consumed") is True,
        "semantic_validity": "unverified"}


def repair_impact(parent, child):
    """Conservative affected closure, not a promise of cache hits or causal blame."""
    old = {n.node_id: asdict(n) for n in parent.nodes}
    new = {n.node_id: asdict(n) for n in child.nodes}
    dirty = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    old_edges = {e.edge_id: asdict(e) for e in parent.edges}
    new_edges = {e.edge_id: asdict(e) for e in child.edges}
    changed_edges = {key for key in old_edges.keys() | new_edges.keys()
                     if old_edges.get(key) != new_edges.get(key)}
    for key in changed_edges:
        for edges in (old_edges, new_edges):
            if key in edges:
                dirty.add(edges[key]["target"])
    changed = set(dirty)
    # An edge-order change can change the ordered parent set passed to an adapter.
    for node in new:
        a = [asdict(e) for e in parent.edges if e.target == node]
        b = [asdict(e) for e in child.edges if e.target == node]
        if a != b:
            dirty.add(node)
            changed.add(node)
    edges = [*parent.edges, *child.edges]
    while True:
        expanded = dirty | {e.target for e in edges if e.source in dirty}
        if expanded == dirty:
            break
        dirty = expanded
    return {"changed_nodes": sorted(changed), "changed_edges": sorted(changed_edges),
        "removed_nodes": sorted(old.keys() - new.keys()),
        "affected_nodes": sorted(dirty & new.keys()),
        "reusable_candidates": sorted(new.keys() - dirty),
        "qualification": "Structural candidates only; cache requires matching task, seed, configuration and material bytes."}


def planner_project(task, parent, records, operation):
    """Only caller-provided runtime observations may cross the planning boundary."""
    for record in records:
        if record.get("task_id") != task.task_id or record.get("status") != "ok":
            raise ValueError("project observation must be a successful evaluation of the current task")
        episode = record.get("episode", "")
        if operation.startswith("test/adaptive/"):
            if episode != operation.rsplit("/", 1)[0]:
                raise ValueError("adaptive project observations must belong to this exact test episode")
        elif operation.startswith(("train/", "factorial/")):
            if episode != "train/" + task.task_id:
                raise ValueError("training project observations must come from this training task")
        else:
            raise ValueError("direct test and validation planning cannot observe generated evaluations")
    refs = reference_view(task.metadata.get("h3_references", []))
    if task.reference_video:
        from evovideo_skill.h3_media import _uri_identity
        source = _uri_identity(task.reference_video)
        same = [r for r in refs if _uri_identity(r.get("uri")) == source]
        if any(r.get("id") == "source-video" and r not in same for r in refs):
            raise ValueError("source-video collides with a different task.reference_video URI")
        if not same:
            refs.append({"id": "source-video", "kind": "video", "uri": task.reference_video,
                         "role": "reference_video"})
    # Keep every paired replicate; never pick the highest scoring seed as context.
    observations = []
    for record in records:
        nodes = record.get("feedback", {}).get("nodes", {})
        observations.append({"evaluation_id": record["evaluation_id"], "graph_id": record["graph_id"],
            "seed": record["seed"], "is_parent": record["graph_id"] == parent.graph_id,
            "artifacts": {node_id: {key: deepcopy(evidence[key]) for key in (
                "artifact_id", "tool", "input_nodes", "input_artifact_ids", "cache_hit",
                "artifact_state") if key in evidence} for node_id, evidence in nodes.items()},
            "quality_scope": "whole output; not a per-node semantic approval"})
    return {"version": VERSION, "references": refs,
        "reference_status": "supplied inputs; semantic correctness is not certified",
        "observations": observations, "has_measured_video": bool(observations),
        "generation_contract": {
            "fixed": ["original task and shot prompts", "task assets", "seed", "model", "rubric"],
            "editable": ["registered conditioning nodes", "reference selection and binding", "dependencies and temporal scope"],
            "forbidden": ["donor-task assets", "final/test-comparison scores", "unsupported model controls"],
            "approval": "A completed node is not a semantically accepted node. Preserve useful context but test uncertain conditions."}}


class ConditioningWorkspace:
    """One immutable attempt directory; node events survive a downstream failure.

    The enclosing runner owns the run lock. A new attempt never overwrites a prior
    interrupted attempt. Provider reconciliation and replay stay in existing code.
    """
    def __init__(self, root, evaluation_id, task, graph, seed, episode):
        self.root = Path(root) / evaluation_id / uuid.uuid4().hex
        self.path = self.root / "state.json"
        self.started_at = time.monotonic()
        self.generation_wall_seconds = None
        self.state = {"version": VERSION, "evaluation_id": evaluation_id,
            "task_id": task.task_id, "graph_id": graph.graph_id, "seed": seed, "episode": episode,
            "status": "running", "stage": "preflight", "sequence": 0,
            "graph": graph_payload(graph), "nodes": {node.node_id: {
                "tool": node.name, "node_type": node.node_type, "status": "planned"} for node in graph.nodes},
            "references": reference_view(task.metadata.get("h3_references", [])),
            "source_video": task.reference_video, "quality": None}
        self.event("attempt_started")

    def event(self, kind, node_id=None, **details):
        self.state["sequence"] += 1
        event = {"sequence": self.state["sequence"], "time": datetime.now(timezone.utc).isoformat(),
                 "event": kind, "node_id": node_id, **details}
        self.state["updated_at"] = event["time"]
        append_json(self.root / "events.jsonl", event)
        write_json(self.path, self.state)

    def node(self, node_id, status, **details):
        self.state["stage"] = "execution"
        self.state["nodes"][node_id].update(status=status, **deepcopy(details))
        self.event("node_" + status, node_id, **details)

    def completed(self, node_id, artifact, evidence):
        self.node(node_id, "completed", artifact=artifact_view(artifact), evidence=deepcopy(evidence))

    def generated(self, artifact):
        self.generation_wall_seconds = time.monotonic() - self.started_at
        self.state.update(stage="verification", output=artifact_view(artifact))
        self.event("generation_completed", generation_wall_seconds=self.generation_wall_seconds)

    def diagnostics(self):
        tools = [node for node in self.state["nodes"].values() if node["node_type"] == "tool"]
        completed = [node for node in tools if node["status"] == "completed"]
        return {"completed_tool_nodes": len(completed),
            "generation_wall_seconds": self.generation_wall_seconds,
            "timing_scope": "client generation phase including queue, cache and media processing; not GPU time",
            "cached_tool_nodes": sum(node.get("evidence", {}).get("cache_hit", False) for node in completed),
            "submitted_reference_count": sum(len(node.get("artifact", {}).get("submitted_conditioning", [])) for node in completed),
            "nodes_with_request_hash": sum(bool(node.get("artifact", {}).get("request_hash")) for node in completed),
            "not_quality_metrics": True}

    def verified(self, record):
        self.state.update(status="verified", stage="complete", quality={
            "score": record["score"], "criterion_scores": record["criterion_scores"],
            "scope": "whole_output", "selection_status": "not_decided_here"},
            acceptance=deepcopy(record.get("acceptance", {"status": "not_applicable"})),
            process_diagnostics=self.diagnostics())
        self.event("verification_completed")

    def failed(self, exc, status="failed"):
        stage = self.state["stage"]
        active = [key for key, value in self.state["nodes"].items() if value["status"] == "running"]
        for key in active:
            self.state["nodes"][key]["status"] = status
        for value in self.state["nodes"].values():
            if value["status"] == "planned":
                value["status"] = "not_executed"
        self.state.update(status=status, failed_stage=stage,
                          error={"type": type(exc).__name__, "message": str(exc)[-2000:]},
                          acceptance={"status": "unknown", "reason": "attempt did not complete verification"},
                          process_diagnostics=self.diagnostics())
        self.event("attempt_" + status, active_nodes=active, stage=stage, error=self.state["error"])


def main():
    parser = argparse.ArgumentParser(description="Inspect task-local project state without calling any models.")
    parser.add_argument("state", type=Path, help="Path to project_states/<evaluation>/<attempt>/state.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    state = json.loads(args.state.read_text())
    if args.json:
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return
    print(f"task={state['task_id']} seed={state['seed']} status={state['status']} stage={state['stage']}")
    for key, node in state["nodes"].items():
        cached = node.get("evidence", {}).get("cache_hit", False)
        print(f"  {key}: {node['tool']} [{node['status']}] cache_hit={cached}")
    if state.get("error"):
        print(f"error: {state['error']['message']}")
    print("Node completion is execution evidence, not a quality approval.")


if __name__ == "__main__":
    main()
