"""Opt-in replay of pure H3 nodes, scoped to one task/seed experiment.

Cache raw artifacts before executor/verifier annotations. Inputs and materialized
bytes are checked; neither a matching filename nor a stable tool name is proof
that an upstream artifact is unchanged.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

from evovideo_skill.models import VideoArtifact
from evovideo_skill.research_protocol import generation_credits, write_json
from evovideo_skill.research_subgraphs import stable_hash
from evovideo_skill.graph_skill import ToolPathGraph


def material_hashes(value):
    result = {}
    def visit(item):
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif isinstance(item, str) and len(item) < 4096:
            path = Path(unquote(urlparse(item).path)) if item.startswith("file://") else Path(item)
            try:
                if path.is_file():
                    if str(path.resolve()) in result:
                        return
                    digest = sha256()
                    with path.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    result[str(path.resolve())] = digest.hexdigest()
            except (OSError, ValueError):
                pass
    visit(value)
    return result


class ConditioningNodeCache:
    def __init__(self, root, namespace, charge):
        self.root = Path(root)
        self.namespace = namespace
        self.charge = charge
        self.hits = []
        self.misses = []

    @staticmethod
    def eligible(spec):
        return spec.name in {
            "mock_text_to_video", "h3_t2va", "h3_fl2va", "h3_ref2va",
            "h3_reference_bank", "h3_reference_select", "h3_reference_pack",
            "h3_frame_extract", "h3_audio_extract", "h3_reference_trim", "h3_av_concat",
        }

    def key(self, task, plan, node, inputs, spec):
        plan_data = asdict(plan)
        # These are graph-level labels, not inputs to the native H3 adapters.
        plan_data.pop("tool_chain", None)
        plan_data.pop("selected_skill_names", None)
        data = [self.namespace, asdict(task), plan_data, asdict(node),
                {k: asdict(v) for k, v in inputs.items()}, spec.to_dict()]
        return stable_hash([data, material_hashes(data)])

    def lookup(self, task, plan, node, inputs, spec):
        if not self.eligible(spec):
            return None
        path = self.root / (self.key(task, plan, node, inputs, spec) + ".json")
        if not path.exists():
            return None
        entry = json.loads(path.read_text())
        files = entry["files"]
        if files != material_hashes(list(files)):
            return None
        return VideoArtifact(**deepcopy(entry["artifact"]))

    def before_node(self, task, node, hit):
        calls, seconds = generation_credits(task, ToolPathGraph("budget", "budget", "", [], [node], []))
        self.charge(calls, seconds, hit)
        (self.hits if hit else self.misses).append(node.node_id)

    def store(self, task, plan, node, inputs, spec, artifact):
        if not self.eligible(spec):
            return
        data = asdict(artifact)
        write_json(self.root / (self.key(task, plan, node, inputs, spec) + ".json"),
                   {"artifact": data, "files": material_hashes(data)})
