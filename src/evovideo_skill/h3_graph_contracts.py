"""Static H3 contracts shared by capability search and executable graph checks."""
from __future__ import annotations

from evovideo_skill.graph_skill import ToolPathGraph
from evovideo_skill.models import VideoTask


H3_CAPABILITY_ALIASES = {
    **{name: "h3_frame_extract" for name in (
        "frame_extraction", "video_frame_extraction", "keyframe_extraction",
        "physical_frame_extraction", "terminal_frame_extraction", "first_frame_extraction",
    )},
    "audio_extraction": "h3_audio_extract",
    "reference_packing": "h3_reference_pack",
    "reference_set_packing": "h3_reference_pack",
    "physical_reference_packaging": "h3_reference_pack",
    "reference_selection": "h3_reference_select",
    "reference_trimming": "h3_reference_trim",
    "task_conditioned_video_generation": "text_to_video",
    "video_concatenation": "h3_av_concat",
}


def validate_h3_reference_flow(graph: ToolPathGraph) -> None:
    """Reject provably invalid roles without opening files or generating upstream media.

    None denotes unknown reference contents, not an empty reference set. Dynamic
    task banks and third-party artifacts still require runtime validation.
    """
    memo: dict[str, list[tuple[str, str]] | None] = {}
    visiting: set[str] = set()

    def references(node_id: str) -> list[tuple[str, str]] | None:
        if node_id in memo:
            return memo[node_id]
        if node_id in visiting:
            return None
        visiting.add(node_id)
        node = graph.node(node_id)
        config = node.config
        parents = [edge.source for edge in graph.edges if edge.target == node_id]
        result = None
        if node.name == "h3_reference_pack":
            result = []
            for binding in config["bindings"]:
                source = graph.node(binding["source"])
                ident = binding.get("reference_id")
                if ident is None and source.name in {
                    "h3_reference_bank", "h3_reference_select", "h3_reference_trim", "h3_reference_pack",
                }:
                    upstream = references(source.node_id)
                    if upstream is None or len(upstream) != 1:
                        result = None
                        break
                    ident = upstream[0][0]
                result.append((ident or source.node_id, binding["role"]))
        elif node.name == "h3_frame_extract":
            role = config.get("role", {"first": "first_frame", "last": "last_frame", "time": "reference_image"}[config["position"]])
            result = [(node_id, role)]
        elif node.name in {"h3_reference_select", "h3_reference_trim"} and len(parents) == 1:
            result = references(parents[0])
            if node.name == "h3_reference_select":
                roles = dict(result or [])
                roles.update(config.get("roles", {}))
                ids = config["reference_ids"]
                if result is not None and any(key not in dict(result) for key in ids):
                    raise ValueError(f"{node_id}: reference_ids contains missing upstream IDs")
                result = [(key, roles[key]) for key in ids] if all(key in roles for key in ids) else None
        visiting.remove(node_id)
        memo[node_id] = result
        return result

    for node in graph.nodes:
        if node.name not in {"h3_fl2va", "h3_ref2va"}:
            continue
        groups = [references(edge.source) for edge in graph.edges if edge.target == node.node_id]
        known = [ref for group in groups if group is not None for ref in group]
        ids = [key for key, _ in known]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{node.node_id}: duplicate upstream reference IDs")
        selected = node.config.get("reference_ids")
        if selected:
            if all(group is not None for group in groups) and set(selected) - {key for key, _ in known}:
                raise ValueError(f"{node.node_id}: reference_ids contains missing upstream IDs")
            known = [ref for ref in known if ref[0] in selected]
        roles = [role for _, role in known]
        frames = [role for role in roles if role in {"first_frame", "last_frame"}]
        ordinary = [role for role in roles if role.startswith("reference_")]
        if frames and ordinary:
            raise ValueError(f"{node.node_id}: H3 frame inputs and reference inputs cannot be mixed; "
                             "use FL2VA endpoints only, or explicitly pack all images as reference_image for Ref2VA")
        if node.name == "h3_ref2va" and frames:
            raise ValueError(f"{node.node_id}: Ref2VA cannot consume first_frame/last_frame; "
                             "use FL2VA or explicitly assign reference_image (soft reference, not a hard boundary)")
        if node.name == "h3_fl2va" and ordinary:
            raise ValueError(f"{node.node_id}: FL2VA accepts only first_frame/last_frame, not ordinary references")
        if any(frames.count(role) > 1 for role in ("first_frame", "last_frame")):
            raise ValueError(f"{node.node_id}: H3 accepts at most one first frame and one last frame")


def validate_h3_composition_duration(graph: ToolPathGraph, task: VideoTask) -> None:
    """Check known terminal concat durations; never retime or truncate a candidate."""
    visiting: set[str] = set()

    def duration(node_id: str) -> float | None:
        if node_id in visiting:
            return None
        visiting.add(node_id)
        node = graph.node(node_id)
        result = None
        if node.name in {"h3_t2va", "h3_fl2va", "h3_ref2va", "mock_text_to_video"}:
            if node.config.get("conditioning_strategy") != "localized_repair":
                default = task.duration_seconds
                if "shot_index" in node.config:
                    index = node.config["shot_index"]
                    shots = task.metadata.get("h3_shots", [])
                    if not 0 <= index < len(shots):
                        raise ValueError(f"{node_id}: shot_index does not exist for task {task.task_id}")
                    default = shots[index].get("duration_seconds", default)
                result = node.config.get("duration_seconds", default)
        elif node.name == "h3_av_concat":
            values = [duration(source) for source in node.config["source_nodes"]]
            if all(value is not None for value in values):
                result = sum(values)
        visiting.remove(node_id)
        return result

    for node in graph.nodes:
        if node.name != "h3_av_concat":
            continue
        pending = [edge.target for edge in graph.edges if edge.source == node.node_id]
        seen = set()
        downstream_tool = False
        while pending:
            target = pending.pop()
            if target in seen:
                continue
            seen.add(target)
            if graph.node(target).node_type == "tool":
                downstream_tool = True
                break
            pending.extend(edge.target for edge in graph.edges if edge.source == target)
        if not downstream_tool:
            total = duration(node.node_id)
            if total is not None and abs(total - task.duration_seconds) > 0.5:
                raise ValueError(f"{node.node_id}: composed duration {total}s does not match task "
                                 f"{task.task_id} duration {task.duration_seconds}s; allocate explicit "
                                 "4..15 second segments for this task before generation")
