"""Shot-local generation and explicit, unverified state-transfer diagnostics."""
from copy import deepcopy
import json

VERSION = "scoped-story-generation-v1"
GENERATORS = {"mock_text_to_video", "h3_t2va", "h3_ref2va", "h3_fl2va"}


def shot_prompt(task, index):
    """Use the declared current state; never replay previous action prose.

    Unstructured/custom shot prompts retain their original behavior. Only shots
    rendered by story_semantics are rebuilt from their authoritative contract.
    """
    from evovideo_skill.story_semantics import render_shots, POLICY
    shot = task.metadata["h3_shots"][index]
    global_text = task.metadata.get("h3_global_constraints", "")
    contract = task.metadata.get("story_contract", {})
    if not contract.get("semantics"):
        return f"{global_text}\n{shot['prompt']}".strip(), None
    rendered = deepcopy(task)
    render_shots(rendered)
    if shot["prompt"] != rendered.metadata["h3_shots"][index]["prompt"]:
        # Do not silently discard custom camera/content instructions.
        return f"{global_text}\n{shot['prompt']}".strip(), None
    rule = contract["shots"][index]
    setup = contract["semantics"]["initial_setup"]
    removed = bool(setup and global_text.startswith(setup))
    persistent = global_text[len(setup):].strip() if removed else global_text
    lines = [f"Generate ONLY shot {index}. This clip is one stage of the story."]
    if persistent:
        lines.append(("Persistent appearance and scene constraints: " if removed else
                      "Global context; any initial state applies only to shot 0: ") + persistent)
    if index == 0:
        lines.append("Initial scene: " + setup)
    else:
        lines.append("Begin in the CURRENT required state below. Do not restart the initial setup "
                     "or repeat prior actions. Preserve the same objects through the cut.")
    lines.append("CURRENT required start state: " + json.dumps(rule["preconditions"], ensure_ascii=False))
    lines.extend("Perform this action in order: " + e["description"] for e in rule["events"])
    lines.append("Required end state: " + json.dumps(rule["postconditions"], ensure_ascii=False))
    if rule.get("invariants"):
        lines.append("Throughout this clip: " + json.dumps(rule["invariants"], ensure_ascii=False))
    lines.extend(["Appearance references define identity/layout, not a command to reset pose or object state. "
                  "A supplied first frame is the actual starting image, not proof it satisfies the required state.", POLICY])
    return "\n".join(lines), {"protocol": VERSION, "shot_index": index,
        "preconditions": deepcopy(rule["preconditions"]), "postconditions": deepcopy(rule["postconditions"]),
        "initial_setup_removed_from_global": removed,
        "state_satisfaction": "unverified; textual requirements do not establish visual continuity"}


def state_transfer_view(task, graph):
    """Recognize explicit last-frame -> first-frame wiring, never visual quality."""
    shots = task.metadata.get("story_contract", {}).get("shots", [])
    if not shots:
        return None
    nodes = {n.node_id: n for n in graph.nodes}
    incoming = {key: [nodes[e.source] for e in graph.edges if e.target == key and e.source in nodes
                      and e.condition == "always"] for key in nodes}
    links = []
    for node in graph.nodes:
        if node.name not in GENERATORS:
            continue
        index = node.config.get("shot_index")
        if not isinstance(index, int) or isinstance(index, bool) or index <= 0:
            continue
        for frame in incoming[node.node_id]:
            if (node.name != "h3_fl2va" or frame.name != "h3_frame_extract"
                    or frame.config.get("position") != "last" or frame.config.get("role") != "first_frame"):
                continue
            for source in incoming[frame.node_id]:
                if source.name in GENERATORS and source.config.get("shot_index") == index - 1:
                    links.append({"shot_index": index, "source_node": source.node_id,
                                  "frame_node": frame.node_id, "target_node": node.node_id})
    return {"protocol": VERSION, "declared_shots": len(shots), "explicit_boundary_links": links,
        "unconfirmed_entry_shots": [i for i in range(1, len(shots)) if not any(l['shot_index'] == i for l in links)],
        "qualification": "Structural wiring only; indirect reference paths are not certified here. "
            "The adapter's submitted conditioning records show actual consumption. A bad boundary can propagate errors.",
        "candidate_recipe": "To test state carry, extract the previous shot's last frame with "
            "h3_frame_extract(position=last, role=first_frame), connect it to h3_fl2va(shot_index=current), "
            "then concatenate the ordered shot videos. Use registered schemas; FL2VA cannot mix ordinary "
            "appearance references with endpoint frames. This is an optional measured intervention, not baseline behavior.",
        "cost_note": "Replacing independent generation with one linked call per shot retains generation call/second "
            "counts but serializes execution and adds frame extraction. Extra repairs cost additional calls/seconds. "
            "Measure quality and wall time; continuity is not guaranteed."}
