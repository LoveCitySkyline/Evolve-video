"""Keep reusable H3 interventions separate from discovery-task prose."""
from __future__ import annotations

import hashlib

from evovideo_skill.models import VideoTask


STRATEGIES = {
    "ordered_actions": (
        "Follow only the actions and entities in the current task. Complete each action before "
        "starting its successor. Track the same objects through every state transition; do not "
        "introduce the final state early. Preserve the requested camera continuity."
    ),
    "preserve_state": (
        "Use only the current task's entities and appearance. Keep static objects and scene "
        "geometry fixed, preserve identity through occlusion, and complete every requested action."
    ),
    "preserve_identity": (
        "Preserve the current task's subject, clothing and distinctive object details across "
        "all transitions. Use supplied visual references for continuity without omitting actions."
    ),
    "localized_repair": (
        "Continue from the supplied boundary using the current task and current video's repair "
        "evidence. Repair the specified interval while preserving identity, physical state and "
        "continuity with the surrounding footage."
    ),
}


def task_prompt_key(task: VideoTask) -> str:
    return hashlib.sha256(task.prompt.encode("utf-8")).hexdigest()


def stage_instruction(task: VideoTask, config: dict) -> tuple[str, bool]:
    strategy = config.get("conditioning_strategy")
    if strategy is not None and strategy not in STRATEGIES:
        raise ValueError(f"Unknown H3 conditioning_strategy: {strategy!r}")
    instruction = STRATEGIES.get(strategy, "")
    literal = config.get("prompt", "")
    bindings = config.get("prompt_task_hashes")
    applied = bool(literal) and (bindings is None or task_prompt_key(task) in bindings)
    if applied:
        instruction = "\n".join(filter(None, (instruction, literal)))
    return instruction, applied
