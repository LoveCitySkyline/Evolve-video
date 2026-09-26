from __future__ import annotations

import re

from evovideo_skill.models import TaskMode, VideoPlan, VideoTask
from evovideo_skill.prompt_rewriter import SkillConditionedPromptRewriter


COLOR_WORDS = ["red", "blue", "yellow", "green", "black", "white", "brown"]
SUBJECT_WORDS = ["woman", "girl", "boy", "man", "robot", "car", "dog", "product", "character", "detective", "dancer", "artist", "cyclist", "runner", "violinist"]
ACTION_WORDS = [
    "walk",
    "walks",
    "turn",
    "turns",
    "wave",
    "waves",
    "run",
    "runs",
    "jump",
    "jumps",
    "land",
    "lands",
    "laugh",
    "laughs",
    "pick",
    "picks",
    "place",
    "places",
    "slice",
    "slices",
    "push",
    "pushes",
    "stir",
    "stirs",
    "move",
    "moves",
    "change",
    "remove",
    "replace",
]


class Planner:
    def __init__(self, prompt_rewriter: SkillConditionedPromptRewriter | None = None):
        self.prompt_rewriter = prompt_rewriter or SkillConditionedPromptRewriter()

    def plan(self, task: VideoTask, selected_skill_names: list[str]) -> VideoPlan:
        prompt = task.prompt.lower()
        intent = {
            "subject": self._first_match(prompt, SUBJECT_WORDS) or "subject",
            "style": self._style(prompt),
            "clothing_color": self._clothing_color(prompt),
            "target_object": self._target_object(prompt),
            "target_color": self._target_color(prompt),
        }
        actions = [word for word in ACTION_WORDS if re.search(rf"\b{word}\b", prompt)]
        temporal_steps = self._temporal_steps(prompt, actions)
        constraints = self._constraints(task, intent)
        tool_chain = self._tool_chain(task, selected_skill_names)
        generation_prompt, rewrite_reasons = self.prompt_rewriter.rewrite(task, selected_skill_names, temporal_steps, constraints)
        return VideoPlan(
            task.task_id,
            intent,
            temporal_steps,
            constraints,
            selected_skill_names,
            tool_chain,
            generation_prompt,
            rewrite_reasons,
        )

    def _temporal_steps(self, prompt: str, actions: list[str]) -> list[str]:
        if not actions:
            return ["establish scene", "maintain scene", "finish scene"]
        parts = re.split(r"\bthen\b|,|;", prompt)
        steps = [part.strip() for part in parts if any(action in part for action in actions)]
        return steps or actions

    def _constraints(self, task: VideoTask, intent: dict[str, str | None]) -> list[str]:
        constraints: list[str] = []
        if intent.get("subject") in {"woman", "girl", "boy", "man", "character"}:
            constraints.append("preserve_identity")
        if intent.get("clothing_color"):
            constraints.append("preserve_clothing_color")
        if task.mode == TaskMode.EDITING and any(token in task.prompt.lower() for token in ["unchanged", "only", "while keeping"]):
            constraints.append("preserve_non_target_regions")
        return constraints

    def _tool_chain(self, task: VideoTask, selected_skill_names: list[str]) -> list[str]:
        selected = set(selected_skill_names)
        if task.mode == TaskMode.EDITING:
            if "video_style_transfer_skill" in selected:
                return ["mock_video_style_transfer"]
            if "region_constrained_editing_skill" in selected:
                return ["mock_region_video_editor"]
            return ["mock_global_video_editor"]
        if "multi_shot_character_continuity_skill" in selected:
            return ["mock_multi_shot_i2v"]
        if selected & {
            "character_consistency_skill",
            "action_keyframe_i2v_skill",
            "multi_keyframe_identity_lock_skill",
            "style_reference_i2v_skill",
        }:
            return ["mock_image_to_video"]
        return ["mock_text_to_video"]

    @staticmethod
    def _first_match(prompt: str, words: list[str]) -> str | None:
        for word in words:
            if re.search(rf"\b{word}\b", prompt):
                return word
        return None

    @staticmethod
    def _clothing_color(prompt: str) -> str | None:
        clothing_words = ["coat", "raincoat", "jacket", "shirt", "dress", "scarf", "helmet"]
        if not any(word in prompt for word in clothing_words):
            return None
        for color in COLOR_WORDS:
            if color in prompt:
                return color
        return None

    @staticmethod
    def _target_color(prompt: str) -> str | None:
        matches = [color for color in COLOR_WORDS if color in prompt]
        return matches[-1] if matches else None

    @staticmethod
    def _target_object(prompt: str) -> str | None:
        for word in ["car", "dog", "person", "background", "sky", "shirt", "coat", "video", "source video"]:
            if word in prompt:
                return word
        return None

    @staticmethod
    def _style(prompt: str) -> str:
        if "anime" in prompt:
            return "anime"
        if "cartoon" in prompt:
            return "cartoon"
        if "watercolor" in prompt:
            return "watercolor"
        if "cinematic" in prompt:
            return "cinematic"
        return "default"
