from __future__ import annotations

from evovideo_skill.models import TaskMode, VideoTask


class SkillConditionedPromptRewriter:
    """Compose the actual generation prompt from selected skills and task metadata."""

    def rewrite(
        self,
        task: VideoTask,
        selected_skill_names: list[str],
        temporal_steps: list[str],
        constraints: list[str],
    ) -> tuple[str, list[str]]:
        prompt = task.prompt.strip()
        clauses: list[str] = []
        reasons: list[str] = []
        skills = set(selected_skill_names)
        metadata = task.metadata or {}
        vbench_dimension = metadata.get("vbench_dimension")
        eval_focus = metadata.get("eval_focus") or []

        if "character_consistency_skill" in skills or "preserve_identity" in constraints:
            clauses.extend(
                [
                    "Maintain the exact same character identity in every frame.",
                    "Preserve the same face, hairstyle, body shape, age, and clothing with no identity replacement or face morphing.",
                ]
            )
            reasons.append("character_consistency_skill/preserve_identity")

        if "preserve_clothing_color" in constraints:
            clauses.append("Keep all specified clothing colors unchanged for the full video; do not let outfits shift color or style.")
            reasons.append("preserve_clothing_color")

        if "temporal_planning_skill" in skills or vbench_dimension in {"human_action", "motion_smoothness", "dynamic_degree", "temporal_style"}:
            ordered = "; ".join(step.strip() for step in temporal_steps if step.strip())
            if ordered:
                clauses.append(f"Show the action sequence in this exact temporal order: {ordered}.")
            clauses.append("Make each action visibly complete before the next action begins; avoid skipping, merging, or reordering steps.")
            reasons.append("temporal_or_motion_skill")

        if "action_keyframe_i2v_skill" in skills:
            ordered = "; ".join(step.strip() for step in temporal_steps if step.strip())
            if ordered:
                clauses.append(f"Use the reference/action keyframe as an anchor, then realize these action beats in order: {ordered}.")
            clauses.append("Keep each key action visually distinct with a clear start, completion, and transition to the next action.")
            reasons.append("top_level_action_keyframe_i2v_skill")

        if "segment_generate_and_repair_skill" in skills:
            ordered = "; ".join(step.strip() for step in temporal_steps if step.strip())
            if ordered:
                clauses.append(f"Treat the video as consecutive temporal segments: {ordered}.")
            clauses.append("Do not compress multiple requested actions into a single static pose; each segment must visibly complete its assigned action.")
            reasons.append("top_level_segment_repair_skill")

        if "multi_keyframe_identity_lock_skill" in skills:
            clauses.append("Use the identity reference across multiple key poses; the same person, face, hairstyle, body shape, clothing, and accessories must persist through pose and camera changes.")
            clauses.append("If the subject changes pose, preserve identity from the reference instead of inventing a new person.")
            reasons.append("top_level_multi_keyframe_identity_lock_skill")

        if "multi_shot_character_continuity_skill" in skills:
            clauses.append("Plan the video as multiple coherent shots using one shared character sheet; preserve the exact same identity, face, hairstyle, body shape, outfit, and accessories across every shot.")
            clauses.append("Each shot should have a clear camera framing and action beat, while continuity of identity, clothing, style, and story state must be maintained between shots.")
            reasons.append("top_level_multi_shot_character_continuity_skill")

        if "style_reference_i2v_skill" in skills:
            clauses.append("Use the style and layout reference as a fixed anchor for lighting, color palette, background geometry, and rendering style.")
            clauses.append("Avoid style drift, background morphing, flicker, and unintended scene transitions across frames.")
            reasons.append("top_level_style_reference_i2v_skill")

        if "video_style_transfer_skill" in skills:
            clauses.append("Transform the source video into the target style while preserving the original motion, scene layout, timing, identity, object positions, and camera movement.")
            clauses.append("Apply the target visual style consistently to every frame; avoid temporal flicker, content warping, identity loss, and frame-to-frame style changes.")
            reasons.append("top_level_video_style_transfer_skill")

        if vbench_dimension in {"multiple_objects", "spatial_relationship", "temporal_style"}:
            clauses.append("Keep every named object visible, stable, and unchanged unless the prompt explicitly says it should move.")
            clauses.append("Avoid object disappearance, duplication, deformation, or unintended color changes.")
            reasons.append(f"vbench_{vbench_dimension}_object_persistence")

        if vbench_dimension in {"background_consistency", "temporal_flickering", "overall_consistency"}:
            clauses.append("Keep the background layout stable across frames with no sudden morphing, flickering, or disappearing scene elements.")
            reasons.append(f"vbench_{vbench_dimension}_stability")

        if vbench_dimension in {"appearance_style", "scene", "aesthetic_quality", "imaging_quality"}:
            clauses.append("Maintain the requested visual style, scene type, lighting, and image quality consistently throughout the video.")
            clauses.append("Avoid artifacts, blur, warped geometry, inconsistent style shifts, or scene contamination.")
            reasons.append(f"vbench_{vbench_dimension}_style_quality")

        if task.mode == TaskMode.EDITING or "region_constrained_editing_skill" in skills or "preserve_non_target_regions" in constraints:
            clauses.append("Only apply the requested change to the target object or region; keep all non-target regions, camera motion, background, and identity unchanged.")
            reasons.append("region_constrained_editing_skill/preserve_non_target_regions")

        if eval_focus:
            clauses.append("Evaluation focus: " + "; ".join(str(item) for item in eval_focus) + ".")
            reasons.append("vbench_eval_focus")

        if not clauses:
            return prompt, []

        deduped = []
        seen = set()
        for clause in clauses:
            if clause not in seen:
                deduped.append(clause)
                seen.add(clause)
        rewritten = prompt + "\n\nGeneration constraints:\n- " + "\n- ".join(deduped)
        return rewritten, reasons
