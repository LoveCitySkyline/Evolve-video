from __future__ import annotations

from evovideo_skill.models import EvaluationReport, FailureReport, FailureType, VideoArtifact, VideoTask


class FailureDiagnoser:
    _REWARD_FAILURES = {
        "cross_shot_identity": FailureType.IDENTITY_DRIFT,
        "subject_consistency": FailureType.IDENTITY_DRIFT,
        "identity_consistency": FailureType.IDENTITY_DRIFT,
        "clothing_consistency": FailureType.CLOTHING_COLOR_DRIFT,
        "prop_persistence": FailureType.OBJECT_PERSISTENCE,
        "object_persistence": FailureType.OBJECT_PERSISTENCE,
        "prop_ownership": FailureType.OBJECT_PERSISTENCE,
        "object_removal_success": FailureType.PROMPT_OMISSION,
        "target_edit_success": FailureType.PROMPT_OMISSION,
        "non_target_preservation": FailureType.EDITING_LEAKAGE,
        "boundary_stability": FailureType.EDITING_LEAKAGE,
        "source_structure_preservation": FailureType.EDITING_LEAKAGE,
        "action_order": FailureType.MOTION_MISMATCH,
        "state_transition_accuracy": FailureType.MOTION_MISMATCH,
        "causal_order": FailureType.MOTION_MISMATCH,
        "interaction_order": FailureType.MOTION_MISMATCH,
        "action_attribution": FailureType.MOTION_MISMATCH,
        "motion_preservation": FailureType.MOTION_MISMATCH,
        "camera_motion_alignment": FailureType.MOTION_MISMATCH,
        "camera_path_order": FailureType.MOTION_MISMATCH,
        "focus_target_accuracy": FailureType.MOTION_MISMATCH,
        "audio_event_alignment": FailureType.MOTION_MISMATCH,
        "speaker_attribution": FailureType.MOTION_MISMATCH,
        "style_alignment": FailureType.STYLE_DRIFT,
        "style_consistency": FailureType.STYLE_DRIFT,
        "temporal_flicker": FailureType.TEMPORAL_FLICKER,
    }

    def diagnose(self, task: VideoTask, artifact: VideoArtifact, report: EvaluationReport) -> FailureReport | None:
        if report.passed:
            return None

        failure_types: list[FailureType] = []
        evidence: list[str] = []
        likely_causes: list[str] = []
        recommended_updates: list[str] = []
        expected_failure_types = self._map_expected_failures(
            (task.metadata or {}).get("expected_failure_modes") or []
        )

        for metric in report.failed_metrics:
            evidence.extend(metric.evidence)
            if metric.name == "identity_consistency":
                failure_types.append(FailureType.IDENTITY_DRIFT)
                likely_causes.append("pure text-to-video generation lacks a stable visual identity anchor")
                recommended_updates.append("create or refine character_consistency_skill")
            elif metric.name == "clothing_color_consistency":
                failure_types.append(FailureType.CLOTHING_COLOR_DRIFT)
                likely_causes.append("visual appearance constraints are not anchored across time")
                recommended_updates.append("track clothing color and add region-level color repair fallback")
            elif metric.name == "background_preservation":
                failure_types.append(FailureType.EDITING_LEAKAGE)
                likely_causes.append("global editing changed non-target regions")
                recommended_updates.append("create or refine region_constrained_editing_skill")
            elif metric.name == "prompt_action_alignment":
                failure_types.append(FailureType.MOTION_MISMATCH)
                likely_causes.append("temporal plan does not explicitly cover all requested actions")
                recommended_updates.append("refine temporal_planning_skill and motion_control_skill")
            elif metric.name == "target_edit_success":
                failure_types.append(FailureType.PROMPT_OMISSION)
                likely_causes.append("target object or edit operation was not grounded")
                recommended_updates.append("add target grounding before editing")
            elif metric.name == "vbench_dimension_alignment":
                observed = self._map_vbench_dimension(
                    str((task.metadata or {}).get("vbench_dimension") or "")
                )
                if observed is not None:
                    failure_types.append(observed)
                likely_causes.append("generated video failed the VBench-inspired dimension-specific check")
                recommended_updates.append("create or refine a skill for the failed VBench dimension")

        self._append_reward_failures(
            task,
            artifact,
            failure_types,
            evidence,
            likely_causes,
            recommended_updates,
        )

        # A strict verifier can fail on a task-specific rubric that has no
        # one-to-one legacy FailureType. Preserve the benchmark's declared
        # failure modes so the mutation planner never receives an empty cause.
        if not failure_types and expected_failure_types:
            failure_types.extend(expected_failure_types)
            evidence.append("strict evaluation failed without a legacy metric mapping")
            likely_causes.append("task-specific verifier criteria failed")
            recommended_updates.append("target the benchmark-declared failure modes")

        unique_failure_types = list(dict.fromkeys(failure_types))
        failed_segments = self._failed_segments(task, artifact, unique_failure_types)
        intervention = self._intervention_plan(unique_failure_types, failed_segments)
        return FailureReport(
            task_id=task.task_id,
            artifact_id=artifact.artifact_id,
            failure_types=unique_failure_types,
            evidence=list(dict.fromkeys(evidence)),
            likely_causes=list(dict.fromkeys(likely_causes)),
            recommended_updates=list(dict.fromkeys(recommended_updates)),
            expected_failure_types=list(dict.fromkeys(expected_failure_types)),
            failed_segments=failed_segments,
            intervention=intervention,
        )

    @staticmethod
    def _failed_segments(
        task: VideoTask,
        artifact: VideoArtifact,
        failure_types: list[FailureType],
    ) -> list[dict[str, object]]:
        vlm = artifact.metadata.get("vlm_evaluation") or {}
        raw_segments = vlm.get("failed_segments") if isinstance(vlm, dict) else None
        normalized: list[dict[str, object]] = []
        if isinstance(raw_segments, list):
            for raw in raw_segments[:3]:
                if not isinstance(raw, dict):
                    continue
                try:
                    start = max(0.0, min(1.0, float(raw.get("start_ratio", 0.0))))
                    end = max(start + 0.05, min(1.0, float(raw.get("end_ratio", 1.0))))
                except (TypeError, ValueError):
                    continue
                normalized.append(
                    {
                        "start_ratio": start,
                        "end_ratio": min(1.0, end),
                        "failed_criteria": [
                            str(item) for item in raw.get("failed_criteria", [])
                            if str(item).strip()
                        ],
                        "diagnosis": str(raw.get("diagnosis") or "localized verifier failure"),
                        "repair_instruction": str(
                            raw.get("repair_instruction")
                            or "Repair only this temporal span while preserving all other frames."
                        ),
                    }
                )
        if normalized:
            return normalized

        values = {failure.value for failure in failure_types}
        if values & {"identity_drift", "clothing_color_drift", "object_persistence_failure"}:
            start, end = 0.45, 0.95
        elif values & {"motion_mismatch", "prompt_omission"}:
            start, end = 0.25, 0.80
        else:
            start, end = 0.30, 0.75
        return [
            {
                "start_ratio": start,
                "end_ratio": end,
                "failed_criteria": sorted(values),
                "diagnosis": "fallback temporal localization from failed task criteria",
                "repair_instruction": (
                    "Correct the failed task requirements in this span only. Preserve identity, "
                    "objects, style, camera, and every healthy frame outside the span."
                ),
                "localization_source": "failure_type_fallback",
                "task_duration_seconds": task.duration_seconds,
            }
        ]

    @staticmethod
    def _intervention_plan(
        failure_types: list[FailureType],
        failed_segments: list[dict[str, object]],
    ) -> dict[str, object]:
        total_span = sum(
            max(0.0, float(item.get("end_ratio", 1.0)) - float(item.get("start_ratio", 0.0)))
            for item in failed_segments
        )
        failures = {failure.value for failure in failure_types}
        if failures <= {"temporal_flicker"}:
            level, strategy = 2, "video_postprocess"
        elif len(failed_segments) == 1 and total_span <= 0.70:
            level, strategy = 4, "localized_segment_repair"
        else:
            level, strategy = 5, "multi_segment_boundary_conditioned_repair"
        return {
            "level": level,
            "strategy": strategy,
            "preserve_healthy_content": True,
            "replace_whole_video": False,
            "failure_types": sorted(failures),
        }

    @classmethod
    def _append_reward_failures(
        cls,
        task: VideoTask,
        artifact: VideoArtifact,
        failure_types: list[FailureType],
        evidence: list[str],
        likely_causes: list[str],
        recommended_updates: list[str],
    ) -> None:
        reward = artifact.metadata.get("task_reward") or {}
        components = reward.get("components") if isinstance(reward, dict) else None
        if not isinstance(components, dict):
            return
        rubric = (task.metadata or {}).get("evaluation") or {}
        rubric = rubric if isinstance(rubric, dict) else {}
        for name, failure_type in cls._REWARD_FAILURES.items():
            if name not in rubric or name not in components:
                continue
            config = rubric.get(name) if isinstance(rubric.get(name), dict) else {}
            try:
                threshold = float(config.get("threshold", 0.75))
                score = float(components[name])
            except (TypeError, ValueError):
                continue
            if score >= threshold:
                continue
            failure_types.append(failure_type)
            evidence.append(f"reward criterion {name}={score:.3f}<{threshold:.3f}")
            likely_causes.append(f"task-specific criterion {name} is below threshold")
            recommended_updates.append(f"select a tool path that directly improves {name}")

    @staticmethod
    def _map_vbench_dimension(dimension: str) -> FailureType | None:
        mapping = {
            "subject_consistency": FailureType.IDENTITY_DRIFT,
            "overall_consistency": FailureType.IDENTITY_DRIFT,
            "color": FailureType.CLOTHING_COLOR_DRIFT,
            "background_consistency": FailureType.EDITING_LEAKAGE,
            "human_action": FailureType.MOTION_MISMATCH,
            "motion_smoothness": FailureType.MOTION_MISMATCH,
            "dynamic_degree": FailureType.MOTION_MISMATCH,
            "temporal_flickering": FailureType.TEMPORAL_FLICKER,
            "appearance_style": FailureType.STYLE_DRIFT,
            "temporal_style": FailureType.STYLE_DRIFT,
            "multiple_objects": FailureType.OBJECT_PERSISTENCE,
            "spatial_relationship": FailureType.OBJECT_PERSISTENCE,
        }
        return mapping.get(dimension.strip().lower())

    @staticmethod
    def _map_expected_failures(expected: list[str]) -> list[FailureType]:
        mapping = {
            FailureType.IDENTITY_DRIFT.value: FailureType.IDENTITY_DRIFT,
            FailureType.CLOTHING_COLOR_DRIFT.value: FailureType.CLOTHING_COLOR_DRIFT,
            FailureType.OBJECT_PERSISTENCE.value: FailureType.OBJECT_PERSISTENCE,
            FailureType.EDITING_LEAKAGE.value: FailureType.EDITING_LEAKAGE,
            FailureType.MOTION_MISMATCH.value: FailureType.MOTION_MISMATCH,
            FailureType.TEMPORAL_FLICKER.value: FailureType.TEMPORAL_FLICKER,
            FailureType.PROMPT_OMISSION.value: FailureType.PROMPT_OMISSION,
            FailureType.STYLE_DRIFT.value: FailureType.STYLE_DRIFT,
        }
        return [mapping[item] for item in expected if item in mapping]
