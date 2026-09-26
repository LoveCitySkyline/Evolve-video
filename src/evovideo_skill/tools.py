from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from copy import deepcopy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evovideo_skill.models import TaskMode, VideoArtifact, VideoPlan, VideoTask


@dataclass
class ToolExecutionContext:
    node_id: str
    node_config: dict[str, Any]
    input_artifacts: dict[str, VideoArtifact] = field(default_factory=dict)
    all_artifacts: dict[str, VideoArtifact] = field(default_factory=dict)
    state: Any = None

    @property
    def latest_artifact(self) -> VideoArtifact | None:
        return next(reversed(self.input_artifacts.values()), None)


class VideoTool(ABC):
    name: str

    @abstractmethod
    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        raise NotImplementedError

    def run_with_context(
        self,
        task: VideoTask,
        plan: VideoPlan,
        context: ToolExecutionContext,
    ) -> VideoArtifact:
        """Execute with upstream artifacts while preserving legacy tools."""
        return self.run(task, plan)


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()[:10]
    return digest


class TaskReferenceVideoTool(VideoTool):
    """Materialize the task's source video as a typed graph-root artifact."""

    name = "task_reference_video"

    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = Path(
            output_dir
            or os.environ.get("EVOVIDEO_REFERENCE_VIDEO_OUTPUT_DIR", "outputs/reference_videos")
        )

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        reference = str(task.reference_video or "").strip()
        if not reference:
            raise RuntimeError(
                f"task {task.task_id!r} requires reference_video for a source-video tool path"
            )
        source = Path(reference).expanduser()
        if source.is_file():
            video_url = source.resolve().as_uri()
        elif reference.startswith(("http://", "https://", "file://")):
            video_url = reference
        else:
            raise RuntimeError(f"task reference video does not exist: {reference}")

        from evovideo_skill.video_processing import VideoProcessor

        processed = VideoProcessor(
            self.output_dir,
            sample_count=max(1, int(os.environ.get("VLM_MAX_IMAGES", "6"))),
        ).process(video_url, f"{task.task_id}-source", task.prompt, {"plan": plan.__dict__})
        return VideoArtifact(
            artifact_id=_stable_id(task.task_id, self.name, processed.local_video_path),
            task_id=task.task_id,
            prompt=task.prompt,
            mode=task.mode,
            tool_chain=[self.name],
            frames=processed.frames,
            metadata={
                "artifact_type": "video",
                "local_video_path": processed.local_video_path,
                "reference_video": processed.local_video_path,
                "source_video": processed.local_video_path,
                "sampled_frame_paths": processed.sampled_frame_paths,
                "reference_video_root": True,
            },
        )


class MockTextToVideoTool(VideoTool):
    name = "mock_text_to_video"

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        frames = []
        prompt = task.prompt.lower()
        subject = plan.intent.get("subject", "subject")
        clothing_color = plan.intent.get("clothing_color")
        for idx in range(max(4, task.duration_seconds)):
            frame = {
                "index": idx,
                "subject": subject,
                "action": plan.temporal_steps[min(idx * len(plan.temporal_steps) // max(1, task.duration_seconds), len(plan.temporal_steps) - 1)],
                "style": plan.intent.get("style", "cinematic"),
                "background_changed": False,
                "target_edit_success": False,
            }
            if clothing_color:
                frame["clothing_color"] = clothing_color
            if any(word in prompt for word in ["woman", "girl", "boy", "man", "detective", "artist", "cyclist", "runner", "violinist", "dancer"]):
                frame["identity"] = "person_a" if idx < task.duration_seconds // 2 else "person_b"
            if clothing_color and idx >= task.duration_seconds // 2:
                frame["clothing_color"] = "dark brown"
            if ("correct order" in prompt or "no skipped action" in prompt) and idx >= task.duration_seconds // 2:
                frame["action"] = "ambiguous motion"
            frames.append(frame)
        return VideoArtifact(
            artifact_id=_stable_id(task.task_id, self.name, task.prompt),
            task_id=task.task_id,
            prompt=task.prompt,
            mode=task.mode,
            tool_chain=[self.name],
            frames=frames,
            metadata={"generator": "deterministic_mock", "known_failure": "identity_drift_for_people"},
        )

    def run_with_context(
        self,
        task: VideoTask,
        plan: VideoPlan,
        context: ToolExecutionContext,
    ) -> VideoArtifact:
        artifact = self.run(task, plan)
        temporal_states = []
        for upstream in context.input_artifacts.values():
            temporal_states.extend(upstream.metadata.get("temporal_states") or [])
        if temporal_states:
            for index, frame in enumerate(artifact.frames):
                frame["action"] = temporal_states[
                    min(index * len(temporal_states) // max(1, len(artifact.frames)), len(temporal_states) - 1)
                ]
        artifact.metadata["upstream_conditioning_consumed"] = bool(context.input_artifacts)
        artifact.metadata["planning_conditioning"] = list(dict.fromkeys(temporal_states))
        return artifact


class MockImageToVideoTool(VideoTool):
    name = "mock_image_to_video"

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        frames = []
        subject = plan.intent.get("subject", "subject")
        clothing_color = plan.intent.get("clothing_color")
        for idx in range(max(4, task.duration_seconds)):
            frames.append(
                {
                    "index": idx,
                    "subject": subject,
                    "identity": "anchored_identity",
                    "clothing_color": clothing_color,
                    "action": plan.temporal_steps[min(idx * len(plan.temporal_steps) // max(1, task.duration_seconds), len(plan.temporal_steps) - 1)],
                    "style": plan.intent.get("style", "cinematic"),
                    "background_changed": False,
                    "target_edit_success": False,
                }
            )
        return VideoArtifact(
            artifact_id=_stable_id(task.task_id, self.name, task.prompt),
            task_id=task.task_id,
            prompt=task.prompt,
            mode=task.mode,
            tool_chain=["mock_keyframe_generator", self.name],
            frames=frames,
            metadata={"generator": "deterministic_mock", "visual_anchor": True},
        )


class MockMultiShotImageToVideoTool(VideoTool):
    name = "mock_multi_shot_i2v"

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        frames = []
        subject = plan.intent.get("subject", "subject")
        clothing_color = plan.intent.get("clothing_color")
        shot_count = max(2, min(4, len(plan.temporal_steps)))
        for idx in range(max(4, task.duration_seconds)):
            shot_id = min(idx * shot_count // max(1, task.duration_seconds), shot_count - 1)
            frames.append(
                {
                    "index": idx,
                    "shot_id": shot_id,
                    "subject": subject,
                    "identity": "locked_character_sheet_identity",
                    "clothing_color": clothing_color,
                    "action": plan.temporal_steps[min(shot_id, len(plan.temporal_steps) - 1)],
                    "style": plan.intent.get("style", "cinematic"),
                    "background_changed": False,
                    "target_edit_success": False,
                    "cross_shot_identity_locked": True,
                }
            )
        return VideoArtifact(
            artifact_id=_stable_id(task.task_id, self.name, task.prompt),
            task_id=task.task_id,
            prompt=task.prompt,
            mode=task.mode,
            tool_chain=["mock_shot_planner", "mock_character_sheet_generator", "mock_keyframe_generator", self.name, "mock_shot_stitcher"],
            frames=frames,
            metadata={"generator": "deterministic_mock", "multi_shot": True, "character_sheet": True},
        )


class MockVideoStyleTransferTool(VideoTool):
    name = "mock_video_style_transfer"

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        frames = []
        for idx in range(max(4, task.duration_seconds)):
            frames.append(
                {
                    "index": idx,
                    "subject": plan.intent.get("subject", "source_subject"),
                    "identity": "source_identity_preserved",
                    "clothing_color": plan.intent.get("clothing_color"),
                    "action": plan.temporal_steps[min(idx * len(plan.temporal_steps) // max(1, task.duration_seconds), len(plan.temporal_steps) - 1)],
                    "style": "anime_style_locked",
                    "background_changed": False,
                    "target_edit_success": True,
                    "source_motion_preserved": True,
                    "temporal_flicker": False,
                }
            )
        return VideoArtifact(
            artifact_id=_stable_id(task.task_id, self.name, task.prompt),
            task_id=task.task_id,
            prompt=task.prompt,
            mode=TaskMode.EDITING,
            tool_chain=["mock_scene_splitter", "mock_structure_extractor", "mock_style_reference_generator", self.name, "mock_temporal_deflicker"],
            frames=frames,
            metadata={"generator": "deterministic_mock", "video_style_transfer": True, "target_style": "anime"},
        )


class MockRegionVideoEditor(VideoTool):
    name = "mock_region_video_editor"

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        frames = []
        for idx in range(max(4, task.duration_seconds)):
            frames.append(
                {
                    "index": idx,
                    "subject": plan.intent.get("target_object", "car"),
                    "target_color": plan.intent.get("target_color", "blue"),
                    "target_edit_success": True,
                    "background_changed": False,
                    "identity": "not_applicable",
                    "clothing_color": None,
                    "action": "edited target region only",
                    "style": plan.intent.get("style", "source_video"),
                }
            )
        return VideoArtifact(
            artifact_id=_stable_id(task.task_id, self.name, task.prompt),
            task_id=task.task_id,
            prompt=task.prompt,
            mode=TaskMode.EDITING,
            tool_chain=["mock_object_tracker", "mock_temporal_mask", self.name],
            frames=frames,
            metadata={"generator": "deterministic_mock", "region_constrained": True},
        )


class MockGlobalVideoEditor(VideoTool):
    name = "mock_global_video_editor"

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        frames = []
        for idx in range(max(4, task.duration_seconds)):
            frames.append(
                {
                    "index": idx,
                    "subject": plan.intent.get("target_object", "car"),
                    "target_color": plan.intent.get("target_color", "blue"),
                    "target_edit_success": True,
                    "background_changed": True,
                    "identity": "not_applicable",
                    "clothing_color": None,
                    "action": "global edit",
                    "style": "color shifted",
                }
            )
        return VideoArtifact(
            artifact_id=_stable_id(task.task_id, self.name, task.prompt),
            task_id=task.task_id,
            prompt=task.prompt,
            mode=TaskMode.EDITING,
            tool_chain=[self.name],
            frames=frames,
            metadata={"generator": "deterministic_mock", "known_failure": "editing_leakage"},
        )


class MockSegmentRepairTool(VideoTool):
    """Deterministic video-to-video repair used only by mock harness runs."""

    name = "segment_repair"

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        return self.run_with_context(task, plan, ToolExecutionContext(self.name, {}))

    def run_with_context(
        self,
        task: VideoTask,
        plan: VideoPlan,
        context: ToolExecutionContext,
    ) -> VideoArtifact:
        upstream = context.latest_artifact
        if upstream is None:
            raise RuntimeError("segment repair requires an upstream video")
        frames = deepcopy(upstream.frames)
        for index, frame in enumerate(frames):
            if plan.temporal_steps:
                frame["action"] = plan.temporal_steps[
                    min(index * len(plan.temporal_steps) // max(1, len(frames)), len(plan.temporal_steps) - 1)
                ]
            frame["segment_repaired"] = True
        metadata = deepcopy(upstream.metadata)
        metadata.update(
            {
                "artifact_type": "video",
                "segment_repair_applied": True,
                "upstream_conditioning_consumed": True,
            }
        )
        return VideoArtifact(
            _stable_id(task.task_id, context.node_id, upstream.artifact_id),
            task.task_id,
            task.prompt,
            upstream.mode,
            list(upstream.tool_chain) + [self.name],
            frames,
            metadata,
        )


class FailureSegmentLocalizerTool(VideoTool):
    """Convert clip-level verifier evidence into an executable temporal repair plan."""

    name = "failed_segment_localizer"

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        raise RuntimeError("failed segment localization requires the verified Wan draft")

    def run_with_context(
        self,
        task: VideoTask,
        plan: VideoPlan,
        context: ToolExecutionContext,
    ) -> VideoArtifact:
        source = context.latest_artifact
        if source is None:
            raise RuntimeError("failed segment localization requires an upstream video")
        vlm = source.metadata.get("vlm_evaluation") or {}
        if isinstance(vlm, dict) and (vlm.get("evaluation_status") == "needs_review"
                                     or str(vlm.get("evaluation_status", "")).startswith("failed")):
            raise RuntimeError("Cannot localize defects from unavailable or inconclusive verifier evidence")
        segments = self._normalize_segments(
            vlm.get("failed_segments") if isinstance(vlm, dict) else None,
            task,
            source,
        )
        total_span = sum(item["end_ratio"] - item["start_ratio"] for item in segments)
        level = 4 if len(segments) == 1 and total_span <= 0.70 else 5
        localization = {
            "segments": segments,
            "intervention_level": level,
            "strategy": (
                "localized_segment_repair"
                if level == 4
                else "multi_segment_boundary_conditioned_repair"
            ),
            "preserve_healthy_content": True,
            "replace_whole_video": False,
            "source_artifact_id": source.artifact_id,
        }
        metadata = deepcopy(source.metadata)
        metadata.update(
            {
                "artifact_type": "segment_plan",
                "failure_localization": localization,
                "repair_instruction": " ".join(
                    str(item["repair_instruction"]) for item in segments
                ),
                "source_video": source.metadata.get("local_video_path"),
                "upstream_conditioning_consumed": True,
            }
        )
        return VideoArtifact(
            _stable_id(task.task_id, context.node_id, source.artifact_id),
            task.task_id,
            task.prompt,
            source.mode,
            list(source.tool_chain) + [self.name],
            deepcopy(source.frames),
            metadata,
        )

    @staticmethod
    def _normalize_segments(
        raw_segments: Any,
        task: VideoTask,
        source: VideoArtifact,
    ) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        if isinstance(raw_segments, list):
            for raw in raw_segments[:3]:
                if not isinstance(raw, dict):
                    continue
                try:
                    start = max(0.0, min(0.95, float(raw.get("start_ratio", 0.0))))
                    end = max(start + 0.05, min(1.0, float(raw.get("end_ratio", 1.0))))
                except (TypeError, ValueError):
                    continue
                segments.append(
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
                            or "Repair this temporal span only while preserving healthy frames."
                        ),
                    }
                )
        if segments:
            return segments
        failure_types = set((source.metadata.get("vlm_evaluation") or {}).get("failure_types", []))
        if failure_types & {"identity_drift", "clothing_color_drift"}:
            start, end = 0.45, 0.95
        elif failure_types & {"motion_mismatch", "prompt_omission"}:
            start, end = 0.25, 0.80
        else:
            start, end = 0.30, 0.75
        return [
            {
                "start_ratio": start,
                "end_ratio": end,
                "failed_criteria": sorted(str(item) for item in failure_types),
                "diagnosis": "fallback localization from clip-level verifier evidence",
                "repair_instruction": (
                    "Correct only the failed actions or appearance in this span. Preserve the "
                    "Wan draft before and after it without restyling or replacing healthy content."
                ),
                "duration_seconds": task.duration_seconds,
            }
        ]


class BoundaryFrameExtractorTool(VideoTool):
    """Extract materialized frames around a localized failure for I2V conditioning."""

    name = "boundary_frame_extractor"

    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = Path(
            output_dir
            or os.environ.get("EVOVIDEO_BOUNDARY_FRAME_DIR", "outputs/boundary_frames")
        )

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        raise RuntimeError("boundary extraction requires a Wan draft and segment plan")

    def run_with_context(
        self,
        task: VideoTask,
        plan: VideoPlan,
        context: ToolExecutionContext,
    ) -> VideoArtifact:
        artifacts = list(context.input_artifacts.values())
        segment_artifact = next(
            (item for item in artifacts if item.metadata.get("failure_localization")),
            None,
        )
        source = next(
            (
                item for item in artifacts
                if item.metadata.get("artifact_type") == "video"
                and item.metadata.get("local_video_path")
            ),
            segment_artifact,
        )
        if segment_artifact is None or source is None:
            raise RuntimeError("boundary extraction requires video pixels and failure localization")
        video_path = source.metadata.get("local_video_path") or segment_artifact.metadata.get("source_video")
        segments = segment_artifact.metadata["failure_localization"]["segments"]
        segment = segments[0]
        if not video_path or not Path(str(video_path)).is_file():
            if source.metadata.get("generator") != "deterministic_mock" or not source.frames:
                raise RuntimeError("boundary extraction requires a materialized local Wan draft")
            start_index = min(
                len(source.frames) - 1,
                max(0, round(float(segment["start_ratio"]) * (len(source.frames) - 1))),
            )
            end_index = min(
                len(source.frames) - 1,
                max(0, round(float(segment["end_ratio"]) * (len(source.frames) - 1))),
            )
            boundary_frames = [
                {**deepcopy(source.frames[start_index]), "boundary_role": "before_failure"},
                {**deepcopy(source.frames[end_index]), "boundary_role": "after_failure"},
            ]
            return VideoArtifact(
                _stable_id(task.task_id, context.node_id, segment_artifact.artifact_id),
                task.task_id,
                task.prompt,
                source.mode,
                list(dict.fromkeys(source.tool_chain + segment_artifact.tool_chain + [self.name])),
                boundary_frames,
                {
                    "artifact_type": "image",
                    "failure_localization": deepcopy(segment_artifact.metadata["failure_localization"]),
                    "repair_instruction": segment_artifact.metadata.get("repair_instruction"),
                    "materialized_boundary_frames": False,
                    "mock_boundary_frames": True,
                    "upstream_conditioning_consumed": True,
                },
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        start_path = self._extract(
            str(video_path),
            max(0.0, float(segment["start_ratio"]) - 0.02),
            self.output_dir / f"{task.task_id}-{context.node_id}-start.png",
        )
        end_path = self._extract(
            str(video_path),
            min(1.0, float(segment["end_ratio"]) + 0.02),
            self.output_dir / f"{task.task_id}-{context.node_id}-end.png",
        )
        return VideoArtifact(
            _stable_id(task.task_id, context.node_id, segment_artifact.artifact_id),
            task.task_id,
            task.prompt,
            source.mode,
            list(dict.fromkeys(source.tool_chain + segment_artifact.tool_chain + [self.name])),
            [
                {"index": 0, "frame_path": start_path, "boundary_role": "before_failure"},
                {"index": 1, "frame_path": end_path, "boundary_role": "after_failure"},
            ],
            {
                "artifact_type": "image",
                "reference_image": start_path,
                "boundary_end_image": end_path,
                "keyframes": [start_path, end_path],
                "failure_localization": deepcopy(segment_artifact.metadata["failure_localization"]),
                "repair_instruction": segment_artifact.metadata.get("repair_instruction"),
                "source_video": str(video_path),
                "materialized_boundary_frames": True,
                "upstream_conditioning_consumed": True,
            },
        )

    @staticmethod
    def _extract(video_path: str, ratio: float, output: Path) -> str:
        try:
            import cv2
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("opencv-python is required for boundary extraction") from exc
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"failed to open Wan draft {video_path}")
        count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, min(count - 1, max(0, round(ratio * (count - 1)))))
        ok, frame = capture.read()
        capture.release()
        if not ok or not cv2.imwrite(str(output), frame):
            raise RuntimeError(f"failed to materialize boundary frame {output}")
        return str(output.resolve())


class SegmentStitcherTool(VideoTool):
    """Replace only localized failed spans and retain healthy Wan draft pixels."""

    name = "segment_stitcher"

    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = Path(
            output_dir
            or os.environ.get("EVOVIDEO_SEGMENT_STITCH_DIR", "outputs/segment_stitches")
        )

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        raise RuntimeError("segment stitching requires a Wan draft, repair plan, and repaired segment")

    def run_with_context(
        self,
        task: VideoTask,
        plan: VideoPlan,
        context: ToolExecutionContext,
    ) -> VideoArtifact:
        artifacts = list(context.input_artifacts.values())
        localization = next(
            (item for item in artifacts if item.metadata.get("failure_localization")),
            None,
        )
        if localization is None:
            raise RuntimeError("segment stitcher requires failure localization")
        source_path = localization.metadata.get("source_video") or localization.metadata.get("local_video_path")
        repaired = next(
            (
                item for item in reversed(artifacts)
                if item.metadata.get("artifact_type") == "video"
                and item is not localization
                and str(item.metadata.get("local_video_path")) != str(source_path)
            ),
            None,
        )
        segments = localization.metadata["failure_localization"]["segments"]
        repaired_path = repaired.metadata.get("local_video_path") if repaired is not None else None
        if (
            source_path
            and repaired_path
            and Path(str(source_path)).is_file()
            and Path(str(repaired_path)).is_file()
        ):
            output_path = self._stitch_video(
                task,
                str(source_path),
                str(repaired_path),
                segments[0],
                context.node_id,
                str(repaired.metadata.get("repair_timeline", "segment")),
            )
            from evovideo_skill.video_processing import VideoProcessor

            processed = VideoProcessor(
                self.output_dir,
                sample_count=max(16, len(localization.metadata.get("sampled_frame_paths") or []),
                                 len(repaired.metadata.get("sampled_frame_paths") or [])),
            ).process(Path(output_path).resolve().as_uri(), f"{task.task_id}-stitched", task.prompt)
            frames = processed.frames
            sampled = processed.sampled_frame_paths
            local_video_path = processed.local_video_path
        else:
            original = next((item for item in artifacts if item is not repaired and item.frames), localization)
            repaired_frames = (
                repaired.frames
                if repaired is not None and repaired.frames
                else original.frames
            )
            frames = deepcopy(original.frames)
            start = int(float(segments[0]["start_ratio"]) * len(frames))
            end = max(start + 1, int(float(segments[0]["end_ratio"]) * len(frames)))
            for index in range(start, min(end, len(frames))):
                source_index = min(
                    len(repaired_frames) - 1,
                    max(0, round((index - start) * len(repaired_frames) / max(1, end - start))),
                )
                frames[index] = deepcopy(repaired_frames[source_index])
                frames[index]["index"] = index
                frames[index]["segment_repaired"] = True
            sampled = [
                str(item) for item in localization.metadata.get("sampled_frame_paths", [])
            ]
            local_video_path = source_path
        metadata = deepcopy(localization.metadata)
        # Localization carries draft evidence for planning. Once pixels change,
        # those scores are stale and the final artifact must be evaluated again.
        for key in (
            "vlm_evaluation",
            "vlm_evaluation_error",
            "vlm_model",
            "task_reward",
        ):
            metadata.pop(key, None)
        metadata.update(
            {
                "artifact_type": "video",
                "local_video_path": local_video_path,
                "sampled_frame_paths": sampled,
                "healthy_content_preserved": True,
                "whole_video_regenerated": False,
                "repaired_segments": deepcopy(segments[:1]),
                "unrepaired_segments": deepcopy(segments[1:]),
                "repair_timeline": repaired.metadata.get("repair_timeline", "segment") if repaired else "segment",
                "audio_policy": "preserve_original_source_tracks",
                "upstream_conditioning_consumed": True,
            }
        )
        chain = list(dict.fromkeys(
            tool for artifact in artifacts for tool in artifact.tool_chain
        )) + [self.name]
        return VideoArtifact(
            _stable_id(task.task_id, context.node_id, *(item.artifact_id for item in artifacts)),
            task.task_id,
            task.prompt,
            localization.mode,
            chain,
            frames,
            metadata,
        )

    def _stitch_video(
        self,
        task: VideoTask,
        source_path: str,
        repair_path: str,
        segment: dict[str, Any],
        node_id: str,
        repair_timeline: str = "segment",
    ) -> str:
        try:
            import cv2
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("opencv-python is required for segment stitching") from exc
        ffmpeg = shutil.which(os.environ.get("EVOVIDEO_FFMPEG_BIN", "ffmpeg"))
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required for healthy-content-preserving segment stitching")
        capture = cv2.VideoCapture(source_path)
        if not capture.isOpened():
            raise RuntimeError(f"failed to open Wan draft {source_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 16.0
        frames = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = max(2, int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = max(2, int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        capture.release()
        duration = frames / fps
        start = max(0.0, min(duration, float(segment["start_ratio"]) * duration))
        end = max(start + 1.0 / fps, min(duration, float(segment["end_ratio"]) * duration))
        repair_duration = end - start
        repair_capture = cv2.VideoCapture(repair_path)
        if not repair_capture.isOpened():
            raise RuntimeError(f"failed to open repair video {repair_path}")
        repair_fps = float(repair_capture.get(cv2.CAP_PROP_FPS))
        repair_frames = int(repair_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        repair_capture.release()
        if repair_fps <= 0 or repair_frames <= 0:
            raise RuntimeError("repair video has no valid duration")
        available_duration = repair_frames / repair_fps
        if repair_timeline == "segment":
            repair_filter = f"setpts=(PTS-STARTPTS)*{repair_duration / available_duration:.9f},"
        elif repair_timeline == "full_video":
            if available_duration + 1 / repair_fps < end:
                raise RuntimeError("full-video repair does not cover the target interval")
            repair_filter = f"trim=start={start:.6f}:end={end:.6f},setpts=PTS-STARTPTS,"
        else:
            raise RuntimeError(f"unknown repair timeline: {repair_timeline}")
        filters: list[str] = []
        labels: list[str] = []
        if start > 1.0 / fps:
            filters.append(f"[0:v]trim=start=0:end={start:.6f},setpts=PTS-STARTPTS[vpre]")
            labels.append("[vpre]")
        filters.append(
            f"[1:v]{repair_filter}scale={width}:{height},setsar=1,fps={fps:.6f},"
            f"tpad=stop_mode=clone:stop_duration={repair_duration:.6f},"
            f"trim=duration={repair_duration:.6f},setpts=PTS-STARTPTS[vrepair]"
        )
        labels.append("[vrepair]")
        if end < duration - 1.0 / fps:
            filters.append(
                f"[0:v]trim=start={end:.6f}:end={duration:.6f},setpts=PTS-STARTPTS[vpost]"
            )
            labels.append("[vpost]")
        filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0[vout]")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output = self.output_dir / (
            f"{task.task_id}-{node_id}-{_stable_id(source_path, repair_path, str(segment), repair_timeline)}.mp4"
        )
        command = [
            ffmpeg, "-y", "-i", source_path, "-i", repair_path,
            "-filter_complex", ";".join(filters), "-map", "[vout]",
            "-map", "0:a?", "-c:a", "aac", "-t", f"{duration:.6f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not output.is_file():
            raise RuntimeError(
                "segment stitching failed: " + (completed.stderr or completed.stdout)[-2000:]
            )
        return str(output.resolve())


class ArtifactTransformTool(VideoTool):
    """Deterministic artifact-producing implementation for graph support nodes."""

    def __init__(self, name: str):
        self.name = name

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        return self.run_with_context(task, plan, ToolExecutionContext(self.name, {}))

    def run_with_context(
        self,
        task: VideoTask,
        plan: VideoPlan,
        context: ToolExecutionContext,
    ) -> VideoArtifact:
        upstream = context.latest_artifact
        if upstream is None:
            frames = [
                {
                    "index": index,
                    "subject": plan.intent.get("subject", "subject"),
                    "action": step,
                    "style": plan.intent.get("style", "default"),
                }
                for index, step in enumerate(plan.temporal_steps)
            ]
            mode = task.mode
            metadata: dict[str, Any] = {}
            chain: list[str] = []
        else:
            frames = deepcopy(upstream.frames)
            mode = upstream.mode
            metadata = deepcopy(upstream.metadata)
            chain = list(upstream.tool_chain)

        # Multi-input planning nodes must preserve both real pixel references and
        # symbolic plans instead of silently keeping only the latest predecessor.
        for input_artifact in context.input_artifacts.values():
            for key in ("sampled_frame_paths", "temporal_states"):
                values = input_artifact.metadata.get(key) or []
                if values:
                    metadata[key] = list(dict.fromkeys([*(metadata.get(key) or []), *values]))
            for key in ("reference_image", "local_video_path", "source_video"):
                value = input_artifact.metadata.get(key)
                if value and not metadata.get(key):
                    metadata[key] = value

        artifact_type = {
            "extract_reference_identity_frame": "identity_reference",
            "scene_splitter": "shot_plan",
            "character_sheet_generator": "character_sheet",
            "temporal_decomposer": "temporal_plan",
            "keyframe_generator": "keyframes",
            "structure_extractor": "structure_motion_map",
            "temporal_deflicker": "video",
            "object_tracker": "tracked_regions",
        }.get(self.name, "intermediate")
        metadata.update(
            {
                "artifact_type": artifact_type,
                "producer_node": context.node_id,
                "producer_tool": self.name,
                "upstream_artifact_ids": [item.artifact_id for item in context.input_artifacts.values()],
            }
        )
        if self.name == "extract_reference_identity_frame":
            metadata["reference_image"] = self._reference_frame(upstream)
        elif self.name == "character_sheet_generator":
            metadata["character_sheet"] = {
                "subject": plan.intent.get("subject"),
                "clothing_color": plan.intent.get("clothing_color"),
            }
        elif self.name == "temporal_decomposer":
            metadata["temporal_states"] = list(plan.temporal_steps)
        elif self.name == "keyframe_generator":
            materialized = [
                str(path) for path in metadata.get("sampled_frame_paths") or []
                if isinstance(path, str) and Path(path).is_file()
            ]
            metadata["keyframes"] = materialized or [frame.get("frame_path") or frame for frame in frames]
        elif self.name == "structure_extractor":
            metadata["source_video"] = task.reference_video or metadata.get("local_video_path")
            metadata["structure_preserved"] = True
        elif self.name == "object_tracker":
            metadata["tracked_target"] = plan.intent.get("target_object")
        elif self.name == "temporal_deflicker":
            metadata["temporal_deflicker_applied"] = True
            for frame in frames:
                frame["temporal_flicker"] = False

        return VideoArtifact(
            artifact_id=_stable_id(task.task_id, context.node_id, self.name, *(item.artifact_id for item in context.input_artifacts.values())),
            task_id=task.task_id,
            prompt=task.prompt,
            mode=mode,
            tool_chain=chain + [self.name],
            frames=frames,
            metadata=metadata,
        )

    @staticmethod
    def _reference_frame(artifact: VideoArtifact | None) -> str | dict[str, Any] | None:
        if artifact is None:
            return None
        sampled = artifact.metadata.get("sampled_frame_paths") or []
        if sampled:
            return sampled[0]
        for frame in artifact.frames:
            if frame.get("frame_path"):
                return frame["frame_path"]
        return artifact.frames[0] if artifact.frames else None


class ArtifactBridgeTool(VideoTool):
    """Materialize deterministic adapters between repository-specific artifact contracts."""

    def __init__(self, name: str, output_dir: str | Path | None = None):
        self.name = name
        self.output_dir = Path(output_dir or os.environ.get("EVOVIDEO_BRIDGE_OUTPUT_DIR", "outputs/artifact_bridges"))

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        return self.run_with_context(task, plan, ToolExecutionContext(self.name, {}))

    def run_with_context(
        self,
        task: VideoTask,
        plan: VideoPlan,
        context: ToolExecutionContext,
    ) -> VideoArtifact:
        upstream = context.latest_artifact
        if upstream is None:
            raise RuntimeError(f"artifact bridge {self.name} requires an upstream artifact")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = dict(context.node_config.get("target_contract") or {})
        if self.name == "bridge_extract_reference_frame":
            return self._extract_frame(task, upstream, context, target)
        if self.name == "bridge_materialize_image":
            return self._materialize_image(task, plan, upstream, context, target)
        if self.name == "bridge_compose_reference_images":
            return self._compose_reference_images(task, plan, upstream, context, target)
        if self.name == "bridge_normalize_image":
            return self._normalize_image(task, upstream, context, target)
        if self.name == "bridge_normalize_video":
            return self._normalize_video(task, plan, upstream, context, target)
        raise RuntimeError(f"unsupported artifact bridge {self.name!r}")

    def _compose_reference_images(
        self,
        task: VideoTask,
        plan: VideoPlan,
        upstream: VideoArtifact,
        context: ToolExecutionContext,
        target: dict[str, Any],
    ) -> VideoArtifact:
        """Materialize several semantic/image controls into one adapter input."""
        try:
            import cv2
            import numpy as np
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("opencv-python and numpy are required for reference composition") from exc

        artifacts = list(context.input_artifacts.values()) or [upstream]
        panels: list[Any] = []
        roles: list[str] = []
        for artifact in artifacts:
            sources = self._materialized_image_sources(artifact)
            if sources:
                image = cv2.imread(sources[0], cv2.IMREAD_COLOR)
                if image is not None:
                    panels.append(image)
                    roles.append(str(artifact.metadata.get("artifact_type") or "image"))
                    continue
            panel = np.full((576, 768, 3), (242, 242, 242), dtype=np.uint8)
            artifact_type = str(artifact.metadata.get("artifact_type") or "image")
            self._draw_semantic_storyboard(panel, task, plan, artifact, artifact_type)
            panels.append(panel)
            roles.append(artifact_type)
        if not panels:
            raise RuntimeError("reference composition received no usable visual artifacts")

        width = int(target.get("width") or 1024)
        height = int(target.get("height") or 576)
        canvas = np.full((height, width, 3), (242, 242, 242), dtype=np.uint8)
        panel_width = max(1, width // len(panels))
        for index, image in enumerate(panels):
            scale = min(panel_width / image.shape[1], height / image.shape[0])
            resized = cv2.resize(
                image,
                (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
                interpolation=cv2.INTER_LANCZOS4,
            )
            x0 = index * panel_width + max(0, (panel_width - resized.shape[1]) // 2)
            y0 = max(0, (height - resized.shape[0]) // 2)
            canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
        output = self.output_dir / (
            f"{task.task_id}-{context.node_id}-{_stable_id(*(a.artifact_id for a in artifacts))}.png"
        )
        if not cv2.imwrite(str(output), canvas):
            raise RuntimeError(f"failed to write composed reference image {output}")
        result = self._image_artifact(
            task,
            upstream,
            context,
            {**target, "artifact_type": "image", "semantic_role": "composed_reference"},
            str(output),
            "composed_reference_images",
        )
        result.tool_chain = list(dict.fromkeys(
            tool for artifact in artifacts for tool in artifact.tool_chain
        )) + [self.name]
        result.metadata.update({
            "composed_reference_roles": roles,
            "composed_upstream_artifact_ids": [artifact.artifact_id for artifact in artifacts],
        })
        return result

    def _extract_frame(
        self,
        task: VideoTask,
        upstream: VideoArtifact,
        context: ToolExecutionContext,
        target: dict[str, Any],
    ) -> VideoArtifact:
        sampled = [str(item) for item in upstream.metadata.get("sampled_frame_paths", []) if isinstance(item, str)]
        role = str(target.get("semantic_role") or "first_frame")
        source = sampled[-1] if sampled and role == "last_frame" else (sampled[0] if sampled else None)
        if source is None:
            video_path = upstream.metadata.get("local_video_path")
            if not video_path:
                raise RuntimeError("frame bridge requires sampled frames or local_video_path")
            source = self._read_video_frame(str(video_path), role, task.task_id)
        if target.get("width") or target.get("height") or target.get("formats"):
            source = self._normalize_source_image(str(source), target, task.task_id, context.node_id, upstream.artifact_id)
        return self._image_artifact(task, upstream, context, target, source, "extracted_reference_frame")

    def _materialize_image(
        self,
        task: VideoTask,
        plan: VideoPlan,
        upstream: VideoArtifact,
        context: ToolExecutionContext,
        target: dict[str, Any],
    ) -> VideoArtifact:
        """Compile symbolic keyframe/character metadata into an adapter-readable image."""
        try:
            import cv2
            import numpy as np
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("opencv-python and numpy are required for image materialization") from exc

        artifact_type = str(upstream.metadata.get("artifact_type") or "image")
        width = int(target.get("width") or (1024 if artifact_type == "keyframes" else 768))
        height = int(target.get("height") or (576 if artifact_type == "keyframes" else 768))
        canvas = np.full((height, width, 3), (242, 242, 242), dtype=np.uint8)
        sources = self._materialized_image_sources(upstream)
        loaded = [cv2.imread(path, cv2.IMREAD_COLOR) for path in sources]
        loaded = [image for image in loaded if image is not None]
        if loaded:
            panel_width = max(1, width // len(loaded))
            for index, image in enumerate(loaded):
                scale = min(panel_width / image.shape[1], height / image.shape[0])
                resized = cv2.resize(
                    image,
                    (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
                    interpolation=cv2.INTER_LANCZOS4,
                )
                x0 = index * panel_width + max(0, (panel_width - resized.shape[1]) // 2)
                y0 = max(0, (height - resized.shape[0]) // 2)
                canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
            mode = "reference_contact_sheet"
        else:
            self._draw_semantic_storyboard(canvas, task, plan, upstream, artifact_type)
            mode = "semantic_storyboard"

        formats = target.get("formats") or ["png"]
        suffix = str(formats[0]).lower().lstrip(".")
        suffix = "jpg" if suffix == "jpeg" else suffix
        output = self.output_dir / (
            f"{task.task_id}-{context.node_id}-{_stable_id(upstream.artifact_id, str(target))}.{suffix}"
        )
        if not cv2.imwrite(str(output), canvas):
            raise RuntimeError(f"failed to write materialized bridge image {output}")
        result = self._image_artifact(
            task, upstream, context, target, str(output), "materialized_symbolic_image"
        )
        result.metadata["materialization_mode"] = mode
        if artifact_type == "keyframes":
            result.metadata["keyframes"] = [str(output)]
        if artifact_type == "character_sheet":
            result.metadata["character_sheet_path"] = str(output)
        return result

    @staticmethod
    def _materialized_image_sources(artifact: VideoArtifact) -> list[str]:
        candidates: list[Any] = [artifact.metadata.get("reference_image")]
        candidates.extend(artifact.metadata.get("sampled_frame_paths") or [])
        candidates.extend(artifact.metadata.get("keyframes") or [])
        for frame in artifact.frames:
            candidates.append(frame.get("frame_path"))
        paths: list[str] = []
        for candidate in candidates:
            if isinstance(candidate, str) and Path(candidate).is_file() and candidate not in paths:
                paths.append(candidate)
        return paths[:6]

    @staticmethod
    def _draw_semantic_storyboard(
        canvas: Any,
        task: VideoTask,
        plan: VideoPlan,
        upstream: VideoArtifact,
        artifact_type: str,
    ) -> None:
        import cv2

        height, width = canvas.shape[:2]
        title = f"{artifact_type}: {plan.intent.get('subject', 'subject')}"
        lines = [title]
        clothing = plan.intent.get("clothing_color")
        if clothing:
            lines.append(f"appearance: {clothing} clothing")
        temporal = upstream.metadata.get("temporal_states") or plan.temporal_steps
        lines.extend(f"{index + 1}. {step}" for index, step in enumerate(temporal[:6]))
        y = max(36, height // 10)
        for line in lines:
            safe = str(line).encode("ascii", errors="replace").decode("ascii")[:90]
            cv2.putText(
                canvas,
                safe,
                (max(20, width // 20), y),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.5, min(width, height) / 900),
                (24, 24, 24),
                2,
                cv2.LINE_AA,
            )
            y += max(34, height // max(8, len(lines) + 2))

    def _normalize_image(
        self,
        task: VideoTask,
        upstream: VideoArtifact,
        context: ToolExecutionContext,
        target: dict[str, Any],
    ) -> VideoArtifact:
        source = upstream.metadata.get("reference_image")
        if not isinstance(source, str):
            sampled = upstream.metadata.get("sampled_frame_paths") or []
            source = str(sampled[0]) if sampled else None
        if not source or not Path(source).exists():
            raise RuntimeError("image bridge requires a materialized local reference_image")
        output = self._normalize_source_image(
            source, target, task.task_id, context.node_id, upstream.artifact_id
        )
        return self._image_artifact(task, upstream, context, target, str(output), "normalized_image")

    def _normalize_source_image(
        self,
        source: str,
        target: dict[str, Any],
        task_id: str,
        node_id: str,
        artifact_id: str,
    ) -> str:
        try:
            import cv2
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("opencv-python is required for image contract normalization") from exc
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read bridge image {source}")
        width = int(target.get("width") or image.shape[1])
        height = int(target.get("height") or image.shape[0])
        if (width, height) != (image.shape[1], image.shape[0]):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
        formats = target.get("formats") or ["png"]
        suffix = str(formats[0]).lower().lstrip(".")
        suffix = "jpg" if suffix == "jpeg" else suffix
        output = self.output_dir / f"{task_id}-{node_id}-{_stable_id(artifact_id, str(target))}.{suffix}"
        if not cv2.imwrite(str(output), image):
            raise RuntimeError(f"failed to write normalized bridge image {output}")
        return str(output)

    def _normalize_video(
        self,
        task: VideoTask,
        plan: VideoPlan,
        upstream: VideoArtifact,
        context: ToolExecutionContext,
        target: dict[str, Any],
    ) -> VideoArtifact:
        source = upstream.metadata.get("local_video_path")
        if not source or not Path(str(source)).exists():
            raise RuntimeError("video bridge requires a materialized local_video_path")
        ffmpeg = shutil.which(os.environ.get("EVOVIDEO_FFMPEG_BIN", "ffmpeg"))
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required for video contract normalization")
        formats = target.get("formats") or ["mp4"]
        suffix = str(formats[0]).lower().lstrip(".")
        output = self.output_dir / f"{task.task_id}-{context.node_id}-{_stable_id(upstream.artifact_id, str(target))}.{suffix}"
        command = [ffmpeg, "-y", "-i", str(source)]
        filters: list[str] = []
        if target.get("width") and target.get("height"):
            filters.append(f"scale={int(target['width'])}:{int(target['height'])}")
        if target.get("fps"):
            filters.append(f"fps={float(target['fps']):g}")
        if filters:
            command.extend(["-vf", ",".join(filters)])
        command.extend(["-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)])
        completed = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
        if completed.returncode != 0 or not output.exists():
            raise RuntimeError(f"video bridge ffmpeg failed: {(completed.stderr or completed.stdout)[-1000:]}")
        from evovideo_skill.video_processing import VideoProcessor

        processed = VideoProcessor(self.output_dir, sample_count=6).process(
            output.resolve().as_uri(), output.stem, task.prompt, {"generation_prompt": plan.generation_prompt}
        )
        metadata = deepcopy(upstream.metadata)
        metadata.update(
            {
                "artifact_type": "video",
                "local_video_path": processed.local_video_path,
                "sampled_frame_paths": processed.sampled_frame_paths,
                "artifact_contract": target,
                "bridge_operation": "normalized_video",
                "upstream_conditioning_consumed": True,
            }
        )
        return VideoArtifact(
            _stable_id(task.task_id, context.node_id, upstream.artifact_id), task.task_id, task.prompt,
            upstream.mode, list(upstream.tool_chain) + [self.name], processed.frames or deepcopy(upstream.frames), metadata,
        )

    def _image_artifact(
        self,
        task: VideoTask,
        upstream: VideoArtifact,
        context: ToolExecutionContext,
        target: dict[str, Any],
        path: str,
        operation: str,
    ) -> VideoArtifact:
        metadata = deepcopy(upstream.metadata)
        metadata.update(
            {
                "artifact_type": str(target.get("artifact_type") or "image"),
                "reference_image": str(path),
                "artifact_contract": target,
                "bridge_operation": operation,
                "upstream_conditioning_consumed": True,
            }
        )
        return VideoArtifact(
            _stable_id(task.task_id, context.node_id, upstream.artifact_id), task.task_id, task.prompt,
            upstream.mode, list(upstream.tool_chain) + [self.name], deepcopy(upstream.frames), metadata,
        )

    def _read_video_frame(self, video_path: str, role: str, task_id: str) -> str:
        try:
            import cv2
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("opencv-python is required for video frame extraction") from exc
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"failed to open bridge video {video_path}")
        if role == "last_frame":
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, count - 1))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"failed to extract {role} from {video_path}")
        output = self.output_dir / f"{task_id}-{role}-{_stable_id(video_path, role)}.png"
        if not cv2.imwrite(str(output), frame):
            raise RuntimeError(f"failed to write bridge frame {output}")
        return str(output)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, VideoTool] = {}
        self._specs: dict[str, Any] = {}

    def register(self, tool: VideoTool, spec: Any | None = None) -> None:
        self._tools[tool.name] = tool
        self._specs[tool.name] = spec or self._infer_spec(tool.name)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._specs.pop(name, None)

    def get(self, name: str) -> VideoTool:
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def available_names(self) -> set[str]:
        return {
            name
            for name, spec in self._specs.items()
            if name in self._tools and bool(getattr(spec, "verified", True))
        }

    def spec(self, name: str) -> Any:
        return self._specs[name]

    def manifests(self) -> list[dict[str, Any]]:
        return [self._specs[name].to_dict() for name in sorted(self.available_names())]

    def find_by_capability(self, capability: str) -> list[Any]:
        return [
            spec for name, spec in self._specs.items()
            if name in self.available_names() and getattr(spec, "capability", None) == capability
        ]

    def validate_connection(self, producer_name: str, consumer_name: str) -> tuple[bool, str]:
        check = self.connection_contract_check(producer_name, consumer_name)
        if check.compatible:
            return True, check.reason
        return False, check.reason

    def connection_contract_check(
        self,
        producer_name: str,
        consumer_name: str,
        produced_contract: dict[str, Any] | None = None,
    ):
        from evovideo_skill.artifact_contracts import ArtifactContract, check_contracts, input_contracts, output_contract

        producer = self.spec(producer_name)
        consumer = self.spec(consumer_name)
        consumer_tool = self.get(consumer_name)
        manifest = getattr(consumer_tool, "manifest", None)
        bindings = getattr(manifest, "input_bindings", {}) if manifest is not None else {}
        accepted = input_contracts(consumer, bindings)
        if not accepted:
            from evovideo_skill.artifact_contracts import ContractCheck

            return ContractCheck(False, f"{consumer_name} is a source tool and declares no upstream artifact inputs")
        produced = ArtifactContract.from_value(produced_contract) if produced_contract else output_contract(producer)
        return check_contracts(produced, accepted)

    @classmethod
    def suggested_spec(cls, name: str) -> Any:
        """Return the typed contract used when requesting an unavailable tool."""
        return cls._infer_spec(name)

    @staticmethod
    def _infer_spec(name: str) -> Any:
        from evovideo_skill.tool_onboarding import ToolSpec

        definitions: dict[str, dict[str, Any]] = {
            "task_reference_video": {"capability": "reference_video_source", "input_types": (), "output_type": "video", "output_bindings": ("reference_video", "reference_image"), "estimated_cost": 0.05},
            "mock_text_to_video": {"capability": "text_to_video", "input_types": ("temporal_plan",), "output_type": "video"},
            "mock_image_to_video": {"capability": "image_conditioned_video_generation", "input_types": ("identity_reference", "keyframes", "image", "video"), "output_type": "video", "consumes_upstream": True},
            "mock_multi_shot_i2v": {"capability": "multi_shot_identity_conditioned_generation", "input_types": ("character_sheet", "keyframes", "image", "video"), "output_type": "video", "consumes_upstream": True},
            "mock_video_style_transfer": {"capability": "video_style_transfer", "input_types": ("video", "structure_motion_map"), "output_type": "video", "consumes_upstream": True},
            "mock_global_video_editor": {"capability": "global_video_editing", "input_types": ("video",), "output_type": "video", "consumes_upstream": True},
            "mock_region_video_editor": {"capability": "region_video_editing", "input_types": ("video", "tracked_regions"), "output_type": "video", "consumes_upstream": True},
            "segment_repair": {"capability": "failed_segment_repair", "input_types": ("video",), "output_type": "video", "consumes_upstream": True, "estimated_cost": 1.5},
            "failed_segment_localizer": {"capability": "failure_segment_localization", "input_types": ("video",), "output_type": "segment_plan", "consumes_upstream": True, "estimated_cost": 0.15},
            "boundary_frame_extractor": {
                "capability": "boundary_frame_extraction",
                "input_types": ("video", "segment_plan"),
                "output_type": "image",
                "output_bindings": ("reference_image",),
                "output_contract": {
                    "artifact_type": "image",
                    "semantic_role": "first_frame",
                    "formats": ["png"],
                    "transport": ["local_path"],
                    "materialized": True,
                    "required_bindings": ["reference_image"],
                },
                "consumes_upstream": True,
                "estimated_cost": 0.15,
            },
            "segment_stitcher": {"capability": "healthy_content_preserving_composition", "input_types": ("video", "segment_plan"), "output_type": "video", "output_bindings": ("reference_video", "reference_image"), "consumes_upstream": True, "estimated_cost": 0.2},
            "extract_reference_identity_frame": {"capability": "identity_reference_extraction", "input_types": ("video",), "output_type": "identity_reference", "output_bindings": ("reference_image",), "consumes_upstream": True, "estimated_cost": 0.2},
            "scene_splitter": {"capability": "scene_splitting", "input_types": ("video", "any"), "output_type": "shot_plan", "estimated_cost": 0.2},
            "character_sheet_generator": {"capability": "character_sheet_generation", "input_types": ("shot_plan", "any"), "output_type": "character_sheet", "estimated_cost": 0.4},
            "temporal_decomposer": {"capability": "temporal_planning", "input_types": ("any",), "output_type": "temporal_plan", "estimated_cost": 0.2},
            "keyframe_generator": {"capability": "keyframe_generation", "input_types": ("temporal_plan", "any"), "output_type": "keyframes", "estimated_cost": 0.5},
            "structure_extractor": {"capability": "structure_motion_extraction", "input_types": ("video",), "output_type": "structure_motion_map", "consumes_upstream": True, "estimated_cost": 0.3},
            "temporal_deflicker": {"capability": "temporal_deflickering", "input_types": ("video",), "output_type": "video", "consumes_upstream": True, "estimated_cost": 0.3},
            "object_tracker": {"capability": "object_tracking", "input_types": ("video",), "output_type": "tracked_regions", "consumes_upstream": True, "estimated_cost": 0.2},
            "bridge_extract_reference_frame": {"capability": "artifact_contract_bridge", "input_types": ("video",), "output_type": "image", "output_bindings": ("reference_image",), "consumes_upstream": True, "estimated_cost": 0.05},
            "bridge_materialize_image": {"capability": "artifact_contract_bridge", "input_types": ("keyframes", "character_sheet", "identity_reference"), "output_type": "image", "output_bindings": ("reference_image",), "consumes_upstream": True, "estimated_cost": 0.08},
            "bridge_compose_reference_images": {"capability": "artifact_contract_bridge", "input_types": ("image", "identity_reference", "keyframes", "character_sheet"), "output_type": "image", "output_bindings": ("reference_image",), "consumes_upstream": True, "estimated_cost": 0.10},
            "bridge_normalize_image": {"capability": "artifact_contract_bridge", "input_types": ("image", "identity_reference"), "output_type": "image", "output_bindings": ("reference_image",), "consumes_upstream": True, "estimated_cost": 0.05},
            "bridge_normalize_video": {"capability": "artifact_contract_bridge", "input_types": ("video",), "output_type": "video", "output_bindings": ("reference_video", "reference_image"), "consumes_upstream": True, "estimated_cost": 0.15},
        }
        definition = definitions.get(name, {"capability": name, "input_types": ("any",), "output_type": "intermediate"})
        return ToolSpec(name=name, backend="builtin", description=f"Registered {name} tool", **definition)

    @classmethod
    def with_mock_tools(cls, bridge_output_dir: str | Path | None = None) -> "ToolRegistry":
        registry = cls()
        registry.register(TaskReferenceVideoTool(bridge_output_dir))
        registry.register(MockTextToVideoTool())
        registry.register(MockImageToVideoTool())
        registry.register(MockMultiShotImageToVideoTool())
        registry.register(MockVideoStyleTransferTool())
        registry.register(MockGlobalVideoEditor())
        registry.register(MockRegionVideoEditor())
        registry.register(MockSegmentRepairTool())
        registry.register(FailureSegmentLocalizerTool())
        registry.register(BoundaryFrameExtractorTool(bridge_output_dir))
        registry.register(SegmentStitcherTool(bridge_output_dir))
        for name in (
            "extract_reference_identity_frame",
            "scene_splitter",
            "character_sheet_generator",
            "temporal_decomposer",
            "keyframe_generator",
            "structure_extractor",
            "temporal_deflicker",
            "object_tracker",
        ):
            registry.register(ArtifactTransformTool(name))
        for name in (
            "bridge_extract_reference_frame",
            "bridge_materialize_image",
            "bridge_compose_reference_images",
            "bridge_normalize_image",
            "bridge_normalize_video",
        ):
            registry.register(ArtifactBridgeTool(name, bridge_output_dir))
        return registry

    @classmethod
    def with_artifact_tools(cls, bridge_output_dir: str | Path | None = None) -> "ToolRegistry":
        """Build a real-runtime registry with analysis/planning tools only.

        Pixel-changing tools must be backed by a real adapter.  In particular,
        the metadata-only mock deflicker must never be advertised as executable
        in a real Wan/API run.
        """
        registry = cls()
        registry.register(TaskReferenceVideoTool(bridge_output_dir))
        registry.register(FailureSegmentLocalizerTool())
        registry.register(BoundaryFrameExtractorTool(bridge_output_dir))
        registry.register(SegmentStitcherTool(bridge_output_dir))
        for name in (
            "extract_reference_identity_frame",
            "scene_splitter",
            "character_sheet_generator",
            "temporal_decomposer",
            "keyframe_generator",
            "structure_extractor",
            "object_tracker",
        ):
            registry.register(ArtifactTransformTool(name))
        for name in (
            "bridge_extract_reference_frame",
            "bridge_materialize_image",
            "bridge_compose_reference_images",
            "bridge_normalize_image",
            "bridge_normalize_video",
        ):
            registry.register(ArtifactBridgeTool(name, bridge_output_dir))
        return registry
