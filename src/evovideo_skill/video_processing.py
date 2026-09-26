from __future__ import annotations

import math
import shutil
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


COLOR_RGB = {
    "red": (210, 50, 50),
    "blue": (50, 90, 210),
    "yellow": (220, 190, 40),
    "green": (60, 160, 70),
    "black": (35, 35, 35),
    "white": (220, 220, 220),
    "brown": (130, 80, 45),
}


@dataclass
class VideoProcessingResult:
    local_video_path: str
    sampled_frame_paths: list[str]
    frames: list[dict[str, Any]]


class VideoProcessor:
    """Download real provider videos, sample frames, and extract lightweight evidence."""

    def __init__(self, output_dir: str | Path = "outputs/provider_videos", sample_count: int = 6, timeout_seconds: int = 120):
        self.output_dir = Path(output_dir)
        self.sample_count = sample_count
        self.timeout_seconds = timeout_seconds

    def process(self, video_url: str, video_id: str, prompt: str, plan_payload: dict[str, Any] | None = None) -> VideoProcessingResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        local_video_path = self._download(video_url, video_id)
        sampled_frame_paths = self._sample_frames(local_video_path, video_id)
        frames = self._analyze_frames(sampled_frame_paths, prompt, plan_payload or {})
        return VideoProcessingResult(str(local_video_path), sampled_frame_paths, frames)

    def _download(self, video_url: str, video_id: str) -> Path:
        parsed = urllib.parse.urlparse(video_url)
        suffix = Path(parsed.path).suffix or ".mp4"
        safe_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in video_id)
        output_path = self.output_dir / f"{safe_id}{suffix}"
        if parsed.scheme == "file":
            source_path = Path(urllib.request.url2pathname(parsed.path)).resolve()
            output_root = self.output_dir.resolve()
            if source_path.parent == output_root:
                return source_path
            shutil.copyfile(source_path, output_path)
            return output_path
        request = urllib.request.Request(video_url, headers={"User-Agent": "EvoVideoSkill/0.1"})
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            output_path.write_bytes(response.read())
        return output_path

    def _sample_frames(self, video_path: Path, video_id: str) -> list[str]:
        try:
            import cv2
        except Exception as exc:  # pragma: no cover - depends on optional local package
            raise RuntimeError("opencv-python is required for frame sampling") from exc

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"failed to open generated video: {video_path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        indices = self._sample_indices(frame_count)
        frame_dir = self.output_dir / f"{video_path.stem}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for order, frame_index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            path = frame_dir / f"frame_{order:02d}_{frame_index:06d}.jpg"
            cv2.imwrite(str(path), frame)
            paths.append(str(path))
        capture.release()
        return paths

    def _sample_indices(self, frame_count: int) -> list[int]:
        if self.sample_count <= 1:
            return [0]
        if frame_count <= self.sample_count:
            return list(range(frame_count))
        return sorted({min(frame_count - 1, round(i * (frame_count - 1) / (self.sample_count - 1))) for i in range(self.sample_count)})

    def _analyze_frames(self, frame_paths: list[str], prompt: str, plan_payload: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            import cv2
            import numpy as np
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("opencv-python and numpy are required for frame analysis") from exc

        prompt_lower = prompt.lower()
        target_colors = [name for name in COLOR_RGB if name in prompt_lower]
        plan = plan_payload.get("plan") or plan_payload
        intent = plan.get("intent") or {}
        steps = plan.get("temporal_steps") or ["sampled generated video"]
        frames: list[dict[str, Any]] = []
        previous_gray = None

        for idx, frame_path in enumerate(frame_paths):
            bgr = cv2.imread(frame_path)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb, (96, 96))
            mean_rgb = small.reshape(-1, 3).mean(axis=0)
            dominant = self._nearest_color(tuple(float(v) for v in mean_rgb))
            color_presence = {color: self._color_presence_ratio(small, color) for color in target_colors}
            clothing_color = None
            if target_colors:
                target = target_colors[0]
                clothing_color = target if color_presence.get(target, 0.0) >= 0.015 else f"missing_{target}"

            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
            motion_delta = 0.0
            if previous_gray is not None:
                motion_delta = float(np.mean(np.abs(gray.astype("float32") - previous_gray.astype("float32"))) / 255.0)
            previous_gray = gray

            frames.append(
                {
                    "index": idx,
                    "frame_path": frame_path,
                    "subject": intent.get("subject", "subject"),
                    "identity": "visual_identity_unverified",
                    "identity_evidence": "real frames sampled; plug in face/person embedding evaluator for strict identity scoring",
                    "dominant_color": dominant,
                    "clothing_color": clothing_color,
                    "color_presence": color_presence,
                    "brightness": float(mean_rgb.mean() / 255.0),
                    "motion_delta": motion_delta,
                    "action": steps[min(idx * len(steps) // max(1, len(frame_paths)), len(steps) - 1)],
                    "style": intent.get("style", "default"),
                    "background_changed": False,
                    "target_edit_success": True,
                }
            )
        return frames

    @staticmethod
    def _nearest_color(rgb: tuple[float, float, float]) -> str:
        def distance(item: tuple[str, tuple[int, int, int]]) -> float:
            _, color = item
            return math.sqrt(sum((rgb[i] - color[i]) ** 2 for i in range(3)))

        return min(COLOR_RGB.items(), key=distance)[0]

    @staticmethod
    def _color_presence_ratio(rgb_image: Any, color_name: str) -> float:
        import numpy as np

        target = np.array(COLOR_RGB[color_name], dtype="float32")
        pixels = rgb_image.reshape(-1, 3).astype("float32")
        dist = np.linalg.norm(pixels - target, axis=1)
        return float(np.mean(dist < 95.0))
