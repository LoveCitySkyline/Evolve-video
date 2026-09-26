from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import struct
import urllib.parse
import urllib.request
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evovideo_skill.models import TaskMode, VideoTask, utc_now
from evovideo_skill.planning import Planner
from evovideo_skill.tools import ToolRegistry


@dataclass
class AssetBootstrapRecord:
    asset_id: str
    asset_type: str
    status: str
    task_id: str | None = None
    path: str | None = None
    prompt: str | None = None
    seed: int | None = None
    generator: str | None = None
    sha256: str | None = None
    error: str | None = None


@dataclass
class AssetBootstrapReport:
    enabled: bool
    records: list[AssetBootstrapRecord] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    @property
    def generated_count(self) -> int:
        return sum(record.status == "generated" for record in self.records)

    @property
    def reused_count(self) -> int:
        return sum(record.status == "reused" for record in self.records)

    @property
    def failed_count(self) -> int:
        return sum(record.status == "failed" for record in self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "generated_count": self.generated_count,
            "reused_count": self.reused_count,
            "failed_count": self.failed_count,
            "records": [asdict(record) for record in self.records],
            "created_at": self.created_at,
        }


class BenchmarkAssetBootstrapper:
    """Materialize missing benchmark inputs once and persist their manifest paths."""

    def __init__(
        self,
        registry: ToolRegistry,
        output_dir: str | Path | None = None,
        *,
        seed: int = 20260827,
        force: bool = False,
        strict: bool = True,
    ):
        self.registry = registry
        self.output_dir = Path(output_dir).expanduser().resolve() if output_dir else None
        self.seed = seed
        self.force = force
        self.strict = strict
        self.planner = Planner()

    def run(self, tasks: list[VideoTask]) -> AssetBootstrapReport:
        report = AssetBootstrapReport(enabled=True)
        manifests = self._manifest_tasks(tasks)
        for manifest_path, manifest_tasks in manifests.items():
            self._bootstrap_manifest(manifest_path, manifest_tasks, report)
        if report.failed_count and self.strict:
            failures = "; ".join(
                f"{record.asset_id}: {record.error}" for record in report.records if record.status == "failed"
            )
            raise RuntimeError(f"benchmark asset bootstrap failed: {failures}")
        return report

    @staticmethod
    def _manifest_tasks(tasks: list[VideoTask]) -> dict[Path, list[VideoTask]]:
        grouped: dict[Path, list[VideoTask]] = {}
        for task in tasks:
            value = (task.metadata or {}).get("asset_manifest_path")
            if value:
                grouped.setdefault(Path(value).expanduser().resolve(), []).append(task)
        return grouped

    def _bootstrap_manifest(
        self,
        manifest_path: Path,
        tasks: list[VideoTask],
        report: AssetBootstrapReport,
    ) -> None:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = payload.get("assets", []) if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            raise ValueError(f"asset manifest must contain a list: {manifest_path}")
        task_by_asset = self._tasks_by_asset(tasks)
        default_dir = self.output_dir or manifest_path.parent / "assets" / self._suite_slug(manifest_path)

        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("asset_id"):
                continue
            asset_id = str(entry["asset_id"])
            task = task_by_asset.get(asset_id)
            if task is None:
                continue
            asset_type = str(entry.get("type") or self._declared_asset_type(task, asset_id))
            existing = self._resolve_manifest_path(manifest_path, entry.get("path"))
            if existing and self._asset_available(existing) and not self.force:
                self._apply_to_tasks(tasks, asset_id, asset_type, existing)
                report.records.append(
                    AssetBootstrapRecord(asset_id, asset_type, "reused", task.task_id, existing)
                )
                continue

            extension = ".wav" if asset_type == "audio" else ".mp4"
            target = self._local_target(existing, default_dir / f"{asset_id}{extension}")
            prompt = None
            asset_seed = self._asset_seed(asset_id)
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                if asset_type == "audio":
                    prompt = task.prompt
                    self._synthesize_audio(task, target, asset_seed)
                    generator = "evovideo-deterministic-audio-bootstrap"
                elif asset_type in {"source_video", "video"}:
                    prompt = self._source_video_prompt(task)
                    generator = self._generate_source_video(task, prompt, target, asset_seed)
                else:
                    raise ValueError(f"unsupported asset type {asset_type!r}")
                digest = self._sha256(target)
                entry["path"] = self._portable_manifest_path(manifest_path, target)
                entry["bootstrap"] = {
                    "generator": generator,
                    "task_id": task.task_id,
                    "prompt": prompt,
                    "seed": asset_seed,
                    "sha256": digest,
                    "created_at": utc_now(),
                }
                self._atomic_write_json(manifest_path, payload)
                resolved = str(target.resolve())
                self._apply_to_tasks(tasks, asset_id, asset_type, resolved)
                report.records.append(
                    AssetBootstrapRecord(
                        asset_id,
                        asset_type,
                        "generated",
                        task.task_id,
                        resolved,
                        prompt,
                        asset_seed,
                        generator,
                        digest,
                    )
                )
                print(f"[asset bootstrap] generated {asset_id} -> {resolved}", flush=True)
            except Exception as exc:
                report.records.append(
                    AssetBootstrapRecord(
                        asset_id,
                        asset_type,
                        "failed",
                        task.task_id,
                        str(target),
                        prompt,
                        asset_seed,
                        error=str(exc),
                    )
                )
                print(f"[asset bootstrap] failed {asset_id}: {exc}", flush=True)

    @staticmethod
    def _tasks_by_asset(tasks: list[VideoTask]) -> dict[str, VideoTask]:
        result: dict[str, VideoTask] = {}
        for task in tasks:
            declared = (task.metadata or {}).get("input_assets") or {}
            if not isinstance(declared, dict):
                continue
            for key in ("audio", "source_video"):
                if declared.get(key):
                    result.setdefault(str(declared[key]), task)
        return result

    @staticmethod
    def _declared_asset_type(task: VideoTask, asset_id: str) -> str:
        declared = (task.metadata or {}).get("input_assets") or {}
        return "audio" if str(declared.get("audio")) == asset_id else "source_video"

    @staticmethod
    def _suite_slug(manifest_path: Path) -> str:
        stem = manifest_path.stem.removesuffix("_assets")
        return stem.removeprefix("complex_video_bench_") or stem

    @staticmethod
    def _resolve_manifest_path(manifest_path: Path, raw_path: Any) -> str | None:
        if not raw_path:
            return None
        value = str(raw_path)
        if value.startswith(("http://", "https://")):
            return value
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = manifest_path.parent / candidate
        return str(candidate.resolve())

    @staticmethod
    def _asset_available(value: str) -> bool:
        if value.startswith(("http://", "https://")):
            return True
        return Path(value).is_file() and Path(value).stat().st_size > 0

    @staticmethod
    def _local_target(existing: str | None, fallback: Path) -> Path:
        if existing and not existing.startswith(("http://", "https://")):
            return Path(existing)
        return fallback.resolve()

    @staticmethod
    def _portable_manifest_path(manifest_path: Path, target: Path) -> str:
        try:
            return str(target.resolve().relative_to(manifest_path.parent.resolve()))
        except ValueError:
            return str(target.resolve())

    def _asset_seed(self, asset_id: str) -> int:
        digest = hashlib.sha256(f"{self.seed}:{asset_id}".encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF

    def _generate_source_video(
        self,
        source_task: VideoTask,
        prompt: str,
        target: Path,
        seed: int,
    ) -> str:
        specs = self.registry.find_by_capability("text_to_video")
        if not specs:
            raise RuntimeError("no verified text_to_video tool is registered")
        spec = specs[0]
        tool = self.registry.get(spec.name)
        task = VideoTask(
            task_id=f"asset-bootstrap-{self._safe_id(target.stem)}",
            prompt=prompt,
            mode=TaskMode.GENERATION,
            duration_seconds=source_task.duration_seconds,
            metadata={"generation_seed": seed, "benchmark_asset_bootstrap": True},
        )
        artifact = tool.run(task, self.planner.plan(task, []))
        location = artifact.metadata.get("local_video_path") or artifact.metadata.get("video_url")
        if not location:
            raise RuntimeError(
                f"tool {spec.name!r} returned no materialized video path; "
                "a mock-only runtime cannot bootstrap benchmark video assets"
            )
        self._materialize_location(str(location), target)
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f"generated source video is empty: {target}")
        return f"{getattr(spec, 'backend', 'runtime')}:{spec.name}"

    @staticmethod
    def _materialize_location(location: str, target: Path) -> None:
        parsed = urllib.parse.urlparse(location)
        if parsed.scheme in {"http", "https"}:
            with urllib.request.urlopen(location, timeout=300) as response, target.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            return
        source = Path(urllib.request.url2pathname(parsed.path)) if parsed.scheme == "file" else Path(location)
        source = source.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"generated video path does not exist: {source}")
        if source != target.resolve():
            shutil.copy2(source, target)

    @staticmethod
    def _source_video_prompt(task: VideoTask) -> str:
        metadata = task.metadata or {}
        family = metadata.get("task_family") or metadata.get("category")
        if family == "compositional_editing":
            location_match = re.search(r"source video in the (.*?):", task.prompt, flags=re.IGNORECASE)
            location = location_match.group(1) if location_match else "described scene"
            original_objects = [
                str(operation.get("target"))
                for operation in metadata.get("edit_operations", [])
                if isinstance(operation, dict) and operation.get("target")
            ]
            objects = ", and ".join(original_objects) or "all original target objects"
            return (
                f"An unedited realistic source video in the {location}. Clearly and continuously show {objects}. "
                "Include natural subject and camera motion, realistic shadows and reflections, and brief object "
                "occlusion followed by recovery. Keep all original objects present; do not replace or remove them."
            )
        if family == "video_stylization":
            subject_match = re.search(r"provided (.*?)\.", task.prompt, flags=re.IGNORECASE)
            subject = subject_match.group(1) if subject_match else "scene described by the task"
            return (
                f"A realistic live-action source video of the {subject}. Preserve clear faces, body poses, object "
                "layout, action timing, and camera motion. Photorealistic untreated footage, no animation, no "
                "illustration, no charcoal, ink-wash, or stylized rendering."
            )
        raise ValueError(f"automatic source-video prompt is unsupported for task family {family!r}")

    @staticmethod
    def _synthesize_audio(task: VideoTask, target: Path, seed: int) -> None:
        sample_rate = 48_000
        duration = max(1.0, float(task.duration_seconds))
        frame_count = int(sample_rate * duration)
        rng = random.Random(seed)
        impact_times = [duration * fraction for fraction in (0.22, 0.52, 0.8)]
        samples = bytearray()
        for index in range(frame_count):
            time_s = index / sample_rate
            value = 350.0 * math.sin(2.0 * math.pi * 95.0 * time_s)
            value += rng.uniform(-240.0, 240.0)
            for impact_index, impact_time in enumerate(impact_times):
                delta = time_s - impact_time
                if 0.0 <= delta < 0.32:
                    decay = math.exp(-18.0 * delta)
                    frequency = 90.0 + 55.0 * impact_index
                    value += 23_000.0 * decay * math.sin(2.0 * math.pi * frequency * delta)
                    value += 7_000.0 * decay * rng.uniform(-1.0, 1.0)
            integer = max(-32_768, min(32_767, int(value)))
            samples.extend(struct.pack("<h", integer))
        with wave.open(str(target), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(bytes(samples))

    @staticmethod
    def _apply_to_tasks(tasks: list[VideoTask], asset_id: str, asset_type: str, path: str) -> None:
        for task in tasks:
            metadata = task.metadata or {}
            declared = metadata.get("input_assets") or {}
            if not isinstance(declared, dict):
                continue
            matched = False
            if asset_type == "audio" and str(declared.get("audio")) == asset_id:
                metadata["local_audio_path"] = path
                matched = True
            if asset_type in {"source_video", "video"} and str(declared.get("source_video")) == asset_id:
                task.reference_video = path
                matched = True
            if matched:
                missing = [item for item in metadata.get("missing_required_assets", []) if item != asset_id]
                if missing:
                    metadata["missing_required_assets"] = missing
                else:
                    metadata.pop("missing_required_assets", None)
                task.metadata = metadata

    @staticmethod
    def _safe_id(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-") or "asset"

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _atomic_write_json(path: Path, payload: Any) -> None:
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + os.linesep, encoding="utf-8")
        temporary.replace(path)
