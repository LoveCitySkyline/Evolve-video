from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evovideo_skill.models import VideoTask


@dataclass
class BenchmarkSuite:
    name: str
    tasks: list[VideoTask]
    description: str = ""
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path, name: str | None = None, limit: int | None = None) -> "BenchmarkSuite":
        path = Path(path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            tasks_payload = payload.get("tasks", [])
            suite_name = name or payload.get("name") or path.stem
            description = payload.get("description", "")
            tags = list(payload.get("tags", []))
        else:
            tasks_payload = payload
            suite_name = name or path.stem
            description = ""
            tags = []
        tasks = [VideoTask.from_dict(item) for item in tasks_payload]
        cls._resolve_asset_manifest(path, tasks)
        cls._resolve_h3_references(path, tasks)
        if limit is not None:
            tasks = tasks[:limit]
        return cls(suite_name, tasks, description, tags)

    @staticmethod
    def _resolve_h3_references(benchmark_path: Path, tasks: list[VideoTask]) -> None:
        from evovideo_skill.h3_api import media_path

        for task in tasks:
            refs = task.metadata.get("h3_references")
            if refs is None:
                continue
            if not isinstance(refs, list) or not all(isinstance(ref, dict) for ref in refs):
                raise ValueError(f"{task.task_id}: h3_references must be a list of objects")
            for ref in refs:
                uri = ref.get("uri")
                if not isinstance(uri, str) or not uri:
                    raise ValueError(f"{task.task_id}: every H3 reference needs a uri")
                local = media_path(uri)
                if local is None:
                    continue
                if not Path(uri).expanduser().is_absolute() and not uri.startswith("file://"):
                    local = (benchmark_path.parent / uri).expanduser().resolve()
                ref["uri"] = str(local)
                if local.is_file():
                    digest = hashlib.sha256()
                    with local.open("rb") as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(block)
                    ref["content_sha256"] = digest.hexdigest()

    @staticmethod
    def _resolve_asset_manifest(benchmark_path: Path, tasks: list[VideoTask]) -> None:
        asset_path = benchmark_path.with_name(f"{benchmark_path.stem}_assets.json")
        if not asset_path.is_file():
            return
        with asset_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        entries = payload.get("assets", []) if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            return
        assets: dict[str, str | None] = {}
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("asset_id"):
                continue
            raw_path = entry.get("path")
            if raw_path:
                value = str(raw_path)
                if not value.startswith(("http://", "https://")):
                    candidate = Path(value).expanduser()
                    if not candidate.is_absolute():
                        candidate = asset_path.parent / candidate
                    value = str(candidate.resolve())
                assets[str(entry["asset_id"])] = value
            else:
                assets[str(entry["asset_id"])] = None

        for task in tasks:
            metadata = task.metadata or {}
            declared = metadata.get("input_assets") or {}
            if not isinstance(declared, dict):
                continue
            missing: list[str] = []
            audio_id = declared.get("audio")
            if audio_id:
                resolved = assets.get(str(audio_id))
                if resolved:
                    metadata["local_audio_path"] = resolved
                elif declared.get("required"):
                    missing.append(str(audio_id))
            source_id = declared.get("source_video")
            if source_id:
                resolved = assets.get(str(source_id))
                if resolved:
                    task.reference_video = resolved
                elif declared.get("required"):
                    missing.append(str(source_id))
            metadata["asset_manifest_path"] = str(asset_path)
            if missing:
                metadata["missing_required_assets"] = sorted(set(missing))
            else:
                metadata.pop("missing_required_assets", None)
            task.metadata = metadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "tasks": [asdict(task) for task in self.tasks],
        }


def load_suites(paths: list[str], limit_per_suite: int | None = None) -> list[BenchmarkSuite]:
    return [BenchmarkSuite.from_file(path, limit=limit_per_suite) for path in paths]
