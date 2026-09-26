from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvolutionCacheConfig:
    cache_dir: str | Path
    enabled: bool = True


class EvolutionRunCache:
    """Content-addressed cache for expensive graph/task validation rollouts."""

    def __init__(self, config: EvolutionCacheConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        if config.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(namespace: str, payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"{namespace}-{digest}"

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.config.enabled:
            return None
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, key: str, payload: dict[str, Any]) -> None:
        if not self.config.enabled:
            return
        path = self.cache_dir / f"{key}.json"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        temporary.replace(path)

    def delete(self, key: str) -> None:
        if not self.config.enabled:
            return
        path = self.cache_dir / f"{key}.json"
        try:
            path.unlink()
        except FileNotFoundError:
            pass
