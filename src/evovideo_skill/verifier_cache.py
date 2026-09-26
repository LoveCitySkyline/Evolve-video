"""Content-addressed judge results, independent of the evolution run directory."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable


class VerifierCache:
    VERSION = "visual-evidence-v1"

    def __init__(self, root: str | Path | None, namespace: str = "default"):
        self.root = Path(root).expanduser() if root else None
        self.namespace = namespace

    @staticmethod
    def file_hash(value: str) -> str | None:
        parsed = urllib.parse.urlparse(value)
        if parsed.scheme in {"http", "https"}:
            return None
        path = Path(urllib.request.url2pathname(parsed.path) if parsed.scheme == "file" else value)
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def evaluate(self, endpoint: str, payload: dict[str, Any], media: list[str],
                 evaluate: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        hashes = [self.file_hash(path) for path in media if path]
        request_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        descriptor = {"version": self.VERSION, "namespace": self.namespace,
                      "endpoint": endpoint, "request_sha256": request_hash, "media_sha256": hashes}
        key = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
        images = [item["image_url"]["url"] for message in payload.get("messages", [])
                  for item in message.get("content", []) if isinstance(item, dict) and item.get("type") == "image_url"]
        # Mutable remote media and missing files cannot establish content identity.
        enabled = self.root is not None and all(h is not None for h in hashes) and all(
            url.startswith("data:") for url in images
        )

        def call() -> dict[str, Any]:
            result = evaluate()
            result["verifier_cache"] = {**descriptor, "key": key, "hit": False, "enabled": enabled}
            return result

        if not enabled:
            return call()
        from evovideo_skill.h3_api import portable_interprocess_lock

        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{key}.json"
        with portable_interprocess_lock(self.root / f"{key}.lock", 3600):
            try:
                stored = json.loads(path.read_text())
                if stored.get("verifier_cache", {}).get("key") == key:
                    stored["verifier_cache"]["hit"] = True
                    return stored
            except (OSError, ValueError, AttributeError):
                pass
            result = call()
            fd, temporary = tempfile.mkstemp(prefix=key, suffix=".tmp", dir=self.root)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(result, handle, ensure_ascii=False, allow_nan=False)
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
            return result
