from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from evovideo_skill.tool_onboarding import (
    CapabilityRequest,
    CommandToolManifest,
    OnboardingResult,
    ToolOnboardingError,
)


@dataclass(frozen=True)
class PreparedCapability:
    request: CapabilityRequest
    required: bool = True


class PreparedToolStore:
    """Portable index for repository tools built before graph evolution."""

    SCHEMA_VERSION = 1

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.catalog_path = self.root / "tool_catalog.json"
        self.report_path = self.root / "prepare_report.json"
        self.acquisition_state_dir = self.root / "acquisition_state"

    @staticmethod
    def load_capabilities(path: str | Path) -> list[PreparedCapability]:
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        entries = payload.get("capabilities", payload) if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            raise ToolOnboardingError(
                "prepared capability file must be a list or a {'capabilities': [...]} object"
            )
        capabilities: list[PreparedCapability] = []
        for index, raw in enumerate(entries):
            if not isinstance(raw, dict) or not str(raw.get("capability", "")).strip():
                raise ToolOnboardingError(f"capability entry {index} has no capability")
            request = CapabilityRequest(
                capability=str(raw["capability"]),
                preferred_backend=raw.get("preferred_backend"),
                suggested_tool_name=raw.get("suggested_tool_name"),
                reason=str(raw.get("reason", "")),
                required_input_types=[str(item) for item in raw.get("required_input_types", [])],
            )
            capabilities.append(PreparedCapability(request, bool(raw.get("required", True))))
        return capabilities

    def publish(
        self,
        manifests: Iterable[CommandToolManifest],
        capabilities: list[PreparedCapability],
        results: list[OnboardingResult],
        executable_tool_names: set[str],
    ) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        unique: dict[str, CommandToolManifest] = {}
        for manifest in manifests:
            if manifest.spec.verified and manifest.spec.name in executable_tool_names:
                unique[manifest.spec.name] = manifest
        catalog = {
            "schema_version": self.SCHEMA_VERSION,
            "generated_by": "evovideo_skill prepare-tools",
            "tools": [unique[name].to_dict() for name in sorted(unique)],
        }
        result_by_key = {
            self._request_key(result.request): result
            for result in results
        }
        capability_rows: list[dict[str, Any]] = []
        required_failures: list[str] = []
        for item in capabilities:
            result = result_by_key.get(self._request_key(item.request))
            status = result.status if result is not None else "missing"
            satisfied = status in {"registered", "already_registered"}
            if item.required and not satisfied:
                required_failures.append(item.request.capability)
            capability_rows.append(
                {
                    "request": asdict(item.request),
                    "required": item.required,
                    "satisfied": satisfied,
                    "status": status,
                    "tool_name": result.tool_name if result is not None else None,
                    "evidence": list(result.evidence) if result is not None else [],
                }
            )
        report = {
            "schema_version": self.SCHEMA_VERSION,
            "catalog_path": str(self.catalog_path),
            "prepared_tool_count": len(unique),
            "prepared_tools": sorted(unique),
            "required_failures": sorted(set(required_failures)),
            "capabilities": capability_rows,
        }
        self._atomic_write(self.catalog_path, catalog)
        self._atomic_write(self.report_path, report)
        return report

    @staticmethod
    def _request_key(request: CapabilityRequest) -> tuple[str, str | None, str | None, tuple[str, ...]]:
        return (
            request.capability,
            request.preferred_backend,
            request.suggested_tool_name,
            tuple(sorted(request.required_input_types)),
        )

    @staticmethod
    def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
