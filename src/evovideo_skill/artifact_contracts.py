from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


IMAGE_TYPES = {"image", "identity_reference"}
SYMBOLIC_IMAGE_TYPES = {"keyframes", "character_sheet"}


@dataclass(frozen=True)
class ArtifactContract:
    artifact_type: str
    semantic_role: str = "any"
    formats: tuple[str, ...] = ()
    transport: tuple[str, ...] = ()
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    num_frames: int | None = None
    materialized: bool | None = None
    required_bindings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: ArtifactContract | dict[str, Any] | str) -> ArtifactContract:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value)
        payload = dict(value)
        payload["artifact_type"] = str(payload.get("artifact_type") or payload.get("type") or "any")
        payload.pop("type", None)
        for key in ("formats", "transport", "required_bindings"):
            raw = payload.get(key, ())
            payload[key] = (raw,) if isinstance(raw, str) else tuple(raw or ())
        for key in ("width", "height", "num_frames"):
            if payload.get(key) is not None and payload.get(key) != "":
                payload[key] = int(payload[key])
        if payload.get("fps") is not None and payload.get("fps") != "":
            payload["fps"] = float(payload["fps"])
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: item for key, item in payload.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("formats", "transport", "required_bindings"):
            payload[key] = list(payload[key])
        return payload


@dataclass(frozen=True)
class ContractCheck:
    compatible: bool
    reason: str
    target: ArtifactContract | None = None
    bridges: tuple[tuple[str, dict[str, Any]], ...] = ()


def default_output_contract(spec: Any) -> ArtifactContract:
    artifact_type = str(getattr(spec, "output_type", "intermediate"))
    bindings = tuple(getattr(spec, "output_bindings", ()))
    materialized = any(item in {"reference_image", "reference_video"} for item in bindings) if bindings else None
    transport = ("local_path",) if materialized else ("memory",)
    formats: tuple[str, ...] = ()
    if artifact_type == "video" and materialized:
        formats = ("mp4",)
    elif artifact_type in IMAGE_TYPES and materialized:
        formats = ("png", "jpg", "jpeg")
    return ArtifactContract(
        artifact_type=artifact_type,
        semantic_role="identity_reference" if artifact_type == "identity_reference" else "any",
        formats=formats,
        transport=transport,
        materialized=materialized,
        required_bindings=bindings,
    )


def output_contract(spec: Any) -> ArtifactContract:
    raw = getattr(spec, "output_contract", None)
    return ArtifactContract.from_value(raw) if raw else default_output_contract(spec)


def input_contracts(spec: Any, input_bindings: dict[str, str] | None = None) -> list[ArtifactContract]:
    raw_contracts = list(getattr(spec, "input_contracts", ()) or ())
    if raw_contracts:
        return [ArtifactContract.from_value(item) for item in raw_contracts]
    bindings = input_bindings or {}
    contracts: list[ArtifactContract] = []
    for artifact_type in tuple(getattr(spec, "input_types", ())):
        binding = bindings.get(artifact_type)
        physical = binding in {"reference_image", "reference_video"}
        contracts.append(
            ArtifactContract(
                artifact_type=artifact_type,
                transport=("local_path",) if physical else (),
                materialized=True if physical else None,
                required_bindings=(binding,) if binding else (),
            )
        )
    return contracts


def check_contracts(produced: ArtifactContract, accepted: list[ArtifactContract]) -> ContractCheck:
    reasons: list[str] = []
    bridge_candidates: list[tuple[ArtifactContract, str, tuple[tuple[str, dict[str, Any]], ...]]] = []
    for target in accepted:
        direct, reason = _directly_compatible(produced, target)
        if direct:
            return ContractCheck(True, reason, target)
        bridges = _bridge_plan(produced, target)
        if bridges:
            bridge_candidates.append((target, reason, bridges))
        reasons.append(reason)
    if bridge_candidates:
        target, reason, bridges = bridge_candidates[0]
        return ContractCheck(False, reason, target, bridges)
    return ContractCheck(False, "; ".join(reasons) or "consumer declares no input artifact contract")


def _directly_compatible(produced: ArtifactContract, target: ArtifactContract) -> tuple[bool, str]:
    source_type = produced.artifact_type
    target_type = target.artifact_type
    type_ok = target_type in {"any", source_type} or (
        source_type in IMAGE_TYPES and target_type == "image" and produced.materialized is not False
    )
    if not type_ok:
        return False, f"artifact type {source_type} does not satisfy {target_type}"
    role_ok = (
        target.semantic_role == "any"
        or produced.semantic_role == target.semantic_role
        or (source_type == "video" and target.semantic_role == "source_video")
    )
    if not role_ok:
        return False, f"semantic role {produced.semantic_role} does not satisfy {target.semantic_role}"
    if target.materialized is True and produced.materialized is not True:
        return False, f"{source_type} is symbolic but consumer requires a materialized artifact"
    if target.transport and produced.transport and not set(target.transport).intersection(produced.transport):
        return False, f"transport {produced.transport} does not satisfy {target.transport}"
    if target.formats and produced.formats and not set(target.formats).intersection(produced.formats):
        return False, f"format {produced.formats} does not satisfy {target.formats}"
    if target.width and produced.width and target.width != produced.width:
        return False, f"width {produced.width} does not satisfy {target.width}"
    if target.height and produced.height and target.height != produced.height:
        return False, f"height {produced.height} does not satisfy {target.height}"
    if target.fps and produced.fps and abs(target.fps - produced.fps) > 0.01:
        return False, f"fps {produced.fps} does not satisfy {target.fps}"
    return True, f"{source_type} satisfies the consumer artifact contract"


def _bridge_plan(
    produced: ArtifactContract,
    target: ArtifactContract,
) -> tuple[tuple[str, dict[str, Any]], ...]:
    config = {"target_contract": target.to_dict()}
    if produced.artifact_type == "video" and target.artifact_type in IMAGE_TYPES | {"image"}:
        return (("bridge_extract_reference_frame", config),)
    if (
        produced.artifact_type in IMAGE_TYPES
        and target.artifact_type in IMAGE_TYPES | {"image"}
        and produced.materialized is True
        and target.semantic_role in {"any", produced.semantic_role}
    ):
        return (("bridge_normalize_image", config),)
    if (
        produced.artifact_type in SYMBOLIC_IMAGE_TYPES | IMAGE_TYPES
        and target.artifact_type in SYMBOLIC_IMAGE_TYPES | IMAGE_TYPES | {"image"}
        and produced.materialized is not True
    ):
        return (("bridge_materialize_image", config),)
    if produced.artifact_type == "video" and target.artifact_type == "video":
        return (("bridge_normalize_video", config),)
    return ()
