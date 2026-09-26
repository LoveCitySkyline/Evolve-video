from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class TaskMode(str, Enum):
    GENERATION = "generation"
    EDITING = "editing"


class FailureType(str, Enum):
    IDENTITY_DRIFT = "identity_drift"
    CLOTHING_COLOR_DRIFT = "clothing_color_drift"
    OBJECT_PERSISTENCE = "object_persistence_failure"
    EDITING_LEAKAGE = "editing_leakage"
    MOTION_MISMATCH = "motion_mismatch"
    TEMPORAL_FLICKER = "temporal_flicker"
    PROMPT_OMISSION = "prompt_omission"
    STYLE_DRIFT = "style_drift"


@dataclass
class VideoTask:
    task_id: str
    prompt: str
    mode: TaskMode = TaskMode.GENERATION
    duration_seconds: int = 6
    reference_video: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mode = TaskMode(self.mode)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoTask":
        payload = dict(data)
        payload["mode"] = TaskMode(payload.get("mode", TaskMode.GENERATION))
        return cls(**payload)


@dataclass
class VideoPlan:
    task_id: str
    intent: dict[str, Any]
    temporal_steps: list[str]
    constraints: list[str]
    selected_skill_names: list[str]
    tool_chain: list[str]
    generation_prompt: str
    prompt_rewrite_reasons: list[str] = field(default_factory=list)


@dataclass
class VideoArtifact:
    artifact_id: str
    task_id: str
    prompt: str
    mode: TaskMode
    tool_chain: list[str]
    frames: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mode = TaskMode(self.mode)

    def frame_values(self, key: str) -> list[Any]:
        return [frame.get(key) for frame in self.frames]


@dataclass
class EvaluationMetric:
    name: str
    score: float
    threshold: float
    passed: bool
    evidence: list[str] = field(default_factory=list)
    applicable: bool = True


@dataclass
class EvaluationReport:
    artifact_id: str
    task_id: str
    metrics: list[EvaluationMetric]

    @property
    def passed(self) -> bool:
        return bool(self.active_metrics) and all(metric.passed for metric in self.active_metrics)

    @property
    def active_metrics(self) -> list[EvaluationMetric]:
        return [metric for metric in self.metrics if metric.applicable]

    @property
    def score(self) -> float:
        active = self.active_metrics
        return sum(metric.score for metric in active) / len(active) if active else 0.0

    @property
    def failed_metrics(self) -> list[EvaluationMetric]:
        return [metric for metric in self.active_metrics if not metric.passed]


@dataclass
class FailureReport:
    task_id: str
    artifact_id: str
    failure_types: list[FailureType]
    evidence: list[str]
    likely_causes: list[str]
    recommended_updates: list[str]
    expected_failure_types: list[FailureType] = field(default_factory=list)
    failed_segments: list[dict[str, Any]] = field(default_factory=list)
    intervention: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillCandidate:
    skill: "SkillCard"
    parent_failure: FailureReport
    proposal_reason: str


@dataclass
class SkillValidationReport:
    skill_name: str
    validation_task_ids: list[str]
    baseline_score: float
    candidate_score: float
    baseline_pass_rate: float
    candidate_pass_rate: float
    quality_gain: float
    estimated_cost: float
    accepted: bool
    evidence: list[str] = field(default_factory=list)


@dataclass
class EvoSkillSelectionReport:
    candidates: list[SkillValidationReport]
    selected_skill_names: list[str]


@dataclass
class SkillStats:
    uses: int = 0
    successes: int = 0
    failures: int = 0
    avg_score_delta: float = 0.0

    def record(self, success: bool, score_delta: float = 0.0) -> None:
        self.uses += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1
        previous = self.avg_score_delta * (self.uses - 1)
        self.avg_score_delta = (previous + score_delta) / self.uses


@dataclass
class SkillCard:
    skill_name: str
    version: str
    description: str
    triggers: list[str]
    failure_conditions: list[str]
    inputs: list[str]
    procedure: list[str]
    tools: list[str]
    evaluators: list[str]
    fallbacks: list[str]
    anti_patterns: list[str]
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    stats: SkillStats = field(default_factory=SkillStats)
    memory: dict[str, list[str]] = field(default_factory=lambda: {"success_cases": [], "failure_cases": []})
    validation: SkillValidationReport | None = None

    def matches(self, task: VideoTask, failure_types: list[FailureType] | None = None) -> bool:
        text = f"{task.prompt} {TaskMode(task.mode).value}".lower()
        trigger_hit = any(trigger.lower() in text for trigger in self.triggers)
        if failure_types:
            failure_hit = any(ft.value in self.failure_conditions for ft in failure_types)
            return trigger_hit or failure_hit
        return trigger_hit

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillCard":
        payload = dict(data)
        payload["stats"] = SkillStats(**payload.get("stats", {}))
        if payload.get("validation"):
            payload["validation"] = SkillValidationReport(**payload["validation"])
        return cls(**payload)
