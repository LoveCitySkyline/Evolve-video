from __future__ import annotations

import hashlib
import json
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from evovideo_skill.models import utc_now


SelectionStrategy = Literal["best", "random", "round_robin"]


@dataclass
class ProgramMetrics:
    quality: float = 0.0
    pass_rate: float = 0.0
    stability: float = 0.0
    estimated_cost: float = 0.0
    task_count: int = 0
    cache_hits: int = 0
    sample_count: int = 0
    execution_coverage: float = 1.0
    execution_error_count: int = 0
    candidate_tool_coverage: float = 1.0
    reward_objectives: list[str] = field(default_factory=list)
    reward_component_means: dict[str, float] = field(default_factory=dict)

    def utility(self, cost_weight: float = 0.0) -> float:
        return self.quality - max(0.0, cost_weight) * self.estimated_cost


@dataclass
class GraphProgram:
    name: str
    graph_ids: list[str]
    parent: str | None = None
    iteration: int = 0
    proposal: str = ""
    justification: str = ""
    mutation: list[dict[str, Any]] = field(default_factory=list)
    metrics: ProgramMetrics | None = None
    status: str = "candidate"
    created_at: str = field(default_factory=utc_now)

    def fingerprint(self) -> str:
        payload = {
            "graph_ids": sorted(self.graph_ids),
            "parent": self.parent,
            "mutation": self.mutation,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GraphProgram":
        data = dict(payload)
        if data.get("metrics") is not None:
            data["metrics"] = ProgramMetrics(**data["metrics"])
        return cls(**data)


class ProgramRegistry:
    """File-backed EvoSkill-style program versions and Pareto frontier."""

    def __init__(self, root: str | Path, cost_weight: float = 0.03):
        self.root = Path(root)
        self.program_dir = self.root / "programs"
        self.frontier_path = self.root / "frontier.json"
        self.current_path = self.root / "current.json"
        self.cost_weight = cost_weight
        self.program_dir.mkdir(parents=True, exist_ok=True)

    def create(self, program: GraphProgram) -> None:
        path = self._path(program.name)
        if path.exists():
            raise ValueError(f"program already exists: {program.name}")
        self._write_program(program)

    def upsert(self, program: GraphProgram) -> None:
        self._write_program(program)

    def get(self, name: str) -> GraphProgram:
        with self._path(name).open("r", encoding="utf-8") as handle:
            return GraphProgram.from_dict(json.load(handle))

    def list_programs(self) -> list[GraphProgram]:
        programs = []
        for path in sorted(self.program_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as handle:
                programs.append(GraphProgram.from_dict(json.load(handle)))
        return programs

    def switch_to(self, name: str) -> GraphProgram:
        program = self.get(name)
        self._write_json(self.current_path, {"program": name, "updated_at": utc_now()})
        return program

    def current(self) -> GraphProgram | None:
        if not self.current_path.exists():
            return None
        with self.current_path.open("r", encoding="utf-8") as handle:
            name = json.load(handle).get("program")
        return self.get(name) if name else None

    def frontier(self) -> list[GraphProgram]:
        if not self.frontier_path.exists():
            return []
        with self.frontier_path.open("r", encoding="utf-8") as handle:
            names = json.load(handle).get("programs", [])
        programs = [self.get(name) for name in names if self._path(name).exists()]
        return sorted(programs, key=self._rank_key, reverse=True)

    def update_frontier(self, name: str, max_size: int) -> bool:
        candidate = self.get(name)
        if candidate.metrics is None:
            raise ValueError(f"program has no metrics: {name}")
        pool = {program.name: program for program in self.frontier()}
        pool[name] = candidate
        nondominated = [
            program for program in pool.values()
            if not any(self._dominates(other, program) for other in pool.values() if other.name != program.name)
        ]
        nondominated.sort(key=self._rank_key, reverse=True)
        kept = nondominated[:max(1, max_size)]
        kept_names = {program.name for program in kept}
        for program in pool.values():
            program.status = "frontier" if program.name in kept_names else "archived"
            self.upsert(program)
        self._write_json(
            self.frontier_path,
            {"programs": [program.name for program in kept], "updated_at": utc_now()},
        )
        return name in kept_names

    def select(self, strategy: SelectionStrategy = "best", iteration: int = 0) -> GraphProgram | None:
        frontier = self.frontier()
        if not frontier:
            return None
        if strategy == "random":
            return random.choice(frontier)
        if strategy == "round_robin":
            return frontier[iteration % len(frontier)]
        return frontier[0]

    def best(self) -> GraphProgram | None:
        return self.select("best")

    def lineage(self, name: str) -> list[str]:
        lineage = []
        current: str | None = name
        visited: set[str] = set()
        while current and current not in visited:
            visited.add(current)
            lineage.append(current)
            current = self.get(current).parent
        return lineage

    def diff(self, left: str, right: str) -> dict[str, Any]:
        left_program = self.get(left)
        right_program = self.get(right)
        return {
            "left": left,
            "right": right,
            "added_graphs": sorted(set(right_program.graph_ids) - set(left_program.graph_ids)),
            "removed_graphs": sorted(set(left_program.graph_ids) - set(right_program.graph_ids)),
            "left_metrics": asdict(left_program.metrics) if left_program.metrics else None,
            "right_metrics": asdict(right_program.metrics) if right_program.metrics else None,
            "mutation": right_program.mutation,
        }

    def reset(self) -> None:
        if self.program_dir.exists():
            shutil.rmtree(self.program_dir)
        self.program_dir.mkdir(parents=True, exist_ok=True)
        for path in (self.frontier_path, self.current_path):
            if path.exists():
                path.unlink()

    def _path(self, name: str) -> Path:
        safe_name = name.replace("/", "_")
        return self.program_dir / f"{safe_name}.json"

    def _write_program(self, program: GraphProgram) -> None:
        self._write_json(self._path(program.name), program.to_dict())

    def _rank_key(self, program: GraphProgram) -> tuple[float, float, float, float, float]:
        metrics = program.metrics or ProgramMetrics()
        return (
            metrics.utility(self.cost_weight),
            metrics.quality,
            metrics.pass_rate,
            metrics.stability,
            -metrics.estimated_cost,
        )

    @staticmethod
    def _dominates(left: GraphProgram, right: GraphProgram) -> bool:
        if left.metrics is None or right.metrics is None:
            return False
        no_worse = (
            left.metrics.quality >= right.metrics.quality
            and left.metrics.pass_rate >= right.metrics.pass_rate
            and left.metrics.stability >= right.metrics.stability
            and left.metrics.estimated_cost <= right.metrics.estimated_cost
        )
        strictly_better = (
            left.metrics.quality > right.metrics.quality
            or left.metrics.pass_rate > right.metrics.pass_rate
            or left.metrics.stability > right.metrics.stability
            or left.metrics.estimated_cost < right.metrics.estimated_cost
        )
        return no_worse and strictly_better

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        temporary.replace(path)
