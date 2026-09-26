from __future__ import annotations

import json
from pathlib import Path

from evovideo_skill.models import FailureType, SkillCard, VideoTask, utc_now


class SkillMemory:
    """Persistent skill library stored as JSON skill cards."""

    def __init__(self, memory_dir: str | Path):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, skill_name: str) -> Path:
        safe = skill_name.replace("/", "_")
        return self.memory_dir / f"{safe}.json"

    def list_skills(self) -> list[SkillCard]:
        cards: list[SkillCard] = []
        for path in sorted(self.memory_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as handle:
                cards.append(SkillCard.from_dict(json.load(handle)))
        return cards

    def get(self, skill_name: str) -> SkillCard | None:
        path = self._path_for(skill_name)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return SkillCard.from_dict(json.load(handle))

    def upsert(self, skill: SkillCard) -> None:
        skill.updated_at = utc_now()
        path = self._path_for(skill.skill_name)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(skill.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        self.materialize_folder(skill)

    def materialize_folder(self, skill: SkillCard) -> None:
        """Materialize an EvoSkill-style structured reusable skill folder."""

        skill_dir = self.memory_dir / "skills" / skill.skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        with (skill_dir / "skill.json").open("w", encoding="utf-8") as handle:
            json.dump(skill.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        with (skill_dir / "SKILL.md").open("w", encoding="utf-8") as handle:
            handle.write(self._skill_markdown(skill))
        with (skill_dir / "examples.json").open("w", encoding="utf-8") as handle:
            json.dump(skill.memory, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        with (skill_dir / "tools.py").open("w", encoding="utf-8") as handle:
            handle.write(self._tools_stub(skill))
        with (skill_dir / "evaluators.py").open("w", encoding="utf-8") as handle:
            handle.write(self._evaluators_stub(skill))

    def retrieve(self, task: VideoTask, failure_types: list[FailureType] | None = None) -> list[SkillCard]:
        matches = [skill for skill in self.list_skills() if skill.matches(task, failure_types)]
        return sorted(matches, key=lambda skill: (skill.stats.successes, skill.stats.uses), reverse=True)

    def record_case(self, skill_name: str, task_id: str, success: bool, note: str) -> None:
        skill = self.get(skill_name)
        if skill is None:
            return
        key = "success_cases" if success else "failure_cases"
        skill.memory.setdefault(key, []).append(f"{task_id}: {note}")
        skill.stats.record(success=success)
        self.upsert(skill)

    @staticmethod
    def _skill_markdown(skill: SkillCard) -> str:
        validation = skill.validation
        validation_text = "Not validated yet."
        if validation is not None:
            validation_text = (
                f"Quality gain: {validation.quality_gain:.3f}\n"
                f"Candidate score: {validation.candidate_score:.3f}\n"
                f"Baseline score: {validation.baseline_score:.3f}\n"
                f"Candidate pass rate: {validation.candidate_pass_rate:.3f}\n"
                f"Estimated cost: {validation.estimated_cost:.3f}\n"
            )
        return (
            f"# {skill.skill_name}\n\n"
            f"Version: {skill.version}\n\n"
            f"{skill.description}\n\n"
            "## Triggers\n\n"
            + "\n".join(f"- {item}" for item in skill.triggers)
            + "\n\n## Procedure\n\n"
            + "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(skill.procedure))
            + "\n\n## Tools\n\n"
            + "\n".join(f"- {item}" for item in skill.tools)
            + "\n\n## Evaluators\n\n"
            + "\n".join(f"- {item}" for item in skill.evaluators)
            + "\n\n## Fallbacks\n\n"
            + "\n".join(f"- {item}" for item in skill.fallbacks)
            + "\n\n## Anti-Patterns\n\n"
            + "\n".join(f"- {item}" for item in skill.anti_patterns)
            + "\n\n## Held-Out Validation\n\n"
            + validation_text
        )

    @staticmethod
    def _tools_stub(skill: SkillCard) -> str:
        tools = ", ".join(repr(tool) for tool in skill.tools)
        return (
            '"""Executable tool hooks for this materialized skill.\n\n'
            "The main repo routes these symbolic tool names through ToolRegistry.\n"
            '"""\n\n'
            f"REQUIRED_TOOLS = [{tools}]\n"
        )

    @staticmethod
    def _evaluators_stub(skill: SkillCard) -> str:
        evaluators = ", ".join(repr(evaluator) for evaluator in skill.evaluators)
        return (
            '"""Evaluator hooks for this materialized skill."""\n\n'
            f"REQUIRED_EVALUATORS = [{evaluators}]\n"
        )
