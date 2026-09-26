from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evovideo_skill.models import utc_now


@dataclass
class EvolutionFeedback:
    iteration: int
    parent_program: str
    child_program: str | None
    task_ids: list[str]
    failure_types: list[str]
    proposal: str
    justification: str
    outcome: str
    parent_score: float
    child_score: float | None
    graph_id: str | None = None
    evidence: list[str] = field(default_factory=list)
    discovery_task_ids: list[str] = field(default_factory=list)
    validation_task_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)


class FeedbackJournal:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.jsonl_path = self.root / "feedback_history.jsonl"
        self.markdown_path = self.root / "feedback_history.md"
        self.root.mkdir(parents=True, exist_ok=True)

    def append(self, entry: EvolutionFeedback) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        with self.markdown_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"\n## Iteration {entry.iteration}: {entry.outcome}\n\n"
                f"- Parent: `{entry.parent_program}`\n"
                f"- Child: `{entry.child_program or 'none'}`\n"
                f"- Discovery tasks: {', '.join(entry.discovery_task_ids or entry.task_ids) or 'none'}\n"
                f"- Validation tasks: {', '.join(entry.validation_task_ids) or 'none'}\n"
                f"- Failures: {', '.join(entry.failure_types) or 'none'}\n"
                f"- Proposal: {entry.proposal}\n"
                f"- Justification: {entry.justification}\n"
                f"- Score: {entry.parent_score:.4f} -> "
                f"{entry.child_score if entry.child_score is not None else 'n/a'}\n"
            )

    def entries(self) -> list[EvolutionFeedback]:
        if not self.jsonl_path.exists():
            return []
        entries = []
        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entries.append(EvolutionFeedback(**json.loads(line)))
        return entries

    def rejected_graph_ids(self) -> set[str]:
        return {
            entry.graph_id
            for entry in self.entries()
            if entry.graph_id and entry.outcome in {"rejected", "dominated", "duplicate"}
        }
