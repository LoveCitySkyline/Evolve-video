from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, field
from typing import Any

from evovideo_skill.models import VideoTask


def task_category(task: VideoTask) -> str:
    metadata = task.metadata or {}
    for key in ("category", "vbench_dimension", "dimension"):
        value = metadata.get(key)
        if value:
            return str(value)
    failures = metadata.get("expected_failure_modes", [])
    if failures:
        return str(failures[0])
    return task.mode.value


@dataclass
class EvolutionDataset:
    train: list[VideoTask]
    validation: list[VideoTask]
    test: list[VideoTask]
    categories: dict[str, list[str]]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "train": [task.task_id for task in self.train],
            "validation": [task.task_id for task in self.validation],
            "test": [task.task_id for task in self.test],
            "categories": self.categories,
            "seed": self.seed,
        }


def stratified_task_split(
    tasks: list[VideoTask],
    train_ratio: float = 0.5,
    validation_ratio: float = 0.25,
    seed: int = 0,
) -> EvolutionDataset:
    if not tasks:
        return EvolutionDataset([], [], [], {}, seed)
    if train_ratio <= 0 or validation_ratio < 0 or train_ratio + validation_ratio > 1:
        raise ValueError("split ratios must satisfy train_ratio > 0 and train_ratio + validation_ratio <= 1")

    grouped: dict[str, list[VideoTask]] = {}
    for task in tasks:
        grouped.setdefault(task_category(task), []).append(task)

    predefined = [str((task.metadata or {}).get("split", "")).lower() for task in tasks]
    valid_splits = {"train", "validation", "test"}
    if predefined and all(split in valid_splits for split in predefined):
        categories = {
            category: [task.task_id for task in group]
            for category, group in sorted(grouped.items())
        }
        return EvolutionDataset(
            train=[task for task, split in zip(tasks, predefined) if split == "train"],
            validation=[task for task, split in zip(tasks, predefined) if split == "validation"],
            test=[task for task, split in zip(tasks, predefined) if split == "test"],
            categories=categories,
            seed=seed,
        )

    train: list[VideoTask] = []
    validation: list[VideoTask] = []
    test: list[VideoTask] = []
    for category in sorted(grouped):
        group = list(grouped[category])
        category_seed = int(hashlib.sha256(f"{seed}:{category}".encode()).hexdigest()[:8], 16)
        random.Random(category_seed).shuffle(group)
        n_items = len(group)
        n_train = max(1, int(n_items * train_ratio))
        n_validation = int(n_items * validation_ratio)
        if n_items >= 3 and n_validation == 0:
            n_validation = 1
        if n_train + n_validation > n_items:
            n_validation = max(0, n_items - n_train)
        train.extend(group[:n_train])
        validation.extend(group[n_train : n_train + n_validation])
        test.extend(group[n_train + n_validation :])

    # Tiny suites still need a held-out signal. Move, never duplicate, a sample.
    if not validation and len(train) > 1:
        validation.append(train.pop())
    if not test and len(train) > 2:
        test.append(train.pop())
    if not validation:
        validation = list(train)

    categories = {
        category: [task.task_id for task in group]
        for category, group in sorted(grouped.items())
    }
    return EvolutionDataset(train, validation, test, categories, seed)


@dataclass
class RoundRobinTaskSampler:
    tasks: list[VideoTask]
    category_offset: int = 0
    per_category_offset: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._pools: dict[str, list[VideoTask]] = {}
        for task in self.tasks:
            self._pools.setdefault(task_category(task), []).append(task)
        for category in self._pools:
            self.per_category_offset.setdefault(category, 0)

    def sample(self, categories_per_batch: int, samples_per_category: int) -> list[VideoTask]:
        categories = sorted(self._pools)
        if not categories:
            return []
        selected: list[VideoTask] = []
        count = min(max(1, categories_per_batch), len(categories))
        for index in range(count):
            category = categories[(self.category_offset + index) % len(categories)]
            pool = self._pools[category]
            for _ in range(min(max(1, samples_per_category), len(pool))):
                offset = self.per_category_offset[category] % len(pool)
                selected.append(pool[offset])
                self.per_category_offset[category] += 1
        self.category_offset = (self.category_offset + count) % len(categories)
        return selected

    def sample_homogeneous(self, batch_size: int) -> list[VideoTask]:
        """Sample one task family per evolution round.

        A mutation proposal must have one causal target. Mixing unrelated task
        families in the same failure batch makes both LLM planning and paired
        validation ambiguous, so category coverage is rotated across rounds.
        """
        categories = sorted(self._pools)
        if not categories:
            return []
        category = categories[self.category_offset % len(categories)]
        pool = self._pools[category]
        selected: list[VideoTask] = []
        for _ in range(min(max(1, batch_size), len(pool))):
            offset = self.per_category_offset[category] % len(pool)
            selected.append(pool[offset])
            self.per_category_offset[category] += 1
        self.category_offset = (self.category_offset + 1) % len(categories)
        return selected

    def state_dict(self) -> dict[str, Any]:
        return {
            "category_offset": self.category_offset,
            "per_category_offset": dict(self.per_category_offset),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.category_offset = int(state.get("category_offset", 0))
        saved = state.get("per_category_offset", {})
        for category in self._pools:
            self.per_category_offset[category] = int(saved.get(category, 0))
