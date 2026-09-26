from __future__ import annotations

import tempfile
import unittest
import json
from collections import Counter
from pathlib import Path

from evovideo_skill.benchmarks import BenchmarkSuite
from evovideo_skill.complex_benchmark import FAMILY_COUNTS, generate_tasks, validate_tasks, write_benchmark
from evovideo_skill.evolution_data import stratified_task_split
from scripts.generate_complex_video_bench_mini50 import (
    DIFFICULTIES,
    PRESET_ALLOCATIONS,
    PRESET_ORDER,
    _split_targets,
    build_focused_mini5,
    build_nested_subsets,
)


class ComplexBenchmarkTests(unittest.TestCase):
    def test_mini15_prepared_pool_covers_every_pixel_changing_task_family(self):
        payload = json.loads(
            Path("configs/prepared_tool_capabilities_mini15.json").read_text(encoding="utf-8")
        )
        capabilities = {item["capability"] for item in payload["capabilities"]}

        self.assertTrue({
            "image_conditioned_video_generation",
            "multi_shot_identity_conditioned_generation",
            "motion_conditioned_video_generation",
            "global_video_editing",
            "region_video_editing",
            "failed_segment_repair",
            "video_style_transfer",
            "temporal_deflickering",
            "audio_conditioned_video_generation",
        }.issubset(capabilities))

    def test_generates_exact_balanced_suite(self):
        tasks = generate_tasks()
        stats = validate_tasks(tasks)
        self.assertEqual(stats["task_count"], 1000)
        self.assertEqual(stats["family_counts"], FAMILY_COUNTS)
        self.assertEqual(stats["split_counts"], {"train": 600, "validation": 200, "test": 200})
        self.assertEqual(len({task["prompt"] for task in tasks}), 1000)

    def test_scenario_groups_do_not_cross_splits(self):
        group_splits = {}
        for task in generate_tasks():
            metadata = task["metadata"]
            group_splits.setdefault(metadata["scenario_group"], set()).add(metadata["split"])
        self.assertTrue(all(len(splits) == 1 for splits in group_splits.values()))

    def test_audio_tasks_declare_task_conditioned_reward(self):
        audio_tasks = [
            task
            for task in generate_tasks()
            if task["metadata"]["task_family"] == "audio_video_sync"
        ]

        self.assertTrue(audio_tasks)
        for task in audio_tasks:
            reward = task["metadata"]["reward"]
            self.assertEqual(reward["objective"], "task_conditioned_multimodal_reward")
            self.assertEqual(reward["video_quality_weight"], 0.5)
            self.assertIn("audio_event_alignment", task["metadata"]["evaluation"])

    def test_written_suite_loads_and_preserves_predefined_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_benchmark(tmp)
            suite = BenchmarkSuite.from_file(paths["benchmark"])
            dataset = stratified_task_split(suite.tasks, train_ratio=0.1, validation_ratio=0.1, seed=999)
            self.assertEqual((len(dataset.train), len(dataset.validation), len(dataset.test)), (600, 200, 200))
            self.assertEqual(Counter(task.metadata["split"] for task in suite.tasks), {"train": 600, "validation": 200, "test": 200})
            self.assertTrue(Path(paths["assets"]).exists())
            asset_payload = json.loads(Path(paths["assets"]).read_text(encoding="utf-8"))
            asset_ids = [asset["asset_id"] for asset in asset_payload["assets"]]
            self.assertEqual(len(asset_ids), 70)
            self.assertEqual(len(asset_ids), len(set(asset_ids)))

    def test_benchmark_loader_materializes_audio_and_source_asset_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "media"
            media.mkdir()
            (media / "track.wav").write_bytes(b"RIFF")
            (media / "source.mp4").write_bytes(b"video")
            benchmark = root / "suite.json"
            benchmark.write_text(json.dumps({"tasks": [{
                "task_id": "audio",
                "prompt": "sync audio",
                "metadata": {
                    "input_assets": {"audio": "a1", "source_video": "v1", "required": True}
                },
            }]}), encoding="utf-8")
            (root / "suite_assets.json").write_text(json.dumps({"assets": [
                {"asset_id": "a1", "type": "audio", "path": "media/track.wav"},
                {"asset_id": "v1", "type": "source_video", "path": "media/source.mp4"},
            ]}), encoding="utf-8")

            task = BenchmarkSuite.from_file(benchmark).tasks[0]

            self.assertEqual(task.metadata["local_audio_path"], str((media / "track.wav").resolve()))
            self.assertEqual(task.reference_video, str((media / "source.mp4").resolve()))
            self.assertNotIn("missing_required_assets", task.metadata)

    def test_mini_subsets_are_nested_stratified_and_difficulty_balanced(self):
        subsets = build_nested_subsets(generate_tasks())
        previous_ids: set[str] = set()
        for preset in PRESET_ORDER:
            tasks = subsets[preset]
            task_ids = {task["task_id"] for task in tasks}
            allocation = PRESET_ALLOCATIONS[preset]
            self.assertTrue(previous_ids.issubset(task_ids))
            self.assertEqual(Counter(task["metadata"]["task_family"] for task in tasks), allocation)
            self.assertEqual(
                Counter(task["metadata"]["difficulty"] for task in tasks),
                Counter({difficulty: len(tasks) // len(DIFFICULTIES) for difficulty in DIFFICULTIES}),
            )
            self.assertEqual(
                len({task["metadata"]["scenario_group"] for task in tasks}),
                len(tasks),
            )
            expected_splits = Counter()
            for family_index, total in enumerate(allocation.values()):
                expected_splits.update(_split_targets(total, family_index))
            self.assertEqual(Counter(task["metadata"]["split"] for task in tasks), expected_splits)
            previous_ids = task_ids

    def test_mini5_is_executable_single_family_and_split_complete(self):
        tasks = build_focused_mini5(generate_tasks())

        self.assertEqual(len(tasks), 5)
        self.assertEqual({task["metadata"]["task_family"] for task in tasks}, {"multi_shot_identity"})
        self.assertEqual({task["metadata"]["difficulty"] for task in tasks}, set(DIFFICULTIES))
        self.assertEqual(Counter(task["metadata"]["split"] for task in tasks), {
            "train": 3,
            "validation": 1,
            "test": 1,
        })
        self.assertEqual(len({task["metadata"]["scenario_group"] for task in tasks}), 5)
        self.assertTrue(all(task["metadata"].get("asset_status") != "manifest_required" for task in tasks))


if __name__ == "__main__":
    unittest.main()
