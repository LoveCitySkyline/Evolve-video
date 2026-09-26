from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from evovideo_skill.benchmarks import BenchmarkSuite
from evovideo_skill.evolution_data import stratified_task_split
from evovideo_skill.h3_api import build_request
from evovideo_skill.h3_cli import preflight
from evovideo_skill.h3_local import build_local_request
from evovideo_skill.harness import HarnessConfig
from evovideo_skill.previs_benchmark import FAMILIES, ROOT, SPLITS, build, compile_suite, compile_task, load_scenarios
from evovideo_skill.vlm_evaluator import QwenVLEvaluator


class PrevisBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.records = load_scenarios()

    def test_independent_family_split_coverage(self):
        self.assertEqual(len(self.records), 24)
        self.assertEqual(Counter((r["family"], r["split"]) for r in self.records),
                         Counter({(f, s): 1 for f in FAMILIES for s in SPLITS}))
        self.assertEqual(len({r["id"] for r in self.records}), 24)

    def test_generation_is_deterministic_and_does_not_mutate_source(self):
        before = copy.deepcopy(self.records)
        self.assertEqual(compile_suite(self.records), compile_suite(self.records))
        self.assertEqual(self.records, before)
        self.assertEqual(len({t["prompt"] for t in compile_suite(self.records)["tasks"]}), 24)

    def test_prompt_operator_neutral_and_no_required_assets(self):
        for task in compile_suite(self.records)["tasks"]:
            self.assertNotRegex(task["prompt"].lower(), r"blender|seedance|minimax|gpt-?6")
            self.assertFalse(task["metadata"]["input_assets"]["required"])
            self.assertEqual(task["metadata"]["h3_references"], [])
            self.assertEqual(task["mode"], "generation")
            self.assertEqual(task["duration_seconds"], 12)

    def test_both_h3_request_builders_accept_all_briefs(self):
        for task in compile_suite(self.records)["tasks"]:
            for builder in (build_request, build_local_request):
                request = builder(task["prompt"], "t2va", [], task["duration_seconds"], "768P")
                self.assertNotIn("seed", request)

    def test_continuous_constraints_do_not_become_shot_permission(self):
        for record in self.records:
            task = compile_task(record)
            if record["continuous"]:
                self.assertNotIn("h3_shots", task["metadata"])
                self.assertIn("no cuts", task["prompt"])
            else:
                self.assertEqual(sum(s["duration_seconds"] for s in task["metadata"]["h3_shots"]), 12)

    def test_scoring_is_preregistered_semantic_proxy(self):
        suite = compile_suite(self.records)
        self.assertFalse(suite["protocol"]["official_leaderboard_comparable"])
        self.assertFalse(suite["protocol"]["automatic_precision_verifier_implemented"])
        for task in suite["tasks"]:
            meta = task["metadata"]
            self.assertEqual(meta["reward"]["video_quality_weight"], 0.6)
            self.assertEqual(len(meta["evaluation"]), 3)
            self.assertNotIn("h3_audio_criteria", meta)
            for criterion in meta["evaluation"].values():
                self.assertEqual(criterion["weight"], 1.0)
                self.assertEqual(criterion["threshold"], 0.9)
                self.assertEqual(criterion["measurement_type"], "vlm_semantic_proxy")
                self.assertIn("insufficient", criterion["description"])

    def test_loader_and_splits_no_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = build(directory)
            suite = BenchmarkSuite.from_file(paths["complex_video_bench_previs24"])
            split = stratified_task_split(suite.tasks)
            self.assertEqual([len(split.train), len(split.validation), len(split.test)], [8, 8, 8])
            sets = [set(t.task_id for t in tasks) for tasks in (split.train, split.validation, split.test)]
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            smoke = BenchmarkSuite.from_file(paths["complex_video_bench_previs_smoke6"])
            split = stratified_task_split(smoke.tasks)
            self.assertEqual([len(split.train), len(split.validation), len(split.test)], [2, 2, 2])
            self.assertTrue({t.task_id for t in smoke.tasks}.issubset({t.task_id for t in suite.tasks}))

    def test_h3_entry_preflight_without_credentials_or_network(self):
        config = HarnessConfig.from_file("configs/h3_local_graph_harness.json")
        with tempfile.TemporaryDirectory() as directory:
            config.task_files = [str(build(directory)["complex_video_bench_previs_smoke6"])]
            with patch.dict("os.environ", {}, clear=True), patch("evovideo_skill.h3_cli.shutil.which", return_value="available"), patch("evovideo_skill.h3_cli.importlib.util.find_spec", return_value=object()):
                summary = preflight(config)
            self.assertEqual(summary["preflight"], "passed")
            self.assertEqual(summary["suites"][0]["reference_tasks"], 0)
            self.assertEqual(summary["api_calls_during_preflight"], 0)

    def test_verifier_receives_timeline_and_criterion_descriptions(self):
        suite = BenchmarkSuite.from_file(ROOT / "complex_video_bench_previs24.json")
        task = suite.tasks[0]
        # Prompt construction is a pure static helper, not a network request.
        prompt = QwenVLEvaluator._prompt(task, frame_count=8)
        self.assertIn(task.metadata["state_timeline"][0]["description"], prompt)
        self.assertIn("vlm_semantic_proxy", prompt)
        self.assertIn("insufficient", prompt)

    def test_stored_artifacts_match_source_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, path in build(directory).items():
                self.assertEqual(path.read_bytes(), (ROOT / path.name).read_bytes(), name)
            for suite in ("complex_video_bench_previs24", "complex_video_bench_previs_smoke6"):
                stats = json.loads((ROOT / f"{suite}_stats.json").read_text())
                self.assertEqual(stats["sha256"], hashlib.sha256((ROOT / f"{suite}.json").read_bytes()).hexdigest())

    def test_old_benchmark_untouched_and_ids_disjoint(self):
        old_path = Path("benchmarks/complex_video_bench_1k/complex_video_bench_1k.json")
        old = old_path.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            build(directory)
        self.assertEqual(old_path.read_bytes(), old)
        old_ids = {t["task_id"] for t in json.loads(old)["tasks"]}
        self.assertFalse(old_ids & {t["task_id"] for t in compile_suite(self.records)["tasks"]})
        with self.assertRaises(ValueError):
            build(old_path.parent)

    def test_invalid_source_rejected(self):
        variants = []
        duplicate = copy.deepcopy(self.records)
        duplicate[1]["id"] = duplicate[0]["id"]
        variants.append(duplicate)
        missing_beat = copy.deepcopy(self.records)
        missing_beat[0]["beats"].pop()
        variants.append(missing_beat)
        biased = copy.deepcopy(self.records)
        biased[0]["setup"] = "Use Blender to solve this task."
        variants.append(biased)
        for records in variants:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.json"
                path.write_text(json.dumps(records))
                with self.assertRaises(ValueError):
                    load_scenarios(path)


if __name__ == "__main__":
    unittest.main()
