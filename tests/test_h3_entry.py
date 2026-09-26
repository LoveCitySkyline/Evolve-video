import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evovideo_skill.benchmarks import BenchmarkSuite
from evovideo_skill.h3_cli import preflight
from evovideo_skill.harness import HarnessConfig
from evovideo_skill.h3_evidence import H3MultimodalEvaluator
from evovideo_skill.api_tools import VideoApiError
from evovideo_skill.models import VideoTask, VideoArtifact, TaskMode


class H3EntryTests(unittest.TestCase):
    def test_default_preflight_is_offline_and_split_disjoint(self):
        config = HarnessConfig.from_file("configs/h3_native_graph_harness.json")
        with patch.dict("os.environ", {"PATH": os.environ.get("PATH", "")}, clear=True):
            result = preflight(config)
        self.assertEqual(result["api_calls_during_preflight"], 0)
        self.assertEqual(result["suites"][0]["tasks"], 9)
        self.assertEqual(result["replicates"], 3)

    def test_provider_and_replicate_preflight_guards(self):
        config = HarnessConfig.from_file("configs/h3_native_graph_harness.json")
        with patch.dict("os.environ", {"PROVIDER": "local-wan"}, clear=True), self.assertRaisesRegex(ValueError, "stale PROVIDER"):
            preflight(config)
        config.evaluation_seeds = [42]
        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(ValueError, "3 evaluation"):
            preflight(config)

    def test_local_reference_fingerprint_changes_with_asset(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            media = root / "hero.png"
            media.write_bytes(b"first")
            path = root / "suite.json"
            path.write_text(json.dumps({"tasks": [{"task_id": "ref", "prompt": "test", "metadata": {
                "h3_references": [{"id": "hero", "kind": "image", "uri": "hero.png", "role": "reference_image"}]}}]}))
            first = BenchmarkSuite.from_file(path).tasks[0].metadata["h3_references"][0]
            media.write_bytes(b"second")
            second = BenchmarkSuite.from_file(path).tasks[0].metadata["h3_references"][0]
            self.assertEqual(first["uri"], str(media.resolve()))
            self.assertNotEqual(first["content_sha256"], second["content_sha256"])

    def test_audio_verifier_overrides_only_declared_audio_criteria(self):
        class Visual:
            model = "visual-test"
            def evaluate(self, task, artifact):
                return {"criterion_scores": {"sync": 0, "identity": .6}}
            def _normalize_benchmark_result(self, task, result):
                pass
        with tempfile.TemporaryDirectory() as root:
            evaluator = H3MultimodalEvaluator(Visual(), ["audio-evaluator"], root)
            task = VideoTask("av", "Say hello", metadata={"h3_audio_criteria": ["sync"], "evaluation": {"sync": {}, "identity": {}}})
            artifact = VideoArtifact("clip", "av", task.prompt, TaskMode.GENERATION, [], [], {"local_video_path": "video.mp4"})
            def run(argv, **kwargs):
                Path(argv[-1]).write_text(json.dumps({"verifier": "test-fixture-v1", "criterion_scores": {"sync": .8, "identity": 1}, "criterion_evidence": {"sync": "observed test signal"}}))
            with patch("evovideo_skill.h3_evidence.subprocess.run", side_effect=run):
                result = evaluator.evaluate(task, artifact)
            self.assertEqual(result["criterion_scores"], {"sync": .8, "identity": .6})
            self.assertTrue(result["audio_evidence_available"])
            def invalid(argv, **kwargs):
                Path(argv[-1]).write_text("{}")
            with patch("evovideo_skill.h3_evidence.subprocess.run", side_effect=invalid), self.assertRaises(VideoApiError):
                evaluator.evaluate(task, artifact)


if __name__ == "__main__":
    unittest.main()
