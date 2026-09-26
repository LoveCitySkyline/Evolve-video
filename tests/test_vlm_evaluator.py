import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import MagicMock, patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evovideo_skill.evaluators import EvaluatorSuite
from evovideo_skill.api_tools import VideoApiError
from evovideo_skill.models import TaskMode, VideoArtifact, VideoTask
from evovideo_skill.vlm_evaluator import QwenVLEvaluator, VLMEvidenceAugmenter


class FakeVLM:
    model = "fake-qwen3-vl"

    def evaluate(self, task, artifact):
        return {
            "model": self.model,
            "identity_consistency_score": 0.4,
            "clothing_color_score": 0.3,
            "action_alignment_score": 0.2,
            "background_preservation_score": 1.0,
            "target_edit_success_score": 1.0,
            "identity_evidence": "face changes across frames",
            "clothing_evidence": "red coat disappears",
            "action_evidence": "wave is not visible",
        }


class MissingFrameVLM:
    model = "fake-qwen3-vl"

    def evaluate(self, task, artifact):
        raise VideoApiError("Qwen-VL evaluation requires sampled frame paths")


class FailingApiVLM:
    model = "fake-qwen3-vl"

    def evaluate(self, task, artifact):
        raise VideoApiError("request timed out")


class VLMIntegrationTest(unittest.TestCase):
    def test_source_and_candidate_frames_are_labeled_for_paired_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jpg"
            candidate = root / "candidate.jpg"
            source.write_bytes(b"source")
            candidate.write_bytes(b"candidate")
            task = VideoTask(
                "paired-edit",
                "Stylize the source while preserving its motion.",
                mode=TaskMode.EDITING,
                reference_video=str(root / "source.mp4"),
                metadata={"evaluation": {"motion_preservation": {"threshold": 0.9}}},
            )
            artifact = VideoArtifact(
                "candidate",
                task.task_id,
                task.prompt,
                task.mode,
                ["style_tool"],
                [],
                {
                    "sampled_frame_paths": [str(candidate)],
                    "source_sampled_frame_paths": [str(source)],
                },
            )
            evaluator = QwenVLEvaluator(api_key="test", max_images=2)
            response = {
                "choices": [{"message": {"content": '{"criterion_scores":{"motion_preservation":1.0}}'}}]
            }

            with patch.object(evaluator, "_post", return_value=response) as post:
                result = evaluator.evaluate(task, artifact)

            content = post.call_args.args[1]["messages"][0]["content"]
            labels = [item["text"] for item in content if item["type"] == "text"]
            self.assertTrue(any("REFERENCE SOURCE FRAMES" in item for item in labels))
            self.assertTrue(any("CANDIDATE OUTPUT FRAMES" in item for item in labels))
            self.assertEqual(sum(item["type"] == "image_url" for item in content), 2)
            self.assertEqual(result["source_frame_count"], 1)
            self.assertEqual(result["candidate_frame_count"], 1)

    def test_transient_tls_url_error_is_retried(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"ok": true}'
        evaluator = QwenVLEvaluator(
            api_key="test", max_retries=2, retry_backoff_seconds=0
        )
        with patch(
            "urllib.request.urlopen",
            side_effect=[urllib.error.URLError("TLS connection closed"), response],
        ) as urlopen:
            result = evaluator._post("/chat/completions", {"model": "test"})

        self.assertEqual(result, {"ok": True})
        self.assertEqual(urlopen.call_count, 2)

    def test_vlm_metadata_overrides_evaluator_scores(self):
        task = VideoTask(
            task_id="vlm-test",
            prompt="A young woman in a red coat walks, then turns and waves.",
            mode=TaskMode.GENERATION,
        )
        artifact = VideoArtifact(
            artifact_id="a",
            task_id=task.task_id,
            prompt=task.prompt,
            mode=task.mode,
            tool_chain=["test"],
            frames=[{"identity": "same", "clothing_color": "red", "action": "walks turns waves"} for _ in range(3)],
            metadata={},
        )
        VLMEvidenceAugmenter(FakeVLM()).augment(task, artifact)
        report = EvaluatorSuite().evaluate(task, artifact)

        scores = {metric.name: metric.score for metric in report.metrics}
        self.assertEqual(scores["identity_consistency"], 0.4)
        self.assertEqual(scores["clothing_color_consistency"], 0.3)
        self.assertEqual(scores["prompt_action_alignment"], 0.2)
        self.assertEqual(
            {metric.name for metric in report.active_metrics},
            {"identity_consistency", "clothing_color_consistency", "prompt_action_alignment"},
        )
        self.assertAlmostEqual(report.score, 0.3)
        self.assertFalse(report.passed)
        self.assertEqual(artifact.metadata["vlm_model"], "fake-qwen3-vl")

    def test_missing_frames_become_failed_evidence_instead_of_crashing(self):
        task = VideoTask("missing-video", "A woman walks and waves.")
        artifact = VideoArtifact(
            artifact_id="missing",
            task_id=task.task_id,
            prompt=task.prompt,
            mode=task.mode,
            tool_chain=["local_wan"],
            frames=[],
            metadata={"video_processing_error": "failed to decode MP4"},
        )

        VLMEvidenceAugmenter(MissingFrameVLM()).augment(task, artifact)
        report = EvaluatorSuite().evaluate(task, artifact)

        self.assertEqual(artifact.metadata["vlm_evaluation"]["evaluation_status"], "failed_missing_frames")
        self.assertIn("failed to decode MP4", artifact.metadata["vlm_evaluation_error"])
        self.assertFalse(report.passed)

    def test_vlm_api_failure_with_frames_becomes_failed_evidence(self):
        task = VideoTask("vlm-timeout", "A woman walks and waves.")
        artifact = VideoArtifact(
            artifact_id="video",
            task_id=task.task_id,
            prompt=task.prompt,
            mode=task.mode,
            tool_chain=["local_wan"],
            frames=[{"frame_path": "/tmp/frame.jpg"}],
            metadata={"sampled_frame_paths": ["/tmp/frame.jpg"]},
        )

        VLMEvidenceAugmenter(FailingApiVLM()).augment(task, artifact)
        report = EvaluatorSuite().evaluate(task, artifact)

        self.assertEqual(artifact.metadata["vlm_evaluation"]["evaluation_status"], "failed_api")
        self.assertIn("request timed out", artifact.metadata["vlm_evaluation_error"])
        self.assertFalse(report.passed)


if __name__ == "__main__":
    unittest.main()
