import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.api_tools import VideoApiError
from evovideo_skill.gemini_verifier import GeminiGraphVerifier, VerifierEvidenceUnavailable
from evovideo_skill.graph_evolver import h3_run_should_stop
from evovideo_skill.models import VideoTask
from evovideo_skill.runtime import RuntimeSettings, build_vlm_augmenter
from evovideo_skill.tools import SegmentStitcherTool
from evovideo_skill.vlm_evaluator import VLMEvidenceAugmenter


class GeminiGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict(os.environ, {"GEMINI_API_KEY": "test-only"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.settings = RuntimeSettings(enable_vlm_eval=True, vlm_provider="gemini",
            video_output_dir=self.temp.name, vlm_model="gemini-3.1-pro-preview")

    def test_runtime_uses_native_video_without_dashscope_credentials(self):
        evaluator = build_vlm_augmenter(self.settings).evaluator
        self.assertIsInstance(evaluator, GeminiGraphVerifier)
        self.assertEqual(evaluator.primary.profile["transport"], "gemini_video")
        self.assertEqual(evaluator.primary.profile["fps"], 4)
        self.assertEqual(evaluator.review.profile["fps"], 8)
        self.assertEqual(evaluator.primary.profile["api_key_env"], "GEMINI_API_KEY")

    def test_invalid_model_fps_and_missing_credentials_fail_early(self):
        self.settings.vlm_model = "qwen3-vl-plus"
        with self.assertRaisesRegex(ValueError, "stale VLM_MODEL"):
            GeminiGraphVerifier(self.settings)
        self.settings.vlm_model = "gemini-3.1-pro-preview"
        self.settings.vlm_review_fps = 1
        with self.assertRaisesRegex(ValueError, "VLM_VIDEO_FPS"):
            GeminiGraphVerifier(self.settings)
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(VideoApiError, "GEMINI_API_KEY"):
            GeminiGraphVerifier(self.settings)

    def test_only_inconclusive_not_low_score_triggers_review(self):
        evaluator = GeminiGraphVerifier(self.settings)
        low = {"evaluation_status": "complete", "criterion_scores": {"action": 0}, "verification_metadata": {}}
        uncertain = {"evaluation_status": "needs_review", "verification_metadata": {"unobserved_criteria": ["action"]}}
        task = VideoTask("t", "Move")
        with patch.object(evaluator.primary, "evaluate", return_value=low), patch.object(evaluator.review, "evaluate") as review:
            self.assertEqual(evaluator.evaluate(task, None)["criterion_scores"]["action"], 0)
            review.assert_not_called()
        with patch.object(evaluator.primary, "evaluate", return_value=uncertain), patch.object(evaluator.review, "evaluate", return_value=low) as review:
            result = evaluator.evaluate(task, None)
            self.assertEqual(len(result["verification_metadata"]["evidence_passes"]), 2)
            review.assert_called_once()

    def test_unknown_and_api_errors_never_become_zero_quality_training_examples(self):
        evaluator = GeminiGraphVerifier(self.settings)
        uncertain = {"evaluation_status": "needs_review", "verification_metadata": {"unobserved_criteria": ["action"]}}
        with patch.object(evaluator.primary, "evaluate", return_value=uncertain), patch.object(evaluator.review, "evaluate", return_value=uncertain):
            with self.assertRaises(VerifierEvidenceUnavailable) as caught:
                VLMEvidenceAugmenter(evaluator).augment(VideoTask("t", "Move"), None)
        self.assertTrue(h3_run_should_stop(caught.exception))
        with patch.object(evaluator.primary, "evaluate", side_effect=VideoApiError("HTTP 429")):
            with self.assertRaises(VerifierEvidenceUnavailable):
                evaluator.evaluate(VideoTask("t", "Move"), None)

    def test_unrelated_condition_profiles_do_not_override_graph_judge(self):
        with patch.dict(os.environ, {"CONDITION_RUNTIME_VERIFIER_MODEL": "other-model"}):
            evaluator = GeminiGraphVerifier(self.settings)
        self.assertEqual(evaluator.primary.model, "gemini-3.1-pro-preview")


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires ffmpeg")
class RepairTimelineTests(unittest.TestCase):
    def test_entire_segment_is_retimed_and_full_video_is_cropped_at_target(self):
        import cv2

        with TemporaryDirectory() as folder:
            root = Path(folder)
            source, repair = root / "source.mp4", root / "repair.mp4"
            base = ["ffmpeg", "-v", "error", "-nostdin", "-y"]
            subprocess.run(base + ["-f", "lavfi", "-i", "color=green:s=64x48:r=12:d=6",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=6", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source)], check=True)
            subprocess.run(base + ["-f", "lavfi", "-i", "color=red:s=64x48:r=12:d=2",
                "-f", "lavfi", "-i", "color=blue:s=64x48:r=12:d=2", "-filter_complex",
                "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-c:v", "libx264", str(repair)], check=True)
            tool = SegmentStitcherTool(root)
            task = VideoTask("t", "A color transition", duration_seconds=6)
            span = {"start_ratio": 1/3, "end_ratio": 2/3}
            def pixel(path, second):
                capture = cv2.VideoCapture(str(path))
                capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
                ok, frame = capture.read()
                capture.release()
                self.assertTrue(ok)
                return frame.mean(axis=(0, 1))
            segmented = tool._stitch_video(task, str(source), str(repair), span, "segment", "segment")
            self.assertEqual(pixel(segmented, 2.3).argmax(), 2)  # red opening
            self.assertEqual(pixel(segmented, 3.6).argmax(), 0)  # blue ending retained
            self.assertEqual(pixel(segmented, .5).argmax(), 1)
            self.assertEqual(pixel(segmented, 5).argmax(), 1)
            cropped = tool._stitch_video(task, str(source), str(repair), span, "full", "full_video")
            self.assertEqual(pixel(cropped, 2.3).argmax(), 0)  # use seconds 2..4, not 0..2
            info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", cropped]))
            self.assertTrue(any(s["codec_type"] == "audio" for s in info["streams"]))
            self.assertAlmostEqual(float(info["format"]["duration"]), 6, delta=.15)


if __name__ == "__main__":
    unittest.main()
