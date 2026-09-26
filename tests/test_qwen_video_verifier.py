import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.api_tools import VideoApiError
from evovideo_skill.gemini_verifier import QwenGraphVideoVerifier, VerifierEvidenceUnavailable
from evovideo_skill.models import VideoTask
from evovideo_skill.runtime import RuntimeSettings, build_vlm_augmenter


class QwenVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        env = patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-only"}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.settings = RuntimeSettings(enable_vlm_eval=True, vlm_provider="qwen_video",
                                        video_output_dir=self.temp.name)

    def test_native_defaults_without_google_key(self):
        judge = build_vlm_augmenter(self.settings).evaluator
        self.assertIsInstance(judge, QwenGraphVideoVerifier)
        self.assertEqual(judge.model, "qwen3.8-max-0902")
        self.assertEqual(judge.primary.profile["transport"], "dashscope_video")
        self.assertEqual(judge.primary.profile["fps"], 4)
        self.assertEqual(judge.review.profile["fps"], 8)
        self.assertEqual(judge.primary.profile["api_key_env"], "DASHSCOPE_API_KEY")

    def test_invalid_settings_fail_early(self):
        self.settings.vlm_model = "gemini-3.1-pro-preview"
        with self.assertRaisesRegex(ValueError, "stale VLM_MODEL"):
            QwenGraphVideoVerifier(self.settings)
        self.settings.vlm_model = None
        for fps in (0, .05, 11, float("nan")):
            self.settings.vlm_video_fps = fps
            with self.assertRaisesRegex(ValueError, "VLM_VIDEO_FPS"):
                QwenGraphVideoVerifier(self.settings)
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(VideoApiError, "DASHSCOPE_API_KEY"):
            QwenGraphVideoVerifier(self.settings)

    def test_endpoint_precedence(self):
        with patch.dict(os.environ, {"DASHSCOPE_COMPAT_BASE_URL": "https://workspace.example/v1"}):
            judge = QwenGraphVideoVerifier(self.settings)
            self.assertEqual(judge.primary.profile["base_url"], "https://workspace.example/v1")
            self.settings.vlm_base_url = "https://explicit.example/v1"
            judge = QwenGraphVideoVerifier(self.settings)
            self.assertEqual(judge.primary.profile["base_url"], self.settings.vlm_base_url)

    def test_video_json_payload_and_no_secret_in_logs(self):
        judge = QwenGraphVideoVerifier(self.settings).primary
        evidence = [("ANONYMOUS CANDIDATE VIDEO", {"mime": "video/mp4", "data": "AA=="})]
        response = {"choices": [{"message": {"content": '{"criteria": {}}'}}]}
        captured = []
        def urlopen(request, **kwargs):
            captured.append(request)
            return io.BytesIO(json.dumps(response).encode())
        with patch("urllib.request.urlopen", side_effect=urlopen):
            self.assertEqual(judge.request("Return JSON evidence", evidence, "unit"), {"criteria": {}})
        request = captured[0]
        self.assertTrue(request.full_url.endswith("/compatible-mode/v1/chat/completions"))
        self.assertEqual(request.get_header("Authorization"), "Bearer test-only")
        body = json.loads(request.data)
        self.assertEqual(body["model"], "qwen3.8-max-0902")
        self.assertIs(body["enable_thinking"], False)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["messages"][1]["content"][-1], {
            "type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA=="}, "fps": 4})
        self.assertNotIn("test-only", (judge.root / "calls.jsonl").read_text())

    def test_missing_evidence_review_not_score_shopping(self):
        judge = QwenGraphVideoVerifier(self.settings)
        task = VideoTask("t", "Move")
        low = {"evaluation_status": "complete", "criterion_scores": {"action": 0}, "verification_metadata": {}}
        unknown = {"evaluation_status": "needs_review", "verification_metadata": {}}
        with patch.object(judge.primary, "evaluate", return_value=low), patch.object(judge.review, "evaluate") as review:
            result = judge.evaluate(task, None)
            self.assertEqual(result["criterion_scores"]["action"], 0)
            self.assertEqual(result["verification_metadata"]["transport"], "dashscope_video")
            review.assert_not_called()
        with patch.object(judge.primary, "evaluate", return_value=unknown), patch.object(judge.review, "evaluate", return_value=low):
            result = judge.evaluate(task, None)
            self.assertEqual(len(result["verification_metadata"]["evidence_passes"]), 2)
        with patch.object(judge.primary, "evaluate", return_value=unknown), patch.object(judge.review, "evaluate", return_value=unknown):
            with self.assertRaises(VerifierEvidenceUnavailable):
                judge.evaluate(task, None)
        with patch.object(judge.primary, "evaluate", side_effect=VideoApiError("HTTP 429")):
            with self.assertRaises(VerifierEvidenceUnavailable):
                judge.evaluate(task, None)

    def test_h3_default_configs_select_same_judge(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("h3_local_graph_harness", "h3_native_graph_harness", "h3_local_mini50_harness"):
            config = json.loads((root / "configs" / (name + ".json")).read_text())
            self.assertEqual(config["runtime"]["vlm_provider"], "qwen_video")
            self.assertEqual(config["runtime"]["vlm_model"], "qwen3.8-max-0902")
            self.assertTrue(config["output_dir"].endswith("_qwen38_v2"))


if __name__ == "__main__":
    unittest.main()
