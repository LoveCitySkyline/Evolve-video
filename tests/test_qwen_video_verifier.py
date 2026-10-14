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

    def test_json_mode_precondition_for_shared_observations_and_review(self):
        from evovideo_skill.review_observations import VERSION, SYSTEM
        judge = QwenGraphVideoVerifier(self.settings).primary
        evidence = [("Candidate frame", {"mime": "image/png", "data": "AA=="})] * 27
        shared = {"output_contract": {"response_protocol": VERSION},
                  "evidence_manifest": {"evaluation_view": {
                      "kind": "fixed_window_clip", "input_representation": "timestamped_images"}}}
        # Simulate the provider's JSON-mode precondition on actual wire messages,
        # not JSON serialization (which by itself does not include the word json).
        def provider(request, **kwargs):
            body = json.loads(request.data)
            self.assertEqual(body["response_format"], {"type": "json_object"})
            system = body["messages"][0]["content"]
            self.assertIn("json", system.lower())
            prompt = json.loads(body["messages"][1]["content"][0]["text"])
            if prompt.get("output_contract", {}).get("response_protocol") == VERSION:
                self.assertTrue(system.startswith(SYSTEM))
                self.assertNotIn("Return a criteria object", system)
            images = [x for x in body["messages"][1]["content"] if x["type"] == "image_url"]
            self.assertEqual(len(images), 27)
            return io.BytesIO(json.dumps({"choices": [{"message": {
                "content": '{"facts": {}, "judgments": {}}'}}]}).encode())
        for model in ("qwen3.8-max-0902", "qwen3-vl-plus"):
            judge.profile["model"] = model
            for kind, prompt in (("ordinary", {}), ("shared", shared), ("review", shared)):
                with self.subTest(model=model, kind=kind), \
                        patch.object(judge, "response_contract_instructions", "Review the same evidence."), \
                        patch("urllib.request.urlopen", side_effect=provider) as call:
                    self.assertEqual(judge.request(json.dumps(prompt), evidence, model + kind),
                                     {"facts": {}, "judgments": {}})
                    self.assertEqual(call.call_count, 1)

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

    def test_flash_video_frames_and_review_disable_thinking_on_actual_wire(self):
        from evovideo_skill.isolated_observations import VERSION
        self.settings.vlm_model = 'qwen3.8-flash'
        judge = QwenGraphVideoVerifier(self.settings)
        shared = {'output_contract': {'response_protocol': VERSION},
                  'evidence_manifest': {'evaluation_view': {
                      'kind': 'fixed_window_clip', 'input_representation': 'timestamped_images'}}}
        cases = [(judge.primary, {}, 'video/mp4', 1),
                 (judge.primary, shared, 'image/jpeg', 27),
                 (judge.review, shared, 'image/jpeg', 60)]
        def provider(request, **kwargs):
            body = json.loads(request.data)
            self.assertEqual(body['model'], 'qwen3.8-flash')
            self.assertIs(body['enable_thinking'], False)
            self.assertEqual(body['response_format'], {'type': 'json_object'})
            self.assertNotIn('reasoning_effort', body)
            self.assertNotIn('thinking_budget', body)
            self.assertEqual(body['temperature'], 0)
            return io.BytesIO(json.dumps({'choices': [{'message': {'content': '{"facts": {}}'}}]}).encode())
        for index, (verifier, prompt, mime, count) in enumerate(cases):
            with self.subTest(index=index), patch('urllib.request.urlopen', side_effect=provider) as call:
                verifier.request(json.dumps(prompt), [('candidate', {'mime': mime, 'data': 'AA=='})]*count,
                                 f'flash-test-{index}')
                self.assertEqual(call.call_count, 1)
                audit = next(p for p in (verifier.root/'requests').glob('*.json')
                             if json.loads(p.read_text())['operation'] == f'flash-test-{index}')
                self.assertIs(json.loads(audit.read_text())['enable_thinking'], False)

    def test_story350_profiles_use_flash_and_separate_default_output_directories(self):
        from evovideo_skill.conditioning_verifier import resolve_profiles
        root = Path(__file__).resolve().parents[1]
        outputs = set()
        for name in ('debug', 'ks', 'nash', 'net'):
            config = json.loads((root/'configs'/f'h3_story350_{name}.json').read_text())
            settings = RuntimeSettings(**config['runtime'])
            profiles = resolve_profiles(config, settings, require_keys=False, apply_env=False)
            self.assertEqual(settings.vlm_model, 'qwen3.8-flash')
            for phase in ('runtime', 'final'):
                self.assertEqual(profiles[phase]['model'], 'qwen3.8-flash')
                self.assertEqual(profiles[phase]['transport'], 'dashscope_video')
                self.assertIsNone(profiles[phase]['auto_review']['secondary_model'])
                self.assertEqual(profiles[phase]['repeats'], 1)
                self.assertEqual(profiles[phase]['auto_review']['max_calls_per_criterion'], 2)
            self.assertTrue(config['output_dir'].endswith('_qwen38_flash'))
            outputs.add(config['output_dir'])
        self.assertEqual(len(outputs), 4)

    def test_h3_default_configs_select_same_judge(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("h3_local_graph_harness", "h3_native_graph_harness", "h3_local_mini50_harness"):
            config = json.loads((root / "configs" / (name + ".json")).read_text())
            self.assertEqual(config["runtime"]["vlm_provider"], "qwen_video")
            self.assertEqual(config["runtime"]["vlm_model"], "qwen3.8-max-0902")
            self.assertTrue(config["output_dir"].endswith("_qwen38_v2"))


if __name__ == "__main__":
    unittest.main()
