import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

from evovideo_skill.api_tools import VideoApiError
from evovideo_skill.models import VideoArtifact, VideoTask
from evovideo_skill.verifier_cache import VerifierCache
from evovideo_skill.vlm_evaluator import QwenVLEvaluator


class VerifierCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.video = self.root / "video.mp4"
        self.video.write_bytes(b"video-original")
        self.frame = self.root / "frame.jpg"
        self.frame.write_bytes(b"frame-original")
        self.task = VideoTask("task", "Fold and throw a paper airplane.", metadata={"evaluation": {"order": {"weight": 1}}})
        self.artifact = VideoArtifact("video", "task", self.task.prompt, self.task.mode, [], [], {
            "local_video_path": str(self.video), "sampled_frame_paths": [str(self.frame)],
        })

    def evaluator(self, **kwargs):
        return QwenVLEvaluator(api_key="secret-not-logged", cache_dir=self.root / "cache", **kwargs)

    @staticmethod
    def response():
        return {"choices": [{"message": {"content": json.dumps({"criterion_scores": {"order": .8}, "action_alignment_score": .8})}}]}

    def test_persistent_cache_ignores_task_id_and_invalidates_changed_evidence(self):
        with patch.object(QwenVLEvaluator, "_post", return_value=self.response()) as post:
            first = self.evaluator().evaluate(self.task, self.artifact)
            self.task.task_id = "renamed"
            second = self.evaluator().evaluate(self.task, self.artifact)
            self.assertFalse(first["verifier_cache"]["hit"])
            self.assertTrue(second["verifier_cache"]["hit"])
            self.assertEqual(post.call_count, 1)
            self.video.write_bytes(b"video-replaced-at-same-path")
            self.assertFalse(self.evaluator().evaluate(self.task, self.artifact)["verifier_cache"]["hit"])
            self.frame.write_bytes(b"frame-replaced")
            self.assertFalse(self.evaluator().evaluate(self.task, self.artifact)["verifier_cache"]["hit"])
            self.task.metadata["evaluation"]["order"]["description"] = "Before/after ordering"
            self.assertFalse(self.evaluator().evaluate(self.task, self.artifact)["verifier_cache"]["hit"])
            self.assertFalse(self.evaluator(model="other").evaluate(self.task, self.artifact)["verifier_cache"]["hit"])
            self.assertFalse(self.evaluator(cache_namespace="independent-review").evaluate(self.task, self.artifact)["verifier_cache"]["hit"])
            self.assertEqual(post.call_count, 6)
        for path in (self.root / "cache").glob("*.json"):
            self.assertNotIn("secret-not-logged", path.read_text())
            self.assertNotIn("data:image", path.read_text())

    def test_errors_and_empty_answers_are_not_cached(self):
        evaluator = self.evaluator()
        with patch.object(evaluator, "_post", side_effect=VideoApiError("timeout")):
            with self.assertRaises(VideoApiError):
                evaluator.evaluate(self.task, self.artifact)
        with patch.object(evaluator, "_post", return_value={"choices": [{"message": {"content": "{}"}}]}):
            with self.assertRaisesRegex(VideoApiError, "no usable"):
                evaluator.evaluate(self.task, self.artifact)
        self.assertEqual(list((self.root / "cache").glob("*.json")), [])

    def test_concurrent_requests_share_one_completed_result(self):
        cache = VerifierCache(self.root / "cache")
        score = Mock(return_value={"score": .5})
        def run(_):
            return cache.evaluate("endpoint", {"model": "judge"}, [str(self.video)], score)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(run, range(2)))
        self.assertEqual(score.call_count, 1)
        self.assertEqual(sorted(result["verifier_cache"]["hit"] for result in results), [False, True])

    def test_remote_media_bypass_cache_and_corruption_is_recoverable(self):
        cache = VerifierCache(self.root / "cache")
        score = Mock(side_effect=lambda: {"score": .5})
        for _ in range(2):
            result = cache.evaluate("endpoint", {}, ["https://example.test/video.mp4"], score)
            self.assertFalse(result["verifier_cache"]["enabled"])
        result = cache.evaluate("endpoint", {}, [str(self.video)], score)
        (self.root / "cache" / f"{result['verifier_cache']['key']}.json").write_text("broken")
        self.assertFalse(cache.evaluate("endpoint", {}, [str(self.video)], score)["verifier_cache"]["hit"])
        self.assertEqual(score.call_count, 4)
