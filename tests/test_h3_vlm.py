import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evovideo_skill.models import VideoArtifact, VideoTask
from evovideo_skill.vlm_evaluator import QwenVLEvaluator


class H3VLMTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.candidates = self.frames("candidate", 9)
        self.task = VideoTask("h3", "Follow the original reference identity, style and motion.")
        self.artifact = VideoArtifact(
            "output", self.task.task_id, "candidate rewritten prompt", self.task.mode,
            [], [], {"sampled_frame_paths": self.candidates},
        )

    def frames(self, prefix, count):
        paths = []
        for index in range(count):
            path = self.root / f"{prefix}-{index}.jpg"
            path.write_bytes(f"{prefix}-{index}".encode())
            paths.append(str(path))
        return paths

    @staticmethod
    def reference(identifier, kind="image", uri=None, role=None, semantic_role="identity"):
        return {
            "id": identifier, "kind": kind,
            "uri": uri or f"https://example.com/{identifier}.jpg",
            "role": role or f"reference_{kind}",
            "semantic_role": semantic_role, "duration_seconds": 6 if kind != "image" else None,
        }

    def evaluate(self, max_images=6, response=None, evaluator=None):
        evaluator = evaluator or QwenVLEvaluator(api_key="test", max_images=max_images)
        raw = {"choices": [{"message": {"content": json.dumps({"action_alignment_score": .5, **(response or {})})}}]}
        with patch.object(evaluator, "_post", return_value=raw) as post:
            result = evaluator.evaluate(self.task, self.artifact)
        return result, post.call_args.args[1]["messages"][0]["content"]

    @staticmethod
    def urls(content):
        return [item["image_url"]["url"] for item in content if item["type"] == "image_url"]

    @staticmethod
    def labels(content):
        return "\n".join(item["text"] for item in content if item["type"] == "text")

    def test_fixed_images_labeled_and_candidate_reference_overrides_ignored(self):
        local = self.frames("original", 1)[0]
        refs = [
            self.reference("start", uri=Path(local).as_uri(), role="first_frame"),
            self.reference("end", role="last_frame"),
            self.reference("style", semantic_role="style"),
        ]
        self.task.metadata["h3_references"] = refs
        before = copy.deepcopy(refs)
        self.artifact.metadata.update({
            "h3_references": [self.reference("candidate-chosen")],
            "selected_references": [self.reference("also-candidate-chosen")],
            "source_sampled_frame_paths": self.frames("fake-source", 2),
        })
        result, content = self.evaluate(max_images=5)
        self.assertEqual(self.urls(content), [
            QwenVLEvaluator._image_data_url(local), refs[1]["uri"], refs[2]["uri"],
            QwenVLEvaluator._image_data_url(self.candidates[0]),
            QwenVLEvaluator._image_data_url(self.candidates[-1]),
        ])
        labels = self.labels(content)
        for ref in refs:
            self.assertIn(json.dumps(ref), labels)
        self.assertNotIn("candidate-chosen", labels)
        self.assertNotIn("candidate rewritten prompt", labels)
        self.assertNotIn("REFERENCE SOURCE FRAMES", labels)
        self.assertIn("preservation sources", labels)
        self.assertEqual(result["source_frame_count"], 0)
        self.assertEqual(result["verification_metadata"]["reference_ids_observed"], ["start", "end", "style"])
        self.assertEqual(refs, before)

    def test_overflow_reports_omitted_references_and_reserves_candidate(self):
        self.task.metadata["h3_references"] = [self.reference(f"ref-{i}") for i in range(5)]
        for budget in range(1, 7):
            with self.subTest(budget=budget):
                result, content = self.evaluate(max_images=budget)
                verification = result["verification_metadata"]
                count = min(5, budget - 1)
                self.assertEqual(len(self.urls(content)), budget)
                self.assertGreaterEqual(result["candidate_frame_count"], 1)
                self.assertEqual(verification["reference_ids_observed"], [f"ref-{i}" for i in range(count)])
                self.assertEqual(verification["reference_ids_omitted"], [f"ref-{i}" for i in range(count, 5)])
                for ref in verification["references"][count:]:
                    self.assertEqual(ref["omission_reason"], "max_images budget exhausted")
                self.assertIn("Criteria requiring omitted references", self.labels(content))

    def test_candidate_and_legacy_source_sampling_covers_whole_timelines(self):
        sources = self.frames("source", 9)
        self.artifact.metadata["source_sampled_frame_paths"] = sources
        result, content = self.evaluate(max_images=4)
        self.assertEqual(self.urls(content), [
            QwenVLEvaluator._image_data_url(path)
            for path in [sources[0], sources[-1], self.candidates[0], self.candidates[-1]]
        ])
        self.assertEqual(result["source_frame_count"], 2)
        result, content = self.evaluate(max_images=1)
        self.assertEqual(self.urls(content), [QwenVLEvaluator._image_data_url(self.candidates[4])])
        self.assertEqual(result["source_frame_count"], 0)

    def test_candidate_only_sampling_and_frame_metadata_fallback(self):
        self.artifact.metadata.clear()
        self.artifact.frames = [{"frame_path": path} for path in self.candidates]
        _, content = self.evaluate(max_images=3)
        self.assertEqual(self.urls(content), [
            QwenVLEvaluator._image_data_url(self.candidates[i]) for i in (0, 4, 8)
        ])

    def test_video_sampling_cached_and_motion_reference_is_not_source(self):
        video = self.root / "motion.mp4"
        video.write_bytes(b"video")
        frames = self.frames("motion", 9)
        self.task.metadata["h3_references"] = [
            self.reference("motion", kind="video", uri=str(video), semantic_role="motion"),
        ]
        self.task.reference_video = video.resolve().as_uri()
        evaluator = QwenVLEvaluator(api_key="test", max_images=5)
        with patch("evovideo_skill.vlm_evaluator.VideoProcessor") as processor:
            processor.return_value.process.return_value = SimpleNamespace(sampled_frame_paths=frames)
            first, content = self.evaluate(evaluator=evaluator)
            second, again = self.evaluate(evaluator=evaluator)
        processor.assert_called_once()
        self.assertGreaterEqual(processor.call_args.kwargs["sample_count"], 2)
        self.assertEqual(processor.return_value.process.call_args.args[0], video.resolve().as_uri())
        self.assertEqual(self.urls(content), [
            QwenVLEvaluator._image_data_url(path)
            for path in [frames[0], frames[-1], self.candidates[0], self.candidates[4], self.candidates[-1]]
        ])
        self.assertEqual(content, again)
        self.assertEqual(first["verification_metadata"], second["verification_metadata"])
        self.assertEqual(first["source_frame_count"], 0)
        self.assertNotIn("REFERENCE SOURCE FRAMES", self.labels(content))

    def test_distinct_original_source_ignores_artifact_supplied_source(self):
        self.task.metadata["h3_references"] = [self.reference("style", semantic_role="style")]
        self.task.reference_video = "https://example.com/original.mp4"
        sources = self.frames("original-source", 9)
        self.artifact.metadata["source_sampled_frame_paths"] = self.frames("candidate-source", 3)
        with patch("evovideo_skill.vlm_evaluator.VideoProcessor") as processor:
            processor.return_value.process.return_value = SimpleNamespace(sampled_frame_paths=sources)
            result, content = self.evaluate(max_images=5)
        self.assertEqual(processor.return_value.process.call_args.args[0], self.task.reference_video)
        self.assertEqual(result["source_frame_count"], 2)
        self.assertEqual(self.urls(content)[1:3], [QwenVLEvaluator._image_data_url(path) for path in (sources[0], sources[-1])])

    def test_unavailable_visual_references_do_not_consume_budget(self):
        self.task.metadata["h3_references"] = [
            self.reference("missing", uri=str(self.root / "missing.png")),
            self.reference("broken", kind="video", uri="https://example.com/broken.mp4"),
            self.reference("available"),
        ]
        with patch("evovideo_skill.vlm_evaluator.VideoProcessor") as processor:
            processor.return_value.process.side_effect = RuntimeError("decode failed")
            result, content = self.evaluate(max_images=3)
        verification = result["verification_metadata"]
        self.assertEqual(verification["reference_ids_observed"], ["available"])
        self.assertEqual(verification["reference_ids_omitted"], ["missing", "broken"])
        self.assertEqual(len(self.urls(content)), 3)
        self.assertTrue(all(ref["omission_reason"] for ref in verification["references"][:2]))

    def test_audio_metadata_is_explicitly_unavailable_and_cannot_override_manifest(self):
        self.task.metadata["h3_references"] = [self.reference("sound", kind="audio", uri="https://example.com/audio.wav")]
        self.task.metadata["evaluation"] = {"audio_sync": {"weight": 1}}
        response = {"verification_metadata": {"audio_evidence_available": True, "reference_ids_observed": ["sound"]}}
        result, content = self.evaluate(max_images=2, response=response)
        verification = result["verification_metadata"]
        self.assertFalse(verification["audio_evidence_available"])
        self.assertEqual(verification["reference_ids_observed"], [])
        self.assertEqual(verification["reference_ids_omitted"], ["sound"])
        self.assertEqual({item["type"] for item in content}, {"text", "image_url"})
        self.assertEqual(result["criterion_scores"]["audio_sync"], 0.0)
        self.assertIn("audio criteria require an external verifier", self.labels(content))
        self.assertIn("audio metadata are not audio evidence", self.labels(content))
        self.assertEqual(result["candidate_frame_count"], 2)

    def test_invalid_image_budget_rejected(self):
        for budget in (0, -1):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                QwenVLEvaluator(api_key="test", max_images=budget)

    def test_video_cache_is_resampled_when_cached_frames_disappear(self):
        self.task.metadata["h3_references"] = [
            self.reference("motion", kind="video", uri="https://example.com/motion.mp4"),
        ]
        old = self.frames("old", 2)
        new = self.frames("new", 2)
        evaluator = QwenVLEvaluator(api_key="test", max_images=4)
        with patch("evovideo_skill.vlm_evaluator.VideoProcessor") as processor:
            processor.return_value.process.side_effect = [
                SimpleNamespace(sampled_frame_paths=old), SimpleNamespace(sampled_frame_paths=new),
            ]
            self.evaluate(evaluator=evaluator)
            Path(old[0]).unlink()
            _, content = self.evaluate(evaluator=evaluator)
        self.assertEqual(processor.return_value.process.call_count, 2)
        self.assertEqual(self.urls(content)[:2], [QwenVLEvaluator._image_data_url(path) for path in new])

    def test_budget_omitted_video_is_not_processed(self):
        self.task.metadata["h3_references"] = [
            self.reference("image"), self.reference("motion", kind="video"),
        ]
        with patch("evovideo_skill.vlm_evaluator.VideoProcessor") as processor:
            result, _ = self.evaluate(max_images=2)
        processor.assert_not_called()
        self.assertEqual(result["verification_metadata"]["reference_ids_omitted"], ["motion"])


if __name__ == "__main__":
    unittest.main()
