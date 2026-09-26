from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evovideo_skill.conditioning_calibration import calibration_metrics, validate_items


class CalibrationTests(unittest.TestCase):
    def test_calibration_coverage_and_rank_account_for_missing(self):
        items = [{"task": {"task_id": "a"}, "human_scores": {"identity": x}} for x in (.9, .5)]
        predictions = [{"evaluation_status": "complete", "criterion_scores": {"identity": .8}},
                       {"evaluation_status": "needs_review"}]
        result = calibration_metrics(items, predictions)
        self.assertEqual(result["coverage"], .5)
        self.assertEqual(result["pairwise_agreement"], 0)
        self.assertAlmostEqual(result["task_balanced_mae"], .1)
        predictions[1] = {"evaluation_status": "complete", "criterion_scores": {"identity": .4}}
        result = calibration_metrics(items, predictions)
        self.assertEqual(result["pairwise_agreement"], 1)
        self.assertEqual(result["coverage"], 1)

    def test_no_test_selection_or_duplicate_videos(self):
        with TemporaryDirectory() as directory:
            video = Path(directory) / "x.mp4"
            video.write_bytes(b"fixture")
            item = {"split": "calibration", "video_path": str(video), "human_scores": {"identity": .8},
                    "task": {"task_id": "cal", "prompt": "A scene", "metadata": {"evaluation": {"identity": {}}}}}
            validate_items([item])
            with self.assertRaises(ValueError):
                validate_items([item, item])
            item["split"] = "test"
            with self.assertRaisesRegex(ValueError, "cannot use"):
                validate_items([item])


if __name__ == "__main__":
    unittest.main()
