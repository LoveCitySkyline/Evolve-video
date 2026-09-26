from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evovideo_skill.graph_visualization import EvolutionGraphArchive
from evovideo_skill.models import VideoTask


class ComparisonExportTests(unittest.TestCase):
    def test_exports_best_paired_baseline_and_optimized_videos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.mp4"
            weaker = root / "weaker.mp4"
            stronger = root / "stronger.mp4"
            baseline.write_bytes(b"baseline")
            weaker.write_bytes(b"weaker")
            stronger.write_bytes(b"stronger")
            archive = EvolutionGraphArchive(root / "archive")
            common = {
                "task_id": "task-1",
                "passed": False,
                "estimated_cost": 1.0,
                "cache_hit": False,
                "failure_types": [],
                "active_metric_names": ["reward:style_alignment"],
                "evaluation_seed": 42,
                "seed_controlled": True,
            }
            archive.record_execution(
                **common,
                graph_id="baseline_t2v_graph",
                score=0.4,
                tool_chain=["mock_text_to_video"],
                metric_scores={"reward:style_alignment": 0.2},
                artifact_path=str(baseline),
            )
            archive.record_execution(
                **common,
                graph_id="candidate-a",
                score=0.6,
                tool_chain=["task_reference_video", "style-a"],
                metric_scores={"reward:style_alignment": 0.6},
                artifact_path=str(weaker),
            )
            archive.record_execution(
                **common,
                graph_id="candidate-b",
                score=0.8,
                tool_chain=["task_reference_video", "style-b"],
                metric_scores={"reward:style_alignment": 0.9},
                artifact_path=str(stronger),
            )

            paths = archive.export_video_comparisons(
                [VideoTask("task-1", "Stylize the source")],
                top_k=1,
            )

            manifest = json.loads(Path(paths["comparison_manifest"]).read_text())
            self.assertEqual(manifest["pair_count"], 1)
            self.assertAlmostEqual(manifest["pairs"][0]["reward_gain"], 0.4)
            self.assertEqual(manifest["pairs"][0]["optimized"]["graph_id"], "candidate-b")
            pair_dir = Path(paths["comparison_directory"]) / "01_task-1"
            self.assertEqual((pair_dir / "wan_baseline.mp4").read_bytes(), b"baseline")
            self.assertEqual((pair_dir / "optimized.mp4").read_bytes(), b"stronger")


if __name__ == "__main__":
    unittest.main()
