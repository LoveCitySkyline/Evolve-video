from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from evovideo_skill.graph_skill import GraphNode, GraphSkillMemory, ToolPathGraph
from evovideo_skill.harness import GraphHarnessRunner, HarnessConfig, summarize_harness
from evovideo_skill.program_registry import GraphProgram, ProgramMetrics, ProgramRegistry
from evovideo_skill.runtime import RuntimeSettings


class HarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.output_dir = Path("outputs/test_harness_runner")
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_harness_runs_ablation_and_writes_reports(self) -> None:
        config = HarnessConfig(
            name="test_harness",
            output_dir=str(self.output_dir),
            task_files=["benchmarks/video_skill_graph_core.json"],
            limit_per_suite=2,
            max_candidates=4,
            max_mutation_searches=1,
            min_quality_gain=0.01,
            evaluation_seeds=[11, 22],
            min_utility=0.0,
            online_foundation_min_support=1,
            online_foundation_min_advantage=0.0,
            online_foundation_min_stability=0.0,
            full_suite_evaluation=True,
            ablations=["full", "no_foundation"],
            runtime=RuntimeSettings(
                provider="local-fake",
                video_output_dir=str(self.output_dir / "videos"),
            ),
        )

        report = GraphHarnessRunner(config).run()
        summary = summarize_harness(self.output_dir)

        self.assertEqual(len(report.records), 2)
        self.assertTrue((self.output_dir / "harness_report.json").exists())
        self.assertTrue((self.output_dir / "harness_summary.csv").exists())
        self.assertTrue((self.output_dir / "summary.json").exists())
        run_dir = self.output_dir / "video_skill_graph_core_full"
        self.assertTrue((self.output_dir / ".harness.lock").exists())
        self.assertTrue((self.output_dir / "videos" / ".evovideo-runtime.lock").exists())
        self.assertTrue((run_dir / "agent_state").is_dir())
        self.assertTrue((run_dir / "evolution" / "registry" / "frontier.json").exists())
        self.assertTrue((run_dir / "evolution" / "loop_checkpoint.json").exists())
        self.assertTrue((run_dir / "evolution" / "feedback_history.jsonl").exists())
        self.assertTrue((run_dir / "evolution" / "loop_result.json").exists())
        self.assertTrue((run_dir / "evolution" / "weighted_tool_graph.json").exists())
        self.assertTrue((run_dir / "evolution" / "online_foundations.jsonl").exists())
        visualization = run_dir / "evolution" / "graph_visualization"
        self.assertTrue((visualization / "index.html").exists())
        self.assertTrue((visualization / "evolution_graph.json").exists())
        self.assertTrue((visualization / "evolution_graph.graphml").exists())
        self.assertTrue((visualization / "evolution_graph.dot").exists())
        self.assertTrue((visualization / "manifest.jsonl").exists())
        self.assertTrue((visualization / "executions.jsonl").exists())
        self.assertTrue((visualization / "candidate_audits.jsonl").exists())
        payload = json.loads((visualization / "evolution_graph.json").read_text())
        self.assertGreaterEqual(len(payload["snapshots"]), 2)
        self.assertGreater(len(payload["executions"]), 0)
        self.assertIn("weighted_task_graphs", payload)
        self.assertIn("candidate_audits", payload)
        self.assertGreater(len(payload["candidate_audits"]), 0)
        run_record = json.loads((run_dir / "run_record.json").read_text())
        loop_result = json.loads((run_dir / "evolution" / "loop_result.json").read_text())
        checkpoint = json.loads((run_dir / "evolution" / "loop_checkpoint.json").read_text())
        self.assertLessEqual(loop_result["mutation_searches_used"], 1)
        self.assertEqual(loop_result["max_mutation_searches"], 1)
        self.assertEqual(checkpoint["mutation_searches_used"], loop_result["mutation_searches_used"])
        self.assertEqual(run_record["full_suite_task_count"], 2)
        self.assertIsNotNone(run_record["full_suite_baseline_score"])
        self.assertIsNotNone(run_record["full_suite_final_score"])
        validation_seeds = {
            item["evaluation_seed"]
            for item in payload["executions"]
            if item["evaluation_seed"] is not None
        }
        self.assertEqual(validation_seeds, {11, 22})
        exploration_records = [
            item for item in payload["executions"]
            if not item["cache_hit"] and item["task_id"] in {"core-source-video-to-anime-style", "core-region-car-edit-preserve-background"}
        ]
        self.assertTrue(all(item["evaluation_seed"] in {11, 22} for item in exploration_records))
        self.assertEqual(summary["records"], 2)
        self.assertGreaterEqual(
            summary["total_accepted_candidates"] + summary["total_rejected_candidates"],
            1,
        )

    def test_imports_nested_benchmark_frontier_for_warm_start(self) -> None:
        source = self.output_dir / "small" / "small_suite_full"
        source_memory = GraphSkillMemory(source / "memory")
        graph = ToolPathGraph(
            graph_id="accepted_temporal_path",
            skill_name="accepted_temporal_path",
            description="validated on the nested smaller benchmark",
            triggers=["long_horizon_causal"],
            nodes=[GraphNode("plan", "tool", "temporal_decomposer")],
            edges=[],
            stats={"accepted": True, "validated_task_classes": ["long_horizon_causal"]},
        )
        source_memory.upsert_graph(graph)
        registry = ProgramRegistry(source / "evolution" / "registry")
        program = GraphProgram(
            name="small-frontier",
            graph_ids=["baseline_t2v_graph", graph.skill_name],
            metrics=ProgramMetrics(quality=0.7, pass_rate=0.1, stability=1.0),
        )
        registry.create(program)
        registry.update_frontier(program.name, 2)
        target_memory = GraphSkillMemory(self.output_dir / "large" / "memory")
        runner = GraphHarnessRunner(HarnessConfig(
            output_dir=str(self.output_dir / "large"),
            warm_start_run_dir=str(source),
            warm_start_required=True,
        ))

        imported = runner._import_warm_start_frontier(target_memory)

        self.assertEqual(imported, [graph.skill_name])
        self.assertEqual(
            [item.skill_name for item in target_memory.list_graphs()],
            [graph.skill_name],
        )


if __name__ == "__main__":
    unittest.main()
