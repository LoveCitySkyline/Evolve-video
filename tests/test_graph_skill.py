from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from evovideo_skill.evaluators import EvaluatorSuite
from evovideo_skill.evolution_loop import GraphSelfImprovingLoop
from evovideo_skill.foundation_skills import FoundationSkillConsolidator
from evovideo_skill.graph_algorithms import (
    approximate_graph_edit_distance,
    betweenness_centrality,
    k_shortest_simple_paths,
    pagerank,
    wl_kernel_similarity,
)
from evovideo_skill.graph_evolver import GraphPathCandidate, GraphPathValidationReport, GraphToolPathEvolver
from evovideo_skill.graph_mining import GraphMotifMiner
from evovideo_skill.graph_router import FoundationAwareGraphRouter
from evovideo_skill.graph_skill import BoundedGraphEdit, ExperienceRecord, GraphNode, GraphSkillMemory, ToolPathGraph
from evovideo_skill.models import VideoTask
from evovideo_skill.skill_memory import SkillMemory


class GraphSkillTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory_dir = Path("outputs/test_graph_skill_memory")
        shutil.rmtree(self.memory_dir, ignore_errors=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.memory_dir, ignore_errors=True)

    def test_bounded_graph_edit_replaces_node_and_adds_validator(self) -> None:
        graph = GraphToolPathEvolver.baseline_graph()
        graph = graph.apply_edit(
            BoundedGraphEdit(
                "replace_node",
                target="tool_t2v",
                payload={"node_id": "tool_i2v", "node_type": "tool", "name": "mock_image_to_video", "config": {"cost": 1.5}},
            )
        )
        graph = graph.apply_edit(BoundedGraphEdit("add_validator", payload={"validator": "identity_consistency"}))

        self.assertEqual(graph.executable_tool_names(), ["mock_image_to_video"])
        self.assertIn("identity_consistency", graph.validators)
        self.assertEqual(graph.node("tool_i2v").name, "mock_image_to_video")

    def test_pareto_frontier_removes_dominated_graph_paths(self) -> None:
        reports = [
            GraphPathValidationReport("cheap_good", ["a"], 0.7, 0.9, 0.2, 1.0, 0.95, True, ["mock_image_to_video"], []),
            GraphPathValidationReport("expensive_worse", ["a"], 0.7, 0.85, 0.15, 2.0, 0.9, True, ["mock_multi_shot_i2v"], []),
            GraphPathValidationReport("rejected", ["a"], 0.7, 0.95, 0.25, 0.8, 0.9, False, ["mock_video_style_transfer"], []),
        ]

        selected = GraphToolPathEvolver.select_pareto_frontier(reports)

        self.assertEqual([report.skill_name for report in selected], ["cheap_good"])

    def test_versioned_candidate_does_not_inherit_parent_acceptance(self) -> None:
        graph = GraphToolPathEvolver.baseline_graph()
        graph.stats.update(
            {
                "accepted": True,
                "quality_gain": 0.2,
                "pass_rate": 1.0,
                "passed_validation_task_ids": ["old-task"],
            }
        )
        candidate = GraphPathCandidate(graph, [], "mutation", [])

        versioned = GraphSelfImprovingLoop._version_candidate(candidate, 2, 1)

        self.assertNotIn("accepted", versioned.graph.stats)
        self.assertNotIn("quality_gain", versioned.graph.stats)
        self.assertNotIn("passed_validation_task_ids", versioned.graph.stats)

    def test_foundation_mining_requires_accepted_rollout_evidence(self) -> None:
        graph_memory = GraphSkillMemory(self.memory_dir)
        skill_memory = SkillMemory(self.memory_dir)
        accepted = ToolPathGraph(
            graph_id="accepted",
            skill_name="accepted",
            description="accepted",
            triggers=["identity"],
            nodes=[GraphNode("a", "tool", "temporal_decomposer", {"cost": 0.2})],
            edges=[],
            stats={
                "accepted": True,
                "pass_rate": 1.0,
                "quality_gain": 0.1,
                "candidate_score": 0.9,
                "passed_validation_task_ids": ["task-a", "task-b"],
            },
        )
        inherited_but_rejected = ToolPathGraph(
            graph_id="rejected",
            skill_name="rejected",
            description="rejected",
            triggers=["identity"],
            nodes=[GraphNode("b", "tool", "bridge_extract_reference_frame", {"cost": 0.2})],
            edges=[],
            stats={
                "accepted": True,
                "pass_rate": 1.0,
                "quality_gain": 0.4,
                "candidate_score": 0.95,
                "passed_validation_task_ids": ["task-a", "task-b"],
            },
        )
        graph_memory.upsert_graph(accepted)
        graph_memory.upsert_graph(inherited_but_rejected)
        for task_id in ("task-a", "task-b"):
            graph_memory.append_experience(
                ExperienceRecord(task_id, "prompt", [], "accepted", ["temporal_decomposer"], 0.9, 0.2, True)
            )

        report = FoundationSkillConsolidator(
            graph_memory,
            skill_memory,
            miner=GraphMotifMiner(min_support=2, min_utility=-1.0),
        ).consolidate()

        selected_tools = {tool for motif in report.selected_motifs for tool in motif.tools}
        self.assertIn("temporal_decomposer", selected_tools)
        self.assertNotIn("bridge_extract_reference_frame", selected_tools)

    def test_graph_evolution_discovers_and_persists_tool_path_skill(self) -> None:
        task = VideoTask(
            "graph-test-red-coat",
            "The same woman in a red coat walks, then turns, then waves while keeping the same face and red coat.",
        )
        evolver = GraphToolPathEvolver(
            skill_memory=SkillMemory(self.memory_dir),
            graph_memory=GraphSkillMemory(self.memory_dir),
            evaluators=EvaluatorSuite(identity_threshold=0.95, clothing_threshold=0.95, action_threshold=0.95, inclusive_threshold=False),
            min_quality_gain=0.01,
        )

        report = evolver.evolve([task], max_candidates=4)

        self.assertTrue(report.candidates)
        self.assertTrue(report.selected_graphs)
        self.assertTrue((self.memory_dir / "graph_skills" / "identity_reference_i2v_graph.json").exists())
        self.assertTrue((self.memory_dir / "identity_reference_i2v_graph.json").exists())
        self.assertTrue((self.memory_dir / "skills" / "identity_reference_i2v_graph" / "SKILL.md").exists())
        experiences = GraphSkillMemory(self.memory_dir).list_experiences()
        self.assertTrue(any(record.accepted for record in experiences))

    def test_foundation_skill_consolidation_and_routing(self) -> None:
        tasks = [
            VideoTask("graph-test-person", "The same woman in a red coat walks, then turns, then waves while keeping the same face."),
            VideoTask("graph-test-action", "A chef places a tomato on a board, slices it, pushes it into a pan, and stirs in the correct order."),
        ]
        graph_memory = GraphSkillMemory(self.memory_dir)
        skill_memory = SkillMemory(self.memory_dir)
        evolver = GraphToolPathEvolver(
            skill_memory=skill_memory,
            graph_memory=graph_memory,
            evaluators=EvaluatorSuite(identity_threshold=0.95, clothing_threshold=0.95, action_threshold=0.95, inclusive_threshold=False),
            min_quality_gain=0.01,
        )
        evolver.evolve(tasks, max_candidates=6)
        consolidator = FoundationSkillConsolidator(
            graph_memory=graph_memory,
            skill_memory=skill_memory,
            miner=GraphMotifMiner(min_support=2, min_utility=0.0),
        )

        report = consolidator.consolidate()
        routes = FoundationAwareGraphRouter(graph_memory).route(
            VideoTask("unseen", "The same character keeps the same face across shots and performs ordered actions."),
            failure_types=["identity_drift", "motion_mismatch"],
        )

        self.assertTrue(report.foundation_skills)
        self.assertTrue((self.memory_dir / "graph_skills" / "foundation" / "foundation_report.json").exists())
        self.assertTrue(any(route.metadata.get("foundation_skill") for route in routes))
        self.assertTrue(any(skill.skill_name.startswith("foundation_") for skill in skill_memory.list_skills()))

    def test_graph_algorithms_on_global_tool_graph(self) -> None:
        tasks = [
            VideoTask("graph-test-person", "The same woman in a red coat walks, then turns, then waves while keeping the same face."),
            VideoTask("graph-test-action", "A chef places a tomato on a board, slices it, pushes it into a pan, and stirs in the correct order."),
        ]
        graph_memory = GraphSkillMemory(self.memory_dir)
        evolver = GraphToolPathEvolver(
            skill_memory=SkillMemory(self.memory_dir),
            graph_memory=graph_memory,
            evaluators=EvaluatorSuite(identity_threshold=0.95, clothing_threshold=0.95, action_threshold=0.95, inclusive_threshold=False),
            min_quality_gain=0.01,
        )
        evolver.evolve(tasks, max_candidates=6)
        graphs = graph_memory.list_graphs()
        global_graph = GraphMotifMiner(min_support=1, min_utility=-1.0).build_global_tool_graph(graphs, graph_memory.list_experiences())

        rank = pagerank(global_graph)
        between = betweenness_centrality(global_graph)
        paths = k_shortest_simple_paths(global_graph, "mock_text_to_video", "mock_image_to_video", k=2)
        similarity = wl_kernel_similarity(graphs[0], graphs[0])
        distance = approximate_graph_edit_distance(graphs[0], graphs[0])

        self.assertTrue(rank)
        self.assertTrue(between)
        self.assertTrue(paths)
        self.assertAlmostEqual(similarity, 1.0)
        self.assertEqual(distance, 0.0)


if __name__ == "__main__":
    unittest.main()
