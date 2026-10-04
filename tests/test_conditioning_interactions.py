from copy import deepcopy
from dataclasses import replace
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.conditioning_interactions import (condition_only, decode_experiment,
    effect_supported, interaction_effect, merge_factors)
from evovideo_skill.conditioning_memory import StrategyMemory
from evovideo_skill.conditioning_planner import ConditioningSmokePlanner
from evovideo_skill.conditioning_runner import (ConditioningRunner, FixedTaskPlanner, validate_config,
                                               EvidenceIncomplete, MeasurementUnavailable)
from evovideo_skill.benchmarks import BenchmarkSuite
from evovideo_skill.evolution_data import stratified_task_split
from evovideo_skill.graph_evolver import GraphToolPathEvolver
from evovideo_skill.graph_skill import GraphSkillMemory
from evovideo_skill.research_protocol import ResearchBudgetExceeded, graph_payload
from evovideo_skill.skill_memory import SkillMemory


ROOT = Path(__file__).resolve().parents[1]


def record(score, seed, key, action=None):
    return {"task_id": "task", "seed": seed, "score": score, "evaluation_id": key,
            "criterion_scores": {"identity": score, "action": score if action is None else action}}


class FactorTests(unittest.TestCase):
    def setUp(self):
        self.anchor = GraphToolPathEvolver.baseline_graph()
        self.a, self.b = deepcopy(self.anchor), deepcopy(self.anchor)
        self.a.nodes[-1].config["reference_ids"] = ["identity"]
        self.b.nodes[-1].config["shot_index"] = 1

    def test_field_merge_preserves_anchor(self):
        original = graph_payload(self.anchor)
        merged = merge_factors(self.anchor, self.a, self.b)
        self.assertEqual(merged.nodes[-1].config["reference_ids"], ["identity"])
        self.assertEqual(merged.nodes[-1].config["shot_index"], 1)
        self.assertEqual(graph_payload(self.anchor), original)
        condition_only(self.anchor, merged)

    def test_shared_list_edits_are_not_independent(self):
        self.b.nodes[-1].config["reference_ids"] = ["motion"]
        with self.assertRaisesRegex(ValueError, "overlapping"):
            merge_factors(self.anchor, self.a, self.b)

    def test_delete_modify_conflict(self):
        self.b.nodes.pop()
        with self.assertRaisesRegex(ValueError, "overlapping"):
            merge_factors(self.anchor, self.a, self.b)

    def test_prompt_rewriting_is_not_a_condition_intervention(self):
        self.a.nodes[-1].config["prompt"] = "Do something different"
        with self.assertRaisesRegex(ValueError, "non-conditioning"):
            condition_only(self.anchor, self.a)

    def test_reject_text_strategy_on_existing_and_new_nodes_and_preserve_cost(self):
        for key in ("conditioning_strategy", "prompt_task_hashes"):
            existing = deepcopy(self.anchor)
            existing.nodes[-1].config[key] = "ordered_actions"
            with self.assertRaisesRegex(ValueError, "non-conditioning"):
                condition_only(self.anchor, existing)
            added = deepcopy(self.anchor)
            node = deepcopy(added.nodes[-1])
            node.node_id = "additional"
            node.config[key] = "ordered_actions"
            added.nodes.append(node)
            with self.assertRaisesRegex(ValueError, "new conditioning nodes"):
                condition_only(self.anchor, added)
        candidate = deepcopy(self.anchor)
        candidate.nodes[-1].config.pop("cost")
        with self.assertRaisesRegex(ValueError, "non-conditioning field changed: cost"):
            condition_only(self.anchor, candidate)

    def test_interaction_and_negative_dimension(self):
        cells = {k: [record(score, s, f"{k}/{s}", action) for s in (1, 2, 3)] for k, score, action in (
            ("anchor", .4, .8), ("a", .5, .8), ("b", .5, .8), ("joint", .8, .6))}
        result = interaction_effect(cells)
        self.assertAlmostEqual(result["quality"]["mean"], .2)
        self.assertAlmostEqual(result["metrics"]["action"]["mean"], -.2)
        self.assertEqual(result["quality"]["standard_error"], 0)
        cfg = {"selection_min_gain": .01, "max_metric_regression": .05}
        self.assertFalse(effect_supported(result["marginal_effects"]["joint"], cfg))

    def test_missing_or_unpaired_cells_cannot_fake_interaction(self):
        cells = {k: [record(.5, s, f"{k}/{s}") for s in (1, 2, 3)] for k in ("anchor", "a", "b", "joint")}
        with self.assertRaises(ValueError):
            interaction_effect({k: v for k, v in cells.items() if k != "joint"})
        cells["a"][0]["seed"] = 9
        with self.assertRaisesRegex(ValueError, "unpaired"):
            interaction_effect(cells)
        cells["a"][0]["seed"] = 1
        del cells["a"][0]["criterion_scores"]["action"]
        with self.assertRaises(ValueError):
            interaction_effect(cells)


class InteractionRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = json.loads((ROOT / "configs/conditioning_interactions_smoke.json").read_text())
        validate_config(self.config)
        self.dataset = stratified_task_split(BenchmarkSuite.from_file(ROOT / self.config["task_file"]).tasks)
        self.ev = GraphToolPathEvolver(SkillMemory(self.root / "skills"), GraphSkillMemory(self.root / "graphs"), planner=FixedTaskPlanner())
        self.runner = ConditioningRunner(self.dataset, self.ev, ConditioningSmokePlanner(), self.root / "run", self.config, {"provider": "local-fake"})
        self.stdout = redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def test_four_cells_resume_and_frozen_effects(self):
        self.runner.learn()
        reports = [json.loads(p.read_text()) for p in (self.runner.root / "interactions").glob("*.json")]
        self.assertEqual(len(reports), 2)
        self.assertTrue(all(set(r["cells"]) == {"anchor", "a", "b", "joint"} for r in reports))
        self.assertTrue(all(r["selected_cell"] == "parent" for r in reports))
        joint = [StrategyMemory.view(e) for e in self.runner.memory.entries.values()
                 if e["strategy"]["name"] == "joint_scope_binding"]
        self.assertTrue(joint[0]["evidence"]["conditioning_interactions"])
        with patch.object(self.ev, "rollout", side_effect=AssertionError("replay")):
            resumed = ConditioningRunner(self.dataset, self.ev, ConditioningSmokePlanner(), self.runner.root, self.config, {"provider": "local-fake"})
            resumed.learn()
        frozen = self.runner.validate_and_freeze()
        original = frozen.read_bytes()
        self.assertEqual(self.runner.test(frozen)["heldout_gain"], 0)
        self.assertEqual(frozen.read_bytes(), original)

    def test_single_factor_control_does_not_execute_joint(self):
        self.runner.config["search_mode"] = "single"
        self.runner.learn()
        report = json.loads((self.runner.root / "interactions/0000.json").read_text())
        self.assertIsNone(report["interaction"])
        self.assertNotIn("joint", report["cells"])

    def test_quality_fixture_selects_joint_but_not_hidden_reference(self):
        original = self.ev.rollout
        def execute(task, graph, **kwargs):
            result = original(task, graph, **kwargs)
            config = graph.nodes[-1].config
            a, b = "duration_seconds" in config, "reference_ids" in config
            score = .8 if a and b else .5 if a or b else .4
            result.reward = replace(result.reward, score=score)
            return result
        with patch.object(self.ev, "rollout", side_effect=execute):
            self.runner.learn()
        report = json.loads((self.runner.root / "interactions/0000.json").read_text())
        self.assertEqual(report["selected_cell"], "joint")
        self.assertAlmostEqual(report["interaction"]["quality"]["mean"], .2)

    def test_budget_exhaustion_does_not_become_negative_quality(self):
        self.runner.ledger.max_calls = 1
        with self.assertRaises(ResearchBudgetExceeded):
            self.runner.learn()
        self.assertFalse(self.runner.state["learned"])
        self.assertEqual(self.runner.memory.entries, {})

    def test_project_evidence_reaches_planner_and_replay_does_not_create_attempt(self):
        task = self.dataset.train[0]
        episode = "train/" + task.task_id
        record = self.runner.evaluate(task, self.runner.baseline, 42, episode)
        state_path = Path(record["project_state_path"])
        state = json.loads(state_path.read_text())
        self.assertEqual(state["status"], "verified")
        self.assertEqual(state["process_diagnostics"]["completed_tool_nodes"], 1)
        self.assertTrue(record["process_diagnostics"]["not_quality_metrics"])
        self.runner.propose(task, self.runner.baseline, [record], [], "factorial/0", experiment=True)
        proposal = json.loads(next((self.runner.root / "proposals").glob("*.json")).read_text())
        self.assertTrue(proposal["request"]["project_state"]["has_measured_video"])
        self.assertEqual(set(proposal["repair_impact"]), {"anchor", "a", "b", "joint"})
        with patch.object(self.ev, "rollout", side_effect=AssertionError("should use evaluation cache")):
            self.assertEqual(self.runner.evaluate(task, self.runner.baseline, 42, episode), record)
        self.assertEqual(len(list(state_path.parent.parent.glob("*/state.json"))), 1)

    def test_failed_verifier_keeps_generated_project_artifacts(self):
        from evovideo_skill.conditioning_runner import MeasurementUnavailable
        task = self.dataset.train[0]
        with patch.object(self.runner, "check_measurement", side_effect=MeasurementUnavailable("ambiguous evidence")):
            with self.assertRaises(MeasurementUnavailable):
                self.runner.evaluate(task, self.runner.baseline, 42, "train/" + task.task_id)
        state = json.loads(next((self.runner.root / "project_states").glob("*/*/state.json")).read_text())
        self.assertEqual(state["status"], "needs_review")
        self.assertEqual(state["failed_stage"], "verification")
        self.assertIsNone(state["quality"])
        self.assertTrue(state["output"]["artifact_id"])
        self.assertEqual(self.runner.memory.entries, {})
        self.assertFalse(list((self.runner.root / "evaluations").glob("*.json")))

    def test_unknown_candidate_excludes_whole_round_and_continues_without_refund(self):
        self.config["candidate_review_policy"] = "skip-experiment"
        self.runner = ConditioningRunner(self.dataset, self.ev, ConditioningSmokePlanner(),
            self.root / "skip-run", self.config, {"provider": "local-fake"})
        original = self.ev.rollout
        unknowns = []
        def execute(task, graph, **kwargs):
            result = original(task, graph, **kwargs)
            cfg = graph.nodes[-1].config
            if (not unknowns and "reference_ids" in cfg and "duration_seconds" not in cfg
                    and task.metadata["evaluation_seed"] == 456):
                unknowns.append(self.runner.ledger.reserved_calls)
                result.artifact.metadata["vlm_evaluation"] = {
                    "evaluation_status": "needs_review", "verification_metadata": {
                        "unobserved_criteria": ["holder"], "disagreement_criteria": [], "scope_issues": {},
                        "verifier_protocol": "boundary-grounded-video-evidence-v9", "judgment_path": "/evidence"},
                    "criterion_observations": {"holder": [{"status": "unobserved", "score": None}]}}
            return result
        with patch.object(self.ev, "rollout", side_effect=execute), \
                patch.object(self.runner.memory, "observe", wraps=self.runner.memory.observe) as observe:
            self.runner.learn()
        self.assertEqual(len(unknowns), 1)
        self.assertTrue(observe.called)
        self.assertTrue(all("factorial/0/" not in call.args[4] for call in observe.call_args_list))
        report = json.loads((self.runner.root / "interactions/0000.json").read_text())
        self.assertEqual(report["status"], "evidence_incomplete")
        self.assertEqual((report["failed_cell"], report["failed_seed"]), ("b", 456))
        self.assertEqual(len(report["cells"]["b"]), 2)
        self.assertEqual(report["unexecuted_cells"], ["joint"])
        self.assertEqual(report["selected_cell"], "parent")
        self.assertEqual(report["comparisons_to_parent"], {})
        self.assertIsNone(report["interaction"])
        self.assertEqual(report["budget"]["reserved_calls"], unknowns[0])
        self.assertGreaterEqual(self.runner.ledger.reserved_calls, unknowns[0])
        self.assertEqual(json.loads((self.runner.root / "interactions/0001.json").read_text())["status"], "complete")
        summary = json.loads((self.runner.root / "learning_summary.json").read_text())["evidence_exclusions"]
        self.assertEqual(summary["excluded_iterations"], [0])
        self.assertEqual(summary["reported_experiments"], 2)
        audit = json.loads(next((self.runner.root / "unobserved_evaluations").glob("*.json")).read_text())
        self.assertEqual(audit["seed"], 456)
        self.assertNotIn("score", audit)
        self.assertEqual(audit["criterion_observations"]["holder"][0]["score"], None)
        self.assertTrue(Path(audit["project_state_path"]).exists())
        self.assertFalse((self.runner.root / "evaluations" / (audit["evaluation_id"] + ".json")).exists())
        with patch.object(self.ev, "rollout", side_effect=AssertionError("must not rerun excluded experiments")):
            resumed = ConditioningRunner(self.dataset, self.ev, ConditioningSmokePlanner(),
                self.runner.root, self.config, {"provider": "local-fake"})
            resumed.learn()

    def test_unknown_baseline_and_api_failures_still_stop(self):
        self.runner.config["candidate_review_policy"] = "skip-experiment"
        with patch.object(self.runner, "evaluate", side_effect=EvidenceIncomplete("baseline unknown", {})):
            with self.assertRaises(EvidenceIncomplete):
                self.runner.learn()
        original = self.runner.evaluate
        def evaluate(task, graph, seed, episode):
            if "reference_ids" in graph.nodes[-1].config:
                raise MeasurementUnavailable("API failure")
            return original(task, graph, seed, episode)
        with patch.object(self.runner, "evaluate", side_effect=evaluate):
            with self.assertRaisesRegex(MeasurementUnavailable, "API failure"):
                self.runner.learn()
        self.assertFalse(self.runner.state["learned"])
        self.assertFalse(list((self.runner.root / "interactions").glob("*.json")))

    def test_unknown_candidate_default_policy_still_stops(self):
        original = self.runner.evaluate
        def evaluate(task, graph, seed, episode):
            if "reference_ids" in graph.nodes[-1].config:
                raise EvidenceIncomplete("candidate unknown", {})
            return original(task, graph, seed, episode)
        with patch.object(self.runner, "evaluate", side_effect=evaluate):
            with self.assertRaises(EvidenceIncomplete):
                self.runner.learn()
        self.assertFalse(self.runner.state["learned"])

    def test_skip_policy_is_validated(self):
        for policy, mode in (("ignore", "factorial"), ("skip-experiment", "legacy")):
            with self.subTest(policy=policy, mode=mode), self.assertRaises(ValueError):
                validate_config({**self.config, "candidate_review_policy": policy, "search_mode": mode})


if __name__ == "__main__":
    unittest.main()
