from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.benchmarks import BenchmarkSuite
from evovideo_skill.conditioning_interactions import interaction_effect
from evovideo_skill.conditioning_planner import ConditioningSmokePlanner
from evovideo_skill.conditioning_repair import (check_local_candidate, dominators,
    output_spans, preservation_report, repair_frontier)
from evovideo_skill.conditioning_runner import ConditioningRunner, FixedTaskPlanner, validate_config
from evovideo_skill.conditioning_search import (SignedInteractionGraph, context_view,
    decode_pool, factor_descriptor, search_options, select_pair)
from evovideo_skill.evolution_data import stratified_task_split
from evovideo_skill.graph_evolver import GraphToolPathEvolver
from evovideo_skill.graph_skill import GraphNode, GraphEdge, ToolPathGraph, GraphSkillMemory
from evovideo_skill.models import VideoTask
from evovideo_skill.models import VideoArtifact
from evovideo_skill.research_protocol import graph_payload
from evovideo_skill.skill_memory import SkillMemory
from evovideo_skill.tool_onboarding import ToolSpec
from evovideo_skill.tools import VideoTool


ROOT = Path(__file__).resolve().parents[1]


def record(score=.5, seed=1, task_id="task", identity=None):
    return {"task_id": task_id, "seed": seed, "score": score, "evaluation_id": f"{task_id}/{seed}/{score}",
            "criterion_scores": {"quality": score, "identity": score if identity is None else identity}}


def factorial(scores, task_id="task"):
    return {key: [record(value, s, task_id) for s in (1, 2, 3)] for key, value in zip(
        ("anchor", "a", "b", "joint"), scores)}


def strategy(name):
    return {"name": name, "instruction": "Select visual references for a generation stage.",
            "hypothesis": "Condition choice may alter consistency.", "risks": ["Motion may regress."],
            "required_references": {}}


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = json.loads((ROOT / "configs/conditioning_graph_search_smoke.json").read_text())
        validate_config(self.config)
        self.options = search_options(self.config)
        self.task = VideoTask("task", "A person walks.")
        self.anchor = GraphToolPathEvolver.baseline_graph()
        self.graphs = {}
        for key, field, value in (("a", "duration_seconds", 6), ("b", "reference_ids", ["identity"]),
                                  ("c", "role", "reference_video")):
            graph = deepcopy(self.anchor)
            graph.nodes[-1].config[field] = value
            self.graphs[key] = graph
        self.ev = GraphToolPathEvolver(SkillMemory(self.root / "skills"), GraphSkillMemory(self.root / "graphs"))
        self.frontier = {"status": "unlocalized", "boundaries": []}
        self.before = [record(seed=s) for s in (1, 2, 3)]
        self.context = context_view(self.task, self.before)
        self.desc = {k: factor_descriptor(self.task, self.anchor, v, self.context) for k, v in self.graphs.items()}

    def pool(self):
        return {"anchor": graph_payload(self.anchor), "factors": [
            {"id": k, "graph": graph_payload(g), "strategy": strategy(k)} for k, g in self.graphs.items()]}

    def select(self, evidence, raw=None, remaining=None):
        anchor, factors, errors = decode_pool(raw or self.pool(), self.task, self.anchor,
                                             self.ev.executor, self.config, self.frontier)
        chosen, audit = select_pair(self.task, self.anchor, anchor, factors, self.before,
            evidence, self.ev.executor, self.config, self.frontier, remaining or {"calls": 100, "seconds": 3000})
        return chosen, audit, errors

    def test_measured_negative_interaction_changes_next_choice(self):
        graph = SignedInteractionGraph()
        self.assertEqual(self.select(graph)[1]["selected_pair"], "a+b")
        interaction = interaction_effect(factorial([.5, .6, .6, .2]))
        graph.observe(self.task, self.desc, interaction, "factorial/0")
        chosen, audit, _ = self.select(graph)
        self.assertNotEqual(audit["selected_pair"], "a+b")
        rejected_pair = next(r for r in audit["ranked_pairs"] if r["pair_id"] == "a+b")
        self.assertAlmostEqual(rejected_pair["predicted_gain_from_anchor"], -.3)
        self.assertEqual(set(chosen[0]), {"anchor", "a", "b", "joint"})

    def test_task_balancing_idempotence_and_train_only(self):
        graph = SignedInteractionGraph()
        effect = interaction_effect(factorial([.4, .6, .6, .8]))
        graph.observe(self.task, self.desc, effect, "factorial/0")
        saved = deepcopy(graph.data)
        graph.observe(self.task, self.desc, effect, "factorial/0")
        self.assertEqual(graph.data, saved)
        for i in range(1, 10):
            graph.observe(self.task, self.desc, effect, f"factorial/{i}")
        other = deepcopy(self.task)
        other.task_id = "other"
        negative = interaction_effect(factorial([.5, .3, .3, .1], "other"))
        graph.observe(other, self.desc, negative, "factorial/10")
        estimate = graph.estimate(graph.data["nodes"][self.desc["a"]["factor_id"]], self.options)
        self.assertEqual(estimate["task_support"], 2)
        self.assertAlmostEqual(estimate["mean"], 0)
        with self.assertRaisesRegex(ValueError, "training"):
            graph.observe(self.task, self.desc, effect, "test/0")

    def test_signatures_keep_role_scope_and_anonymize_references(self):
        self.task.metadata["h3_references"] = [{"id": "actor", "kind": "image", "uri": "/private/actor.png"}]
        self.graphs["b"].nodes[-1].config["reference_ids"] = ["actor"]
        first = factor_descriptor(self.task, self.anchor, self.graphs["b"], self.context)
        other = deepcopy(self.task)
        other.task_id = "new-task"
        other.metadata["h3_references"][0].update(id="someone", uri="/different/photo.png")
        graph = deepcopy(self.graphs["b"])
        graph.nodes[-1].config["reference_ids"] = ["someone"]
        second = factor_descriptor(other, self.anchor, graph, self.context)
        self.assertEqual(first["factor_id"], second["factor_id"])
        self.assertNotIn("/private/actor", json.dumps(first))
        changed_context = {**self.context, "family": "another"}
        self.assertNotEqual(first["factor_id"], factor_descriptor(other, self.anchor, graph, changed_context)["factor_id"])
        graph.nodes[-1].config["position"] = "last"
        self.assertNotEqual(first["factor_id"], factor_descriptor(other, self.anchor, graph, self.context)["factor_id"])

    def test_invalid_factor_is_audited_while_other_pairs_survive(self):
        raw = self.pool()
        raw["factors"][0]["graph"]["nodes"][-1]["config"]["prompt"] = "rewrite"
        chosen, audit, errors = self.select(SignedInteractionGraph(), raw)
        self.assertIsNotNone(chosen)
        self.assertEqual(audit["selected_pair"], "b+c")
        self.assertEqual(len(errors), 1)

    def test_overlapping_pairs_and_insufficient_budget_are_not_executed(self):
        raw = self.pool()
        raw["factors"][2]["graph"] = deepcopy(raw["factors"][1]["graph"])
        raw["factors"][2]["graph"]["nodes"][-1]["config"]["reference_ids"] = ["different"]
        _, audit, _ = self.select(SignedInteractionGraph(), raw)
        self.assertTrue(any("overlapping" in r["error"] for r in audit["rejected_pairs"]))
        selected, audit, _ = self.select(SignedInteractionGraph(), remaining={"calls": 1, "seconds": 1})
        self.assertIsNone(selected)
        self.assertIsNone(audit["selected_pair"])

    def test_unknown_settings_and_invalid_numbers_fail_early(self):
        for key, value in (("pool_size", 20), ("exploration_beta", float("nan")),
                           ("enabled", "true"), ("unknown", 1)):
            config = deepcopy(self.config)
            config["active_graph_search"][key] = value
            with self.assertRaises(ValueError):
                validate_config(config)
        config = deepcopy(self.config)
        config["search_mode"] = "single"
        with self.assertRaisesRegex(ValueError, "factorial"):
            validate_config(config)

    def test_joint_strategy_requires_union_of_external_assets(self):
        self.task.metadata["h3_references"] = [
            {"id": "one", "kind": "image"}, {"id": "two", "kind": "image"}]
        raw = self.pool()
        raw["factors"] = raw["factors"][:2]
        raw["factors"][0]["graph"]["nodes"][-1]["config"].update(reference_id="one")
        raw["factors"][1]["graph"]["nodes"][-1]["config"].update(reference_ids=["two"])
        for factor in raw["factors"]:
            factor["strategy"]["required_references"] = {"image": 1}
        chosen, _, _ = self.select(SignedInteractionGraph(), raw)
        self.assertEqual(chosen[1]["joint"]["required_references"]["image"], 2)


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.task = VideoTask("task", "Three shots.", duration_seconds=18)
        self.nodes = [GraphNode("bank", "tool", "h3_reference_bank")]
        self.edges = []
        for index in range(1, 4):
            self.nodes.extend([GraphNode(f"s{index}", "tool", "h3_reference_select"),
                GraphNode(f"g{index}", "tool", "h3_ref2va", {"duration_seconds": 6})])
            self.edges.extend([GraphEdge(f"bs{index}", "bank", f"s{index}"),
                               GraphEdge(f"sg{index}", f"s{index}", f"g{index}"),
                               GraphEdge(f"gc{index}", f"g{index}", "concat")])
        self.nodes.append(GraphNode("concat", "tool", "h3_av_concat", {"source_nodes": ["g1", "g2", "g3"]}))
        self.graph = ToolPathGraph("g", "g", "", [], self.nodes, self.edges)
        self.options = search_options({"search_mode": "factorial", "active_graph_search": {"boundary_limit": 1}})
        self.records = [{"feedback": {"failed_segments": [{"start_ratio": 1/3, "end_ratio": 2/3,
                                                           "failed_criteria": ["background"]}]}}]

    def test_separator_protects_independent_shots_and_counts_union(self):
        frontier = repair_frontier(self.task, self.graph, self.records, self.options)
        boundary = frontier["boundaries"][0]
        self.assertEqual(frontier["targets"], ["g2"])
        self.assertEqual(boundary["affected_nodes"], ["concat", "g2"])
        self.assertEqual(boundary["generation_calls"], 1)
        self.assertEqual(boundary["generated_seconds"], 6)
        child = deepcopy(self.graph)
        child.node("g2").config["role"] = "reference_video"
        self.assertEqual(check_local_candidate(self.graph, child, frontier)["status"], "within_boundary")
        child.node("g1").config["role"] = "reference_video"
        with self.assertRaisesRegex(ValueError, "boundary"):
            check_local_candidate(self.graph, child, frontier)

    def test_downstream_dependent_shot_is_in_regeneration_closure(self):
        self.graph.edges.append(GraphEdge("propagation", "g2", "g3"))
        frontier = repair_frontier(self.task, self.graph, self.records, self.options)
        self.assertIn("g3", frontier["boundaries"][0]["affected_nodes"])
        self.assertEqual(frontier["boundaries"][0]["generation_calls"], 2)
        self.assertEqual(frontier["boundaries"][0]["other_output_nodes_affected"], ["g3"])

    def test_multiple_influence_branches_need_separator_covering_every_path(self):
        self.graph.nodes.append(GraphNode("other", "tool", "h3_reference_bank"))
        self.graph.edges.append(GraphEdge("other-g2", "other", "g2"))
        frontier = repair_frontier(self.task, self.graph, self.records, self.options)
        self.assertEqual(frontier["boundaries"][0]["nodes"], ["g2"])
        self.assertNotIn("bank", dominators(self.graph)["g2"])

    def test_no_timeline_or_no_failure_means_no_fabricated_localization(self):
        self.assertEqual(repair_frontier(self.task, self.graph, [], self.options)["status"], "unlocalized")
        self.graph.node("concat").name = "unknown_video_editor"
        self.assertEqual(output_spans(self.task, self.graph), [])
        self.assertEqual(repair_frontier(self.task, self.graph, self.records, self.options)["status"], "unlocalized")

    def test_inserting_new_processor_after_local_output_is_allowed(self):
        frontier = repair_frontier(self.task, self.graph, self.records, self.options)
        child = deepcopy(self.graph)
        child.nodes.append(GraphNode("repair", "tool", "h3_ref2va"))
        # Analyze the attachment; executable contract checking is a separate gate.
        child.edges.append(GraphEdge("attach", "g2", "repair"))
        self.assertEqual(check_local_candidate(self.graph, child, frontier)["status"], "within_boundary")

    def test_preservation_rejects_side_effect_even_if_overall_quality_improves(self):
        report = preservation_report([record(.5, identity=.95)], [record(.8, identity=.90)], .85, .03)
        self.assertFalse(report["passed"])
        self.assertEqual(report["violations"][0]["key"], "identity")
        self.assertTrue(preservation_report([record(.5, identity=.95)], [record(.8, identity=.94)], .85, .03)["passed"])

    def test_protected_time_window_cannot_disappear(self):
        left, right = record(), record(.8)
        left["artifact"] = {"metadata": {"vlm_evaluation": {
            "verification_metadata": {"windows": [{"start_seconds": 0, "end_seconds": 6}]},
            "criterion_observations": {"identity": [{"segments": [
                {"segment_id": 0, "status": "observed", "score": .95}]}]}}}}
        report = preservation_report([left], [right], .85, .03)
        self.assertFalse(report["passed"])
        self.assertEqual(report["violations"][0]["scope"], "segment")
        self.assertIsNone(report["violations"][0]["after"])


class ActiveRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = json.loads((ROOT / "configs/conditioning_graph_search_smoke.json").read_text())
        validate_config(self.config)
        self.dataset = stratified_task_split(BenchmarkSuite.from_file(ROOT / self.config["task_file"]).tasks)
        self.ev = GraphToolPathEvolver(SkillMemory(self.root / "skills"), GraphSkillMemory(self.root / "graphs"),
                                      planner=FixedTaskPlanner())
        self.planner = ConditioningSmokePlanner()
        self.runner = self.make_runner()
        self.stdout = redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def make_runner(self):
        return ConditioningRunner(self.dataset, self.ev, self.planner, self.root / "run", self.config, {"provider": "local-fake"})

    def test_active_full_run_freezes_train_evidence_and_exports_decisions(self):
        self.runner.learn()
        self.assertEqual(self.runner.state["interaction_cursor"]["next_index"], 2)
        self.assertTrue(self.runner.signed_graph.data["edges"])
        decisions = list((self.runner.root / "search_decisions").glob("*.json"))
        self.assertEqual(len(decisions), 2)
        self.assertEqual(json.loads(decisions[0].read_text())["audit"]["selected_pair"], "a+b")
        frozen = self.runner.validate_and_freeze()
        original = frozen.read_bytes()
        signed = deepcopy(self.runner.signed_graph.data)
        self.assertEqual(self.runner.test(frozen)["heldout_gain"], 0)
        self.assertEqual(signed, self.runner.signed_graph.data)
        self.assertEqual(original, frozen.read_bytes())

    def test_completed_round_is_not_replayed_after_interrupt(self):
        original = self.runner.propose
        def interrupt(*args, **kwargs):
            if args[4] == "factorial/1":
                raise KeyboardInterrupt()
            return original(*args, **kwargs)
        with patch.object(self.runner, "propose", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.runner.learn()
        self.assertEqual(self.runner.state["interaction_cursor"]["next_index"], 1)
        resumed = self.make_runner()
        with patch.object(resumed, "propose", wraps=resumed.propose) as propose:
            resumed.learn()
        self.assertEqual([call.args[4] for call in propose.call_args_list], ["factorial/1"])
        for entry in resumed.signed_graph.data["edges"].values():
            self.assertEqual(len(entry["observations"]), len(set(entry["observations"])))

    def test_mid_factorial_resume_keeps_selected_pair_and_cached_generation(self):
        original = self.runner.evaluate
        counter = 0
        def interrupt(*args, **kwargs):
            nonlocal counter
            counter += 1
            if counter == 8:
                raise KeyboardInterrupt()
            return original(*args, **kwargs)
        with patch.object(self.runner, "evaluate", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.runner.learn()
        decision = json.loads(next((self.runner.root / "search_decisions").glob("*.json")).read_text())
        resumed = self.make_runner()
        with patch("evovideo_skill.conditioning_runner.select_pair", wraps=select_pair) as select:
            resumed.learn()
        self.assertEqual(select.call_count, 1)  # Only the next round selects anew.
        report = json.loads((self.runner.root / "interactions/0000.json").read_text())
        self.assertEqual(report["active_search"], decision)

    def test_executor_reuses_independent_branch_but_reruns_dependent_suffix(self):
        calls = []
        class FixtureTool(VideoTool):
            def __init__(self, name):
                self.name = name

            def run(self, task, plan):
                raise AssertionError("artifact context required")

            def run_with_context(self, task, plan, context):
                calls.append(context.node_id)
                return VideoArtifact(str(len(calls)), task.task_id, task.prompt, task.mode, [self.name], [],
                    {"artifact_type": "video", "upstream_conditioning_consumed": bool(context.input_artifacts)})

        for name in ("mock_text_to_video", "h3_ref2va", "h3_av_concat"):
            self.ev.tools.register(FixtureTool(name), ToolSpec(name=name, capability=name,
                input_types=() if name == "mock_text_to_video" else ("video",), output_type="video",
                consumes_upstream=name != "mock_text_to_video", backend="builtin"))
        graph = ToolPathGraph("fixture", "fixture", "", [], [
            GraphNode("g1", "tool", "mock_text_to_video"),
            GraphNode("g2", "tool", "mock_text_to_video"),
            GraphNode("g3", "tool", "h3_ref2va"),
            GraphNode("concat", "tool", "h3_av_concat", {"source_nodes": ["g1", "g2", "g3"]})], [
            GraphEdge("dependency", "g2", "g3"), GraphEdge("c1", "g1", "concat"),
            GraphEdge("c2", "g2", "concat"), GraphEdge("c3", "g3", "concat")])
        task = self.dataset.train[0]
        self.runner.evaluate(task, graph, 42, "train/" + task.task_id)
        changed = deepcopy(graph)
        changed.node("g2").config["reference_ids"] = ["replacement"]
        result = self.runner.evaluate(task, changed, 42, "train/" + task.task_id)
        self.assertEqual(result["reused_nodes"], ["g1"])
        self.assertEqual(result["executed_nodes"], ["g2", "g3", "concat"])


if __name__ == "__main__":
    unittest.main()
