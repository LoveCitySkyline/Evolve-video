from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.benchmarks import BenchmarkSuite
from evovideo_skill.evolution_data import stratified_task_split
from evovideo_skill.evaluators import EvaluatorSuite
from evovideo_skill.graph_executor import ArtifactPassingGraphExecutor
from evovideo_skill.graph_evolver import GraphToolPathEvolver
from evovideo_skill.graph_skill import GraphEdge, GraphNode, GraphSkillMemory, ToolPathGraph
from evovideo_skill.models import VideoTask, VideoArtifact
from evovideo_skill.research_protocol import (
    BudgetLedger, ResearchBudgetExceeded, composition_template, decode_graph, feedback_view,
    generation_credits, graph_payload, validate_candidate, validate_splits,
)
from evovideo_skill.research_runner import ResearchArmRunner, SmokePlanner
from evovideo_skill.research_subgraphs import SubgraphLibrary, remap_config
from evovideo_skill.skill_memory import SkillMemory
from evovideo_skill.tools import ToolRegistry, VideoTool
from evovideo_skill.tool_onboarding import ToolSpec


ROOT = Path(__file__).resolve().parents[1]


class RecordingPlanner(SmokePlanner):
    def __init__(self):
        self.requests = []

    def propose(self, request):
        self.requests.append(deepcopy(request))
        return super().propose(request)


class DummyTool(VideoTool):
    def __init__(self, name):
        self.name = name

    def run(self, task, plan):
        return VideoArtifact(self.name, task.task_id, task.prompt, task.mode, [self.name], [])


def library_tools():
    registry = ToolRegistry()
    for name, inputs, output in (("draft", (), "video"), ("extract", ("video",), "image"),
                                 ("i2v", ("image",), "video")):
        registry.register(DummyTool(name), ToolSpec(name=name, capability=name,
                          input_types=inputs, output_type=output, consumes_upstream=bool(inputs), backend="builtin"))
    return registry


def donor_graph():
    return ToolPathGraph("donor", "donor", "", [], [
        GraphNode("d", "tool", "draft"), GraphNode("x", "tool", "extract", {"position": "last"}),
        GraphNode("v", "tool", "i2v", {"strength": 0.6})],
        [GraphEdge("dx", "d", "x"), GraphEdge("xv", "x", "v")])


class ResearchProtocolTests(unittest.TestCase):
    def test_scalar_feedback_excludes_evidence(self):
        with TemporaryDirectory() as tmp:
            ev = GraphToolPathEvolver(SkillMemory(Path(tmp) / "s"), GraphSkillMemory(Path(tmp) / "g"))
            result = ev.rollout(VideoTask("x", "A woman turns and waves."), ev.baseline_graph())
            result.artifact.metadata["vlm_evaluation"] = {"criterion_evidence": {"action": "secret"}}
            self.assertEqual(set(feedback_view(result, "scalar")), {"score"})
            local = feedback_view(result, "local")
            self.assertTrue(local["nodes"])
            self.assertEqual(local["criterion_evidence"]["action"], "secret")
            self.assertEqual(next(iter(local["nodes"].values()))["semantic_postcondition"], "unknown")

    def test_prompt_rejects_structural_edits(self):
        registry = ToolRegistry.with_mock_tools()
        executor = ArtifactPassingGraphExecutor(registry, EvaluatorSuite())
        base = GraphToolPathEvolver.baseline_graph()
        payload = graph_payload(base)
        payload["nodes"][-1]["config"]["prompt"] = "Preserve identity."
        validate_candidate(decode_graph(payload), base, executor, "prompt", 16, 24)
        with self.assertRaisesRegex(ValueError, "fixed-topology"):
            validate_candidate(composition_template(base), base, executor, "prompt", 16, 24)

    def test_forbid_mutable_verifier_and_task_override(self):
        base = GraphToolPathEvolver.baseline_graph()
        payload = graph_payload(base)
        payload["reward"] = 1
        with self.assertRaises(ValueError):
            decode_graph(payload)
        base.nodes.append(GraphNode("j", "verifier", "identity_consistency"))
        with self.assertRaisesRegex(ValueError, "fixed external verifier"):
            validate_candidate(base, base, ArtifactPassingGraphExecutor(ToolRegistry.with_mock_tools(), EvaluatorSuite()),
                               "graph", 16, 24)

    def test_call_and_seconds_budget_and_long_direct(self):
        task = VideoTask("long", "x", duration_seconds=20,
                         metadata={"h3_shots": [{"duration_seconds": 10}, {"duration_seconds": 10}]})
        self.assertEqual(generation_credits(task, GraphToolPathEvolver.baseline_graph()), (2, 20))
        ledger = BudgetLedger(3, 25)
        ledger.reserve(2, 20)
        with self.assertRaises(ResearchBudgetExceeded):
            ledger.reserve(1, 6)
        self.assertEqual(ledger.reserved_calls, 2)
        with self.assertRaises(ResearchBudgetExceeded):
            ledger.reserve(2, 2)

    def test_split_leakage_including_scenarios(self):
        d = stratified_task_split(BenchmarkSuite.from_file(ROOT / "examples/research_smoke_tasks.json").tasks)
        validate_splits(d)
        d.train[0].metadata["scenario_id"] = "same"
        d.test[0].metadata["scenario_id"] = "same"
        with self.assertRaises(ValueError):
            validate_splits(d)


class SubgraphTests(unittest.TestCase):
    def test_exact_renaming_invariant_but_config_sensitive(self):
        library = SubgraphLibrary(library_tools())
        graph = donor_graph()
        ids = library.mine(graph, "identity", {"task_id": "a", "gain": .1})
        renamed = deepcopy(graph)
        mapping = {"d": "another_d", "x": "another_x", "v": "another_v"}
        for node in renamed.nodes:
            node.node_id = mapping[node.node_id]
        for edge in renamed.edges:
            edge.source, edge.target = mapping[edge.source], mapping[edge.target]
        same = library.mine(renamed, "identity", {"task_id": "b", "gain": .1})
        self.assertEqual(set(ids), set(same))
        self.assertTrue(all(len(library.fragments[i].observations) == 2 for i in ids))
        renamed.nodes[-1].config["strength"] = .8
        changed = library.mine(renamed, "identity", {"task_id": "c", "gain": .1})
        self.assertFalse(set(ids) & set(changed))

    def test_splice_and_execute_with_real_artifact_edges(self):
        registry = library_tools()
        library = SubgraphLibrary(registry)
        ids = library.mine(donor_graph(), "identity", {"task_id": "a", "gain": .1})
        fragment = next(library.fragments[i] for i in ids if len(library.fragments[i].nodes) == 2)
        base = ToolPathGraph("b", "b", "", [], [GraphNode("draft", "tool", "draft")], [])
        graph = library.append(base, fragment.fragment_id, "draft", "identity")
        executor = ArtifactPassingGraphExecutor(registry, EvaluatorSuite())
        executor.validate_graph(graph)
        task = VideoTask("task", "prompt")
        from evovideo_skill.planning import Planner
        result = executor.execute(task, Planner().plan(task, []), graph)
        self.assertEqual(result.executed_tools, ["draft", "extract", "i2v"])
        self.assertEqual(len(result.artifact.metadata["node_evidence"]), 3)
        with self.assertRaisesRegex(ValueError, "applicability"):
            library.append(base, fragment.fragment_id, "draft", "other")

    def test_bindings_remap_and_task_bound_fragment_exclusion(self):
        self.assertEqual(remap_config({"bindings": [{"source": "a", "role": "first_frame"}],
                                       "source_nodes": ["a", "b"]}, {"a": "new"}),
                         {"bindings": [{"source": "new", "role": "first_frame"}], "source_nodes": ["new", "b"]})
        graph = donor_graph()
        graph.nodes[-1].config["reference_ids"] = ["donor-only"]
        self.assertEqual(SubgraphLibrary(library_tools()).mine(graph, "a", {"task_id": "a"}), [])


class ResearchRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = stratified_task_split(BenchmarkSuite.from_file(ROOT / "examples/research_smoke_tasks.json").tasks)
        self.config = {"max_searches": 2, "searches_per_task": 1, "max_generation_calls": 100,
                       "max_generated_seconds": 600, "evaluation_seeds": [42, 123, 456],
                       "max_nodes": 16, "max_edits": 24, "min_gain": .02}
        self.ev = GraphToolPathEvolver(SkillMemory(self.root / "s"), GraphSkillMemory(self.root / "g"))
        real_rollout = self.ev.rollout

        def fixture(task, graph):
            result = real_rollout(task, graph)
            quality = .8 if any(n.config.get("prompt") for n in graph.nodes) else .5
            result.reward = replace(result.reward, score=quality)
            result.artifact.metadata["vlm_evaluation"] = {"criterion_evidence": {"action": "localized evidence"}}
            return result

        self.ev.rollout = fixture
        self.planner = RecordingPlanner()

    def runner(self, arm="graph_scalar", protocol=None):
        return ResearchArmRunner(arm, self.dataset, self.ev, self.planner, self.root / arm,
                                 protocol or {"provider": "local-fake"}, self.config)

    def test_frozen_test_no_planner_and_resume_no_replay(self):
        runner = self.runner()
        result = runner.run()
        self.assertEqual(result["status"], "complete")
        self.assertAlmostEqual(result["heldout_gain"], .3)
        self.assertEqual(len(self.planner.requests), 2)
        for request in self.planner.requests:
            self.assertNotIn("smoke-test", json.dumps(request))
            self.assertNotIn("localized evidence", json.dumps(request))
            self.assertTrue(all(set(x) == {"score"} for x in request["feedback"]))
        previous_calls = result["budget"]["reserved_calls"]
        with patch.object(self.ev, "rollout", side_effect=AssertionError("replayed")):
            resumed = self.runner().run()
        self.assertEqual(resumed["budget"]["reserved_calls"], previous_calls)
        with self.assertRaisesRegex(ValueError, "protocol changed"):
            self.runner(protocol={"provider": "other"})

    def test_memory_arm_separation(self):
        none = self.runner("graph_none")
        none.run()
        self.assertTrue(all(request["memory"] == {} for request in self.planner.requests))
        self.planner.requests.clear()
        whole = self.runner("graph_whole")
        whole.run()
        self.assertTrue(self.planner.requests[1]["memory"]["whole_paths"])

    def test_validated_subgraph_reused_on_second_training_task(self):
        for name, inputs, output in (("test_extract", ("video",), "image"), ("test_i2v", ("image",), "video")):
            self.ev.tools.register(DummyTool(name), ToolSpec(name=name, capability=name,
                input_types=inputs, output_type=output, consumes_upstream=True, backend="builtin"))
        class FragmentPlanner:
            reused = False

            def propose(self, request):
                fragments = request["memory"].get("fragments", [])
                match = next((f for f in fragments if len(f["nodes"]) == 2), None)
                if match:
                    self.reused = True
                    return {"reuse_fragment_id": match["fragment_id"], "source_node": "tool_t2v"}
                graph = deepcopy(request["parent"])
                graph["nodes"] += [
                    {"node_id": "extract", "node_type": "tool", "name": "test_extract", "config": {}},
                    {"node_id": "repair", "node_type": "tool", "name": "test_i2v", "config": {"prompt": "Preserve identity"}}]
                graph["edges"] += [as_edge("up", "tool_t2v", "extract"), as_edge("down", "extract", "repair")]
                return {"graph": graph}

        def as_edge(ident, source, target):
            return {"edge_id": ident, "source": source, "target": target, "condition": "always", "config": {}}

        self.planner = FragmentPlanner()
        result = self.runner("graph_subgraph").run()
        self.assertTrue(self.planner.reused)
        self.assertAlmostEqual(result["heldout_gain"], .3)
        fragments = json.loads((self.root / "graph_subgraph/subgraphs.json").read_text())
        self.assertTrue(any(len(f["observations"]) == 2 for f in fragments))

    def test_exhausted_budget_is_censored_not_gain_zero(self):
        self.config["max_generation_calls"] = 1
        result = self.runner().run()
        self.assertEqual(result["status"], "budget_censored")
        self.assertIsNone(result["heldout_gain"])

    def test_bad_verifier_not_quality_sample(self):
        old = self.ev.rollout

        def bad(task, graph):
            result = old(task, graph)
            result.artifact.metadata["vlm_evaluation"] = {"evaluation_status": "failed_api"}
            return result

        self.ev.rollout = bad
        with self.assertRaisesRegex(RuntimeError, "verifier unavailable"):
            self.runner().run()
        self.assertFalse((self.root / "graph_scalar" / "executions.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
