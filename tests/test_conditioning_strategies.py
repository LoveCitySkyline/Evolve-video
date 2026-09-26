from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.benchmarks import BenchmarkSuite
from evovideo_skill.conditioning_cache import ConditioningNodeCache
from evovideo_skill.conditioning_memory import StrategyMemory, paired_effect, portable_recipe, task_state, validate_strategy
from evovideo_skill.conditioning_planner import ConditioningSmokePlanner, decode_proposal
from evovideo_skill.conditioning_runner import ConditioningRunner, EpisodeBudgetExceeded, FixedTaskPlanner, validate_config
from evovideo_skill.evolution_data import stratified_task_split
from evovideo_skill.graph_evolver import GraphToolPathEvolver
from evovideo_skill.graph_skill import GraphNode, GraphEdge, GraphSkillMemory
from evovideo_skill.models import VideoArtifact, VideoTask
from evovideo_skill.research_protocol import graph_payload, write_json
from evovideo_skill.research_subgraphs import stable_hash
from evovideo_skill.skill_memory import SkillMemory
from evovideo_skill.tool_onboarding import ToolSpec
from evovideo_skill.tools import VideoTool


ROOT = Path(__file__).resolve().parents[1]


def strategy():
    return {"name": "clean_identity_anchor", "instruction": "Use a clean identity reference when continuity is weak.",
            "hypothesis": "A clean anchor may avoid propagating visual errors.",
            "risks": ["May constrain motion."], "required_references": {}}


def effect(gain=.1):
    return {"gain": gain, "metric_deltas": {"identity": gain, "action": -.01}, "pairs": []}


class MemoryTests(unittest.TestCase):
    def test_scope_negative_evidence_and_freeze(self):
        memory = StrategyMemory()
        task = VideoTask("train", "An actor moves", metadata={"category": "identity"})
        graph = GraphToolPathEvolver.baseline_graph()
        key = memory.observe(task, strategy(), graph, effect(), "one")
        memory.observe(task, strategy(), graph, effect(), "one")
        memory.observe(VideoTask("train2", "Another actor", metadata={"category": "identity"}),
                       strategy(), graph, effect(-.2), "two")
        retrieved = memory.retrieve(task)
        self.assertEqual(retrieved[0]["evidence"]["experiment_count"], 2)
        self.assertEqual(retrieved[0]["evidence"]["status"], "mixed")
        self.assertIn("action", retrieved[0]["evidence"]["observed_risks"])
        self.assertEqual(memory.retrieve(VideoTask("other", "test", metadata={"category": "style"})), [])
        frozen = StrategyMemory(memory.snapshot(), frozen=True)
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            frozen.observe(task, strategy(), graph, effect(), "three")
        self.assertEqual(frozen.snapshot(), memory.snapshot())
        self.assertIn(key, memory.entries)

    def test_reference_requirements_and_donor_rejection(self):
        task = VideoTask("task-id", "Specific private donor prompt")
        value = strategy()
        value["required_references"] = {"image": 1}
        with self.assertRaisesRegex(ValueError, "unavailable"):
            validate_strategy(value, task)
        value = strategy()
        value["instruction"] = task.prompt
        with self.assertRaisesRegex(ValueError, "donor"):
            validate_strategy(value, task)
        self.assertEqual(task_state(task)["observations"], [])

    def test_recipe_scrubs_donor_literals(self):
        graph = GraphToolPathEvolver.baseline_graph()
        graph.nodes[-1].config.update(prompt="secret actor", reference_ids=["private-reference"],
                                     duration_seconds=12, shot_index=3)
        recipe = portable_recipe(graph)
        self.assertFalse(recipe["executable"])
        self.assertNotIn("secret actor", json.dumps(recipe))
        self.assertNotIn("private-reference", json.dumps(recipe))
        self.assertEqual(recipe["nodes"][-1]["config"]["shot_index"], "$target_shot_index")

    def test_paired_observations_require_matching_seeds(self):
        a = {"task_id": "t", "seed": 1, "evaluation_id": "a", "score": .6, "reward": {"components": {"motion": .8}}}
        b = {**a, "evaluation_id": "b", "score": .7, "reward": {"components": {"motion": .6}}}
        result = paired_effect([a], [b])
        self.assertAlmostEqual(result["gain"], .1)
        self.assertAlmostEqual(result["metric_deltas"]["motion"], -.2)
        with self.assertRaises(ValueError):
            paired_effect([a], [{**b, "seed": 2}])


class RecordingPlanner(ConditioningSmokePlanner):
    def __init__(self):
        self.requests = []

    def propose(self, request):
        self.requests.append(deepcopy(request))
        return super().propose(request)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = stratified_task_split(BenchmarkSuite.from_file(ROOT / "examples/research_smoke_tasks.json").tasks)
        self.config = json.loads((ROOT / "configs/conditioning_smoke.json").read_text())
        validate_config(self.config)
        self.ev = GraphToolPathEvolver(SkillMemory(self.root / "skills"), GraphSkillMemory(self.root / "graphs"),
                                      planner=FixedTaskPlanner())
        self.planner = RecordingPlanner()
        self.runner = ConditioningRunner(self.dataset, self.ev, self.planner, self.root / "run", self.config,
                                         {"provider": "local-fake"})

    def frozen(self):
        self.runner.learn()
        return self.runner.validate_and_freeze()

    def test_all_protocols_and_resume_without_replay(self):
        frozen = self.frozen()
        original = frozen.read_bytes()
        trained = deepcopy(self.runner.state["entries"])
        self.planner.requests.clear()
        direct = self.runner.test(frozen)
        self.assertEqual(direct["status"], "complete")
        self.assertEqual(len(direct["pairs"]), len(self.dataset.test) * 3)
        self.assertTrue(all(not q["state"]["observations"] for q in self.planner.requests))
        self.assertTrue(all("final_scores" not in json.dumps(q) for q in self.planner.requests))
        self.planner.requests.clear()
        adaptive = self.runner.test(frozen, "adaptive", "none")
        self.assertEqual(adaptive["status"], "complete")
        self.assertTrue(all(not q["memory"] for q in self.planner.requests))
        self.assertTrue(any(q["state"]["observations"] for q in self.planner.requests))
        self.assertEqual(frozen.read_bytes(), original)
        self.assertEqual(self.runner.state["entries"], trained)
        with patch.object(self.ev, "rollout", side_effect=AssertionError("replayed")), patch.object(
                self.planner, "propose", side_effect=AssertionError("replanned")):
            resumed = self.runner.test(frozen)
        self.assertEqual(resumed["pairs"], direct["pairs"])

    def test_direct_does_not_choose_baseline_by_final_score(self):
        frozen = self.frozen()
        self.planner.requests.clear()
        seen = []
        def judge(task, record, root):
            committed = json.loads((root / "committed_selections.json").read_text())
            self.assertEqual(len(committed), len(self.dataset.test) * 3)
            seen.append(record["evaluation_id"])
            return {"score": 1.0 if record["graph_id"] == self.runner.baseline.graph_id else .1}
        with patch.object(self.runner, "final_score", side_effect=judge):
            result = self.runner.test(frozen)
        self.assertLess(result["heldout_gain"], 0)
        self.assertTrue(all(p["candidate_graph_id"] != self.runner.baseline.graph_id for p in result["pairs"]))
        self.assertEqual(len(seen), len(self.dataset.test) * 6)

    def test_snapshot_tamper_and_test_leakage(self):
        frozen = self.frozen()
        data = json.loads(frozen.read_text())
        data["admitted_ids"].append("forged")
        write_json(frozen, data)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.runner.load_frozen(frozen)
        data.pop("content_hash")
        data["source_task_ids"].append(self.dataset.test[0].task_id)
        data["content_hash"] = stable_hash(data)
        write_json(frozen, data)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.runner.load_frozen(frozen)

    def test_episode_budget_and_failed_verifier(self):
        self.config["test_max_generation_calls"] = 1
        self.runner.charge(1, 6, False, "test/a")
        self.runner.charge(1, 6, True, "test/a")
        with self.assertRaises(EpisodeBudgetExceeded):
            self.runner.charge(1, 6, False, "test/a")
        artifact = VideoArtifact("a", "t", "x", "generation", [], [],
                                 {"vlm_evaluation": {"evaluation_status": "failed_api"}})
        with self.assertRaisesRegex(RuntimeError, "verifier"):
            self.runner.check_measurement(artifact, .5)

    def test_validation_never_updates_training_memory(self):
        self.runner.learn()
        entries = self.runner.memory.snapshot()
        self.runner.validate_and_freeze()
        self.assertEqual(entries, self.runner.memory.snapshot())
        for request in self.planner.requests:
            if request["operation"].startswith("validation/"):
                self.assertEqual(request["state"]["observations"], [])

    def test_cache_invalidates_changed_files_and_inputs(self):
        task = VideoTask("cache", "A scene")
        graph = self.runner.baseline
        node = graph.nodes[-1]
        plan = FixedTaskPlanner().plan(task, [])
        spec = self.ev.tools.spec(node.name)
        media = self.root / "input.mp4"
        media.write_bytes(b"first")
        artifact = VideoArtifact("a", task.task_id, task.prompt, task.mode, [node.name], [],
                                 {"local_video_path": str(media)})
        cache = ConditioningNodeCache(self.root / "cache", "v1", lambda *args: None)
        cache.store(task, plan, node, {}, spec, artifact)
        found = cache.lookup(task, plan, node, {}, spec)
        self.assertIsNotNone(found)
        found.metadata["vlm_evaluation"] = {"score": 1}
        self.assertNotIn("vlm_evaluation", cache.lookup(task, plan, node, {}, spec).metadata)
        self.assertIsNone(cache.lookup(task, plan, replace(node, config={"prompt": "changed"}), {}, spec))
        media.write_bytes(b"different")
        self.assertIsNone(cache.lookup(task, plan, node, {}, spec))

    def test_unregistered_tools_and_unknown_strategies(self):
        task = self.dataset.train[0]
        request = {"parent": graph_payload(self.runner.baseline), "memory": []}
        raw = self.planner.propose(request)
        raw["used_strategy_ids"] = ["invented"]
        with self.assertRaisesRegex(ValueError, "unknown"):
            decode_proposal(raw, task, self.runner.baseline, self.ev.executor, self.config, [])
        raw["used_strategy_ids"] = []
        raw["graph"]["nodes"][-1]["name"] = "uninstalled_i2v"
        with self.assertRaises(RuntimeError):
            decode_proposal(raw, task, self.runner.baseline, self.ev.executor, self.config, [])

    def test_real_executor_reuses_only_unchanged_upstream(self):
        calls = []
        class FixtureTool(VideoTool):
            def __init__(self, name, output):
                self.name, self.output = name, output

            def run(self, task, plan):
                raise AssertionError("must pass artifacts")

            def run_with_context(self, task, plan, context):
                calls.append(context.node_id)
                return VideoArtifact(str(len(calls)), task.task_id, task.prompt, task.mode, [self.name], [],
                    {"artifact_type": self.output, "upstream_conditioning_consumed": bool(context.input_artifacts)})

        for name, inputs, output in (("mock_text_to_video", (), "video"),
                ("h3_frame_extract", ("video",), "image"), ("h3_ref2va", ("image",), "video")):
            self.ev.tools.register(FixtureTool(name, output), ToolSpec(name=name, capability=name,
                input_types=inputs, output_type=output, consumes_upstream=bool(inputs), backend="builtin"))
        graph = deepcopy(self.runner.baseline)
        graph.nodes += [GraphNode("frame", "tool", "h3_frame_extract", {"position": "last"}),
                        GraphNode("repair", "tool", "h3_ref2va", {"prompt": "first strategy"})]
        graph.edges += [GraphEdge("a", "tool_t2v", "frame"), GraphEdge("b", "frame", "repair")]
        task = self.dataset.train[0]
        self.runner.evaluate(task, graph, 42, "train/replay")
        changed = deepcopy(graph)
        changed.nodes[-1].config["prompt"] = "second strategy"
        result = self.runner.evaluate(task, changed, 42, "train/replay")
        self.assertEqual(result["reused_nodes"], ["tool_t2v", "frame"])
        self.assertEqual(result["executed_nodes"], ["repair"])
        self.assertEqual(calls, ["tool_t2v", "frame", "repair", "repair"])
        changed.nodes[1].config["prompt"] = "new upstream"
        result = self.runner.evaluate(task, changed, 42, "train/replay")
        self.assertEqual(result["reused_nodes"], [])

    def test_api_planner_forbids_filesystem_enabled_codex(self):
        from evovideo_skill.conditioning_planner import ConditioningPlanner
        class Proposer:
            codex = object()
        with self.assertRaisesRegex(ValueError, "filesystem"):
            ConditioningPlanner(Proposer())

    def test_resume_learning_does_not_refit_or_replan(self):
        self.frozen()
        resumed = ConditioningRunner(self.dataset, self.ev, self.planner, self.root / "run", self.config,
                                     {"provider": "local-fake"})
        with patch.object(self.planner, "propose", side_effect=AssertionError("replanned")):
            resumed.learn()
            resumed.validate_and_freeze()
        self.assertEqual(resumed.memory.snapshot(), self.runner.memory.snapshot())

    def test_positive_training_evidence_can_be_validated_and_retrieved(self):
        original = self.ev.rollout
        def fixture(task, graph, **kwargs):
            rollout = original(task, graph, **kwargs)
            score = .8 if any(n.config.get("prompt") for n in graph.nodes) else .5
            rollout.reward = replace(rollout.reward, score=score)
            return rollout
        with patch.object(self.ev, "rollout", side_effect=fixture):
            frozen = self.frozen()
        memory, admitted, _ = self.runner.load_frozen(frozen)
        self.assertTrue(admitted)
        retrieved = self.runner.test_memory(memory, admitted, self.dataset.test[0], "strategy")
        self.assertEqual(retrieved[0]["deployment_status"], "validated")
        self.assertEqual(retrieved[0]["evidence"]["task_support"], 2)
        self.assertTrue(retrieved[0]["evidence"]["contextual_effects"])
        paths = self.runner.test_memory(memory, admitted, self.dataset.test[0], "paths")
        self.assertNotIn("evidence", paths[0])
        self.assertNotIn("strategy", paths[0])


if __name__ == "__main__":
    unittest.main()
