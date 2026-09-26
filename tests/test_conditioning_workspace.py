from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evovideo_skill.conditioning_workspace import (ConditioningWorkspace, artifact_view,
    planner_project, repair_impact)
from evovideo_skill.conditioning_cache import ConditioningNodeCache
from evovideo_skill.evaluators import EvaluatorSuite
from evovideo_skill.graph_evolver import validate_h3_node_configs
from evovideo_skill.graph_executor import ArtifactPassingGraphExecutor, GraphToolExecutionError
from evovideo_skill.graph_skill import GraphNode, GraphEdge, ToolPathGraph
from evovideo_skill.models import VideoTask, VideoArtifact, VideoPlan
from evovideo_skill.tools import VideoTool, ToolRegistry
from evovideo_skill.tool_onboarding import ToolSpec


class Source(VideoTool):
    name = "mock_text_to_video"

    def __init__(self):
        self.calls = 0

    def run(self, task, plan):
        self.calls += 1
        return VideoArtifact("clip", task.task_id, task.prompt, task.mode, [self.name], [], {
            "artifact_type": "video", "h3_conditioning": [{"id": "actor", "kind": "image",
                "uri": "/reference.png", "role": "reference_image", "api_key": "not-for-observer"}],
            "h3_request_hash": "request", "upstream_conditioning_consumed": True,
            "vlm_evaluation": {"secret_final_label": 1}})


class Broken(VideoTool):
    name = "broken"

    def run(self, task, plan):
        raise RuntimeError("decoder failed")


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = VideoTask("task", "A person walks.")
        self.graph = ToolPathGraph("g", "g", "", [], [
            GraphNode("source", "tool", "mock_text_to_video"),
            GraphNode("sink", "tool", "broken")], [GraphEdge("link", "source", "sink")])
        self.plan = VideoPlan(self.task.task_id, {}, [], [], [], [], self.task.prompt)
        self.registry = ToolRegistry()
        self.source = Source()
        self.registry.register(self.source)
        self.registry.register(Broken(), ToolSpec("broken", "test failing video processor",
            input_types=("video",), output_type="video", consumes_upstream=True, backend="builtin"))
        self.executor = ArtifactPassingGraphExecutor(self.registry, EvaluatorSuite())

    def workspace(self, graph=None):
        return ConditioningWorkspace(self.root, "evaluation", self.task, graph or self.graph, 1, "train/task")

    def test_downstream_failure_keeps_completed_prefix_and_cache(self):
        for attempt in range(2):
            workspace = self.workspace()
            cache = ConditioningNodeCache(self.root / "cache", {}, lambda *args: None)
            try:
                self.executor.execute(self.task, self.plan, self.graph, node_cache=cache, observer=workspace)
            except GraphToolExecutionError as exc:
                workspace.failed(exc)
            state = json.loads(workspace.path.read_text())
            self.assertEqual(state["nodes"]["source"]["status"], "completed")
            self.assertEqual(state["nodes"]["sink"]["status"], "failed")
            self.assertEqual(state["failed_stage"], "execution")
            self.assertIsNone(state["quality"])
            evidence = state["nodes"]["source"]["evidence"]
            self.assertEqual(evidence["cache_hit"], bool(attempt))
            self.assertEqual(evidence["artifact_state"]["semantic_validity"], "unverified")
            self.assertNotIn("not-for-observer", workspace.path.read_text())
            events = [json.loads(line) for line in (workspace.root / "events.jsonl").read_text().splitlines()]
            self.assertEqual([e["sequence"] for e in events], list(range(1, len(events) + 1)))
        self.assertEqual(self.source.calls, 1)
        self.assertEqual(len(list((self.root / "evaluation").glob("*/state.json"))), 2)

    def test_generated_and_verified_are_distinct(self):
        graph = deepcopy(self.graph)
        graph.nodes.pop()
        graph.edges.clear()
        workspace = self.workspace(graph)
        self.executor.execute(self.task, self.plan, graph, observer=workspace)
        self.assertEqual(workspace.state["stage"], "verification")
        self.assertIsNone(workspace.state["quality"])
        workspace.verified({"score": .2, "criterion_scores": {"identity": .2}})
        self.assertEqual(workspace.state["quality"]["selection_status"], "not_decided_here")
        self.assertEqual(workspace.state["nodes"]["source"]["artifact"]["semantic_validity"], "unverified")

    def test_verifier_error_does_not_blame_completed_tool(self):
        workspace = self.workspace()
        artifact = self.source.run(self.task, self.plan)
        workspace.completed("source", artifact, {})
        workspace.generated(artifact)
        workspace.failed(RuntimeError("unobserved"), "needs_review")
        self.assertEqual(workspace.state["nodes"]["source"]["status"], "completed")
        self.assertEqual(workspace.state["nodes"]["sink"]["status"], "not_executed")
        self.assertEqual(workspace.state["failed_stage"], "verification")

    def test_artifact_view_excludes_scores_and_unknown_metadata(self):
        data = artifact_view(self.source.run(self.task, self.plan))
        self.assertTrue(data["adapter_reports_consumption"])
        self.assertEqual(data["submitted_conditioning"][0]["id"], "actor")
        self.assertNotIn("secret_final_label", json.dumps(data))
        self.assertNotIn("api_key", json.dumps(data))

    def record(self, episode="train/task"):
        return {"status": "ok", "task_id": "task", "episode": episode, "graph_id": "g",
            "evaluation_id": "ev", "seed": 1, "feedback": {"nodes": {}}, "score": 1,
            "final_score": "must-not-be-visible"}

    def test_project_is_episode_scoped_not_a_global_memory_query(self):
        data = planner_project(self.task, self.graph, [self.record()], "factorial/0")
        self.assertTrue(data["has_measured_video"])
        self.assertNotIn("must-not-be-visible", json.dumps(data))
        for operation in ("test/direct/strategy/task/1/0", "validation/strategy/task"):
            with self.assertRaises(ValueError):
                planner_project(self.task, self.graph, [self.record()], operation)
        with self.assertRaises(ValueError):
            planner_project(self.task, self.graph, [self.record("comparison/direct/strategy/task/1")], "train/0")
        with self.assertRaises(ValueError):
            planner_project(self.task, self.graph, [self.record("test/adaptive/strategy/task/2")],
                            "test/adaptive/strategy/task/1/1")
        data = planner_project(self.task, self.graph, [self.record("test/adaptive/strategy/task/1")],
                               "test/adaptive/strategy/task/1/1")
        self.assertTrue(data["has_measured_video"])
        foreign = {**self.record(), "task_id": "other"}
        with self.assertRaises(ValueError):
            planner_project(self.task, self.graph, [foreign], "train/0")

    def test_direct_project_has_no_fabricated_measurement(self):
        self.task.metadata["h3_references"] = [{"id": "actor", "kind": "image", "uri": "/actor.png"}]
        data = planner_project(self.task, self.graph, [], "test/direct/strategy/task/1/0")
        self.assertFalse(data["has_measured_video"])
        self.assertEqual(data["references"][0]["id"], "actor")

    def test_project_reference_ids_match_bank_uri_normalization(self):
        path = self.root / "source.mp4"
        self.task.reference_video = str(path)
        self.task.metadata["h3_references"] = [{"id": "existing", "kind": "video", "uri": path.as_uri()}]
        data = planner_project(self.task, self.graph, [], "validation/strategy/task")
        self.assertEqual([r["id"] for r in data["references"]], ["existing"])
        self.task.metadata["h3_references"].append({"id": "source-video", "kind": "video", "uri": "/different.mp4"})
        with self.assertRaisesRegex(ValueError, "collides"):
            planner_project(self.task, self.graph, [], "validation/strategy/task")

    def test_repair_closure_preserves_independent_branch(self):
        self.graph.nodes.extend([GraphNode("independent", "tool", "mock_text_to_video"),
                                 GraphNode("join", "tool", "h3_av_concat")])
        self.graph.edges.extend([GraphEdge("left", "sink", "join"), GraphEdge("right", "independent", "join")])
        child = deepcopy(self.graph)
        child.nodes[1].config["reference_ids"] = ["actor"]
        result = repair_impact(self.graph, child)
        self.assertEqual(result["affected_nodes"], ["join", "sink"])
        self.assertEqual(result["reusable_candidates"], ["independent", "source"])
        child = deepcopy(self.graph)
        child.edges.pop(0)
        result = repair_impact(self.graph, child)
        self.assertIn("sink", result["affected_nodes"])
        self.assertIn("join", result["affected_nodes"])
        child = deepcopy(self.graph)
        child.edges.reverse()
        self.assertIn("join", repair_impact(self.graph, child)["affected_nodes"])

    def test_single_input_native_tool_rejected_before_generation(self):
        graph = ToolPathGraph("g", "g", "", [], [
            GraphNode("one", "tool", "h3_t2va"), GraphNode("two", "tool", "h3_t2va"),
            GraphNode("extract", "tool", "h3_frame_extract", {"position": "first"})],
            [GraphEdge("a", "one", "extract"), GraphEdge("b", "two", "extract")])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            validate_h3_node_configs(graph, {"h3_t2va", "h3_fl2va", "h3_ref2va", "h3_frame_extract"}, local=True)

    def test_native_pack_rejects_video_bound_as_image(self):
        graph = ToolPathGraph("g", "g", "", [], [GraphNode("clip", "tool", "h3_t2va"),
            GraphNode("pack", "tool", "h3_reference_pack", {"bindings": [
                {"source": "clip", "kind": "image", "role": "first_frame"}]})],
            [GraphEdge("a", "clip", "pack")])
        with self.assertRaisesRegex(ValueError, "does not match producer"):
            validate_h3_node_configs(graph, {"h3_t2va", "h3_ref2va", "h3_fl2va", "h3_reference_pack"}, local=True)


if __name__ == "__main__":
    unittest.main()
