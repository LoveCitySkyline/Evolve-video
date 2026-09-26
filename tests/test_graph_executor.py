from __future__ import annotations

import unittest
from unittest.mock import Mock

from evovideo_skill.evaluators import EvaluatorSuite
from evovideo_skill.graph_executor import (
    ArtifactPassingGraphExecutor,
    GraphExecutionError,
    GraphToolExecutionError,
)
from evovideo_skill.graph_skill import GraphEdge, GraphNode, ToolPathGraph, VideoGenerationState
from evovideo_skill.models import TaskMode, VideoArtifact, VideoPlan, VideoTask
from evovideo_skill.tools import (
    FailureSegmentLocalizerTool,
    SegmentStitcherTool,
    ToolExecutionContext,
    ToolRegistry,
    VideoTool,
    _stable_id,
)
from evovideo_skill.tool_onboarding import CommandToolManifest, DeclarativeCommandVideoTool, ToolSpec


class SourceTool(VideoTool):
    name = "source_tool"

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        return VideoArtifact(
            artifact_id=_stable_id(task.task_id, self.name),
            task_id=task.task_id,
            prompt=task.prompt,
            mode=task.mode,
            tool_chain=[self.name],
            frames=[{"index": 0, "identity": "a", "action": "walk"}],
            metadata={"payload": "source-value"},
        )


class SinkTool(VideoTool):
    name = "sink_tool"

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        raise AssertionError("artifact-aware executor must call run_with_context")

    def run_with_context(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext) -> VideoArtifact:
        upstream = context.latest_artifact
        if upstream is None:
            raise AssertionError("sink did not receive an upstream artifact")
        return VideoArtifact(
            artifact_id=_stable_id(task.task_id, self.name, upstream.artifact_id),
            task_id=task.task_id,
            prompt=task.prompt,
            mode=task.mode,
            tool_chain=[self.name],
            frames=list(upstream.frames),
            metadata={"consumed_artifact": upstream.artifact_id, "payload": upstream.metadata["payload"]},
        )


class FailingSinkTool(VideoTool):
    name = "failing_sink_tool"

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        raise RuntimeError("intentional adapter failure")


class GraphExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.registry.register(SourceTool())
        self.registry.register(SinkTool())
        self.executor = ArtifactPassingGraphExecutor(self.registry, EvaluatorSuite())
        self.task = VideoTask("artifact-flow", "A person walks.")
        self.plan = VideoPlan(
            task_id=self.task.task_id,
            intent={"subject": "person"},
            temporal_steps=["walk"],
            constraints=[],
            selected_skill_names=[],
            tool_chain=[],
            generation_prompt=self.task.prompt,
        )

    def test_executes_all_tool_nodes_and_passes_artifacts(self) -> None:
        graph = ToolPathGraph(
            graph_id="flow",
            skill_name="flow",
            description="source to sink",
            triggers=["generation"],
            nodes=[
                GraphNode("trigger", "trigger", "generation"),
                GraphNode("source", "tool", "source_tool"),
                GraphNode("sink", "tool", "sink_tool"),
            ],
            edges=[
                GraphEdge("e1", "trigger", "source"),
                GraphEdge("e2", "source", "sink"),
            ],
        )

        result = self.executor.execute(self.task, self.plan, graph)

        self.assertEqual(result.executed_tools, ["source_tool", "sink_tool"])
        self.assertEqual(result.artifact.tool_chain, ["source_tool", "sink_tool"])
        self.assertEqual(result.artifact.metadata["consumed_artifact"], result.node_artifacts["source"].artifact_id)
        self.assertEqual(result.artifact.metadata["payload"], "source-value")
        self.assertEqual(set(result.artifact.metadata["artifact_lineage"]), {"source", "sink"})

    def test_localizer_receives_fresh_verifier_evidence_without_explicit_verifier_node(self):
        self.registry.register(SourceTool(), ToolSpec("source_tool", "text_to_video", output_type="video"))
        self.registry.register(FailureSegmentLocalizerTool(), ToolSpec(
            "failed_segment_localizer", "failure_segment_localization",
            input_types=("video",), output_type="segment_plan", consumes_upstream=True,
        ))
        self.registry.register(SinkTool(), ToolSpec(
            "sink_tool", "video_repair", backend="builtin",
            input_types=("segment_plan",), output_type="video",
        ))
        augmenter = Mock()
        def verify(task, artifact):
            artifact.metadata["vlm_evaluation"] = {"failed_segments": [{
                "start_ratio": .75, "end_ratio": 1.0,
                "repair_instruction": "Restore the current task's ending action",
            }]}
            return artifact
        augmenter.augment.side_effect = verify
        graph = ToolPathGraph("localize", "localize", "", [], [
            GraphNode("draft", "tool", "source_tool"),
            GraphNode("plan", "tool", "failed_segment_localizer"),
            GraphNode("output", "tool", "sink_tool"),
        ], [GraphEdge("e", "draft", "plan"), GraphEdge("repair", "plan", "output")])
        result = ArtifactPassingGraphExecutor(self.registry, EvaluatorSuite(), augmenter).execute(
            self.task, self.plan, graph)
        augmenter.augment.assert_called_once()
        segment = result.node_artifacts["plan"].metadata["failure_localization"]["segments"][0]
        self.assertEqual(segment["start_ratio"], .75)
        self.assertEqual(segment["repair_instruction"], "Restore the current task's ending action")

    def test_rejects_cycles_before_execution(self) -> None:
        graph = ToolPathGraph(
            graph_id="cycle",
            skill_name="cycle",
            description="invalid",
            triggers=[],
            nodes=[GraphNode("a", "tool", "source_tool"), GraphNode("b", "tool", "sink_tool")],
            edges=[GraphEdge("e1", "a", "b"), GraphEdge("e2", "b", "a")],
        )
        with self.assertRaises(GraphExecutionError):
            self.executor.execute(self.task, self.plan, graph)

    def test_tool_failure_identifies_exact_node_and_completed_prefix(self) -> None:
        self.registry.register(FailingSinkTool())
        graph = ToolPathGraph(
            "precise-failure", "precise-failure", "source then failure", [],
            [
                GraphNode("source", "tool", "source_tool"),
                GraphNode("failure", "tool", "failing_sink_tool"),
            ],
            [GraphEdge("e", "source", "failure")],
        )

        with self.assertRaises(GraphToolExecutionError) as raised:
            self.executor.execute(self.task, self.plan, graph)

        self.assertEqual(raised.exception.tool_name, "failing_sink_tool")
        self.assertEqual(raised.exception.node_id, "failure")
        self.assertEqual(raised.exception.executed_tools, ["source_tool"])

    def test_verifier_condition_controls_repair_and_forwards_artifact(self) -> None:
        failing_task = VideoTask("repair", "A person walks, then turns.")
        graph = ToolPathGraph(
            graph_id="conditional-repair",
            skill_name="conditional-repair",
            description="repair only failed action alignment",
            triggers=["motion_mismatch"],
            nodes=[
                GraphNode("source", "tool", "source_tool"),
                GraphNode("verify", "verifier", "prompt_action_alignment", {"threshold": 0.8}),
                GraphNode("repair", "tool", "sink_tool"),
            ],
            edges=[
                GraphEdge("e1", "source", "verify"),
                GraphEdge("e2", "verify", "repair", "action_score<threshold"),
            ],
        )

        result = self.executor.execute(failing_task, self.plan, graph)

        self.assertEqual(result.executed_tools, ["source_tool", "sink_tool"])
        self.assertEqual(result.state.verifier_scores["prompt_action_alignment"], 0.5)
        self.assertEqual(result.artifact.metadata["consumed_artifact"], result.node_artifacts["source"].artifact_id)

    def test_verifier_lte_condition_triggers_at_exact_evolution_threshold(self) -> None:
        verifier = GraphNode(
            "verify", "verifier", "prompt_action_alignment", {"threshold": 0.95}
        )
        repair = GraphNode("repair", "tool", "sink_tool")
        graph = ToolPathGraph(
            "threshold", "threshold", "inclusive repair threshold", [],
            [verifier, repair],
            [GraphEdge("repair_on_failure", "verify", "repair", "action_score<=threshold")],
        )
        state = VideoGenerationState("task", "prompt", graph.graph_id)
        state.verifier_scores["prompt_action_alignment"] = 0.95

        self.assertTrue(self.executor._condition_passes(graph.edges[0], graph, state))

    def test_localized_repair_preserves_healthy_frames_and_forces_final_reevaluation(self) -> None:
        source = VideoArtifact(
            "draft",
            self.task.task_id,
            self.task.prompt,
            self.task.mode,
            ["mock_text_to_video"],
            [{"index": index, "pixel_source": f"wan-{index}"} for index in range(6)],
            {
                "artifact_type": "video",
                "vlm_evaluation": {
                    "action_alignment_score": 0.2,
                    "failure_types": ["motion_mismatch"],
                    "failed_segments": [{
                        "start_ratio": 0.33,
                        "end_ratio": 0.67,
                        "failed_criteria": ["action_order"],
                        "diagnosis": "middle action is missing",
                        "repair_instruction": "restore the middle action only",
                    }],
                },
                "vlm_model": "draft-verifier",
                "task_reward": {"score": 0.2},
            },
        )
        localization = FailureSegmentLocalizerTool().run_with_context(
            self.task,
            self.plan,
            ToolExecutionContext("localize", {}, {"verify": source}),
        )
        repaired = VideoArtifact(
            "repair",
            self.task.task_id,
            self.task.prompt,
            self.task.mode,
            ["mock_image_to_video"],
            [{"index": index, "pixel_source": f"repair-{index}"} for index in range(3)],
            {"artifact_type": "video", "upstream_conditioning_consumed": True},
        )

        result = SegmentStitcherTool().run_with_context(
            self.task,
            self.plan,
            ToolExecutionContext(
                "stitch",
                {},
                {"draft": source, "segment_plan": localization, "repair": repaired},
            ),
        )

        self.assertEqual(result.frames[0]["pixel_source"], "wan-0")
        self.assertEqual(result.frames[-1]["pixel_source"], "wan-5")
        self.assertTrue(all(result.frames[index]["segment_repaired"] for index in (1, 2, 3)))
        self.assertNotIn("vlm_evaluation", result.metadata)
        self.assertNotIn("task_reward", result.metadata)
        self.assertTrue(result.metadata["healthy_content_preserved"])
        self.assertFalse(result.metadata["whole_video_regenerated"])

    def test_rejects_upstream_consumer_attached_only_to_trigger(self) -> None:
        self.registry.register(
            SinkTool(),
            ToolSpec(
                name="sink_tool",
                capability="video_editing",
                input_types=("video",),
                output_type="video",
                consumes_upstream=True,
            ),
        )
        graph = ToolPathGraph(
            graph_id="empty-consumer",
            skill_name="empty-consumer",
            description="invalid empty consumer",
            triggers=[],
            nodes=[
                GraphNode("trigger", "trigger", "generation"),
                GraphNode("sink", "tool", "sink_tool"),
            ],
            edges=[GraphEdge("e1", "trigger", "sink")],
        )

        with self.assertRaisesRegex(GraphExecutionError, "no tool/verifier producer"):
            self.executor.execute(self.task, self.plan, graph)

    def test_verifier_preserves_artifact_type_during_static_validation(self) -> None:
        self.registry.register(
            SourceTool(),
            ToolSpec("source_tool", "text_to_video", output_type="video"),
        )
        self.registry.register(
            SinkTool(),
            ToolSpec(
                "sink_tool", "keyframe_to_video", input_types=("keyframes",),
                output_type="video", consumes_upstream=True,
            ),
        )
        graph = ToolPathGraph(
            "typed-verifier", "typed-verifier", "invalid forwarded type", [],
            [
                GraphNode("source", "tool", "source_tool"),
                GraphNode("verify", "verifier", "prompt_action_alignment"),
                GraphNode("sink", "tool", "sink_tool"),
            ],
            [GraphEdge("e1", "source", "verify"), GraphEdge("e2", "verify", "sink")],
        )

        with self.assertRaisesRegex(GraphExecutionError, "forwarded from source"):
            self.executor.validate_graph(graph)

    def test_symbolic_keyframes_cannot_feed_tool_requiring_real_image_binding(self) -> None:
        self.registry.register(
            SourceTool(),
            ToolSpec("source_tool", "keyframe_planning", output_type="keyframes"),
        )
        manifest = CommandToolManifest.from_dict({
            "name": "sink_tool",
            "capability": "image_conditioned_video_generation",
            "input_types": ["keyframes"],
            "output_type": "video",
            "verified": True,
            "consumes_upstream": True,
            "command": ["python", "adapter.py", "{reference_image}", "{output_video}"],
            "input_bindings": {"keyframes": "reference_image"},
        })
        self.registry.register(DeclarativeCommandVideoTool(manifest, "outputs/test_graph_executor"), manifest.spec)
        graph = ToolPathGraph(
            "physical-binding", "physical-binding", "requires pixels", [],
            [GraphNode("source", "tool", "source_tool"), GraphNode("sink", "tool", "sink_tool")],
            [GraphEdge("e", "source", "sink")],
        )

        with self.assertRaisesRegex(GraphExecutionError, "symbolic.*materialized"):
            self.executor.validate_graph(graph)


if __name__ == "__main__":
    unittest.main()
