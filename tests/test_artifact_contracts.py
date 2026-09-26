from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evovideo_skill.artifact_contracts import ArtifactContract, check_contracts
from evovideo_skill.evaluators import EvaluatorSuite
from evovideo_skill.graph_executor import ArtifactPassingGraphExecutor
from evovideo_skill.graph_evolver import GraphPathCandidate, GraphToolPathEvolver
from evovideo_skill.graph_skill import GraphEdge, GraphNode, GraphSkillMemory, ToolPathGraph
from evovideo_skill.models import TaskMode, VideoArtifact, VideoPlan, VideoTask
from evovideo_skill.skill_memory import SkillMemory
from evovideo_skill.tool_onboarding import ToolSpec
from evovideo_skill.tools import ArtifactBridgeTool, ToolExecutionContext, ToolRegistry, VideoTool


class NoopTool(VideoTool):
    def __init__(self, name: str):
        self.name = name

    def run(self, task, plan):
        raise AssertionError("static contract test must not execute repository tools")


class MaterializedImageSink(VideoTool):
    name = "materialized_image_sink"

    def run(self, task, plan):
        raise AssertionError("artifact-aware execution is required")

    def run_with_context(self, task, plan, context):
        upstream = context.latest_artifact
        reference = Path(str(upstream.metadata.get("reference_image")))
        if not reference.is_file():
            raise AssertionError("sink did not receive a materialized image")
        return VideoArtifact(
            "sink", task.task_id, task.prompt, task.mode, [self.name], [],
            {
                "artifact_type": "video",
                "reference_consumed": str(reference),
                "upstream_conditioning_consumed": True,
            },
        )


class ArtifactContractTests(unittest.TestCase):
    def test_direct_contract_is_preferred_over_an_available_bridge(self) -> None:
        produced = ArtifactContract("video", transport=("local_path",), materialized=True)
        accepted = [
            ArtifactContract("image", semantic_role="first_frame", materialized=True),
            ArtifactContract("video", semantic_role="source_video", materialized=True),
        ]

        check = check_contracts(produced, accepted)

        self.assertTrue(check.compatible)
        self.assertEqual(check.target.artifact_type, "video")

    def test_video_to_first_frame_contract_requests_executable_bridge(self) -> None:
        check = check_contracts(
            ArtifactContract("video", formats=("mp4",), transport=("local_path",), materialized=True),
            [ArtifactContract(
                "image", semantic_role="first_frame", formats=("png",),
                transport=("local_path",), width=64, height=32, materialized=True,
            )],
        )

        self.assertFalse(check.compatible)
        self.assertEqual(check.bridges[0][0], "bridge_extract_reference_frame")

    def test_symbolic_keyframes_request_materialization_bridge(self) -> None:
        check = check_contracts(
            ArtifactContract("keyframes", transport=("memory",), materialized=False),
            [ArtifactContract(
                "image", semantic_role="first_frame", formats=("png",),
                transport=("local_path",), materialized=True,
                required_bindings=("reference_image",),
            )],
        )

        self.assertFalse(check.compatible)
        self.assertEqual(check.bridges[0][0], "bridge_materialize_image")

    def test_evolver_inserts_contract_bridge_into_candidate_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry.with_artifact_tools()
            registry.register(
                NoopTool("source_video"),
                ToolSpec(
                    "source_video", "text_to_video", output_type="video",
                    output_bindings=("reference_video", "reference_image"), verified=True,
                ),
            )
            registry.register(
                NoopTool("first_frame_i2v"),
                ToolSpec(
                    "first_frame_i2v", "image_conditioned_video_generation",
                    input_types=("image",), output_type="video", consumes_upstream=True, verified=True,
                    input_contracts=({
                        "artifact_type": "image", "semantic_role": "first_frame",
                        "formats": ["png"], "transport": ["local_path"],
                        "width": 64, "height": 32, "materialized": True,
                        "required_bindings": ["reference_image"],
                    },),
                ),
            )
            evolver = GraphToolPathEvolver(
                SkillMemory(Path(tmpdir) / "skills"),
                GraphSkillMemory(Path(tmpdir) / "graphs"),
                tools=registry,
                evaluators=EvaluatorSuite(),
            )
            graph = ToolPathGraph(
                "contract-path", "contract-path", "contract path", [],
                [GraphNode("source", "tool", "source_video"), GraphNode("sink", "tool", "first_frame_i2v")],
                [GraphEdge("raw", "source", "sink")],
            )
            candidate = GraphPathCandidate(graph, [], "align contracts", [])

            executable = evolver._executable_candidates([candidate])

            self.assertEqual(executable, [candidate])
            self.assertNotIn("raw", {edge.edge_id for edge in graph.edges})
            self.assertIn("bridge_extract_reference_frame", graph.tool_names())
            self.assertTrue(graph.stats["artifact_contract_repairs"])

    def test_evolver_inserts_task_reference_video_root_for_editing_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry.with_artifact_tools()
            registry.register(
                NoopTool("source_video_editor"),
                ToolSpec(
                    "source_video_editor", "global_video_editing",
                    input_types=("video",), output_type="video",
                    consumes_upstream=True, verified=True,
                ),
            )
            evolver = GraphToolPathEvolver(
                SkillMemory(Path(tmpdir) / "skills"),
                GraphSkillMemory(Path(tmpdir) / "graphs"),
                tools=registry,
                evaluators=EvaluatorSuite(),
            )
            graph = ToolPathGraph(
                "source-edit", "source-edit", "edit the supplied source video", [],
                [GraphNode("editor", "tool", "source_video_editor")],
                [],
                stats={
                    "planning_context": {
                        "source_video_available": True,
                        "reference_video_task_ids": ["edit-task"],
                    }
                },
            )
            candidate = GraphPathCandidate(graph, [], "source-conditioned edit", [])

            executable = evolver._executable_candidates([candidate])

            self.assertEqual(executable, [candidate])
            self.assertIn("task_reference_video", graph.tool_names())
            source = next(node for node in graph.nodes if node.name == "task_reference_video")
            self.assertTrue(any(
                edge.source == source.node_id and edge.target == "editor"
                for edge in graph.edges
            ))
            self.assertTrue(graph.stats["automatic_graph_repairs"])

    def test_frame_bridge_materializes_target_resolution_and_format(self) -> None:
        try:
            import cv2
            import numpy as np
        except Exception as exc:  # pragma: no cover
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "frame.jpg"
            cv2.imwrite(str(source), np.zeros((20, 30, 3), dtype=np.uint8))
            task = VideoTask("bridge", "A subject moves.")
            plan = VideoPlan(task.task_id, {}, ["move"], [], [], [], task.prompt)
            upstream = VideoArtifact(
                "upstream", task.task_id, task.prompt, TaskMode.GENERATION, ["source"], [],
                {"artifact_type": "video", "sampled_frame_paths": [str(source)]},
            )
            target = {
                "artifact_type": "image", "semantic_role": "first_frame", "formats": ["png"],
                "transport": ["local_path"], "width": 64, "height": 32, "materialized": True,
            }
            tool = ArtifactBridgeTool("bridge_extract_reference_frame", root / "bridges")

            result = tool.run_with_context(
                task, plan, ToolExecutionContext("bridge_node", {"target_contract": target}, {"source": upstream})
            )

            output = Path(result.metadata["reference_image"])
            image = cv2.imread(str(output))
            self.assertEqual(output.suffix, ".png")
            self.assertEqual((image.shape[1], image.shape[0]), (64, 32))
            self.assertEqual(result.metadata["artifact_contract"]["semantic_role"], "first_frame")

    def test_symbolic_keyframes_are_materialized_as_a_local_storyboard(self) -> None:
        try:
            import cv2
        except Exception as exc:  # pragma: no cover
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task = VideoTask("keyframes", "A woman walks, turns, and waves.")
            plan = VideoPlan(
                task.task_id,
                {"subject": "woman", "clothing_color": "red"},
                ["walk", "turn", "wave"], [], [], [], task.prompt,
            )
            upstream = VideoArtifact(
                "symbolic", task.task_id, task.prompt, TaskMode.GENERATION,
                ["keyframe_generator"], [],
                {"artifact_type": "keyframes", "temporal_states": plan.temporal_steps},
            )
            target = {
                "artifact_type": "image", "semantic_role": "first_frame",
                "formats": ["png"], "transport": ["local_path"],
                "width": 320, "height": 180, "materialized": True,
                "required_bindings": ["reference_image"],
            }
            tool = ArtifactBridgeTool("bridge_materialize_image", root / "bridges")

            result = tool.run_with_context(
                task, plan, ToolExecutionContext("materialize", {"target_contract": target}, {"source": upstream})
            )

            output = Path(result.metadata["reference_image"])
            image = cv2.imread(str(output))
            self.assertTrue(output.is_file())
            self.assertEqual((image.shape[1], image.shape[0]), (320, 180))
            self.assertEqual(result.metadata["materialization_mode"], "semantic_storyboard")

    def test_evolver_materializes_keyframes_before_real_image_adapter(self) -> None:
        try:
            import cv2  # noqa: F401
        except Exception as exc:  # pragma: no cover
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry = ToolRegistry.with_artifact_tools(root / "bridges")
            registry.register(
                MaterializedImageSink(),
                ToolSpec(
                    "materialized_image_sink", "image_conditioned_video_generation",
                    input_types=("image",), output_type="video", consumes_upstream=True,
                    input_contracts=({
                        "artifact_type": "image", "semantic_role": "first_frame",
                        "formats": ["png"], "transport": ["local_path"],
                        "materialized": True, "required_bindings": ["reference_image"],
                    },),
                ),
            )
            evolver = GraphToolPathEvolver(
                SkillMemory(root / "skills"), GraphSkillMemory(root / "graphs"),
                tools=registry, evaluators=EvaluatorSuite(),
            )
            graph = ToolPathGraph(
                "keyframe-path", "keyframe-path", "materialize keyframes", [],
                [
                    GraphNode("plan", "tool", "temporal_decomposer"),
                    GraphNode("keyframes", "tool", "keyframe_generator"),
                    GraphNode("sink", "tool", "materialized_image_sink"),
                ],
                [
                    GraphEdge("plan_to_keyframes", "plan", "keyframes"),
                    GraphEdge("keyframes_to_sink", "keyframes", "sink"),
                ],
            )
            candidate = GraphPathCandidate(graph, [], "materialize", [])

            executable = evolver._executable_candidates([candidate])
            result = ArtifactPassingGraphExecutor(registry, EvaluatorSuite()).execute(
                VideoTask("materialized-flow", "A woman walks and waves."),
                VideoPlan(
                    "materialized-flow", {"subject": "woman"}, ["walk", "wave"],
                    [], [], [], "A woman walks and waves.",
                ),
                graph,
            )

            self.assertEqual(executable, [candidate])
            self.assertIn("bridge_materialize_image", graph.tool_names())
            self.assertTrue(Path(result.artifact.metadata["reference_consumed"]).is_file())


if __name__ == "__main__":
    unittest.main()
