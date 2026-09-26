from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from evovideo_skill.api_tools import VideoApiError
from evovideo_skill.evaluators import EvaluatorSuite
from evovideo_skill.graph_executor import ArtifactPassingGraphExecutor, GraphExecutionError
from evovideo_skill.graph_skill import GraphEdge, GraphNode, ToolPathGraph
from evovideo_skill.h3_api import H3Client, H3Config, H3GenerationTool, H3SubmissionUnknown, H3PollingInterrupted, build_request, register_h3_tools
from evovideo_skill.models import VideoPlan, VideoTask
from evovideo_skill.tools import ToolExecutionContext, ToolRegistry


class FakeHTTP:
    def __init__(self, url="https://example.test/output.mp4", statuses=None):
        self.url = url
        self.statuses = list(statuses or ["succeeded"])
        self.calls = []
        self.count = 0

    def request_json(self, method, url, headers, payload=None):
        self.calls.append((method, url, payload))
        if method == "POST":
            self.count += 1
            return {"task_id": f"job-{self.count}"}
        state = self.statuses.pop(0) if self.statuses else "succeeded"
        return {"task": {"status": state, "content": {"url": self.url}, "error": {"message": "rejected"}}}


def reference(kind="image", role="reference_image", ident="hero"):
    return {"id": ident, "kind": kind, "role": role, "uri": f"https://example.test/{ident}", "duration_seconds": 4}


class H3RequestTests(unittest.TestCase):
    def test_mode_wire_formats(self):
        refs = [reference(), reference("video", "reference_video", "motion"), reference("audio", "reference_audio", "voice")]
        req = build_request("Keep identity and motion", "ref2va", refs, 6)
        self.assertEqual([r["type"] for r in req["content"]], ["text", "image_url", "video_url", "audio_url"])
        self.assertNotIn("seed", req)
        self.assertEqual(build_request("Ending", "fl2va", [reference(role="last_frame")], 6)["ratio"], "adaptive")

    def test_invalid_requests_fail_before_network(self):
        cases = [
            ("t2va", [reference()]),
            ("fl2va", []),
            ("ref2va", [reference(role="first_frame"), reference(ident="style")]),
            ("ref2va", [reference("audio", "reference_audio", "voice")]),
            ("ref2va", [reference(), reference()]),
            ("ref2va", [reference(ident=str(n)) for n in range(10)]),
        ]
        for mode, refs in cases:
            with self.subTest(mode=mode, refs=refs), self.assertRaises(VideoApiError):
                build_request("Generate", mode, refs, 6)
        with self.assertRaises(VideoApiError):
            build_request("Generate", "t2va", [], 6, ratio="adaptive")
        with self.assertRaises(VideoApiError):
            build_request("Generate", "t2va", [], 3)

    def test_reference_duration_budget(self):
        refs = [reference("video", "reference_video", str(n)) for n in range(2)]
        refs[0]["duration_seconds"] = 12
        with self.assertRaisesRegex(VideoApiError, "total reference video"):
            build_request("Move", "ref2va", refs, 6)
        refs[0].pop("duration_seconds")
        with self.assertRaisesRegex(VideoApiError, "requires duration_seconds"):
            build_request("Move", "ref2va", refs, 6)


class H3LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = H3Config(output_dir=self.temp.name, max_api_calls=2, poll_interval_seconds=0)
        self.payload = build_request("A rolling ball", "t2va", [], 6)

    def test_reuse_and_replicate_budget(self):
        http = FakeHTTP()
        client = H3Client(self.config, "test-key", http)
        first = client.generate(self.payload, {"replicate_label": 1})
        second = client.generate(self.payload, {"replicate_label": 1})
        self.assertEqual(first[:2], second[:2])
        client.generate(self.payload, {"replicate_label": 2})
        with self.assertRaisesRegex(VideoApiError, "budget exhausted"):
            client.generate(self.payload, {"replicate_label": 3})
        self.assertEqual(http.count, 2)
        self.assertNotIn("test-key", "".join(path.read_text() for path in client.jobs.glob("*.json")))

    def test_poll_resume_never_resubmits(self):
        http = FakeHTTP(statuses=["running"])
        client = H3Client(self.config, "test-key", http)
        original = http.request_json
        def interrupted(method, *args, **kwargs):
            if method == "GET":
                raise TimeoutError("temporary poll failure")
            return original(method, *args, **kwargs)
        with patch.object(http, "request_json", side_effect=interrupted), self.assertRaises(H3PollingInterrupted):
            client.generate(self.payload, {})
        http.statuses = ["succeeded"]
        client.generate(self.payload, {})
        self.assertEqual(http.count, 1)

    def test_unknown_post_outcome_requires_reconciliation(self):
        http = FakeHTTP()
        client = H3Client(self.config, "test-key", http)
        with patch.object(http, "request_json", side_effect=TimeoutError("unknown outcome")), self.assertRaises(H3SubmissionUnknown):
            client.generate(self.payload, {})
        with self.assertRaisesRegex(VideoApiError, "outcome unknown"):
            client.generate(self.payload, {})
        self.assertEqual(http.count, 0)

    def test_terminal_failure_not_resubmitted(self):
        http = FakeHTTP(statuses=["failed"])
        client = H3Client(self.config, "test-key", http)
        for _ in range(2):
            with self.assertRaises(VideoApiError):
                client.generate(self.payload, {})
        self.assertEqual(http.count, 1)


class H3ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / "fixture.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=64x64:r=24:d=4",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=4", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", str(self.video)], check=True, capture_output=True)
        self.http = FakeHTTP(self.video.as_uri())
        self.client = H3Client(H3Config(output_dir=str(self.root / "out"), poll_interval_seconds=0), "key", self.http)
        self.registry = ToolRegistry()
        register_h3_tools(self.registry, self.client)
        self.task = VideoTask("task", "A red object moves", duration_seconds=4, metadata={"generation_seed": 123})
        self.plan = VideoPlan("task", {}, ["move"], [], [], [], self.task.prompt)

    def test_real_artifact_passing_t2va_frame_fl2va(self):
        graph = ToolPathGraph("h3-test", "h3-test", "Test", ["generation"], [
            GraphNode("draft", "tool", "h3_t2va"),
            GraphNode("frame", "tool", "h3_frame_extract", {"position": "last", "role": "last_frame"}),
            GraphNode("repair", "tool", "h3_fl2va"),
        ], [GraphEdge("a", "draft", "frame"), GraphEdge("b", "frame", "repair")])
        result = ArtifactPassingGraphExecutor(self.registry, EvaluatorSuite()).execute(self.task, self.plan, graph)
        posts = [call[2] for call in self.http.calls if call[0] == "POST"]
        self.assertEqual(len(posts), 2)
        self.assertEqual(posts[1]["content"][1]["role"], "last_frame")
        self.assertTrue(posts[1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertTrue(result.artifact.metadata["has_audio"])
        self.assertEqual(result.artifact.metadata["replicate_label"], 123)
        self.assertEqual(len(result.node_artifacts), 3)
        self.assertTrue(result.artifact.metadata["sampled_frame_paths"])

    def test_localized_repair_renders_segment_and_marks_timeline(self):
        draft = self.registry.get("h3_t2va").run(self.task, self.plan)
        frame = self.registry.get("h3_frame_extract").run_with_context(
            self.task, self.plan, ToolExecutionContext("frame", {"position": "first"}, {"draft": draft}))
        segment = {"start_ratio": .25, "end_ratio": .75,
                   "diagnosis": "middle transition absent", "repair_instruction": "finish the middle transition"}
        frame.metadata.update(failure_localization={"segments": [segment]}, repair_instruction=segment["repair_instruction"])
        result = self.registry.get("h3_fl2va").run_with_context(
            self.task, self.plan, ToolExecutionContext("repair", {"conditioning_strategy": "localized_repair"}, {"frame": frame}))
        self.assertEqual(result.metadata["repair_timeline"], "segment")
        self.assertEqual(result.metadata["repair_segment"], segment)
        self.assertIn("ONLY a replacement segment", result.metadata["generation_prompt"])
        self.assertIn("middle transition absent", result.metadata["generation_prompt"])

    def test_localized_repair_without_timeline_fails_before_submission(self):
        with self.assertRaisesRegex(VideoApiError, "failure_localization"):
            self.registry.get("h3_t2va").run_with_context(self.task, self.plan,
                ToolExecutionContext("repair", {"conditioning_strategy": "localized_repair"}))
        self.assertEqual(self.http.count, 0)

    def test_direct_baseline_uses_task_references(self):
        task = replace(self.task, reference_video=str(self.video))
        artifact = self.registry.get("mock_text_to_video").run(task, self.plan)
        self.assertEqual(artifact.metadata["h3_mode"], "ref2va")
        self.assertEqual(artifact.metadata["h3_conditioning"][0]["id"], "source-video")

    def test_repaired_terminal_sends_task_duration_to_model(self):
        from types import SimpleNamespace
        from evovideo_skill.graph_evolver import GraphPathCandidate, GraphToolPathEvolver
        from evovideo_skill.graph_skill import GraphSkillMemory
        from evovideo_skill.skill_memory import SkillMemory

        task = replace(self.task, duration_seconds=9)
        graph = ToolPathGraph("duration", "duration", "", [], [
            GraphNode("final", "tool", "h3_t2va", {"duration_seconds": 8}),
        ], [])
        evolver = GraphToolPathEvolver(SkillMemory(self.root / "skills"),
                                      GraphSkillMemory(self.root / "graphs"), tools=self.registry)
        candidates = evolver._prepare_executable_candidates(
            [GraphPathCandidate(graph, [], "duration test", [])], [SimpleNamespace(task=task)])
        self.assertEqual(len(candidates), 1)
        self.registry.get("h3_t2va").run_with_context(
            task, self.plan, ToolExecutionContext("final", candidates[0].graph.node("final").config))
        payload = next(c[2] for c in self.http.calls if c[0] == "POST")
        self.assertEqual(payload["duration"], 9)
        # The test server deliberately returns the original four-second fixture;
        # production output-duration validation is tested separately below.

    def test_discovery_prompt_is_not_injected_into_heldout_task(self):
        from evovideo_skill.h3_prompt_binding import task_prompt_key

        training = replace(self.task, prompt="A chef cuts a tomato then stirs the slices.")
        validation = replace(self.task, prompt="A person folds and throws a paper airplane.")
        config = {"prompt": "Keep the tomato out of the pan until cutting finishes.",
                  "prompt_task_hashes": [task_prompt_key(training)],
                  "conditioning_strategy": "ordered_actions"}
        plan = replace(self.plan, generation_prompt=validation.prompt)
        result = self.registry.get("h3_t2va").run_with_context(
            validation, plan, ToolExecutionContext("generate", config))
        wire_prompt = next(c[2]["content"][0]["text"] for c in self.http.calls if c[0] == "POST")
        self.assertIn("paper airplane", wire_prompt)
        self.assertIn("Complete each action", wire_prompt)
        self.assertNotIn("tomato", wire_prompt)
        self.assertNotIn("pan", wire_prompt)
        self.assertFalse(result.metadata["h3_prompt_binding"]["literal_applied"])
        from evovideo_skill.h3_prompt_binding import stage_instruction
        self.assertIn("tomato", stage_instruction(training, config)[0])
        # A reused ID with a changed prompt cannot revive discovery literals.
        self.assertFalse(stage_instruction(validation, config)[1])

    def test_reusable_shots_concat_and_duration_guard(self):
        task = replace(self.task, duration_seconds=8, metadata={"h3_global_constraints": "Preserve the red object.",
            "h3_shots": [{"prompt": "First scene", "duration_seconds": 4}, {"prompt": "Second scene", "duration_seconds": 4}]})
        graph = ToolPathGraph("shots", "shots", "Two shots", ["generation"], [
            GraphNode("one", "tool", "h3_t2va", {"shot_index": 0}),
            GraphNode("two", "tool", "h3_t2va", {"shot_index": 1}),
            GraphNode("join", "tool", "h3_av_concat", {"source_nodes": ["one", "two"]}),
        ], [GraphEdge("one-join", "one", "join"), GraphEdge("two-join", "two", "join")])
        executor = ArtifactPassingGraphExecutor(self.registry, EvaluatorSuite())
        result = executor.execute(task, self.plan, graph)
        self.assertTrue(result.artifact.metadata["has_audio"])
        self.assertAlmostEqual(result.artifact.metadata["duration_seconds"], 8, delta=.2)
        posts = [call[2] for call in self.http.calls if call[0] == "POST"]
        self.assertIn("First scene", posts[0]["content"][0]["text"])
        self.assertNotIn("Second scene", posts[0]["content"][0]["text"])
        with self.assertRaisesRegex(GraphExecutionError, "does not match"):
            executor.execute(replace(task, duration_seconds=10), self.plan, graph)

    def test_long_direct_baseline_is_explicit_independent_shot_calls(self):
        task = replace(self.task, duration_seconds=16, metadata={
            "h3_shots": [{"prompt": f"Stage {index}", "duration_seconds": 4} for index in range(4)]})
        result = self.registry.get("mock_text_to_video").run(task, self.plan)
        self.assertEqual(self.http.count, 4)
        self.assertEqual(result.metadata["direct_generation_calls"], 4)
        self.assertEqual(result.metadata["baseline_protocol"], "independent_native_shot_calls_then_av_concat")
        self.assertTrue(result.metadata["has_audio"])
        self.assertAlmostEqual(result.metadata["duration_seconds"], 16, delta=.2)


if __name__ == "__main__":
    unittest.main()
