from __future__ import annotations

import errno
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from evovideo_skill.api_tools import VideoApiError
from evovideo_skill.evaluators import EvaluatorSuite
from evovideo_skill.graph_executor import ArtifactPassingGraphExecutor
from evovideo_skill.graph_skill import GraphEdge, GraphNode, ToolPathGraph
from evovideo_skill.h3_api import H3BudgetExceeded, H3PollingInterrupted, H3SubmissionUnknown, register_h3_tools
from evovideo_skill.h3_local import H3LocalClient, H3LocalConfig, build_local_request
from evovideo_skill.models import VideoPlan, VideoTask
from evovideo_skill.tools import ToolRegistry


class LocalHTTP:
    def __init__(self):
        self.calls = []
        self.status = "completed"

    def request_json(self, method, url, headers, payload=None):
        self.calls.append((method, url, payload, headers))
        if method == "POST":
            return {"id": "video-same-id", "status": "queued"}
        if url.endswith("/health"):
            return {"status": "ok"}
        if url.endswith("/model_info"):
            return {"architectures": ["MiniMaxH3Pipeline"], "model_path": "/models/H3"}
        if url.endswith("/server_info"):
            return {"version": "fixture"}
        return {"id": "video-same-id", "status": self.status, "error": "fixture failure"}

    @property
    def posts(self):
        return [call for call in self.calls if call[0] == "POST"]


class LocalH3Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.video = self.root / "source.mp4"
        self.image = self.root / "first.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=blue:s=64x64:r=24:d=4",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=4", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", str(self.video)], check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(self.video), "-frames:v", "1", str(self.image)], check=True, capture_output=True)
        self.http = LocalHTTP()
        self.config = H3LocalConfig(output_dir=str(self.root / "out"), poll_interval_seconds=0, max_api_calls=5)
        self.client = H3LocalClient(self.config, self.http)
        self.payload = build_local_request("A blue object moves", "t2va", [], 4)
        self.download = patch("evovideo_skill.h3_local._local_open", side_effect=lambda *a, **k: io.BytesIO(self.video.read_bytes()))
        self.download.start()
        self.addCleanup(self.download.stop)

    def ref(self, role="first_frame", kind="image"):
        return {"id": "hero", "kind": kind, "role": role, "uri": str(self.image if kind == "image" else self.video)}

    def test_native_payload_no_cloud_transport(self):
        request = build_local_request("Move", "fl2va", [self.ref("last_frame")], 4)
        self.assertEqual(request["conditions"], [{"type": "image", "uri": self.image.as_uri(), "role": "keyframe", "frame_index": -1}])
        self.assertEqual(request["target"]["aspect_ratio"], "auto")
        self.assertNotIn("content", request)
        self.assertNotIn("seed", request)
        request = build_local_request("Use motion", "ref2va", [self.ref("reference_video", "video")], 4)
        self.assertEqual(request["conditions"][0]["role"], "reference")
        self.assertIn("<Audio 1>", request["prompt"])
        for resolution in ("2K", "720P"):
            with self.assertRaises(VideoApiError):
                build_local_request("Move", "t2va", [], 4, resolution)

    def test_reject_remote_and_missing_and_unsupported_modes(self):
        for uri in ("https://example.test/a.png", "mm_file://abc", str(self.root / "missing.png")):
            with self.subTest(uri=uri), self.assertRaises(VideoApiError):
                build_local_request("Move", "fl2va", [{**self.ref(), "uri": uri}], 4)
        for mode, refs in (("fl2va", []), ("v2v", []), ("t2va", [self.ref()])):
            with self.assertRaises(VideoApiError):
                build_local_request("Move", mode, refs, 4)

    def test_health_checks_model_without_generation(self):
        health = self.client.check_health()
        self.assertEqual(set(health), {"fl2va", "ref2va"})
        self.assertEqual(self.http.posts, [])
        with patch.object(self.http, "request_json", return_value={"status": "ok", "architectures": ["WanPipeline"]}):
            with self.assertRaisesRegex(H3PollingInterrupted, "not a recognized H3"):
                self.client.check_health()

    def test_seed_reuse_routing_and_endpoint_id_collision(self):
        first = self.client.generate(self.payload, {"replicate_label": 42})
        again = self.client.generate(self.payload, {"replicate_label": 42})
        self.assertEqual(first[0], again[0])
        self.assertEqual(len(self.http.posts), 1)
        ref_request = build_local_request("Move", "ref2va", [self.ref("reference_image")], 4)
        second = self.client.generate(ref_request, {"replicate_label": 42})
        self.assertNotEqual(first[0], second[0])
        self.assertIn(":30010/v1/videos", self.http.posts[0][1])
        self.assertIn(":30011/v1/videos", self.http.posts[1][1])
        self.assertEqual(self.http.posts[0][2]["seed"], 42)
        self.assertEqual(self.http.posts[0][2]["quality"], "lossless")
        self.assertNotIn("Authorization", self.http.posts[0][3])
        self.assertTrue(Path(first[2]["local_video_path"]).is_file())

    def test_nas_without_flock_uses_atomic_directory_fallback(self):
        unsupported = OSError(errno.ENOSYS, "Function not implemented")
        with patch("evovideo_skill.h3_api.fcntl.flock", side_effect=unsupported):
            first = self.client.generate(self.payload, {"replicate_label": 42})
            replay = self.client.generate(self.payload, {"replicate_label": 42})
        self.assertEqual(first[0], replay[0])
        self.assertEqual(len(self.http.posts), 1)
        self.assertEqual(list(self.client.jobs.glob("*.lock.d")), [])
        self.assertFalse((self.client.jobs / ".lock.d").exists())

    def test_seed_and_revision_and_input_bytes_invalidate_cache(self):
        self.client.generate(self.payload, {"replicate_label": 42})
        self.client.generate(self.payload, {"replicate_label": 43})
        other = H3LocalClient(replace(self.config, model_revision="new-snapshot"), self.http)
        other.generate(self.payload, {"replicate_label": 42})
        request = build_local_request("Move", "fl2va", [self.ref()], 4)
        self.client.generate(request, {"replicate_label": 42})
        self.image.write_bytes(self.image.read_bytes() + b"metadata-change")
        self.client.generate(request, {"replicate_label": 42})
        self.assertEqual(len(self.http.posts), 5)
        with self.assertRaises(H3BudgetExceeded):
            self.client.generate(self.payload, {"replicate_label": 44})

    def test_health_prevents_reusing_experiment_after_deployment_change(self):
        self.client.check_health()
        self.client.generate(self.payload, {"replicate_label": 42})
        self.client.check_health()
        other = H3LocalClient(replace(self.config, model_revision="different-weights"), self.http)
        with self.assertRaisesRegex(H3PollingInterrupted, "deployment metadata changed"):
            other.check_health()

    def test_native_keyframes_are_canonical_and_smoke_uses_both_services(self):
        from evovideo_skill.h3_local_smoke import smoke_graph
        refs = [self.ref("last_frame"), {**self.ref(), "id": "start"}]
        request = build_local_request("Move", "fl2va", refs, 4)
        self.assertEqual([condition["frame_index"] for condition in request["conditions"]], [0, -1])
        self.assertEqual(smoke_graph().tool_names(), ["h3_t2va", "h3_frame_extract", "h3_fl2va", "h3_reference_pack", "h3_ref2va"])

    def test_copied_generation_directory_reuses_completed_video_without_post_or_download(self):
        self.client.check_health()
        identity = {"task_id": "camera", "node_id": "tool_t2v", "replicate_label": 42}
        first = self.client.generate(self.payload, identity)
        new_root = self.root / "new_run" / "videos"
        shutil.copytree(self.client.root, new_root)
        other = H3LocalClient(replace(self.config, output_dir=str(new_root)), self.http)
        other.check_health()
        with patch("evovideo_skill.h3_local._local_open", side_effect=AssertionError("must reuse local video")):
            replay = other.generate(self.payload, identity)
        self.assertEqual(len(self.http.posts), 1)
        self.assertEqual(Path(replay[2]["local_video_path"]).parent, new_root)
        self.assertEqual(Path(replay[2]["local_video_path"]).read_bytes(), Path(first[2]["local_video_path"]).read_bytes())

    def test_poll_resume_does_not_resubmit(self):
        original = self.http.request_json
        def interrupted(method, *args, **kwargs):
            if method == "GET":
                raise TimeoutError("poll connection lost")
            return original(method, *args, **kwargs)
        with patch.object(self.http, "request_json", side_effect=interrupted), self.assertRaises(H3PollingInterrupted):
            self.client.generate(self.payload, {})
        self.client.generate(self.payload, {})
        self.assertEqual(len(self.http.posts), 1)

    def test_recreated_reference_paths_cannot_bypass_uncertain_submission(self):
        request = build_local_request("Move", "fl2va", [self.ref()], 4)
        with patch.object(self.http, "request_json", side_effect=TimeoutError("post lost")), self.assertRaises(H3SubmissionUnknown):
            self.client.generate(request, {"replicate_label": 42})
        regenerated = self.root / "new-uuid-keyframe.png"
        regenerated.write_bytes(self.image.read_bytes())
        replay = build_local_request("Move", "fl2va", [{**self.ref(), "uri": str(regenerated)}], 4)
        with self.assertRaises(H3SubmissionUnknown):
            self.client.generate(replay, {"replicate_label": 42})
        self.assertEqual(len(list(self.client.jobs.glob("*.json"))), 1)
        self.assertEqual(self.http.posts, [])

    def test_download_resume_no_partial_output(self):
        with patch("evovideo_skill.h3_local._local_open", return_value=io.BytesIO(b"bad mp4")), self.assertRaises(H3PollingInterrupted):
            self.client.generate(self.payload, {})
        self.assertEqual(list(self.client.root.glob("*.part")), [])
        self.assertEqual(list(self.client.root.glob("*.mp4")), [])
        self.client.generate(self.payload, {})
        self.assertEqual(len(self.http.posts), 1)

    def test_uncertain_post_and_terminal_failure_not_retried(self):
        with patch.object(self.http, "request_json", side_effect=TimeoutError("post lost")), self.assertRaises(H3SubmissionUnknown):
            self.client.generate(self.payload, {})
        with self.assertRaises(H3SubmissionUnknown):
            self.client.generate(self.payload, {})
        self.http.status = "failed"
        for _ in range(2):
            with self.assertRaisesRegex(VideoApiError, "fail"):
                self.client.generate(self.payload, {"replicate_label": 43})
        self.assertEqual(len(self.http.posts), 1)

    def test_invalid_local_configuration_and_seed(self):
        for kwargs in ({"resolution": "2K"}, {"quality": "high"}, {"max_api_calls": 0},
                       {"ref2va_url": self.config.fl2va_url}, {"fl2va_url": "http://user:secret@localhost:30010"}):
            with self.assertRaises(VideoApiError):
                H3LocalClient(replace(self.config, **kwargs), self.http)
        for seed in (True, -1, "42", 2**32):
            with self.assertRaises(VideoApiError):
                self.client.generate(self.payload, {"replicate_label": seed})

    def test_real_frame_passing_and_ref_routing(self):
        registry = ToolRegistry()
        register_h3_tools(registry, self.client)
        task = VideoTask("local", "A blue object moves", duration_seconds=4, metadata={"generation_seed": 123})
        plan = VideoPlan(task.task_id, {}, ["move"], [], [], [], task.prompt)
        graph = ToolPathGraph("local-path", "local-path", "Test", ["generation"], [
            GraphNode("draft", "tool", "h3_t2va"),
            GraphNode("frame", "tool", "h3_frame_extract", {"position": "last", "role": "last_frame"}),
            GraphNode("repair", "tool", "h3_fl2va"),
            GraphNode("pack", "tool", "h3_reference_pack", {"bindings": [
                {"source": "repair", "kind": "video", "role": "reference_video"}]}),
            GraphNode("reference", "tool", "h3_ref2va"),
        ], [GraphEdge("a", "draft", "frame"), GraphEdge("b", "frame", "repair"),
            GraphEdge("c", "repair", "pack"), GraphEdge("d", "pack", "reference")])
        result = ArtifactPassingGraphExecutor(registry, EvaluatorSuite()).execute(task, plan, graph)
        self.assertEqual([post[2]["task"] for post in self.http.posts], ["t2va", "fl2va", "ref2va"])
        self.assertEqual([post[2]["seed"] for post in self.http.posts], [123] * 3)
        self.assertEqual(self.http.posts[1][2]["conditions"][0]["frame_index"], -1)
        self.assertTrue(result.artifact.metadata["provider_seed_control"])
        self.assertEqual(result.artifact.metadata["provider"], "local-h3")
        self.assertTrue(result.artifact.metadata["has_audio"])
        self.assertTrue(result.artifact.metadata["sampled_frame_paths"])
        self.assertEqual(result.artifact.metadata["generation_seed"], 123)
        replay = ArtifactPassingGraphExecutor(registry, EvaluatorSuite()).execute(task, plan, graph)
        self.assertEqual(len(self.http.posts), 3)
        self.assertEqual(replay.artifact.metadata["local_video_path"], result.artifact.metadata["local_video_path"])


if __name__ == "__main__":
    unittest.main()
