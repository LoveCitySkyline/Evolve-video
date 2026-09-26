import sys
import tempfile
import textwrap
import unittest
import json
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evovideo_skill.api_tools import VideoApiError, WanLocalCliImageToVideoTool, WanLocalCliTextToVideoTool
from evovideo_skill.models import TaskMode, VideoArtifact, VideoPlan, VideoTask
from evovideo_skill.runtime import RuntimeSettings, _local_wan_registry
from evovideo_skill.tools import ToolExecutionContext


class LocalWanCliTest(unittest.TestCase):
    def test_t2v_runtime_does_not_advertise_fake_i2v_or_editing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo = root / "Wan"
            ckpt = root / "t2v"
            repo.mkdir()
            ckpt.mkdir()
            registry = _local_wan_registry(
                RuntimeSettings(
                    provider="local-wan",
                    wan_repo=str(repo),
                    wan_ckpt_dir=str(ckpt),
                    video_output_dir=str(root / "out"),
                )
            )

            self.assertIn("mock_text_to_video", registry.available_names())
            self.assertNotIn("mock_image_to_video", registry.available_names())
            self.assertNotIn("mock_region_video_editor", registry.available_names())
            self.assertNotIn("temporal_deflicker", registry.available_names())

    def test_wan_cli_tool_runs_local_generate_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            wan_repo = root / "Wan"
            ckpt_dir = root / "ckpt"
            output_dir = root / "out"
            wan_repo.mkdir()
            ckpt_dir.mkdir()
            generate_py = wan_repo / "generate.py"
            generate_py.write_text(
                textwrap.dedent(
                    """
                    import argparse
                    from pathlib import Path

                    parser = argparse.ArgumentParser()
                    parser.add_argument("--task")
                    parser.add_argument("--size")
                    parser.add_argument("--ckpt_dir")
                    parser.add_argument("--prompt")
                    parser.add_argument("--save_file")
                    parser.add_argument("--offload_model")
                    args, _ = parser.parse_known_args()
                    Path(args.save_file).write_bytes(b"fake mp4")
                    print("saved", args.save_file)
                    """
                ),
                encoding="utf-8",
            )
            tool = WanLocalCliTextToVideoTool.from_arg_string(
                wan_repo=wan_repo,
                ckpt_dir=ckpt_dir,
                output_dir=output_dir,
                tool_name="mock_text_to_video",
                python_bin=sys.executable,
                extra_args="--offload_model True",
            )
            task = VideoTask(
                "local-wan-smoke",
                "A robot waves.",
                metadata={"generation_seed": 42},
            )
            plan = VideoPlan(
                task_id=task.task_id,
                intent={"subject": "robot"},
                temporal_steps=["waves"],
                constraints=[],
                selected_skill_names=[],
                tool_chain=["mock_text_to_video"],
                generation_prompt="A robot waves.",
            )

            artifact = tool.run(task, plan)

            self.assertEqual(artifact.metadata["provider"], "wan-local-cli")
            self.assertIn("--save_file", artifact.metadata["command"])
            self.assertTrue(Path(artifact.metadata["local_video_path"]).exists())
            self.assertEqual(artifact.tool_chain, ["mock_text_to_video"])
            events = [
                json.loads(line)
                for line in (output_dir / "wan_invocations.jsonl").read_text().splitlines()
            ]
            self.assertEqual(events[0]["status"], "started")
            self.assertEqual(events[-1]["status"], "completed_without_frames")
            self.assertTrue(Path(events[0]["expected_output_path"]).is_absolute())
            self.assertEqual(events[0]["generation_seed"], 42)
            self.assertIn("--base_seed", events[0]["command"])
            seed_index = events[0]["command"].index("--base_seed")
            self.assertEqual(events[0]["command"][seed_index + 1], "42")

    def test_success_without_mp4_reports_output_discovery_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            wan_repo = root / "Wan"
            ckpt_dir = root / "ckpt"
            wan_repo.mkdir()
            ckpt_dir.mkdir()
            (wan_repo / "generate.py").write_text("print('finished without output')\n", encoding="utf-8")
            tool = WanLocalCliTextToVideoTool.from_arg_string(
                wan_repo=wan_repo,
                ckpt_dir=ckpt_dir,
                output_dir=root / "out",
                python_bin=sys.executable,
                extra_args="--offload_model True",
            )
            task = VideoTask("missing-output", "A robot waves.")
            plan = VideoPlan(task.task_id, {}, ["waves"], [], [], ["mock_text_to_video"], task.prompt)

            with self.assertRaisesRegex(VideoApiError, "no new MP4 was found"):
                tool.run(task, plan)

    def test_silent_wan_process_is_terminated_at_runtime_timeout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            wan_repo = root / "Wan"
            ckpt_dir = root / "ckpt"
            wan_repo.mkdir()
            ckpt_dir.mkdir()
            (wan_repo / "generate.py").write_text(
                "import time\ntime.sleep(30)\n",
                encoding="utf-8",
            )
            tool = WanLocalCliTextToVideoTool.from_arg_string(
                wan_repo=wan_repo,
                ckpt_dir=ckpt_dir,
                output_dir=root / "out",
                python_bin=sys.executable,
                extra_args="--offload_model True",
                timeout_seconds=1,
            )
            task = VideoTask("wan-timeout", "A robot waves.")
            plan = VideoPlan(task.task_id, {}, ["waves"], [], [], [tool.name], task.prompt)

            with self.assertRaisesRegex(VideoApiError, "timed out after 1s"):
                tool.run(task, plan)

    def test_completed_output_is_recovered_when_wan_does_not_exit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            wan_repo = root / "Wan"
            ckpt_dir = root / "ckpt"
            wan_repo.mkdir()
            ckpt_dir.mkdir()
            (wan_repo / "generate.py").write_text(
                textwrap.dedent(
                    """
                    import argparse
                    import time
                    from pathlib import Path
                    parser = argparse.ArgumentParser()
                    parser.add_argument("--save_file", required=True)
                    args, _ = parser.parse_known_args()
                    Path(args.save_file).write_bytes(b"completed mp4")
                    print("saved", args.save_file, flush=True)
                    time.sleep(30)
                    """
                ),
                encoding="utf-8",
            )
            tool = WanLocalCliTextToVideoTool.from_arg_string(
                wan_repo=wan_repo,
                ckpt_dir=ckpt_dir,
                output_dir=root / "out",
                python_bin=sys.executable,
                extra_args="--offload_model True",
                timeout_seconds=2,
            )
            task = VideoTask("wan-saved-before-exit", "A robot waves.")
            plan = VideoPlan(task.task_id, {}, ["waves"], [], [], [tool.name], task.prompt)

            with patch.dict(
                "os.environ",
                {
                    "WAN_VALIDATE_OUTPUT_FFPROBE": "0",
                    "WAN_OUTPUT_FINALIZE_GRACE_SECONDS": "0",
                },
            ):
                artifact = tool.run(task, plan)

            self.assertTrue(Path(artifact.metadata["local_video_path"]).exists())
            self.assertIn("recovered completed output", artifact.metadata["stdout_tail"])

    def test_i2v_adapter_passes_real_upstream_reference_to_wan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            wan_repo = root / "Wan"
            ckpt_dir = root / "i2v-ckpt"
            output_dir = root / "out"
            wan_repo.mkdir()
            ckpt_dir.mkdir()
            reference = root / "identity.png"
            reference.write_bytes(b"image")
            (wan_repo / "generate.py").write_text(
                textwrap.dedent(
                    """
                    import argparse
                    from pathlib import Path
                    parser = argparse.ArgumentParser()
                    parser.add_argument("--image", required=True)
                    parser.add_argument("--save_file", required=True)
                    args, _ = parser.parse_known_args()
                    assert Path(args.image).exists()
                    Path(args.save_file).write_bytes(b"fake mp4")
                    """
                ),
                encoding="utf-8",
            )
            tool = WanLocalCliImageToVideoTool.from_arg_string(
                wan_repo=wan_repo,
                ckpt_dir=ckpt_dir,
                output_dir=output_dir,
                tool_name="mock_image_to_video",
                task_name="i2v-test",
                python_bin=sys.executable,
                extra_args="--offload_model True",
            )
            task = VideoTask("i2v", "Keep the same person.")
            plan = VideoPlan(task.task_id, {}, ["walk"], [], [], [tool.name], task.prompt)
            upstream = VideoArtifact(
                "ref", task.task_id, task.prompt, TaskMode.GENERATION, ["extract"], [],
                {"artifact_type": "identity_reference", "reference_image": str(reference)},
            )
            artifact = tool.run_with_context(
                task,
                plan,
                ToolExecutionContext("i2v", {}, {"reference": upstream}),
            )

            self.assertTrue(artifact.metadata["upstream_conditioning_consumed"])
            self.assertIn("--image", artifact.metadata["command"])
            self.assertIn(str(reference), artifact.metadata["command"])


if __name__ == "__main__":
    unittest.main()
