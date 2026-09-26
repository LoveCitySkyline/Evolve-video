from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

from evovideo_skill.benchmark_assets import BenchmarkAssetBootstrapper
from evovideo_skill.benchmarks import BenchmarkSuite
from evovideo_skill.models import VideoArtifact, VideoPlan, VideoTask
from evovideo_skill.tools import ToolRegistry, VideoTool


class MaterializedT2VTool(VideoTool):
    name = "test_text_to_video"

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.calls = 0
        self.prompts: list[str] = []

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        self.calls += 1
        self.prompts.append(plan.generation_prompt)
        path = self.output_dir / f"generated-{self.calls}.mp4"
        path.write_bytes(b"test-mp4-content")
        return VideoArtifact(
            artifact_id=f"artifact-{self.calls}",
            task_id=task.task_id,
            prompt=task.prompt,
            mode=task.mode,
            tool_chain=[self.name],
            frames=[],
            metadata={"local_video_path": str(path)},
        )


class BenchmarkAssetBootstrapTests(unittest.TestCase):
    def test_generates_missing_assets_updates_manifest_and_reuses_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = root / "mini.json"
            manifest = root / "mini_assets.json"
            benchmark.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "task_id": "audio-task",
                                "prompt": "Three objects fall and each impact matches its sound.",
                                "duration_seconds": 1,
                                "metadata": {
                                    "task_family": "audio_video_sync",
                                    "input_assets": {"audio": "audio-1", "required": True},
                                },
                            },
                            {
                                "task_id": "edit-task",
                                "prompt": (
                                    "Create a consistent edit of the source video in the cafe: "
                                    "replace only the black umbrella with a yellow umbrella and remove the thermos."
                                ),
                                "mode": "editing",
                                "duration_seconds": 1,
                                "metadata": {
                                    "task_family": "compositional_editing",
                                    "edit_operations": [
                                        {"operation": "replace", "target": "black umbrella", "value": "yellow umbrella"},
                                        {"operation": "remove", "target": "thermos"},
                                    ],
                                    "input_assets": {"source_video": "video-1", "required": True},
                                },
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            manifest.write_text(
                json.dumps(
                    {
                        "assets": [
                            {"asset_id": "audio-1", "type": "audio", "path": None},
                            {"asset_id": "video-1", "type": "source_video", "path": None},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            tool = MaterializedT2VTool(root)
            registry = ToolRegistry()
            registry.register(
                tool,
                SimpleNamespace(
                    name=tool.name,
                    capability="text_to_video",
                    backend="test-runtime",
                    verified=True,
                ),
            )

            suite = BenchmarkSuite.from_file(benchmark)
            report = BenchmarkAssetBootstrapper(registry, root / "assets", seed=7).run(suite.tasks)

            self.assertEqual(report.generated_count, 2)
            self.assertEqual(report.failed_count, 0)
            self.assertEqual(tool.calls, 1)
            audio_task, edit_task = suite.tasks
            audio_path = Path(audio_task.metadata["local_audio_path"])
            video_path = Path(edit_task.reference_video or "")
            self.assertTrue(audio_path.is_file())
            self.assertTrue(video_path.is_file())
            with wave.open(str(audio_path), "rb") as handle:
                self.assertEqual(handle.getframerate(), 48_000)
                self.assertEqual(handle.getnchannels(), 1)
            self.assertIn("black umbrella", tool.prompts[0])
            self.assertIn("thermos", tool.prompts[0])
            self.assertIn("do not replace or remove", tool.prompts[0].lower())
            persisted = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertTrue(all(item["path"] for item in persisted["assets"]))
            self.assertTrue(all(item.get("bootstrap", {}).get("sha256") for item in persisted["assets"]))

            reloaded = BenchmarkSuite.from_file(benchmark)
            reused = BenchmarkAssetBootstrapper(registry, root / "assets", seed=7).run(reloaded.tasks)
            self.assertEqual(reused.generated_count, 0)
            self.assertEqual(reused.reused_count, 2)
            self.assertEqual(tool.calls, 1)


if __name__ == "__main__":
    unittest.main()
