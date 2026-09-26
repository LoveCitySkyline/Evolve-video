import copy
import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from evovideo_skill.benchmarks import BenchmarkSuite
from evovideo_skill.h3_api import task_references
from evovideo_skill.h3_cli import preflight
from evovideo_skill.h3_mini50 import (SOURCE, attach_assets, digest, pin_experiment, prepare,
                                    segment_task, verify_prepared, write_json)
from evovideo_skill.h3_omni_verifier import comparison_video, evaluate, request_scores
from evovideo_skill.harness import HarnessConfig
from evovideo_skill.models import VideoTask


class H3Mini50Tests(unittest.TestCase):
    def setUp(self):
        self.suite = json.loads(SOURCE.read_text())

    def test_segmentation_preserves_original_tasks_and_rubric(self):
        before = copy.deepcopy(self.suite)
        adapted = [segment_task(task) for task in self.suite["tasks"]]
        self.assertEqual(self.suite, before)
        self.assertEqual(sum("h3_shots" in task["metadata"] for task in adapted), 8)
        for old, task in zip(self.suite["tasks"], adapted):
            for key in ("task_id", "prompt", "duration_seconds", "mode"):
                self.assertEqual(task[key], old[key])
            for key in ("split", "evaluation", "reward", "temporal_steps"):
                self.assertEqual(task["metadata"].get(key), old["metadata"].get(key))
            if task["duration_seconds"] > 15:
                shots = task["metadata"]["h3_shots"]
                self.assertEqual(sum(s["duration_seconds"] for s in shots), task["duration_seconds"])
                self.assertTrue(all(4 <= s["duration_seconds"] <= 15 for s in shots))
                self.assertEqual(shots[0]["source_end_seconds"], shots[1]["source_start_seconds"])
                for step in task["metadata"].get("temporal_steps", []):
                    self.assertEqual(sum(step in shot["prompt"] for shot in shots), 1)

    def test_missing_inputs_report_without_generation_or_source_edits(self):
        before = SOURCE.read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, "17 inputs missing"):
                prepare(SOURCE, SOURCE.with_name(SOURCE.stem + "_assets.json"), root)
            report = json.loads((root / "prepare_report.json").read_text())
            self.assertEqual(report["status"], "missing_assets")
            self.assertFalse((root / "prepared.lock.json").exists())
        self.assertEqual(SOURCE.read_bytes(), before)

    def fixture_manifest(self, root):
        entries = []
        for task in self.suite["tasks"]:
            for key in ("audio", "source_video"):
                ident = task["metadata"].get("input_assets", {}).get(key)
                if ident:
                    path = root / (ident + ".fixture")
                    path.write_text(str(task["duration_seconds"]))
                    entry = {"asset_id": ident, "type": key, "path": str(path)}
                    if key == "audio":
                        anchor = root / f"{ident}.png"
                        anchor.write_text("1")
                        entry["image_path"] = str(anchor)
                    entries.append(entry)
        path = root / "assets.json"
        write_json(path, {"assets": entries})
        return path

    @staticmethod
    def fake_probe(path, kind):
        return {"duration_seconds": float(path.read_text()), "has_audio": kind == "audio"}

    @staticmethod
    def fake_trim(source, target, start, duration):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(duration))

    @staticmethod
    def fake_native_probe(path):
        path = Path(path)
        kind = "audio" if path.name.startswith("audio") and path.suffix != ".png" else "video"
        return {"format": {"duration": float(path.read_text())}, "streams": [{"codec_type": kind, "width": 100}]}

    def test_prepare_freeze_preflight_resume_and_asset_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = self.fixture_manifest(root)
            output = root / "prepared"
            with patch("evovideo_skill.h3_mini50._probe", side_effect=self.fake_probe), patch(
                "evovideo_skill.h3_mini50.trim_video", side_effect=self.fake_trim), redirect_stdout(io.StringIO()):
                report = prepare(SOURCE, manifest, output)
                self.assertEqual(report["status"], "awaiting_asset_review")
                with self.assertRaisesRegex(ValueError, "not frozen"):
                    verify_prepared(output)
                report = prepare(SOURCE, manifest, output, approve_assets=True)
                self.assertEqual(report["split_counts"], {"train": 28, "validation": 11, "test": 11})
                self.assertEqual(report["audio_tasks"], 3)
                self.assertEqual(report["reference_tasks"], 17)
                task_file = verify_prepared(output)
                config = HarnessConfig.from_file("configs/h3_local_mini50_harness.json")
                config.task_files = [str(task_file)]
                config.runtime.h3_audio_verifier_command = '["audio-judge"]'
                with patch.dict(os.environ, {"PATH": os.environ.get("PATH", "")}, clear=True), patch(
                    "evovideo_skill.h3_api.probe_media", side_effect=self.fake_native_probe), patch(
                    "evovideo_skill.h3_local.probe_media", side_effect=self.fake_native_probe):
                    summary = preflight(config)
                self.assertEqual(summary["suites"][0]["tasks"], 50)
                self.assertEqual(summary["replicates"], 3)
                task_hash = digest(task_file)
                prepare(SOURCE, manifest, output)
                self.assertEqual(digest(task_file), task_hash)
                tasks = BenchmarkSuite.from_file(task_file).tasks
                audio = next(t for t in tasks if t.metadata.get("h3_audio_criteria"))
                self.assertEqual({r["kind"] for r in task_references(audio)}, {"image", "audio"})
                long_source = next(t for t in tasks if t.metadata.get("h3_segmented_source"))
                self.assertEqual(len(task_references(long_source, 0)), 1)
                self.assertNotEqual(task_references(long_source, 0), task_references(long_source, 1))
                self.assertTrue(all(ref["uri"] != long_source.reference_video for ref in task_references(long_source)))
                with self.assertRaisesRegex(Exception, "shot_index"):
                    task_references(long_source, True)
                Path(audio.metadata["h3_asset_lock"][0]["path"]).write_text("tampered")
                with self.assertRaisesRegex(ValueError, "frozen input changed"):
                    verify_prepared(output)

    def test_source_duration_mismatch_not_silently_trimmed(self):
        raw = next(t for t in self.suite["tasks"] if t["metadata"].get("input_assets", {}).get("audio"))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            media = root / "wrong.wav"
            media.write_text("2")
            ident = raw["metadata"]["input_assets"]["audio"]
            with patch("evovideo_skill.h3_mini50._probe", side_effect=self.fake_probe), self.assertRaisesRegex(ValueError, "must match task"):
                attach_assets(segment_task(raw), {ident: {"path": str(media)}}, root)

    def test_generated_inputs_checkpoint_and_reuse_without_more_model_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def generate(raw, ident, kind, output, client):
                path = root / f"{ident}.fixture"
                path.write_text(str(raw["duration_seconds"]))
                result = {"asset_id": ident, "type": kind, "path": str(path)}
                if kind == "audio":
                    image = root / f"{ident}.png"
                    image.write_text("1")
                    result["image_path"] = str(image)
                return result
            manifest = SOURCE.with_name(SOURCE.stem + "_assets.json")
            with patch("evovideo_skill.h3_mini50.generate_asset", side_effect=generate) as call, patch(
                "evovideo_skill.runtime.build_h3_local_client") as build, patch(
                "evovideo_skill.h3_mini50._probe", side_effect=self.fake_probe), patch(
                "evovideo_skill.h3_mini50.trim_video", side_effect=self.fake_trim), redirect_stdout(io.StringIO()):
                prepare(SOURCE, manifest, root, generate_missing=True)
                self.assertEqual(call.call_count, 17)
                build.return_value.check_health.assert_called_once()
                prepare(SOURCE, manifest, root, generate_missing=True, approve_assets=True)
                self.assertEqual(call.call_count, 17)
                self.assertTrue(verify_prepared(root).is_file())

    def test_malformed_shot_reference_ids_rejected(self):
        task = VideoTask("x", "test", metadata={"h3_references": [], "h3_shots": [{"reference_ids": ["missing"]}]})
        with self.assertRaisesRegex(Exception, "reference_ids"):
            task_references(task, 0)

    def test_cli_creates_missing_manifest_only_in_prepared_directory(self):
        from evovideo_skill.h3_mini50 import main
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "mini50.json"
            source.write_bytes(SOURCE.read_bytes())
            output = root / "prepared"
            with patch("sys.argv", ["h3_mini50", "prepare", "--source", str(source), "--prepared-dir", str(output)]), patch(
                "evovideo_skill.h3_mini50.prepare") as call:
                main()
            self.assertFalse((root / "mini50_assets.json").exists())
            self.assertEqual(len(json.loads((output / "assets_template.json").read_text())["assets"]), 17)
            self.assertEqual(call.call_args.args[1], output.resolve() / "assets_template.json")

    def test_all_50_preflight_with_real_media_and_segment_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            entries = []
            anchor = root / "anchor.png"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=blue:s=64x48", "-frames:v", "1", str(anchor)], check=True)
            for task in self.suite["tasks"]:
                for key in ("source_video", "audio"):
                    ident = task["metadata"].get("input_assets", {}).get(key)
                    if not ident:
                        continue
                    duration = task["duration_seconds"]
                    media = root / f"{key}-{duration}.{'wav' if key == 'audio' else 'mp4'}"
                    if not media.exists():
                        signal = f"sine=duration={duration}" if key == "audio" else f"testsrc2=size=64x48:rate=4:duration={duration}"
                        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", signal, str(media)], check=True)
                    entries.append({"asset_id": ident, "path": str(media), **({"image_path": str(anchor)} if key == "audio" else {})})
            manifest = root / "manifest.json"
            write_json(manifest, {"assets": entries})
            with redirect_stdout(io.StringIO()):
                report = prepare(SOURCE, manifest, root / "prepared", approve_assets=True)
            config = HarnessConfig.from_file("configs/h3_local_mini50_harness.json")
            config.task_files = [report["task_file"]]
            config.runtime.h3_audio_verifier_command = '["offline-contract-test"]'
            with patch.dict(os.environ, {"PATH": os.environ.get("PATH", "")}, clear=True):
                summary = preflight(config)
            self.assertEqual(summary["suites"][0]["tasks"], 50)
            self.assertEqual(summary["suites"][0]["reference_tasks"], 17)
            self.assertEqual(summary["api_calls_during_preflight"], 0)

    def test_protocol_pin_allows_only_call_budget_increase(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            root = Path(folder)
            tasks = root / "tasks.json"
            tasks.write_text("{}")
            output = root / "experiment"
            pin_experiment(tasks, output, False)
            with patch.dict(os.environ, {"H3_MAX_API_CALLS": "1000"}):
                pin_experiment(tasks, output, True)
            with patch.dict(os.environ, {"H3_OMNI_MODEL": "different-judge"}), self.assertRaisesRegex(ValueError, "protocol changed"):
                pin_experiment(tasks, output, True)
            with self.assertRaisesRegex(ValueError, "already exists"):
                pin_experiment(tasks, output, False)

    def test_no_audio_frame_only_evaluation_allowed(self):
        config = HarnessConfig.from_file("configs/h3_local_mini50_harness.json")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tasks.json"
            subset = copy.deepcopy(json.loads(Path("benchmarks/h3_native_pilot.json").read_text()))
            subset["tasks"][0]["metadata"]["task_family"] = "audio_video_sync"
            write_json(path, subset)
            config.task_files = [str(path)]
            with patch.dict(os.environ, {"PATH": os.environ.get("PATH", "")}, clear=True), self.assertRaisesRegex(ValueError, "h3_audio_criteria"):
                preflight(config)


class OmniVerifierTests(unittest.TestCase):
    def test_streamed_video_request_contains_audio_bearing_file(self):
        class Response(io.BytesIO):
            pass
        content = json.dumps({"criterion_scores": {"sync": .4}, "criterion_evidence": {"sync": "Impact at 1s precedes visual contact."}})
        stream = ('data: ' + json.dumps({"choices": [{"delta": {"content": content}}]}) + '\n\ndata: [DONE]\n').encode()
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fixture-key"}, clear=True):
            root = Path(folder)
            video = root / "av.mp4"
            video.write_bytes(b"mock-audio-video")
            with patch("urllib.request.urlopen", return_value=Response(stream)) as call:
                result, model = request_scores(video, "Evaluate fixed criteria", root)
                payload = json.loads(call.call_args.args[0].data)
                self.assertTrue(payload["stream"])
                self.assertEqual(payload["modalities"], ["text"])
                self.assertIn("base64,", payload["messages"][0]["content"][1]["video_url"]["url"])
                self.assertEqual(result["criterion_scores"]["sync"], .4)
                self.assertEqual(model, "qwen3-omni-flash")
            with patch("urllib.request.urlopen", return_value=Response(stream.replace(b'data: [DONE]', b''))), self.assertRaisesRegex(ValueError, "complete"):
                request_scores(video, "test", root)

    def test_real_av_comparison_preserves_both_intervals(self):
        from evovideo_skill.h3_media import _probe
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio, video = root / "ref.wav", root / "candidate.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", str(audio)], check=True)
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=red:s=96x64:r=12:d=2", "-f", "lavfi", "-i",
                            "sine=frequency=880:duration=2", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(video)], check=True)
            comparison = root / "comparison.mp4"
            rd, vd = comparison_video(audio, video, comparison)
            info = _probe(comparison, "video")
            self.assertTrue(info["has_audio"])
            self.assertAlmostEqual(info["duration_seconds"], rd + vd, delta=.15)
            request = {"prompt": "Match the tone", "candidate_video": str(video),
                       "references": [{"kind": "audio", "uri": str(audio)}], "criteria": {"sync": {}}}
            with patch("evovideo_skill.h3_omni_verifier.request_scores", return_value=({"criterion_scores": {"sync": .2},
                       "criterion_evidence": {"sync": "The candidate tone differs from the fixed reference at 0s."}}, "fixture-model")):
                result = evaluate(request, root)
            self.assertEqual(result["criterion_scores"]["sync"], .2)
            self.assertFalse(result["exact_sync_verified"])
            with patch("evovideo_skill.h3_omni_verifier.request_scores", return_value=({"criterion_scores": {"sync": True},
                       "criterion_evidence": {"sync": "text"}}, "fixture-model")), self.assertRaisesRegex(ValueError, "Invalid"):
                evaluate(request, root)


if __name__ == "__main__":
    unittest.main()
