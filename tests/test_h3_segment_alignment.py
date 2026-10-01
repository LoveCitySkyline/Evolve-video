"""Reproduce 277-frame single clips and 452-frame two-shot source videos."""
import copy
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

from evovideo_skill.h3_mini50 import attach_assets, digest, segment_task, verify_task_assets
from evovideo_skill.h3_media import _probe
from evovideo_skill.models import VideoTask


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class SegmentAlignmentTests(unittest.TestCase):
    def clip(self, path, frames, color="red", frequency=440):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        f"color={color}:size=64x48:rate=24", "-f", "lavfi", "-i",
                        f"sine=frequency={frequency}:sample_rate=32000:duration={frames / 24}",
                        "-frames:v", str(frames), "-c:v", "libx264", "-c:a", "aac", str(path)], check=True)

    def fixture(self, root):
        task = segment_task({"task_id": "test", "prompt": "Two scenes", "duration_seconds": 18,
                             "metadata": {"input_assets": {"source_video": "source_video_006"}}})
        generation = root / "asset_generation"
        jobs = generation / "h3_local_jobs"
        jobs.mkdir(parents=True)
        clips = [generation / f"shot-{i}.mp4" for i in range(2)]
        for i, path in enumerate(clips):
            self.clip(path, 226, ["red", "blue"][i], [440, 880][i])
            record = {"request_hash": f"request-{i}", "status": "completed", "model_revision": "test-revision",
                      "identity": {"task_id": "input-source_video_006", "node_id": f"direct-direct-shot-{i}",
                                   "replicate_label": 42},
                      "request": {"seconds": 9}, "local_video_path": str(path)}
            (jobs / f"{i}.json").write_text(json.dumps(record))
        original = generation / "concat.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clips[0]), "-i", str(clips[1]),
                        "-filter_complex", "[0:v]setpts=PTS-STARTPTS,fps=24[v0];[1:v]setpts=PTS-STARTPTS,fps=24[v1];"
                        "[0:a]apad,atrim=duration=9.416666667,asetpts=PTS-STARTPTS[a0];"
                        "[1:a]apad,atrim=duration=9.416666667,asetpts=PTS-STARTPTS[a1];"
                        "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]",
                        "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-c:a", "aac", str(original)], check=True)
        entry = {"asset_id": "source_video_006", "path": str(original),
                 "provenance": {"synthetic": True, "generator": "local-h3", "seed": 42,
                                "model_revision": "test-revision"}}
        return task, entry, clips, jobs

    def test_277_frames_require_explicit_single_shot_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "original.mp4"
            self.clip(source, 277)
            self.assertAlmostEqual(_probe(source, "video")["duration_seconds"], 277 / 24, places=5)
            task = {"task_id": "single", "duration_seconds": 11, "metadata": {"input_assets": {"source_video": "edit_source_019"}}}
            entry = {"path": str(source), "provenance": {"synthetic": True, "generator": "local-h3"}}
            with self.assertRaisesRegex(ValueError, "actual=11.541667"):
                attach_assets(task, {"edit_source_019": entry}, root, align_media=True)
            result = attach_assets(task, {"edit_source_019": entry}, root, align_media=True, max_video_overrun=.6)
            self.assertAlmostEqual(_probe(Path(result["reference_video"]), "video")["duration_seconds"], 11, places=3)

    def test_452_frames_rebuilt_with_visual_and_audio_transition_at_nine_seconds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            task, entry, clips, jobs = self.fixture(root)
            originals = {str(p): digest(p) for p in [Path(entry["path"]), *clips]}
            self.assertAlmostEqual(_probe(Path(entry["path"]), "video")["duration_seconds"], 452 / 24, places=5)
            result = attach_assets(task, {entry["asset_id"]: entry}, root, align_media=True, max_video_overrun=.6)
            aligned = result["reference_video"]
            record = result["metadata"]["h3_video_alignment"][0]
            self.assertEqual(record["operation"], "trim_each_recorded_shot_then_concat")
            self.assertEqual([s["target_start_seconds"] for s in record["source_shots"]], [0, 9])
            self.assertAlmostEqual(record["source_shots"][1]["source_start_seconds"], 226 / 24, places=5)
            self.assertAlmostEqual(record["duration_seconds"], 18, places=3)
            for p, before in originals.items():
                self.assertEqual(digest(Path(p)), before)
            # At 9.125s a total-tail trim would still show red and play 440 Hz.
            pixel = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", "9.125", "-i", aligned,
                                             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            self.assertGreater(pixel[2], pixel[0] + 100)
            audio = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", "9.125", "-i", aligned,
                                             "-t", "0.1", "-map", "0:a:0", "-ac", "1", "-ar", "32000", "-f", "f32le", "-"])
            samples = struct.unpack("<" + "f" * (len(audio) // 4), audio)
            crossings = sum((a < 0) != (b < 0) for a, b in zip(samples, samples[1:]))
            self.assertGreater(crossings, 140)
            self.assertLess(crossings, 210)
            again = attach_assets(task, {entry["asset_id"]: entry}, root, align_media=True, max_video_overrun=.6)
            self.assertEqual(again, result)
            verify_task_assets(VideoTask.from_dict(result))
            clips[1].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "frozen input changed"):
                verify_task_assets(VideoTask.from_dict(result))

    def test_missing_ambiguous_and_mismatched_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            task, entry, clips, jobs = self.fixture(root)
            record_path = jobs / "1.json"
            saved = record_path.read_text()
            record_path.unlink()
            with self.assertRaisesRegex(ValueError, "found 0"):
                attach_assets(task, {entry["asset_id"]: entry}, root, align_media=True, max_video_overrun=.6)
            record_path.write_text(saved)
            duplicate = jobs / "duplicate.json"
            duplicate.write_text(saved)
            with self.assertRaisesRegex(ValueError, "found 2"):
                attach_assets(task, {entry["asset_id"]: entry}, root, align_media=True, max_video_overrun=.6)
            duplicate.unlink()
            for field, value in [("seconds", 8)]:
                invalid = json.loads(saved)
                invalid["request"][field] = value
                record_path.write_text(json.dumps(invalid))
                with self.assertRaisesRegex(ValueError, "found 0"):
                    attach_assets(task, {entry["asset_id"]: entry}, root, align_media=True, max_video_overrun=.6)

    def test_explicit_provenance_is_hashed_and_short_shots_are_not_extended(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            task, entry, clips, jobs = self.fixture(root)
            entry["provenance"]["source_shots"] = [
                {"shot_index": i, "path": str(path), "sha256": digest(path), "requested_duration_seconds": 9}
                for i, path in enumerate(clips)]
            result = attach_assets(task, {entry["asset_id"]: entry}, root, align_media=True, max_video_overrun=.6)
            self.assertEqual(result["metadata"]["h3_video_alignment"][0]["source_shots"][0]["evidence"], "source_shots")
            altered = copy.deepcopy(entry)
            altered["provenance"]["source_shots"][0]["sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "changed or missing"):
                attach_assets(task, {entry["asset_id"]: altered}, root, align_media=True)
            # Even a permissive upper limit must not extend an undersized native shot.
            shifted = copy.deepcopy(task)
            shifted["metadata"]["h3_shots"][0]["duration_seconds"] = 10
            shifted["metadata"]["h3_shots"][1]["duration_seconds"] = 8
            entry["provenance"]["source_shots"][0]["requested_duration_seconds"] = 10
            entry["provenance"]["source_shots"][1]["requested_duration_seconds"] = 8
            with self.assertRaisesRegex(ValueError, "short clips cannot be extended"):
                attach_assets(shifted, {entry["asset_id"]: entry}, root, align_media=True, max_video_overrun=2)
