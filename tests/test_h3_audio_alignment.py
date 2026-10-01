"""Real-media regression coverage for the 12.256s synthetic audio failure."""
import copy
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evovideo_skill.h3_mini50 import SOURCE, attach_assets, digest, prepare, verify_prepared
import test_h3_mini50


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class AudioAlignmentTests(unittest.TestCase):
    def video_fixture(self, root, duration=9.416667, audio=True):
        video = root / "original.mp4"
        command = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                   f"testsrc2=size=64x48:rate=24:duration={duration}"]
        if audio:
            command += ["-f", "lavfi", "-i", f"sine=sample_rate=32000:duration={duration}", "-c:a", "aac"]
        subprocess.run(command + ["-c:v", "libx264", "-t", str(duration), str(video)], check=True)
        task = {"task_id": "video-test", "duration_seconds": 9,
                "metadata": {"input_assets": {"source_video": "edit_source_018"}}}
        entry = {"asset_id": "edit_source_018", "path": str(video),
                 "provenance": {"synthetic": True, "generator": "local-h3"}}
        return task, entry

    def test_video_trim_preserves_original_and_optional_audio(self):
        from evovideo_skill.h3_mini50 import _probe
        for audio in (True, False):
            with self.subTest(audio=audio), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                task, entry = self.video_fixture(root, audio=audio)
                source_hash = digest(Path(entry["path"]))
                with self.assertRaisesRegex(ValueError, "--align-generated-media"):
                    attach_assets(task, {entry["asset_id"]: entry}, root, align_audio=True)
                result = attach_assets(task, {entry["asset_id"]: entry}, root, align_media=True)
                record = result["metadata"]["h3_video_alignment"][0]
                self.assertAlmostEqual(record["duration_seconds"], 9, places=3)
                self.assertEqual(record["has_audio"], audio)
                self.assertEqual(result["reference_video"], record["path"])
                self.assertEqual(digest(Path(entry["path"])), source_hash)
                self.assertEqual(len(result["metadata"]["h3_asset_lock"]), 2)
                if audio:
                    stream = _probe(Path(record["path"]), "video")["audio_streams"][0]
                    self.assertAlmostEqual(float(stream["duration"]), 9, delta=.05)

    def test_video_alignment_rejects_short_external_or_large_overrun(self):
        for duration, provenance, message in [(8.6, True, "only trims"), (9.75, True, "only trims"),
                                               (9.416667, False, "provenance")]:
            with self.subTest(duration=duration, provenance=provenance), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                task, entry = self.video_fixture(root, duration)
                if not provenance:
                    entry.pop("provenance")
                with self.assertRaisesRegex(ValueError, message):
                    attach_assets(task, {entry["asset_id"]: entry}, root, align_media=True)
                self.assertFalse((root / "aligned_video").exists())

    def fixture(self, root, duration):
        audio = root / "original.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        f"sine=frequency=440:sample_rate=32000:duration={duration}", str(audio)], check=True)
        image = root / "anchor.ppm"
        image.write_bytes(b"P6\n2 2\n255\n" + bytes([255, 0, 0]) * 4)
        task = {"task_id": "audio-test", "duration_seconds": 12,
                "metadata": {"input_assets": {"audio": "audio_track_006"}}}
        entry = {"asset_id": "audio_track_006", "path": str(audio), "image_path": str(image),
                 "provenance": {"synthetic": True, "generator": "local-h3"}}
        return task, entry

    def test_explicit_alignment_preserves_source_locks_and_review_evidence(self):
        for seconds, operation in [(12.256, "trim_tail"), (11.7, "pad_tail_silence")]:
            with self.subTest(seconds=seconds), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                task, entry = self.fixture(root, seconds)
                entries = {entry["asset_id"]: entry}
                before = copy.deepcopy(entries)
                source_hash = digest(Path(entry["path"]))
                with self.assertRaisesRegex(ValueError, "--align-generated-audio"):
                    attach_assets(task, entries, root)
                result = attach_assets(task, entries, root, align_audio=True)
                record = result["metadata"]["h3_audio_alignment"][0]
                self.assertEqual(record["operation"], operation)
                self.assertAlmostEqual(record["duration_seconds"], 12, places=3)
                self.assertEqual(digest(Path(entry["path"])), source_hash)
                self.assertEqual(entries, before)
                self.assertEqual(len(result["metadata"]["h3_asset_lock"]), 3)
                self.assertEqual(result["metadata"]["local_audio_path"], record["path"])
                again = attach_assets(task, entries, root, align_audio=True)
                self.assertEqual(again, result)

    def test_rejects_external_sources_and_large_duration_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            task, entry = self.fixture(root, 12.256)
            entry["provenance"] = {}
            with self.assertRaisesRegex(ValueError, "provenance"):
                attach_assets(task, {entry["asset_id"]: entry}, root, align_audio=True)
            task, entry = self.fixture(root, 12.6)
            with self.assertRaisesRegex(ValueError, "0.5s alignment limit"):
                attach_assets(task, {entry["asset_id"]: entry}, root, align_audio=True)
            self.assertFalse((root / "aligned_audio").exists())

    def test_prepare_reuses_saved_audio_without_h3_and_freezes_derived_asset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture = test_h3_mini50.H3Mini50Tests()
            fixture.setUp()
            manifest = fixture.fixture_manifest(root)
            suite = json.loads(SOURCE.read_text())
            audio_task = next(t for t in suite["tasks"] if t["metadata"].get("input_assets", {}).get("audio") == "audio_track_006")
            self.assertEqual(audio_task["duration_seconds"], 12)
            _, entry = self.fixture(root, 12.256)
            _, video_entry = self.video_fixture(root)
            data = json.loads(manifest.read_text())
            data["assets"] = [e for e in data["assets"] if e["asset_id"] not in {entry["asset_id"], video_entry["asset_id"]}]
            manifest.write_text(json.dumps(data))
            output = root / "prepared"
            output.mkdir()
            (output / "generated_assets.json").write_text(json.dumps({"assets": [entry, video_entry]}))
            from evovideo_skill.h3_mini50 import _probe, trim_video
            def probe(path, kind):
                fake = path.suffix in {".fixture", ".png"} or path.parent.name == "reference_segments"
                return fixture.fake_probe(path, kind) if fake else _probe(path, kind)
            def trim(source, target, start, duration):
                method = fixture.fake_trim if target.parent.name == "reference_segments" else trim_video
                return method(source, target, start, duration)
            with patch("evovideo_skill.h3_mini50._probe", side_effect=probe), patch(
                "evovideo_skill.h3_mini50.trim_video", side_effect=trim), patch(
                "evovideo_skill.h3_mini50.generate_asset") as generate:
                with self.assertRaisesRegex(ValueError, "Asset validation failed"):
                    prepare(SOURCE, manifest, output)
                failed = json.loads((output / "prepare_report.json").read_text())
                self.assertEqual(failed["status"], "invalid_assets")
                self.assertEqual(len(failed["invalid_assets"]), 2)
                report = prepare(SOURCE, manifest, output, align_media=True)
                generate.assert_not_called()
                self.assertEqual(report["status"], "awaiting_asset_review")
                self.assertEqual(len(report["audio_alignments"]), 1)
                self.assertEqual(len(report["video_alignments"]), 1)
                self.assertFalse((output / "prepared.lock.json").exists())
                prepare(SOURCE, manifest, output, approve_assets=True, align_media=True)
                verify_prepared(output)
                Path(entry["path"]).write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "frozen input changed"):
                    verify_prepared(output)
