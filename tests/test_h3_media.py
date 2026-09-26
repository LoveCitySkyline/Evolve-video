from __future__ import annotations

import array
import copy
import json
import math
import shutil
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from evovideo_skill.h3_media import (
    H3AudioExtractTool,
    H3AVConcatTool,
    H3FrameExtractTool,
    H3ReferenceBankTool,
    H3ReferencePackTool,
    H3ReferenceSelectTool,
    H3ReferenceTrimTool,
    register_h3_media_tools,
)
from evovideo_skill.models import VideoArtifact, VideoPlan, VideoTask
from evovideo_skill.tools import ToolExecutionContext, ToolRegistry


def run_media(argv: list[str]) -> bytes:
    return subprocess.run(argv, check=True, capture_output=True, timeout=120).stdout


def probe(path: str | Path) -> dict:
    return json.loads(run_media(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))


class H3RegistrationTests(unittest.TestCase):
    def test_specs_are_typed_verified_builtins(self):
        registry = ToolRegistry()
        with tempfile.TemporaryDirectory() as tmp:
            register_h3_media_tools(registry, tmp)
            expected = {
                "h3_reference_bank": ((), "h3_reference_set"),
                "h3_reference_select": (("h3_reference_set",), "h3_reference_set"),
                "h3_reference_pack": (("image", "video", "audio", "h3_reference_set"), "h3_reference_set"),
                "h3_frame_extract": (("video",), "image"),
                "h3_audio_extract": (("video",), "audio"),
                "h3_reference_trim": (("h3_reference_set",), "h3_reference_set"),
                "h3_av_concat": (("video",), "video"),
            }
            self.assertEqual(registry.available_names(), set(expected))
            for name, types in expected.items():
                spec = registry.spec(name)
                self.assertEqual((spec.input_types, spec.output_type), types)
                self.assertTrue(spec.verified)
                self.assertEqual(spec.backend, "builtin")
                self.assertEqual(spec.provenance, "builtin")
                self.assertIn("Config:", spec.description)
                self.assertTrue(spec.output_contract["materialized"])
                self.assertEqual(registry.get(name).output_dir, Path(tmp).resolve())
            for producer, consumer in [("h3_reference_bank", "h3_reference_select"),
                                       ("h3_frame_extract", "h3_reference_pack"),
                                       ("h3_av_concat", "h3_frame_extract")]:
                self.assertTrue(registry.validate_connection(producer, consumer)[0])
            self.assertIn("semantic_roles", registry.spec("h3_reference_select").description)
            self.assertIn("24fps, 32kHz stereo", registry.spec("h3_av_concat").description)
            self.assertIn("WAV pcm_s16le", registry.spec("h3_reference_trim").description)

    def test_default_constructors(self):
        for cls in (H3ReferenceBankTool, H3ReferenceSelectTool, H3ReferencePackTool,
                    H3FrameExtractTool, H3AudioExtractTool, H3ReferenceTrimTool, H3AVConcatTool):
            self.assertIsInstance(cls().output_dir, Path)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe required")
class H3MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="h3 media ' fixtures ")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.red = cls.root / "red with sound.mp4"
        cls.blue = cls.root / "blue silent.mp4"
        cls.multi = cls.root / "two soundtracks.mp4"
        cls.audio = cls.root / "tone.wav"
        cls.image = cls.root / "still.png"
        common = ["ffmpeg", "-v", "error", "-nostdin", "-y"]
        run_media(common + ["-f", "lavfi", "-i", "color=c=red:s=64x48:r=12:d=1",
                           "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=1",
                           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(cls.red)])
        run_media(common + ["-f", "lavfi", "-i", "color=c=blue:s=80x60:r=20:d=1",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(cls.blue)])
        run_media(common + ["-i", str(cls.red), "-f", "lavfi", "-i", "sine=frequency=880:duration=1",
                           "-map", "0:v", "-map", "0:a", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", str(cls.multi)])
        run_media(common + ["-f", "lavfi", "-i", "sine=frequency=660:duration=1", str(cls.audio)])
        run_media(common + ["-i", str(cls.red), "-frames:v", "1", str(cls.image)])

    def setUp(self):
        self.out_temp = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(self.out_temp.cleanup)
        self.output = Path(self.out_temp.name)
        self.task = VideoTask("h3-test", "red then blue", metadata={"h3_references": [
            {"id": "portrait", "kind": "image", "uri": self.image.as_uri(), "role": "reference_image", "semantic_role": "identity"},
            {"id": "clip", "kind": "video", "uri": str(self.red), "role": "reference_video", "duration_seconds": 99},
            {"id": "music", "kind": "audio", "uri": str(self.audio), "role": "reference_audio"},
        ]})
        self.plan = VideoPlan(self.task.task_id, {}, ["red", "blue"], [], [], [], self.task.prompt)

    def execute(self, cls, config=None, parents=None):
        return cls(self.output).run_with_context(self.task, self.plan, ToolExecutionContext("node", config or {}, parents or {}))

    def artifact(self, path, kind="video", ident="parent"):
        return VideoArtifact(ident, self.task.task_id, self.task.prompt, self.task.mode, ["fixture"], [],
                             {"artifact_type": kind, f"local_{kind}_path": str(path)})

    def bank(self):
        return self.execute(H3ReferenceBankTool)

    def test_reference_pack_preserves_current_task_repair_evidence(self):
        image = self.artifact(self.image, kind="image")
        image.metadata["repair_instruction"] = "Restore the current ending action"
        image.metadata["failure_localization"] = {"segments": [{"start_ratio": .3, "end_ratio": .6}]}
        config = {"bindings": [{"source": "frame", "kind": "image", "role": "first_frame"}]}
        result = self.execute(H3ReferencePackTool, config, {"frame": image})
        self.assertEqual(result.metadata["repair_instruction"], image.metadata["repair_instruction"])
        self.assertEqual(result.metadata["failure_localization"], image.metadata["failure_localization"])
        image.task_id = "other-task"
        result = self.execute(H3ReferencePackTool, config, {"frame": image})
        self.assertNotIn("repair_instruction", result.metadata)
        self.assertNotIn("failure_localization", result.metadata)

    def test_reference_pack_rejects_symbolic_storyboard(self):
        image = self.artifact(self.image, kind="image")
        image.metadata["materialization_mode"] = "semantic_storyboard"
        config = {"bindings": [{"source": "frame", "kind": "image", "role": "first_frame"}]}
        with self.assertRaisesRegex(ValueError, "Symbolic storyboard"):
            self.execute(H3ReferencePackTool, config, {"frame": image})

    def test_audio_extract_retains_observed_sound_and_duration(self):
        result = self.execute(H3AudioExtractTool, {"start_seconds": .1, "end_seconds": .8}, {"clip": self.artifact(self.red)})
        path = result.metadata["local_audio_path"]
        self.assertAlmostEqual(result.metadata["duration_seconds"], .7, delta=.03)
        stream = probe(path)["streams"][0]
        self.assertEqual(stream["codec_type"], "audio")
        self.assertEqual(stream["sample_rate"], "32000")
        self.assertEqual(stream["channels"], 2)
        self.assert_tone(path, .1, .3, 440)
        self.assertEqual(result.metadata["source_nodes"], ["clip"])

    def test_audio_extract_rejects_silence_and_invalid_bounds(self):
        with self.assertRaisesRegex(ValueError, "silent"):
            self.execute(H3AudioExtractTool, parents={"clip": self.artifact(self.blue)})
        with self.assertRaisesRegex(ValueError, "bounds"):
            self.execute(H3AudioExtractTool, {"start_seconds": .9, "end_seconds": .2}, {"clip": self.artifact(self.red)})

    def require_processor(self):
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("VideoProcessor requires opencv-python and numpy")

    def rms(self, path, start, duration, track=0):
        pcm = run_media(["ffmpeg", "-v", "error", "-i", str(path), "-ss", str(start), "-t", str(duration),
                         "-map", f"0:a:{track}", "-f", "f32le", "-ac", "1", "-ar", "8000", "pipe:1"])
        samples = array.array("f")
        samples.frombytes(pcm)
        self.assertTrue(samples)
        return math.sqrt(sum(value * value for value in samples) / len(samples))

    def assert_tone(self, path, start, duration, frequency, track=0):
        pcm = run_media(["ffmpeg", "-v", "error", "-i", str(path), "-ss", str(start), "-t", str(duration),
                         "-map", f"0:a:{track}", "-f", "f32le", "-ac", "1", "-ar", "8000", "pipe:1"])
        samples = array.array("f")
        samples.frombytes(pcm)
        self.assertTrue(samples)
        crossings = sum(left <= 0 < right for left, right in zip(samples, samples[1:]))
        self.assertAlmostEqual(crossings * 8000 / len(samples), frequency, delta=10)

    def rgb(self, path, timestamp=0):
        return tuple(run_media(["ffmpeg", "-v", "error", "-i", str(path), "-ss", str(timestamp),
                                "-frames:v", "1", "-vf", "scale=1:1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"])[:3])

    def test_bank_validates_materialized_files_and_probes_duration(self):
        original = copy.deepcopy(self.task.metadata)
        self.task.reference_video = str(self.blue)
        bank = self.bank()
        self.assertEqual(bank.metadata["artifact_type"], "h3_reference_set")
        refs = bank.metadata["h3_references"]
        self.assertEqual([ref["id"] for ref in refs], ["portrait", "clip", "music", "source-video"])
        self.assertTrue(all(Path(ref["uri"]).is_file() for ref in refs))
        self.assertAlmostEqual(refs[1]["duration_seconds"], 1, delta=0.1)
        self.assertTrue(bank.metadata["reference_media_info"]["clip"]["has_audio"])
        self.assertTrue(bank.metadata["reference_media_info"]["music"]["audio_streams"])
        self.assertEqual(self.task.metadata, original)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_bank_run_supports_source_video_only(self):
        self.task.metadata = {}
        self.task.reference_video = str(self.red)
        result = H3ReferenceBankTool(self.output).run(self.task, self.plan)
        self.assertEqual(result.metadata["h3_references"][0]["id"], "source-video")

    def test_bank_rejects_colliding_source_id_and_unused_parents(self):
        with self.assertRaises(ValueError):
            self.execute(H3ReferenceBankTool, parents={"unused": self.artifact(self.red)})
        self.task.reference_video = str(self.red)
        self.task.metadata["h3_references"].append({
            "id": "source-video", "kind": "video", "role": "reference_video", "uri": str(self.blue),
        })
        with self.assertRaisesRegex(ValueError, "collides"):
            self.bank()

    def test_bank_deduplicates_existing_source_uri_preserving_id_and_meaning(self):
        self.task.reference_video = str(self.red)
        original = copy.deepcopy(self.task.metadata)
        for ident, uri in [("clip", str(self.red)), ("source-video", str(self.red)),
                           ("source-video", self.red.as_uri()), ("clip", str(self.red.resolve()))]:
            with self.subTest(ident=ident, uri=uri):
                self.task.metadata = copy.deepcopy(original)
                self.task.metadata["h3_references"][1].update(id=ident, uri=uri, semantic_role="source motion")
                before = copy.deepcopy(self.task.metadata)
                refs = self.bank().metadata["h3_references"]
                self.assertEqual([ref["id"] for ref in refs], ["portrait", ident, "music"])
                self.assertEqual(refs[1]["semantic_role"], "source motion")
                self.assertEqual(self.task.metadata, before)

    def test_bank_accepts_null_optional_durations_and_probes_actual_values(self):
        self.task.reference_video = str(self.red)
        self.task.metadata["h3_references"][1]["id"] = "source-video"
        for ref in self.task.metadata["h3_references"]:
            ref["duration_seconds"] = None
        refs = self.bank().metadata["h3_references"]
        self.assertNotIn("duration_seconds", refs[0])
        for ref in refs[1:]:
            self.assertAlmostEqual(ref["duration_seconds"], 1, delta=0.1)
        self.assertTrue(all(ref["duration_seconds"] is None for ref in self.task.metadata["h3_references"]))

    def test_bank_rejects_missing_fake_mistyped_and_duplicate_references(self):
        good = self.task.metadata["h3_references"][0]
        corrupt = self.output / "fake.png"
        corrupt.write_text("not an image")
        cases = [[], {}, [good, good], [{**good, "uri": str(self.output / "missing.png")}],
                 [{**good, "uri": str(corrupt)}], [{**good, "uri": str(self.red)}],
                 [{**good, "kind": "video"}], [{**good, "role": "reference_audio"}],
                 [{**good, "duration_seconds": float("nan")}], [{**good, "id": []}],
                 [{**good, "kind": []}], [{**good, "role": []}], [{**good, "semantic_role": ""}]]
        for values in cases:
            with self.subTest(values=values):
                self.task.metadata["h3_references"] = values
                with self.assertRaises((ValueError, RuntimeError)):
                    self.bank()

    def test_select_orders_roles_without_mutating_parent(self):
        bank = self.bank()
        original = copy.deepcopy(bank)
        selected = self.execute(H3ReferenceSelectTool, {"reference_ids": ["music", "portrait"], "roles": {"portrait": "first_frame"}}, {"bank": bank})
        self.assertEqual([ref["id"] for ref in selected.metadata["h3_references"]], ["music", "portrait"])
        self.assertEqual(selected.metadata["h3_references"][1]["role"], "first_frame")
        self.assertEqual(selected.metadata["h3_references"][1]["semantic_role"], "identity")
        self.assertEqual(bank, original)
        for config in [{}, {"reference_ids": []}, {"reference_ids": ["absent"]},
                       {"reference_ids": ["music", "music"]}, {"reference_ids": [[]]},
                       {"reference_ids": ["music"], "roles": {"music": "first_frame"}},
                       {"reference_ids": ["music"], "roles": {"portrait": "last_frame"}}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.execute(H3ReferenceSelectTool, config, {"bank": bank})

    def test_pack_uses_ordered_upstream_bindings_not_task_uris(self):
        bank = self.bank()
        self.task.metadata = {"h3_references": [{"uri": "https://invalid.example/fake.png"}]}
        parents = {"bank": bank, "image": self.artifact(self.image, "image", "image-id"),
                   "video": self.artifact(self.blue, ident="video-id"), "audio": self.artifact(self.audio, "audio", "audio-id")}
        bindings = [{"source": "audio", "kind": "audio", "role": "reference_audio"},
                    {"source": "bank", "reference_id": "portrait", "kind": "image", "role": "first_frame"},
                    {"source": "video", "kind": "video", "role": "reference_video"},
                    {"source": "image", "kind": "image", "role": "last_frame", "semantic_role": "ending"}]
        packed = self.execute(H3ReferencePackTool, {"bindings": bindings}, parents)
        refs = packed.metadata["h3_references"]
        self.assertEqual([ref["id"] for ref in refs], ["audio", "portrait", "video", "image"])
        self.assertEqual([ref["role"] for ref in refs], ["reference_audio", "first_frame", "reference_video", "last_frame"])
        self.assertEqual(packed.metadata["source_nodes"], ["audio", "bank", "video", "image"])
        self.assertEqual(refs[-1]["semantic_role"], "ending")
        self.assertEqual(refs[2]["uri"], str(self.blue.resolve()))

    def test_pack_direct_ids_stay_selectable_across_new_artifact_ids(self):
        bindings = [{"source": "frame", "kind": "image", "role": "first_frame"},
                    {"source": "sound", "kind": "audio", "role": "reference_audio", "reference_id": "theme"}]
        for run in range(2):
            with self.subTest(run=run):
                packed = self.execute(H3ReferencePackTool, {"bindings": bindings}, {
                    "frame": self.artifact(self.image, "image", f"image-run-{run}"),
                    "sound": self.artifact(self.audio, "audio", f"audio-run-{run}"),
                })
                self.assertEqual([ref["id"] for ref in packed.metadata["h3_references"]], ["frame", "theme"])
                self.assertEqual(packed.metadata["source_artifact_ids"], [f"image-run-{run}", f"audio-run-{run}"])
                selected = self.execute(H3ReferenceSelectTool, {"reference_ids": ["theme", "frame"]}, {"pack": packed})
                self.assertEqual([ref["id"] for ref in selected.metadata["h3_references"]], ["theme", "frame"])

    def test_select_reassigns_semantic_roles_independently_without_mutation(self):
        bank = self.bank()
        original = copy.deepcopy(bank)
        selected = self.execute(H3ReferenceSelectTool, {
            "reference_ids": ["music", "portrait"], "roles": {"portrait": "first_frame"},
            "semantic_roles": {"music": "ambient soundtrack", "portrait": "opening pose"},
        }, {"bank": bank})
        refs = selected.metadata["h3_references"]
        self.assertEqual([(ref["role"], ref["semantic_role"]) for ref in refs],
                         [("reference_audio", "ambient soundtrack"), ("first_frame", "opening pose")])
        semantic_only = self.execute(H3ReferenceSelectTool, {
            "reference_ids": ["portrait"], "semantic_roles": {"portrait": "lighting reference"},
        }, {"bank": bank})
        self.assertEqual(semantic_only.metadata["h3_references"][0]["role"], "reference_image")
        self.assertEqual(bank, original)
        for semantic_roles in (None, [], {"clip": "not selected"}, {"music": ""},
                               {"music": "  "}, {"music": None}, {"music": 1}):
            with self.subTest(semantic_roles=semantic_roles), self.assertRaises(ValueError):
                self.execute(H3ReferenceSelectTool, {"reference_ids": ["music"], "semantic_roles": semantic_roles}, {"bank": bank})

    def test_pack_rejects_ambiguous_set_wrong_kind_nonparent_and_fake_artifact(self):
        parents = {"bank": self.bank(), "fake": self.artifact(self.image, "character_sheet")}
        cases = [[], [{"source": "missing", "kind": "image", "role": "first_frame"}],
                 [{"source": "bank", "kind": "image", "role": "first_frame"}],
                 [{"source": "bank", "reference_id": "music", "kind": "image", "role": "first_frame"}],
                 [{"source": "bank", "reference_id": "missing", "kind": "image", "role": "first_frame"}],
                 [{"source": "fake", "kind": "image", "role": "first_frame"}]]
        for bindings in cases:
            with self.subTest(bindings=bindings), self.assertRaises(ValueError):
                self.execute(H3ReferencePackTool, {"bindings": bindings}, parents)
        context = ToolExecutionContext("node", {"bindings": [{"source": "not-parent", "kind": "image", "role": "first_frame"}]}, {}, {"not-parent": self.artifact(self.image, "image")})
        with self.assertRaises(ValueError):
            H3ReferencePackTool(self.output).run_with_context(self.task, self.plan, context)

    def test_pack_singleton_set_and_duplicate_ids(self):
        selected = self.execute(H3ReferenceSelectTool, {"reference_ids": ["portrait"]}, {"bank": self.bank()})
        binding = {"source": "set", "kind": "image", "role": "last_frame"}
        packed = self.execute(H3ReferencePackTool, {"bindings": [binding]}, {"set": selected})
        self.assertEqual(packed.metadata["h3_references"][0]["id"], "portrait")
        self.assertEqual(packed.metadata["source_lineage"][0]["parents"][0]["source_node"], "bank")
        with self.assertRaisesRegex(ValueError, "duplicate reference id"):
            self.execute(H3ReferencePackTool, {"bindings": [binding, binding]}, {"set": selected})

    def test_frame_extract_first_last_time_are_real_pixels(self):
        changing = self.output / "changing.mp4"
        run_media(["ffmpeg", "-v", "error", "-y", "-i", str(self.red), "-i", str(self.blue),
                   "-filter_complex", "[1:v]scale=64:48,setsar=1,fps=12[b];[0:v][b]concat=n=2:v=1:a=0[v]",
                   "-map", "[v]", "-c:v", "libx264", str(changing)])
        for position, extra, red in [("first", {}, True), ("last", {}, False), ("time", {"time_seconds": 1.5}, False)]:
            with self.subTest(position=position):
                result = self.execute(H3FrameExtractTool, {"position": position, **extra}, {"video": self.artifact(changing)})
                path = result.metadata["reference_image"]
                self.assertTrue(Path(path).is_file())
                self.assertEqual(result.metadata["artifact_type"], "image")
                self.assertEqual(result.metadata["h3_role"], result.metadata["role"])
                self.assertEqual(result.metadata["h3_role"], {
                    "first": "first_frame", "last": "last_frame", "time": "reference_image",
                }[position])
                rgb = self.rgb(path)
                self.assertGreater(rgb[0] if red else rgb[2], 200)
                self.assertEqual(result.metadata["source_artifact_ids"], ["parent"])
        overridden = self.execute(H3FrameExtractTool, {"position": "first", "role": "last_frame"},
                                  {"video": self.artifact(changing)})
        self.assertEqual(overridden.metadata["h3_role"], "last_frame")

    def test_frame_extract_rejects_invalid_selectors_and_bounds(self):
        for config in [{}, {"position": []}, {"position": "middle"}, {"position": "time"},
                       {"position": "time", "time_seconds": -0.1}, {"position": "time", "time_seconds": 1},
                       {"position": "time", "time_seconds": float("inf")},
                       {"position": "time", "time_seconds": True},
                       {"position": "time", "time_seconds": "0.5"},
                       {"position": "first", "time_seconds": 0}, {"position": "first", "role": "reference_audio"}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.execute(H3FrameExtractTool, config, {"video": self.artifact(self.red)})
        with self.assertRaises(ValueError):
            self.execute(H3FrameExtractTool, {"position": "first"}, {"a": self.artifact(self.red), "b": self.artifact(self.blue)})

    def test_trim_video_materializes_shorter_video_with_all_audio_and_frames(self):
        self.require_processor()
        self.task.metadata["h3_references"][1]["uri"] = str(self.multi)
        bank = self.bank()
        original = copy.deepcopy(bank)
        result = self.execute(H3ReferenceTrimTool, {"reference_id": "clip", "start_seconds": 0.25, "end_seconds": 0.75}, {"bank": bank})
        refs = result.metadata["h3_references"]
        path = refs[1]["uri"]
        self.assertNotEqual(path, str(self.multi))
        self.assertAlmostEqual(refs[1]["duration_seconds"], 0.5, delta=0.1)
        audio = [s for s in probe(path)["streams"] if s["codec_type"] == "audio"]
        self.assertEqual(len(audio), 2)
        for track in range(2):
            self.assertGreater(self.rms(path, 0.1, 0.2, track), 0.02)
        self.assertTrue(result.frames)
        self.assertTrue(all(Path(frame["frame_path"]).is_file() for frame in result.frames))
        self.assertEqual(refs[0], bank.metadata["h3_references"][0])
        self.assertEqual(refs[2], bank.metadata["h3_references"][2])
        self.assertEqual(bank, original)

    def test_trim_audio_is_real_and_keeps_semantics(self):
        bank = self.bank()
        result = self.execute(H3ReferenceTrimTool, {"reference_id": "music", "start_seconds": 0.2, "end_seconds": 0.6}, {"bank": bank})
        ref = result.metadata["h3_references"][2]
        self.assertEqual(ref["role"], "reference_audio")
        self.assertAlmostEqual(ref["duration_seconds"], 0.4, delta=0.03)
        self.assertEqual(Path(ref["uri"]).suffix, ".wav")
        info = probe(ref["uri"])
        self.assertEqual(info["format"]["format_name"], "wav")
        self.assertEqual([s["codec_type"] for s in info["streams"]], ["audio"])
        self.assertEqual(info["streams"][0]["codec_name"], "pcm_s16le")
        self.assertGreater(self.rms(ref["uri"], 0, 0.2), 0.02)
        self.assert_tone(ref["uri"], 0, 0.3, 660)
        self.assertTrue(result.metadata["trimmed_media"]["has_audio"])

    def test_trim_rejects_invalid_bounds_and_image_references(self):
        bank = self.bank()
        config = {"reference_id": "clip", "start_seconds": 0, "end_seconds": 0.5}
        for change in [{"reference_id": "portrait"}, {"reference_id": "absent"}, {"start_seconds": -1},
                       {"start_seconds": 0.5}, {"end_seconds": 0}, {"end_seconds": 2},
                       {"start_seconds": float("nan")}, {"end_seconds": True}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.execute(H3ReferenceTrimTool, {**config, **change}, {"bank": bank})

    def test_concat_normalizes_order_retains_sound_and_adds_only_missing_silence(self):
        self.require_processor()
        parents = {"red": self.artifact(self.red, ident="red-id"), "blue": self.artifact(self.blue, ident="blue-id")}
        result = self.execute(H3AVConcatTool, {"source_nodes": ["blue", "red"]}, parents)
        path = result.metadata["local_video_path"]
        info = probe(path)
        self.assertAlmostEqual(float(info["format"]["duration"]), 2, delta=0.1)
        self.assertTrue(any(s["codec_type"] == "audio" for s in info["streams"]))
        video = next(s for s in info["streams"] if s["codec_type"] == "video")
        self.assertEqual((video["width"], video["height"]), (80, 60))
        self.assertEqual(Fraction(video["avg_frame_rate"]), 24)
        for stream in info["streams"]:
            if stream["codec_type"] == "audio":
                self.assertEqual(int(stream["sample_rate"]), 32000)
                self.assertEqual(stream["channels"], 2)
                self.assertEqual(stream["channel_layout"], "stereo")
        self.assertLess(self.rms(path, 0.2, 0.5), 0.001)
        self.assertGreater(self.rms(path, 1.2, 0.5), 0.02)
        self.assert_tone(path, 1.2, 0.5, 440)
        self.assertGreater(self.rgb(path, 0.4)[2], 200)
        self.assertGreater(self.rgb(path, 1.4)[0], 200)
        self.assertEqual(result.metadata["source_nodes"], ["blue", "red"])
        self.assertEqual(result.metadata["source_artifact_ids"], ["blue-id", "red-id"])
        self.assertEqual([entry["silent_tracks_added"] for entry in result.metadata["concat_sources"]], [1, 0])
        self.assertTrue(result.metadata["has_audio"])
        self.assertTrue(result.frames)

    def test_concat_keeps_multiple_soundtracks_and_repeated_sources(self):
        self.require_processor()
        result = self.execute(H3AVConcatTool, {"source_nodes": ["multi", "multi"]}, {"multi": self.artifact(self.multi)})
        path = result.metadata["local_video_path"]
        self.assertEqual(len(result.metadata["audio_streams"]), 2)
        for track in range(2):
            for start in (0.2, 1.2):
                self.assertGreater(self.rms(path, start, 0.5, track), 0.02)
                self.assert_tone(path, start, 0.5, 440 if track == 0 else 880, track)

    def test_concat_all_silent_still_materializes_audio(self):
        self.require_processor()
        result = self.execute(H3AVConcatTool, {"source_nodes": ["silent"]}, {"silent": self.artifact(self.blue)})
        self.assertEqual(len(result.metadata["audio_streams"]), 1)
        self.assertLess(self.rms(result.metadata["local_video_path"], 0.2, 0.5), 0.001)

    def test_concat_preserves_delayed_audio_alignment(self):
        self.require_processor()
        delayed = self.output / "delayed audio.mp4"
        run_media(["ffmpeg", "-v", "error", "-y", "-i", str(self.blue), "-itsoffset", "0.4",
                   "-i", str(self.audio), "-map", "0:v", "-map", "1:a", "-t", "1", "-c:v", "copy", "-c:a", "aac", str(delayed)])
        result = self.execute(H3AVConcatTool, {"source_nodes": ["delayed", "red"]},
                              {"delayed": self.artifact(delayed), "red": self.artifact(self.red)})
        path = result.metadata["local_video_path"]
        self.assertLess(self.rms(path, 0.05, 0.2), 0.001)
        self.assertGreater(self.rms(path, 0.55, 0.2), 0.02)
        self.assertGreater(self.rms(path, 1.2, 0.5), 0.02)
        self.assertEqual([entry["silent_tracks_added"] for entry in result.metadata["concat_sources"]], [0, 0])

    def test_concat_requires_explicit_parent_selection(self):
        parents = {"video": self.artifact(self.red), "image": self.artifact(self.image, "image")}
        for config in [{}, {"source_nodes": []}, {"source_nodes": "video"}, {"source_nodes": ["absent"]},
                       {"source_nodes": ["image"]}, {"source_nodes": [[]]}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.execute(H3AVConcatTool, config, parents)

    def test_subprocess_uses_argv_timeout_and_reports_failures(self):
        with patch("evovideo_skill.h3_media.subprocess.run", side_effect=subprocess.TimeoutExpired("ffprobe", 120)) as run:
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                self.bank()
            args, kwargs = run.call_args
            self.assertIsInstance(args[0], list)
            self.assertGreater(kwargs["timeout"], 0)
            self.assertFalse(kwargs.get("shell", False))
        with patch("evovideo_skill.h3_media.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "ffprobe is required"):
                self.bank()


if __name__ == "__main__":
    unittest.main()
