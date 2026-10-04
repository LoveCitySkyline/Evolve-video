from copy import deepcopy
import base64
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.api_tools import VideoApiError
from evovideo_skill.conditioning_runner import ConditioningRunner, MeasurementUnavailable
from evovideo_skill.conditioning_verifier import (ConditioningVideoVerifier, GENERIC,
    parse_judgment, resolve_profiles, windows, combine_window_judgments)
from evovideo_skill.models import TaskMode, VideoArtifact, VideoTask
from evovideo_skill.runtime import RuntimeSettings


def observed(score=.8, count=1):
    return {"status": "observed", "score": score, "confidence": .7,
            "evidence": "Visible movement at 00:01 with identity preserved.",
            "segments": [{"segment_id": i, "status": "observed", "score": score,
                          "evidence": "Visible movement in this interval."} for i in range(count)]}


def profile(**kwargs):
    return resolve_profiles({"verifier": {"runtime": kwargs}}, RuntimeSettings(), require_keys=False)["runtime"]


class EvidenceContractTests(unittest.TestCase):
    def test_global_minimum_uses_fixed_windows_without_discarding_global_lows_or_unknowns(self):
        spans = [{"segment_id": i} for i in range(3)]
        full = observed(1, 3)
        full["segments"][2].update(status="unobserved", score=None,
                                   evidence="No frames for 12–18s.")
        components = [observed(1) for _ in spans]
        for i, row in enumerate(components):
            row["segments"][0]["segment_id"] = i
        before = deepcopy(full)
        result = combine_window_judgments(full, components, spans)
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["full_video_judgment"], before)
        self.assertEqual(full, before)
        self.assertEqual(result["segments"][2]["evidence_source"], "fixed_window_clip")
        components[2]["segments"][0]["score"] = .2
        self.assertEqual(combine_window_judgments(full, components, spans)["score"], .2)
        components[2]["segments"][0]["score"] = 1
        full["score"] = .1  # Cross-cut failure cannot be rescued by stable isolated clips.
        self.assertEqual(combine_window_judgments(full, components, spans)["score"], .1)
        full["score"] = 1
        full["segments"][0]["score"] = .3
        self.assertEqual(combine_window_judgments(full, components, spans)["score"], .3)
        full.update(status="unobserved", score=None)
        self.assertIsNone(combine_window_judgments(full, components, spans)["score"])
        full.update(status="observed", score=1)
        components[2].update(status="unobserved", score=None)
        components[2]["segments"][0].update(status="unobserved", score=None)
        self.assertIsNone(combine_window_judgments(full, components, spans)["score"])
        with self.assertRaisesRegex(ValueError, "every fixed-window"):
            combine_window_judgments(full, components[:2], spans)

    def test_global_window_components_allow_inapplicability_but_not_inconsistent_statuses(self):
        row = observed()
        row.update(status="not_applicable", score=None)
        row["segments"][0].update(status="not_applicable", score=None)
        rubric = {"geometry": {"story_shot_index": 0, "window_component_of_global": True, "mandatory": True}}
        parsed = parse_judgment({"criteria": {"geometry": row}}, rubric, [{"segment_id": 0}])["geometry"]
        result = combine_window_judgments(observed(), [parsed], [{"segment_id": 0}])
        self.assertEqual(result["status"], "unobserved")
        self.assertIsNone(result["score"])
        row.update(status="observed", score=1)
        with self.assertRaisesRegex(ValueError, "matching top-level"):
            parse_judgment({"criteria": {"geometry": row}}, rubric, [{"segment_id": 0}])

    def test_observation_basis_requires_consistent_status_without_choosing_scores(self):
        rubric = {"event": {"story_shot_index": 0, "evidence_status_contract": "visible-outcome-v1"}}
        spans = [{"segment_id": 0}]
        for basis, status, score in (("visible_match", "observed", 1),
                                     ("visible_mismatch", "observed", 0),
                                     ("visible_mismatch", "observed", .25),
                                     ("insufficient_evidence", "unobserved", None)):
            row = observed(score)
            row.update(observation_basis=basis, status=status)
            row["segments"][0]["status"] = status
            original = deepcopy(row)
            with self.subTest(basis=basis, score=score):
                parsed = parse_judgment({"criteria": {"event": row}}, rubric, spans)["event"]
                self.assertEqual(parsed, original)
                self.assertEqual(row, original)
        for basis, top, segment in (("visible_mismatch", "unobserved", "unobserved"),
                                    ("insufficient_evidence", "observed", "observed"),
                                    ("visible_match", "observed", "unobserved"),
                                    (None, "observed", "observed"),
                                    ([], "observed", "observed")):
            row = observed()
            row.update(observation_basis=basis, status=top, score=None if top == "unobserved" else .8)
            row["segments"][0].update(status=segment, score=None if segment == "unobserved" else .8)
            original = deepcopy(row)
            with self.subTest(basis=basis, top=top, segment=segment):
                with self.assertRaisesRegex(ValueError, "observation_basis"):
                    parse_judgment({"criteria": {"event": row}}, rubric, spans)
                self.assertEqual(row, original)

    def test_real_relay_baton_response_preserves_target_scores_and_evidence(self):
        from evovideo_skill.story_contracts import prepare_story_task, acceptance_report
        root = Path(__file__).resolve().parents[1]
        tasks = json.loads((root / "benchmarks/story350/story350_smoke15.json").read_text())["tasks"]
        task = prepare_story_task(VideoTask.from_dict(next(t for t in tasks if t["task_id"] == "story350-relay_baton")))
        raw = json.loads((root / "tests/fixtures/relay_baton_shot_judgments.json").read_text())
        original = deepcopy(raw)
        rubric = {k: task.metadata["evaluation"][k] for k in raw["criteria"]}
        parsed = parse_judgment(raw, rubric, windows(task))
        for name, row in parsed.items():
            self.assertEqual(row["score"], raw["criteria"][name]["score"])
            target = rubric[name]["story_shot_index"]
            self.assertEqual(row["segments"][target], raw["criteria"][name]["segments"][0])
            for other in row["segments"]:
                if other["segment_id"] != target:
                    self.assertEqual(other["status"], "not_applicable")
                    self.assertIsNone(other["score"])
                    self.assertEqual(other["applicability_source"], "original_task_contract")
        self.assertEqual(raw, original)
        artifact = VideoArtifact("a", task.task_id, "", "generation", [], [], {"vlm_evaluation": {
            "criterion_observations": {k: [v] for k, v in parsed.items()},
            "verification_metadata": {"windows": windows(task)}}})
        report = acceptance_report(task, artifact)
        self.assertEqual(report["status"], "failed")
        for name, row in raw["criteria"].items():
            self.assertEqual(report["checks"][name]["score"], row["score"])
            self.assertEqual(report["checks"][name]["status"], "passed" if row["score"] >= .9 else "failed")

    def test_shot_scope_never_fills_missing_target_or_invalid_segment_ids(self):
        spans = [{"segment_id": i} for i in range(3)]
        rubric = {"shot": {"story_shot_index": 1}}
        for ids in ([], [0], [0, 2], [1, 1], [1, 3], ["1"], [True]):
            with self.subTest(ids=ids):
                row = observed()
                row["segments"] = [{**observed()["segments"][0], "segment_id": i} for i in ids]
                with self.assertRaisesRegex(ValueError, "shot:"):
                    parse_judgment({"criteria": {"shot": row}}, rubric, spans)
        for index in (True, "1", -1, 3, None):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "story_shot_index"):
                parse_judgment({"criteria": {"shot": observed()}}, {"shot": {"story_shot_index": index}}, spans)

    def test_shot_scope_unknown_is_not_promoted_and_target_cannot_be_na(self):
        spans = [{"segment_id": i} for i in range(3)]
        rubric = {"shot": {"story_shot_index": 0}}
        row = observed()
        row["segments"][0].update(status="unobserved", score=None)
        parsed = parse_judgment({"criteria": {"shot": row}}, rubric, spans)["shot"]
        self.assertEqual(parsed["status"], "unobserved")
        self.assertIsNone(parsed["score"])
        row["segments"][0]["status"] = "not_applicable"
        with self.assertRaisesRegex(ValueError, "mandatory criterion"):
            parse_judgment({"criteria": {"shot": row}}, rubric, spans)

    def test_global_and_unscoped_story_criteria_still_require_all_windows(self):
        spans = [{"segment_id": i} for i in range(3)]
        for name in ("identity_consistency_score", "story.s0.event.example"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "required=\\[0, 1, 2\\]"):
                parse_judgment({"criteria": {name: observed()}}, {name: {}}, spans)
        row = observed(.25, 3)
        result = parse_judgment({"criteria": {"shot": row}}, {"shot": {"story_shot_index": 1}}, spans)
        self.assertEqual(result["shot"], row)

    def test_missing_and_boolean_scores_are_rejected(self):
        spans = [{"segment_id": 0}]
        with self.assertRaises(ValueError):
            parse_judgment({"criteria": {}}, {"identity": {}}, spans)
        row = observed(True)
        with self.assertRaises(ValueError):
            parse_judgment({"criteria": {"identity": row}}, {"identity": {}}, spans)

    def test_unobservable_is_not_zero(self):
        row = observed()
        row.update(status="unobserved", score=None)
        result = parse_judgment({"criteria": {"identity": row}}, {"identity": {}}, [{"segment_id": 0}])
        self.assertIsNone(result["identity"]["score"])
        row["score"] = 0
        with self.assertRaises(ValueError):
            parse_judgment({"criteria": {"identity": row}}, {"identity": {}}, [{"segment_id": 0}])

    def test_minimum_over_segments_and_window_coverage(self):
        row = observed(.9, 2)
        row["segments"][1]["score"] = .2
        rubric = {"identity": {"aggregation": "minimum_over_segments"}}
        spans = [{"segment_id": 0}, {"segment_id": 1}]
        result = parse_judgment({"criteria": {"identity": row}}, rubric, spans)
        self.assertEqual(result["identity"]["score"], .2)
        row["segments"][1]["segment_id"] = 0
        with self.assertRaises(ValueError):
            parse_judgment({"criteria": {"identity": row}}, rubric, spans)

    def test_identity_scope_contradictions_request_review_without_inflating_score(self):
        rubric = {"identity_across_shots": {"scoring_scope": "visible_appearance_only"}}
        for score, appearance, basis in ((0, "stable", "structure"), (.5, "stable", "appearance"),
                                         (1, "changed", "appearance"), (.8, "unobservable", "appearance")):
            row = observed(score)
            row["scope_checks"] = {"appearance_status": appearance, "structure_status": "absent", "score_basis": basis}
            result = parse_judgment({"criteria": {"identity_across_shots": row}}, rubric, [{"segment_id": 0}])["identity_across_shots"]
            self.assertEqual(result["status"], "unobserved")
            self.assertIsNone(result["score"])
            self.assertTrue(result["scope_issues"])
        row = observed(1)
        row["scope_checks"] = {"appearance_status": "stable", "structure_status": "absent", "score_basis": "appearance"}
        result = parse_judgment({"criteria": {"identity_across_shots": row}}, rubric, [{"segment_id": 0}])["identity_across_shots"]
        self.assertEqual(result["score"], 1)
        row["segments"][0]["score"] = 0
        self.assertEqual(parse_judgment({"criteria": {"identity_across_shots": row}}, rubric, [{"segment_id": 0}])["identity_across_shots"]["status"], "unobserved")

    def test_identity_scope_missing_checks_are_not_silently_accepted(self):
        result = parse_judgment({"criteria": {"identity_across_shots": observed()}},
            {"identity_across_shots": {"scoring_scope": "visible_appearance_only"}}, [{"segment_id": 0}])
        self.assertEqual(result["identity_across_shots"]["status"], "unobserved")

    def test_fixed_task_windows_not_candidate_selected(self):
        task = VideoTask("x", "scene", duration_seconds=12,
            metadata={"h3_shots": [{"duration_seconds": 4}, {"duration_seconds": 8}]})
        self.assertEqual(windows(task)[1]["start_seconds"], 4)
        task.metadata = {}
        self.assertEqual(len(windows(task)), 2)

    def test_independent_final_model_configuration(self):
        config = {"verifier": {"require_independent_final": True}}
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "different final"):
            resolve_profiles(config, RuntimeSettings(), False)
        env = {"CONDITION_FINAL_VERIFIER_TRANSPORT": "gemini_video",
               "CONDITION_FINAL_VERIFIER_MODEL": "test-gemini",
               "CONDITION_FINAL_VERIFIER_BASE_URL": "https://generativelanguage.googleapis.com/v1beta",
               "CONDITION_FINAL_VERIFIER_API_KEY_ENV": "GEMINI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            result = resolve_profiles(config, RuntimeSettings(), False)
        self.assertTrue(result["independent_model"])
        self.assertEqual(result["final"]["transport"], "gemini_video")

    def test_review_status_stops_quality_admission(self):
        artifact = VideoArtifact("a", "t", "x", "generation", [], [], {"vlm_evaluation": {
            "evaluation_status": "needs_review", "verification_metadata": {"judgment_path": "/evidence"}}})
        with self.assertRaisesRegex(MeasurementUnavailable, "/evidence"):
            ConditioningRunner.check_measurement(artifact, .9)

    def test_only_pure_unknown_evidence_is_eligible_for_training_exclusion(self):
        from evovideo_skill.conditioning_runner import EvidenceIncomplete
        artifact = VideoArtifact("a", "t", "x", "generation", [], [], {"vlm_evaluation": {
            "evaluation_status": "needs_review", "verification_metadata": {
                "judgment_path": "/evidence", "unobserved_criteria": ["holder"],
                "disagreement_criteria": [], "scope_issues": {}, "verifier_protocol": "v9"}}})
        with self.assertRaises(EvidenceIncomplete):
            ConditioningRunner.check_measurement(artifact, .9)
        for changes in ({"disagreement_criteria": ["holder"]}, {"scope_issues": {"holder": ["wrong scope"]}},
                        {"verifier_protocol": None}, {"unobserved_criteria": []}):
            copy = deepcopy(artifact)
            copy.metadata["vlm_evaluation"]["verification_metadata"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(MeasurementUnavailable) as caught:
                ConditioningRunner.check_measurement(copy, .9)
            self.assertNotIsInstance(caught.exception, EvidenceIncomplete)

    def test_credentials_cannot_be_embedded_in_logged_profiles(self):
        for changes in ({"api_key": "secret"}, {"base_url": "https://example.test/v1?key=secret"}):
            with self.assertRaises(ValueError):
                profile(**changes)


class VerifierTests(unittest.TestCase):
    def test_group_routing_preserves_references_and_excludes_other_windows(self):
        verifier = ConditioningVideoVerifier(profile(criteria_per_call=2), self.root)
        criteria = {"global": {}, "shot2a": {"story_shot_index": 2},
                    "shot0": {"story_shot_index": 0}, "shot2b": {"story_shot_index": 2},
                    "shot2c": {"story_shot_index": 2}}
        groups = list(verifier.criterion_groups(list(criteria), criteria))
        self.assertEqual([list(g) for g in groups], [["global"], ["shot2a", "shot2b"], ["shot2c"], ["shot0"]])
        clips = [{"segment_id": i, "media_label": f"clip{i}", "source_time_offset_seconds": 6 * i} for i in range(3)]
        for i, clip in enumerate(clips):
            clip["boundary_frames"] = [{"media_label": f"first{i}"}, {"media_label": f"last{i}"}]
        manifest = {"full_candidate_label": "full", "window_clips": clips}
        evidence = [(label, {"data": label}) for label in ("reference", "full", "clip0", "clip1", "clip2")]
        evidence += [(label, {"data": label}) for i in range(3) for label in (f"first{i}", f"last{i}")]
        before = deepcopy(manifest)
        media, view = verifier.group_evidence(evidence, manifest, groups[1])
        self.assertEqual([label for label, _ in media], ["reference", "clip2", "first2", "last2"])
        self.assertEqual(view["evaluation_view"]["source_time_offset_seconds"], 12)
        self.assertEqual(view["window_clips"], [clips[2]])
        media, view = verifier.group_evidence(evidence, manifest, groups[0])
        self.assertEqual([label for label, _ in media], ["reference", "full"])
        self.assertEqual(view["evaluation_view"]["kind"], "full_video")
        self.assertEqual(manifest, before)
        with self.assertRaisesRegex(ValueError, "missing physically extracted"):
            verifier.group_evidence([(label, m) for label, m in evidence if label != "clip2"], manifest, groups[1])
        with self.assertRaisesRegex(ValueError, "missing boundary frame"):
            verifier.group_evidence(evidence[:-1], manifest, groups[1])
        with self.assertRaisesRegex(ValueError, "exactly one temporal scope"):
            verifier.group_evidence(evidence, manifest, criteria)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires ffmpeg")
    def test_physical_window_clips_contain_only_correct_frames_and_use_local_time(self):
        video = self.root / "three_colors.mp4"
        command = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
        for color in ("red", "green", "blue"):
            command += ["-f", "lavfi", "-i", f"color=c={color}:s=96x64:r=4:d=6"]
        subprocess.run(command + ["-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
            "-map", "[v]", "-c:v", "libx264", str(video)], check=True, capture_output=True)
        self.task.duration_seconds = 18
        self.task.metadata = {"h3_shots": [{"duration_seconds": 6} for _ in range(3)],
            "evaluation": {"scene_geometry": {"aggregation": "minimum_over_segments"},
                           **{f"story.s{i}.event": {"story_shot_index": i} for i in range(3)}}}
        self.artifact.metadata["local_video_path"] = str(video)
        verifier = ConditioningVideoVerifier(profile(), self.root / "judge")
        evidence, manifest = verifier.evidence(self.task, self.artifact)
        self.assertEqual(len(manifest["window_clips"]), 3)
        for i, clip in enumerate(manifest["window_clips"]):
            medium = dict(evidence)[clip["media_label"]]
            path = self.root / f"decoded-window-{i}.mp4"
            path.write_bytes(base64.b64decode(medium["data"]))
            self.assertEqual(clip["sampled_frame_count"], 12)
            self.assertAlmostEqual(clip["clip_duration_seconds"], 6)
            self.assertEqual(clip["source_time_offset_seconds"], i * 6)
            pixels = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", "scale=1:1",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], check=True, capture_output=True).stdout
            self.assertEqual(len(pixels), 36)
            self.assertTrue(all(max(range(3), key=lambda c: pixels[j + c]) == i for j in range(0, len(pixels), 3)))
            frames = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path)],
                check=True, capture_output=True).stdout)["frames"]
            self.assertEqual(float(frames[0]["best_effort_timestamp_time"]), 0)
            self.assertEqual(float(frames[-1]["best_effort_timestamp_time"]), 5.5)
        with patch("evovideo_skill.conditioning_verifier.subprocess.run", wraps=subprocess.run) as run:
            self.assertEqual(verifier.evidence(self.task, self.artifact), (evidence, manifest))
        self.assertFalse(any(call.args[0][0] == "ffmpeg" for call in run.call_args_list))
        calls = []
        def request(prompt, media, operation):
            payload = json.loads(prompt)
            view = payload["evidence_manifest"]["evaluation_view"]
            calls.append(view)
            self.assertEqual(len(media), 1 if view["kind"] == "full_video" else 3)
            if view["kind"] == "fixed_window_clip":
                self.assertEqual([m["mime"] for _, m in media], ["video/mp4", "image/png", "image/png"])
                self.assertEqual([b["boundary"] for b in view["boundary_frames"]], ["first", "last"])
                self.assertLess(view["boundary_frames"][-1]["source_timestamp_seconds"], view["end_seconds"])
            rows = {}
            for name, rule in payload["criteria"].items():
                row = observed(0, 3 if "story_shot_index" not in rule else 1)
                if "story_shot_index" in rule:
                    row["segments"][0]["segment_id"] = rule["story_shot_index"]
                    row["observation_basis"] = "visible_mismatch"
                    self.assertEqual(view["segment_id"], rule["story_shot_index"])
                rows[name] = row
            return {"criteria": rows}
        with patch.object(verifier, "request", side_effect=request):
            result = verifier.evaluate(self.task, self.artifact)
        self.assertEqual(result["evaluation_status"], "complete")
        self.assertEqual([v["kind"] for v in calls], ["full_video"] + ["fixed_window_clip"] * 3)
        self.assertEqual(calls[-1]["source_time_offset_seconds"], 12)
        self.assertEqual(result["criterion_scores"]["scene_geometry"], 0)
        geometry = result["criterion_observations"]["scene_geometry"][0]
        self.assertEqual(len(geometry["fixed_window_judgments"]), 3)
        self.assertEqual(geometry["full_video_judgment"]["score"], 0)
        # A global minimum criterion requires physical clips even without story.* metrics.
        self.task.metadata["evaluation"] = {"scene_geometry": {"aggregation": "minimum_over_segments"}}
        self.assertEqual(len(verifier.evidence(self.task, self.artifact)[1]["window_clips"]), 3)
        folder = Path(result["verification_metadata"]["judgment_path"])
        self.assertEqual(json.loads((folder / "group-003.evidence.json").read_text())["evaluation_view"]["segment_id"], 2)
        self.task.duration_seconds = 24
        self.task.metadata["h3_shots"][-1]["duration_seconds"] = 12
        with self.assertRaisesRegex(ValueError, "does not cover fixed window 2"):
            verifier.evidence(self.task, self.artifact)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires ffmpeg")
    def test_boundary_last_frame_preserves_change_after_last_uniform_sample(self):
        video = self.root / "late_change.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
            "color=c=red:s=96x64:r=24:d=6", "-vf", "drawbox=x=0:y=0:w=iw:h=ih:color=blue:t=fill:enable='eq(n,143)'",
            "-c:v", "libx264", str(video)], check=True, capture_output=True)
        verifier = ConditioningVideoVerifier(profile(), self.root / "judge")
        span = {"segment_id": 0, "start_seconds": 0, "end_seconds": 6}
        sampled = verifier.media(video, "video", span)
        times = verifier.frame_times(video, sampled["source_hash"])
        frames = verifier.boundary_frames(video, span, times, sampled["source_hash"])
        last = frames[-1][1]["boundary_metadata"]
        self.assertEqual(last["source_frame_index"], 143)
        self.assertAlmostEqual(last["source_timestamp_seconds"], 143 / 24, places=5)
        self.assertGreater(last["clip_timestamp_seconds"], 5.5)
        def pixels(data, suffix):
            path = self.root / ("inspect" + suffix)
            path.write_bytes(base64.b64decode(data))
            return subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", "scale=1:1",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], check=True, capture_output=True).stdout
        sampled_pixels = pixels(sampled["data"], ".mp4")
        self.assertTrue(all(sampled_pixels[j] > sampled_pixels[j + 2] for j in range(0, len(sampled_pixels), 3)))
        first_pixel = pixels(frames[0][1]["data"], ".png")
        last_pixel = pixels(frames[-1][1]["data"], ".png")
        self.assertGreater(first_pixel[0], first_pixel[2])
        self.assertGreater(last_pixel[2], last_pixel[0])
        # Adjacent windows must not borrow the next window's first frame.
        split = {"segment_id": 0, "start_seconds": 0, "end_seconds": 3}
        before_cut = verifier.boundary_frames(video, split, times, sampled["source_hash"])
        self.assertEqual(before_cut[-1][1]["boundary_metadata"]["source_frame_index"], 71)
        with self.assertRaisesRegex(ValueError, "no original candidate frames"):
            verifier.boundary_frames(video, {"segment_id": 1, "start_seconds": 6, "end_seconds": 7}, times, sampled["source_hash"])

    def test_global_minimum_preplans_all_windows_for_every_repeat_and_keeps_unknown(self):
        self.task.metadata["evaluation"] = {"scene_geometry": {"aggregation": "minimum_over_segments"}}
        verifier = ConditioningVideoVerifier(profile(repeats=2), self.root)
        calls = []
        def request(prompt, media, operation):
            data = json.loads(prompt)
            view = data["evidence_manifest"]["evaluation_view"]
            calls.append(view["kind"])
            rows = {k: observed(1) for k in data["criteria"]}
            if view["kind"] == "full_video":
                self.assertEqual(data["criteria"]["scene_geometry"]["aggregation"], "full_video_assessment")
                rows["scene_geometry"]["segments"][0].update(status="unobserved", score=None)
            else:
                self.assertTrue(data["criteria"]["scene_geometry"]["window_component_of_global"])
                if len(calls) == 4:
                    rows["scene_geometry"].update(status="unobserved", score=None)
                    rows["scene_geometry"]["segments"][0].update(status="unobserved", score=None)
            return {"criteria": rows}
        with patch.object(verifier, "evidence", return_value=(self.evidence, self.manifest)), patch.object(
                verifier, "request", side_effect=request):
            result = verifier.evaluate(self.task, self.artifact)
            self.assertEqual(result, verifier.evaluate(self.task, self.artifact))
        self.assertEqual(calls, ["full_video", "full_video", "fixed_window_clip", "fixed_window_clip"])
        self.assertEqual(result["evaluation_status"], "needs_review")
        self.assertNotIn("scene_geometry", result["criterion_scores"])
        self.assertEqual(result["criterion_observations"]["scene_geometry"][0]["score"], 1)
        self.assertIsNone(result["criterion_observations"]["scene_geometry"][1]["score"])

    def test_story_status_contract_and_identity_context_preserve_valid_unknown_or_low_score(self):
        name = "story.s0.event.walk"
        identity = "A has auburn hair and a teal jacket. B has gray hair and a navy apron."
        self.task.metadata.update(h3_global_constraints=identity, evaluation={name: {"story_shot_index": 0}})
        original_task = deepcopy(self.task.metadata)
        for basis, status, score in (("visible_mismatch", "observed", 0),
                                     ("insufficient_evidence", "unobserved", None)):
            with self.subTest(basis=basis):
                verifier = ConditioningVideoVerifier(profile(), self.root / basis)
                prompts = []
                def request(prompt, evidence, operation):
                    data = json.loads(prompt)
                    prompts.append(data)
                    rows = {k: observed() for k in data["criteria"]}
                    if name in rows:
                        rows[name].update(observation_basis=basis, status=status, score=score,
                            evidence="The required action is missing; occlusion prevents deciding whether it occurred.")
                        rows[name]["segments"][0].update(status=status, score=score)
                    return {"criteria": rows}
                with patch.object(verifier, "evidence", return_value=(self.evidence, self.manifest)), patch.object(
                        verifier, "request", side_effect=request) as call:
                    result = verifier.evaluate(self.task, self.artifact)
                self.assertEqual(call.call_count, 2)
                self.assertEqual(prompts[1]["frozen_identity_context"]["requirements"], identity)
                self.assertEqual(prompts[1]["criteria"][name]["evidence_status_contract"], "visible-outcome-v1")
                self.assertEqual(prompts[1]["output_contract"]["observation_basis"]["required_for"], [name])
                if status == "observed":
                    self.assertEqual(result["evaluation_status"], "complete")
                    self.assertEqual(result["criterion_scores"][name], 0)
                else:
                    self.assertEqual(result["evaluation_status"], "needs_review")
                    self.assertNotIn(name, result["criterion_scores"])
                    folder = Path(result["verification_metadata"]["judgment_path"])
                    self.assertTrue((folder / "needs_review.json").exists())
                self.assertEqual(self.task.metadata, original_task)

    def test_contradictory_story_status_has_one_correction_with_same_media_and_can_remain_unknown(self):
        name = "story.s0.event.walk"
        self.task.metadata["evaluation"] = {name: {"story_shot_index": 0}}
        verifier = ConditioningVideoVerifier(profile(), self.root)
        calls = []
        def request(prompt, media, operation):
            data = json.loads(prompt)
            calls.append((data, media))
            rows = {k: observed() for k in data["criteria"]}
            if name in rows:
                rows[name].update(status="unobserved", score=None,
                    observation_basis="visible_mismatch" if len(calls) == 2 else "insufficient_evidence")
                rows[name]["segments"][0].update(status="unobserved", score=None)
            return {"criteria": rows}
        with patch.object(verifier, "evidence", return_value=(self.evidence, self.manifest)), patch.object(
                verifier, "request", side_effect=request):
            result = verifier.evaluate(self.task, self.artifact)
            self.assertEqual(result, verifier.evaluate(self.task, self.artifact))
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[1][1], calls[2][1])
        for key in ("original_task", "criteria", "evidence_manifest", "frozen_identity_context"):
            self.assertEqual(calls[1][0][key], calls[2][0][key])
        self.assertIn("observation_basis", calls[2][0]["format_feedback"]["error"])
        self.assertEqual(result["evaluation_status"], "needs_review")
        self.assertNotIn(name, result["criterion_scores"])

    def test_scoped_prompt_accepts_sparse_low_score_without_correction_and_caches_raw(self):
        verifier = ConditioningVideoVerifier(profile(), self.root)
        spans = [{"segment_id": i, "start_seconds": i * 6, "end_seconds": (i + 1) * 6} for i in range(3)]
        rubric = {"story.s1.event.pass": {"story_shot_index": 1}, "global": {}}
        scoped = observed(.0)
        scoped["segments"][0]["segment_id"] = 1
        raw = {"criteria": {"story.s1.event.pass": scoped, "global": observed(.25, 3)}}
        path = self.root / "group.json"
        with patch.object(verifier, "request", return_value=raw) as request:
            result = verifier._observe_group(path, {"criteria": rubric}, self.evidence, "test", rubric, spans)
            cached = verifier._observe_group(path, {"criteria": rubric}, self.evidence, "test", rubric, spans)
        self.assertEqual(request.call_count, 1)
        contract = json.loads(request.call_args.args[0])["output_contract"]
        self.assertEqual(contract["required_segment_ids"], {"story.s1.event.pass": [1], "global": [0, 1, 2]})
        self.assertEqual(contract["temporal_windows"], spans)
        self.assertEqual(result, cached)
        self.assertEqual(result["story.s1.event.pass"]["score"], 0)
        self.assertEqual(json.loads(path.read_text()), raw)
        self.assertFalse(path.with_suffix(".correction-1.raw.json").exists())

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = VideoTask("private_task_id", "A person runs", metadata={"evaluation": {"identity": {}}})
        self.artifact = VideoArtifact("private_id", self.task.task_id, "mutated private prompt", "generation", ["best_graph"], [],
            {"local_video_path": "/not-used.mp4", "vlm_evaluation": {"score": 1}, "graph_id": "winner"})
        self.evidence = [("ANONYMOUS CANDIDATE VIDEO", {"mime": "video/mp4", "data": "AA==", "source_hash": "abc"})]
        self.manifest = {"windows": windows(self.task), "candidate_hash": "abc", "references": [], "fps": 2}
        self.manifest["full_candidate_label"] = self.evidence[0][0]
        self.manifest["window_clips"] = [{**windows(self.task)[0], "media_label": "FIXED WINDOW 0",
                                         "source_time_offset_seconds": 0}]
        self.evidence.append(("FIXED WINDOW 0", {"mime": "video/mp4", "data": "AQ==", "source_hash": "abc"}))

    def test_blinded_prompt_and_success_cache(self):
        verifier = ConditioningVideoVerifier(profile(), self.root)
        requests = []
        def request(prompt, media, operation):
            requests.append(prompt)
            data = json.loads(prompt)
            return {"criteria": {k: observed() for k in data["criteria"]}}
        with patch.object(verifier, "evidence", return_value=(self.evidence, self.manifest)), patch.object(verifier, "request", side_effect=request):
            result = verifier.evaluate(self.task, self.artifact)
            again = verifier.evaluate(self.task, self.artifact)
        self.assertEqual(result, again)
        self.assertEqual(len(requests), 1)
        self.assertEqual(result["evaluation_status"], "complete")
        for secret in ("private_task_id", "private_id", "best_graph", "mutated private prompt", '"winner"'):
            self.assertNotIn(secret, requests[0])

    def test_unobserved_and_repeat_disagreement_need_review(self):
        verifier = ConditioningVideoVerifier(profile(repeats=2), self.root)
        replies = [{"criteria": {k: observed(score) for k in [*[k for k in GENERIC if k != "target_edit_success_score"], "identity"]}} for score in (.9, .3)]
        with patch.object(verifier, "evidence", return_value=(self.evidence, self.manifest)), patch.object(verifier, "request", side_effect=replies):
            result = verifier.evaluate(self.task, self.artifact)
        self.assertEqual(result["evaluation_status"], "needs_review")
        self.assertIn("identity", result["verification_metadata"]["disagreement_criteria"])
        self.assertFalse(list(self.root.glob("judgments/*/result.json")))

    def test_mandatory_unobserved_has_no_criterion_score(self):
        verifier = ConditioningVideoVerifier(profile(), self.root)
        raw = {"criteria": {k: observed() for k in [*[k for k in GENERIC if k != "target_edit_success_score"], "identity"]}}
        raw["criteria"]["identity"].update(status="unobserved", score=None)
        with patch.object(verifier, "evidence", return_value=(self.evidence, self.manifest)), patch.object(verifier, "request", return_value=raw):
            result = verifier.evaluate(self.task, self.artifact)
        self.assertNotIn("identity", result["criterion_scores"])
        self.assertEqual(result["evaluation_status"], "needs_review")

    def test_report45_scope_contract_and_contradiction_are_audited(self):
        self.task.metadata["evaluation"] = {
            "identity_across_shots": {"description": "Face and glasses remain consistent"},
            "shot_and_action_coverage": {"description": "Three requested shots"}}
        verifier = ConditioningVideoVerifier(profile(criteria_per_call=8), self.root)
        prompts = []
        def request(prompt, media, operation):
            data = json.loads(prompt)
            prompts.append(data)
            rows = {k: observed(1) for k in data["criteria"]}
            rows["identity_across_shots"].update(score=0, scope_checks={
                "appearance_status": "stable", "structure_status": "absent", "score_basis": "structure"})
            rows["shot_and_action_coverage"] = observed(.25)
            return {"criteria": rows}
        with patch.object(verifier, "evidence", return_value=(self.evidence, self.manifest)), patch.object(verifier, "request", side_effect=request):
            result = verifier.evaluate(self.task, self.artifact)
        self.assertEqual(prompts[0]["criteria"]["identity_across_shots"]["scoring_scope"], "visible_appearance_only")
        self.assertNotIn("scoring_scope", self.task.metadata["evaluation"]["identity_across_shots"])
        self.assertEqual(result["evaluation_status"], "needs_review")
        self.assertNotIn("identity_across_shots", result["criterion_scores"])
        self.assertEqual(result["criterion_scores"]["shot_and_action_coverage"], .25)
        self.assertIn("identity_across_shots", result["verification_metadata"]["scope_issues"])
        self.assertFalse(list(self.root.glob("judgments/*/result.json")))
        self.assertTrue(all("identity_across_shots" not in s["failed_criteria"] for s in result["failed_segments"]))

    def test_generation_edit_metric_is_host_na_but_declared_rubric_stays_mandatory(self):
        for mode, operations, declared in (("generation", [], False), ("editing", [], False),
                                          ("generation", [{"operation": "replace"}], False),
                                          ("generation", [], True)):
            with self.subTest(mode=mode, operations=operations, declared=declared):
                self.task.mode = TaskMode(mode)
                self.task.metadata["edit_operations"] = operations
                self.task.metadata["evaluation"] = {"identity": {}}
                if declared:
                    self.task.metadata["evaluation"]["target_edit_success_score"] = "Required edit"
                verifier = ConditioningVideoVerifier(profile(), self.root / str((mode, bool(operations), declared)))
                requested = []
                def request(prompt, media, operation):
                    criteria = json.loads(prompt)["criteria"]
                    requested.extend(criteria)
                    rows = {k: observed() for k in criteria}
                    if "target_edit_success_score" in rows:
                        rows["target_edit_success_score"].update(status="unobserved", score=None,
                            evidence="Original reference missing; cannot verify the requested edit.")
                    return {"criteria": rows}
                with patch.object(verifier, "evidence", return_value=(self.evidence, self.manifest)), patch.object(
                        verifier, "request", side_effect=request):
                    result = verifier.evaluate(self.task, self.artifact)
                na = mode == "generation" and not operations and not declared
                self.assertEqual(result["evaluation_status"], "complete" if na else "needs_review")
                self.assertEqual("target_edit_success_score" in requested, not na)
                row = result["criterion_observations"]["target_edit_success_score"][0]
                self.assertEqual(row["status"], "not_applicable" if na else "unobserved")
                self.assertIsNone(row["score"])
                if na:
                    self.assertEqual(row["applicability_source"], "original_task_contract")
                    self.assertNotIn("target_edit_success_score", result["criterion_scores"])
                else:
                    self.assertIn("target_edit_success_score", result["verification_metadata"]["unobserved_criteria"])

    def test_explicit_generic_rubric_cannot_be_not_applicable(self):
        row = observed()
        row.update(status="not_applicable", score=None)
        with self.assertRaisesRegex(ValueError, "mandatory criterion"):
            parse_judgment({"criteria": {"target_edit_success_score": row}},
                           {"target_edit_success_score": {"mandatory": True}}, [{"segment_id": 0}])

    def test_wrong_criterion_keys_are_corrected_once_with_same_evidence(self):
        verifier = ConditioningVideoVerifier(profile(), self.root)
        calls = []
        def request(prompt, evidence, operation):
            data = json.loads(prompt)
            calls.append((data, evidence, operation))
            rows = {k: observed(.2) for k in data["criteria"]}
            if len(calls) == 1:
                rows["renamed_identity"] = rows.pop("identity")
            return {"criteria": rows}
        with patch.object(verifier, "evidence", return_value=(self.evidence, self.manifest)), patch.object(
                verifier, "request", side_effect=request):
            result = verifier.evaluate(self.task, self.artifact)
            self.assertEqual(result, verifier.evaluate(self.task, self.artifact))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1], calls[1][1])
        for key in ("original_task", "criteria", "evidence_manifest"):
            self.assertEqual(calls[0][0][key], calls[1][0][key])
        feedback = calls[1][0]["format_feedback"]
        self.assertIn("missing=['identity']", feedback["error"])
        self.assertIn("unexpected=['renamed_identity']", feedback["error"])
        self.assertEqual(result["criterion_scores"]["identity"], .2)
        folder = Path(result["verification_metadata"]["judgment_path"])
        self.assertIn("renamed_identity", json.loads((folder / "group-000-repeat-0.raw.json").read_text())["criteria"])
        self.assertTrue((folder / "group-000-repeat-0.correction-1.raw.json").exists())

    def test_format_retry_exhaustion_is_persistent_and_never_substitutes_scores(self):
        verifier = ConditioningVideoVerifier(profile(), self.root)
        with patch.object(verifier, "evidence", return_value=(self.evidence, self.manifest)), patch.object(
                verifier, "request", return_value={"criteria": {}}) as request:
            for _ in range(2):
                with self.assertRaisesRegex(VideoApiError, "after one correction"):
                    verifier.evaluate(self.task, self.artifact)
        self.assertEqual(request.call_count, 2)
        self.assertFalse(list(self.root.glob("judgments/*/result.json")))
        self.assertFalse(list(self.root.glob("judgments/*/group-000-repeat-0.json")))
        self.assertTrue(list(self.root.glob("judgments/*/*.format-1.json")))

    def test_valid_unobserved_response_is_not_retried_for_better_scores(self):
        verifier = ConditioningVideoVerifier(profile(), self.root)
        def request(prompt, evidence, operation):
            rows = {k: observed() for k in json.loads(prompt)["criteria"]}
            rows["identity"].update(status="unobserved", score=None)
            return {"criteria": rows}
        with patch.object(verifier, "evidence", return_value=(self.evidence, self.manifest)), patch.object(
                verifier, "request", side_effect=request) as call:
            result = verifier.evaluate(self.task, self.artifact)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(result["evaluation_status"], "needs_review")
        self.assertNotIn("identity", result["criterion_scores"])

    def test_interrupted_format_correction_reuses_original_invalid_response(self):
        verifier = ConditioningVideoVerifier(profile(), self.root)
        with patch.object(verifier, "evidence", return_value=(self.evidence, self.manifest)), patch.object(
                verifier, "request", side_effect=[{"criteria": {}}, KeyboardInterrupt()]):
            with self.assertRaises(KeyboardInterrupt):
                verifier.evaluate(self.task, self.artifact)
        def corrected(prompt, evidence, operation):
            data = json.loads(prompt)
            self.assertIn("format_feedback", data)
            return {"criteria": {k: observed() for k in data["criteria"]}}
        with patch.object(verifier, "evidence", return_value=(self.evidence, self.manifest)), patch.object(
                verifier, "request", side_effect=corrected) as request:
            result = verifier.evaluate(self.task, self.artifact)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(result["evaluation_status"], "complete")

    def test_native_transport_payloads_and_no_secret_logs(self):
        media = self.evidence + [("CANDIDATE BOUNDARY FRAME LAST at 5.958333s",
                                  {"mime": "image/png", "data": "AQ==", "source_hash": "boundary"})]
        for transport in ("dashscope_video", "gemini_video"):
            verifier = ConditioningVideoVerifier(profile(transport=transport, api_key_env="TEST_KEY"), self.root / transport)
            response = {"choices": [{"message": {"content": '{"criteria":{}}'}}],
                        "candidates": [{"content": {"parts": [{"text": '{"criteria":{}}'}]}}]}
            captured = []
            def urlopen(request, **kwargs):
                captured.append(json.loads(request.data))
                return io.BytesIO(json.dumps(response).encode())
            with patch.dict(os.environ, {"TEST_KEY": "secret-not-loggable"}), patch("urllib.request.urlopen", side_effect=urlopen):
                self.assertEqual(verifier.request("prompt", media, "unit"), {"criteria": {}})
            if transport == "gemini_video":
                parts = captured[0]["contents"][0]["parts"]
                self.assertTrue(all(part["video_metadata"]["fps"] == 2 for part in parts if "video_metadata" in part))
                self.assertEqual(parts[-1]["inline_data"]["mime_type"], "image/png")
                self.assertNotIn("video_metadata", parts[-1])
            else:
                parts = captured[0]["messages"][1]["content"]
                self.assertTrue(all(part["fps"] == 2 for part in parts if part["type"] == "video_url"))
                self.assertEqual(parts[-1]["type"], "image_url")
                self.assertTrue(parts[-1]["image_url"]["url"].startswith("data:image/png;base64,"))
            self.assertIn("CANDIDATE BOUNDARY FRAME LAST at 5.958333s", json.dumps(captured[0]))
            self.assertNotIn("secret-not-loggable", (verifier.root / "calls.jsonl").read_text())

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires ffmpeg")
    def test_real_media_preparation_and_no_silent_truncation(self):
        video = self.root / "input.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=8",
                        "-t", "2", "-c:v", "libx264", str(video)], check=True)
        verifier = ConditioningVideoVerifier(profile(), self.root / "judge")
        self.artifact.metadata["local_video_path"] = str(video)
        evidence, manifest = verifier.evidence(self.task, self.artifact)
        self.assertTrue(evidence[-1][1]["data"])
        self.assertAlmostEqual(manifest["candidate_duration_seconds"], 2)
        self.assertFalse(manifest["full_rate_motion_verified"])
        verifier.profile["max_media_bytes"] = 1
        with self.assertRaisesRegex(ValueError, "size budget"):
            verifier.evidence(self.task, self.artifact)


if __name__ == "__main__":
    unittest.main()
