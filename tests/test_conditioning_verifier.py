from copy import deepcopy
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
    parse_judgment, resolve_profiles, windows)
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

    def test_credentials_cannot_be_embedded_in_logged_profiles(self):
        for changes in ({"api_key": "secret"}, {"base_url": "https://example.test/v1?key=secret"}):
            with self.assertRaises(ValueError):
                profile(**changes)


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = VideoTask("private_task_id", "A person runs", metadata={"evaluation": {"identity": {}}})
        self.artifact = VideoArtifact("private_id", self.task.task_id, "mutated private prompt", "generation", ["best_graph"], [],
            {"local_video_path": "/not-used.mp4", "vlm_evaluation": {"score": 1}, "graph_id": "winner"})
        self.evidence = [("ANONYMOUS CANDIDATE VIDEO", {"mime": "video/mp4", "data": "AA==", "source_hash": "abc"})]
        self.manifest = {"windows": windows(self.task), "candidate_hash": "abc", "references": [], "fps": 2}

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
        for transport in ("dashscope_video", "gemini_video"):
            verifier = ConditioningVideoVerifier(profile(transport=transport, api_key_env="TEST_KEY"), self.root / transport)
            response = {"choices": [{"message": {"content": '{"criteria":{}}'}}],
                        "candidates": [{"content": {"parts": [{"text": '{"criteria":{}}'}]}}]}
            captured = []
            def urlopen(request, **kwargs):
                captured.append(json.loads(request.data))
                return io.BytesIO(json.dumps(response).encode())
            with patch.dict(os.environ, {"TEST_KEY": "secret-not-loggable"}), patch("urllib.request.urlopen", side_effect=urlopen):
                self.assertEqual(verifier.request("prompt", self.evidence, "unit"), {"criteria": {}})
            if transport == "gemini_video":
                self.assertEqual(captured[0]["contents"][0]["parts"][-1]["video_metadata"]["fps"], 2)
            else:
                self.assertEqual(captured[0]["messages"][1]["content"][-1]["type"], "video_url")
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
