import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from evovideo_skill import cli, h3_cli
from evovideo_skill.api_tools import VideoApiError
from evovideo_skill.benchmarks import BenchmarkSuite
from evovideo_skill.harness import HarnessConfig
from evovideo_skill.models import VideoTask
from evovideo_skill.runtime import RuntimeSettings, build_runtime, settings_from_namespace, with_env_overrides


class H3LocalEntryTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.media_commands = patch("evovideo_skill.h3_cli.shutil.which", return_value="/test/bin/tool")
        self.media_commands.start()
        self.addCleanup(self.media_commands.stop)
        self.media_modules = patch("evovideo_skill.h3_cli.importlib.util.find_spec", return_value=object())
        self.media_modules.start()
        self.addCleanup(self.media_modules.stop)
        self.tasks = [VideoTask(f"task-{index}", "A robot waves.", duration_seconds=6) for index in range(9)]
        self.suites = patch("evovideo_skill.h3_cli.load_suites", return_value=[BenchmarkSuite("local", self.tasks)])
        self.suites.start()
        self.addCleanup(self.suites.stop)

    def config(self, **settings):
        return HarnessConfig(
            auto_bootstrap_assets=False,
            runtime=RuntimeSettings(provider="local-h3", enable_vlm_eval=True, **settings),
        )

    def test_defaults_and_env_overrides_round_trip_through_config(self):
        defaults = with_env_overrides(RuntimeSettings(provider="local-h3"))
        self.assertEqual(defaults.resolution, "768P")
        self.assertEqual(defaults.h3_fl2va_url, "http://127.0.0.1:30010")
        self.assertEqual(defaults.h3_ref2va_url, "http://127.0.0.1:30011")
        self.assertEqual(defaults.h3_local_model_revision, "unspecified")
        self.assertEqual(defaults.h3_local_quality, "lossless")
        env = {
            "PROVIDER": "local-h3", "H3_FL2VA_URL": "http://worker:31010",
            "H3_REF2VA_URL": "http://worker:31011", "H3_LOCAL_MODEL_REVISION": "revision-123",
            "H3_LOCAL_QUALITY": "extra-high", "H3_MAX_API_CALLS": "17",
            "H3_TIMEOUT_SECONDS": "1800", "H3_HTTP_TIMEOUT_SECONDS": "30",
        }
        with patch.dict(os.environ, env):
            resolved = with_env_overrides(RuntimeSettings())
        self.assertEqual(resolved.h3_fl2va_url, env["H3_FL2VA_URL"])
        self.assertEqual(resolved.h3_ref2va_url, env["H3_REF2VA_URL"])
        self.assertEqual(resolved.h3_local_model_revision, "revision-123")
        self.assertEqual(resolved.h3_local_quality, "extra-high")
        self.assertEqual(resolved.h3_max_api_calls, 17)
        self.assertEqual(resolved.timeout_seconds, 1800)
        self.assertEqual(resolved.h3_http_timeout_seconds, 30)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "config.json"
            path.write_text(json.dumps({"runtime": asdict(resolved)}))
            restored = HarnessConfig.from_file(path).runtime
        self.assertEqual(asdict(restored), asdict(resolved))

    def test_default_preflight_is_offline_and_reports_seed_protocol(self):
        with patch("evovideo_skill.h3_cli.build_h3_local_client") as client, patch(
            "evovideo_skill.h3_cli.build_request", side_effect=AssertionError("cloud transport used")
        ):
            result = h3_cli.preflight(self.config(h3_max_api_calls=17))
        client.assert_not_called()
        self.assertEqual(result["resolution"], "768P")
        self.assertEqual(result["comparison_protocol"], "matched_task_seed_replicates")
        self.assertEqual(result["replicates"], 3)
        self.assertTrue(result["provider_seed_control"])
        self.assertIn("not bit-identical", result["seed_control_note"])
        self.assertIsNone(result["server_health"])
        self.assertEqual(result["api_calls_during_preflight"], 0)
        self.assertEqual(result["max_api_calls"], 17)
        self.assertEqual(result["h3_max_api_calls"], 17)
        self.assertEqual(result["h3_local_quality"], "lossless")
        self.assertEqual(result["h3_local_model_revision"], "unspecified")

    def test_local_credentials_do_not_require_minimax(self):
        config = self.config(enable_llm_mutation=True, graph_planner_backend="codex")
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "vlm-test"}):
            self.assertEqual(h3_cli.preflight(config, require_credentials=True)["preflight"], "passed")
        with self.assertRaisesRegex(ValueError, "DASHSCOPE_API_KEY"):
            h3_cli.preflight(config, require_credentials=True)

    def test_local_retains_codex_and_api_planner_guards(self):
        config = self.config(enable_llm_mutation=True, graph_planner_backend="codex")
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "vlm-test"}), patch(
            "evovideo_skill.h3_cli.shutil.which", side_effect=lambda name: None if name == "codex" else "/test/bin/tool"
        ), self.assertRaisesRegex(ValueError, "Codex planner"):
            h3_cli.preflight(config, require_credentials=True)
        config.runtime.graph_planner_backend = "api"
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "vlm-test"}), patch(
            "evovideo_skill.llm_graph_mutation.mutation_config_from_env", return_value=SimpleNamespace(api_key=None)
        ), self.assertRaisesRegex(ValueError, "graph planner endpoint"):
            h3_cli.preflight(config, require_credentials=True)

    def test_check_server_returns_health_and_propagates_failure(self):
        with patch("evovideo_skill.h3_cli.build_h3_local_client") as client:
            health = {"fl2va": {"status": "ok"}, "ref2va": {"status": "ok"}}
            client.return_value.check_health.return_value = health
            result = h3_cli.preflight(self.config(), check_server=True)
            self.assertEqual(result["server_health"], health)
            client.return_value.check_health.assert_called_once_with()
            client.return_value.generate.assert_not_called()
            client.return_value.check_health.side_effect = VideoApiError("unhealthy Ref2VA")
            with self.assertRaisesRegex(VideoApiError, "unhealthy Ref2VA"):
                h3_cli.preflight(self.config(), check_server=True)

    def test_local_replicates_and_quality_guards(self):
        config = self.config()
        config.evaluation_seeds = [42, 42, 123]
        with self.assertRaisesRegex(ValueError, "3 evaluation"):
            h3_cli.preflight(config)
        with self.assertRaisesRegex(ValueError, "lossless or extra-high"):
            h3_cli.preflight(self.config(h3_local_quality="high"))
        with self.assertRaises((ValueError, VideoApiError)):
            h3_cli.preflight(self.config(resolution="2K"))
        for seed in (True, -1, 2**32, "42", 1.5):
            config.evaluation_seeds = [seed, 123, 456]
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "evaluation seeds must be integers"):
                h3_cli.preflight(config)

    def test_native_preflight_accepts_file_above_cloud_inline_limit(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "large.mp4"
            with source.open("wb") as handle:
                handle.truncate(51 * 1024 * 1024)
            self.tasks[0].metadata["h3_references"] = [
                {"id": "clip", "kind": "video", "role": "reference_video", "uri": str(source)}
            ]
            info = {"streams": [{"codec_type": "video", "width": 768}], "format": {"duration": "6"}}
            with patch("evovideo_skill.h3_api.probe_media", return_value=info), patch(
                "evovideo_skill.h3_local.probe_media", return_value=info
            ), patch("evovideo_skill.h3_api._transport_uri", side_effect=AssertionError("cloud inline transport used")):
                result = h3_cli.preflight(self.config())
            self.assertEqual(result["preflight"], "passed")
            self.assertEqual(result["suites"][0]["reference_tasks"], 1)

    def test_native_builder_receives_modes_shots_and_raw_references(self):
        image = {"id": "frame", "kind": "image", "role": "first_frame", "uri": "/shared/frame.png"}
        reference = {"id": "source", "kind": "video", "role": "reference_video", "uri": "/shared/large.mp4"}
        self.tasks[0].metadata["h3_references"] = [image]
        self.tasks[1].metadata["h3_references"] = [reference]
        self.tasks[1].duration_seconds = 20
        self.tasks[1].metadata["h3_shots"] = [
            {"prompt": "First shot.", "duration_seconds": 10},
            {"prompt": "Second shot.", "duration_seconds": 10},
        ]
        with patch("evovideo_skill.h3_local.build_local_request", return_value={}) as native, patch(
            "evovideo_skill.h3_cli.build_request", side_effect=AssertionError("cloud transport used")
        ):
            h3_cli.preflight(self.config())
        native.assert_any_call("A robot waves.", "fl2va", [image], 6, "768P", "16:9")
        native.assert_any_call("First shot.", "ref2va", [reference], 10, "768P", "16:9")
        native.assert_any_call("Second shot.", "ref2va", [reference], 10, "768P", "16:9")
        native.assert_any_call("A robot waves.", "t2va", [], 6, "768P", "16:9")
        self.assertEqual(native.call_count, 10)
        self.assertTrue(all(len(call.args) == 6 and not call.kwargs for call in native.call_args_list))

    def test_runtime_builds_local_client_checks_health_and_reuses_registration(self):
        with tempfile.TemporaryDirectory() as root:
            settings = RuntimeSettings(
                provider="local-h3", video_output_dir=root, h3_max_api_calls=19,
                h3_fl2va_url="http://worker:30010", h3_ref2va_url="http://worker:30011",
                h3_local_model_revision="revision-123", h3_local_quality="extra-high",
                timeout_seconds=1800, h3_http_timeout_seconds=30, poll_interval_seconds=2,
                sample_frames=12, enable_vlm_eval=True, h3_audio_verifier_command='["audio-test"]',
            )
            def local_client(config):
                return Mock(config=config, root=Path(root), jobs=Path(root) / "jobs")
            with patch("evovideo_skill.h3_local.H3LocalClient", side_effect=local_client) as local, patch(
                "evovideo_skill.h3_api.H3Client", side_effect=AssertionError("cloud client used")
            ), patch("evovideo_skill.h3_api.register_h3_tools") as register, patch(
                "evovideo_skill.runtime.build_vlm_augmenter", return_value=SimpleNamespace(evaluator=Mock())
            ) as vlm, redirect_stdout(io.StringIO()):
                registry, augmenter = build_runtime(settings)
            config = local.call_args.args[0]
            self.assertEqual(config.resolution, "768P")
            self.assertEqual(config.fl2va_url, settings.h3_fl2va_url)
            self.assertEqual(config.ref2va_url, settings.h3_ref2va_url)
            self.assertEqual(config.model_revision, "revision-123")
            self.assertEqual(config.quality, "extra-high")
            self.assertEqual(config.max_api_calls, 19)
            self.assertEqual(config.timeout_seconds, 1800)
            self.assertEqual(config.http_timeout_seconds, 30)
            self.assertEqual(config.poll_interval_seconds, 2)
            self.assertEqual(config.sample_frames, 12)
            client = register.call_args.args[1]
            register.assert_called_once_with(registry, client)
            client.check_health.assert_called_once_with()
            vlm.assert_called_once()
            self.assertEqual(type(augmenter.evaluator).__name__, "H3MultimodalEvaluator")

    def test_runtime_unhealthy_server_prevents_registration(self):
        with patch("evovideo_skill.runtime.build_h3_local_client") as client, patch(
            "evovideo_skill.h3_api.register_h3_tools"
        ) as register:
            client.return_value.check_health.side_effect = VideoApiError("unhealthy FL2VA")
            with self.assertRaisesRegex(VideoApiError, "unhealthy FL2VA"):
                build_runtime(RuntimeSettings(provider="local-h3"))
            register.assert_not_called()

    def test_real_local_client_registers_native_tools_without_cloud_credentials(self):
        with tempfile.TemporaryDirectory() as root, patch(
            "evovideo_skill.h3_local.H3LocalClient.check_health", return_value={"fl2va": "ok", "ref2va": "ok"}
        ) as health, redirect_stdout(io.StringIO()):
            registry, augmenter = build_runtime(RuntimeSettings(
                provider="local-h3", video_output_dir=root, h3_local_model_revision="test-revision",
            ))
            for name in ("mock_text_to_video", "h3_t2va", "h3_fl2va", "h3_ref2va"):
                self.assertEqual(registry.spec(name).backend, "local-h3")
                self.assertTrue(registry.get(name).client.provider_seed_control)
                self.assertEqual(registry.get(name).client.config.resolution, "768P")
            self.assertIn("h3_reference_bank", registry.available_names())
            self.assertIsNone(augmenter)
            health.assert_called_once_with()

    def test_entry_checks_server_only_when_requested_or_running(self):
        for action, flag, expected in (("preflight", [], False), ("preflight", ["--check-server"], True), ("run", [], True)):
            with self.subTest(action=action, flag=flag), patch(
                "sys.argv", ["h3-entry", action, *flag]
            ), patch("evovideo_skill.h3_cli.HarnessConfig.from_file", return_value=self.config()) as load, patch(
                "evovideo_skill.h3_cli.preflight", return_value={}
            ) as preflight, patch("evovideo_skill.h3_cli.GraphHarnessRunner") as runner, redirect_stdout(io.StringIO()):
                runner.return_value.run.return_value = SimpleNamespace(output_dir="output", records=[])
                h3_cli.main()
                preflight.assert_called_once_with(load.return_value, require_credentials=action == "run", check_server=expected)
                self.assertEqual(runner.call_count, int(action == "run"))

    def test_entry_unhealthy_server_prevents_run(self):
        with patch("sys.argv", ["h3-entry", "run"]), patch(
            "evovideo_skill.h3_cli.HarnessConfig.from_file", return_value=self.config()
        ), patch("evovideo_skill.h3_cli.preflight", side_effect=VideoApiError("unhealthy")), patch(
            "evovideo_skill.h3_cli.GraphHarnessRunner"
        ) as runner, self.assertRaisesRegex(VideoApiError, "unhealthy"):
            h3_cli.main()
        runner.assert_not_called()

    def test_generic_cli_accepts_local_fields_and_existing_budget_name(self):
        with patch("sys.argv", [
            "evovideo", "graph-evolve", "--provider", "local-h3",
            "--h3-fl2va-url", "http://worker:31010", "--h3-ref2va-url", "http://worker:31011",
            "--h3-local-model-revision", "revision-123", "--h3-local-quality", "extra-high",
            "--h3-max-api-calls", "7",
        ]), patch("evovideo_skill.cli.run_graph_evolve") as run:
            cli.main()
        settings = settings_from_namespace(run.call_args.args[0])
        self.assertEqual(settings.provider, "local-h3")
        self.assertEqual(settings.h3_fl2va_url, "http://worker:31010")
        self.assertEqual(settings.h3_ref2va_url, "http://worker:31011")
        self.assertEqual(settings.h3_local_model_revision, "revision-123")
        self.assertEqual(settings.h3_local_quality, "extra-high")
        self.assertEqual(settings.h3_max_api_calls, 7)

    def test_cloud_preflight_keeps_resolution_protocol_and_credentials(self):
        config = self.config()
        config.runtime.provider = "minimax-h3"
        with patch("evovideo_skill.h3_cli.build_h3_local_client") as local:
            result = h3_cli.preflight(config, check_server=True)
        local.assert_not_called()
        self.assertEqual(result["resolution"], "2K")
        self.assertEqual(result["comparison_protocol"], "task_matched_unseeded_replicates")
        self.assertNotIn("server_health", result)
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "vlm-test"}), self.assertRaisesRegex(ValueError, "MINIMAX_API_KEY"):
            h3_cli.preflight(config, require_credentials=True)


if __name__ == "__main__":
    unittest.main()
