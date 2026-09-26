import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from evovideo_skill.h3_cli import preflight
from evovideo_skill.harness import HarnessConfig


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "configs/h3_local_graph_harness.json"


class LocalH3EntryTests(unittest.TestCase):
    def shell(self, script, *args, env=None, cwd=None):
        return subprocess.run(
            ["bash", str(SCRIPTS / script), *args],
            cwd=cwd or ROOT,
            env={**os.environ, **(env or {})},
            capture_output=True,
            text=True,
            timeout=15,
        )

    def executable(self, root, source):
        path = Path(root) / "fake executable"
        path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(source))
        path.chmod(0o755)
        return str(path)

    def test_shell_syntax(self):
        for name in ("start_h3_local_servers.sh", "run_harness_local_h3_graph.sh",
                     "check_local_h3_graph.sh"):
            with self.subTest(script=name):
                result = subprocess.run(["bash", "-n", str(SCRIPTS / name)],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_config_changes_only_local_runtime_and_output_identity(self):
        local = json.loads(CONFIG.read_text())
        native = json.loads((ROOT / "configs/h3_native_graph_harness.json").read_text())
        self.assertEqual(local["name"], "h3_local_graph_harness")
        self.assertEqual(local["output_dir"], "outputs/harness_h3_local_qwen38_v2")
        expected = {
            "provider": "local-h3",
            "resolution": "768P",
            "video_output_dir": "outputs/harness_h3_local_qwen38_v2/videos",
            "h3_fl2va_url": "http://127.0.0.1:30010",
            "h3_ref2va_url": "http://127.0.0.1:30011",
            "h3_local_model_revision": "unspecified",
            "h3_local_quality": "lossless",
        }
        for key, value in expected.items():
            self.assertEqual(local["runtime"][key], value)
        self.assertEqual(local["runtime"], {**native["runtime"], **expected})
        for key in native.keys() - {"name", "output_dir", "runtime"}:
            self.assertEqual(local[key], native[key], key)

    def test_default_preflight_is_offline_without_generation_credentials(self):
        config = HarnessConfig.from_file(str(CONFIG))
        with patch.dict(os.environ, {}, clear=True), \
                patch("evovideo_skill.h3_cli.shutil.which", return_value="/test/bin/tool"), \
                patch("evovideo_skill.h3_cli.importlib.util.find_spec", return_value=object()), \
                patch("evovideo_skill.h3_cli.build_h3_local_client") as client:
            result = preflight(config)
        client.assert_not_called()
        self.assertEqual(result["provider"], "local-h3")
        self.assertEqual(result["resolution"], "768P")
        self.assertEqual(result["api_calls_during_preflight"], 0)
        self.assertEqual(result["suites"][0]["tasks"], 9)
        self.assertEqual(result["replicates"], 3)
        self.assertEqual(result["h3_local_model_revision"], "unspecified")
        self.assertEqual(result["h3_local_quality"], "lossless")
        self.assertIsNone(result["server_health"])

    def test_preflight_overrides_and_local_credentials(self):
        config = HarnessConfig.from_file(str(CONFIG))
        env = {
            "DASHSCOPE_API_KEY": "test-only",
            "H3_FL2VA_URL": "http://127.0.0.1:31010",
            "H3_REF2VA_URL": "http://127.0.0.1:31011",
            "H3_LOCAL_MODEL_REVISION": "pinned-test-revision",
            "H3_LOCAL_QUALITY": "extra-high",
            "H3_BASE_URL": "https://cloud.invalid",
        }
        with patch.dict(os.environ, env, clear=True), \
                patch("evovideo_skill.h3_cli.shutil.which", return_value="/test/bin/tool"), \
                patch("evovideo_skill.h3_cli.importlib.util.find_spec", return_value=object()), \
                patch("evovideo_skill.h3_cli.build_h3_local_client") as client:
            result = preflight(config, require_credentials=True)
        client.assert_not_called()
        for field in ("h3_fl2va_url", "h3_ref2va_url", "h3_local_model_revision", "h3_local_quality"):
            self.assertEqual(result[field], env[field.upper()])

    def test_local_entry_rejects_high_quality(self):
        config = HarnessConfig.from_file(str(CONFIG))
        with patch.dict(os.environ, {"H3_LOCAL_QUALITY": "high"}, clear=True), \
                self.assertRaisesRegex(ValueError, "lossless or extra-high"):
            preflight(config)

    def test_wrappers_delegate_and_preserve_overrides(self):
        with tempfile.TemporaryDirectory() as root:
            fake = self.executable(root, """
                import json, os, sys
                print(json.dumps({"args": sys.argv[1:], "cwd": os.getcwd(),
                                  "env": dict(os.environ)}))
                sys.exit(17)
            """)
            env = {
                "EVOVIDEO_PYTHON": fake,
                "PROVIDER": "minimax-h3",
                "PYTHONPATH": "/existing/pythonpath",
                "H3_FL2VA_URL": "http://127.0.0.1:31010",
                "H3_REF2VA_URL": "http://127.0.0.1:31011",
                "H3_LOCAL_MODEL_REVISION": "test-model-revision",
                "H3_LOCAL_QUALITY": "extra-high",
            }
            for script, action in (("check_local_h3_graph.sh", "preflight"),
                                   ("run_harness_local_h3_graph.sh", "run")):
                with self.subTest(script=script):
                    args = ["--tasks", "/tmp/tasks with spaces.json", "--output-dir", "/tmp/local results"]
                    result = self.shell(script, *args, env=env, cwd=root)
                    self.assertEqual(result.returncode, 17, result.stderr)
                    recorded = json.loads(result.stdout)
                    self.assertEqual(recorded["args"], ["-m", "evovideo_skill.h3_cli", action,
                                                       "--config", "configs/h3_local_graph_harness.json", *args])
                    self.assertEqual(Path(recorded["cwd"]), ROOT)
                    self.assertEqual(recorded["env"]["PROVIDER"], "local-h3")
                    self.assertEqual(recorded["env"]["PYTHONPATH"], f"{ROOT}/src:/existing/pythonpath")
                    for key in env.keys() - {"PROVIDER", "PYTHONPATH"}:
                        self.assertEqual(recorded["env"][key], env[key])

    def test_dry_run_uses_documented_topology_without_starting_services(self):
        with tempfile.TemporaryDirectory() as root:
            logs = Path(root) / "logs"
            result = self.shell("start_h3_local_servers.sh", "--dry-run", env={
                "H3_SGLANG_BIN": "/not installed/sglang",
                "H3_MODEL_PATH": "/modelweights/pinned snapshot",
                "H3_SERVER_LOG_DIR": str(logs),
                "CUDA_VISIBLE_DEVICES": "99",
            })
            self.assertEqual(result.returncode, 0, result.stderr)
            commands = [shlex.split(line) for line in result.stdout.splitlines()]
            self.assertEqual(len(commands), 2)
            topologies = zip(
                commands,
                ["0,1,2,3", "4,5,6,7"],
                ["fl2va", "ref2va"],
                ["30010", "30011"],
                ["30105", "30205"],
                ["30110", "30210"],
            )
            for args, gpus, variant, port, master, scheduler in topologies:
                self.assertEqual(args[:3], [f"CUDA_VISIBLE_DEVICES={gpus}", "/not installed/sglang", "serve"])
                self.assertEqual(args[3:], [
                    "--model-path", "/modelweights/pinned snapshot",
                    "--model-type", "diffusion",
                    "--backend", "sglang",
                    "--model-id", "MiniMax-H3",
                    "--pipeline-class-name", "MiniMaxH3Pipeline",
                    "--num-gpus", "4",
                    "--tp-size", "2",
                    "--ulysses-degree", "2",
                    "--encoder-parallel", "auto",
                    "--performance-mode", "speed",
                    "--host", "127.0.0.1",
                    "--strict-ports",
                    "--model-variant", variant,
                    "--port", port,
                    "--master-port", master,
                    "--scheduler-port", scheduler,
                ])
            self.assertFalse(logs.exists())

    def test_launcher_rejects_arguments_and_missing_binary(self):
        for args in (("--unknown",), ("--dry-run", "--unknown")):
            result = self.shell("start_h3_local_servers.sh", *args)
            self.assertEqual(result.returncode, 2)
        result = self.shell("start_h3_local_servers.sh", env={"H3_SGLANG_BIN": "/no/such/sglang"})
        self.assertEqual(result.returncode, 127)
        self.assertIn("SGLang executable not found", result.stderr)

    def test_launcher_stops_peer_and_propagates_server_failure(self):
        with tempfile.TemporaryDirectory() as root:
            model = Path(root) / "minimaxh3"
            for relative in ("model_index.json", "FL2VA/model_index.json",
                             "Ref2VA/model_index.json"):
                checkpoint_file = model / relative
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_file.write_text("{}")
            fake = self.executable(root, """
                import json, os, pathlib, signal, sys, time
                root = pathlib.Path(os.environ["FAKE_SERVER_ROOT"])
                variant = sys.argv[sys.argv.index("--model-variant") + 1]
                def stop(signum, frame):
                    (root / (variant + ".stopped")).touch()
                    sys.exit(0)
                signal.signal(signal.SIGTERM, stop)
                (root / (variant + ".json")).write_text(json.dumps({
                    "argv": sys.argv[1:], "gpus": os.environ["CUDA_VISIBLE_DEVICES"]}))
                if variant == "fl2va":
                    deadline = time.monotonic() + 5
                    while not (root / "ref2va.json").exists() and time.monotonic() < deadline:
                        time.sleep(.05)
                    sys.exit(23)
                while True:
                    time.sleep(.05)
            """)
            fake_nvcc = Path(root) / "nvcc"
            fake_nvcc.write_text("#!/usr/bin/env bash\n# Deliberately has no c++20 help text.\nexit 0\n")
            fake_nvcc.chmod(0o755)
            result = self.shell("start_h3_local_servers.sh", env={
                "H3_SGLANG_BIN": fake, "FAKE_SERVER_ROOT": root,
                "H3_MODEL_PATH": str(model),
                "H3_SERVER_LOG_DIR": str(Path(root) / "logs"),
                "H3_NVCC_BIN": str(fake_nvcc),
            })
            self.assertEqual(result.returncode, 23, result.stderr)
            self.assertTrue((Path(root) / "ref2va.stopped").exists())
            for variant, gpus in (("fl2va", "0,1,2,3"), ("ref2va", "4,5,6,7")):
                recorded = json.loads((Path(root) / f"{variant}.json").read_text())
                self.assertEqual(recorded["gpus"], gpus)
                model_path = Path(recorded["argv"][recorded["argv"].index("--model-path") + 1])
                self.assertEqual(model_path.name, "MiniMax-H3")
                self.assertEqual(model_path.resolve(), model.resolve())
            self.assertIn("stopping both services", result.stderr)


if __name__ == "__main__":
    unittest.main()
