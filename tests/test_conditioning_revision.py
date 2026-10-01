"""Exercise CLI revision gates without model or API calls."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.conditioning_runner import main


class RuntimeReached(Exception):
    pass


class RevisionGateTests(unittest.TestCase):
    def invoke(self, tier, dry_run, revision=None):
        name = "h3_conditioning_research.json" if tier == "research" else "h3_conditioning_graph_search.json"
        config = json.loads((Path("configs") / name).read_text())
        config["task_file"] = "examples/research_smoke_tasks.json"
        env = {} if revision is None else {"H3_LOCAL_MODEL_REVISION": revision}
        with TemporaryDirectory() as folder:
            config["output_dir"] = str(Path(folder) / "run")
            path = Path(folder) / "config.json"
            path.write_text(json.dumps(config))
            argv = ["conditioning", "--config", str(path)] + (["--dry-run"] if dry_run else [])
            output = io.StringIO()
            with patch.dict(os.environ, env, clear=True), patch("sys.argv", argv), redirect_stdout(output), patch(
                "evovideo_skill.conditioning_verifier.resolve_profiles", return_value=None), patch(
                "evovideo_skill.h3_cli.preflight") as preflight, patch(
                "evovideo_skill.conditioning_runner.build_runtime", side_effect=RuntimeReached) as runtime:
                error = None
                try:
                    main()
                except (RuntimeReached, ValueError) as exc:
                    error = exc
                return output.getvalue(), error, preflight, runtime

    def test_dry_run_allows_unknown_revision_for_both_tiers(self):
        for tier in ("pilot", "research"):
            with self.subTest(tier=tier):
                output, error, preflight, runtime = self.invoke(tier, True)
                self.assertIsNone(error)
                self.assertIn('"h3_local_model_revision": "unspecified"', output)
                self.assertIn("WARNING", output)
                self.assertFalse(preflight.call_args.kwargs["require_credentials"])
                runtime.assert_not_called()

    def test_pilot_execution_allows_unknown_but_keeps_credentials_check(self):
        for revision in (None, "", "   ", "unspecified"):
            with self.subTest(revision=revision):
                output, error, preflight, runtime = self.invoke("pilot", False, revision)
                self.assertIsInstance(error, RuntimeReached)
                self.assertTrue(preflight.call_args.kwargs["require_credentials"])
                self.assertEqual(runtime.call_args.args[0].h3_local_model_revision, "unspecified")

    def test_research_execution_requires_revision(self):
        for revision in (None, "", "   ", "unspecified"):
            with self.subTest(revision=revision):
                output, error, preflight, runtime = self.invoke("research", False, revision)
                self.assertIsInstance(error, ValueError)
                self.assertIn("research runs require", str(error))
                preflight.assert_not_called()
                runtime.assert_not_called()

    def test_declared_revision_is_preserved(self):
        revision = "a" * 40
        output, error, preflight, runtime = self.invoke("research", False, revision)
        self.assertIsInstance(error, RuntimeReached)
        self.assertEqual(runtime.call_args.args[0].h3_local_model_revision, revision)
        self.assertNotIn("unknown weight revision", output)
