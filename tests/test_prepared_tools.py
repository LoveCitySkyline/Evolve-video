from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from evovideo_skill.prepared_tools import PreparedCapability, PreparedToolStore
from evovideo_skill.runtime import RuntimeSettings, build_runtime
from evovideo_skill.tool_onboarding import (
    CapabilityRequest,
    CommandToolManifest,
    OnboardingResult,
    ToolOnboardingManager,
)
from evovideo_skill.tools import ToolRegistry


class PreparedToolStoreTests(unittest.TestCase):
    @staticmethod
    def _manifest(name: str = "prepared_deflicker") -> CommandToolManifest:
        return CommandToolManifest.from_dict(
            {
                "name": name,
                "capability": "temporal_deflickering",
                "input_types": ["video"],
                "output_type": "video",
                "backend": "python",
                "verified": True,
                "provenance": "github:example/deflicker@abc123",
                "command": [sys.executable, "-c", "raise SystemExit(0)"],
                "output_arg": "",
            }
        )

    def test_publish_and_restore_frozen_catalog_without_acquirer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PreparedToolStore(Path(tmpdir) / "shared")
            request = CapabilityRequest(
                "temporal_deflickering", required_input_types=["video"]
            )
            report = store.publish(
                [self._manifest()],
                [PreparedCapability(request, required=True)],
                [OnboardingResult(request, "registered", "prepared_deflicker")],
                {"prepared_deflicker"},
            )

            registry = ToolRegistry()
            manager = ToolOnboardingManager(
                registry,
                store.catalog_path,
                Path(tmpdir) / "runtime",
                restore_catalog_tools=True,
            )

            self.assertEqual(report["required_failures"], [])
            self.assertTrue(registry.has("prepared_deflicker"))
            self.assertIsNone(manager.open_world_acquirer)
            self.assertEqual(
                registry.spec("prepared_deflicker").capability,
                "temporal_deflickering",
            )

    def test_required_failure_is_recorded_while_catalog_is_published(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PreparedToolStore(tmpdir)
            request = CapabilityRequest(
                "image_conditioned_video_generation", required_input_types=["image"]
            )
            report = store.publish(
                [],
                [PreparedCapability(request, required=True)],
                [OnboardingResult(request, "blocked", evidence=["build failed"])],
                set(),
            )

            self.assertEqual(
                report["required_failures"],
                ["image_conditioned_video_generation"],
            )
            self.assertEqual(json.loads(store.catalog_path.read_text())["tools"], [])
            self.assertTrue(store.report_path.exists())

    def test_restore_filters_invalid_legacy_physical_tool_from_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PreparedToolStore(Path(tmpdir) / "shared")
            invalid_audio = CommandToolManifest.from_dict(
                {
                    "name": "legacy_audio_wrapper",
                    "capability": "audio_conditioned_video_generation",
                    "input_types": ["audio"],
                    "output_type": "video",
                    "backend": "micromamba",
                    "verified": True,
                    "command": [
                        sys.executable,
                        "-c",
                        "raise SystemExit(0)",
                        "--seed",
                        "{seed}",
                        "--output",
                        "{output_video}",
                    ],
                    "output_arg": "",
                }
            )
            store.publish(
                [self._manifest(), invalid_audio],
                [],
                [],
                {"prepared_deflicker", "legacy_audio_wrapper"},
            )

            registry = ToolRegistry()
            manager = ToolOnboardingManager(
                registry,
                store.catalog_path,
                Path(tmpdir) / "runtime",
                restore_catalog_tools=True,
            )

            self.assertTrue(registry.has("prepared_deflicker"))
            self.assertFalse(registry.has("legacy_audio_wrapper"))
            self.assertNotIn(
                "legacy_audio_wrapper",
                {item["name"] for item in manager.discoverable_tools()},
            )
            self.assertTrue(
                any("reference_audio" in item for item in manager.restore_evidence)
            )

    def test_runtime_restores_frozen_tools_without_llm_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PreparedToolStore(Path(tmpdir) / "shared")
            request = CapabilityRequest(
                "temporal_deflickering", required_input_types=["video"]
            )
            store.publish(
                [self._manifest()],
                [PreparedCapability(request, required=True)],
                [OnboardingResult(request, "registered", "prepared_deflicker")],
                {"prepared_deflicker"},
            )

            registry, vlm_augmenter = build_runtime(
                RuntimeSettings(
                    provider="local-fake",
                    video_output_dir=str(Path(tmpdir) / "videos"),
                    agent_state_dir=str(Path(tmpdir) / "state"),
                    tool_catalog_path=str(store.catalog_path),
                    restore_catalog_tools=True,
                    enable_llm_mutation=False,
                    enable_open_world_tools=False,
                )
            )

            self.assertTrue(registry.has("prepared_deflicker"))
            self.assertIsNone(vlm_augmenter)

    def test_capability_file_canonicalizes_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "capabilities.json"
            path.write_text(
                json.dumps(
                    {
                        "capabilities": [
                            {
                                "capability": "i2v",
                                "required_input_types": ["image"],
                                "required": False,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            loaded = PreparedToolStore.load_capabilities(path)

            self.assertEqual(
                loaded[0].request.capability,
                "image_conditioned_video_generation",
            )
            self.assertFalse(loaded[0].required)


if __name__ == "__main__":
    unittest.main()
