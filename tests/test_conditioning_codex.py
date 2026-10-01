"""CLI wiring and real condition-graph decoding, with no remote model calls."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.codex_agents import CodexExecClient, CodexExecConfig, CodexGraphMutationProposer, CodexAgentError
from evovideo_skill.conditioning_planner import CodexConditioningPlanner, ConditioningSmokePlanner, codex_plan_schema, decode_codex_plan, conditioning_h3_grammar, configuration_contract, conditioning_system
from evovideo_skill.conditioning_runner import main, validate_config
from evovideo_skill.conditioning_search import decode_pool
from evovideo_skill.graph_evolver import GraphToolPathEvolver
from evovideo_skill.graph_skill import GraphSkillMemory
from evovideo_skill.h3_cli import preflight
from evovideo_skill.harness import HarnessConfig
from evovideo_skill.llm_graph_mutation import GraphMutationConfig
from evovideo_skill.models import VideoTask
from evovideo_skill.research_protocol import graph_payload
from evovideo_skill.runtime import RuntimeSettings, build_graph_mutation_proposer
from evovideo_skill.skill_memory import SkillMemory


def wire_plan(plan):
    result = deepcopy(plan)
    def graph(value):
        for node in value["nodes"]:
            node["config_json"] = json.dumps(node.pop("config"))
        for edge in value["edges"]:
            edge.pop("config", None)
    def strategy(value):
        value["required_references"] = {**{k: 0 for k in ("image", "video", "audio")}, **value["required_references"]}
    if "anchor" in result:
        graph(result["anchor"])
        for item in result.get("factors", []) or [result["a"], result["b"]]:
            graph(item["graph"])
            strategy(item["strategy"])
        if "joint_strategy" in result:
            strategy(result["joint_strategy"])
    else:
        graph(result["graph"])
        strategy(result["strategy"])
    return result


class CodexConditioningTests(unittest.TestCase):
    def test_cli_returns_conditioning_pool_and_preserves_graph_validation(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            task = VideoTask("codex-test", "A person walks.")
            parent = GraphToolPathEvolver.baseline_graph()
            request = {"experiment": "active_factorial", "condition_only": True,
                       "parent": graph_payload(parent), "task": {"task_id": task.task_id, "duration_seconds": 6}}
            expected = ConditioningSmokePlanner().propose(request)
            calls = []
            def execute(command, **kwargs):
                calls.append((command, kwargs))
                schema = json.loads(Path(command[command.index("--output-schema") + 1]).read_text())
                self.assertEqual(schema["required"], ["anchor", "factors"])
                strategy_schema = schema["properties"]["factors"]["items"]["properties"]["strategy"]
                self.assertFalse(strategy_schema["additionalProperties"])
                self.assertNotIn("used_strategy_ids", strategy_schema["properties"])
                Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps(wire_plan(expected)))
                return subprocess.CompletedProcess(command, 0, "", "")
            client = CodexExecClient(CodexExecConfig(codex_bin=sys.executable, job_root=str(root / "jobs")), runner=execute)
            proposer = CodexGraphMutationProposer(GraphMutationConfig(), client)
            with patch.object(proposer, "_post", side_effect=AssertionError("generic mutation envelope must not be used")), patch.dict(
                os.environ, {"DASHSCOPE_API_KEY": "not-for-planner", "CODEX_GRAPH_TIMEOUT_SECONDS": "123"}):
                planner = CodexConditioningPlanner(proposer)
                result = planner.propose(request)
            self.assertEqual(result, expected)
            command, kwargs = calls[0]
            self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
            self.assertEqual(command[command.index("--ask-for-approval") + 1], "never")
            self.assertNotIn("--approve-for-me", command)
            self.assertNotIn("--search", command)
            self.assertIn('web_search="disabled"', command)
            self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", command)
            self.assertNotIn("DASHSCOPE_API_KEY", kwargs["env"])
            self.assertEqual(kwargs["timeout"], 123)
            self.assertIsNone(planner.last_usage)
            self.assertEqual(Path(kwargs["cwd"]), Path(planner.last_job_dir))
            self.assertIn("Keep all prompts unchanged", kwargs["input"])
            self.assertIn("Every strategy object MUST contain exactly these five fields", kwargs["input"])
            self.assertIn("Never omit required_references", kwargs["input"])
            self.assertIn("A trigger is only a starting event", kwargs["input"])
            config = json.loads(Path("configs/conditioning_graph_search_smoke.json").read_text())
            validate_config(config)
            evolver = GraphToolPathEvolver(SkillMemory(root / "skills"), GraphSkillMemory(root / "graphs"))
            anchor, factors, errors = decode_pool(result, task, parent, evolver.executor, config, {"status": "unlocalized"})
            self.assertEqual(len(factors), 2)
            self.assertEqual(errors, [])
            invalid = deepcopy(result)
            invalid["factors"][0]["graph"]["nodes"][-1]["name"] = "nonexistent_tool"
            _, retained, errors = decode_pool(invalid, task, parent, evolver.executor, config, {"status": "unlocalized"})
            self.assertEqual(len(retained), 1)
            self.assertEqual(len(errors), 1)

    def test_bad_codex_envelope_fails_without_api_fallback(self):
        with TemporaryDirectory() as folder:
            client = CodexExecClient(CodexExecConfig(job_root=folder))
            planner = CodexConditioningPlanner(CodexGraphMutationProposer(GraphMutationConfig(), client))
            with patch.object(planner.codex, "run_json", return_value=({"mutation_json": "{}"}, [], folder)):
                with self.assertRaisesRegex(CodexAgentError, "structured plan"):
                    planner.propose({})

    def test_all_wire_modes_roundtrip_full_graphs_and_frozen_cost(self):
        parent = graph_payload(GraphToolPathEvolver.baseline_graph())
        for experiment in (None, "factorial", "active_factorial"):
            request = {"parent": parent, "task": {"duration_seconds": 6}, "condition_only": True,
                       "memory": [], "experiment": experiment}
            plan = ConditioningSmokePlanner().propose(request)
            decoded = decode_codex_plan(wire_plan(plan), request)
            self.assertEqual(decoded, plan)
            self.assertEqual(configuration_contract(parent)["frozen_parent_configs"]["tool_t2v"],
                             {"cost": parent["nodes"][-1]["config"]["cost"]})

    def test_wire_strategy_rejects_extra_ids_and_nonobject_config(self):
        request = {"experiment": "active_factorial", "parent": graph_payload(GraphToolPathEvolver.baseline_graph()),
                   "task": {"duration_seconds": 6}}
        raw = wire_plan(ConditioningSmokePlanner().propose(request))
        raw["factors"][0]["strategy"]["used_strategy_ids"] = []
        with self.assertRaisesRegex(ValueError, "five declared fields"):
            decode_codex_plan(raw, request)
        del raw["factors"][0]["strategy"]["used_strategy_ids"]
        raw["anchor"]["nodes"][-1]["config_json"] = "[]"
        with self.assertRaisesRegex(ValueError, "JSON object"):
            decode_codex_plan(raw, request)

    def test_conditioning_grammar_no_longer_teaches_forbidden_text_edits(self):
        from evovideo_skill.llm_graph_mutation import OpenAICompatibleGraphMutationProposer
        tools = {"h3_t2va", "h3_ref2va"}
        general = OpenAICompatibleGraphMutationProposer._h3_native_planner(tools, local=True)
        guide = conditioning_h3_grammar(tools)
        self.assertTrue(any("Use config.conditioning_strategy=" in r for r in general["rules"]))
        self.assertFalse(any("conditioning_strategy" in r or "config.prompt" in r for r in guide["rules"]))
        for node in guide["node_examples"]:
            self.assertFalse({"prompt", "conditioning_strategy", "prompt_task_hashes"} & set(node["config"]))
        prompt = conditioning_system({"experiment": "active_factorial", "condition_only": True})
        self.assertIn("complete snapshots, NOT node/edge deltas", prompt)
        self.assertIn("Preserve every other existing config field EXACTLY, including cost", prompt)
        self.assertIn("Do not include used_strategy_ids anywhere", prompt)
        self.assertNotIn("used_strategy_ids must be a subset", prompt)

    def test_codex_factory_and_preflight_need_no_graph_api_key(self):
        config = json.loads(Path("configs/h3_conditioning_graph_search.json").read_text())
        config["runtime"]["graph_planner_backend"] = "codex"
        settings = RuntimeSettings(**config["runtime"])
        with patch.dict(os.environ, {"PATH": os.environ["PATH"], "CODEX_BIN": sys.executable,
                                     "DASHSCOPE_API_KEY": "test-verifier-key"}, clear=True):
            proposer = build_graph_mutation_proposer(settings)
            self.assertIsInstance(proposer, CodexGraphMutationProposer)
            self.assertFalse(proposer.config.api_key)
            preflight(HarnessConfig(name="codex-test", task_files=["examples/research_smoke_tasks.json"],
                                   runtime=settings, evaluation_seeds=[42, 123, 456]), require_credentials=True)

    def test_main_allows_codex_pilot_dry_run_and_rejects_research(self):
        for tier in ("pilot", "research"):
            with self.subTest(tier=tier), TemporaryDirectory() as folder:
                name = "h3_conditioning_research.json" if tier == "research" else "h3_conditioning_graph_search.json"
                config = json.loads((Path("configs") / name).read_text())
                config["task_file"] = "examples/research_smoke_tasks.json"
                path = Path(folder) / "config.json"
                path.write_text(json.dumps(config))
                output = io.StringIO()
                with patch.dict(os.environ, {"GRAPH_PLANNER_BACKEND": "codex", "PATH": os.environ["PATH"]}, clear=True), patch(
                    "sys.argv", ["conditioning", "--config", str(path), "--dry-run"]), patch(
                    "evovideo_skill.conditioning_verifier.resolve_profiles", return_value=None), redirect_stdout(output):
                    if tier == "research":
                        with self.assertRaisesRegex(ValueError, "research tier requires"):
                            main()
                    else:
                        main()
                        self.assertIn('"graph_planner_backend": "codex"', output.getvalue())
