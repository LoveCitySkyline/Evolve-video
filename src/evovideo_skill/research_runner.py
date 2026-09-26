"""Isolated research arms: fixed generator, budget caps, frozen held-out evaluation."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import fcntl
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

from evovideo_skill.benchmarks import BenchmarkSuite
from evovideo_skill.evolution_data import stratified_task_split
from evovideo_skill.graph_evolver import GraphToolPathEvolver, h3_run_should_stop
from evovideo_skill.graph_skill import GraphSkillMemory, ToolPathGraph
from evovideo_skill.graph_visualization import EvolutionGraphArchive
from evovideo_skill.research_protocol import (
    ARMS, BudgetLedger, ResearchBudgetExceeded, append_json, applicability,
    composition_template, decode_graph, feedback_view, generation_credits, graph_payload,
    validate_candidate, validate_splits, write_json,
)
from evovideo_skill.research_subgraphs import SubgraphLibrary, stable_hash
from evovideo_skill.runtime import (
    RuntimeSettings, build_evaluator_suite, build_graph_mutation_proposer,
    build_runtime, with_env_overrides,
)
from evovideo_skill.skill_memory import SkillMemory
from evovideo_skill.tools import ToolRegistry


class ResearchPlanner:
    """One planner request per search, no hidden repair retries or installation."""

    def __init__(self, proposer):
        self.proposer = proposer
        self.last_usage = None

    def propose(self, request: dict) -> dict:
        self.last_usage = None
        system = (
            "Optimize a frozen video agent. Return one JSON object with graph OR reuse_fragment_id and source_node. "
            "A graph has only nodes and edges, using the exact supplied node/edge schema. "
            "Do not modify the task, source assets, scoring rules, seeds, model, or verifier. "
            "Use generic stage instructions that transfer to unseen tasks in the same task class. "
            "Do not hard-code this task's scene, characters, file paths or target answers into a shared policy. "
            "Only registered tools may execute. No installation, internet research, or file inspection. "
            "For prompt/composition search, copy the parent graph exactly and change only video nodes' config.prompt. "
            "For graph search, change topology/config within the bounded edit budget; same tool may repeat. "
            "Use explicit artifact bindings and preserve all requested content and total duration. "
            "Unknown postconditions are not verified. Localized feedback is not causal proof. "
            "If using a retrieved fragment, return its ID and a terminal source_node in the parent graph."
        )
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(request, ensure_ascii=False)}]
        if hasattr(self.proposer, "codex"):
            schema = {"type": "object", "properties": {"proposal_json": {"type": "string"}},
                      "required": ["proposal_json"], "additionalProperties": False}
            envelope, _, _ = self.proposer.codex.run_json(
                "research-proposal", system + "\n" + messages[1]["content"]
                + "\nReturn the JSON-serialized proposal in proposal_json.", schema)
            return self.proposer._parse_json(envelope["proposal_json"])
        config = self.proposer.config
        payload = {"model": config.model, "messages": messages,
                   "max_completion_tokens": config.max_output_tokens}
        if config.reasoning_effort:
            payload["reasoning_effort"] = config.reasoning_effort
        response = self.proposer._post(payload)
        self.last_usage = response.get("usage")
        return self.proposer._parse_json(self.proposer._response_content(response))


class SmokePlanner:
    """Offline plumbing test. Scores from this run are not experimental evidence."""

    def propose(self, request: dict) -> dict:
        graph = deepcopy(request["parent"])
        for node in graph["nodes"]:
            if node["name"] == "mock_text_to_video":
                node["config"]["prompt"] = "Preserve requested identity and execute every action in order."
        return {"graph": graph}


class ResearchArmRunner:
    def __init__(self, arm: str, dataset, evolver: GraphToolPathEvolver, planner,
                 root: Path, protocol: dict, config: dict):
        self.arm = arm
        self.search, self.sharing, self.feedback_mode = ARMS[arm]
        self.dataset, self.evolver, self.planner = dataset, evolver, planner
        self.root, self.config = root, config
        self.root.mkdir(parents=True, exist_ok=True)
        self.protocol = protocol
        self.protocol_hash = stable_hash(protocol)
        self.library = SubgraphLibrary(evolver.tools)
        self.archive = EvolutionGraphArchive(root / "graph_visualization")
        self.state_path = root / "checkpoint.json"
        self.seeds = config["evaluation_seeds"]
        self.baseline = evolver.baseline_graph()
        self.parent = composition_template(self.baseline) if self.search == "composition" else self.baseline
        self.evolver.executor.validate_graph(self.parent)
        validation_keys = {applicability(t) for t in dataset.validation}
        self.search_tasks = [t for t in dataset.train if applicability(t) in validation_keys]
        if not self.search_tasks:
            raise ValueError("no train task has a disjoint validation task with matching applicability")
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            if self.state["protocol_hash"] != self.protocol_hash:
                raise ValueError("research protocol changed; use a new output directory")
        else:
            self.state = {"protocol_hash": self.protocol_hash, "iteration": 0, "searches": 0,
                          "policies": {}, "history": [], "pending": None, "pareto": {},
                          "task_parents": {}, "attempt_history": {},
                          "ledger": asdict(BudgetLedger(config["max_generation_calls"], config["max_generated_seconds"])),
                          "phase": "search"}
            self.save()
        self.ledger = BudgetLedger(**self.state["ledger"])
        for item in self.state["history"]:
            self.library.mine(ToolPathGraph.from_dict(item["graph"]), item["applicability"], item["observation"])

    def save(self) -> None:
        if hasattr(self, "ledger"):
            self.state["ledger"] = asdict(self.ledger)
        write_json(self.state_path, self.state)

    def evaluate(self, task, graph, seed: int, phase: str) -> dict:
        key = stable_hash([self.protocol_hash, asdict(task), graph_payload(graph), seed])
        output = self.root / "evaluations" / f"{key}.json"
        if output.exists():
            return json.loads(output.read_text())
        calls, seconds = generation_credits(task, graph)
        self.ledger.reserve(calls, seconds)
        # Reserve before dispatch. An interrupted request never gets free credits.
        self.save()
        sampled = deepcopy(task)
        sampled.metadata.update(generation_seed=seed, evaluation_seed=seed, replicate_label=seed)
        start = time.monotonic()
        print(f"[research] arm={self.arm} phase={phase} task={task.task_id} seed={seed} "
              f"graph={graph.graph_id} credits={self.ledger.reserved_calls}/{self.ledger.max_calls}", flush=True)
        try:
            rollout = self.evolver.rollout(sampled, graph)
            vlm = rollout.artifact.metadata.get("vlm_evaluation", {})
            if isinstance(vlm, dict) and str(vlm.get("evaluation_status", "")).startswith("failed"):
                raise RuntimeError("verifier unavailable; this video has no valid quality measurement")
            record = {"status": "ok", "score": rollout.score,
                      "feedback": feedback_view(rollout, "local"),
                      "video": rollout.artifact.metadata.get("local_video_path"),
                      "reward": rollout.reward.to_dict() if rollout.reward else None,
                      "node_evidence": rollout.artifact.metadata.get("node_evidence", {}),
                      "seed_controlled": rollout.artifact.metadata.get("generation_seed_applied"),
                      "state": rollout.state.to_dict()}
            if not isinstance(record["score"], (int, float)) or not 0 <= record["score"] <= 1:
                raise ValueError("nonfinite/out-of-range verifier score")
            if self.protocol.get("provider") == "local-h3" and record["seed_controlled"] is not True:
                raise RuntimeError("local H3 did not confirm the paired seed; stop this comparison")
        except Exception as exc:
            # Operational failures must not be scored as low-quality videos.
            append_json(self.root / "errors.jsonl", {"task_id": task.task_id, "phase": phase,
                        "graph": graph_payload(graph), "error": str(exc), "type": type(exc).__name__})
            raise
        record.update(task_id=task.task_id, seed=seed, graph_id=graph.graph_id, phase=phase,
                      reserved_calls=calls, reserved_generated_seconds=seconds,
                      wall_seconds=time.monotonic() - start)
        write_json(output, record)
        append_json(self.root / "executions.jsonl", record)
        self.archive.record_execution(task_id=task.task_id, graph_id=graph.graph_id,
            score=record["score"], passed=rollout.evaluation.passed, tool_chain=rollout.artifact.tool_chain,
            estimated_cost=seconds, cache_hit=False, failure_types=[],
            metric_scores={m.name: m.score for m in rollout.evaluation.active_metrics},
            evaluation_seed=seed, seed_controlled=record["seed_controlled"], artifact_path=record["video"])
        return record

    def paired(self, tasks, candidate, phase) -> tuple[float, list[dict]]:
        pairs = []
        for task in tasks:
            for seed in self.seeds:
                base = self.evaluate(task, self.baseline, seed, phase)
                child = self.evaluate(task, candidate, seed, phase)
                pairs.append({"task_id": task.task_id, "seed": seed, "base": base["score"],
                              "candidate": child["score"], "delta": child["score"] - base["score"],
                              "baseline_video": base["video"], "candidate_video": child["video"]})
        return statistics.mean(p["delta"] for p in pairs), pairs

    def _memory(self, key: str) -> dict:
        if self.sharing == "none":
            return {}
        if self.sharing == "whole":
            return {"whole_paths": [graph_payload(ToolPathGraph.from_dict(h["graph"]))
                                    for h in self.state["history"] if h["applicability"] == key][-8:]}
        fragments = self.library.retrieve(key)
        # No diagnostic strings or media paths enter the scalar-only planner.
        for fragment in fragments:
            fragment["observations"] = [{k: o[k] for k in ("task_id", "gain")} for o in fragment["observations"]]
        return {"fragments": fragments}

    def search_iteration(self, iteration: int) -> None:
        task = self.search_tasks[(iteration // self.config.get("searches_per_task", 3)) % len(self.search_tasks)]
        key = applicability(task)
        saved_parent = self.state["task_parents"].get(task.task_id)
        parent = ToolPathGraph.from_dict(saved_parent["graph"]) if saved_parent else self.parent
        validation = [t for t in self.dataset.validation if applicability(t) == key]
        if not validation:
            append_json(self.root / "iterations.jsonl", {"iteration": iteration, "task_id": task.task_id,
                        "status": "skipped_no_disjoint_matching_validation"})
            return
        baseline = [self.evaluate(task, parent, seed, "train") for seed in self.seeds]
        pending = self.state["pending"]
        if pending is None:
            if self.state["searches"] >= self.config["max_searches"]:
                return
            feedback = [r["feedback"] if self.feedback_mode == "local" else {"score": r["score"]} for r in baseline]
            metadata = {k: deepcopy(task.metadata[k]) for k in ("temporal_steps", "constraints", "evaluation",
                        "h3_shots", "h3_references", "h3_audio_criteria", "h3_global_constraints") if k in task.metadata}
            # Native reference descriptions can contain paths; never include prior rollout paths in scalar feedback.
            request = {"search": self.search, "feedback_mode": self.feedback_mode,
                       "task": {"prompt": task.prompt, "duration_seconds": task.duration_seconds, "metadata": metadata},
                       "parent": graph_payload(parent), "feedback": feedback,
                       "previous_attempts": self.state["attempt_history"].get(task.task_id, [])[-3:],
                       "memory": self._memory(key), "max_nodes": self.config["max_nodes"],
                       "max_edits": self.config["max_edits"],
                       "tools": [self.evolver.tools.spec(n).to_dict() for n in sorted(self.evolver.tools.available_names())]}
            if self.protocol.get("provider") == "local-h3":
                from evovideo_skill.llm_graph_mutation import OpenAICompatibleGraphMutationProposer
                request["h3_native_planner"] = OpenAICompatibleGraphMutationProposer._h3_native_planner(
                    self.evolver.tools.available_names(), local=True)
            self.state["searches"] += 1
            # Interrupted planner requests consume a search slot, and are not replayed for free.
            self.state["pending"] = {"iteration": iteration, "status": "planner_started"}
            self.save()
            write_json(self.root / "proposals" / f"{iteration:04d}_request.json", request)
            start = time.monotonic()
            try:
                raw = self.planner.propose(request)
                pending = {"iteration": iteration, "status": "proposed", "raw": raw}
            except Exception as exc:
                pending = {"iteration": iteration, "status": "planner_error", "error": str(exc)}
            append_json(self.root / "planner_costs.jsonl", {"iteration": iteration,
                        "wall_seconds": time.monotonic() - start, "status": pending["status"],
                        "usage": getattr(self.planner, "last_usage", None)})
            self.state["pending"] = pending
            self.save()
        if pending["status"] != "proposed":
            append_json(self.root / "iterations.jsonl", pending)
            return
        try:
            raw = pending["raw"]
            if "reuse_fragment_id" in raw:
                if self.sharing != "subgraph" or set(raw) != {"reuse_fragment_id", "source_node"}:
                    raise ValueError("fragment reuse is unavailable in this arm")
                graph = self.library.append(parent, raw["reuse_fragment_id"], raw["source_node"], key)
            else:
                if set(raw) != {"graph"}:
                    raise ValueError("proposal may contain only graph")
                graph = decode_graph(raw["graph"])
            validate_candidate(graph, parent, self.evolver.executor, self.search,
                               self.config["max_nodes"], self.config["max_edits"])
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            append_json(self.root / "iterations.jsonl", {"iteration": iteration, "status": "invalid",
                        "error": str(exc), "proposal": pending["raw"]})
            return
        write_json(self.root / "graphs" / f"{iteration:04d}.json", graph.to_dict())
        try:
            train_gain, train_pairs = self.paired([task], graph, "train")
            train_results = [self.evaluate(task, graph, seed, "train") for seed in self.seeds]
            train_score = statistics.mean(r["score"] for r in train_results)
            if train_score > statistics.mean(r["score"] for r in baseline):
                self.state["task_parents"][task.task_id] = {"graph": graph.to_dict(), "score": train_score}
            self.state["attempt_history"].setdefault(task.task_id, []).append({
                "graph": graph_payload(graph), "feedback": [
                    r["feedback"] if self.feedback_mode == "local" else {"score": r["score"]}
                    for r in train_results]})
            self.save()
            gain, pairs = self.paired(validation, graph, "validation")
        except ResearchBudgetExceeded:
            raise
        except Exception as exc:
            if h3_run_should_stop(exc):
                raise
            # An unrendered candidate is an execution failure, never a negative quality measurement.
            append_json(self.root / "iterations.jsonl", {"iteration": iteration, "status": "execution_error",
                        "error": str(exc), "graph_id": graph.graph_id})
            return
        incumbent = self.state["policies"].get(key)
        threshold = incumbent["gain"] if incumbent else 0.0
        accepted = gain >= self.config["min_gain"] and gain > threshold
        item = {"iteration": iteration, "task_id": task.task_id, "status": "accepted" if accepted else "rejected",
                "gain": gain, "train_gain": train_gain, "pairs": pairs, "train_pairs": train_pairs,
                "graph_id": graph.graph_id, "budget": asdict(self.ledger)}
        if accepted:
            self.state["policies"][key] = {"graph": graph.to_dict(), "gain": gain}
        if gain >= self.config["min_gain"] and train_gain >= self.config["min_gain"]:
            history = {"graph": graph.to_dict(), "applicability": key,
                       "observation": {"task_id": task.task_id, "gain": gain, "validation_pairs": pairs}}
            if history not in self.state["history"]:
                self.state["history"].append(history)
            self.library.mine(graph, key, history["observation"])
        cost = statistics.mean(generation_credits(t, graph)[1] for t in validation)
        frontier = self.state["pareto"].setdefault(key, [])
        point = {"graph_id": graph.graph_id, "validation_gain": gain, "generated_seconds_per_task": cost}
        points = [p for p in frontier if p["graph_id"] != graph.graph_id] + [point]
        self.state["pareto"][key] = [p for p in points if not any(
            q["validation_gain"] >= p["validation_gain"] and
            q["generated_seconds_per_task"] <= p["generated_seconds_per_task"] and
            (q["validation_gain"] > p["validation_gain"] or
             q["generated_seconds_per_task"] < p["generated_seconds_per_task"])
            for q in points)]
        self.save()
        append_json(self.root / "iterations.jsonl", item)
        self.archive.record_graph(graph, stage="research_validation", status=item["status"],
                                  iteration=iteration, metadata=item)
        self.archive.export(programs=[], feedback=[], weighted_categories={},
                            frontier=[p["graph"]["graph_id"] for p in self.state["policies"].values()])
        print(f"[research] arm={self.arm} iteration={iteration} {item['status']} validation_gain={gain:+.4f}", flush=True)

    def run(self) -> dict:
        if self.state["phase"] == "complete":
            return json.loads((self.root / "summary.json").read_text())
        try:
            for iteration in range(self.state["iteration"], self.config["max_searches"]):
                self.search_iteration(iteration)
                self.state.update(iteration=iteration + 1, pending=None)
                self.save()
            self.state["phase"] = "frozen_test"
            self.save()
            write_json(self.root / "frozen_policies.json", self.state["policies"])
            write_json(self.root / "subgraphs.json", [f.to_dict() for f in self.library.fragments.values()])
            write_json(self.root / "pareto.json", self.state["pareto"])
            pairs = []
            routed = 0
            for task in self.dataset.test:
                policy = self.state["policies"].get(applicability(task))
                routed += policy is not None
                graph = ToolPathGraph.from_dict(policy["graph"]) if policy else self.baseline
                _, task_pairs = self.paired([task], graph, "test")
                pairs.extend(task_pairs)
            by_task = {}
            for pair in pairs:
                by_task.setdefault(pair["task_id"], []).append(pair["delta"])
            gains = [statistics.mean(x) for x in by_task.values()]
            rng = random.Random(0)
            boot = sorted(statistics.mean(rng.choices(gains, k=len(gains))) for _ in range(2000))
            summary = {"status": "complete", "arm": self.arm, "heldout_task_count": len(by_task),
                       "heldout_gain": statistics.mean(gains), "task_bootstrap_95ci": [boot[49], boot[1949]],
                       "routed_tasks": routed, "searches": self.state["searches"], "pairs": pairs,
                       "eligible_train_tasks": [t.task_id for t in self.search_tasks],
                       "total_train_tasks": len(self.dataset.train),
                       "budget": asdict(self.ledger), "protocol": self.protocol,
                       "limitations": ["Equal budget caps, not necessarily equal realized compute.",
                           "Reservations include cached and failed/interrupted attempts; they are not GPU seconds.",
                           "No human calibration or causal node-credit claim; confidence intervals cluster by task."]}
            write_json(self.root / "summary.json", summary)
            self.archive.export(programs=[], feedback=[], weighted_categories={},
                                frontier=[p["graph"]["graph_id"] for p in self.state["policies"].values()])
            self.state["phase"] = "complete"
            self.save()
            return summary
        except ResearchBudgetExceeded as exc:
            summary = {"status": "budget_censored", "arm": self.arm, "reason": str(exc),
                       "budget": asdict(self.ledger), "heldout_gain": None}
            write_json(self.root / "summary.json", summary)
            return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/h3_research_questions.json")
    parser.add_argument("--arm", choices=list(ARMS) + ["all"], default="all")
    parser.add_argument("--output-dir")
    parser.add_argument("--continue", dest="resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    config.setdefault("max_nodes", 16)
    config.setdefault("max_edits", 24)
    config.setdefault("min_gain", 0.02)
    seeds = config["evaluation_seeds"]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError("research requires at least three distinct integer seeds")
    if any(config[k] <= 0 for k in ("max_searches", "max_generation_calls", "max_generated_seconds")):
        raise ValueError("all research budgets must be positive")
    source = Path(config["task_file"])
    tasks = BenchmarkSuite.from_file(source).tasks
    dataset = stratified_task_split(tasks)
    validate_splits(dataset)
    runtime = with_env_overrides(RuntimeSettings(**config["runtime"]))
    if args.smoke:
        runtime = replace(runtime, provider="local-fake", enable_vlm_eval=False, enable_llm_mutation=False)
    elif runtime.provider != "local-h3":
        raise ValueError("this controlled protocol supports local-h3; use --smoke for offline plumbing")
    if not args.smoke:
        if os.environ.get("VIDEO_OUTPUT_DIR") or os.environ.get("AGENT_STATE_DIR"):
            raise ValueError("unset VIDEO_OUTPUT_DIR and AGENT_STATE_DIR: research arms need isolated directories")
        if not runtime.enable_vlm_eval or not runtime.enable_llm_mutation:
            raise ValueError("real research needs VLM evaluation and LLM planning enabled")
        if runtime.restore_catalog_tools or runtime.enable_open_world_tools or runtime.enable_mcp_tools:
            raise ValueError("research freezes the tool set: unset RESTORE_CATALOG_TOOLS, ENABLE_OPEN_WORLD_TOOLS and ENABLE_MCP_TOOLS")
        if runtime.h3_local_model_revision == "unspecified":
            raise ValueError("pin H3_LOCAL_MODEL_REVISION to the actual deployed checkpoint")
        if any(t.metadata.get("h3_audio_criteria") for t in tasks) and not runtime.h3_audio_verifier_command:
            runtime = replace(runtime, h3_audio_verifier_command=json.dumps(
                [sys.executable, "-m", "evovideo_skill.h3_omni_verifier"]))
        from evovideo_skill.harness import HarnessConfig
        from evovideo_skill.h3_cli import preflight
        preflight(HarnessConfig(name="research", task_files=[str(source)], runtime=runtime,
                               evaluation_seeds=seeds),
                  require_credentials=not args.dry_run, check_server=False)
    root = Path(args.output_dir or config["output_dir"]).resolve()
    arms = list(ARMS) if args.arm == "all" else [args.arm]
    description = {"arms": arms, "split": dataset.to_dict(), "budget_per_arm": {
        k: config[k] for k in ("max_searches", "max_generation_calls", "max_generated_seconds")},
        "note": "Generation credits include train, validation and held-out evaluation. No GPU/API calls in dry-run."}
    eligible = [t.task_id for t in dataset.train
                if applicability(t) in {applicability(v) for v in dataset.validation}]
    description["eligible_train_tasks"] = eligible
    description["excluded_train_tasks"] = [t.task_id for t in dataset.train if t.task_id not in eligible]
    print(f"[research] independent-validation coverage: {len(eligible)}/{len(dataset.train)} training tasks", flush=True)
    if args.dry_run:
        print(json.dumps(description, indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for arm in arms:
        arm_root = root / arm
        arm_root.mkdir(exist_ok=True)
        with (arm_root / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (arm_root / "checkpoint.json").exists() and not args.resume:
                raise ValueError(f"{arm} already exists; use --continue or a new output directory")
            settings = replace(runtime, video_output_dir=str(arm_root / "videos"),
                               agent_state_dir=str(arm_root / "agent_state"),
                               enable_open_world_tools=False, enable_mcp_tools=False, restore_catalog_tools=False)
            tools, augmenter = (ToolRegistry.with_mock_tools(), None) if args.smoke else build_runtime(settings)
            proposer = None if args.smoke else build_graph_mutation_proposer(settings)
            # Pin actual LLM resolution, not a config label overridden by the environment.
            llm = {k: v for k, v in asdict(proposer.config).items() if k != "api_key"} if proposer else {}
            codex = getattr(proposer, "codex", None)
            if codex:
                llm["codex"] = {"model": codex.config.model, "reasoning_effort": codex.config.reasoning_effort}
                print("[research] WARNING: Codex has filesystem access; feedback isolation is prompt-only. "
                      "Use the API planner for the controlled RQ3 comparison.", flush=True)
            protocol = {"version": "graph-research-v1", "arm": arm, "provider": settings.provider,
                        "task_hash": stable_hash([asdict(t) for t in tasks]),
                        "settings": asdict(settings), "planner": llm,
                        "experiment": {k: v for k, v in config.items() if k not in {"runtime", "output_dir"}},
                        "tool_manifest": [tools.spec(n).to_dict() for n in sorted(tools.available_names())],
                        "evidence_type": "synthetic_smoke" if args.smoke else "real_video",
                        "feedback_isolation": "prompt_only_not_enforced" if codex else "api_payload_only"}
            protocol["code_hash"] = stable_hash({p.name: stable_hash(p.read_bytes().hex())
                for p in Path(__file__).parent.glob("*.py")})
            protocol["verifier_environment"] = {k: v for k, v in os.environ.items()
                if k.startswith(("H3_OMNI_", "VLM_", "EVOVIDEO_VLM_")) and "KEY" not in k}
            write_json(arm_root / "protocol_requested.json", protocol)
            evolver = GraphToolPathEvolver(SkillMemory(arm_root / "skill_memory"),
                        GraphSkillMemory(arm_root / "graph_memory"), tools=tools,
                        evaluators=build_evaluator_suite(settings), vlm_augmenter=augmenter,
                        template_mutations_enabled=False)
            runner = ResearchArmRunner(arm, dataset, evolver,
                        SmokePlanner() if args.smoke else ResearchPlanner(proposer), arm_root, protocol, config)
            summaries.append(runner.run())
    write_json(root / ("comparison_" + args.arm + ".json"), summaries)
    for summary in summaries:
        print(f"{summary['arm']}: {summary['status']} heldout_gain={summary.get('heldout_gain')}")


if __name__ == "__main__":
    main()
