"""Learn conditioning strategies; freeze them before direct/adaptive held-out use."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import time

from evovideo_skill.benchmarks import BenchmarkSuite
from evovideo_skill.conditioning_cache import ConditioningNodeCache, material_hashes
from evovideo_skill.conditioning_memory import StrategyMemory, paired_effect, task_payload, task_state
from evovideo_skill.conditioning_planner import ConditioningPlanner, ConditioningSmokePlanner, decode_proposal
from evovideo_skill.conditioning_interactions import (connection_manifest, decode_experiment,
    effect_supported, interaction_effect)
from evovideo_skill.conditioning_workspace import (VERSION as WORKSPACE_VERSION,
    ConditioningWorkspace, planner_project, repair_impact)
from evovideo_skill.conditioning_repair import repair_frontier, preservation_report
from evovideo_skill.conditioning_search import (SignedInteractionGraph, context_view,
    decode_pool, search_options, select_pair)
from evovideo_skill.evolution_data import stratified_task_split
from evovideo_skill.graph_evolver import GraphToolPathEvolver, h3_run_should_stop
from evovideo_skill.graph_skill import GraphSkillMemory, ToolPathGraph
from evovideo_skill.graph_visualization import EvolutionGraphArchive
from evovideo_skill.models import VideoArtifact
from evovideo_skill.planning import Planner
from evovideo_skill.research_protocol import (BudgetLedger, ResearchBudgetExceeded, append_json, decode_graph,
    feedback_view, graph_payload, validate_splits, write_json)
from evovideo_skill.research_subgraphs import stable_hash
from evovideo_skill.runtime import (RuntimeSettings, build_evaluator_suite, build_graph_mutation_proposer,
    build_runtime, with_env_overrides)
from evovideo_skill.skill_memory import SkillMemory
from evovideo_skill.tools import ToolRegistry


class FixedTaskPlanner(Planner):
    def plan(self, task, selected_skill_names):
        # Graph labels never silently activate legacy skill-conditioned rewriting.
        return super().plan(task, [])


class EpisodeBudgetExceeded(ResearchBudgetExceeded):
    pass


class MeasurementUnavailable(RuntimeError):
    pass


def balanced_training_order(tasks):
    families = {}
    for task in tasks:
        families.setdefault(task_state(task)["family"], []).append(task)
    return [group[index] for index in range(max(map(len, families.values())))
            for group in families.values() if index < len(group)]


def scenario_id(task):
    return str(task.metadata.get("scenario_group") or task.metadata.get("scenario_id") or task.task_id)


class ConditioningRunner:
    def __init__(self, dataset, evolver, planner, root, config, signature, final_augmenter=None):
        self.dataset, self.evolver, self.planner = dataset, evolver, planner
        self.root, self.config, self.signature = Path(root), config, signature
        self.root.mkdir(parents=True, exist_ok=True)
        self.final_augmenter = final_augmenter or evolver.vlm_augmenter
        self.baseline = evolver.baseline_graph()
        self.archive = EvolutionGraphArchive(self.root / "graph_visualization")
        self.seeds = config["evaluation_seeds"]
        self.protocol = {"version": "conditioning-strategies-v1", "signature": signature,
            "workspace_protocol": WORKSPACE_VERSION,
            "config": {k: v for k, v in config.items() if k not in {"output_dir", "runtime"}},
            "dataset_hash": stable_hash([asdict(t) for group in
                (dataset.train, dataset.validation, dataset.test) for t in group]),
            "asset_hashes": material_hashes([asdict(t) for group in
                (dataset.train, dataset.validation, dataset.test) for t in group])}
        self.protocol_hash = stable_hash(self.protocol)
        self.checkpoint = self.root / "checkpoint.json"
        if self.checkpoint.exists():
            self.state = json.loads(self.checkpoint.read_text())
            if self.state["protocol_hash"] != self.protocol_hash:
                raise ValueError("conditioning protocol/assets changed; use a new output directory")
        else:
            self.state = {"protocol_hash": self.protocol_hash, "learned": False, "validated": False,
                "entries": {}, "admitted_ids": [], "episodes": {}, "planner_requests": 0,
                "ledger": asdict(BudgetLedger(config["max_generation_calls"], config["max_generated_seconds"]))}
        self.ledger = BudgetLedger(**self.state["ledger"])
        self.memory = StrategyMemory(self.state["entries"])
        self.search_options = search_options(config)
        self.signed_graph = SignedInteractionGraph(self.state.get("signed_interaction_graph"))
        self.active_selection = None
        write_json(self.root / "protocol.json", self.protocol)
        self.save()

    def save(self):
        self.state["ledger"] = asdict(self.ledger)
        write_json(self.checkpoint, self.state)

    def charge(self, calls, seconds, hit, episode):
        if hit:
            return
        usage = self.state["episodes"].setdefault(episode, {"calls": 0, "seconds": 0})
        if episode.startswith("test/") and (
                usage["calls"] + calls > self.config["test_max_generation_calls"] or
                usage["seconds"] + seconds > self.config["test_max_generated_seconds"]):
            raise EpisodeBudgetExceeded("test episode generation budget exhausted")
        self.ledger.reserve(calls, seconds)
        usage["calls"] += calls
        usage["seconds"] += seconds
        self.save()  # Reserve before dispatch, including failed/uncertain submissions.

    def evaluate(self, task, graph, seed, episode):
        sampled = deepcopy(task)
        sampled.metadata.update(generation_seed=seed, evaluation_seed=seed, replicate_label=seed)
        ident = stable_hash([self.protocol_hash, asdict(sampled), graph_payload(graph), episode])
        path = self.root / "evaluations" / (ident + ".json")
        if path.exists():
            record = json.loads(path.read_text())
            if record["status"] != "ok":
                raise RuntimeError(record["error"])
            if record["files"] != material_hashes(list(record["files"])):
                raise RuntimeError("saved evaluation media changed or disappeared; do not reuse this experiment")
            return record
        cache = ConditioningNodeCache(self.root / "node_cache" / stable_hash([episode, task.task_id, seed]),
            self.signature, lambda c, s, hit: self.charge(c, s, hit, episode))
        started = time.monotonic()
        workspace = ConditioningWorkspace(self.root / "project_states", ident, sampled, graph, seed, episode)
        print(f"[conditioning] generate episode={episode} task={task.task_id} seed={seed} graph={graph.graph_id}", flush=True)
        print(f"[conditioning] project_state={workspace.path}", flush=True)
        try:
            rollout = self.evolver.rollout(sampled, graph, node_cache=cache, observer=workspace)
            self.check_measurement(rollout.artifact, rollout.score)
            if self.signature.get("provider") == "local-h3" and rollout.artifact.metadata.get("generation_seed_applied") is not True:
                raise RuntimeError("local H3 did not confirm the fixed evaluation seed")
            record = {"status": "ok", "evaluation_id": ident, "task_id": task.task_id, "seed": seed,
                "episode": episode, "graph_id": graph.graph_id, "score": rollout.score,
                "feedback": feedback_view(rollout, "local"), "reward": rollout.reward.to_dict(),
                "artifact": asdict(rollout.artifact), "files": material_hashes(asdict(rollout.artifact)),
                "criterion_scores": deepcopy(rollout.artifact.metadata.get("vlm_evaluation", {}).get("criterion_scores", {})),
                "verification": deepcopy(rollout.artifact.metadata.get("vlm_evaluation", {}).get("verification_metadata", {})),
                "video": rollout.artifact.metadata.get("local_video_path"),
                "project_state_path": str(workspace.path),
                "process_diagnostics": workspace.diagnostics(),
                "reused_nodes": cache.hits, "executed_nodes": cache.misses,
                "wall_seconds": time.monotonic() - started}
            workspace.verified(record)
            write_json(path, record)
            append_json(self.root / "executions.jsonl", {k: v for k, v in record.items() if k != "artifact"})
            self.archive.record_execution(task_id=task.task_id, graph_id=graph.graph_id, score=rollout.score,
                passed=rollout.evaluation.passed, tool_chain=rollout.artifact.tool_chain,
                estimated_cost=self.state["episodes"].get(episode, {}).get("seconds", 0),
                cache_hit=bool(cache.hits), failure_types=[], artifact_path=record["video"],
                metric_scores={m.name: m.score for m in rollout.evaluation.active_metrics})
            return record
        except ResearchBudgetExceeded as exc:
            workspace.failed(exc, "budget_exhausted")
            raise
        except Exception as exc:
            workspace.failed(exc, "needs_review" if isinstance(exc, MeasurementUnavailable) else "failed")
            append_json(self.root / "errors.jsonl", {"episode": episode, "task_id": task.task_id,
                "graph_id": graph.graph_id, "error": str(exc), "type": type(exc).__name__,
                "project_state_path": str(workspace.path)})
            # Uncertain provider jobs are resumable, never converted to quality failures.
            if not h3_run_should_stop(exc) and not isinstance(exc, MeasurementUnavailable):
                write_json(path, {"status": "execution_error", "error": str(exc)})
            raise
        except KeyboardInterrupt as exc:
            workspace.failed(exc, "interrupted")
            raise

    @staticmethod
    def check_measurement(artifact, score):
        vlm = artifact.metadata.get("vlm_evaluation", {})
        if str(vlm.get("evaluation_status", "")).startswith("failed"):
            raise MeasurementUnavailable("verifier unavailable; no valid quality observation")
        if vlm.get("evaluation_status") == "needs_review":
            details = vlm.get("verification_metadata", {})
            raise MeasurementUnavailable("verifier evidence is incomplete or disputed; inspect " + str(details.get("judgment_path", "verifier logs")))
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("nonfinite/out-of-range quality score")
        if artifact.metadata.get("task_reward", {}).get("missing_metrics"):
            raise MeasurementUnavailable("required rubric metrics are unobserved; no valid complete quality measurement")

    def propose(self, task, parent, records, memory, operation, required_id=None, experiment=False):
        self.active_selection = None
        project = planner_project(task, parent, records, operation)
        active = experiment and self.search_options["enabled"]
        path = self.root / "proposals" / (stable_hash(operation) + ".json")
        if path.exists():
            envelope = json.loads(path.read_text())
        else:
            request = {"operation": operation, "task": task_payload(task),
                "condition_only": self.config.get("search_mode", "legacy") != "legacy",
                "state": task_state(task, [r["feedback"] for r in records]),
                "parent": graph_payload(parent), "memory": memory,
                "project_state": project,
                "required_strategy_id": required_id, "max_nodes": self.config["max_nodes"],
                "max_edits": self.config["max_edits"],
                "tools": [self.evolver.tools.spec(n).to_dict() for n in sorted(self.evolver.tools.available_names())]}
            if experiment:
                request.update(experiment="factorial", search_mode=self.config["search_mode"],
                               parent_connections=connection_manifest(parent))
            if active:
                frontier = (repair_frontier(task, parent, records, self.search_options)
                    if self.search_options["local_repair"] else
                    {"status": "unlocalized", "reason": "local repair ablation disabled", "boundaries": []})
                request.update(experiment="active_factorial", factor_pool_size=self.search_options["pool_size"],
                    repair_frontier=frontier,
                    signed_interaction_graph=self.signed_graph.view(self.search_options, context_view(task, records)))
            elif self.search_options["enabled"] and self.search_options["local_repair"] and records:
                request["repair_frontier"] = repair_frontier(task, parent, records, self.search_options)
            if self.signature.get("provider") == "local-h3":
                from evovideo_skill.llm_graph_mutation import OpenAICompatibleGraphMutationProposer
                request["h3_native_planner"] = OpenAICompatibleGraphMutationProposer._h3_native_planner(
                    self.evolver.tools.available_names(), local=True)
            envelope = {"status": "interrupted_planner", "operation": operation, "request": request}
            write_json(path, envelope)
            self.state["planner_requests"] += 1
            self.save()
            started = time.monotonic()
            try:
                envelope.update(status="proposed", raw=self.planner.propose(request))
            except Exception as exc:
                envelope.update(status="planner_error", error=str(exc))
            envelope.update(wall_seconds=time.monotonic() - started, usage=getattr(self.planner, "last_usage", None))
            write_json(path, envelope)
        if envelope["status"] != "proposed":
            append_json(self.root / "candidate_audits.jsonl", {"operation": operation, **envelope})
            return None
        try:
            if active:
                # The chosen pool/pair is committed before generation and reused on
                # resume, even after part of its factorial has already executed.
                decision = envelope.get("active_selection")
                if decision is None:
                    frontier = envelope["request"]["repair_frontier"]
                    anchor, factors, errors = decode_pool(envelope["raw"], task, parent,
                        self.evolver.executor, self.config, frontier)
                    chosen, audit = select_pair(task, parent, anchor, factors, records, self.signed_graph,
                        self.evolver.executor, self.config, frontier,
                        {"calls": self.ledger.max_calls - self.ledger.reserved_calls,
                         "seconds": self.ledger.max_seconds - self.ledger.reserved_seconds})
                    decision = {"audit": audit, "invalid_factors": errors, "frontier": frontier}
                    if chosen is not None:
                        graphs, strategies, descriptors = chosen
                        decision.update(graphs={k: graph_payload(g) for k, g in graphs.items()},
                                        strategies=strategies, descriptors=descriptors)
                    envelope["active_selection"] = decision
                    write_json(path, envelope)
                self.active_selection = decision
                write_json(self.root / "search_decisions" / (stable_hash(operation) + ".json"), decision)
                if "graphs" not in decision:
                    append_json(self.root / "candidate_audits.jsonl", {"operation": operation,
                        "status": "no_feasible_pair", **decision})
                    return None
                return {k: decode_graph(g) for k, g in decision["graphs"].items()}, decision["strategies"]
            if experiment:
                graphs, strategies = decode_experiment(envelope["raw"], task, parent, self.evolver.executor, self.config)
                envelope["repair_impact"] = {label: repair_impact(parent if label == "anchor" else graphs["anchor"], graph)
                                             for label, graph in graphs.items()}
                write_json(path, envelope)
                return graphs, strategies
            graph, strategy = decode_proposal(envelope["raw"], task, parent, self.evolver.executor,
                                             self.config, envelope["request"]["memory"], required_id)
            if "repair_frontier" in envelope["request"]:
                from evovideo_skill.conditioning_repair import check_local_candidate
                envelope["local_repair"] = check_local_candidate(parent, graph, envelope["request"]["repair_frontier"])
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            append_json(self.root / "candidate_audits.jsonl", {"operation": operation,
                "status": "invalid", "error": str(exc), "raw": envelope["raw"]})
            return None
        envelope["repair_impact"] = repair_impact(parent, graph)
        write_json(path, envelope)
        write_json(self.root / "graphs" / (graph.graph_id + ".json"), graph.to_dict())
        self.archive.record_graph(graph, stage=operation.split("/")[0], status="proposed",
                                  iteration=self.state["planner_requests"], metadata={"operation": operation})
        return graph, strategy

    def learn(self):
        if self.state["learned"]:
            return
        if self.config.get("search_mode", "legacy") != "legacy":
            return self.learn_interactions()
        parents = {}
        order = balanced_training_order(self.dataset.train)
        visited = []
        for index in range(self.config["max_searches"]):
            task = order[(index // self.config["searches_per_task"]) % len(order)]
            visited.append(task.task_id)
            parent = parents.get(task.task_id, self.baseline)
            episode = "train/" + task.task_id
            before = [self.evaluate(task, parent, seed, episode) for seed in self.seeds]
            memory = self.memory.retrieve(task, self.config["retrieval_limit"])
            proposal = self.propose(task, parent, before, memory, f"train/{index}")
            if proposal is None:
                continue
            graph, strategy = proposal
            if graph_payload(graph) == graph_payload(parent):
                continue
            try:
                after = [self.evaluate(task, graph, seed, episode) for seed in self.seeds]
            except ResearchBudgetExceeded:
                raise
            except Exception as exc:
                if h3_run_should_stop(exc) or isinstance(exc, MeasurementUnavailable):
                    raise
                continue
            effect = paired_effect(before, after)
            key = self.memory.observe(task, strategy, graph, effect, f"train/{index}", before_graph=parent)
            if effect["gain"] > self.config["selection_min_gain"]:
                parents[task.task_id] = graph
            self.state["entries"] = self.memory.snapshot()
            self.save()
            write_json(self.root / "interventions" / f"{index:04d}.json", {"strategy_id": key,
                "task_id": task.task_id, "before_graph": graph_payload(parent), "after_graph": graph_payload(graph),
                "effect": effect, "state": task_state(task)})
            self.archive.record_graph(graph, stage="conditioning_learning", status="observed", iteration=index,
                                      metadata={"strategy_id": key, "gain": effect["gain"]})
            self.archive.export(programs=[], feedback=[], weighted_categories={}, frontier=[])
            print(f"[conditioning] learned experiment={index} strategy={key} gain={effect['gain']:+.4f}", flush=True)
        self.state["learned"] = True
        self.save()
        write_json(self.root / "learning_summary.json", {"train_task_count": len(self.dataset.train),
            "visited_train_tasks": sorted(set(visited)), "max_searches": self.config["max_searches"],
            "strategy_count": len(self.memory.entries), "budget": asdict(self.ledger),
            "note": "The fixed budget may cover only a family-balanced subset of training tasks."})

    def learn_interactions(self):
        cursor = self.state.get("interaction_cursor", {"next_index": 0, "parents": {}, "visited": []})
        parents = {key: decode_graph(value) for key, value in cursor["parents"].items()}
        visited = list(cursor["visited"])
        order = balanced_training_order(self.dataset.train)
        for index in range(cursor["next_index"], self.config["max_searches"]):
            task = order[(index // self.config["searches_per_task"]) % len(order)]
            visited.append(task.task_id)
            parent = parents.get(task.task_id, self.baseline)
            episode = "train/" + task.task_id
            before = [self.evaluate(task, parent, s, episode) for s in self.seeds]
            proposal = self.propose(task, parent, before, self.memory.retrieve(task, self.config["retrieval_limit"]),
                                    f"factorial/{index}", experiment=True)
            if proposal is None:
                self.commit_interaction_cursor(index, parents, visited)
                continue
            graphs, strategies = proposal
            labels = ("anchor", "a", "b", "joint") if self.config["search_mode"] == "factorial" else ("anchor", "a", "b")
            cells, errors = {}, {}
            for label in labels:
                graph = graphs[label]
                write_json(self.root / "graphs" / (graph.graph_id + ".json"), graph.to_dict())
                self.archive.record_graph(graph, stage="factorial", status="proposed", iteration=index,
                    metadata={"cell": label, "connections": connection_manifest(graph)})
                try:
                    cells[label] = [self.evaluate(task, graph, s, episode) for s in self.seeds]
                except ResearchBudgetExceeded:
                    raise
                except Exception as exc:
                    if h3_run_should_stop(exc) or isinstance(exc, MeasurementUnavailable):
                        raise
                    errors[label] = str(exc)
            interaction = interaction_effect(cells) if set(cells) == {"anchor", "a", "b", "joint"} else None
            if interaction and self.active_selection:
                self.signed_graph.observe(task, self.active_selection["descriptors"], interaction, f"factorial/{index}")
            if "anchor" in cells:
                for label in ("a", "b", "joint"):
                    if label not in cells:
                        continue
                    effect = paired_effect(cells["anchor"], cells[label])
                    if label == "joint" and interaction:
                        effect["interaction"] = {"quality": interaction["quality"], "metrics": interaction["metrics"],
                            "factors": {k: strategies[k] for k in ("a", "b")},
                            "attribution": interaction["attribution"]}
                    self.memory.observe(task, strategies[label], graphs[label], effect, f"factorial/{index}/{label}",
                                        before_graph=graphs["anchor"])
            comparisons = {k: paired_effect(before, records) for k, records in cells.items()}
            preservation = {k: preservation_report(before, records,
                self.search_options["preservation_threshold"], self.search_options["preservation_tolerance"])
                for k, records in cells.items()} if self.search_options["enabled"] and self.search_options["local_repair"] else {}
            eligible = [k for k, effect in comparisons.items() if effect_supported(effect, self.config)
                        and preservation.get(k, {"passed": True})["passed"]]
            winner = max(eligible, key=lambda k: comparisons[k]["gain"]) if eligible else "parent"
            if winner != "parent":
                parents[task.task_id] = graphs[winner]
            for label in labels:
                self.archive.record_graph(graphs[label], stage="factorial", iteration=index,
                    status="selected" if label == winner else "execution_error" if label in errors else "observed",
                    metadata={"cell": label, "gain_to_parent": comparisons.get(label, {}).get("gain"),
                              "connections": connection_manifest(graphs[label])})
            report = {"iteration": index, "task_id": task.task_id, "search_mode": self.config["search_mode"],
                "parent_graph": parent.graph_id, "selected_cell": winner,
                "status": "complete" if not errors else "partial_execution_failure",
                "interaction": interaction, "comparisons_to_parent": comparisons, "execution_errors": errors,
                "preservation": preservation, "active_search": deepcopy(self.active_selection),
                "selected_graph": graph_payload(parents.get(task.task_id, parent)),
                "connections": {k: connection_manifest(graphs[k]) for k in labels},
                "cells": {k: [{"evaluation_id": r["evaluation_id"], "seed": r["seed"], "score": r["score"],
                              "video": r["video"]} for r in records] for k, records in cells.items()}}
            write_json(self.root / "interactions" / f"{index:04d}.json", report)
            self.state["entries"] = self.memory.snapshot()
            self.commit_interaction_cursor(index, parents, visited)
            self.archive.export(programs=[], feedback=[], weighted_categories={}, frontier=[])
            observed = f"{interaction['quality']['mean']:+.4f}" if interaction else "unmeasured"
            print(f"[conditioning] experiment={index} interaction={observed} selected={winner} errors={len(errors)}", flush=True)
        self.state["learned"] = True
        self.save()
        write_json(self.root / "learning_summary.json", {"search_mode": self.config["search_mode"],
            "train_task_count": len(self.dataset.train), "visited_train_tasks": sorted(set(visited)),
            "strategy_count": len(self.memory.entries), "budget": asdict(self.ledger),
            "active_graph_search": self.search_options,
            "signed_graph_nodes": len(self.signed_graph.data["nodes"]),
            "signed_graph_edges": len(self.signed_graph.data["edges"]),
            "note": "Local paired-seed graph interventions. No real improvement is guaranteed."})

    def commit_interaction_cursor(self, index, parents, visited):
        self.state["interaction_cursor"] = {"next_index": index + 1,
            "parents": {k: graph_payload(g) for k, g in parents.items()}, "visited": list(visited)}
        self.state["signed_interaction_graph"] = deepcopy(self.signed_graph.data)
        self.save()
        if self.search_options["enabled"]:
            write_json(self.root / "signed_interaction_graph.json", self.signed_graph.data)
            write_json(self.root / "signed_interaction_summary.json", self.signed_graph.view(self.search_options))

    def validate_and_freeze(self):
        if not self.state["learned"]:
            raise ValueError("learn before validation")
        frozen = self.root / "frozen_strategies.json"
        if self.state["validated"]:
            return frozen
        candidates = sorted(self.memory.entries.values(), key=lambda e: (
            -StrategyMemory.view(e)["evidence"]["mean_train_gain"], e["strategy_id"]))
        admitted = []
        reports = []
        for entry in candidates[:self.config["max_validation_strategies"]]:
            key = entry["strategy_id"]
            tasks = [t for t in self.dataset.validation if StrategyMemory.matches(entry, task_state(t))]
            tasks = tasks[:self.config["validation_tasks_per_strategy"]]
            effects, failures, preservation = [], [], []
            for task in tasks:
                # Validation observations are never fed back to the training memory/planner.
                proposal = self.propose(task, self.baseline, [], [StrategyMemory.view(entry)],
                                        f"validation/{key}/{task.task_id}", required_id=key)
                if proposal is None:
                    failures.append(task.task_id)
                    continue
                try:
                    before = [self.evaluate(task, self.baseline, s, "validation/" + task.task_id) for s in self.seeds]
                    after = [self.evaluate(task, proposal[0], s, "validation/" + task.task_id) for s in self.seeds]
                    effects.append(paired_effect(before, after))
                    if self.search_options["enabled"] and self.search_options["local_repair"]:
                        preservation.append(preservation_report(before, after,
                            self.search_options["preservation_threshold"], self.search_options["preservation_tolerance"]))
                except ResearchBudgetExceeded:
                    raise
                except Exception as exc:
                    if h3_run_should_stop(exc) or isinstance(exc, MeasurementUnavailable):
                        raise
                    failures.append(task.task_id)
            gain = statistics.mean(e["gain"] for e in effects) if effects else None
            accepted = (len(effects) >= self.config["min_validation_tasks"] and not failures
                        and gain >= self.config["min_gain"]
                        and all(v >= -self.config["max_metric_regression"]
                                for e in effects for v in e["metric_deltas"].values()))
            if self.config.get("search_mode", "legacy") != "legacy":
                accepted = accepted and all(effect_supported(e, self.config) for e in effects)
            accepted = accepted and all(r["passed"] for r in preservation)
            if accepted:
                admitted.append(key)
            reports.append({"strategy_id": key, "accepted": accepted, "gain": gain,
                "tasks": [t.task_id for t in tasks], "execution_failures": failures, "effects": effects,
                "preservation": preservation})
        write_json(self.root / "validation_reports.json", reports)
        data = {"version": "conditioning-strategies-v1", "signature": self.signature,
            "entries": self.memory.snapshot(), "admitted_ids": admitted,
            "signed_interaction_graph": deepcopy(self.signed_graph.data),
            "source_task_ids": [t.task_id for t in self.dataset.train + self.dataset.validation],
            "source_scenarios": [scenario_id(t) for t in self.dataset.train + self.dataset.validation],
            "learning_protocol": self.protocol_hash,
            "note": "Effects are training observations; admission is held-out validation selection, not causal proof."}
        data["content_hash"] = stable_hash(data)
        write_json(frozen, data)
        self.state.update(validated=True, admitted_ids=admitted)
        self.save()
        return frozen

    def load_frozen(self, path):
        data = json.loads(Path(path).read_text())
        content_hash = data.pop("content_hash")
        if stable_hash(data) != content_hash:
            raise ValueError("frozen strategy checksum mismatch")
        if data["signature"] != self.signature:
            raise ValueError("frozen strategy generator/tools/verifier/planner signature mismatch")
        if set(data["source_task_ids"]) & {t.task_id for t in self.dataset.test} or set(data["source_scenarios"]) & {
                scenario_id(t) for t in self.dataset.test}:
            raise ValueError("training/validation data overlaps held-out test")
        return StrategyMemory(data["entries"], frozen=True), set(data["admitted_ids"]), content_hash

    def test_memory(self, memory, admitted, task, arm):
        if arm == "none":
            return []
        warnings = {k for k, e in memory.entries.items() if StrategyMemory.view(e)["evidence"]["mean_train_gain"] <= 0}
        values = memory.retrieve(task, self.config["retrieval_limit"], admitted | warnings)
        for value in values:
            value["deployment_status"] = "validated" if value["strategy_id"] in admitted else "negative_training_evidence_only"
        if arm == "paths":
            values = [{k: v[k] for k in ("strategy_id", "scope", "recipe")} for v in values
                      if v["strategy_id"] in admitted]
        return values

    def runtime_better(self, incumbent, candidate):
        effect = paired_effect([incumbent], [candidate])
        if self.search_options["enabled"] and self.search_options["local_repair"]:
            if not preservation_report([incumbent], [candidate], self.search_options["preservation_threshold"],
                                       self.search_options["preservation_tolerance"])["passed"]:
                return False
        return (effect["gain"] > self.config["selection_min_gain"] and
                all(v >= -self.config["max_metric_regression"] for v in effect["metric_deltas"].values()))

    def test(self, frozen, mode="direct", arm="strategy"):
        memory, admitted, digest = self.load_frozen(frozen)
        root = self.root / "test" / mode / arm
        root.mkdir(parents=True, exist_ok=True)
        plan_path = root / "test_protocol.json"
        protocol = {"memory_hash": digest, "mode": mode, "arm": arm, "run_protocol": self.protocol_hash}
        if plan_path.exists() and json.loads(plan_path.read_text()) != protocol:
            raise ValueError("test memory or protocol changed; use a new output directory")
        write_json(plan_path, protocol)
        selected = []
        for task in self.dataset.test:
            for seed in self.seeds:
                episode = f"test/{mode}/{arm}/{task.task_id}/{seed}"
                selection_path = root / "selections" / (stable_hash([task.task_id, seed]) + ".json")
                if selection_path.exists():
                    selected.append(json.loads(selection_path.read_text()))
                    continue
                retrieved = self.test_memory(memory, admitted, task, arm)
                parent, incumbent = self.baseline, None
                history, attempts = [], []
                limit = 1 if mode == "direct" else self.config["test_max_attempts"]
                for attempt in range(limit):
                    proposal = self.propose(task, parent, history, retrieved, f"{episode}/{attempt}")
                    graph = proposal[0] if proposal is not None else parent
                    try:
                        record = self.evaluate(task, graph, seed, episode)
                    except EpisodeBudgetExceeded:
                        attempts.append({"attempt": attempt, "status": "budget_exhausted"})
                        break
                    except ResearchBudgetExceeded:
                        raise
                    except Exception as exc:
                        if h3_run_should_stop(exc) or isinstance(exc, MeasurementUnavailable):
                            raise
                        attempts.append({"attempt": attempt, "status": "execution_error", "error": str(exc)})
                        break
                    attempts.append({"attempt": attempt, "status": "ok", "evaluation_id": record["evaluation_id"]})
                    # Direct transfer commits its only output, regardless of score.
                    if incumbent is None or (mode == "adaptive" and self.runtime_better(incumbent, record)):
                        parent, incumbent = graph, record
                    history.append(record)
                if incumbent is None:
                    # Predeclared operational fallback, not a score-based oracle choice.
                    incumbent = self.evaluate(task, self.baseline, seed, episode)
                    parent = self.baseline
                    attempts.append({"status": "baseline_operational_fallback"})
                selection = {"task_id": task.task_id, "seed": seed, "evaluation_id": incumbent["evaluation_id"],
                    "graph_id": parent.graph_id, "video": incumbent["video"], "memory_hash": digest,
                    "retrieved_ids": [m["strategy_id"] for m in retrieved], "attempts": attempts,
                    "episode": episode, "budget": self.state["episodes"].get(episode, {})}
                write_json(selection_path, selection)
                selected.append(selection)
        # All outputs are committed before any final scoring or baseline comparison.
        write_json(root / "committed_selections.json", selected)
        pairs = []
        tasks = {t.task_id: t for t in self.dataset.test}
        for selection in selected:
            task, seed = tasks[selection["task_id"]], selection["seed"]
            candidate = json.loads((self.root / "evaluations" / (selection["evaluation_id"] + ".json")).read_text())
            baseline = self.evaluate(task, self.baseline, seed, f"comparison/{mode}/{arm}/{task.task_id}/{seed}")
            base_score = self.final_score(task, baseline, root)
            score = self.final_score(task, candidate, root)
            pairs.append({"task_id": task.task_id, "seed": seed, "baseline": base_score["score"],
                "candidate": score["score"], "delta": score["score"] - base_score["score"],
                "baseline_video": baseline["video"], "candidate_video": candidate["video"],
                "candidate_graph_id": selection["graph_id"], "episode_budget": selection["budget"],
                "baseline_criteria": base_score.get("criterion_scores", {}),
                "candidate_criteria": score.get("criterion_scores", {}),
                "criterion_deltas": {k: score["criterion_scores"][k] - v for k, v in base_score.get("criterion_scores", {}).items()
                                     if k in score.get("criterion_scores", {})}})
        if self.load_frozen(frozen)[2] != digest or memory.snapshot() != json.loads(Path(frozen).read_text())["entries"]:
            raise RuntimeError("frozen memory changed during held-out evaluation")
        by_task = {}
        for pair in pairs:
            by_task.setdefault(pair["task_id"], []).append(pair["delta"])
        gains = [statistics.mean(x) for x in by_task.values()]
        rng = random.Random(0)
        boot = sorted(statistics.mean(rng.choices(gains, k=len(gains))) for _ in range(2000))
        result = {"status": "complete", **protocol, "heldout_gain": statistics.mean(gains),
            "task_bootstrap_95ci": [boot[49], boot[1949]], "pairs": pairs,
            "test_tasks": len(by_task), "admitted_strategies": len(admitted),
            "independent_final_model": self.signature.get("verifier_profiles", {}).get("independent_model", False)
                if self.signature.get("verifier_profiles") else False,
            "planner_requests_total": self.state["planner_requests"], "ledger": asdict(self.ledger),
            "limitations": ["Runtime selection uses a fixed verifier, never final scores.",
                "Final assessment is a fresh call; same-model assessor bias remains unless independently calibrated.",
                "This is not an official VBench or StoryBench score.",
                "Budget is reserved native calls/duration, not measured GPU time; cache hits are reported separately.",
                "Path-memory control uses portable whole-graph sketches, not donor-specific executable paths."]}
        write_json(root / "summary.json", result)
        # Human review sees both videos under random labels, never their scores or graphs.
        blind, key = [], []
        for index, pair in enumerate(pairs):
            swap = rng.choice([False, True])
            videos = [pair["baseline_video"], pair["candidate_video"]]
            if swap:
                videos.reverse()
            for position, video in enumerate(videos):
                if video:
                    source = Path(video).resolve()
                    target = root / "human_review_media" / f"pair-{index:04d}-{'AB'[position]}.mp4"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        try:
                            os.link(source, target)
                        except OSError:
                            shutil.copyfile(source, target)
                    videos[position] = str(target.resolve())
            blind.append({"pair_id": index, "prompt": tasks[pair["task_id"]].prompt,
                          "rubric": tasks[pair["task_id"]].metadata.get("evaluation", {}),
                          "video_A": videos[0], "video_B": videos[1]})
            key.append({"pair_id": index, "task_id": pair["task_id"], "seed": pair["seed"],
                        "candidate_label": "A" if swap else "B"})
        write_json(root / "human_review_pairs.json", blind)
        write_json(root / "human_review_key.json", key)
        self.archive.export(programs=[], feedback=[], weighted_categories={}, frontier=[])
        print(f"[conditioning] {mode}/{arm} heldout_gain={result['heldout_gain']:+.4f} summary={root / 'summary.json'}", flush=True)
        return result

    def final_score(self, task, record, root):
        path = root / "final_scores" / (record["evaluation_id"] + ".json")
        if record["files"] != material_hashes(list(record["files"])):
            raise RuntimeError("selected media changed before final scoring")
        if path.exists():
            return json.loads(path.read_text())
        artifact = VideoArtifact(**deepcopy(record["artifact"]))
        # Do not show the assessor earlier scores, graph rationale, or strategy IDs.
        artifact.metadata = {k: v for k, v in artifact.metadata.items() if k in {
            "local_video_path", "sampled_frame_paths", "duration_seconds", "has_audio", "provider",
            "source_sampled_frame_paths", "reference_video", "real_video_processed"}}
        if self.final_augmenter is not None:
            artifact = self.final_augmenter.augment(task, artifact)
        report = self.evolver.evaluators.evaluate(task, artifact)
        reward = self.evolver.reward_router.evaluate(task, artifact, report)
        artifact.metadata["task_reward"] = reward.to_dict()
        self.check_measurement(artifact, reward.score)
        result = {"evaluation_id": record["evaluation_id"], "score": reward.score, "reward": reward.to_dict(),
                  "metrics": [asdict(m) for m in report.active_metrics], "assessment": "post_commit_fresh_verifier",
                  "criterion_scores": deepcopy(artifact.metadata.get("vlm_evaluation", {}).get("criterion_scores", {})),
                  "verification": deepcopy(artifact.metadata.get("vlm_evaluation", {}).get("verification_metadata", {}))}
        write_json(path, result)
        return result


def validate_config(config):
    defaults = {"searches_per_task": 2, "max_nodes": 16, "max_edits": 24, "retrieval_limit": 4,
        "max_validation_strategies": 8, "validation_tasks_per_strategy": 2, "min_validation_tasks": 1,
        "min_gain": .02, "max_metric_regression": .05, "selection_min_gain": .01,
        "test_max_attempts": 3, "test_max_generation_calls": 12, "test_max_generated_seconds": 180}
    defaults.update(search_mode="legacy", min_positive_seed_fraction=2 / 3, gain_se_multiplier=1.0)
    for key, value in defaults.items():
        config.setdefault(key, value)
    if config["search_mode"] not in {"legacy", "single", "factorial"}:
        raise ValueError("search_mode must be legacy, single or factorial")
    search_options(config)
    if not 0 <= config["min_positive_seed_fraction"] <= 1 or not math.isfinite(config["gain_se_multiplier"]) or config["gain_se_multiplier"] < 0:
        raise ValueError("invalid interaction acceptance settings")
    seeds = config["evaluation_seeds"]
    if not isinstance(seeds, list) or len(seeds) < 3 or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError("at least three distinct integer seeds required")
    for key in ("max_searches", "searches_per_task", "max_nodes", "max_edits", "retrieval_limit",
                "max_validation_strategies", "validation_tasks_per_strategy", "min_validation_tasks",
                "test_max_attempts", "test_max_generation_calls", "max_generation_calls"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("max_generated_seconds", "test_max_generated_seconds"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("min_gain", "max_metric_regression", "selection_min_gain"):
        if not math.isfinite(config[key]) or not 0 <= config[key] <= 1:
            raise ValueError(f"invalid {key}")
    if config["min_validation_tasks"] > config["validation_tasks_per_strategy"]:
        raise ValueError("minimum validation support exceeds validation task budget")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/h3_conditioning_strategies.json")
    parser.add_argument("--phase", choices=["learn", "test", "all"], default="all")
    parser.add_argument("--test-protocol", choices=["direct", "adaptive", "both"], default="direct")
    parser.add_argument("--memory-mode", choices=["strategy", "paths", "none", "all"], default="strategy")
    parser.add_argument("--memory", help="Frozen strategy snapshot for test-only runs")
    parser.add_argument("--output-dir")
    parser.add_argument("--task-file", help="Prepared, immutable task manifest")
    parser.add_argument("--search-mode", choices=["legacy", "single", "factorial"])
    parser.add_argument("--active-search", choices=["on", "off"], help="Use a signed interaction graph to select a factorial pair")
    parser.add_argument("--local-repair", choices=["on", "off"], help="Use bounded separators and preservation gates")
    parser.add_argument("--max-searches", type=int)
    parser.add_argument("--require-independent-final", action="store_true")
    parser.add_argument("--continue", dest="resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.memory and args.phase != "test":
        parser.error("--memory is only accepted for --phase test")
    config = json.loads(Path(args.config).read_text())
    for key in ("task_file", "search_mode", "max_searches"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    for arg, key in (("active_search", "enabled"), ("local_repair", "local_repair")):
        if getattr(args, arg) is not None:
            config.setdefault("active_graph_search", {})[key] = getattr(args, arg) == "on"
    if args.require_independent_final:
        config.setdefault("verifier", {})["require_independent_final"] = True
    validate_config(config)
    tasks = BenchmarkSuite.from_file(config["task_file"]).tasks
    dataset = stratified_task_split(tasks)
    validate_splits(dataset)
    settings = with_env_overrides(RuntimeSettings(**config["runtime"]))
    from evovideo_skill.conditioning_verifier import resolve_profiles, build_conditioning_verifier
    profiles = None if args.smoke else resolve_profiles(config, settings, require_keys=not args.dry_run)
    if args.smoke:
        settings = replace(settings, provider="local-fake", enable_vlm_eval=False, enable_llm_mutation=False)
    else:
        if settings.provider != "local-h3" or not settings.enable_vlm_eval or not settings.enable_llm_mutation:
            raise ValueError("real conditioning experiments require local-h3, VLM evaluation and API LLM planning")
        if settings.graph_planner_backend != "api":
            raise ValueError("use GRAPH_PLANNER_BACKEND=api for enforceable test-data isolation")
        if settings.restore_catalog_tools or settings.enable_open_world_tools or settings.enable_mcp_tools:
            raise ValueError("freeze tools: disable catalog restoration, open-world acquisition and MCP")
        if os.environ.get("VIDEO_OUTPUT_DIR") or os.environ.get("AGENT_STATE_DIR"):
            raise ValueError("unset VIDEO_OUTPUT_DIR and AGENT_STATE_DIR for isolated experiments")
        if settings.h3_local_model_revision == "unspecified":
            raise ValueError("pin H3_LOCAL_MODEL_REVISION to the deployed checkpoint")
        if any(t.metadata.get("h3_audio_criteria") for t in tasks) and not settings.h3_audio_verifier_command:
            settings = replace(settings, h3_audio_verifier_command=json.dumps(
                [sys.executable, "-m", "evovideo_skill.h3_omni_verifier"]))
        from evovideo_skill.h3_cli import preflight
        from evovideo_skill.harness import HarnessConfig
        verifier_keys = None
        if profiles:
            verifier_keys = [profiles[p]["api_key_env"] for p in ("runtime", "final")]
            if settings.h3_audio_verifier_command and "evovideo_skill.h3_omni_verifier" in json.loads(settings.h3_audio_verifier_command):
                verifier_keys.append("DASHSCOPE_API_KEY")
        preflight(HarnessConfig(name="conditioning", task_files=[config["task_file"]], runtime=settings,
                  evaluation_seeds=config["evaluation_seeds"]), require_credentials=not args.dry_run, check_server=False,
                  visual_credential_envs=verifier_keys)
    if profiles and not profiles["independent_model"]:
        print("[conditioning] WARNING: runtime/final use the same model; fresh/repeated calls are not independent-model verification", flush=True)
    if args.dry_run:
        print(json.dumps({"phase": args.phase, "test_protocol": args.test_protocol,
            "split": dataset.to_dict(), "config": {k: v for k, v in config.items() if k != "runtime"},
            "resolved_verifiers": profiles,
            "note": "No generation/planning API calls. Test variants share budget caps, not equal actual compute."}, indent=2))
        return
    root = Path(args.output_dir or config["output_dir"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "checkpoint.json").exists() and not args.resume:
            raise ValueError("output exists; use --continue or a new directory")
        settings = replace(settings, video_output_dir=str(root / "videos"), agent_state_dir=str(root / "agent_state"))
        tools, augmenter = (ToolRegistry.with_mock_tools(), None) if args.smoke else build_runtime(settings, build_verifier=not bool(profiles))
        proposer = None if args.smoke else build_graph_mutation_proposer(settings)
        planner = ConditioningSmokePlanner() if args.smoke else ConditioningPlanner(proposer)
        final_augmenter = augmenter
        if profiles:
            augmenter = build_conditioning_verifier(profiles["runtime"], root / "verifier" / "runtime", settings)
            final_augmenter = build_conditioning_verifier(profiles["final"], root / "verifier" / "final", settings)
            write_json(root / "verifier_profiles.json", profiles)
        elif not args.smoke and config.get("final_vlm_model"):
            _, final_augmenter = build_runtime(replace(settings, vlm_model=config["final_vlm_model"],
                video_output_dir=str(root / "final_assessor"), agent_state_dir=str(root / "final_assessor_state")))
            if final_augmenter.evaluator.model != config["final_vlm_model"]:
                raise ValueError("VLM_MODEL environment override conflicts with final_vlm_model; unset VLM_MODEL")
        # Paths and secrets are excluded; model/tool/verifier behavior must match snapshots.
        runtime_signature = {k: v for k, v in asdict(settings).items()
            if not any(word in k for word in ("key", "output_dir", "state_dir"))}
        signature = {"provider": settings.provider, "runtime": runtime_signature,
            "verifier_profiles": profiles,
            "final_vlm_model": profiles["final"]["model"] if profiles else config.get("final_vlm_model") or settings.vlm_model,
            "tool_manifest": [tools.spec(n).to_dict() for n in sorted(tools.available_names())],
            "planner": {k: v for k, v in asdict(proposer.config).items() if "key" not in k} if proposer else {},
            "code_hash": stable_hash({p.name: stable_hash(p.read_bytes().hex()) for p in Path(__file__).parent.glob("*.py")}),
            "verifier_environment": {k: v for k, v in os.environ.items()
                if k.startswith(("H3_OMNI_", "VLM_", "EVOVIDEO_VLM_")) and not any(s in k for s in ("KEY", "TOKEN", "SECRET"))},
            "evidence_type": "synthetic_smoke" if args.smoke else "real_video"}
        evolver = GraphToolPathEvolver(SkillMemory(root / "skills"), GraphSkillMemory(root / "graphs_memory"),
            tools=tools, planner=FixedTaskPlanner(), evaluators=build_evaluator_suite(settings),
            vlm_augmenter=augmenter, template_mutations_enabled=False)
        runner = ConditioningRunner(dataset, evolver, planner, root, config, signature, final_augmenter)
        try:
            if args.phase in {"learn", "all"}:
                runner.learn()
                frozen = runner.validate_and_freeze()
                print(f"[conditioning] frozen strategies={frozen}", flush=True)
            else:
                frozen = Path(args.memory) if args.memory else root / "frozen_strategies.json"
            if args.phase in {"test", "all"}:
                modes = ["direct", "adaptive"] if args.test_protocol == "both" else [args.test_protocol]
                arms = ["strategy", "paths", "none"] if args.memory_mode == "all" else [args.memory_mode]
                for mode in modes:
                    for arm in arms:
                        runner.test(frozen, mode, arm)
        except ResearchBudgetExceeded as exc:
            write_json(root / "incomplete.json", {"status": "budget_censored", "heldout_gain": None,
                "reason": str(exc), "ledger": asdict(runner.ledger)})
            print(f"[conditioning] budget_censored: {exc}; no complete held-out gain", flush=True)
            raise SystemExit(2)
        except MeasurementUnavailable as exc:
            write_json(root / "incomplete.json", {"status": "verifier_unavailable", "heldout_gain": None,
                "reason": str(exc), "ledger": asdict(runner.ledger)})
            print(f"[conditioning] stopped: {exc}; repair verifier connectivity and use --continue", flush=True)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
