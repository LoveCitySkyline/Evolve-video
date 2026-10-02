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
from evovideo_skill import conditioning_bargaining as bargaining
from evovideo_skill.conditioning_cost import (cost_options, profile as cost_profile, selection_gain,
    pareto_points, penalty as cost_penalty, delta as cost_delta)
from evovideo_skill.conditioning_memory import StrategyMemory, paired_effect, task_payload, task_state, metric_vector
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
    feedback_view, graph_payload, generation_credits, validate_splits, write_json)
from evovideo_skill.research_subgraphs import stable_hash
from evovideo_skill.runtime import (RuntimeSettings, build_evaluator_suite, build_graph_mutation_proposer,
    build_runtime, with_env_overrides)
from evovideo_skill.skill_memory import SkillMemory
from evovideo_skill.tools import ToolRegistry
from evovideo_skill.story_contracts import (prepare_story_task, validate_story_graph,
    acceptance_report, preserves_mandatory)


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


def training_coverage(tasks, config):
    from collections import Counter
    order = balanced_training_order(tasks)
    count = min(config["max_searches"], len(order) * config["searches_per_task"])
    visited = [order[i % len(order)] for i in range(count)]
    all_families = {task_state(t)["family"] for t in tasks}
    covered = Counter(task_state(t)["family"] for t in visited)
    return {"searches": count, "unique_tasks": len({t.task_id for t in visited}),
            "family_searches": dict(covered), "uncovered_families": sorted(all_families - covered.keys())}


def scenario_id(task):
    return str(task.metadata.get("scenario_group") or task.metadata.get("scenario_id") or task.task_id)


class ConditioningRunner:
    def __init__(self, dataset, evolver, planner, root, config, signature, final_augmenter=None):
        self.dataset, self.evolver, self.planner = dataset, evolver, planner
        for task in dataset.train + dataset.validation + dataset.test:
            prepare_story_task(task)
        self.root, self.config, self.signature = Path(root), config, signature
        self.root.mkdir(parents=True, exist_ok=True)
        self.final_augmenter = final_augmenter or evolver.vlm_augmenter
        self.baseline = evolver.baseline_graph()
        self.archive = EvolutionGraphArchive(self.root / "graph_visualization")
        self.seeds = config["evaluation_seeds"]
        self.protocol = {"version": "conditioning-strategies-v2", "signature": signature,
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

    def charge(self, calls, seconds, hit, episode, request_id=None):
        if hit or not calls:
            return
        reservations = self.state.setdefault("reservations", {})
        if request_id in reservations:
            previous = reservations[request_id]
            if (previous["episode"], previous["calls"], previous["seconds"]) != (episode, calls, seconds):
                raise ValueError("reservation identity reused with different cost or episode")
            return
        usage = self.state["episodes"].setdefault(episode, {"calls": 0, "seconds": 0})
        if episode.startswith("test/") and (
                usage["calls"] + calls > self.config["test_max_generation_calls"] or
                usage["seconds"] + seconds > self.config["test_max_generated_seconds"]):
            raise EpisodeBudgetExceeded("test episode generation budget exhausted")
        self.ledger.reserve(calls, seconds)
        usage["calls"] += calls
        usage["seconds"] += seconds
        if request_id is not None:
            reservations[request_id] = {"episode": episode, "calls": calls, "seconds": seconds,
                                        "status": "reserved"}
        self.save()  # One atomic checkpoint owns budget and idempotency together.

    def complete_reservation(self, request_id):
        reservation = self.state.get("reservations", {}).get(request_id)
        if reservation is not None:
            reservation["status"] = "completed"
            self.save()

    def evaluate(self, task, graph, seed, episode):
        validate_story_graph(task, graph)
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
            self.signature, lambda c, s, hit, key: self.charge(c, s, hit, episode, key),
            self.complete_reservation)
        started = time.monotonic()
        usage_before = deepcopy(self.state["episodes"].get(episode, {"calls": 0, "seconds": 0}))
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
                "generation_cost": cost_profile(task, graph, cost_options(self.config)),
                "reserved_budget_delta": {k: self.state["episodes"].get(episode, {}).get(k, 0) - usage_before.get(k, 0)
                                          for k in ("calls", "seconds")},
                "reused_nodes": cache.hits, "executed_nodes": cache.misses,
                "wall_seconds": time.monotonic() - started}
            if bargaining.options(self.config)['enabled']:
                record['bargaining_objective'] = bargaining.options(self.config)
                try:
                    record['bargaining_profile'] = bargaining.profile([record], record['bargaining_objective'])
                except ValueError as exc:
                    raise MeasurementUnavailable(str(exc)) from exc
            record["acceptance"] = acceptance_report(sampled, rollout.artifact)
            record["feedback"]["acceptance"] = deepcopy(record["acceptance"])
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

    def _retry_invalid_proposal(self, path, envelope, error, task, parent, records,
                                memory, operation, required_id, experiment):
        # One correction attempt per operation, durably recorded before calling
        # the planner. Never execute invalid graphs or change validation rules.
        envelope.setdefault("validation_failures", []).append({
            "error": error, "raw": envelope["raw"], "request": envelope["request"],
            "usage": envelope.get("usage"), "wall_seconds": envelope.get("wall_seconds"),
            "active_selection": envelope.get("active_selection")})
        if envelope.get("validation_retry_count", 0) >= 1:
            envelope.update(status="validation_failed", error=error)
            write_json(path, envelope)
            print(f"[conditioning] planner validation failed after correction operation={operation}: "
                  f"{error}; proposal={path}", flush=True)
            return None
        request = deepcopy(envelope["request"])
        request["validation_feedback"] = {"error": error, "failed_proposal": envelope["raw"],
            "instruction": "Correct this proposal and return the complete original response schema. "
                           "All original constraints and budgets still apply."}
        envelope.update(status="interrupted_planner", request=request, validation_retry_count=1)
        envelope.pop("active_selection", None)
        write_json(path, envelope)
        self.state["planner_requests"] += 1
        self.save()
        print(f"[conditioning] planner validation retry=1/1 operation={operation}: {error}", flush=True)
        started = time.monotonic()
        try:
            envelope.update(status="proposed", raw=self.planner.propose(request))
        except Exception as exc:
            envelope.update(status="planner_error", error=str(exc))
        envelope.update(wall_seconds=time.monotonic() - started, usage=getattr(self.planner, "last_usage", None))
        write_json(path, envelope)
        return self.propose(task, parent, records, memory, operation, required_id, experiment)

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
                "cost_objective": cost_options(self.config),
                "parent_generation_cost": cost_profile(task, parent, cost_options(self.config)),
                "bargaining": bargaining.options(self.config),
                "selection_instruction": ("Use fixed-reference multi-objective bargaining; keep each quality dimension, costs and conflicts explicit. Missing evidence is unknown. Staying with the parent is allowed."
                    if bargaining.options(self.config)['enabled'] else "Use the original configured quality/net-gain objective."),
                "cost_instruction": "Quality does not necessarily improve with longer paths. Keep original quality evidence separate from cold graph calls/video seconds. Prefer supported net gains under the configured cost objective. Node config.cost is NOT a measured cost and must remain fixed.",
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
                from evovideo_skill.conditioning_planner import conditioning_h3_grammar
                request["h3_native_planner"] = conditioning_h3_grammar(self.evolver.tools.available_names())
            if request["condition_only"]:
                from evovideo_skill.conditioning_planner import configuration_contract
                request["configuration_contract"] = configuration_contract(request["parent"])
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
                    failures = decision.get("invalid_factors", []) + decision["audit"].get("rejected_pairs", [])
                    correctable = [row for row in failures if not row["error"].startswith("insufficient budget")]
                    if correctable:
                        return self._retry_invalid_proposal(path, envelope, json.dumps(correctable),
                            task, parent, records, memory, operation, required_id, experiment)
                    return None
                print(f"[conditioning] selected executable pair operation={operation} "
                      f"pair={decision['audit'].get('selected_pair')}; starting matched-seed candidate evaluation", flush=True)
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
                "status": "invalid", "error": str(exc), "raw": envelope["raw"],
                "validation_retry_count": envelope.get("validation_retry_count", 0)})
            return self._retry_invalid_proposal(path, envelope, str(exc),
                task, parent, records, memory, operation, required_id, experiment)
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
        for index in range(min(self.config["max_searches"], len(order) * self.config["searches_per_task"])):
            task = order[index % len(order)]
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
            if (effect_supported(effect, self.config) if cost_options(self.config)["enabled"] or bargaining.options(self.config)['enabled']
                    else effect["gain"] > self.config["selection_min_gain"]):
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
        for index in range(cursor["next_index"], min(self.config["max_searches"], len(order) * self.config["searches_per_task"])):
            task = order[index % len(order)]
            if bargaining.options(self.config)['enabled'] and self.state.get('stopping', {}).get(task.task_id, {}).get('stop'):
                self.commit_interaction_cursor(index, parents, visited)
                continue
            visited.append(task.task_id)
            parent = parents.get(task.task_id, self.baseline)
            episode = "train/" + task.task_id
            before = [self.evaluate(task, parent, s, episode) for s in self.seeds]
            proposal = self.propose(task, parent, before, self.memory.retrieve(task, self.config["retrieval_limit"]),
                                    f"factorial/{index}", experiment=True)
            if proposal is None:
                print(f"[conditioning] skipped experiment operation=factorial/{index} task={task.task_id}: "
                      "no executable candidate pair; baseline only, no interaction evidence learned; "
                      f"inspect {self.root / 'candidate_audits.jsonl'}", flush=True)
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
                            "cost_interaction": interaction.get("cost_interaction"),
                            "factors": {k: strategies[k] for k in ("a", "b")},
                            "attribution": interaction["attribution"]}
                    self.memory.observe(task, strategies[label], graphs[label], effect, f"factorial/{index}/{label}",
                                        before_graph=graphs["anchor"])
            comparisons = {k: paired_effect(before, records) for k, records in cells.items()}
            preservation = {k: preservation_report(before, records,
                self.search_options["preservation_threshold"], self.search_options["preservation_tolerance"])
                for k, records in cells.items()} if self.search_options["enabled"] and self.search_options["local_repair"] else {}
            decision_checks = {k: {"supported_gain": effect_supported(effect, self.config),
                "preservation": preservation.get(k, {"passed": True})["passed"],
                "mandatory_preserved": all(preserves_mandatory(a, b) for a, b in zip(before, cells[k]))}
                for k, effect in comparisons.items()}
            eligible = [k for k, checks in decision_checks.items() if all(checks.values())]
            bargaining_points = []
            if bargaining.options(self.config)['enabled']:
                bargaining_points = bargaining.frontier({'parent': before, **cells}, bargaining.options(self.config))
                # Ineligible paths cannot dominate an otherwise admissible choice.
                admissible_points = bargaining.frontier({'parent': before, **{k: cells[k] for k in eligible}}, bargaining.options(self.config))
                allowed = {p['cell'] for p in admissible_points if p['pareto']}
                eligible = [k for k in eligible if k in allowed]
                winner = max(eligible, key=lambda k: (comparisons[k]['bargaining']['after']['score'],
                    comparisons[k]['bargaining']['after']['nash_log'])) if eligible else 'parent'
                if not errors:
                    self.state.setdefault('stopping', {})[task.task_id] = bargaining.update_stopping(
                        self.state.get('stopping', {}).get(task.task_id), winner != 'parent', self.config)
            else:
                winner = max(eligible, key=lambda k: selection_gain(comparisons[k], self.config)) if eligible else "parent"
            if winner != "parent":
                parents[task.task_id] = graphs[winner]
            for label in labels:
                self.archive.record_graph(graphs[label], stage="factorial", iteration=index,
                    status="selected" if label == winner else "execution_error" if label in errors else "observed",
                    metadata={"cell": label, "gain_to_parent": comparisons.get(label, {}).get("gain"),
                              "cost_effect": comparisons.get(label, {}).get("cost_effect"),
                              "connections": connection_manifest(graphs[label])})
            report = {"iteration": index, "task_id": task.task_id, "search_mode": self.config["search_mode"],
                "parent_graph": parent.graph_id, "selected_cell": winner,
                "status": "complete" if not errors else "partial_execution_failure",
                "interaction": interaction, "comparisons_to_parent": comparisons, "execution_errors": errors,
                "preservation": preservation, "active_search": deepcopy(self.active_selection),
                "cost_objective": cost_options(self.config),
                "pareto_frontier": pareto_points({"parent": before, **cells}, cost_options(self.config)),
                "selection_gains": {k: selection_gain(e, self.config) for k, e in comparisons.items()},
                "decision_checks": decision_checks,
                "bargaining_frontier": bargaining_points,
                "stopping": self.state.get('stopping', {}).get(task.task_id),
                "selected_graph": graph_payload(parents.get(task.task_id, parent)),
                "connections": {k: connection_manifest(graphs[k]) for k in labels},
                "cells": {k: [{"evaluation_id": r["evaluation_id"], "seed": r["seed"], "score": r["score"],
                              "video": r["video"], "generation_cost": r.get("generation_cost"),
                              "reserved_budget_delta": r.get("reserved_budget_delta"),
                              "generation_wall_seconds": r.get("process_diagnostics", {}).get("generation_wall_seconds"),
                              "evaluation_wall_seconds": r.get("wall_seconds")} for r in records] for k, records in cells.items()}}
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
            "cost_objective": cost_options(self.config),
            "signed_graph_nodes": len(self.signed_graph.data["nodes"]),
            "signed_graph_edges": len(self.signed_graph.data["edges"]),
            "note": "Local paired-seed graph interventions. No real improvement is guaranteed."})

    def commit_interaction_cursor(self, index, parents, visited):
        self.state["interaction_cursor"] = {"next_index": index + 1,
            "parents": {k: graph_payload(g) for k, g in parents.items()}, "visited": list(visited)}
        self.state["signed_interaction_graph"] = deepcopy(self.signed_graph.data)
        self.save()
        if self.search_options["enabled"]:
            audits = [json.loads(p.read_text())["audit"] for p in (self.root / "search_decisions").glob("*.json")]
            coverages = [a.get("evidence_coverage", {}) for a in audits]
            write_json(self.root / "search_evidence_coverage.json", {
                "decisions": len(audits),
                "decisions_using_exact_evidence": sum("exact" in c.get("selected_sources", []) for c in coverages),
                "decisions_using_backoff_evidence": sum("structural_backoff" in c.get("selected_sources", []) for c in coverages),
                "decisions_using_only_prior": sum(bool(c.get("selected_sources")) and set(c["selected_sources"]) == {"prior"} for c in coverages),
                "independent_task_support_per_edge": [len({o["task_id"] for o in e["observations"].values()})
                    for e in self.signed_graph.data["edges"].values()]})
            write_json(self.root / "signed_interaction_graph.json", self.signed_graph.data)
            write_json(self.root / "signed_interaction_summary.json", self.signed_graph.view(self.search_options))
            from evovideo_skill.conditioning_cost_report import export_cost_report
            export_cost_report(self.root, self.signed_graph.view(self.search_options), cost_options(self.config))

    def validate_and_freeze(self):
        if not self.state["learned"]:
            raise ValueError("learn before validation")
        frozen = self.root / "frozen_strategies.json"
        if self.state["validated"]:
            return frozen
        candidates = sorted(self.memory.entries.values(), key=lambda e: (
            -StrategyMemory.view(e)["evidence"]["mean_selection_gain"], e["strategy_id"]))
        admitted = []
        reports = []
        for entry in candidates[:self.config["max_validation_strategies"]]:
            key = entry["strategy_id"]
            tasks = [t for t in self.dataset.validation if StrategyMemory.matches(entry, task_state(t))]
            tasks = tasks[:self.config["validation_tasks_per_strategy"]]
            effects, failures, preservation, mandatory_preserved = [], [], [], []
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
                    mandatory_preserved.append(all(preserves_mandatory(a, b) for a, b in zip(before, after)))
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
            objective = cost_options(self.config)
            bargain = bargaining.options(self.config)
            net_gain = statistics.mean(selection_gain(e, self.config) for e in effects) if effects else None
            accepted = (len(effects) >= self.config["min_validation_tasks"] and not failures
                        and net_gain >= (bargain['min_validation_gain'] if bargain['enabled'] else
                            objective["min_validation_net_gain"] if objective["enabled"] else self.config["min_gain"])
                        and all(v >= -self.config["max_metric_regression"]
                                for e in effects for v in e["metric_deltas"].values()))
            if self.config.get("search_mode", "legacy") != "legacy" or objective["enabled"] or bargain['enabled']:
                accepted = accepted and all(effect_supported(e, self.config) for e in effects)
            accepted = accepted and all(r["passed"] for r in preservation) and all(mandatory_preserved)
            if accepted:
                admitted.append(key)
            reports.append({"strategy_id": key, "accepted": accepted, "gain": gain,
                "selection_gain": net_gain, "cost_objective": objective,
                "bargaining_objective": bargain,
                "tasks": [t.task_id for t in tasks], "execution_failures": failures, "effects": effects,
                "preservation": preservation, "mandatory_preserved": mandatory_preserved,
                "required_task_support": self.config["min_validation_tasks"],
                "insufficient_support": len(effects) < self.config["min_validation_tasks"]})
        write_json(self.root / "validation_reports.json", reports)
        data = {"version": "conditioning-strategies-v1", "signature": self.signature,
            "cost_objective": cost_options(self.config),
            "entries": self.memory.snapshot(), "admitted_ids": admitted,
            "bargaining_objective": bargaining.options(self.config),
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
        if data.get("cost_objective", cost_options({})) != cost_options(self.config):
            raise ValueError("frozen strategy cost objective mismatch; revalidate with the chosen weights")
        if data.get('bargaining_objective', bargaining.options({})) != bargaining.options(self.config):
            raise ValueError('frozen bargaining objective mismatch; revalidate before testing')
        if set(data["source_task_ids"]) & {t.task_id for t in self.dataset.test} or set(data["source_scenarios"]) & {
                scenario_id(t) for t in self.dataset.test}:
            raise ValueError("training/validation data overlaps held-out test")
        return StrategyMemory(data["entries"], frozen=True), set(data["admitted_ids"]), content_hash

    def test_memory(self, memory, admitted, task, arm):
        if arm == "none":
            return []
        warnings = {k for k, e in memory.entries.items() if StrategyMemory.view(e)["evidence"]["mean_selection_gain"] <= 0}
        values = memory.retrieve(task, self.config["retrieval_limit"], admitted | warnings)
        for value in values:
            value["deployment_status"] = "validated" if value["strategy_id"] in admitted else "negative_training_evidence_only"
        if arm == "paths":
            values = [{k: v[k] for k in ("strategy_id", "scope", "recipe", "structural_contract")} for v in values
                      if v["strategy_id"] in admitted]
        return values

    def runtime_better(self, incumbent, candidate, *, paired=True):
        if not preserves_mandatory(incumbent, candidate):
            return False
        if paired:
            effect = paired_effect([incumbent], [candidate])
        else:
            # Independent-seed best-of-N selection, not a paired effect estimate.
            if incumbent["task_id"] != candidate["task_id"]:
                raise ValueError("cannot compare different tasks")
            a, b = metric_vector(incumbent), metric_vector(candidate)
            if a.keys() != b.keys():
                return False
            effect = {"gain": candidate["score"] - incumbent["score"],
                      "metric_deltas": {k: b[k] - a[k] for k in a}}
        if self.search_options["enabled"] and self.search_options["local_repair"]:
            if not preservation_report([incumbent], [candidate], self.search_options["preservation_threshold"],
                                       self.search_options["preservation_tolerance"], paired=paired)["passed"]:
                return False
        if (cost_options(self.config)["enabled"] or bargaining.options(self.config)['enabled']) and paired:
            return effect_supported(effect, self.config)
        # Independent-seed matched-budget baseline draws use the SAME graph:
        # their cold cost is equal, so quality ordering remains appropriate.
        return (effect["gain"] > self.config["selection_min_gain"] and
                all(v >= -self.config["max_metric_regression"] for v in effect["metric_deltas"].values()))

    def comparison_baseline(self, task, seed, selection, mode, arm):
        episode = f"comparison/{mode}/{arm}/{task.task_id}/{seed}"
        calls, seconds = generation_credits(task, self.baseline)
        matched = self.config["comparison_baseline"] == "matched_budget"
        budget = selection["budget"]
        count = min(budget.get("calls", 0) // calls, int(budget.get("seconds", 0) // seconds)) if matched else 1
        if count < 1:
            raise ResearchBudgetExceeded("candidate budget cannot fund even one matched baseline; no fair comparison")
        # Spend at most the candidate's actual reserved native calls AND seconds.
        # Select with runtime scores only, before any final assessment.
        incumbent, repeats = None, []
        for index in range(count):
            replicate = seed if index == 0 else int(stable_hash([task.task_id, seed, index, "baseline"])[:8], 16)
            record = self.evaluate(task, self.baseline, replicate, episode)
            repeats.append({"seed": replicate, "evaluation_id": record["evaluation_id"]})
            if incumbent is None or self.runtime_better(incumbent, record, paired=False):
                incumbent = record
        return incumbent, {"mode": self.config["comparison_baseline"], "replicates": repeats,
            "reserved_budget": deepcopy(self.state["episodes"].get(episode, {})),
            "candidate_budget": deepcopy(budget),
            "qualification": "Native calls/duration matched from below; not equal GPU time or token cost."}

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
                stopping = None
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
                    improved = incumbent is None or (mode == "adaptive" and self.runtime_better(incumbent, record))
                    if improved:
                        parent, incumbent = graph, record
                    history.append(record)
                    if mode == 'adaptive' and bargaining.options(self.config)['enabled']:
                        stopping = bargaining.update_stopping(stopping, improved, self.config)
                        if stopping['stop']:
                            break
                if incumbent is None:
                    # Predeclared operational fallback, not a score-based oracle choice.
                    incumbent = self.evaluate(task, self.baseline, seed, episode)
                    parent = self.baseline
                    attempts.append({"status": "baseline_operational_fallback"})
                selection = {"task_id": task.task_id, "seed": seed, "evaluation_id": incumbent["evaluation_id"],
                    "graph_id": parent.graph_id, "video": incumbent["video"], "memory_hash": digest,
                    "retrieved_ids": [m["strategy_id"] for m in retrieved], "attempts": attempts,
                    "episode": episode, "budget": self.state["episodes"].get(episode, {})}
                selection['stopping'] = stopping
                write_json(selection_path, selection)
                selected.append(selection)
        # All outputs are committed before any final scoring or baseline comparison.
        write_json(root / "committed_selections.json", selected)
        pairs = []
        tasks = {t.task_id: t for t in self.dataset.test}
        for selection in selected:
            task, seed = tasks[selection["task_id"]], selection["seed"]
            candidate = json.loads((self.root / "evaluations" / (selection["evaluation_id"] + ".json")).read_text())
            baseline, comparison = self.comparison_baseline(task, seed, selection, mode, arm)
            base_score = self.final_score(task, baseline, root)
            score = self.final_score(task, candidate, root)
            objective = cost_options(self.config)
            bc, cc = baseline["generation_cost"], candidate["generation_cost"]
            dc = cost_penalty(cc, objective) - cost_penalty(bc, objective)
            pairs.append({"task_id": task.task_id, "seed": seed, "baseline": base_score["score"],
                "candidate": score["score"], "delta": score["score"] - base_score["score"],
                "cost_tradeoff": {"baseline": bc, "candidate": cc, **cost_delta(bc, cc),
                    "cost_penalty_delta": dc, "net_gain": score["score"] - base_score["score"] - dc},
                "baseline_video": baseline["video"], "candidate_video": candidate["video"],
                "candidate_graph_id": selection["graph_id"], "episode_budget": selection["budget"],
                "comparison_control": comparison,
                "baseline_criteria": base_score.get("criterion_scores", {}),
                "candidate_criteria": score.get("criterion_scores", {}),
                "baseline_acceptance": base_score["acceptance"],
                "candidate_acceptance": score["acceptance"],
                "criterion_deltas": {k: score["criterion_scores"][k] - v for k, v in base_score.get("criterion_scores", {}).items()
                                     if k in score.get("criterion_scores", {})}})
            if bargaining.options(self.config)['enabled']:
                o = bargaining.options(self.config)
                def final_profile(assessment, cost):
                    return bargaining.profile([{**assessment, 'generation_cost': cost,
                        'feedback': {'metrics': {m['name']: m for m in assessment['metrics']}}}], o)
                bp, cp = final_profile(base_score, bc), final_profile(score, cc)
                pairs[-1]['bargaining'] = {'baseline': bp, 'candidate': cp, 'gain': cp['score']-bp['score']}
        if self.load_frozen(frozen)[2] != digest or memory.snapshot() != json.loads(Path(frozen).read_text())["entries"]:
            raise RuntimeError("frozen memory changed during held-out evaluation")
        by_task = {}
        for pair in pairs:
            by_task.setdefault(pair["task_id"], []).append(pair["delta"])
        gains = [statistics.mean(x) for x in by_task.values()]
        rng = random.Random(0)
        boot = sorted(statistics.mean(rng.choices(gains, k=len(gains))) for _ in range(2000))
        result = {"status": "complete", **protocol, "heldout_gain": statistics.mean(gains),
            "cost_objective": cost_options(self.config),
            "heldout_net_gain": statistics.mean(statistics.mean(p["cost_tradeoff"]["net_gain"]
                for p in pairs if p["task_id"] == task_id) for task_id in by_task),
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
        contract_pairs = [p for p in pairs if p["candidate_acceptance"]["status"] != "not_applicable"]
        result['bargaining_objective'] = bargaining.options(self.config)
        if bargaining.options(self.config)['enabled']:
            result['heldout_bargaining_gain'] = statistics.mean(statistics.mean(p['bargaining']['gain']
                for p in pairs if p['task_id'] == task_id) for task_id in by_task)
        result["contract_acceptance"] = {
            "evaluated_outputs": len(contract_pairs),
            "baseline_pass_rate": (statistics.mean(p["baseline_acceptance"]["status"] == "passed" for p in contract_pairs)
                                   if contract_pairs else None),
            "candidate_pass_rate": (statistics.mean(p["candidate_acceptance"]["status"] == "passed" for p in contract_pairs)
                                    if contract_pairs else None),
            "note": "Experiment completion is separate from story acceptance; rates count task/seed outputs."}
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
                  "acceptance": acceptance_report(task, artifact),
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
    defaults.update(search_mode="legacy", min_positive_seed_fraction=2 / 3, gain_se_multiplier=1.0,
                    comparison_baseline="single")
    for key, value in defaults.items():
        config.setdefault(key, value)
    if config["search_mode"] not in {"legacy", "single", "factorial"}:
        raise ValueError("search_mode must be legacy, single or factorial")
    search_options(config)
    cost_options(config)
    bargain = bargaining.options(config)
    if bargain['enabled'] and not search_options(config)['enabled']:
        raise ValueError('bargaining requires active factorial graph search')
    if config["comparison_baseline"] not in {"single", "matched_budget"}:
        raise ValueError("comparison_baseline must be single or matched_budget")
    if config.get("experiment_tier", "pilot") not in {"pilot", "research"}:
        raise ValueError("experiment_tier must be pilot or research")
    if config.get("experiment_tier") == "research" and (
            config["min_validation_tasks"] < 2 or config["comparison_baseline"] != "matched_budget"
            or config.get("verifier", {}).get("require_independent_final") is not True):
        raise ValueError("research tier requires multiple validation tasks, matched budget and independent final verification")
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
    task_path = Path(config["task_file"]).resolve()
    if not task_path.is_file():
        hint = (
            "Run bash scripts/prepare_complex_video_bench_mini50_h3.sh first. "
            "Supply --asset-manifest with existing inputs, or explicitly use --generate-missing "
            "after starting H3 to generate them; review assets before --approve-assets. "
            if task_path.name == "complex_video_bench_mini50_h3.json" else ""
        )
        parser.error(
            f"Task manifest not found: {task_path}. "
            "--dry-run still requires the task manifest; it does not prepare assets. "
            + hint + "Use --task-file /absolute/path/to/prepared.json if already prepared elsewhere."
        )
    tasks = BenchmarkSuite.from_file(config["task_file"]).tasks
    for task in tasks:
        prepare_story_task(task)
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
        if settings.graph_planner_backend not in {"api", "codex"}:
            raise ValueError("GRAPH_PLANNER_BACKEND must be api or codex")
        if settings.graph_planner_backend == "codex":
            if config.get("experiment_tier", "pilot") == "research":
                raise ValueError("research tier requires GRAPH_PLANNER_BACKEND=api for held-out data isolation; Codex CLI is supported for pilot")
            print("[conditioning] Codex CLI planner enabled for pilot; no graph-planner API key required. "
                  "Filesystem read isolation is not guaranteed.", flush=True)
        if settings.restore_catalog_tools or settings.enable_open_world_tools or settings.enable_mcp_tools:
            raise ValueError("freeze tools: disable catalog restoration, open-world acquisition and MCP")
        if os.environ.get("VIDEO_OUTPUT_DIR") or os.environ.get("AGENT_STATE_DIR"):
            raise ValueError("unset VIDEO_OUTPUT_DIR and AGENT_STATE_DIR for isolated experiments")
        revision = settings.h3_local_model_revision
        if not revision or not revision.strip() or revision.strip() == "unspecified":
            settings = replace(settings, h3_local_model_revision="unspecified")
            if config.get("experiment_tier", "pilot") == "research" and not args.dry_run:
                raise ValueError("research runs require H3_LOCAL_MODEL_REVISION to identify the deployed checkpoint; pilot and --dry-run allow unspecified")
            print("[conditioning] WARNING: H3_LOCAL_MODEL_REVISION is unspecified; continuing "
                  "pilot/dry-run with unknown weight revision. Record the actual revision before a research run.", flush=True)
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
            "h3_local_model_revision": settings.h3_local_model_revision,
            "graph_planner_backend": settings.graph_planner_backend,
            "training_coverage": training_coverage(dataset.train, config),
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
        if args.smoke:
            planner = ConditioningSmokePlanner()
        elif settings.graph_planner_backend == "codex":
            from evovideo_skill.conditioning_planner import CodexConditioningPlanner
            planner = CodexConditioningPlanner(proposer)
        else:
            planner = ConditioningPlanner(proposer)
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
        if not args.smoke and settings.graph_planner_backend == "codex":
            signature["planner"]["codex_exec"] = {k: v for k, v in asdict(planner.codex.config).items()
                                                   if k != "job_root"}
            signature["planner"]["held_out_read_isolation"] = "not_enforced_pilot_only"
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
            print(f"[conditioning] stopped: {exc}; inspect verifier evidence/errors; resume only after resolving the cause", flush=True)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
