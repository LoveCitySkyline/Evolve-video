from __future__ import annotations

import json
import hashlib
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from evovideo_skill.evolution_cache import EvolutionCacheConfig, EvolutionRunCache
from evovideo_skill.evolution_data import EvolutionDataset, RoundRobinTaskSampler
from evovideo_skill.evolution_feedback import EvolutionFeedback, FeedbackJournal
from evovideo_skill.graph_executor import GraphToolExecutionError
from evovideo_skill.graph_evolver import (
    GraphPathCandidate, GraphPathRollout, GraphToolPathEvolver, h3_registry_active, h3_run_should_stop,
)
from evovideo_skill.graph_skill import ExperienceRecord, GraphSkillMemory, ToolPathGraph, graph_score_stability
from evovideo_skill.graph_visualization import EvolutionGraphArchive
from evovideo_skill.models import FailureReport, SkillValidationReport, VideoTask, utc_now
from evovideo_skill.online_foundation import TaskConditionedFoundationPromoter
from evovideo_skill.program_registry import GraphProgram, ProgramMetrics, ProgramRegistry, SelectionStrategy
from evovideo_skill.weighted_tool_graph import (
    H3_COMPARISON_PROTOCOL, H3_LOCAL_COMPARISON_PROTOCOL, H3_LOCAL_EVALUATION_PROTOCOL,
    WeightedToolGraphMemory, graph_behavior_fingerprint, h3_local_active, h3_local_seed_applied, task_class,
)


@dataclass
class EvolutionLoopConfig:
    max_iterations: int = 5
    frontier_size: int = 3
    no_improvement_limit: int = 3
    selection_strategy: SelectionStrategy = "best"
    categories_per_batch: int = 3
    samples_per_category: int = 1
    homogeneous_task_batches: bool = False
    max_proposals_per_iteration: int = 3
    max_mutation_searches: int | None = 8
    min_quality_gain: float = 0.02
    max_task_metric_regression: float = 0.05
    min_task_metric_gain: float = 0.0
    cache_enabled: bool = True
    evaluation_seeds: list[int] = field(default_factory=lambda: [42, 123, 456])
    h3_min_replicates: int = 3
    continue_mode: bool = False
    cost_weight: float = 0.03
    path_prior_weight: float = 0.5
    online_foundation_enabled: bool = True
    online_foundation_interval: int = 1
    online_foundation_min_gain: float = 0.0
    exploratory_admission_enabled: bool = True
    exploratory_min_worst_seed_gain: float = 0.05
    exploratory_min_stability_gain: float = 0.02
    exploratory_max_pair_regression: float = 0.03
    runtime_failure_circuit_breaker: bool = True
    full_suite_evaluation: bool = False
    allow_seed_holdout_validation: bool = False
    seed_holdout_validation_seed: int = 123
    near_miss_repair_enabled: bool = False


@dataclass
class TaskRolloutSummary:
    task_id: str
    graph_id: str
    score: float
    passed: bool
    failure_types: list[str]
    metric_scores: dict[str, float]
    estimated_cost: float
    tool_chain: list[str]
    active_metric_names: list[str] = field(default_factory=list)
    cache_hit: bool = False
    evaluation_seed: int | None = None
    seed_controlled: bool | None = None
    provider_seed_control: bool | None = None
    replicate_label: int | None = None
    evaluation_protocol: str | None = None
    actual_tool_edges: list[tuple[str, str]] | None = None
    execution_error: str | None = None
    failed_tool_name: str | None = None
    artifact_path: str | None = None
    runtime_log: str | None = None
    tool_provenance: dict[str, dict[str, Any]] = field(default_factory=dict)
    reward_objective: str = "video_quality"
    reward_components: dict[str, float] = field(default_factory=dict)
    reward_weights: dict[str, float] = field(default_factory=dict)
    missing_reward_metrics: list[str] = field(default_factory=list)
    verifier_evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgramEvaluation:
    program_name: str
    metrics: ProgramMetrics
    rollouts: list[TaskRolloutSummary]


@dataclass
class EvolutionIteration:
    iteration: int
    parent_program: str
    sampled_task_ids: list[str]
    failure_types: list[str]
    proposed_programs: list[str]
    accepted_programs: list[str]
    rejected_programs: list[str]
    frontier: list[str]
    status: str
    promoted_foundations: list[str] = field(default_factory=list)
    exploratory_programs: list[str] = field(default_factory=list)
    mutation_searches_used: int = 0
    mutation_searches_remaining: int | None = None


@dataclass
class EvolutionLoopResult:
    best_program: str
    best_metrics: ProgramMetrics
    frontier: list[str]
    iterations: list[EvolutionIteration]
    dataset: dict[str, Any]
    program_root: str
    feedback_path: str
    cache_dir: str
    validation_evaluation: ProgramEvaluation
    test_evaluation: ProgramEvaluation | None
    full_baseline_evaluation: ProgramEvaluation | None
    full_final_evaluation: ProgramEvaluation | None
    mutation_searches_used: int = 0
    max_mutation_searches: int | None = None
    mutation_search_budget_exhausted: bool = False
    visualization_paths: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    @property
    def accepted_candidate_count(self) -> int:
        return sum(len(item.accepted_programs) for item in self.iterations)

    @property
    def rejected_candidate_count(self) -> int:
        return sum(len(item.rejected_programs) for item in self.iterations)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GraphSelfImprovingLoop:
    """EvoSkill-style iterative program evolution for graph-structured video agents."""

    def __init__(
        self,
        config: EvolutionLoopConfig,
        dataset: EvolutionDataset,
        evolver: GraphToolPathEvolver,
        graph_memory: GraphSkillMemory,
        state_dir: str | Path,
        runtime_signature: dict[str, Any] | None = None,
        weighted_tool_graph: WeightedToolGraphMemory | None = None,
        foundation_promoter: TaskConditionedFoundationPromoter | None = None,
        warm_start_graph_ids: list[str] | None = None,
    ):
        self.config = config
        self.dataset = dataset
        self.evolver = evolver
        self.graph_memory = graph_memory
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ProgramRegistry(self.state_dir / "registry", cost_weight=config.cost_weight)
        self.feedback = FeedbackJournal(self.state_dir)
        self.cache = EvolutionRunCache(
            EvolutionCacheConfig(self.state_dir / "cache", enabled=config.cache_enabled)
        )
        self.checkpoint_path = self.state_dir / "loop_checkpoint.json"
        self.iteration_log_path = self.state_dir / "iterations.jsonl"
        self.runtime_signature = runtime_signature or {}
        self.weighted_tool_graph = weighted_tool_graph
        self.foundation_promoter = foundation_promoter
        self.warm_start_graph_ids = list(dict.fromkeys(warm_start_graph_ids or []))
        self.foundation_log_path = self.state_dir / "online_foundations.jsonl"
        self.runtime_decision_path = self.state_dir / "runtime_decisions.jsonl"
        self.runtime_decisions: list[dict[str, Any]] = []
        self.runtime_session_id = utc_now()
        self.graph_archive = EvolutionGraphArchive(self.state_dir / "graph_visualization")
        self._foundation_attempts: set[str] = set()
        self._runtime_tool_failures: dict[str, str] = {}
        self._mutation_searches_used = 0
        self._exploration_visits: dict[str, int] = {}
        if config.allow_seed_holdout_validation:
            self._augment_validation_with_seed_holdouts()
        sampler_tasks = list(dataset.train)
        if config.homogeneous_task_batches and (not config.allow_seed_holdout_validation or self._h3_active()):
            validation_classes = {task_class(task) for task in dataset.validation}
            category_matched = [
                task for task in sampler_tasks
                if task_class(task) in validation_classes
            ]
            if category_matched:
                excluded_classes = sorted({
                    task_class(task) for task in sampler_tasks
                    if task_class(task) not in validation_classes
                })
                sampler_tasks = category_matched
                if excluded_classes:
                    print(
                        "[evolution] excluded training-only task classes from mutation sampling "
                        "because no paired validation task exists: "
                        + ", ".join(excluded_classes)
                    )
        self.sampler = RoundRobinTaskSampler(sampler_tasks)
        self._graphs: dict[str, ToolPathGraph] = {
            graph.skill_name: graph for graph in graph_memory.list_graphs()
        }

    def _augment_validation_with_seed_holdouts(self) -> None:
        """Give compact singleton task classes a fixed-seed validation context."""
        if self._h3_active():
            return
        validation_classes = {task_class(task) for task in self.dataset.validation}
        by_class: dict[str, list[VideoTask]] = {}
        for task in self.dataset.train:
            by_class.setdefault(task_class(task), []).append(task)
        additions: list[VideoTask] = []
        for category, tasks in sorted(by_class.items()):
            if category in validation_classes:
                continue
            source = sorted(tasks, key=lambda task: task.task_id)[0]
            additions.append(
                VideoTask(
                    task_id=(
                        f"{source.task_id}__seed_holdout_"
                        f"{self.config.seed_holdout_validation_seed}"
                    ),
                    prompt=source.prompt,
                    mode=source.mode,
                    duration_seconds=source.duration_seconds,
                    reference_video=source.reference_video,
                    metadata={
                        **(source.metadata or {}),
                        "generation_seed": self.config.seed_holdout_validation_seed,
                        "seed_holdout_validation": True,
                        "seed_holdout_source_task_id": source.task_id,
                    },
                )
            )
        if additions:
            self.dataset.validation.extend(additions)
            print(
                "[evolution] added fixed-seed validation holdouts for compact task classes: "
                + ", ".join(task_class(task) for task in additions)
            )

    def run(self) -> EvolutionLoopResult:
        start_iteration, no_improvement = self._initialize_or_resume()
        iteration_records: list[EvolutionIteration] = self._read_iteration_records()

        for iteration in range(start_iteration, self.config.max_iterations + 1):
            if self._mutation_search_budget_exhausted():
                break
            parent = self.registry.select(self.config.selection_strategy, iteration)
            if parent is None:
                raise RuntimeError("evolution frontier is empty")
            self.registry.switch_to(parent.name)
            repair_graph, repair_tasks = self._next_near_miss_repair(iteration)
            if repair_graph is not None:
                sampled = repair_tasks
            elif self.config.homogeneous_task_batches:
                sampled = self.sampler.sample_homogeneous(
                    self.config.categories_per_batch * self.config.samples_per_category
                )
            else:
                sampled = self.sampler.sample(
                    self.config.categories_per_batch,
                    self.config.samples_per_category,
                )
            training_rollouts = []
            for task in sampled:
                exploration_task = self._exploration_variant(task, iteration)
                rollout = self.evolver.rollout(
                    exploration_task,
                    repair_graph or self._select_graph(parent, exploration_task),
                )
                if repair_graph is not None:
                    gate = repair_graph.stats.get("gate_evidence", {})
                    followup_reason = repair_graph.stats.get("near_miss_repair", {}).get(
                        "reason", "Positive validation gain but task-metric regression")
                    repair_evidence = {
                        "source": "validation_selection_feedback",
                        "repair_graph_id": repair_graph.graph_id,
                        "mean_paired_gain": gate.get("mean_paired_gain"),
                        "task_metric_mean_deltas": gate.get("task_metric_mean_deltas", {}),
                        "max_task_metric_regression": gate.get("max_task_metric_regression"),
                        "paired_improvement_fraction": gate.get("paired_improvement_fraction"),
                        "paired_tie_fraction": gate.get("paired_tie_fraction"),
                        "reason": followup_reason,
                        "instruction": "Preserve gains, fix actual regressed criteria, or improve robustness when gains occur only in some replicates. Do not invent a regression for tied scores. Use reusable current-task conditioning. This is validation feedback, not localized evidence for this training video.",
                    }
                    if rollout.failure is None:
                        rollout.failure = FailureReport(
                            task_id=exploration_task.task_id, artifact_id=rollout.artifact.artifact_id,
                            failure_types=[], evidence=[followup_reason],
                            likely_causes=[], recommended_updates=[repair_evidence["instruction"]],
                        )
                    rollout.failure.intervention["near_miss_repair"] = repair_evidence
                training_rollouts.append(rollout)
                self._archive_rollout(rollout)
            failures = [rollout for rollout in training_rollouts if rollout.failure is not None]
            failure_types = sorted(
                {
                    failure.value
                    for rollout in failures
                    for failure in rollout.failure.failure_types
                }
            )
            if not failures:
                no_improvement += 1
                promoted = self._refresh_online_foundations(
                    iteration,
                    parent,
                )
                if promoted:
                    no_improvement = 0
                record = EvolutionIteration(
                    iteration,
                    parent.name,
                    [task.task_id for task in sampled],
                    [],
                    [],
                    [],
                    [],
                    [program.name for program in self.registry.frontier()],
                    "foundation_promoted" if promoted else "all_sampled_tasks_passed",
                    promoted,
                    mutation_searches_used=self._mutation_searches_used,
                    mutation_searches_remaining=self._remaining_mutation_searches(),
                )
                iteration_records.append(record)
                self._append_iteration(record)
                self._save_checkpoint(iteration, no_improvement)
                if no_improvement >= self.config.no_improvement_limit:
                    break
                continue

            raw_candidates = self.evolver.propose_candidates(training_rollouts)
            self._archive_candidate_audits(iteration, failures)
            novel_candidates = self._novel_candidates(
                parent,
                raw_candidates,
                iteration,
            )
            candidates = self._select_candidate_portfolio(
                novel_candidates,
                failures,
                min(
                    1 if repair_graph is not None else self.config.max_proposals_per_iteration,
                    self._remaining_mutation_searches(default=self.config.max_proposals_per_iteration),
                ),
            )
            self._mutation_searches_used += len(candidates)
            self.graph_archive.record_candidate_audit(
                {
                    "iteration": iteration,
                    "task_ids": [item.task.task_id for item in failures],
                    "failure_types": failure_types,
                    "stage": "mutation_search_budget",
                    "status": (
                        "exhausted" if self._mutation_search_budget_exhausted() else "active"
                    ),
                    "selected_graph_ids": [candidate.graph.graph_id for candidate in candidates],
                    "searches_consumed": len(candidates),
                    "searches_used": self._mutation_searches_used,
                    "searches_remaining": self._remaining_mutation_searches(),
                    "search_limit": self.config.max_mutation_searches,
                    "reward_objective": self._reward_objective(failures),
                }
            )
            selected_graph_ids = {candidate.graph.graph_id for candidate in candidates}
            for candidate in novel_candidates:
                selection = candidate.graph.stats.get("portfolio_selection", {})
                selected = candidate.graph.graph_id in selected_graph_ids
                eligible = selection.get("eligible", True)
                self.graph_archive.record_candidate_audit(
                    {
                        "iteration": iteration,
                        "task_ids": [item.task.task_id for item in failures],
                        "failure_types": failure_types,
                        "stage": "candidate_portfolio",
                        "graph_id": candidate.graph.graph_id,
                        "status": "selected" if selected else ("deferred" if eligible else "rejected"),
                        "selection": selection,
                        "reason": (
                            "selected for validation"
                            if selected
                            else selection.get("rejection_reason")
                            if not eligible
                            else "deferred by candidate budget after mechanism-family diversification"
                        ),
                    }
                )
            deferred_candidates = sorted(
                [
                    candidate
                    for candidate in novel_candidates
                    if candidate.graph.graph_id not in selected_graph_ids
                    and candidate.graph.stats.get("portfolio_selection", {}).get("eligible", True)
                ],
                key=lambda candidate: (
                    float(
                        candidate.graph.stats.get("portfolio_selection", {}).get("score", 0.0)
                    ),
                    candidate.graph.graph_id,
                ),
                reverse=True,
            )
            proposed: list[str] = []
            accepted: list[str] = []
            exploratory: list[str] = []
            rejected: list[str] = []
            parent_evaluation = self.evaluate_program(parent, self.dataset.validation)
            parent.metrics = parent_evaluation.metrics
            self.registry.upsert(parent)
            runtime_candidates = self._runtime_replenished_candidates(
                candidates,
                deferred_candidates,
                iteration=iteration,
                failures=failures,
            )
            for index, candidate in enumerate(runtime_candidates, start=1):
                versioned = self._version_candidate(candidate, iteration, index)
                self._graphs[versioned.graph.skill_name] = versioned.graph
                child_name = f"iter-{iteration:03d}-graph-{index:02d}"
                child = GraphProgram(
                    name=child_name,
                    parent=parent.name,
                    graph_ids=list(dict.fromkeys(parent.graph_ids + [versioned.graph.skill_name])),
                    iteration=iteration,
                    proposal=versioned.reason,
                    justification="; ".join(edit.reason for edit in versioned.edits if edit.reason),
                    mutation=[asdict(edit) for edit in versioned.edits],
                )
                self.registry.create(child)
                proposed.append(child_name)
                validation_tasks = self._candidate_validation_tasks(versioned)
                if not validation_tasks:
                    rejection_reason = "no_task_class_matched_validation_tasks"
                    child.metrics = ProgramMetrics(
                        execution_coverage=0.0,
                        candidate_tool_coverage=0.0,
                    )
                    child.status = "rejected"
                    self.registry.upsert(child)
                    rejected.append(child_name)
                    versioned.graph.stats.update(
                        {
                            "program": child_name,
                            "parent_program": parent.name,
                            "accepted": False,
                            "admission": "rejected",
                            "rejection_reason": rejection_reason,
                            "validation_task_ids": [],
                        }
                    )
                    self.graph_memory.upsert_graph(versioned.graph)
                    self.graph_archive.record_graph(
                        versioned.graph,
                        stage="candidate_validation",
                        status="rejected",
                        iteration=iteration,
                        program=child_name,
                        parent_program=parent.name,
                        metadata={
                            "failure_types": failure_types,
                            "proposal": versioned.reason,
                            "rejection_reason": rejection_reason,
                        },
                    )
                    self.graph_archive.record_candidate_audit(
                        {
                            "iteration": iteration,
                            "task_ids": [rollout.task.task_id for rollout in failures],
                            "failure_types": failure_types,
                            "stage": "candidate_validation",
                            "graph_id": versioned.graph.graph_id,
                            "status": "rejected",
                            "reason": rejection_reason,
                        }
                    )
                    continue
                parent_tools = {
                    tool
                    for graph_id in parent.graph_ids
                    for tool in (self._graphs.get(graph_id).tool_names() if self._graphs.get(graph_id) else [])
                }
                required_candidate_tools = set(versioned.graph.tool_names()).difference(parent_tools)
                candidate_parent_evaluation = self.evaluate_program(parent, validation_tasks)
                forced_evaluation = self.evaluate_program(
                    child,
                    validation_tasks,
                    forced_graph_id=versioned.graph.skill_name,
                    required_tool_names=required_candidate_tools,
                )
                forced_passes_gate, forced_gate_evidence = self._passes_validation_gate(
                    candidate_parent_evaluation,
                    forced_evaluation,
                )
                self._set_candidate_routing_scope(
                    versioned,
                    candidate_parent_evaluation,
                    forced_evaluation,
                )
                # The forced rollout discovers where a path helps. Admission is
                # decided only after the production router can select that path
                # and preserve the parent route on incompatible task profiles.
                evaluation = self.evaluate_program(
                    child,
                    validation_tasks,
                    required_tool_names=required_candidate_tools,
                )
                passes_gate, gate_evidence = self._passes_validation_gate(
                    candidate_parent_evaluation,
                    evaluation,
                )
                routed_evaluation = self.evaluate_program(child, self.dataset.validation)
                child.metrics = routed_evaluation.metrics
                gain = float(gate_evidence["mean_paired_gain"])
                exploratory_gate = False
                if not passes_gate and self.config.exploratory_admission_enabled:
                    exploratory_gate, exploratory_evidence = self._passes_exploratory_gate(
                        candidate_parent_evaluation,
                        evaluation,
                    )
                    gate_evidence.update(exploratory_evidence)
                gate_evidence["strict_gate_passed"] = passes_gate
                gate_evidence["exploratory_gate_passed"] = exploratory_gate
                gate_evidence["post_routing_validation"] = True
                gate_evidence["forced_probe_passed"] = forced_passes_gate
                gate_evidence["forced_probe"] = forced_gate_evidence
                gate_evidence["forced_probe_quality"] = forced_evaluation.metrics.quality
                gate_evidence["normal_route_quality"] = evaluation.metrics.quality
                gate_evidence["search_reward"] = gain
                gate_evidence["reward_objective"] = self._reward_objective(
                    evaluation.rollouts
                )
                validation_credit_allowed = passes_gate or exploratory_gate
                self._record_candidate_validation_credit(
                    versioned.graph,
                    validation_tasks,
                    evaluation,
                    positive_credit_allowed=validation_credit_allowed,
                )
                # Accepted quality evidence may inform later routing. Rejected
                # candidates stay deferred so cached or diagnostic rollouts do
                # not accidentally earn positive graph credit later.
                versioned.graph.stats["defer_weighted_credit"] = not validation_credit_allowed
                child.status = "validated" if passes_gate else ("exploratory" if exploratory_gate else "rejected")
                self.registry.upsert(child)
                admitted = (passes_gate or exploratory_gate) and self.registry.update_frontier(
                    child.name, self.config.frontier_size
                )
                outcome = (
                    "accepted" if admitted and passes_gate
                    else "exploratory" if admitted and exploratory_gate
                    else "dominated" if passes_gate or exploratory_gate
                    else "rejected"
                )
                versioned.graph.stats.update(
                    {
                        "program": child_name,
                        "parent_program": parent.name,
                        "quality_gain": gain,
                        "candidate_score": evaluation.metrics.quality,
                        "routed_validation_score": child.metrics.quality,
                        "pass_rate": evaluation.metrics.pass_rate,
                        "stability": evaluation.metrics.stability,
                        "estimated_cost": evaluation.metrics.estimated_cost,
                        "accepted": outcome == "accepted",
                        "exploratory": outcome == "exploratory",
                        "admission": outcome,
                        "validation_task_ids": sorted({summary.task_id for summary in evaluation.rollouts}),
                        "passed_validation_task_ids": sorted({
                            summary.task_id for summary in evaluation.rollouts if summary.passed
                        }),
                        "validation_seeds": sorted({
                            summary.evaluation_seed
                            for summary in evaluation.rollouts
                            if summary.evaluation_seed is not None
                        }),
                        "gate_evidence": gate_evidence,
                    }
                )
                followup_reason = self._candidate_followup_reason(gate_evidence)
                if (self.config.near_miss_repair_enabled and self._h3_active()
                        and outcome == "rejected" and gain > 0
                        and followup_reason is not None
                        and evaluation.metrics.execution_error_count == 0
                        and gate_evidence.get("paired_coverage") == 1.0
                        and gate_evidence.get("h3_replicate_gate_passed") is True
                        and repair_graph is None):
                    versioned.graph.stats["near_miss_repair"] = {
                        "status": "pending", "attempts": 0,
                        "training_task_ids": [r.task.task_id for r in failures],
                        "gain": gain,
                        "reason": followup_reason,
                    }
                    self.graph_archive.record_candidate_audit({
                        "iteration": iteration, "stage": "near_miss_repair",
                        "graph_id": versioned.graph.graph_id, "status": "pending",
                        "reason": followup_reason,
                    })
                if admitted and passes_gate:
                    accepted.append(child_name)
                    versioned.graph.stats.update(
                        {
                            "program": child_name,
                            "parent_program": parent.name,
                            "quality_gain": gain,
                            "candidate_score": evaluation.metrics.quality,
                            "routed_validation_score": child.metrics.quality,
                            "pass_rate": evaluation.metrics.pass_rate,
                            "stability": evaluation.metrics.stability,
                            "estimated_cost": evaluation.metrics.estimated_cost,
                            "accepted": True,
                            "validation_task_ids": sorted({summary.task_id for summary in evaluation.rollouts}),
                            "passed_validation_task_ids": sorted({summary.task_id for summary in evaluation.rollouts if summary.passed}),
                            "validation_seeds": sorted({summary.evaluation_seed for summary in evaluation.rollouts if summary.evaluation_seed is not None}),
                            "gate_evidence": gate_evidence,
                        }
                    )
                    skill = versioned.graph.to_skill_card()
                    skill.validation = SkillValidationReport(
                        skill_name=skill.skill_name,
                        validation_task_ids=[summary.task_id for summary in evaluation.rollouts],
                        baseline_score=candidate_parent_evaluation.metrics.quality,
                        candidate_score=evaluation.metrics.quality,
                        baseline_pass_rate=candidate_parent_evaluation.metrics.pass_rate,
                        candidate_pass_rate=evaluation.metrics.pass_rate,
                        quality_gain=gain,
                        estimated_cost=evaluation.metrics.estimated_cost,
                        accepted=True,
                        evidence=[
                            f"program:{child_name}",
                            f"parent:{parent.name}",
                            *failure_types,
                        ],
                    )
                    self.evolver.skill_memory.upsert(skill)
                    task_by_id = {task.task_id: task for task in self.dataset.validation}
                    for summary in evaluation.rollouts:
                        task = task_by_id.get(summary.task_id)
                        if task is None:
                            continue
                        self.graph_memory.append_experience(
                            ExperienceRecord(
                                task_id=task.task_id,
                                prompt=task.prompt,
                                failure_types=summary.failure_types,
                                graph_id=versioned.graph.skill_name,
                                tool_path=summary.tool_chain,
                                score=summary.score,
                                cost=summary.estimated_cost,
                                accepted=True,
                                evidence=[
                                    f"{name}={score:.3f}"
                                    for name, score in summary.metric_scores.items()
                                ],
                            )
                        )
                elif admitted and exploratory_gate:
                    exploratory.append(child_name)
                    versioned.graph.stats.update(
                        {
                            "program": child_name,
                            "parent_program": parent.name,
                            "quality_gain": gain,
                            "candidate_score": evaluation.metrics.quality,
                            "routed_validation_score": child.metrics.quality,
                            "pass_rate": evaluation.metrics.pass_rate,
                            "stability": evaluation.metrics.stability,
                            "estimated_cost": evaluation.metrics.estimated_cost,
                            "accepted": False,
                            "exploratory": True,
                            "admission": "exploratory_frontier",
                            "validation_task_ids": sorted({summary.task_id for summary in evaluation.rollouts}),
                            "validation_seeds": sorted({
                                summary.evaluation_seed
                                for summary in evaluation.rollouts
                                if summary.evaluation_seed is not None
                            }),
                            "gate_evidence": gate_evidence,
                        }
                    )
                else:
                    rejected.append(child_name)
                self.graph_memory.upsert_graph(versioned.graph)
                self.graph_archive.record_graph(
                    versioned.graph,
                    stage="candidate_validation",
                    status=outcome,
                    iteration=iteration,
                    program=child_name,
                    parent_program=parent.name,
                    metadata={
                        "quality": evaluation.metrics.quality,
                        "routed_validation_quality": child.metrics.quality,
                        "quality_gain": gain,
                        "pass_rate": evaluation.metrics.pass_rate,
                        "stability": evaluation.metrics.stability,
                        "estimated_cost": evaluation.metrics.estimated_cost,
                        "failure_types": failure_types,
                        "proposal": versioned.reason,
                        "gate_evidence": gate_evidence,
                    },
                )
                self.feedback.append(
                    EvolutionFeedback(
                        iteration=iteration,
                        parent_program=parent.name,
                        child_program=child_name,
                        task_ids=[rollout.task.task_id for rollout in failures],
                        failure_types=failure_types,
                        proposal=versioned.reason,
                        justification=child.justification,
                        outcome=outcome,
                        parent_score=candidate_parent_evaluation.metrics.quality,
                        child_score=evaluation.metrics.quality,
                        graph_id=versioned.graph.skill_name,
                        evidence=[
                            metric.name
                            for rollout in failures
                            for metric in rollout.evaluation.failed_metrics
                        ],
                        discovery_task_ids=[
                            rollout.task.task_id for rollout in failures
                        ],
                        validation_task_ids=sorted({
                            summary.task_id for summary in evaluation.rollouts
                        }),
                    )
                )

            if accepted or exploratory:
                no_improvement = 0
                status = "frontier_updated" if accepted else "exploratory_frontier_updated"
            else:
                no_improvement += 1
                status = "no_candidate_improved" if candidates else "no_novel_candidate"
            promoted = self._refresh_online_foundations(iteration, parent)
            if promoted:
                no_improvement = 0
                status = "frontier_and_foundation_updated" if accepted else "foundation_promoted"
            if self._mutation_search_budget_exhausted():
                status = f"{status}_mutation_budget_exhausted"
            record = EvolutionIteration(
                iteration,
                parent.name,
                [task.task_id for task in sampled],
                failure_types,
                proposed,
                accepted,
                rejected,
                [program.name for program in self.registry.frontier()],
                status,
                promoted,
                exploratory_programs=exploratory,
                mutation_searches_used=self._mutation_searches_used,
                mutation_searches_remaining=self._remaining_mutation_searches(),
            )
            iteration_records.append(record)
            self._append_iteration(record)
            self._save_checkpoint(iteration, no_improvement)
            if (
                no_improvement >= self.config.no_improvement_limit
                or self._mutation_search_budget_exhausted()
            ):
                break

        best = self.registry.best()
        if best is None or best.metrics is None:
            raise RuntimeError("evolution completed without a scored program")
        validation = self.evaluate_program(best, self.dataset.validation)
        test_evaluation = self.evaluate_program(best, self.dataset.test) if self.dataset.test else None
        full_baseline_evaluation = None
        full_final_evaluation = None
        if self.config.full_suite_evaluation:
            full_tasks = self._all_dataset_tasks()
            baseline = self.registry.get("base")
            full_baseline_evaluation = self.evaluate_program(baseline, full_tasks)
            full_final_evaluation = (
                full_baseline_evaluation
                if best.name == baseline.name
                else self.evaluate_program(best, full_tasks)
            )
        visualization_paths = self.graph_archive.export(
            programs=self.registry.list_programs(),
            feedback=self.feedback.entries(),
            weighted_categories=(
                self.weighted_tool_graph.categories_summary(self.evolver.tools)
                if self.weighted_tool_graph is not None
                else {}
            ),
            frontier=[program.name for program in self.registry.frontier()],
        )
        visualization_paths.update(
            self.graph_archive.export_video_comparisons(
                self._all_dataset_tasks(),
                top_k=max(1, int(os.environ.get("EVOVIDEO_COMPARISON_TOP_K", "5"))),
            )
        )
        result = EvolutionLoopResult(
            best_program=best.name,
            best_metrics=best.metrics,
            frontier=[program.name for program in self.registry.frontier()],
            iterations=iteration_records,
            dataset=self.dataset.to_dict(),
            program_root=str(self.registry.root),
            feedback_path=str(self.feedback.jsonl_path),
            cache_dir=str(self.cache.cache_dir),
            validation_evaluation=validation,
            test_evaluation=test_evaluation,
            full_baseline_evaluation=full_baseline_evaluation,
            full_final_evaluation=full_final_evaluation,
            mutation_searches_used=self._mutation_searches_used,
            max_mutation_searches=self.config.max_mutation_searches,
            mutation_search_budget_exhausted=self._mutation_search_budget_exhausted(),
            visualization_paths=visualization_paths,
        )
        self._write_json(self.state_dir / "loop_result.json", result.to_dict())
        return result

    def _all_dataset_tasks(self) -> list[VideoTask]:
        tasks: dict[str, VideoTask] = {}
        for task in [*self.dataset.train, *self.dataset.validation, *self.dataset.test]:
            tasks.setdefault(task.task_id, task)
        return list(tasks.values())

    def evaluate_program(
        self,
        program: GraphProgram,
        tasks: list[VideoTask],
        *,
        forced_graph_id: str | None = None,
        required_tool_names: set[str] | None = None,
        verifier_gated: bool = False,
    ) -> ProgramEvaluation:
        evaluation_tasks = self._evaluation_variants(tasks)
        summaries: list[TaskRolloutSummary] = []
        for task in evaluation_tasks:
            if verifier_gated and forced_graph_id is None and program.name != "base":
                summaries.append(self._verifier_gated_rollout_summary(program, task))
                continue
            graph = self._graphs.get(forced_graph_id) if forced_graph_id else self._select_graph(program, task)
            if graph is None:
                raise RuntimeError(f"forced evaluation graph is unavailable: {forced_graph_id}")
            blocked_tool = self._circuit_broken_tool(graph)
            if blocked_tool is not None:
                summary = TaskRolloutSummary(
                    task_id=task.task_id,
                    graph_id=graph.skill_name,
                    score=0.0,
                    passed=False,
                    failure_types=["tool_execution_error"],
                    metric_scores={},
                    estimated_cost=graph.estimated_cost(),
                    tool_chain=graph.tool_names(),
                    evaluation_seed=self._evaluation_label(task),
                    seed_controlled=False,
                    provider_seed_control=self._h3_local() if self._h3_active() else None,
                    replicate_label=self._evaluation_label(task) if self._h3_active() else None,
                    evaluation_protocol=self._h3_summary_protocol() if self._h3_active() else None,
                    execution_error=(
                        f"runtime circuit breaker skipped tool {blocked_tool!r} after a prior fatal failure: "
                        f"{self._runtime_tool_failures[blocked_tool]}"
                    ),
                    failed_tool_name=blocked_tool,
                )
                self._archive_execution(summary)
                self._record_failed_path_evidence(task, graph, summary)
            else:
                summary = self._rollout_summary(task, graph)
                self._record_runtime_tool_failure(graph, summary)
            summaries.append(summary)
        metrics = self._metrics_from_summaries(summaries, required_tool_names)
        return ProgramEvaluation(program.name, metrics, summaries)

    def _verifier_gated_rollout_summary(
        self,
        program: GraphProgram,
        task: VideoTask,
    ) -> TaskRolloutSummary:
        """Repair only observed baseline failures and retain the stronger artifact."""
        baseline_graph = self.evolver.baseline_graph()
        baseline = self._rollout_summary(task, baseline_graph)
        decision: dict[str, Any] = {
            "task_id": task.task_id,
            "evaluation_seed": self._evaluation_label(task),
            "program": program.name,
            "baseline_graph_id": baseline.graph_id,
            "baseline_score": baseline.score,
            "baseline_passed": baseline.passed,
            "baseline_failure_types": list(baseline.failure_types),
            "baseline_artifact_path": baseline.artifact_path,
            "candidate_graph_id": None,
            "candidate_score": None,
            "candidate_passed": None,
            "candidate_failure_types": [],
            "candidate_artifact_path": None,
            "accepted_candidate": False,
            "reused_baseline_artifact": False,
        }
        if baseline.execution_error is None and baseline.passed:
            decision["reason"] = "baseline_passed_verifier_gate"
            decision["selected_graph_id"] = baseline.graph_id
            decision["selected_score"] = baseline.score
            decision["selected_artifact_path"] = baseline.artifact_path
            self._record_runtime_decision(decision)
            return baseline

        observed_failures = {
            str(value).strip().lower()
            for value in baseline.failure_types
            if value and value != "tool_execution_error"
        }
        routed_task = self._task_with_observed_failures(task, observed_failures)
        graph = self._select_graph(
            program,
            routed_task,
            observed_failure_types=observed_failures or None,
        )
        if graph.skill_name == baseline_graph.skill_name:
            decision["reason"] = "no_specialized_graph_for_observed_failure"
            decision["selected_graph_id"] = baseline.graph_id
            decision["selected_score"] = baseline.score
            decision["selected_artifact_path"] = baseline.artifact_path
            self._record_runtime_decision(decision)
            return baseline

        candidate_task, candidate_graph, reused_baseline = self._condition_repair_on_baseline(
            routed_task,
            graph,
            baseline,
        )
        candidate = self._rollout_summary(candidate_task, candidate_graph)
        candidate.estimated_cost += baseline.estimated_cost
        decision.update(
            {
                "candidate_graph_id": graph.skill_name,
                "candidate_score": candidate.score,
                "candidate_passed": candidate.passed,
                "candidate_failure_types": list(candidate.failure_types),
                "candidate_artifact_path": candidate.artifact_path,
                "candidate_execution_error": candidate.execution_error,
                "reused_baseline_artifact": reused_baseline,
            }
        )
        margin = max(
            0.0,
            float(os.environ.get("EVOVIDEO_RUNTIME_REPAIR_MIN_GAIN", "0.02")),
        )
        new_failures = set(candidate.failure_types).difference(baseline.failure_types)
        improved = candidate.score > baseline.score + margin + 1e-12
        accept = (
            candidate.execution_error is None
            and improved
            and not new_failures
        )
        if accept:
            decision["accepted_candidate"] = True
            decision["reason"] = "candidate_improved_without_new_failure"
            decision["selected_graph_id"] = graph.skill_name
            decision["selected_score"] = candidate.score
            decision["selected_artifact_path"] = candidate.artifact_path
            self._record_runtime_decision(decision)
            return candidate

        selected = TaskRolloutSummary(**asdict(baseline))
        selected.estimated_cost = candidate.estimated_cost
        decision["selected_graph_id"] = baseline.graph_id
        decision["selected_score"] = baseline.score
        decision["selected_artifact_path"] = baseline.artifact_path
        if candidate.execution_error is not None:
            decision["reason"] = "candidate_execution_failed_fallback_baseline"
        elif new_failures:
            decision["reason"] = "candidate_introduced_new_failure_fallback_baseline"
        else:
            decision["reason"] = "candidate_gain_below_runtime_margin_fallback_baseline"
        self._record_runtime_decision(decision)
        return selected

    @staticmethod
    def _task_with_observed_failures(
        task: VideoTask,
        observed_failures: set[str],
    ) -> VideoTask:
        metadata = dict(task.metadata or {})
        if observed_failures:
            metadata["expected_failure_modes"] = sorted(observed_failures)
            metadata["runtime_observed_failure_modes"] = sorted(observed_failures)
        return VideoTask(
            task_id=task.task_id,
            prompt=task.prompt,
            mode=task.mode,
            duration_seconds=task.duration_seconds,
            reference_video=task.reference_video,
            metadata=metadata,
        )

    def _condition_repair_on_baseline(
        self,
        task: VideoTask,
        graph: ToolPathGraph,
        baseline: TaskRolloutSummary,
    ) -> tuple[VideoTask, ToolPathGraph, bool]:
        if self._h3_active():
            return task, graph, False
        if os.environ.get("EVOVIDEO_BASELINE_CONDITIONED_REPAIR", "1") == "0":
            return task, graph, False
        if not baseline.artifact_path or not Path(baseline.artifact_path).is_file():
            return task, graph, False
        if not self.evolver.tools.has("task_reference_video"):
            return task, graph, False

        t2v_nodes = [
            node for node in graph.nodes
            if node.node_type == "tool" and node.name == "mock_text_to_video"
        ]
        if not t2v_nodes:
            return task, graph, False
        t2v_node = t2v_nodes[0]
        descendants = self._graph_descendants(graph, t2v_node.node_id)
        has_repair_consumer = any(
            node.node_id in descendants
            and node.node_type == "tool"
            and self.evolver.tools.has(node.name)
            and self.evolver.tools.spec(node.name).consumes_upstream
            for node in graph.nodes
        )
        if not has_repair_consumer:
            return task, graph, False

        conditioned = ToolPathGraph.from_dict(graph.to_dict())
        for node in conditioned.nodes:
            if node.node_id == t2v_node.node_id:
                node.name = "task_reference_video"
                node.config = {**node.config, "cost": 0.05, "runtime_baseline_source": True}
        conditioned.edges = [
            edge for edge in conditioned.edges
            if edge.target != t2v_node.node_id
        ]
        conditioned.stats = {
            **conditioned.stats,
            "runtime_baseline_conditioned": True,
            "source_graph_id": graph.skill_name,
        }
        metadata = {
            **(task.metadata or {}),
            "runtime_baseline_conditioned": True,
            "runtime_baseline_artifact_path": baseline.artifact_path,
        }
        conditioned_task = VideoTask(
            task_id=task.task_id,
            prompt=task.prompt,
            mode=task.mode,
            duration_seconds=task.duration_seconds,
            reference_video=baseline.artifact_path,
            metadata=metadata,
        )
        return conditioned_task, conditioned, True

    @staticmethod
    def _graph_descendants(graph: ToolPathGraph, source_id: str) -> set[str]:
        outgoing: dict[str, list[str]] = {}
        for edge in graph.edges:
            outgoing.setdefault(edge.source, []).append(edge.target)
        descendants: set[str] = set()
        frontier = list(outgoing.get(source_id, []))
        while frontier:
            node_id = frontier.pop()
            if node_id in descendants:
                continue
            descendants.add(node_id)
            frontier.extend(outgoing.get(node_id, []))
        return descendants

    def _record_runtime_decision(self, decision: dict[str, Any]) -> None:
        payload = {
            **decision,
            "runtime_session_id": getattr(self, "runtime_session_id", "unspecified"),
            "created_at": utc_now(),
        }
        self.runtime_decisions.append(payload)
        self.runtime_decision_path.parent.mkdir(parents=True, exist_ok=True)
        with self.runtime_decision_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _metrics_from_summaries(
        summaries: list[TaskRolloutSummary],
        required_tool_names: set[str] | None = None,
    ) -> ProgramMetrics:
        valid = [summary for summary in summaries if summary.execution_error is None]
        # Runtime failures are zero-reward observations, not missing data. If
        # they are dropped, a routed program can inherit unrelated cached
        # baseline scores and appear stronger than the path that actually ran.
        scores = [summary.score if summary.execution_error is None else 0.0 for summary in summaries]
        required_tools = set(required_tool_names or ())
        executed_tools = {
            tool for summary in valid for tool in summary.tool_chain
        }
        candidate_tool_coverage = (
            len(required_tools.intersection(executed_tools)) / len(required_tools)
            if required_tools else 1.0
        )
        metrics = ProgramMetrics(
            quality=sum(scores) / max(1, len(scores)),
            pass_rate=sum(1 for summary in summaries if summary.passed) / max(1, len(summaries)),
            stability=graph_score_stability(scores),
            estimated_cost=sum(summary.estimated_cost for summary in summaries) / max(1, len(summaries)),
            task_count=len({summary.task_id for summary in summaries}),
            cache_hits=sum(1 for summary in summaries if summary.cache_hit),
            sample_count=len(summaries),
            execution_coverage=len(valid) / max(1, len(summaries)),
            execution_error_count=len(summaries) - len(valid),
            candidate_tool_coverage=candidate_tool_coverage,
            reward_objectives=sorted({summary.reward_objective for summary in valid}),
            reward_component_means=GraphSelfImprovingLoop._reward_component_means(valid),
        )
        return metrics

    @staticmethod
    def _reward_component_means(
        summaries: list[TaskRolloutSummary],
    ) -> dict[str, float]:
        values: dict[str, list[float]] = {}
        for summary in summaries:
            for name, score in summary.reward_components.items():
                values.setdefault(name, []).append(float(score))
        return {
            name: sum(scores) / len(scores)
            for name, scores in sorted(values.items())
            if scores
        }

    @staticmethod
    def _reward_objective(items: list[Any]) -> str:
        objectives: set[str] = set()
        for item in items:
            direct = getattr(item, "reward_objective", None)
            reward = getattr(item, "reward", None)
            value = direct or getattr(reward, "objective", None)
            if value:
                objectives.add(str(value))
        if not objectives:
            return "video_quality"
        if len(objectives) == 1:
            return next(iter(objectives))
        return "mixed_task_conditioned_reward"

    def _merge_routed_evaluation(
        self,
        program_name: str,
        baseline: ProgramEvaluation,
        candidate: ProgramEvaluation,
    ) -> ProgramEvaluation:
        """Build full-validation metrics without rerunning unrelated families."""
        replacements = {
            (summary.task_id, summary.evaluation_seed): summary
            for summary in candidate.rollouts
        }
        merged = [
            replacements.get((summary.task_id, summary.evaluation_seed), summary)
            for summary in baseline.rollouts
        ]
        return ProgramEvaluation(
            program_name,
            self._metrics_from_summaries(merged),
            merged,
        )

    def _circuit_broken_tool(self, graph: ToolPathGraph) -> str | None:
        if not self.config.runtime_failure_circuit_breaker:
            return None
        failures = getattr(self, "_runtime_tool_failures", {})
        return next(
            (name for name in graph.tool_names() if name in failures),
            None,
        )

    def _record_runtime_tool_failure(
        self,
        graph: ToolPathGraph,
        summary: TaskRolloutSummary,
    ) -> None:
        if not self.config.runtime_failure_circuit_breaker or not summary.execution_error:
            return
        lowered = summary.execution_error.lower()
        fatal_markers = (
            "timed out after",
            "cuda out of memory",
            "outofmemoryerror",
            "no space left on device",
            "fatal runtime diagnostic",
            "cuda extension architecture mismatch",
            "cuda kernel is incompatible",
            "native extension abi mismatch",
            "was built for sm",
            "no kernel image",
            "invalid device function",
            "undefined symbol",
            "glibcxx",
            "external api credential",
            "missing authentication header",
            "api_key is required",
            "api key is required",
            "hugging face model revision does not exist",
            "hugging face model identifier is invalid",
            "revisionnotfounderror",
            "revision not found",
            "invalid rev id",
            "model checkpoint archive is corrupted",
            "model checkpoint tensor archive is corrupted",
            "pytorchstreamreader failed reading file",
            "invalid header or archive is corrupted",
            "failed finding central directory",
            "safetensorerror",
        )
        if not any(marker in lowered for marker in fatal_markers):
            return
        failed_tool = summary.failed_tool_name or self._infer_failed_tool(graph, summary.execution_error)
        if failed_tool is None or not self.evolver.tools.has(failed_tool):
            return
        if self.evolver.tools.spec(failed_tool).backend in {"venv", "micromamba", "docker", "mcp"}:
            self._runtime_tool_failures.setdefault(failed_tool, summary.execution_error[-2000:])

    @staticmethod
    def _infer_failed_tool(graph: ToolPathGraph, execution_error: str) -> str | None:
        """Conservatively attribute legacy unstructured errors to one named tool."""
        tool_names = list(dict.fromkeys(graph.tool_names()))
        matches = [
            name
            for name in tool_names
            if repr(name) in execution_error or f"tool {name}" in execution_error
        ]
        if len(matches) == 1:
            return matches[0]
        return tool_names[0] if len(tool_names) == 1 else None

    def _initialize_or_resume(self) -> tuple[int, int]:
        if self.config.continue_mode and self.registry.frontier():
            checkpoint = self._read_json(self.checkpoint_path, {})
            current_verifier = self._verifier_protocol()
            saved_verifier = checkpoint.get("verifier_protocol")
            if ((saved_verifier is not None and saved_verifier != current_verifier)
                    or (saved_verifier is None and current_verifier.get("vlm_provider") in {"gemini", "qwen_video"})):
                raise ValueError("Verifier protocol changed or legacy checkpoint has no verifier identity. "
                                 "Use a new output directory to re-evaluate baseline and candidates; "
                                 "do not mix scores from different verifier protocols with --continue.")
            self.sampler.load_state_dict(checkpoint.get("sampler", {}))
            self._exploration_visits = dict(checkpoint.get("exploration_visits", {}))
            self._mutation_searches_used = int(
                checkpoint.get(
                    "mutation_searches_used",
                    self._mutation_searches_from_log(),
                )
            )
            checkpoint_iteration = int(checkpoint.get("iteration", 0))
            no_improvement = int(checkpoint.get("no_improvement", 0))
            recovered = [
                *self._revalidate_stale_boundary_rejections(checkpoint_iteration),
                *self._revalidate_profile_routing_rejections(checkpoint_iteration),
            ]
            if recovered:
                no_improvement = 0
                self._save_checkpoint(checkpoint_iteration, no_improvement)
            return checkpoint_iteration + 1, no_improvement

        self._mutation_searches_used = 0
        self._exploration_visits = {}
        self.registry.reset()
        for path in (
            self.checkpoint_path,
            self.iteration_log_path,
            self.feedback.jsonl_path,
            self.feedback.markdown_path,
            self.foundation_log_path,
            self.state_dir / "loop_result.json",
        ):
            if path.exists():
                path.unlink()
        self.feedback.jsonl_path.touch()
        self.feedback.markdown_path.touch()
        self.foundation_log_path.touch()
        if self.weighted_tool_graph is not None:
            self.weighted_tool_graph.reset()
        self.graph_archive.reset()
        baseline_graph = self.evolver.baseline_graph()
        self.graph_memory.upsert_graph(baseline_graph)
        self._graphs[baseline_graph.skill_name] = baseline_graph
        baseline = GraphProgram(
            name="base",
            graph_ids=[baseline_graph.skill_name],
            iteration=0,
            proposal="single-call text-to-video baseline",
            status="baseline",
        )
        self.registry.create(baseline)
        evaluation = self.evaluate_program(baseline, self.dataset.validation)
        baseline.metrics = evaluation.metrics
        self.registry.upsert(baseline)
        self.registry.update_frontier("base", self.config.frontier_size)
        self.graph_archive.record_graph(
            baseline_graph,
            stage="baseline",
            status="frontier",
            iteration=0,
            program="base",
            metadata={"quality": evaluation.metrics.quality, "pass_rate": evaluation.metrics.pass_rate},
        )
        self._initialize_warm_start_program(baseline)
        return 1, 0

    def _candidate_followup_reason(self, evidence: dict[str, Any]) -> str | None:
        if evidence.get("task_metric_guard_passed") is False:
            return "positive paired gain with task-metric regression; retained for one repair search"
        if (evidence.get("task_metric_guard_passed") is True
                and evidence.get("mean_paired_gain", 0) >= self.config.min_quality_gain
                and evidence.get("paired_improvement_fraction", 1) < .5
                and evidence.get("paired_regression_fraction", 1) == 0
                and evidence.get("candidate_tool_coverage") == 1.0
                and evidence.get("candidate_pass_rate", 0) >= evidence.get("baseline_pass_rate", 0)):
            return ("positive gain with ties and no observed regression; evidence is insufficient for admission; "
                    "retained for one training-only robustness search and fresh candidate validation")
        return None

    def _next_near_miss_repair(self, iteration: int) -> tuple[ToolPathGraph | None, list[VideoTask]]:
        """Use rejected paths as proposal parents, never as accepted production routes."""
        if not self.config.near_miss_repair_enabled or not self._h3_active():
            return None, []
        # Cover the pilot's three families before spending a slot on repair.
        if iteration <= min(3, len({task_class(t) for t in self.dataset.train})):
            return None, []
        for graph in sorted(self._graphs.values(), key=lambda g: g.stats.get("quality_gain", 0), reverse=True):
            repair = graph.stats.get("near_miss_repair", {})
            if repair.get("status") != "pending" or repair.get("attempts", 0) >= 1:
                continue
            tasks = [task for task in self.dataset.train if task.task_id in repair.get("training_task_ids", [])]
            if not tasks:
                continue
            repair.update(status="attempted", attempts=1, iteration=iteration)
            self.graph_memory.upsert_graph(graph)
            self.graph_archive.record_candidate_audit({
                "iteration": iteration, "stage": "near_miss_repair", "status": "attempted",
                "graph_id": graph.graph_id, "task_ids": [t.task_id for t in tasks],
                "reason": "repair proposal uses training tasks; admission remains against the accepted parent",
            })
            return graph, tasks
        return None, []

    def _initialize_warm_start_program(self, baseline: GraphProgram) -> None:
        graph_ids = [
            graph_id
            for graph_id in self.warm_start_graph_ids
            if graph_id != baseline.graph_ids[0] and graph_id in self._graphs
        ]
        if not graph_ids:
            return
        program = GraphProgram(
            name="warm-start",
            parent=baseline.name,
            graph_ids=list(dict.fromkeys([*baseline.graph_ids, *graph_ids])),
            iteration=0,
            proposal="Transfer the validated frontier from the nested smaller benchmark",
            justification="Re-evaluate imported graph paths on this benchmark's held-out validation split",
            mutation=[{
                "op": "warm_start_graph_lineage",
                "target": None,
                "payload": {"graph_ids": graph_ids},
                "reason": "reuse validated tool-path evolution across nested benchmark scales",
            }],
        )
        self.registry.create(program)
        evaluation = self.evaluate_program(program, self.dataset.validation)
        program.metrics = evaluation.metrics
        self.registry.upsert(program)
        admitted = self.registry.update_frontier(program.name, self.config.frontier_size)
        for graph_id in graph_ids:
            graph = self._graphs[graph_id]
            self.graph_archive.record_graph(
                graph,
                stage="cross_scale_warm_start",
                status="frontier" if admitted else "archived",
                iteration=0,
                program=program.name,
                parent_program=baseline.name,
                metadata={
                    "validation_quality": evaluation.metrics.quality,
                    "validation_pass_rate": evaluation.metrics.pass_rate,
                    "source": "nested_smaller_benchmark_frontier",
                },
            )
        print(
            "[evolution] cross-scale warm start "
            f"graphs={len(graph_ids)} quality={evaluation.metrics.quality:.4f} "
            f"admitted={admitted}",
            flush=True,
        )

    def _revalidate_stale_boundary_rejections(self, iteration: int) -> list[str]:
        """Replay candidates rejected only by pre-tolerance floating-point noise.

        The replay creates a merged child of the current frontier instead of
        resurrecting an old program. This preserves skills accepted after the
        stale rejection while applying the current gate to the original paired
        validation task and cached artifacts.
        """
        task_by_id = {task.task_id: task for task in self.dataset.validation}
        candidates: list[tuple[float, GraphProgram, ToolPathGraph, list[VideoTask]]] = []
        for program in self.registry.list_programs():
            if program.status != "rejected":
                continue
            for graph_id in program.graph_ids:
                graph = self._graphs.get(graph_id)
                if graph is None or graph.stats.get("accepted"):
                    continue
                evidence = graph.stats.get("gate_evidence") or {}
                if not self._is_stale_boundary_rejection(evidence):
                    continue
                validation_tasks = [
                    task_by_id[task_id]
                    for task_id in graph.stats.get("validation_task_ids", [])
                    if task_id in task_by_id
                ]
                if not validation_tasks:
                    continue
                candidates.append(
                    (
                        float(evidence.get("mean_paired_gain", 0.0)),
                        program,
                        graph,
                        validation_tasks,
                    )
                )

        promoted: list[str] = []
        replayed_scopes: set[tuple[str, ...]] = set()
        for _, old_program, graph, validation_tasks in sorted(
            candidates,
            key=lambda item: (item[0], item[2].graph_id),
            reverse=True,
        ):
            scope = tuple(sorted(task_class(task) for task in validation_tasks))
            if scope in replayed_scopes:
                continue
            replayed_scopes.add(scope)
            parent = self.registry.best()
            if parent is None or graph.skill_name in parent.graph_ids:
                continue
            parent_evaluation = self.evaluate_program(parent, validation_tasks)
            probe = GraphProgram(
                name=f"resume-revalidation-probe-{len(promoted) + 1:02d}",
                parent=parent.name,
                graph_ids=list(dict.fromkeys([*parent.graph_ids, graph.skill_name])),
                iteration=iteration,
                proposal=f"Revalidate stale boundary rejection {graph.skill_name}",
            )
            parent_tools = {
                tool
                for parent_graph_id in parent.graph_ids
                for tool in (
                    self._graphs[parent_graph_id].tool_names()
                    if parent_graph_id in self._graphs
                    else []
                )
            }
            candidate_evaluation = self.evaluate_program(
                probe,
                validation_tasks,
                forced_graph_id=graph.skill_name,
                required_tool_names=set(graph.tool_names()).difference(parent_tools),
            )
            passes_gate, gate_evidence = self._passes_validation_gate(
                parent_evaluation,
                candidate_evaluation,
            )
            if not passes_gate:
                continue

            child_name = self._next_resume_revalidation_name(iteration)
            child = GraphProgram(
                name=child_name,
                parent=parent.name,
                graph_ids=list(dict.fromkeys([*parent.graph_ids, graph.skill_name])),
                iteration=iteration,
                proposal=(
                    f"Merge {graph.skill_name} after revalidating a stale inclusive-boundary rejection"
                ),
                justification=(
                    f"Replayed {old_program.name} with the current task-metric tolerance and paired held-out artifacts"
                ),
                mutation=[{
                    "op": "merge_revalidated_graph",
                    "target": graph.skill_name,
                    "payload": {"source_program": old_program.name},
                    "reason": "preserve the current frontier while recovering valid historical gain",
                }],
            )
            child.metrics = self.evaluate_program(child, self.dataset.validation).metrics
            self.registry.create(child)
            admitted = self.registry.update_frontier(child.name, self.config.frontier_size)
            outcome = "accepted" if admitted else "dominated"
            gate_evidence.update({
                "strict_gate_passed": True,
                "exploratory_gate_passed": False,
                "resume_revalidation": True,
                "source_program": old_program.name,
            })
            graph.stats.update({
                "program": child.name,
                "parent_program": parent.name,
                "accepted": admitted,
                "admission": outcome,
                "quality_gain": float(gate_evidence["mean_paired_gain"]),
                "candidate_score": candidate_evaluation.metrics.quality,
                "routed_validation_score": child.metrics.quality,
                "gate_evidence": gate_evidence,
                "defer_weighted_credit": not admitted,
            })
            self._record_candidate_validation_credit(
                graph,
                validation_tasks,
                candidate_evaluation,
                positive_credit_allowed=admitted,
            )
            self.graph_memory.upsert_graph(graph)
            self.graph_archive.record_graph(
                graph,
                stage="resume_gate_revalidation",
                status=outcome,
                iteration=iteration,
                program=child.name,
                parent_program=parent.name,
                metadata={
                    "source_program": old_program.name,
                    "quality": candidate_evaluation.metrics.quality,
                    "routed_validation_quality": child.metrics.quality,
                    "quality_gain": gate_evidence["mean_paired_gain"],
                    "gate_evidence": gate_evidence,
                },
            )
            self.feedback.append(
                EvolutionFeedback(
                    iteration=iteration,
                    parent_program=parent.name,
                    child_program=child.name,
                    task_ids=[task.task_id for task in validation_tasks],
                    failure_types=list(graph.stats.get("source_failure_types", [])),
                    proposal=child.proposal,
                    justification=child.justification,
                    outcome=outcome,
                    parent_score=parent_evaluation.metrics.quality,
                    child_score=candidate_evaluation.metrics.quality,
                    graph_id=graph.graph_id,
                    evidence=["resume_gate_revalidation", f"source_program:{old_program.name}"],
                    validation_task_ids=[task.task_id for task in validation_tasks],
                )
            )
            if admitted:
                promoted.append(child.name)
                print(
                    "[evolution] resume revalidated historical graph "
                    f"{graph.skill_name}: gain={gate_evidence['mean_paired_gain']:.4f} "
                    f"merged_program={child.name}",
                    flush=True,
                )
        return promoted

    def _next_resume_revalidation_name(self, iteration: int) -> str:
        index = 1
        while True:
            name = f"resume-{iteration:03d}-revalidated-{index:02d}"
            if not (self.registry.program_dir / f"{name}.json").exists():
                return name
            index += 1

    def _revalidate_profile_routing_rejections(self, iteration: int) -> list[str]:
        """Recover heterogeneous candidates rejected before semantic routing existed.

        Historical forced probes may show that a tool path helps one subtype and
        harms another subtype in the same coarse task class. Rebuild a routing
        profile from those paired observations and admit the graph only when the
        normal production router passes the complete validation gate.
        """
        task_by_id = {task.task_id: task for task in self.dataset.validation}
        eligible: list[tuple[float, GraphProgram, ToolPathGraph, list[VideoTask]]] = []
        for old_program in self.registry.list_programs():
            if old_program.status != "rejected":
                continue
            for graph_id in old_program.graph_ids:
                graph = self._graphs.get(graph_id)
                if graph is None or graph.stats.get("accepted"):
                    continue
                evidence = graph.stats.get("gate_evidence") or {}
                validation_tasks = [
                    task_by_id[task_id]
                    for task_id in graph.stats.get("validation_task_ids", [])
                    if task_id in task_by_id
                ]
                profiles = {
                    json.dumps(self._task_routing_profile(task), sort_keys=True)
                    for task in validation_tasks
                }
                if not self._is_profile_routing_rejection(evidence, profiles):
                    continue
                eligible.append(
                    (
                        float(evidence.get("mean_paired_gain", 0.0)),
                        old_program,
                        graph,
                        validation_tasks,
                    )
                )

        limit = max(0, int(os.environ.get("EVOVIDEO_ROUTING_REVALIDATION_LIMIT", "3")))
        promoted: list[str] = []
        for _, old_program, graph, validation_tasks in sorted(
            eligible,
            key=lambda item: (item[0], item[2].graph_id),
            reverse=True,
        )[:limit]:
            parent = self.registry.best()
            if parent is None or graph.skill_name in parent.graph_ids:
                continue
            parent_tools = {
                tool
                for parent_graph_id in parent.graph_ids
                for tool in (
                    self._graphs[parent_graph_id].tool_names()
                    if parent_graph_id in self._graphs
                    else []
                )
            }
            required_tools = set(graph.tool_names()).difference(parent_tools)
            parent_evaluation = self.evaluate_program(parent, validation_tasks)
            probe = GraphProgram(
                name=f"routing-revalidation-probe-{len(promoted) + 1:02d}",
                parent=parent.name,
                graph_ids=list(dict.fromkeys([*parent.graph_ids, graph.skill_name])),
                iteration=iteration,
                proposal=f"Revalidate semantic routing for {graph.skill_name}",
            )
            forced_evaluation = self.evaluate_program(
                probe,
                validation_tasks,
                forced_graph_id=graph.skill_name,
                required_tool_names=required_tools,
            )
            candidate = GraphPathCandidate(
                graph=graph,
                edits=[],
                reason=probe.proposal,
                parent_failure_types=list(graph.stats.get("source_failure_types", [])),
            )
            self._set_candidate_routing_scope(candidate, parent_evaluation, forced_evaluation)
            routed_evaluation = self.evaluate_program(
                probe,
                validation_tasks,
                required_tool_names=required_tools,
            )
            passes_gate, gate_evidence = self._passes_validation_gate(
                parent_evaluation,
                routed_evaluation,
            )
            if not passes_gate:
                continue

            child_name = self._next_resume_revalidation_name(iteration)
            child = GraphProgram(
                name=child_name,
                parent=parent.name,
                graph_ids=probe.graph_ids,
                iteration=iteration,
                proposal=f"Merge profile-routed historical graph {graph.skill_name}",
                justification=(
                    "The normal router preserves the parent path on regressing subtypes "
                    "and selects this graph only on positively validated semantic profiles."
                ),
                mutation=[{
                    "op": "specialize_graph_route",
                    "target": graph.skill_name,
                    "payload": {
                        "source_program": old_program.name,
                        "profiles": graph.stats.get("validated_routing_profiles", []),
                    },
                    "reason": "recover subtype-specific gain without relaxing metric guards",
                }],
            )
            child.metrics = self.evaluate_program(child, self.dataset.validation).metrics
            self.registry.create(child)
            admitted = self.registry.update_frontier(child.name, self.config.frontier_size)
            outcome = "accepted" if admitted else "dominated"
            gate_evidence.update(
                {
                    "strict_gate_passed": True,
                    "post_routing_validation": True,
                    "profile_routing_revalidation": True,
                    "source_program": old_program.name,
                    "forced_probe_quality": forced_evaluation.metrics.quality,
                    "normal_route_quality": routed_evaluation.metrics.quality,
                }
            )
            graph.stats.update(
                {
                    "program": child.name,
                    "parent_program": parent.name,
                    "accepted": admitted,
                    "admission": outcome,
                    "quality_gain": float(gate_evidence["mean_paired_gain"]),
                    "candidate_score": routed_evaluation.metrics.quality,
                    "routed_validation_score": child.metrics.quality,
                    "gate_evidence": gate_evidence,
                    "defer_weighted_credit": not admitted,
                }
            )
            self._record_candidate_validation_credit(
                graph,
                validation_tasks,
                routed_evaluation,
                positive_credit_allowed=admitted,
            )
            self.graph_memory.upsert_graph(graph)
            self.graph_archive.record_graph(
                graph,
                stage="resume_profile_routing_revalidation",
                status=outcome,
                iteration=iteration,
                program=child.name,
                parent_program=parent.name,
                metadata={
                    "source_program": old_program.name,
                    "quality": routed_evaluation.metrics.quality,
                    "routed_validation_quality": child.metrics.quality,
                    "quality_gain": gate_evidence["mean_paired_gain"],
                    "routing_profiles": graph.stats.get("validated_routing_profiles", []),
                    "gate_evidence": gate_evidence,
                },
            )
            if admitted:
                promoted.append(child.name)
                print(
                    "[evolution] resume recovered profile-routed historical graph "
                    f"{graph.skill_name}: gain={gate_evidence['mean_paired_gain']:.4f} "
                    f"merged_program={child.name}",
                    flush=True,
                )
        return promoted

    def _is_profile_routing_rejection(
        self,
        evidence: dict[str, Any],
        profiles: set[str],
    ) -> bool:
        try:
            gain = float(evidence.get("mean_paired_gain", 0.0))
            regression = float(evidence.get("max_task_metric_regression", 0.0))
            allowed = float(
                evidence.get(
                    "max_allowed_task_metric_regression",
                    self.config.max_task_metric_regression,
                )
            )
        except (TypeError, ValueError):
            return False
        return (
            len(profiles) > 1
            and gain >= self.config.min_quality_gain
            and regression > allowed + 1e-9
            and float(evidence.get("paired_improvement_fraction", 0.0)) >= 0.5
            and float(evidence.get("paired_coverage", 0.0)) == 1.0
            and float(evidence.get("candidate_tool_coverage", 0.0)) == 1.0
            and (evidence.get("h3_replicate_gate_passed") is True if self._h3_active()
                 else evidence.get("seed_control_fraction") in {None, 1.0})
        )

    def _is_stale_boundary_rejection(self, evidence: dict[str, Any]) -> bool:
        if evidence.get("task_metric_guard_passed") is not False:
            return False
        try:
            regression = float(evidence["max_task_metric_regression"])
            allowed = float(evidence["max_allowed_task_metric_regression"])
            gain = float(evidence["mean_paired_gain"])
        except (KeyError, TypeError, ValueError):
            return False
        tolerance = 1e-9
        return (
            regression > allowed
            and regression <= allowed + tolerance
            and gain + tolerance >= self.config.min_quality_gain
            and float(evidence.get("aggregate_quality_gain", gain)) + tolerance >= 0.0
            and float(evidence.get("paired_improvement_fraction", 0.0)) >= 0.5
            and float(evidence.get("paired_coverage", 0.0)) == 1.0
            and float(evidence.get("candidate_tool_coverage", 0.0)) == 1.0
            and float(evidence.get("candidate_pass_rate", 0.0))
            >= float(evidence.get("baseline_pass_rate", 0.0))
            and (evidence.get("h3_replicate_gate_passed") is True if self._h3_active()
                 else evidence.get("seed_control_fraction") in {None, 1.0})
        )

    def _mutation_searches_from_log(self) -> int:
        if not self.iteration_log_path.exists():
            return 0
        total = 0
        with self.iteration_log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "mutation_searches_used" in payload:
                    total = max(total, int(payload["mutation_searches_used"]))
                else:
                    total += len(payload.get("proposed_programs", []))
        return total

    def _select_graph(
        self,
        program: GraphProgram,
        task: VideoTask,
        observed_failure_types: set[str] | None = None,
    ) -> ToolPathGraph:
        expected = (
            {str(item).lower() for item in observed_failure_types}
            if observed_failure_types is not None
            else {
                str(item).lower()
                for item in (task.metadata or {}).get("expected_failure_modes", [])
            }
        )
        prompt = task.prompt.lower()
        ranked: list[tuple[float, ToolPathGraph]] = []
        category = task_class(task)
        task_profile = self._task_routing_profile(task)
        transfer_classes = self._external_transfer_classes(task)
        candidate_ids = list(program.graph_ids)
        candidate_ids.extend(
            graph.skill_name
            for graph in self._graphs.values()
            if graph.stats.get("online_foundation")
            and graph.stats.get("accepted")
            and graph.stats.get("task_class") == category
        )
        for graph_id in dict.fromkeys(candidate_ids):
            graph = self._graphs.get(graph_id)
            if graph is None:
                continue
            if any(
                not self.evolver.tools.has(tool_name)
                for tool_name in graph.tool_names()
            ):
                continue
            if not self._graph_task_inputs_available(graph, task):
                continue
            generic_triggers = {"always", "generation", "video", "foundation_skill"}
            specific_triggers = {
                trigger.lower()
                for trigger in graph.triggers
                if trigger.lower() not in generic_triggers
            }
            validated_classes = {
                str(item).lower()
                for item in graph.stats.get("validated_task_classes", [])
            }
            declared_class = str(graph.stats.get("task_class") or "").lower()
            if declared_class:
                validated_classes.add(declared_class)
            source_failures = {
                str(item).lower()
                for item in graph.stats.get("source_failure_types", [])
            }
            routing_profiles = graph.stats.get("validated_routing_profiles", [])
            if not self._graph_in_scope(
                graph,
                category=category,
                expected=expected,
                prompt=prompt,
                specific_triggers=specific_triggers,
                validated_classes=validated_classes,
                source_failures=source_failures,
                routing_profiles=routing_profiles,
                task_profile=task_profile,
                transfer_classes=transfer_classes,
            ):
                continue
            trigger_hits = sum(
                1 for trigger in specific_triggers
                if trigger in prompt or trigger in expected
            )
            failure_hits = len(expected & (specific_triggers | source_failures))
            exact_class_match = category in validated_classes
            transferred_class_match = bool(transfer_classes & validated_classes)
            class_bonus = 2.5 if exact_class_match else 1.25 if transferred_class_match else 0.0
            exact_profile_match = bool(
                routing_profiles
                and any(self._routing_profile_matches(profile, task_profile) for profile in routing_profiles)
            )
            transferred_profile_match = bool(
                routing_profiles
                and any(
                    self._routing_profile_transfer_matches(profile, task_profile, transfer_classes)
                    for profile in routing_profiles
                )
            )
            profile_bonus = 6.0 if exact_profile_match else 3.0 if transferred_profile_match else 0.0
            learned_gain = float(graph.stats.get("quality_gain", 0.0))
            path_prior = 0.0
            if self.weighted_tool_graph is not None:
                path_prior = self.weighted_tool_graph.path_prior(
                    category, graph.tool_names(),
                    **({"graph_behavior_fingerprint": self._execution_fingerprint(graph)} if self._h3_active() else {}),
                )
            foundation_bonus = 0.75 if graph.stats.get("online_foundation") and graph.stats.get("task_class") == category else 0.0
            score = (
                class_bonus
                + profile_bonus
                + 2.0 * failure_hits
                + trigger_hits
                + learned_gain
                + self.config.path_prior_weight * path_prior
                + foundation_bonus
            )
            ranked.append((score, graph))
        if not ranked:
            return self.evolver.baseline_graph()
        ranked.sort(key=lambda item: (item[0], -item[1].estimated_cost()), reverse=True)
        return ranked[0][1]

    @staticmethod
    def _graph_in_scope(
        graph: ToolPathGraph,
        *,
        category: str,
        expected: set[str],
        prompt: str,
        specific_triggers: set[str],
        validated_classes: set[str],
        source_failures: set[str],
        routing_profiles: list[dict[str, Any]] | None = None,
        task_profile: dict[str, str] | None = None,
        transfer_classes: set[str] | None = None,
    ) -> bool:
        if graph.skill_name == "baseline_t2v_graph":
            return True
        transfer_classes = transfer_classes or set()
        causal_match = bool(
            expected.intersection(specific_triggers | source_failures)
        )
        if routing_profiles:
            exact_match = any(
                GraphSelfImprovingLoop._routing_profile_matches(profile, task_profile or {})
                for profile in routing_profiles
            )
            transfer_match = causal_match and any(
                GraphSelfImprovingLoop._routing_profile_transfer_matches(
                    profile,
                    task_profile or {},
                    transfer_classes,
                )
                for profile in routing_profiles
            )
            return exact_match or transfer_match
        if validated_classes:
            return category in validated_classes or (
                causal_match and bool(transfer_classes.intersection(validated_classes))
            )
        return bool(
            specific_triggers.intersection(expected)
            or source_failures.intersection(expected)
            or any(trigger in prompt for trigger in specific_triggers)
        )

    def _set_candidate_routing_scope(
        self,
        candidate: GraphPathCandidate,
        baseline: ProgramEvaluation,
        evaluation: ProgramEvaluation,
    ) -> None:
        task_by_id = {task.task_id: task for task in self.dataset.validation}
        baseline_by_key = {
            (summary.task_id, summary.evaluation_seed): summary
            for summary in baseline.rollouts
            if summary.execution_error is None
        }
        positive_classes: set[str] = set()
        positive_profiles: list[dict[str, str]] = []
        for summary in evaluation.rollouts:
            reference = baseline_by_key.get((summary.task_id, summary.evaluation_seed))
            task = task_by_id.get(summary.task_id)
            if (
                reference is not None
                and task is not None
                and summary.execution_error is None
                and summary.score > reference.score
            ):
                positive_classes.add(task_class(task))
                profile = self._task_routing_profile(task)
                if profile not in positive_profiles:
                    positive_profiles.append(profile)
        source_class = str(
            candidate.graph.stats.get("portfolio_selection", {}).get("task_class")
            or candidate.graph.stats.get("task_class")
            or ""
        )
        if source_class:
            positive_classes.add(source_class)
        candidate.graph.stats.update(
            {
                "task_class": source_class or "general",
                "validated_task_classes": sorted(positive_classes),
                "validated_routing_profiles": positive_profiles,
                "source_failure_types": sorted(set(candidate.parent_failure_types)),
                "routing_policy": "validated_semantic_profile",
            }
        )

    @staticmethod
    def _task_routing_profile(task: VideoTask) -> dict[str, str]:
        """Return transferable metadata used to specialize otherwise broad task classes."""
        metadata = task.metadata or {}
        profile = {"task_class": task_class(task).strip().lower()}
        for key in ("target_style", "routing_subtype"):
            value = metadata.get(key)
            if value is not None and str(value).strip():
                profile[key] = str(value).strip().lower()
        return profile

    @staticmethod
    def _routing_profile_matches(
        profile: dict[str, Any],
        task_profile: dict[str, str],
    ) -> bool:
        if not isinstance(profile, dict) or not profile:
            return False
        return all(
            str(task_profile.get(str(key), "")).strip().lower()
            == str(value).strip().lower()
            for key, value in profile.items()
        )

    @staticmethod
    def _routing_profile_transfer_matches(
        profile: dict[str, Any],
        task_profile: dict[str, str],
        transfer_classes: set[str],
    ) -> bool:
        """Match a learned profile through an explicit external-benchmark class map."""
        if not isinstance(profile, dict) or not profile or not transfer_classes:
            return False
        source_class = str(profile.get("task_class", "")).strip().lower()
        if source_class not in transfer_classes:
            return False
        return all(
            key == "task_class"
            or str(task_profile.get(str(key), "")).strip().lower()
            == str(value).strip().lower()
            for key, value in profile.items()
        )

    @staticmethod
    def _external_transfer_classes(task: VideoTask) -> set[str]:
        """Map external benchmark semantics onto classes learned on ComplexVideoBench."""
        metadata = task.metadata or {}
        benchmark = str(metadata.get("benchmark") or "").strip().lower()
        if benchmark == "vbench":
            dimensions = metadata.get("vbench_dimensions") or [metadata.get("vbench_dimension")]
            mapping = {
                "subject_consistency": {"multi_shot_identity"},
                "background_consistency": {"compositional_editing", "long_horizon_causal"},
                "temporal_flickering": {"physical_dynamics", "video_stylization"},
                "motion_smoothness": {"camera_control", "long_horizon_causal", "physical_dynamics"},
                "dynamic_degree": {"camera_control", "physical_dynamics"},
                "multiple_objects": {"multi_character_interaction", "long_horizon_causal"},
                "human_action": {"camera_control", "long_horizon_causal"},
                "spatial_relationship": {"multi_character_interaction", "compositional_editing"},
                "scene": {"compositional_editing", "long_horizon_causal"},
                "temporal_style": {"video_stylization", "long_horizon_causal"},
                "appearance_style": {"video_stylization"},
                "overall_consistency": {"multi_shot_identity", "long_horizon_causal"},
            }
            classes: set[str] = set()
            for dimension in dimensions:
                classes.update(mapping.get(str(dimension).strip().lower(), set()))
            return classes
        if benchmark == "storybench":
            return {
                "camera_control",
                "long_horizon_causal",
                "multi_character_interaction",
                "multi_shot_identity",
            }
        return set()

    @staticmethod
    def _graph_task_inputs_available(graph: ToolPathGraph, task: VideoTask) -> bool:
        """Reject paths whose required task-level source assets are unavailable."""
        tools = set(graph.tool_names())
        if "task_reference_video" in tools and not task.reference_video:
            return False
        if "h3_reference_bank" in tools and not (task.metadata or {}).get("h3_references"):
            return False
        return True

    def _candidate_validation_tasks(
        self,
        candidate: GraphPathCandidate,
    ) -> list[VideoTask]:
        source_class = str(
            candidate.graph.stats.get("portfolio_selection", {}).get("task_class")
            or candidate.graph.stats.get("task_class")
            or ""
        ).strip().lower()
        if not source_class or source_class == "general":
            return list(self.dataset.validation)
        validation_matches = [
            task for task in self.dataset.validation
            if task_class(task) == source_class
        ]
        if validation_matches or not self.config.allow_seed_holdout_validation or self._h3_active():
            return validation_matches

        # Some compact benchmark subsets contain singleton task classes. Keep
        # the formal test split untouched and validate those mutations on the
        # same training prompt with a fixed, non-exploration generation seed.
        training_matches = sorted(
            (task for task in self.dataset.train if task_class(task) == source_class),
            key=lambda task: task.task_id,
        )
        if not training_matches:
            return []
        task = training_matches[0]
        return [
            VideoTask(
                task_id=task.task_id,
                prompt=task.prompt,
                mode=task.mode,
                duration_seconds=task.duration_seconds,
                reference_video=task.reference_video,
                metadata={
                    **(task.metadata or {}),
                    "generation_seed": self.config.seed_holdout_validation_seed,
                    "seed_holdout_validation": True,
                },
            )
        ]

    def _runtime_replenished_candidates(
        self,
        selected: list[GraphPathCandidate],
        deferred: list[GraphPathCandidate],
        *,
        iteration: int,
        failures: list[GraphPathRollout],
    ):
        """Replace candidates invalidated by failures discovered earlier this iteration."""
        queue = list(selected)
        reserve = list(deferred)
        target_validations = len(selected)
        yielded = 0
        seen = {candidate.graph.graph_id for candidate in selected}
        failure_types = sorted({
            item.value
            for rollout in failures
            if rollout.failure is not None
            for item in rollout.failure.failure_types
        })
        while queue and yielded < target_validations:
            candidate = queue.pop(0)
            broken_tool = self._circuit_broken_tool(candidate.graph)
            if broken_tool is None:
                yielded += 1
                yield candidate
                continue

            self.graph_archive.record_candidate_audit(
                {
                    "iteration": iteration,
                    "task_ids": [rollout.task.task_id for rollout in failures],
                    "failure_types": failure_types,
                    "stage": "runtime_candidate_replenishment",
                    "graph_id": candidate.graph.graph_id,
                    "status": "circuit_skipped",
                    "failed_tool": broken_tool,
                    "reason": str(self._runtime_tool_failures.get(broken_tool, "fatal runtime failure")),
                }
            )
            replacement = None
            while reserve:
                proposed = reserve.pop(0)
                if proposed.graph.graph_id in seen:
                    continue
                seen.add(proposed.graph.graph_id)
                if self._circuit_broken_tool(proposed.graph) is None:
                    replacement = proposed
                    break
            if replacement is not None:
                queue.append(replacement)
                self.graph_archive.record_candidate_audit(
                    {
                        "iteration": iteration,
                        "task_ids": [rollout.task.task_id for rollout in failures],
                        "failure_types": failure_types,
                        "stage": "runtime_candidate_replenishment",
                        "graph_id": replacement.graph.graph_id,
                        "status": "replacement_selected",
                        "replaces_graph_id": candidate.graph.graph_id,
                        "reason": "selected from the deferred portfolio after a runtime circuit break",
                    }
                )

    def _refresh_online_foundations(
        self,
        iteration: int,
        parent: GraphProgram,
    ) -> list[str]:
        if (
            not self.config.online_foundation_enabled
            or self.foundation_promoter is None
            or iteration % max(1, self.config.online_foundation_interval) != 0
        ):
            return []
        known = {self._graph_fingerprint(graph) for graph in self._graphs.values()}
        promoted: list[str] = []
        proposals = self.foundation_promoter.propose()
        for deferral in getattr(self.foundation_promoter, "last_deferrals", []):
            self.graph_archive.record_candidate_audit({
                "stage": "foundation_reconstruction", "iteration": iteration, **deferral,
            })
        for graph in proposals:
            fingerprint = self._graph_fingerprint(graph)
            native_clone = bool(graph.stats.get("foundation_source_graph_id"))
            if (fingerprint in known and not native_clone) or fingerprint in self._foundation_attempts or graph.skill_name in self._graphs:
                continue
            if native_clone:
                # Exact native clones deliberately match their accepted source;
                # allow one independent foundation validation, never an auto-pass.
                self._foundation_attempts.add(fingerprint)
            category = str(graph.stats.get("task_class", "general"))
            validation_tasks = [
                task for task in self.dataset.validation if task_class(task) == category
            ]
            if not validation_tasks:
                self._foundation_attempts.add(fingerprint)
                graph.stats.update(
                    {
                        "accepted": False,
                        "rejection_reason": "no_category_matched_validation_tasks",
                        "promotion_iteration": iteration,
                    }
                )
                self.graph_archive.record_graph(
                    graph,
                    stage="foundation_validation",
                    status="rejected",
                    iteration=iteration,
                    parent_program=parent.name,
                    metadata={
                        "task_class": category,
                        "rejection_reason": "no_category_matched_validation_tasks",
                    },
                )
                self._append_foundation_event(iteration, graph, [], False)
                known.add(fingerprint)
                continue
            validation_tasks = self._evaluation_variants(validation_tasks)
            summaries = [self._rollout_summary(task, graph) for task in validation_tasks]
            parent_summaries = [
                self._rollout_summary(task, self._select_graph(parent, task))
                for task in validation_tasks
            ]
            valid_pairs = [
                (base, child)
                for base, child in zip(parent_summaries, summaries)
                if base.execution_error is None and child.execution_error is None
            ]
            valid_parent = [base for base, _ in valid_pairs]
            valid_candidates = [child for _, child in valid_pairs]
            quality = sum(item.score for item in valid_candidates) / max(1, len(valid_candidates))
            category_parent_score = sum(item.score for item in valid_parent) / max(1, len(valid_parent))
            pass_rate = sum(item.passed for item in valid_candidates) / max(1, len(valid_candidates))
            parent_pass_rate = sum(item.passed for item in valid_parent) / max(1, len(valid_parent))
            stability = graph_score_stability([item.score for item in valid_candidates])
            gain = quality - category_parent_score
            independent_tasks = {item.task_id for item in valid_candidates if item.passed}
            paired_coverage = len(valid_pairs) / max(1, len(summaries))
            replicate_evidence = self._h3_replicate_evidence(valid_pairs)
            accepted = (
                gain >= self.config.online_foundation_min_gain
                and quality >= category_parent_score
                and pass_rate > 0.0
                and pass_rate >= parent_pass_rate
                and len(independent_tasks) >= 2
                and paired_coverage == 1.0
                and replicate_evidence.get("h3_replicate_gate_passed", True)
            )
            graph.stats.update(
                {
                    "online_validation_score": quality,
                    **replicate_evidence,
                    "online_validation_gain": gain,
                    "online_parent_score": category_parent_score,
                    "pass_rate": pass_rate,
                    "baseline_pass_rate": parent_pass_rate,
                    "passed_validation_task_ids": sorted(independent_tasks),
                    "stability": stability,
                    "accepted": accepted,
                    "promotion_iteration": iteration,
                    "paired_coverage": paired_coverage,
                }
            )
            self.graph_archive.record_graph(
                graph,
                stage="foundation_validation",
                status="promoted" if accepted else "rejected",
                iteration=iteration,
                parent_program=parent.name,
                metadata={
                    "task_class": category,
                    "quality": quality,
                    "parent_quality": category_parent_score,
                    "quality_gain": gain,
                    "pass_rate": pass_rate,
                    "stability": stability,
                    "core_tools": graph.stats.get("foundation_core_tools", []),
                    "paired_coverage": paired_coverage,
                },
            )
            self._append_foundation_event(iteration, graph, summaries, accepted)
            known.add(fingerprint)
            if not accepted:
                continue
            self.graph_memory.upsert_graph(graph)
            self._graphs[graph.skill_name] = graph
            skill = graph.to_skill_card()
            skill.version = "online-foundation-1.0"
            skill.validation = SkillValidationReport(
                skill_name=graph.skill_name,
                validation_task_ids=[item.task_id for item in summaries],
                baseline_score=category_parent_score,
                candidate_score=quality,
                baseline_pass_rate=parent_pass_rate,
                candidate_pass_rate=pass_rate,
                quality_gain=gain,
                estimated_cost=graph.estimated_cost(),
                accepted=True,
                evidence=[
                    f"task_class:{graph.stats.get('task_class')}",
                    f"support:{graph.stats.get('support')}",
                    "online held-out foundation validation",
                ],
            )
            self.evolver.skill_memory.upsert(skill)
            for frontier_program in self.registry.frontier():
                if graph.skill_name not in frontier_program.graph_ids:
                    frontier_program.graph_ids.append(graph.skill_name)
                    self.registry.upsert(frontier_program)
            promoted.append(graph.skill_name)
        return promoted

    def _append_foundation_event(
        self,
        iteration: int,
        graph: ToolPathGraph,
        summaries: list[TaskRolloutSummary],
        accepted: bool,
    ) -> None:
        payload = {
            "iteration": iteration,
            "graph_id": graph.graph_id,
            "task_class": graph.stats.get("task_class"),
            "tools": graph.tool_names(),
            "support": graph.stats.get("support"),
            "path_ucb": graph.stats.get("path_ucb"),
            "quality": graph.stats.get("online_validation_score"),
            "gain": graph.stats.get("online_validation_gain"),
            "accepted": accepted,
            "validation_tasks": [item.task_id for item in summaries],
            "created_at": utc_now(),
        }
        with self.foundation_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _novel_candidates(
        self,
        parent: GraphProgram,
        candidates: list[GraphPathCandidate],
        iteration: int,
    ) -> list[GraphPathCandidate]:
        del parent, iteration
        seen_fingerprints = {self._graph_fingerprint(graph) for graph in self._graphs.values()}
        seen_execution_fingerprints = {
            self._execution_fingerprint(graph) for graph in self._graphs.values()
        }
        unique: dict[str, GraphPathCandidate] = {}
        for candidate in candidates:
            key = self._graph_fingerprint(candidate.graph)
            execution_key = self._execution_fingerprint(candidate.graph)
            if key in seen_fingerprints or execution_key in seen_execution_fingerprints:
                continue
            unique.setdefault(key, candidate)
            seen_execution_fingerprints.add(execution_key)
        return list(unique.values())

    def _select_candidate_portfolio(
        self,
        candidates: list[GraphPathCandidate],
        failures: list[GraphPathRollout],
        limit: int,
    ) -> list[GraphPathCandidate]:
        """Rank candidates by causal fit, quality-gain evidence, and path diversity."""
        if limit <= 0 or not candidates:
            return []
        failure_types = {
            item.value
            for rollout in failures
            if rollout.failure is not None
            for item in rollout.failure.failure_types
        }
        prompts = " ".join(rollout.task.prompt.lower() for rollout in failures)
        category = task_class(failures[0].task) if failures else "general"
        baseline_tools = set(self.evolver.baseline_graph().tool_names())
        scored: list[tuple[float, GraphPathCandidate, str]] = []
        for candidate in candidates:
            tools = candidate.graph.tool_names()
            joined = " ".join(tools).lower()
            components: dict[str, float] = {}
            components["llm_source"] = 0.45 if candidate.graph.stats.get("proposal_source") == "llm" else 0.0
            if candidate.graph.stats.get("proposal_source") == "open_world_invention":
                components["open_world_invention"] = 0.35
                components["structural_novelty"] = 0.5 * float(
                    candidate.graph.stats.get("structural_novelty", 0.0)
                )
            else:
                components["open_world_invention"] = 0.0
                components["structural_novelty"] = 0.0
            components["tool_novelty"] = 0.12 * len(set(tools) - baseline_tools)
            preserves_healthy = bool(
                candidate.graph.stats.get("healthy_content_preservation")
                or "segment_stitcher" in tools
            )
            components["healthy_content_preservation"] = 0.85 if preserves_healthy and not self._h3_active() else 0.0
            destructive_repair_capabilities = {
                "image_conditioned_video_generation",
                "multi_shot_identity_conditioned_generation",
                "motion_conditioned_video_generation",
                "global_video_editing",
                "failed_segment_repair",
            }
            downstream_video_tools = [
                name
                for name in tools
                if name != "mock_text_to_video"
                and self.evolver.tools.has(name)
                and self.evolver.tools.spec(name).output_type == "video"
                and self.evolver.tools.spec(name).capability
                in destructive_repair_capabilities
            ]
            whole_video_regeneration = (
                not self._h3_active()
                and "mock_text_to_video" in tools
                and bool(downstream_video_tools)
                and "segment_stitcher" not in tools
                and not any("style" in name.lower() for name in downstream_video_tools)
            )
            components["whole_video_regeneration_penalty"] = (
                -0.9 if whole_video_regeneration else 0.0
            )
            components["failure_fit"] = 0.0
            if "identity_drift" in failure_types and any(
                token in joined for token in ("identity", "character", "image_to_video", "i2v")
            ):
                components["failure_fit"] += 0.55
            if "motion_mismatch" in failure_types and any(
                token in joined for token in ("temporal", "keyframe", "repair", "editor")
            ):
                components["failure_fit"] += 0.55
            if "temporal_flicker" in failure_types and any(
                token in joined for token in ("deflicker", "stabil", "temporal", "repair")
            ):
                components["failure_fit"] += 0.55
            if "style_drift" in failure_types and any(
                token in joined for token in ("style", "styliz", "rerender", "tokenflow", "rave")
            ):
                components["failure_fit"] += 0.55
            if "editing_leakage" in failure_types and any(
                token in joined for token in ("edit", "track", "mask", "region", "repair")
            ):
                components["failure_fit"] += 0.55
            if "object_persistence_failure" in failure_types and any(
                token in joined for token in ("object", "track", "keyframe", "reference", "identity")
            ):
                components["failure_fit"] += 0.55
            if "prompt_omission" in failure_types and any(
                token in joined
                for token in (
                    "plan", "temporal", "decompos", "scene", "structure",
                    "edit", "track", "mask", "region", "repair",
                )
            ):
                components["failure_fit"] += 0.45
            if any(token in prompts for token in ("shot", "scene", "then", "across")) and any(
                token in joined for token in ("scene_splitter", "character_sheet", "multi_shot")
            ):
                components["multishot_fit"] = 0.65
            else:
                components["multishot_fit"] = 0.0
            if self.weighted_tool_graph is not None:
                components["quality_gain_prior"] = (
                    self.config.path_prior_weight
                    * self.weighted_tool_graph.path_prior(
                        category, tools,
                        **({"graph_behavior_fingerprint": self._execution_fingerprint(candidate.graph)} if self._h3_active() else {}),
                    )
                )
            else:
                components["quality_gain_prior"] = 0.0
            unseeded_dynamic = 0
            for name in tools:
                if not self.evolver.tools.has(name):
                    continue
                spec = self.evolver.tools.spec(name)
                if spec.backend == "builtin":
                    continue
                manifest = getattr(self.evolver.tools.get(name), "manifest", None)
                if manifest is not None and not any("{seed}" in token for token in manifest.command):
                    unseeded_dynamic += 1
            components["uncontrolled_seed_penalty"] = 0.0 if self._h3_active() else -0.8 * unseeded_dynamic
            score = sum(components.values())
            terminal = self._terminal_video_tool(candidate.graph)
            arena = candidate.graph.stats.get("tool_arena_variant", {})
            mechanism = candidate.graph.stats.get("mechanism_family")
            arena_family = str(arena.get("family_id") or "")
            diversity_key = str(
                f"mechanism:{mechanism}"
                if mechanism
                else f"arena:{arena_family}"
                if arena_family
                else terminal
            )
            compatible, incompatibility = self._candidate_task_compatibility(
                candidate, failures
            )
            broken_tool = self._circuit_broken_tool(candidate.graph)
            if broken_tool is not None:
                compatible = False
                failure = getattr(self, "_runtime_tool_failures", {}).get(broken_tool, {})
                failure_error = (
                    failure.get("error", "fatal runtime failure")
                    if isinstance(failure, dict)
                    else str(failure or "fatal runtime failure")
                )
                incompatibility = (
                    f"tool {broken_tool!r} is runtime-circuit-broken for this run: "
                    f"{failure_error}"
                )
            if compatible and failure_types and components["failure_fit"] <= 0.0 and not self._h3_active():
                compatible = False
                incompatibility = (
                    "candidate has no causal match for observed failures: "
                    + ", ".join(sorted(failure_types))
                )
            candidate.graph.stats["portfolio_selection"] = {
                "score": score,
                "components": components,
                "terminal_tool": terminal,
                "diversity_key": diversity_key,
                "task_class": category,
                "reward_objective": self._reward_objective(failures),
                "preserves_healthy_wan_content": preserves_healthy,
                "whole_video_regeneration": whole_video_regeneration,
                "eligible": compatible,
                "rejection_reason": incompatibility if not compatible else None,
            }
            if not compatible:
                continue
            scored.append((score, candidate, diversity_key))
        scored.sort(key=lambda item: (item[0], item[1].graph.graph_id), reverse=True)

        selected: list[GraphPathCandidate] = []
        used_mechanisms: set[str] = set()
        if "motion_mismatch" in failure_types:
            temporal = next(
                (
                    item for item in scored
                    if item[1].graph.stats.get("mechanism_family")
                    == "temporal_prompt_conditioning"
                ),
                None,
            )
            if temporal is not None:
                _, candidate, mechanism_key = temporal
                candidate.graph.stats["portfolio_selection"]["selection_stage"] = (
                    "reserved_causal_temporal_plan"
                )
                selected.append(candidate)
                used_mechanisms.add(mechanism_key)
        remaining = limit - len(selected)
        if os.environ.get("OPEN_WORLD_ARENA_REQUIRE_PAIR", "0") == "1" and remaining >= 2:
            arena_groups: dict[str, list[tuple[float, GraphPathCandidate, str]]] = {}
            for scored_item in scored:
                arena = scored_item[1].graph.stats.get("tool_arena_variant", {})
                family_id = str(arena.get("family_id") or "")
                if family_id:
                    arena_groups.setdefault(family_id, []).append(scored_item)
            complete_groups = [
                items for items in arena_groups.values() if len(items) >= 2
            ]
            if complete_groups:
                paired = max(
                    complete_groups,
                    key=lambda items: (items[0][0] + items[1][0], items[0][1].graph.graph_id),
                )[:2]
                for _, candidate, mechanism_key in paired:
                    candidate.graph.stats["portfolio_selection"]["selection_stage"] = (
                        "paired_tool_arena"
                    )
                    selected.append(candidate)
                    used_mechanisms.add(mechanism_key)
        # Spend the first pass on causally distinct mechanisms. Tool-arena arms
        # remain available, but cannot consume every slot merely because their
        # terminal repository names differ.
        for _, candidate, mechanism_key in scored:
            if len(selected) >= limit:
                break
            if candidate in selected:
                continue
            if mechanism_key in used_mechanisms:
                continue
            candidate.graph.stats["portfolio_selection"]["selection_stage"] = "mechanism_diversity"
            selected.append(candidate)
            used_mechanisms.add(mechanism_key)
        for _, candidate, _ in scored:
            if len(selected) >= limit:
                break
            if candidate not in selected:
                arena = candidate.graph.stats.get("tool_arena_variant", {})
                candidate.graph.stats["portfolio_selection"]["selection_stage"] = (
                    "tool_arena_fill" if arena.get("family_id") else "utility_fill"
                )
                selected.append(candidate)
        return selected

    def _candidate_task_compatibility(
        self,
        candidate: GraphPathCandidate,
        failures: list[GraphPathRollout],
    ) -> tuple[bool, str | None]:
        categories = {task_class(rollout.task) for rollout in failures}
        if len(categories) > 1:
            return False, (
                "candidate planning batch mixes unrelated task classes: "
                + ", ".join(sorted(categories))
            )
        missing_assets = sorted({
            str(asset_id)
            for rollout in failures
            for asset_id in (rollout.task.metadata or {}).get("missing_required_assets", [])
        })
        if missing_assets:
            return False, (
                "required benchmark assets are unresolved in the sibling *_assets.json manifest: "
                + ", ".join(missing_assets)
            )
        if self._h3_active() and any(not self._graph_task_inputs_available(candidate.graph, rollout.task) for rollout in failures):
            return False, "candidate requires task reference assets that are unavailable"
        category = next(iter(categories), "general")
        specs = [
            self.evolver.tools.spec(name)
            for name in candidate.graph.tool_names()
            if self.evolver.tools.has(name)
        ]
        if category == "audio_video_sync":
            audio_video_tools = [
                spec
                for spec in specs
                if spec.output_type == "video" and (
                    "audio" in set(spec.input_types)
                    or (self._h3_active() and spec.name == "h3_ref2va"
                        and "h3_reference_set" in set(spec.input_types))
                )
            ]
            if not audio_video_tools:
                return False, (
                    "audio_video_sync requires a video-producing tool with a physical audio input; "
                    "plain T2V/I2V cannot realize this task"
                )
        return True, None

    def _terminal_video_tool(self, graph: ToolPathGraph) -> str:
        outgoing = {edge.source for edge in graph.edges}
        terminal = [
            node.name
            for node in graph.nodes
            if node.node_type == "tool"
            and node.node_id not in outgoing
            and self.evolver.tools.has(node.name)
            and self.evolver.tools.spec(node.name).output_type == "video"
        ]
        return terminal[-1] if terminal else ""

    @staticmethod
    def _graph_fingerprint(graph: ToolPathGraph) -> str:
        payload = graph.to_dict()
        for key in ("graph_id", "skill_name", "description", "created_at", "updated_at", "stats"):
            payload.pop(key, None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _execution_fingerprint(graph: ToolPathGraph) -> str:
        return graph_behavior_fingerprint(graph)

    @staticmethod
    def _version_candidate(
        candidate: GraphPathCandidate,
        iteration: int,
        index: int,
    ) -> GraphPathCandidate:
        graph = ToolPathGraph.from_dict(candidate.graph.to_dict())
        base_name = graph.skill_name.split("__iter_")[0]
        suffix = f"__iter_{iteration:03d}_{index:02d}"
        graph.skill_name = f"{base_name}{suffix}"
        graph.graph_id = graph.skill_name
        for key in (
            "accepted",
            "exploratory",
            "admission",
            "program",
            "parent_program",
            "quality_gain",
            "candidate_score",
            "pass_rate",
            "stability",
            "estimated_cost",
            "validation_task_ids",
            "passed_validation_task_ids",
            "validation_seeds",
            "gate_evidence",
        ):
            graph.stats.pop(key, None)
        graph.stats["base_skill_name"] = base_name
        graph.stats["proposal_iteration"] = iteration
        graph.stats["defer_weighted_credit"] = True
        return GraphPathCandidate(
            graph=graph,
            edits=list(candidate.edits),
            reason=candidate.reason,
            parent_failure_types=list(candidate.parent_failure_types),
        )

    def _rollout_summary(self, task: VideoTask, graph: ToolPathGraph) -> TaskRolloutSummary:
        if self._h3_active():
            task = self._h3_replicate(task, self._evaluation_label(task))
        key = self._rollout_cache_key(task, graph)
        cached = None
        cached_key = key
        for candidate_key in self._rollout_cache_keys(task, graph):
            cached = self.cache.get(candidate_key)
            if cached is not None:
                cached_key = candidate_key
                break
        if cached is not None:
            artifact_path = cached.get("artifact_path")
            if cached.get("execution_error") or (
                artifact_path
                and not str(artifact_path).startswith(("http://", "https://"))
                and not Path(str(artifact_path)).exists()
            ):
                self.cache.delete(cached_key)
                cached = None
        if cached is not None:
            if cached_key != key:
                # Operational timeout changes do not alter a successful video.
                # Migrate the legacy hit, while failed hits above are discarded.
                self.cache.put(key, cached)
            cached["cache_hit"] = True
            cached["graph_id"] = graph.skill_name
            summary = TaskRolloutSummary(**cached)
            if self._h3_active():
                if not self._h3_local():
                    summary.seed_controlled = False
                    summary.provider_seed_control = False
                summary.replicate_label = self._evaluation_label(task)
                summary.evaluation_protocol = self._h3_summary_protocol()
            self._archive_execution(summary)
            return summary
        try:
            rollout = self.evolver.rollout(task, graph)
        except Exception as exc:
            if self._h3_active() and h3_run_should_stop(exc):
                raise
            failed_tool = exc.tool_name if isinstance(exc, GraphToolExecutionError) else None
            executed_tools = list(exc.executed_tools) if isinstance(exc, GraphToolExecutionError) else []
            if failed_tool and failed_tool not in executed_tools:
                executed_tools.append(failed_tool)
            summary = TaskRolloutSummary(
                task_id=task.task_id,
                graph_id=graph.skill_name,
                score=0.0,
                passed=False,
                failure_types=["tool_execution_error"],
                metric_scores={},
                estimated_cost=graph.estimated_cost(),
                tool_chain=executed_tools or graph.tool_names(),
                evaluation_seed=self._evaluation_label(task),
                seed_controlled=False,
                provider_seed_control=self._h3_local() if self._h3_active() else None,
                replicate_label=self._evaluation_label(task) if self._h3_active() else None,
                evaluation_protocol=self._h3_summary_protocol() if self._h3_active() else None,
                execution_error=f"{type(exc).__name__}: {exc}",
                failed_tool_name=failed_tool,
            )
            self._archive_execution(summary)
            self._record_failed_path_evidence(task, graph, summary)
            return summary
        summary = TaskRolloutSummary(
            task_id=task.task_id,
            graph_id=graph.skill_name,
            score=rollout.score,
            passed=rollout.evaluation.passed,
            failure_types=[
                failure.value for failure in rollout.failure.failure_types
            ] if rollout.failure else [],
            metric_scores={
                **{metric.name: metric.score for metric in rollout.evaluation.metrics},
                **(
                    {f"reward:{name}": score for name, score in rollout.reward.components.items()}
                    if rollout.reward is not None
                    else {}
                ),
            },
            estimated_cost=graph.estimated_cost(),
            tool_chain=list(rollout.artifact.tool_chain),
            active_metric_names=[
                *[metric.name for metric in rollout.evaluation.active_metrics],
                *(
                    [f"reward:{name}" for name in rollout.reward.components]
                    if rollout.reward is not None
                    else []
                ),
            ],
            evaluation_seed=self._evaluation_label(task),
            seed_controlled=self._rollout_seed_controlled(rollout),
            provider_seed_control=(rollout.artifact.metadata.get("provider_seed_control", False)
                                   if self._h3_local() else False if self._h3_active() else None),
            replicate_label=self._evaluation_label(task) if self._h3_active() else None,
            evaluation_protocol=self._h3_summary_protocol() if self._h3_active() else None,
            actual_tool_edges=rollout.artifact.metadata.get("h3_observed_tool_edges", []) if self._h3_active() else None,
            artifact_path=rollout.artifact.metadata.get("local_video_path"),
            runtime_log=rollout.artifact.metadata.get("runtime_log"),
            verifier_evidence=self._verifier_evidence(rollout),
            tool_provenance=self._tool_provenance(rollout.artifact.tool_chain),
            reward_objective=(
                rollout.reward.objective if rollout.reward is not None else "video_quality"
            ),
            reward_components=(
                dict(rollout.reward.components) if rollout.reward is not None else {}
            ),
            reward_weights=(
                dict(rollout.reward.weights) if rollout.reward is not None else {}
            ),
            missing_reward_metrics=(
                list(rollout.reward.missing_metrics) if rollout.reward is not None else []
            ),
        )
        self.cache.put(key, asdict(summary))
        self._archive_execution(summary)
        return summary

    def _rollout_cache_key(self, task: VideoTask, graph: ToolPathGraph) -> str:
        return self._rollout_cache_key_for_runtime(task, graph, self.runtime_signature)

    def _rollout_cache_keys(self, task: VideoTask, graph: ToolPathGraph) -> list[str]:
        keys = [self._rollout_cache_key(task, graph)]
        if self._h3_active():
            return keys
        runtime = self.runtime_signature
        if not isinstance(runtime, dict) or "timeout_seconds" not in runtime:
            return keys
        configured = os.environ.get("EVOVIDEO_CACHE_COMPAT_WAN_TIMEOUTS", "300")
        for value in configured.split(","):
            try:
                timeout = int(value.strip())
            except ValueError:
                continue
            if timeout == runtime.get("timeout_seconds"):
                continue
            legacy_runtime = dict(runtime)
            legacy_runtime["timeout_seconds"] = timeout
            legacy_key = self._rollout_cache_key_for_runtime(task, graph, legacy_runtime)
            if legacy_key not in keys:
                keys.append(legacy_key)
        return keys

    def _rollout_cache_key_for_runtime(
        self,
        task: VideoTask,
        graph: ToolPathGraph,
        runtime_signature: dict[str, Any] | str,
    ) -> str:
        if isinstance(runtime_signature, dict) and runtime_signature.get("vlm_provider") in {"qwen_video", "gemini"}:
            from evovideo_skill.conditioning_verifier import VERIFIER_PROTOCOL_VERSION

            runtime_signature = {**runtime_signature, "evidence_contract_version": VERIFIER_PROTOCOL_VERSION}
        if self._h3_active():
            if isinstance(runtime_signature, dict):
                runtime_signature = {
                    key: value for key, value in runtime_signature.items()
                    if key not in {"h3_max_api_calls", "h3_http_timeout_seconds", "h3_timeout_seconds",
                                   "h3_poll_interval_seconds", "timeout_seconds", "poll_interval_seconds"}
                }
            runtime_signature = {"runtime": runtime_signature, "evaluation_protocol": self._h3_evaluation_protocol()}
        return self.cache.make_key(
            "rollout",
            {
                "runtime": runtime_signature,
                "reward_router": getattr(
                    getattr(getattr(self, "evolver", None), "reward_router", None),
                    "VERSION",
                    "legacy-video-quality",
                ),
                "task": {
                    "task_id": task.task_id,
                    "prompt": task.prompt,
                    "mode": task.mode.value,
                    "duration_seconds": task.duration_seconds,
                    "reference_video": self._artifact_reference(task.reference_video),
                    "metadata": task.metadata,
                },
                "graph_execution": self._execution_fingerprint(graph),
            },
        )

    @staticmethod
    def _artifact_reference(value: str | None) -> dict[str, Any] | None:
        if not value:
            return None
        if value.startswith(("http://", "https://")):
            return {"uri": value}
        path = Path(value).expanduser()
        if not path.is_file():
            return {"path": value, "missing": True}
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "sha256": digest.hexdigest(),
            "size": path.stat().st_size,
        }

    def _record_failed_path_evidence(
        self,
        task: VideoTask,
        graph: ToolPathGraph,
        summary: TaskRolloutSummary,
    ) -> None:
        if self.weighted_tool_graph is None or not summary.execution_error:
            return
        self.weighted_tool_graph.record_rollout(
            task=task,
            graph_id=graph.graph_id,
            tools=summary.tool_chain or graph.tool_names(),
            quality=0.0,
            cost=summary.estimated_cost,
            success=False,
            seed_controlled=summary.seed_controlled,
            execution_error=True,
            failed_tool=summary.failed_tool_name,
            arena_variant=graph.stats.get("tool_arena_variant"),
            evaluation_seed=summary.evaluation_seed,
            event_id=(
                f"execution-error:{task.task_id}:{graph.graph_id}:"
                f"{summary.evaluation_seed}:{hashlib.sha256(summary.execution_error.encode()).hexdigest()[:12]}"
            ),
            **({
                "graph_behavior_fingerprint": self._execution_fingerprint(graph),
                # Exceptions lack exact completed node ids. Do not infer DAG edges from tool order.
                "actual_edges": summary.actual_tool_edges or [],
                "comparison_protocol": self._h3_comparison_protocol(),
            } if self._h3_active() else {}),
        )

    def _record_candidate_validation_credit(
        self,
        graph: ToolPathGraph,
        tasks: list[VideoTask],
        evaluation: ProgramEvaluation,
        *,
        positive_credit_allowed: bool,
    ) -> None:
        """Commit candidate path credit only after task-aware validation gates run."""
        if self.weighted_tool_graph is None:
            return
        task_by_id = {task.task_id: task for task in tasks}
        for summary in evaluation.rollouts:
            if summary.execution_error is not None:
                # Runtime errors are recorded at the point of failure with
                # failed-tool attribution and must not be counted twice.
                continue
            task = task_by_id.get(summary.task_id)
            if task is None:
                continue
            if summary.graph_id != graph.skill_name:
                # A routed validation may deliberately retain the parent path
                # for incompatible profiles. Do not attribute that rollout to
                # the candidate graph.
                continue
            seed_requested = summary.evaluation_seed is not None
            self.weighted_tool_graph.record_rollout(
                task=task,
                graph_id=graph.graph_id,
                tools=summary.tool_chain or graph.tool_names(),
                quality=summary.score,
                cost=summary.estimated_cost,
                success=positive_credit_allowed and summary.passed,
                seed_controlled=summary.seed_controlled if seed_requested else None,
                arena_variant=graph.stats.get("tool_arena_variant"),
                evaluation_seed=summary.evaluation_seed,
                positive_credit_allowed=positive_credit_allowed,
                event_id=(
                    f"validation-credit:{summary.task_id}:{graph.graph_id}:"
                    f"{summary.evaluation_seed}"
                ),
                **({
                    "graph_behavior_fingerprint": self._execution_fingerprint(graph),
                    "actual_edges": summary.actual_tool_edges or [],
                    "comparison_protocol": self._h3_comparison_protocol(),
                } if self._h3_active() else {}),
            )

    def _archive_execution(self, summary: TaskRolloutSummary) -> None:
        self.graph_archive.record_execution(
            task_id=summary.task_id,
            graph_id=summary.graph_id,
            score=summary.score,
            passed=summary.passed,
            tool_chain=summary.tool_chain,
            estimated_cost=summary.estimated_cost,
            cache_hit=summary.cache_hit,
            failure_types=summary.failure_types,
            metric_scores=summary.metric_scores,
            active_metric_names=summary.active_metric_names,
            evaluation_seed=summary.evaluation_seed,
            seed_controlled=summary.seed_controlled,
            execution_error=summary.execution_error,
            failed_tool_name=summary.failed_tool_name,
            artifact_path=summary.artifact_path,
            runtime_log=summary.runtime_log,
            tool_provenance=summary.tool_provenance,
            reward_objective=summary.reward_objective,
            reward_components=summary.reward_components,
            reward_weights=summary.reward_weights,
            missing_reward_metrics=summary.missing_reward_metrics,
            verifier_evidence=summary.verifier_evidence,
        )

    def _archive_rollout(self, rollout: GraphPathRollout) -> None:
        self.graph_archive.record_execution(
            task_id=rollout.task.task_id,
            graph_id=rollout.graph.skill_name,
            score=rollout.score,
            passed=rollout.evaluation.passed,
            tool_chain=list(rollout.artifact.tool_chain),
            estimated_cost=rollout.graph.estimated_cost(),
            cache_hit=False,
            failure_types=[item.value for item in rollout.failure.failure_types]
            if rollout.failure
            else [],
            metric_scores={item.name: item.score for item in rollout.evaluation.metrics},
            active_metric_names=[item.name for item in rollout.evaluation.active_metrics],
            evaluation_seed=self._evaluation_label(rollout.task),
            seed_controlled=self._rollout_seed_controlled(rollout),
            execution_error=None,
            failed_tool_name=None,
            artifact_path=rollout.artifact.metadata.get("local_video_path"),
            runtime_log=rollout.artifact.metadata.get("runtime_log"),
            verifier_evidence=self._verifier_evidence(rollout),
            tool_provenance=self._tool_provenance(rollout.artifact.tool_chain),
            reward_objective=(
                rollout.reward.objective if rollout.reward is not None else "video_quality"
            ),
            reward_components=(
                dict(rollout.reward.components) if rollout.reward is not None else {}
            ),
            reward_weights=(
                dict(rollout.reward.weights) if rollout.reward is not None else {}
            ),
            missing_reward_metrics=(
                list(rollout.reward.missing_metrics) if rollout.reward is not None else []
            ),
        )

    @staticmethod
    def _verifier_evidence(rollout: GraphPathRollout) -> dict[str, Any]:
        result = rollout.artifact.metadata.get("vlm_evaluation") or {}
        if not isinstance(result, dict):
            return {}
        return {key: result[key] for key in (
            "model", "verifier_cache", "candidate_frame_count", "source_frame_count",
            "criterion_evidence", "failed_segments", "verification_metadata",
        ) if key in result}

    def _tool_provenance(self, tool_chain: list[str]) -> dict[str, dict[str, Any]]:
        provenance: dict[str, dict[str, Any]] = {}
        for name in dict.fromkeys(tool_chain):
            if not self.evolver.tools.has(name):
                continue
            spec = self.evolver.tools.spec(name)
            provenance[name] = {
                "backend": spec.backend,
                "model": spec.model,
                "provenance": spec.provenance,
                "capability": spec.capability,
            }
        return provenance

    def _archive_candidate_audits(
        self,
        iteration: int,
        failures: list[GraphPathRollout],
    ) -> None:
        for audit in self.evolver.last_candidate_audit:
            self.graph_archive.record_candidate_audit(
                {
                    "iteration": iteration,
                    "task_ids": [item.task.task_id for item in failures],
                    "failure_types": sorted(
                        {
                            failure.value
                            for item in failures
                            if item.failure is not None
                            for failure in item.failure.failure_types
                        }
                    ),
                    **audit,
                }
            )

    def _h3_active(self) -> bool:
        registry = getattr(getattr(self, "evolver", None), "tools", None)
        available_names = getattr(registry, "available_names", None)
        return (callable(available_names) and h3_registry_active(available_names())) or self._h3_local()

    def _h3_local(self) -> bool:
        return h3_local_active(getattr(getattr(self, "evolver", None), "tools", None),
                               getattr(self, "runtime_signature", None))

    def _h3_evaluation_protocol(self) -> str:
        return H3_LOCAL_EVALUATION_PROTOCOL if self._h3_local() else "h3_unseeded_replicates_v1"

    def _h3_summary_protocol(self) -> str:
        return H3_LOCAL_EVALUATION_PROTOCOL if self._h3_local() else "h3_uncontrolled_replicates"

    def _h3_comparison_protocol(self) -> str:
        return H3_LOCAL_COMPARISON_PROTOCOL if self._h3_local() else H3_COMPARISON_PROTOCOL

    def _rollout_seed_controlled(self, rollout: GraphPathRollout) -> bool:
        if self._h3_local():
            return h3_local_seed_applied(rollout.artifact.metadata, self._evaluation_label(rollout.task))
        if self._h3_active():
            return False
        return rollout.artifact.metadata.get(
            "generation_seed_applied",
            all(self.evolver.tools.spec(name).backend == "builtin" for name in rollout.artifact.tool_chain),
        )

    @staticmethod
    def _evaluation_label(task: VideoTask) -> int | None:
        metadata = task.metadata or {}
        return metadata.get("replicate_label", metadata.get("evaluation_seed", metadata.get("generation_seed")))

    def _h3_replicate(self, task: VideoTask, label: int | None) -> VideoTask:
        metadata = dict(task.metadata or {})
        metadata.pop("exploration_seed", None)
        if self._h3_local() and label is None:
            label = next(iter(self.config.evaluation_seeds), 42)
        if self._h3_local() and type(label) is not int:
            raise ValueError("local-h3 requires an integer replicate generation seed")
        # Both clients use this field for ledger identity; only local H3 applies it.
        metadata["generation_seed"] = label
        metadata["evaluation_seed"] = label
        metadata["replicate_label"] = label
        metadata["provider_seed_control"] = self._h3_local()
        metadata["evaluation_protocol"] = self._h3_evaluation_protocol()
        return replace(task, metadata=metadata)

    def _h3_replicate_evidence(
        self, paired: list[tuple[TaskRolloutSummary, TaskRolloutSummary]],
    ) -> dict[str, Any]:
        if not self._h3_active():
            return {}
        labels: dict[str, list[int | None]] = {}
        for base, child in paired:
            if (base.task_id, base.evaluation_seed) == (child.task_id, child.evaluation_seed):
                labels.setdefault(base.task_id, []).append(base.evaluation_seed)
        minimum = max(3 if self._h3_local() else 1, int(self.config.h3_min_replicates))
        counts = {task_id: len({label for label in values if label is not None})
                  for task_id, values in labels.items()}
        return {
            "h3_min_replicates": minimum,
            "h3_replicates_per_task": counts,
            "h3_replicate_gate_passed": bool(counts) and all(
                count >= minimum and len(labels[task_id]) == count for task_id, count in counts.items()
            ) and (not self._h3_local() or all(
                type(base.evaluation_seed) is int and type(child.evaluation_seed) is int
                and base.seed_controlled is True and child.seed_controlled is True
                and base.provider_seed_control is True and child.provider_seed_control is True
                for base, child in paired
            )),
            "provider_seed_control": self._h3_local(),
            "comparison_protocol": self._h3_comparison_protocol(),
            "statistical_gain_guaranteed": False,
        }

    def _exploration_variant(self, task: VideoTask, iteration: int) -> VideoTask:
        seeds = list(dict.fromkeys(int(seed) for seed in self.config.evaluation_seeds))
        if self._h3_active():
            label = self._evaluation_label(task)
            if label is None and seeds:
                visit = self._exploration_visits.get(task.task_id, 0)
                label = seeds[visit % len(seeds)]
                self._exploration_visits[task.task_id] = visit + 1
            return self._h3_replicate(task, label)
        if not seeds or (task.metadata or {}).get("generation_seed") is not None:
            return task
        seed = seeds[(max(1, iteration) - 1) % len(seeds)]
        return VideoTask(
            task_id=task.task_id,
            prompt=task.prompt,
            mode=task.mode,
            duration_seconds=task.duration_seconds,
            reference_video=task.reference_video,
            metadata={**(task.metadata or {}), "generation_seed": seed, "exploration_seed": True},
        )

    def _passes_validation_gate(
        self,
        parent: ProgramEvaluation,
        candidate: ProgramEvaluation,
    ) -> tuple[bool, dict[str, Any]]:
        paired = [
            (base, child)
            for base, child in zip(parent.rollouts, candidate.rollouts)
            if base.execution_error is None and child.execution_error is None
            and (not self._h3_active() or (base.task_id, base.evaluation_seed) == (child.task_id, child.evaluation_seed))
        ]
        replicate_evidence = self._h3_replicate_evidence(paired)
        gains = [child.score - base.score for base, child in paired]
        improved_fraction = sum(value > 0.0 for value in gains) / max(1, len(gains))
        controlled = [
            base.seed_controlled is True and child.seed_controlled is True
            for base, child in paired
            if base.evaluation_seed is not None or child.evaluation_seed is not None
        ]
        seed_control_fraction = sum(controlled) / len(controlled) if controlled else None
        expected_pairs = (max if self._h3_active() else min)(len(parent.rollouts), len(candidate.rollouts))
        paired_coverage = len(paired) / max(1, expected_pairs)
        aggregate_quality_gain = candidate.metrics.quality - parent.metrics.quality
        gain = sum(gains) / len(gains) if gains else 0.0
        task_guard = self._task_metric_guard(paired)
        passed = (
            gain >= self.config.min_quality_gain
            and candidate.metrics.quality >= parent.metrics.quality
            and candidate.metrics.pass_rate >= parent.metrics.pass_rate
            and improved_fraction >= 0.5
            and paired_coverage == 1.0
            and candidate.metrics.candidate_tool_coverage == 1.0
            and (replicate_evidence["h3_replicate_gate_passed"] if self._h3_active()
                 else seed_control_fraction is None or seed_control_fraction == 1.0)
            and task_guard["task_metric_guard_passed"]
        )
        return passed, {
            "mean_paired_gain": gain,
            "aggregate_quality_gain": aggregate_quality_gain,
            "paired_improvement_fraction": improved_fraction,
            "paired_regression_fraction": sum(value < -1e-9 for value in gains) / max(1, len(gains)),
            "paired_tie_fraction": sum(abs(value) <= 1e-9 for value in gains) / max(1, len(gains)),
            "paired_gains": gains,
            "baseline_pass_rate": parent.metrics.pass_rate,
            "candidate_pass_rate": candidate.metrics.pass_rate,
            "seed_control_fraction": seed_control_fraction,
            "seed_control_required": bool(controlled) and (self._h3_local() or not self._h3_active()),
            "comparison_protocol": self._h3_comparison_protocol() if self._h3_active() else "paired_generation_seeds",
            "valid_pair_count": len(paired),
            "expected_pair_count": expected_pairs,
            "paired_coverage": paired_coverage,
            "candidate_tool_coverage": candidate.metrics.candidate_tool_coverage,
            **task_guard,
            **replicate_evidence,
        }

    def _task_metric_guard(
        self,
        paired: list[tuple[TaskRolloutSummary, TaskRolloutSummary]],
    ) -> dict[str, Any]:
        """Prevent visual-quality gains from masking task-semantic regressions."""
        component_deltas: dict[str, list[float]] = {}
        weighted_pair_gains: list[float] = []
        for base, child in paired:
            names = sorted(
                (set(base.reward_components) & set(child.reward_components))
                - {"video_quality"}
            )
            active = [
                name for name in names
                if max(base.reward_weights.get(name, 0.0), child.reward_weights.get(name, 0.0)) > 0.0
            ]
            if not active:
                continue
            weights = {
                name: max(base.reward_weights.get(name, 0.0), child.reward_weights.get(name, 0.0))
                for name in active
            }
            total = sum(weights.values())
            pair_gain = 0.0
            for name in active:
                delta = float(child.reward_components[name]) - float(base.reward_components[name])
                component_deltas.setdefault(name, []).append(delta)
                pair_gain += weights[name] * delta
            weighted_pair_gains.append(pair_gain / total if total else 0.0)

        applicable = bool(component_deltas)
        max_regression = max(
            (max(0.0, -delta) for values in component_deltas.values() for delta in values),
            default=0.0,
        )
        mean_task_gain = (
            sum(weighted_pair_gains) / len(weighted_pair_gains)
            if weighted_pair_gains
            else 0.0
        )
        # Metric values come from decimal VLM scores, so a boundary delta such
        # as 1.00 - 0.95 can be represented as 0.050000000000000044. Keep the
        # configured guard inclusive without rejecting that numerical noise.
        numeric_tolerance = 1e-9
        passed = (
            not applicable
            or (
                max_regression
                <= self.config.max_task_metric_regression + numeric_tolerance
                and mean_task_gain + numeric_tolerance
                >= self.config.min_task_metric_gain
            )
        )
        return {
            "task_metric_guard_applicable": applicable,
            "task_metric_guard_passed": passed,
            "mean_task_metric_gain": mean_task_gain,
            "max_task_metric_regression": max_regression,
            "max_allowed_task_metric_regression": self.config.max_task_metric_regression,
            "min_required_task_metric_gain": self.config.min_task_metric_gain,
            "task_metric_mean_deltas": {
                name: sum(values) / len(values)
                for name, values in sorted(component_deltas.items())
            },
        }

    def _passes_exploratory_gate(
        self,
        parent: ProgramEvaluation,
        candidate: ProgramEvaluation,
    ) -> tuple[bool, dict[str, Any]]:
        paired = [
            (base, child)
            for base, child in zip(parent.rollouts, candidate.rollouts)
            if base.execution_error is None and child.execution_error is None
            and (not self._h3_active() or (base.task_id, base.evaluation_seed) == (child.task_id, child.evaluation_seed))
        ]
        replicate_evidence = self._h3_replicate_evidence(paired)
        gains = [child.score - base.score for base, child in paired]
        expected_pairs = (max if self._h3_active() else min)(len(parent.rollouts), len(candidate.rollouts))
        paired_coverage = len(paired) / max(1, expected_pairs)
        controlled = [
            base.seed_controlled is True and child.seed_controlled is True
            for base, child in paired
            if base.evaluation_seed is not None or child.evaluation_seed is not None
        ]
        seed_control_fraction = sum(controlled) / len(controlled) if controlled else None
        parent_scores = [base.score for base, _ in paired]
        candidate_scores = [child.score for _, child in paired]
        aggregate_quality_gain = candidate.metrics.quality - parent.metrics.quality
        mean_gain = sum(gains) / len(gains) if gains else 0.0
        worst_case_floor_gain = (
            min(candidate_scores) - min(parent_scores)
            if parent_scores and candidate_scores
            else float("-inf")
        )
        worst_paired_gain = min(gains, default=float("-inf"))
        stability_gain = candidate.metrics.stability - parent.metrics.stability
        max_pair_regression = max((max(0.0, -gain) for gain in gains), default=float("inf"))
        task_guard = self._task_metric_guard(paired)
        passed = (
            bool(paired)
            and mean_gain >= self.config.min_quality_gain
            and candidate.metrics.pass_rate >= parent.metrics.pass_rate
            and worst_case_floor_gain >= self.config.exploratory_min_worst_seed_gain
            and stability_gain >= self.config.exploratory_min_stability_gain
            and max_pair_regression <= self.config.exploratory_max_pair_regression
            and paired_coverage == 1.0
            and candidate.metrics.candidate_tool_coverage == 1.0
            and (replicate_evidence["h3_replicate_gate_passed"] if self._h3_active()
                 else seed_control_fraction is None or seed_control_fraction == 1.0)
            and task_guard["task_metric_guard_passed"]
        )
        return passed, {
            "exploratory_mean_gain": mean_gain,
            "seed_control_fraction": seed_control_fraction,
            "seed_control_required": bool(controlled) and (self._h3_local() or not self._h3_active()),
            "comparison_protocol": self._h3_comparison_protocol() if self._h3_active() else "paired_generation_seeds",
            "aggregate_quality_gain": aggregate_quality_gain,
            # Keep the legacy key for report compatibility, but expose its actual
            # floor-recovery meaning and the true minimum paired delta separately.
            "worst_seed_gain": worst_case_floor_gain,
            "worst_case_floor_gain": worst_case_floor_gain,
            "worst_paired_gain": worst_paired_gain,
            "stability_gain": stability_gain,
            "max_pair_regression": max_pair_regression,
            "exploratory_min_worst_seed_gain": self.config.exploratory_min_worst_seed_gain,
            "exploratory_min_stability_gain": self.config.exploratory_min_stability_gain,
            "exploratory_max_pair_regression": self.config.exploratory_max_pair_regression,
            "candidate_tool_coverage": candidate.metrics.candidate_tool_coverage,
            **task_guard,
            **replicate_evidence,
        }

    def _evaluation_variants(self, tasks: list[VideoTask]) -> list[VideoTask]:
        seeds = list(dict.fromkeys(int(seed) for seed in self.config.evaluation_seeds))
        if self._h3_active():
            return [
                self._h3_replicate(task, label)
                for task in tasks
                for label in (
                    [self._evaluation_label(task)] if (task.metadata or {}).get("evaluation_protocol") == self._h3_evaluation_protocol()
                    else seeds or [self._evaluation_label(task)]
                )
            ]
        if not seeds:
            return list(tasks)
        variants: list[VideoTask] = []
        for task in tasks:
            if (task.metadata or {}).get("generation_seed") is not None:
                variants.append(task)
                continue
            for seed in seeds:
                variants.append(
                    VideoTask(
                        task_id=task.task_id,
                        prompt=task.prompt,
                        mode=task.mode,
                        duration_seconds=task.duration_seconds,
                        reference_video=task.reference_video,
                        metadata={**(task.metadata or {}), "generation_seed": seed},
                    )
                )
        return variants

    def _save_checkpoint(self, iteration: int, no_improvement: int) -> None:
        self._write_json(
            self.checkpoint_path,
            {
                "iteration": iteration,
                "no_improvement": no_improvement,
                "mutation_searches_used": self._mutation_searches_used,
                "max_mutation_searches": self.config.max_mutation_searches,
                "sampler": self.sampler.state_dict(),
                "exploration_visits": self._exploration_visits,
                "verifier_protocol": self._verifier_protocol(),
                "frontier": [program.name for program in self.registry.frontier()],
                "updated_at": utc_now(),
            },
        )

    def _verifier_protocol(self) -> dict[str, Any]:
        protocol = {key: self.runtime_signature.get(key) for key in (
            "vlm_provider", "vlm_model", "vlm_base_url", "vlm_video_fps", "vlm_review_fps",
            "vlm_max_images", "sample_frames", "vlm_cache_namespace", "h3_audio_verifier_command",
        )}
        if protocol["vlm_provider"] in {"qwen_video", "gemini"}:
            from evovideo_skill.conditioning_verifier import VERIFIER_PROTOCOL_VERSION

            protocol["evidence_contract_version"] = VERIFIER_PROTOCOL_VERSION
        return protocol

    def _remaining_mutation_searches(self, default: int | None = None) -> int | None:
        limit = self.config.max_mutation_searches
        if limit is None:
            return default
        return max(0, int(limit) - self._mutation_searches_used)

    def _mutation_search_budget_exhausted(self) -> bool:
        remaining = self._remaining_mutation_searches()
        return remaining is not None and remaining <= 0

    def _append_iteration(self, record: EvolutionIteration) -> None:
        with self.iteration_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def _read_iteration_records(self) -> list[EvolutionIteration]:
        if not self.config.continue_mode or not self.iteration_log_path.exists():
            return []
        records = []
        with self.iteration_log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(EvolutionIteration(**json.loads(line)))
        return records

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return default

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        temporary.replace(path)
