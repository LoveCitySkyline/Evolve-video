from __future__ import annotations

import csv
import json
import os
import shutil
import socket
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - the harness targets Linux/macOS.
    fcntl = None

from evovideo_skill.benchmarks import BenchmarkSuite, load_suites
from evovideo_skill.benchmark_assets import BenchmarkAssetBootstrapper
from evovideo_skill.evolution_data import stratified_task_split
from evovideo_skill.evolution_loop import EvolutionLoopConfig, EvolutionLoopResult, GraphSelfImprovingLoop
from evovideo_skill.foundation_skills import FoundationSkillConsolidator
from evovideo_skill.graph_composition import HistoricalPathComposer
from evovideo_skill.graph_evolver import GraphToolPathEvolver
from evovideo_skill.graph_mining import GraphMotifMiner
from evovideo_skill.graph_router import FoundationAwareGraphRouter
from evovideo_skill.graph_skill import GraphSkillMemory
from evovideo_skill.models import VideoTask, utc_now
from evovideo_skill.online_foundation import OnlineFoundationConfig, TaskConditionedFoundationPromoter
from evovideo_skill.program_registry import ProgramRegistry
from evovideo_skill.runtime import RuntimeSettings, build_evaluator_suite, build_graph_mutation_proposer, build_runtime, with_env_overrides
from evovideo_skill.skill_memory import SkillMemory
from evovideo_skill.weighted_tool_graph import WeightedToolGraphMemory


@dataclass
class HarnessConfig:
    name: str = "video_skill_graph_harness"
    output_dir: str = "outputs/harness_runs"
    task_files: list[str] = field(default_factory=lambda: ["examples/graph_evolution_tasks.json"])
    limit_per_suite: int | None = None
    auto_bootstrap_assets: bool = False
    asset_bootstrap_dir: str | None = None
    asset_bootstrap_seed: int = 20260827
    asset_bootstrap_force: bool = False
    asset_bootstrap_strict: bool = True
    max_candidates: int = 8
    max_mutation_searches: int | None = 8
    min_quality_gain: float = 0.02
    max_task_metric_regression: float = 0.05
    min_task_metric_gain: float = 0.0
    evolution_iterations: int = 5
    frontier_size: int = 3
    no_improvement_limit: int = 3
    selection_strategy: str = "best"
    categories_per_batch: int = 3
    samples_per_category: int = 1
    homogeneous_task_batches: bool = False
    train_ratio: float = 0.5
    validation_ratio: float = 0.25
    split_seed: int = 0
    cache_enabled: bool = True
    evaluation_seeds: list[int] = field(default_factory=lambda: [42, 123, 456])
    continue_mode: bool = False
    program_cost_weight: float = 0.03
    auto_consolidate_foundation: bool = True
    min_support: int = 2
    min_utility: float = 0.02
    max_motif_size: int = 4
    motif_cost_weight: float = 0.08
    mdl_weight: float = 0.02
    max_foundation_skills: int = 5
    path_prior_weight: float = 0.5
    exploration_weight: float = 0.15
    historical_merge_candidates: int = 2
    online_foundation_enabled: bool = True
    online_foundation_interval: int = 1
    online_foundation_min_support: int = 2
    online_foundation_min_advantage: float = 0.0
    online_foundation_min_stability: float = 0.8
    online_foundation_min_gain: float = 0.0
    online_foundation_max_per_category: int = 2
    exploratory_admission_enabled: bool = True
    exploratory_min_worst_seed_gain: float = 0.05
    exploratory_min_stability_gain: float = 0.02
    exploratory_max_pair_regression: float = 0.03
    runtime_failure_circuit_breaker: bool = True
    full_suite_evaluation: bool = False
    allow_seed_holdout_validation: bool = False
    seed_holdout_validation_seed: int = 123
    near_miss_repair_enabled: bool = False
    warm_start_run_dir: str | None = None
    warm_start_required: bool = False
    ablations: list[str] = field(default_factory=lambda: ["full"])
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)

    @classmethod
    def from_file(cls, path: str | Path) -> "HarnessConfig":
        config_path = Path(path)
        with config_path.open("rb") as handle:
            if config_path.suffix.lower() == ".toml":
                try:
                    import tomllib
                except ModuleNotFoundError:
                    import tomli as tomllib
                payload = tomllib.load(handle)
            else:
                payload = json.loads(handle.read().decode("utf-8"))
        payload = dict(payload)
        runtime_payload = payload.pop("runtime", {})
        evolution_payload = payload.pop("evolution", {})
        dataset_payload = payload.pop("dataset", {})
        foundation_payload = payload.pop("foundation", {})
        aliases = {
            "iterations": "evolution_iterations",
            "max_iterations": "evolution_iterations",
            "max_proposals": "max_candidates",
            "mutation_search_budget": "max_mutation_searches",
            "exploratory_enabled": "exploratory_admission_enabled",
        }
        for key, value in evolution_payload.items():
            payload[aliases.get(key, key)] = value
        dataset_aliases = {
            "task_files": "task_files",
            "limit_per_suite": "limit_per_suite",
            "train_ratio": "train_ratio",
            "validation_ratio": "validation_ratio",
            "seed": "split_seed",
        }
        for key, value in dataset_payload.items():
            if key in dataset_aliases:
                payload[dataset_aliases[key]] = value
        foundation_aliases = {
            "enabled": "auto_consolidate_foundation",
            "min_support": "min_support",
            "min_utility": "min_utility",
            "max_motif_size": "max_motif_size",
            "cost_weight": "motif_cost_weight",
            "mdl_weight": "mdl_weight",
            "max_skills": "max_foundation_skills",
            "online_enabled": "online_foundation_enabled",
            "online_interval": "online_foundation_interval",
            "online_min_support": "online_foundation_min_support",
            "online_min_advantage": "online_foundation_min_advantage",
            "online_min_stability": "online_foundation_min_stability",
            "online_min_gain": "online_foundation_min_gain",
            "online_max_per_category": "online_foundation_max_per_category",
        }
        for key, value in foundation_payload.items():
            if key in foundation_aliases:
                payload[foundation_aliases[key]] = value
        if os.environ.get("EVOVIDEO_MAX_MUTATION_SEARCHES") is not None:
            payload["max_mutation_searches"] = int(os.environ["EVOVIDEO_MAX_MUTATION_SEARCHES"])
        if os.environ.get("EVOVIDEO_EVOLUTION_ITERATIONS") is not None:
            payload["evolution_iterations"] = int(os.environ["EVOVIDEO_EVOLUTION_ITERATIONS"])
        if os.environ.get("EVOVIDEO_MAX_CANDIDATES") is not None:
            payload["max_candidates"] = int(os.environ["EVOVIDEO_MAX_CANDIDATES"])
        if os.environ.get("EVOVIDEO_NO_IMPROVEMENT_LIMIT") is not None:
            payload["no_improvement_limit"] = int(os.environ["EVOVIDEO_NO_IMPROVEMENT_LIMIT"])
        if os.environ.get("EVOVIDEO_MAX_TASK_METRIC_REGRESSION") is not None:
            payload["max_task_metric_regression"] = float(
                os.environ["EVOVIDEO_MAX_TASK_METRIC_REGRESSION"]
            )
        if os.environ.get("EVOVIDEO_AUTO_BOOTSTRAP_ASSETS") is not None:
            payload["auto_bootstrap_assets"] = os.environ["EVOVIDEO_AUTO_BOOTSTRAP_ASSETS"] == "1"
        if os.environ.get("EVOVIDEO_ASSET_BOOTSTRAP_DIR"):
            payload["asset_bootstrap_dir"] = os.environ["EVOVIDEO_ASSET_BOOTSTRAP_DIR"]
        if os.environ.get("EVOVIDEO_ASSET_BOOTSTRAP_SEED") is not None:
            payload["asset_bootstrap_seed"] = int(os.environ["EVOVIDEO_ASSET_BOOTSTRAP_SEED"])
        if os.environ.get("EVOVIDEO_ASSET_BOOTSTRAP_FORCE") is not None:
            payload["asset_bootstrap_force"] = os.environ["EVOVIDEO_ASSET_BOOTSTRAP_FORCE"] == "1"
        if os.environ.get("EVOVIDEO_WARM_START_RUN_DIR"):
            payload["warm_start_run_dir"] = os.environ["EVOVIDEO_WARM_START_RUN_DIR"]
        payload["runtime"] = RuntimeSettings(**runtime_payload)
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


@dataclass
class HarnessRunRecord:
    run_id: str
    suite_name: str
    ablation: str
    memory_dir: str
    task_count: int
    baseline_mean_score: float
    selected_graph_count: int
    accepted_candidate_count: int
    rejected_candidate_count: int
    mutation_searches_used: int
    max_mutation_searches: int | None
    mutation_search_budget_exhausted: bool
    foundation_skill_count: int
    online_foundation_skill_count: int
    weighted_task_class_count: int
    mined_motif_count: int
    mean_quality_gain: float
    mean_candidate_score: float
    full_suite_baseline_score: float | None
    full_suite_final_score: float | None
    full_suite_quality_gain: float | None
    full_suite_task_count: int
    mean_cost: float
    route_count: int
    report_path: str
    visualization_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HarnessReport:
    config: dict[str, Any]
    records: list[HarnessRunRecord]
    output_dir: str
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "records": [record.to_dict() for record in self.records],
            "output_dir": self.output_dir,
            "created_at": self.created_at,
        }


class GraphHarnessRunner:
    def __init__(self, config: HarnessConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> HarnessReport:
        runtime = with_env_overrides(self.config.runtime)
        lock_paths = {
            (self.output_dir / ".harness.lock").resolve(),
            (Path(runtime.video_output_dir) / ".evovideo-runtime.lock").resolve(),
        }
        with ExitStack() as stack:
            for lock_path in sorted(lock_paths, key=str):
                stack.enter_context(self._exclusive_lock(lock_path))
            return self._run_locked()

    def _run_locked(self) -> HarnessReport:
        suites = load_suites(self.config.task_files, limit_per_suite=self.config.limit_per_suite)
        records: list[HarnessRunRecord] = []
        self._write_json(self.output_dir / "config.json", self.config.to_dict())
        for suite in suites:
            for ablation in self.config.ablations:
                records.append(self._run_one(suite, ablation))
        report = HarnessReport(self.config.to_dict(), records, str(self.output_dir))
        self._write_json(self.output_dir / "harness_report.json", report.to_dict())
        self._write_csv(self.output_dir / "harness_summary.csv", records)
        return report

    def _run_one(self, suite: BenchmarkSuite, ablation: str) -> HarnessRunRecord:
        run_id = f"{suite.name}_{ablation}".replace("/", "_")
        run_dir = self.output_dir / run_id
        memory_dir = run_dir / "memory"
        if run_dir.exists() and not self.config.continue_mode:
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        settings = with_env_overrides(self._runtime_for_ablation(ablation))
        settings.agent_state_dir = str(run_dir / "agent_state")
        Path(settings.agent_state_dir).mkdir(parents=True, exist_ok=True)
        print(f"  isolated agent state: {Path(settings.agent_state_dir).resolve()}")
        registry, vlm_augmenter = build_runtime(settings)
        if self.config.auto_bootstrap_assets:
            asset_report = BenchmarkAssetBootstrapper(
                registry,
                output_dir=self.config.asset_bootstrap_dir,
                seed=self.config.asset_bootstrap_seed,
                force=self.config.asset_bootstrap_force,
                strict=False,
            ).run(suite.tasks)
            asset_report_path = run_dir / "asset_bootstrap_report.json"
            self._write_json(asset_report_path, asset_report.to_dict())
            print(
                "  benchmark assets: "
                f"generated={asset_report.generated_count} reused={asset_report.reused_count} "
                f"failed={asset_report.failed_count}",
                flush=True,
            )
            if asset_report.failed_count and self.config.asset_bootstrap_strict:
                raise RuntimeError(
                    f"benchmark asset bootstrap failed; inspect {asset_report_path.resolve()}"
                )
        graph_memory = GraphSkillMemory(memory_dir)
        skill_memory = SkillMemory(memory_dir)
        warm_start_graph_ids = self._import_warm_start_frontier(graph_memory)
        weighted_graph = WeightedToolGraphMemory(
            run_dir / "evolution" / "weighted_tool_graph.json",
            cost_weight=self.config.program_cost_weight,
            exploration_weight=self.config.exploration_weight,
        )
        path_composer = HistoricalPathComposer(
            weighted_graph,
            max_candidates=self.config.historical_merge_candidates,
            tool_registry=registry,
        )
        evolver = GraphToolPathEvolver(
            skill_memory=skill_memory,
            graph_memory=graph_memory,
            tools=registry,
            evaluators=build_evaluator_suite(settings),
            vlm_augmenter=vlm_augmenter,
            min_quality_gain=self.config.min_quality_gain,
            mutation_proposer=build_graph_mutation_proposer(settings, registry),
            template_mutations_enabled=settings.template_mutations_enabled,
            weighted_tool_graph=weighted_graph,
            historical_path_composer=path_composer,
        )
        dataset = stratified_task_split(
            suite.tasks,
            train_ratio=self.config.train_ratio,
            validation_ratio=self.config.validation_ratio,
            seed=self.config.split_seed,
        )
        loop = GraphSelfImprovingLoop(
            config=EvolutionLoopConfig(
                max_iterations=self.config.evolution_iterations,
                frontier_size=self.config.frontier_size,
                no_improvement_limit=self.config.no_improvement_limit,
                selection_strategy=self.config.selection_strategy,
                categories_per_batch=self.config.categories_per_batch,
                samples_per_category=self.config.samples_per_category,
                homogeneous_task_batches=self.config.homogeneous_task_batches,
                max_proposals_per_iteration=self.config.max_candidates,
                max_mutation_searches=self.config.max_mutation_searches,
                min_quality_gain=self.config.min_quality_gain,
                max_task_metric_regression=self.config.max_task_metric_regression,
                min_task_metric_gain=self.config.min_task_metric_gain,
                cache_enabled=self.config.cache_enabled,
                evaluation_seeds=self.config.evaluation_seeds,
                continue_mode=self.config.continue_mode,
                cost_weight=self.config.program_cost_weight,
                path_prior_weight=self.config.path_prior_weight,
                online_foundation_enabled=self.config.online_foundation_enabled,
                online_foundation_interval=self.config.online_foundation_interval,
                online_foundation_min_gain=self.config.online_foundation_min_gain,
                exploratory_admission_enabled=self.config.exploratory_admission_enabled,
                exploratory_min_worst_seed_gain=self.config.exploratory_min_worst_seed_gain,
                exploratory_min_stability_gain=self.config.exploratory_min_stability_gain,
                exploratory_max_pair_regression=self.config.exploratory_max_pair_regression,
                runtime_failure_circuit_breaker=self.config.runtime_failure_circuit_breaker,
                full_suite_evaluation=self.config.full_suite_evaluation,
                allow_seed_holdout_validation=self.config.allow_seed_holdout_validation,
                seed_holdout_validation_seed=self.config.seed_holdout_validation_seed,
                near_miss_repair_enabled=self.config.near_miss_repair_enabled,
            ),
            dataset=dataset,
            evolver=evolver,
            graph_memory=graph_memory,
            state_dir=run_dir / "evolution",
            runtime_signature=asdict(settings),
            weighted_tool_graph=weighted_graph,
            foundation_promoter=TaskConditionedFoundationPromoter(
                weighted_graph,
                graph_memory=graph_memory,
                config=OnlineFoundationConfig(
                    min_support=self.config.online_foundation_min_support,
                    min_advantage=self.config.online_foundation_min_advantage,
                    min_stability=self.config.online_foundation_min_stability,
                    max_skills_per_category=self.config.online_foundation_max_per_category,
                ),
                tool_registry=registry,
            ),
            warm_start_graph_ids=warm_start_graph_ids,
        )
        evolution = loop.run()
        foundation_report = None
        if self.config.auto_consolidate_foundation and ablation != "no_foundation":
            foundation_report = FoundationSkillConsolidator(
                graph_memory=graph_memory,
                skill_memory=skill_memory,
                miner=GraphMotifMiner(
                    min_support=self.config.min_support,
                    min_utility=self.config.min_utility,
                    max_motif_size=self.config.max_motif_size,
                    cost_weight=self.config.motif_cost_weight,
                    mdl_weight=self.config.mdl_weight,
                ),
                max_foundation_skills=self.config.max_foundation_skills,
            ).consolidate()
            for foundation in foundation_report.foundation_skills:
                loop.graph_archive.record_graph(
                    foundation.graph,
                    stage="offline_foundation_consolidation",
                    status="promoted",
                    iteration=self.config.evolution_iterations + 1,
                    metadata={
                        "motif_id": foundation.motif.motif_id,
                        "support": foundation.motif.support,
                        "utility": foundation.motif.utility,
                        "source_graph_ids": foundation.motif.source_graph_ids,
                    },
                )
            comparison_paths = {
                key: value for key, value in evolution.visualization_paths.items()
                if key.startswith("comparison_")
            }
            evolution.visualization_paths = loop.graph_archive.export(
                programs=loop.registry.list_programs(),
                feedback=loop.feedback.entries(),
                weighted_categories=weighted_graph.categories_summary(registry),
                frontier=[program.name for program in loop.registry.frontier()],
            )
            evolution.visualization_paths.update(comparison_paths)
        routes = self._route_probe(suite.tasks[0], graph_memory) if suite.tasks else []
        record = self._record(
            run_id,
            suite,
            ablation,
            memory_dir,
            evolution,
            foundation_report,
            len(routes),
            loop,
        )
        self._write_json(run_dir / "run_record.json", record.to_dict())
        self._write_json(run_dir / "evolution_report.json", evolution.to_dict())
        if foundation_report is not None:
            self._write_json(run_dir / "foundation_report_copy.json", foundation_report.to_dict())
        return record

    def _import_warm_start_frontier(self, graph_memory: GraphSkillMemory) -> list[str]:
        if self.config.continue_mode or not self.config.warm_start_run_dir:
            return []
        source = Path(self.config.warm_start_run_dir).expanduser()
        if not (source / "evolution" / "registry").is_dir() and source.is_dir():
            candidates = sorted(source.glob("*_full"))
            if len(candidates) == 1:
                source = candidates[0]
        registry_dir = source / "evolution" / "registry"
        graph_dir = source / "memory" / "graph_skills"
        if not registry_dir.is_dir() or not graph_dir.is_dir():
            message = (
                "warm-start run is incomplete; expected evolution/registry and "
                f"memory/graph_skills under {source}"
            )
            if self.config.warm_start_required:
                raise FileNotFoundError(message)
            print(f"[harness] warning: {message}", flush=True)
            return []
        source_registry = ProgramRegistry(registry_dir)
        frontier = source_registry.best()
        if frontier is None:
            message = f"warm-start run has no frontier: {source}"
            if self.config.warm_start_required:
                raise RuntimeError(message)
            print(f"[harness] warning: {message}", flush=True)
            return []
        imported: list[str] = []
        for graph_id in frontier.graph_ids:
            if graph_id == "baseline_t2v_graph":
                continue
            path = graph_dir / f"{graph_id}.json"
            if not path.is_file():
                if self.config.warm_start_required:
                    raise FileNotFoundError(f"warm-start graph is missing: {path}")
                continue
            with path.open("r", encoding="utf-8") as handle:
                from evovideo_skill.graph_skill import ToolPathGraph

                graph_memory.upsert_graph(ToolPathGraph.from_dict(json.load(handle)))
            imported.append(graph_id)
        if self.config.warm_start_required and not imported:
            raise RuntimeError(f"warm-start frontier contains no transferable graph: {source}")
        print(
            f"  warm-start frontier: source={source.resolve()} "
            f"program={frontier.name} graphs={len(imported)}",
            flush=True,
        )
        return imported

    def _runtime_for_ablation(self, ablation: str) -> RuntimeSettings:
        settings = RuntimeSettings(**asdict(self.config.runtime))
        if ablation == "low_threshold":
            settings.evolve_on_lte_095 = False
            settings.strict_eval = False
        if ablation == "no_vlm":
            settings.enable_vlm_eval = False
        return settings

    @staticmethod
    def _route_probe(task: VideoTask, graph_memory: GraphSkillMemory):
        router = FoundationAwareGraphRouter(graph_memory)
        expected = task.metadata.get("expected_failure_modes", []) if task.metadata else []
        return router.route(task, failure_types=expected, top_k=5)

    @staticmethod
    def _record(
        run_id: str,
        suite: BenchmarkSuite,
        ablation: str,
        memory_dir: Path,
        evolution: EvolutionLoopResult,
        foundation_report,
        route_count: int,
        loop: GraphSelfImprovingLoop,
    ) -> HarnessRunRecord:
        baseline = loop.registry.get("base")
        baseline_metrics = baseline.metrics
        accepted_count = evolution.accepted_candidate_count
        rejected_count = evolution.rejected_candidate_count
        quality_gain = evolution.best_metrics.quality - (baseline_metrics.quality if baseline_metrics else 0.0)
        full_baseline = evolution.full_baseline_evaluation
        full_final = evolution.full_final_evaluation
        full_gain = (
            full_final.metrics.quality - full_baseline.metrics.quality
            if full_baseline is not None and full_final is not None
            else None
        )
        best_program = loop.registry.get(evolution.best_program)
        online_foundations = [
            graph for graph in loop.graph_memory.list_graphs()
            if graph.stats.get("online_foundation") and graph.stats.get("accepted")
        ]
        return HarnessRunRecord(
            run_id=run_id,
            suite_name=suite.name,
            ablation=ablation,
            memory_dir=str(memory_dir),
            task_count=len(suite.tasks),
            baseline_mean_score=baseline_metrics.quality if baseline_metrics else 0.0,
            selected_graph_count=max(0, len(best_program.graph_ids) - 1),
            accepted_candidate_count=accepted_count,
            rejected_candidate_count=rejected_count,
            mutation_searches_used=evolution.mutation_searches_used,
            max_mutation_searches=evolution.max_mutation_searches,
            mutation_search_budget_exhausted=evolution.mutation_search_budget_exhausted,
            foundation_skill_count=len(foundation_report.foundation_skills) if foundation_report is not None else 0,
            online_foundation_skill_count=len(online_foundations),
            weighted_task_class_count=len(loop.weighted_tool_graph.categories) if loop.weighted_tool_graph is not None else 0,
            mined_motif_count=len(foundation_report.motifs) if foundation_report is not None else 0,
            mean_quality_gain=quality_gain,
            mean_candidate_score=evolution.best_metrics.quality,
            full_suite_baseline_score=full_baseline.metrics.quality if full_baseline is not None else None,
            full_suite_final_score=full_final.metrics.quality if full_final is not None else None,
            full_suite_quality_gain=full_gain,
            full_suite_task_count=full_final.metrics.task_count if full_final is not None else 0,
            mean_cost=evolution.best_metrics.estimated_cost,
            route_count=route_count,
            report_path=str(memory_dir / "graph_skills" / "foundation" / "foundation_report.json"),
            visualization_path=evolution.visualization_paths.get("html", ""),
        )

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    @staticmethod
    def _write_csv(path: Path, records: list[HarnessRunRecord]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(HarnessRunRecord.__dataclass_fields__))
            writer.writeheader()
            for record in records:
                writer.writerow(record.to_dict())

    @staticmethod
    @contextmanager
    def _exclusive_lock(path: Path) -> Iterator[None]:
        if fcntl is None:
            raise RuntimeError("harness locking requires fcntl on this platform")
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+", encoding="utf-8")
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.seek(0)
                owner = handle.read().strip() or "unknown owner"
                raise RuntimeError(
                    f"another harness is already using {path.parent}; lock={path}; owner={owner}"
                ) from exc
            handle.seek(0)
            handle.truncate()
            handle.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "host": socket.gethostname(),
                        "acquired_at": utc_now(),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            handle.flush()
            yield
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def summarize_harness(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    report_path = output_dir / "harness_report.json"
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    records = report.get("records", [])
    if not records:
        return {"output_dir": str(output_dir), "records": 0}
    best = max(records, key=lambda item: (item["mean_quality_gain"], item["foundation_skill_count"]))
    summary = {
        "output_dir": str(output_dir),
        "records": len(records),
        "best_run_id": best["run_id"],
        "best_mean_quality_gain": best["mean_quality_gain"],
        "total_foundation_skills": sum(item["foundation_skill_count"] for item in records),
        "total_online_foundation_skills": sum(item.get("online_foundation_skill_count", 0) for item in records),
        "total_weighted_task_classes": sum(item.get("weighted_task_class_count", 0) for item in records),
        "total_accepted_candidates": sum(item["accepted_candidate_count"] for item in records),
        "total_rejected_candidates": sum(item["rejected_candidate_count"] for item in records),
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary
