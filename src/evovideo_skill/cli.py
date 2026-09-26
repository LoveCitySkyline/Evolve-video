from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from evovideo_skill.benchmarks import load_suites
from evovideo_skill.evolution_data import EvolutionDataset, stratified_task_split, task_category
from evovideo_skill.evolution_loop import EvolutionLoopConfig, GraphSelfImprovingLoop
from evovideo_skill.foundation_skills import FoundationSkillConsolidator
from evovideo_skill.graph_algorithms import k_shortest_simple_paths
from evovideo_skill.graph_composition import HistoricalPathComposer
from evovideo_skill.graph_evolver import GraphToolPathEvolver
from evovideo_skill.graph_mining import GraphMotifMiner
from evovideo_skill.graph_router import FoundationAwareGraphRouter
from evovideo_skill.graph_skill import GraphSkillMemory
from evovideo_skill.harness import GraphHarnessRunner, HarnessConfig, summarize_harness
from evovideo_skill.models import VideoTask
from evovideo_skill.open_world_tools import ExternalToolCandidate, ToolApprovalStore, load_tool_plans
from evovideo_skill.program_registry import GraphProgram, ProgramRegistry
from evovideo_skill.prepared_tools import PreparedToolStore
from evovideo_skill.runtime import build_evaluator_suite, build_graph_mutation_proposer, build_runtime, build_tool_onboarding_manager, settings_from_namespace, with_env_overrides
from evovideo_skill.skill_memory import SkillMemory
from evovideo_skill.tools import ToolRegistry
from evovideo_skill.tool_onboarding import ToolOnboardingManager
from evovideo_skill.weighted_tool_graph import WeightedToolGraphMemory


def run_graph_evolve(args: argparse.Namespace) -> None:
    task_path = Path(args.tasks)
    with task_path.open("r", encoding="utf-8") as handle:
        tasks = [VideoTask.from_dict(item) for item in json.load(handle)]
    tasks = tasks[: args.limit_tasks]
    if not tasks:
        raise SystemExit("No tasks available for graph evolution")

    settings = settings_from_namespace(args)
    if args.provider == "local-fake":
        registry = ToolRegistry.with_mock_tools()
        vlm_augmenter = None
        print("Using local fake tools for graph-structured tool-path evolution")
        if args.enable_vlm_eval:
            print("  VLM evaluation is disabled for local fake tools because no real frames are produced")
    else:
        registry, vlm_augmenter = build_runtime(settings)

    skill_memory = SkillMemory(args.memory_dir)
    graph_memory = GraphSkillMemory(args.memory_dir)
    weighted_graph = WeightedToolGraphMemory(Path(args.memory_dir) / "weighted_tool_graph.json")
    evolver = GraphToolPathEvolver(
        skill_memory=skill_memory,
        graph_memory=graph_memory,
        tools=registry,
        evaluators=build_evaluator_suite(settings),
        vlm_augmenter=vlm_augmenter,
        min_quality_gain=args.min_quality_gain,
        mutation_proposer=build_graph_mutation_proposer(settings, registry),
        template_mutations_enabled=settings.template_mutations_enabled,
        weighted_tool_graph=weighted_graph,
        historical_path_composer=HistoricalPathComposer(weighted_graph),
    )
    report = evolver.evolve(tasks, max_candidates=args.max_candidates)

    print("Graph-structured tool-path evolution complete")
    print(f"  tasks: {[task.task_id for task in tasks]}")
    print(f"  baseline_rollouts: {len(report.baseline_rollouts)}")
    for rollout in report.baseline_rollouts:
        failure_types = [item.value for item in rollout.failure.failure_types] if rollout.failure else []
        print(
            f"  - baseline task={rollout.task.task_id} score={rollout.score:.3f} "
            f"passed={rollout.evaluation.passed} failures={failure_types}"
        )
    print(f"  proposed_graph_paths: {len(report.candidates)}")
    for candidate in report.candidates:
        edit_ops = [edit.op for edit in candidate.edits]
        print(f"  - candidate={candidate.graph.skill_name} path={candidate.graph.executable_tool_names(registry.available_names())} edits={edit_ops}")
    print(f"  validation_reports: {len(report.validation_reports)}")
    for validation in report.validation_reports:
        status = "accepted" if validation.accepted else "rejected"
        print(
            f"  - {validation.skill_name}: {status}, "
            f"gain={validation.quality_gain:.3f}, "
            f"candidate={validation.candidate_score:.3f}, "
            f"baseline={validation.baseline_score:.3f}, "
            f"stability={validation.stability:.3f}, "
            f"cost={validation.estimated_cost:.3f}, "
            f"path={validation.tool_path}"
        )
    print(f"  selected_graph_skills: {[graph.skill_name for graph in report.selected_graphs]}")
    print(f"  graph_memory: {Path(args.memory_dir) / 'graph_skills'}")
    if report.selected_graphs:
        print(f"  skill_memory: {args.memory_dir}")
    if getattr(args, "auto_consolidate_foundation", False):
        foundation_report = _consolidate_foundation_from_args(args)
        _print_foundation_report(foundation_report)


def run_consolidate_foundation_skills(args: argparse.Namespace) -> None:
    report = _consolidate_foundation_from_args(args)
    _print_foundation_report(report)


def run_tool_preflight(args: argparse.Namespace) -> None:
    config = HarnessConfig.from_file(args.config)
    settings = with_env_overrides(config.runtime)
    registry, _ = build_runtime(settings)
    manager = build_tool_onboarding_manager(settings, registry)
    print("Executable tool capabilities")
    for manifest in registry.manifests():
        print(
            f"  - {manifest['name']}: capability={manifest['capability']} "
            f"inputs={manifest['input_types']} output={manifest['output_type']} "
            f"backend={manifest['backend']} verified={manifest['verified']}"
        )
    discoverable = manager.discoverable_tools()
    print(f"Discoverable MCP/catalog tools: {len(discoverable)}")
    for manifest in discoverable:
        print(
            f"  - {manifest['name']}: capability={manifest['capability']} "
            f"backend={manifest['backend']} configured={manifest['configured']}"
        )


def run_prepare_tools(args: argparse.Namespace) -> None:
    config = HarnessConfig.from_file(args.config)
    settings = with_env_overrides(config.runtime)
    store = PreparedToolStore(args.store_dir)
    capabilities = store.load_capabilities(args.capabilities)
    settings.enable_open_world_tools = True
    settings.restore_catalog_tools = False
    settings.agent_state_dir = str(store.acquisition_state_dir)
    registry, _ = build_runtime(settings)
    manager = build_tool_onboarding_manager(settings, registry)
    results = manager.onboard_requests([item.request for item in capabilities])
    report = store.publish(
        manager.manifests,
        capabilities,
        results,
        registry.available_names(),
    )

    print("Prepared repository tool phase complete")
    print(f"  store: {store.root}")
    print(f"  catalog: {store.catalog_path}")
    print(f"  prepared_tools: {report['prepared_tools']}")
    for row in report["capabilities"]:
        request = row["request"]
        print(
            f"  - {request['capability']}: status={row['status']} "
            f"required={row['required']} tool={row['tool_name'] or '<none>'}"
        )
    if report["required_failures"] and not args.allow_partial:
        raise SystemExit(
            "Required prepared capabilities are unavailable: "
            + ", ".join(report["required_failures"])
            + f". See {store.report_path}"
        )


def _consolidate_foundation_from_args(args: argparse.Namespace):
    miner = GraphMotifMiner(
        min_support=args.min_support,
        min_utility=args.min_utility,
        max_motif_size=args.max_motif_size,
        cost_weight=args.motif_cost_weight,
        mdl_weight=args.mdl_weight,
    )
    consolidator = FoundationSkillConsolidator(
        graph_memory=GraphSkillMemory(args.memory_dir),
        skill_memory=SkillMemory(args.memory_dir),
        miner=miner,
        max_foundation_skills=args.max_foundation_skills,
    )
    return consolidator.consolidate()


def _print_foundation_report(report) -> None:
    print("Foundation skill consolidation complete")
    print(f"  graph_count: {report.global_graph.graph_count}")
    print(f"  accepted_experience_count: {report.global_graph.accepted_experience_count}")
    print(f"  mined_motifs: {len(report.motifs)}")
    for motif in report.motifs[:8]:
        print(
            f"  - motif={motif.motif_id} tools={motif.tools} "
            f"support={motif.support} utility={motif.utility:.3f} "
            f"gain={motif.avg_gain:.3f} stability={motif.stability:.3f} mdl={motif.mdl_gain:.3f}"
        )
    print(f"  foundation_skills: {[skill.skill_name for skill in report.foundation_skills]}")
    if report.centrality:
        top_nodes = sorted(report.centrality.items(), key=lambda item: item[1]["weighted_degree"], reverse=True)[:5]
        print(f"  top_tool_nodes: {[node for node, _ in top_nodes]}")
    if report.communities:
        grouped: dict[int, list[str]] = {}
        for node, community in report.communities.items():
            grouped.setdefault(community, []).append(node)
        print(f"  communities: {grouped}")
    if report.graph_similarities:
        top_pair = report.graph_similarities[0]
        print(
            "  top_wl_similarity: "
            f"{top_pair['left']}~{top_pair['right']} "
            f"sim={float(top_pair['wl_similarity']):.3f} "
            f"ged={float(top_pair['approx_graph_edit_distance']):.3f}"
        )
    print(f"  report_path: {report.report_path}")


def run_route_graph(args: argparse.Namespace) -> None:
    if args.source_tool and args.target_tool:
        graph_memory = GraphSkillMemory(args.memory_dir)
        global_graph = GraphMotifMiner().build_global_tool_graph(graph_memory.list_graphs(), graph_memory.list_experiences())
        paths = k_shortest_simple_paths(global_graph, args.source_tool, args.target_tool, k=args.top_k)
        print("K-shortest tool-path routing complete")
        print(f"  source_tool: {args.source_tool}")
        print(f"  target_tool: {args.target_tool}")
        print(f"  paths: {len(paths)}")
        for path in paths:
            print(f"  - cost={path.cost:.3f} path={path.path}")
        return
    if args.task_prompt:
        task = VideoTask(args.task_id, args.task_prompt)
    elif args.tasks:
        with Path(args.tasks).open("r", encoding="utf-8") as handle:
            task = VideoTask.from_dict(json.load(handle)[args.task_index])
    else:
        raise SystemExit("--task-prompt or --tasks is required")
    failure_types = [item.strip() for item in (args.failure_types or "").split(",") if item.strip()]
    router = FoundationAwareGraphRouter(GraphSkillMemory(args.memory_dir), cost_weight=args.route_cost_weight)
    routes = router.route(task, failure_types=failure_types, top_k=args.top_k)
    print("Foundation-aware graph routing complete")
    print(f"  task_id: {task.task_id}")
    print(f"  prompt: {task.prompt}")
    print(f"  requested_failures: {failure_types}")
    print(f"  routes: {len(routes)}")
    for route in routes:
        kind = "foundation" if route.metadata.get("foundation_skill") else "task_graph"
        print(
            f"  - {route.graph.skill_name} [{kind}] score={route.score:.3f} "
            f"cost={route.estimated_cost:.3f} tools={route.expanded_tools}"
        )
        print(f"    reason: {route.reason}")


def run_harness(args: argparse.Namespace) -> None:
    config = HarnessConfig.from_file(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.continue_mode:
        config.continue_mode = True
    report = GraphHarnessRunner(config).run()
    print("Graph harness run complete")
    print(f"  output_dir: {report.output_dir}")
    print(f"  records: {len(report.records)}")
    for record in report.records:
        full_suite = (
            f", full_suite_gain={record.full_suite_quality_gain:.3f} "
            f"({record.full_suite_task_count} tasks)"
            if record.full_suite_quality_gain is not None
            else ""
        )
        print(
            f"  - {record.run_id}: gain={record.mean_quality_gain:.3f}, "
            f"accepted={record.accepted_candidate_count}, "
            f"mutations={record.mutation_searches_used}/{record.max_mutation_searches}, "
            f"foundation={record.foundation_skill_count}, motifs={record.mined_motif_count}"
            f"{full_suite}"
        )
        if record.visualization_path:
            print(f"    graph_view={record.visualization_path}")


def _candidate_from_plan(plan: dict) -> ExternalToolCandidate:
    raw = plan["candidate"]
    return ExternalToolCandidate(**{
        key: value for key, value in raw.items()
        if key in ExternalToolCandidate.__dataclass_fields__
    })


def manage_tool_approvals(args: argparse.Namespace) -> None:
    config = HarnessConfig.from_file(args.config)
    settings = with_env_overrides(config.runtime)
    candidates_dir = Path(args.candidates_dir or (
        Path(settings.video_output_dir) / "tool_onboarding" / "open_world" / "candidates"
    )).expanduser()
    approval_file = Path(
        args.approval_file
        or os.environ.get("OPEN_WORLD_VENV_APPROVAL_FILE", "configs/open_world_venv_approvals.json")
    ).expanduser()
    store = ToolApprovalStore(approval_file)
    plans = load_tool_plans(candidates_dir)

    for candidate_id in args.approve or []:
        store.set_decision(candidate_id, True)
        print(f"Approved: {candidate_id}")
    for candidate_id in args.reject or []:
        store.set_decision(candidate_id, False)
        print(f"Rejected: {candidate_id}")

    if not plans:
        print(f"No open-world tool plans found under: {candidates_dir}")
        print(f"Approval file: {approval_file}")
        return

    print("Open-world tool approval queue")
    print(f"  candidates_dir: {candidates_dir}")
    print(f"  approval_file: {approval_file}")
    for index, plan in enumerate(plans, 1):
        candidate = _candidate_from_plan(plan)
        manifest = plan.get("manifest", {})
        spec = manifest.get("spec", {}) if isinstance(manifest, dict) else {}
        print(
            f"  [{index}] {candidate.candidate_id} status={store.status(candidate)} repo={candidate.name} "
            f"capability={spec.get('capability', 'unknown')} license={candidate.license or 'unknown'}"
        )
        print(f"      revision={candidate.revision} plan={plan['plan_path']}")

    if not args.interactive:
        return
    if not sys.stdin.isatty():
        print("Interactive approval skipped because stdin is not a terminal.")
        return
    for plan in plans:
        candidate = _candidate_from_plan(plan)
        if store.status(candidate) != "pending":
            continue
        manifest = plan.get("manifest", {})
        print(f"\nReview {candidate.name} ({candidate.candidate_id})")
        print(f"  URL: {candidate.url}")
        print(f"  revision: {candidate.revision}")
        print(f"  description: {candidate.description or '<none>'}")
        print(f"  install: {plan.get('install_commands', [])}")
        print(f"  command: {manifest.get('command', []) if isinstance(manifest, dict) else []}")
        print(f"  risks: {plan.get('risks', [])}")
        while True:
            answer = input("Decision [y] approve/[n] reject/[s] skip/[q] quit: ").strip().lower()
            if answer in {"q", "quit"}:
                return
            if answer in {"s", "skip", ""}:
                break
            if answer in {"y", "yes"}:
                store.set_decision(candidate.candidate_id, True)
                print(f"Approved: {candidate.candidate_id}")
                break
            if answer in {"n", "no"}:
                store.set_decision(candidate.candidate_id, False)
                print(f"Rejected: {candidate.candidate_id}")
                break
            print("Please enter y, n, s, or q.")


def run_summarize_results(args: argparse.Namespace) -> None:
    summary = summarize_harness(args.output_dir)
    print("Harness summary complete")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def list_programs(args: argparse.Namespace) -> None:
    registry = ProgramRegistry(Path(args.run_dir) / "evolution" / "registry")
    frontier = {program.name for program in registry.frontier()}
    programs = registry.list_programs()
    if not programs:
        print("No evolved programs found.")
        return
    for program in programs:
        marker = "*" if program.name in frontier else " "
        metrics = program.metrics
        score = f"{metrics.quality:.3f}" if metrics else "n/a"
        cost = f"{metrics.estimated_cost:.3f}" if metrics else "n/a"
        print(
            f"{marker} {program.name} parent={program.parent or '-'} "
            f"quality={score} cost={cost} graphs={program.graph_ids}"
        )


def diff_programs(args: argparse.Namespace) -> None:
    registry = ProgramRegistry(Path(args.run_dir) / "evolution" / "registry")
    right = args.right
    if right == "best":
        best = registry.best()
        if best is None:
            raise SystemExit("No frontier program available")
        right = best.name
    payload = registry.diff(args.left, right)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def show_evolution_logs(args: argparse.Namespace) -> None:
    path = Path(args.run_dir) / "evolution" / "iterations.jsonl"
    if not path.exists():
        print("No evolution log found.")
        return
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    for record in records[-args.last :]:
        print(
            f"iter={record['iteration']} parent={record['parent_program']} "
            f"status={record['status']} failures={record['failure_types']} "
            f"accepted={record['accepted_programs']} "
            f"exploratory={record.get('exploratory_programs', [])} frontier={record['frontier']}"
        )


def evaluate_best_program(args: argparse.Namespace) -> None:
    config = HarnessConfig.from_file(args.config)
    config.runtime = with_env_overrides(config.runtime)
    suites = load_suites(config.task_files, limit_per_suite=config.limit_per_suite)
    suite = next((item for item in suites if item.name == args.suite), suites[0] if suites else None)
    if suite is None:
        raise SystemExit("No benchmark suite available")
    run_dir = Path(args.run_dir)
    memory_dir = run_dir / "memory"
    registry_tools, vlm_augmenter = build_runtime(config.runtime)
    graph_memory = GraphSkillMemory(memory_dir)
    evolver = GraphToolPathEvolver(
        skill_memory=SkillMemory(memory_dir),
        graph_memory=graph_memory,
        tools=registry_tools,
        evaluators=build_evaluator_suite(config.runtime),
        vlm_augmenter=vlm_augmenter,
        min_quality_gain=config.min_quality_gain,
    )
    dataset = stratified_task_split(
        suite.tasks,
        train_ratio=config.train_ratio,
        validation_ratio=config.validation_ratio,
        seed=config.split_seed,
    )
    loop = GraphSelfImprovingLoop(
        EvolutionLoopConfig(
            max_iterations=config.evolution_iterations,
            frontier_size=config.frontier_size,
            no_improvement_limit=config.no_improvement_limit,
            selection_strategy=config.selection_strategy,
            min_quality_gain=config.min_quality_gain,
            cache_enabled=config.cache_enabled,
            continue_mode=True,
            cost_weight=config.program_cost_weight,
        ),
        dataset,
        evolver,
        graph_memory,
        run_dir / "evolution",
        runtime_signature=asdict(config.runtime),
    )
    best = loop.registry.best()
    if best is None:
        raise SystemExit("No best program available")
    tasks = dataset.test if args.split == "test" else dataset.validation
    if not tasks:
        raise SystemExit(f"No tasks available in split: {args.split}")
    evaluation = loop.evaluate_program(best, tasks)
    output_path = run_dir / f"best_eval_{args.split}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(evaluation), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(
        f"Best program evaluation complete: program={best.name}, "
        f"split={args.split}, quality={evaluation.metrics.quality:.3f}, "
        f"pass_rate={evaluation.metrics.pass_rate:.3f}, "
        f"cache_hits={evaluation.metrics.cache_hits}/"
        f"{evaluation.metrics.sample_count or evaluation.metrics.task_count}"
    )
    print(f"  report: {output_path}")


def evaluate_external_benchmark(args: argparse.Namespace) -> None:
    """Evaluate copied frozen programs without mutating the source evolution run."""
    config = HarnessConfig.from_file(args.config)
    config.runtime = with_env_overrides(config.runtime)
    suites = load_suites(config.task_files, limit_per_suite=config.limit_per_suite)
    suite = next((item for item in suites if item.name == args.suite), suites[0] if suites else None)
    if suite is None or not suite.tasks:
        raise SystemExit("No external benchmark tasks are available")

    source_run = Path(args.source_run_dir).expanduser()
    if not (source_run / "evolution" / "registry").is_dir() and source_run.is_dir():
        candidates = sorted(source_run.glob("*_full"))
        if len(candidates) == 1:
            source_run = candidates[0]
    source_memory = source_run / "memory"
    source_registry = source_run / "evolution" / "registry"
    if not source_memory.is_dir() or not source_registry.is_dir():
        raise SystemExit(
            "Frozen source run is incomplete; expected memory/ and evolution/registry/ under "
            f"{source_run}"
        )

    output_dir = Path(args.output_dir).expanduser()
    marker = output_dir / "external_eval.json"
    if output_dir.is_dir() and any(output_dir.iterdir()) and not args.continue_mode:
        raise SystemExit(
            f"External evaluation workspace is not empty: {output_dir}. "
            "Use --continue to reuse its paired cache or select a new output directory."
        )
    output_memory = output_dir / "memory"
    output_registry = output_dir / "evolution" / "registry"
    if not args.continue_mode or not output_memory.is_dir():
        shutil.copytree(source_memory, output_memory, dirs_exist_ok=True)
    if not args.continue_mode or not output_registry.is_dir():
        shutil.copytree(source_registry, output_registry, dirs_exist_ok=True)

    config.runtime.agent_state_dir = str(output_dir / "agent_state")
    registry_tools, vlm_augmenter = build_runtime(config.runtime)
    graph_memory = GraphSkillMemory(output_memory)
    evolver = GraphToolPathEvolver(
        skill_memory=SkillMemory(output_memory),
        graph_memory=graph_memory,
        tools=registry_tools,
        evaluators=build_evaluator_suite(config.runtime),
        vlm_augmenter=vlm_augmenter,
        min_quality_gain=config.min_quality_gain,
    )
    categories: dict[str, list[str]] = {}
    for task in suite.tasks:
        categories.setdefault(task_category(task), []).append(task.task_id)
    dataset = EvolutionDataset([], [], list(suite.tasks), categories, config.split_seed)
    loop = GraphSelfImprovingLoop(
        EvolutionLoopConfig(
            max_iterations=0,
            frontier_size=config.frontier_size,
            no_improvement_limit=0,
            selection_strategy=config.selection_strategy,
            min_quality_gain=config.min_quality_gain,
            cache_enabled=config.cache_enabled,
            evaluation_seeds=config.evaluation_seeds,
            continue_mode=True,
            cost_weight=config.program_cost_weight,
            runtime_failure_circuit_breaker=config.runtime_failure_circuit_breaker,
        ),
        dataset,
        evolver,
        graph_memory,
        output_dir / "evolution",
        runtime_signature=asdict(config.runtime),
    )
    program_registry = loop.registry
    best = program_registry.best()
    if best is None:
        raise SystemExit("The copied source run has no best frontier program")
    if args.require_specialized_routing and best.name == "base":
        raise SystemExit(
            "Frozen transfer requires an evolved frontier program, but the source registry's "
            "best program is base. Point --source-run-dir at a run with accepted graph paths."
        )

    requested = [value.strip() for value in args.programs.split(",") if value.strip()]
    programs = []
    for name in requested:
        if name == "best":
            program = best
        elif name == "base":
            program = GraphProgram(
                name="base",
                graph_ids=["baseline_t2v_graph"],
                proposal="single-call text-to-video external baseline",
                status="frozen_baseline",
            )
        else:
            program = program_registry.get(name)
        if all(existing.name != program.name for existing in programs):
            programs.append(program)

    planned_routing = {
        program.name: _external_planned_routing_summary(loop, program, suite.tasks)
        for program in programs
    }
    base_plan = planned_routing.get("base")
    best_plan = planned_routing.get(best.name)
    planned_changed = None
    if base_plan is not None and best_plan is not None and best.name != "base":
        planned_changed = sum(
            base_graph != best_plan["task_routes"].get(task_id)
            for task_id, base_graph in base_plan["task_routes"].items()
        )
    tool_availability = _external_graph_tool_availability(loop, best)
    preflight = {
        "benchmark": suite.name,
        "source_run_dir": str(source_run.resolve()),
        "frozen_program": best.name,
        "planned_routing": planned_routing,
        "planned_changed_tasks": planned_changed,
        "graph_tool_availability": tool_availability,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = output_dir / "routing_preflight.json"
    with preflight_path.open("w", encoding="utf-8") as handle:
        json.dump(preflight, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    for name, summary in planned_routing.items():
        print(
            f"[routing-preflight] program={name} "
            f"specialized={summary['specialized_tasks']}/{summary['task_count']} "
            f"graph_counts={summary['graph_counts']}",
            flush=True,
        )
    print(f"[routing-preflight] report={preflight_path.resolve()}", flush=True)
    if args.require_specialized_routing and planned_changed == 0:
        missing = {
            graph_id: item["missing_tools"]
            for graph_id, item in tool_availability.items()
            if item["missing_tools"]
        }
        raise SystemExit(
            "Frozen best would not route any task differently from the true single-Wan base; "
            "stopped before video generation. "
            f"Unavailable graph tools: {missing or '<none>'}. "
            f"Inspect {preflight_path}."
        )

    evaluations = {}
    for program in programs:
        print(
            f"[external-eval] suite={suite.name} program={program.name} "
            f"tasks={len(suite.tasks)} source={source_run}",
            flush=True,
        )
        evaluations[program.name] = loop.evaluate_program(
            program,
            suite.tasks,
            verifier_gated=program.name != "base",
        )

    routing = {
        name: _external_routing_summary(evaluation)
        for name, evaluation in evaluations.items()
    }
    paired_gain = None
    paired_routing = None
    baseline = evaluations.get("base")
    candidate = evaluations.get(best.name)
    if baseline is not None and candidate is not None and best.name != "base":
        base_rollouts = {
            (item.task_id, item.evaluation_seed): item for item in baseline.rollouts
        }
        gains = [
            item.score - base_rollouts[(item.task_id, item.evaluation_seed)].score
            for item in candidate.rollouts
            if (item.task_id, item.evaluation_seed) in base_rollouts
        ]
        paired_gain = sum(gains) / len(gains) if gains else None
        paired_routing = _external_paired_routing_summary(baseline, candidate)

    report = {
        "benchmark": suite.name,
        "source_run_dir": str(source_run.resolve()),
        "frozen_program": best.name,
        "task_count": len(suite.tasks),
        "programs": requested,
        "runtime_policy": {
            "verifier_gated_repair": True,
            "retain_passing_baseline": True,
            "fallback_on_non_improvement": True,
            "fallback_on_new_failure": True,
            "baseline_conditioned_repair": os.environ.get(
                "EVOVIDEO_BASELINE_CONDITIONED_REPAIR", "1"
            ) != "0",
            "minimum_internal_gain": float(
                os.environ.get("EVOVIDEO_RUNTIME_REPAIR_MIN_GAIN", "0.02")
            ),
        },
        "mean_paired_gain": paired_gain,
        "routing": routing,
        "paired_routing": paired_routing,
        "runtime_gate": _external_runtime_gate_summary(loop.runtime_decisions),
        "runtime_decisions": loop.runtime_decisions,
        "evaluations": {
            name: asdict(evaluation) for name, evaluation in evaluations.items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with marker.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    for name, evaluation in evaluations.items():
        route = routing[name]
        print(
            f"  {name}: quality={evaluation.metrics.quality:.4f} "
            f"pass={evaluation.metrics.pass_rate:.4f} "
            f"coverage={evaluation.metrics.execution_coverage:.4f} "
            f"samples={evaluation.metrics.sample_count} "
            f"specialized_routes={route['specialized_rollouts']}/"
            f"{route['rollout_count']}",
            flush=True,
        )
        print(f"    graph_counts={route['graph_counts']}", flush=True)
    if paired_gain is not None:
        print(f"  mean_paired_gain={paired_gain:+.4f}", flush=True)
    if paired_routing is not None:
        print(
            "  paired_routing: "
            f"changed_graph={paired_routing['changed_graph_pairs']}/"
            f"{paired_routing['pair_count']} "
            f"changed_artifact={paired_routing['changed_artifact_pairs']}/"
            f"{paired_routing['pair_count']}",
            flush=True,
        )
    runtime_gate = report["runtime_gate"]
    if runtime_gate["decision_count"]:
        print(
            "  runtime_gate: "
            f"attempted={runtime_gate['candidate_attempts']} "
            f"accepted={runtime_gate['accepted_candidates']} "
            f"fallbacks={runtime_gate['fallbacks']} "
            f"baseline_pass_retained={runtime_gate['baseline_pass_retained']}",
            flush=True,
        )
        print(f"    reasons={runtime_gate['reason_counts']}", flush=True)
    print(f"  report={marker.resolve()}", flush=True)


def _external_routing_summary(evaluation: ProgramEvaluation) -> dict[str, object]:
    graph_counts = Counter(item.graph_id for item in evaluation.rollouts)
    specialized = sum(
        count for graph_id, count in graph_counts.items()
        if graph_id != "baseline_t2v_graph"
    )
    return {
        "rollout_count": len(evaluation.rollouts),
        "specialized_rollouts": specialized,
        "baseline_rollouts": graph_counts.get("baseline_t2v_graph", 0),
        "cache_hits": sum(bool(item.cache_hit) for item in evaluation.rollouts),
        "execution_errors": sum(item.execution_error is not None for item in evaluation.rollouts),
        "graph_counts": dict(sorted(graph_counts.items())),
    }


def _external_runtime_gate_summary(decisions: list[dict[str, object]]) -> dict[str, object]:
    reasons = Counter(str(item.get("reason") or "unknown") for item in decisions)
    attempts = sum(item.get("candidate_graph_id") is not None for item in decisions)
    accepted = sum(bool(item.get("accepted_candidate")) for item in decisions)
    return {
        "decision_count": len(decisions),
        "candidate_attempts": attempts,
        "accepted_candidates": accepted,
        "fallbacks": attempts - accepted,
        "baseline_pass_retained": reasons.get("baseline_passed_verifier_gate", 0),
        "baseline_conditioned_attempts": sum(
            bool(item.get("reused_baseline_artifact")) for item in decisions
        ),
        "reason_counts": dict(sorted(reasons.items())),
        "candidate_graph_counts": dict(sorted(Counter(
            str(item["candidate_graph_id"])
            for item in decisions
            if item.get("candidate_graph_id") is not None
        ).items())),
    }


def _external_planned_routing_summary(
    loop: GraphSelfImprovingLoop,
    program: GraphProgram,
    tasks: list[VideoTask],
) -> dict[str, object]:
    task_routes = {
        task.task_id: loop._select_graph(program, task).skill_name
        for task in tasks
    }
    graph_counts = Counter(task_routes.values())
    return {
        "task_count": len(tasks),
        "specialized_tasks": sum(
            count for graph_id, count in graph_counts.items()
            if graph_id != "baseline_t2v_graph"
        ),
        "graph_counts": dict(sorted(graph_counts.items())),
        "task_routes": task_routes,
    }


def _external_graph_tool_availability(
    loop: GraphSelfImprovingLoop,
    program: GraphProgram,
) -> dict[str, dict[str, object]]:
    availability: dict[str, dict[str, object]] = {}
    for graph_id in program.graph_ids:
        graph = loop._graphs.get(graph_id)
        if graph is None:
            availability[graph_id] = {
                "graph_found": False,
                "tools": [],
                "missing_tools": [],
            }
            continue
        tools = graph.tool_names()
        availability[graph_id] = {
            "graph_found": True,
            "tools": tools,
            "missing_tools": [name for name in tools if not loop.evolver.tools.has(name)],
            "requires_task_reference_video": "task_reference_video" in tools,
            "validated_task_classes": graph.stats.get("validated_task_classes", []),
            "validated_routing_profiles": graph.stats.get("validated_routing_profiles", []),
            "source_failure_types": graph.stats.get("source_failure_types", []),
        }
    return availability


def _external_paired_routing_summary(
    baseline: ProgramEvaluation,
    candidate: ProgramEvaluation,
) -> dict[str, int]:
    base_rollouts = {
        (item.task_id, item.evaluation_seed): item for item in baseline.rollouts
    }
    pairs = [
        (base_rollouts[(item.task_id, item.evaluation_seed)], item)
        for item in candidate.rollouts
        if (item.task_id, item.evaluation_seed) in base_rollouts
    ]
    return {
        "pair_count": len(pairs),
        "changed_graph_pairs": sum(left.graph_id != right.graph_id for left, right in pairs),
        "changed_artifact_pairs": sum(
            bool(left.artifact_path and right.artifact_path)
            and left.artifact_path != right.artifact_path
            for left, right in pairs
        ),
        "identical_score_pairs": sum(abs(left.score - right.score) <= 1e-12 for left, right in pairs),
    }


def list_skills(args: argparse.Namespace) -> None:
    memory = SkillMemory(args.memory_dir)
    skills = memory.list_skills()
    if not skills:
        print("No skills found.")
        return
    for skill in skills:
        print(f"{skill.skill_name}@{skill.version}")
        print(f"  description: {skill.description}")
        print(f"  triggers: {', '.join(skill.triggers)}")
        print(f"  tools: {', '.join(skill.tools)}")
        print(f"  stats: uses={skill.stats.uses}, successes={skill.stats.successes}, failures={skill.stats.failures}")
        if skill.validation:
            print(
                "  validation: "
                f"gain={skill.validation.quality_gain:.3f}, "
                f"candidate={skill.validation.candidate_score:.3f}, "
                f"baseline={skill.validation.baseline_score:.3f}, "
                f"cost={skill.validation.estimated_cost:.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="VideoSkillGraph paper CLI")
    sub = parser.add_subparsers(required=True)

    graph = sub.add_parser("graph-evolve", help="Self-evolve graph-structured executable video tool paths")
    graph.add_argument("--provider", choices=["local-fake", "local-wan", "auto", "hailuo", "aliyun-wanx", "kling", "minimax-h3", "local-h3"], default="local-fake")
    graph.add_argument("--memory-dir", default="outputs/video_skill_graph_memory")
    graph.add_argument("--tasks", default="examples/graph_evolution_tasks.json")
    graph.add_argument("--limit-tasks", type=int, default=4, help="Limit task count for provider/API cost control")
    graph.add_argument("--max-candidates", type=int, default=8)
    graph.add_argument("--min-quality-gain", type=float, default=0.02)
    graph.add_argument("--duration", type=int, default=6)
    graph.add_argument("--resolution", default=None)
    graph.add_argument("--timeout-seconds", type=int, default=900)
    graph.add_argument("--poll-interval-seconds", type=float, default=5)
    graph.add_argument("--reference-image-url", default=None, help="Optional public image URL for I2V-capable providers")
    graph.add_argument("--video-output-dir", default="outputs/provider_videos")
    graph.add_argument("--sample-frames", type=int, default=6)
    graph.add_argument("--h3-fl2va-url", default="http://127.0.0.1:30010")
    graph.add_argument("--h3-ref2va-url", default="http://127.0.0.1:30011")
    graph.add_argument("--h3-local-model-revision", default="unspecified")
    graph.add_argument("--h3-local-quality", choices=["lossless", "extra-high"], default="lossless")
    graph.add_argument("--h3-max-api-calls", type=int, default=40, help="H3 generation request budget, including local H3 requests")
    graph.add_argument("--skip-video-processing", action="store_true", help="Skip download/frame sampling for generated video URLs")
    graph.add_argument("--enable-vlm-eval", action="store_true", help="Evaluate sampled frames with Qwen/Qwen3-VL through DashScope")
    graph.add_argument("--vlm-model", default=None, help="Default: qwen3-vl-plus, or set EVOVIDEO_VLM_MODEL")
    graph.add_argument("--vlm-base-url", default=None, help="Default: DashScope OpenAI-compatible base URL")
    graph.add_argument("--vlm-timeout-seconds", type=int, default=120)
    graph.add_argument("--vlm-max-images", type=int, default=6)
    graph.add_argument("--strict-eval", action="store_true", default=True)
    graph.add_argument("--evolve-on-lte-095", action="store_true", default=True, help="Trigger graph evolution when identity/clothing/action/vbench scores are <= 0.95")
    graph.add_argument("--identity-threshold", type=float, default=None)
    graph.add_argument("--clothing-threshold", type=float, default=None)
    graph.add_argument("--action-threshold", type=float, default=None)
    graph.add_argument("--background-threshold", type=float, default=None)
    graph.add_argument("--target-edit-threshold", type=float, default=None)
    graph.add_argument("--vbench-threshold", type=float, default=None)
    graph.add_argument("--t2v-model", default=None)
    graph.add_argument("--i2v-model", default=None)
    graph.add_argument("--edit-model", default=None)
    graph.add_argument("--wan-repo", default=None, help="Path to an open-source Wan repo containing generate.py")
    graph.add_argument("--wan-ckpt-dir", default=None, help="Path to local Wan checkpoint directory")
    graph.add_argument("--wan-task", default="t2v-1.3B")
    graph.add_argument("--wan-size", default="832*480")
    graph.add_argument("--wan-python", default="python")
    graph.add_argument("--wan-extra-args", default=None, help="Extra Wan generate.py args, parsed with shell-style quoting")
    graph.add_argument("--wan-save-arg", default="--save_file", help="Wan generate.py output-file argument; set to empty if unsupported")
    graph.add_argument("--wan-seed-arg", default="--base_seed")
    graph.add_argument("--wan-i2v-ckpt-dir", default=None)
    graph.add_argument("--wan-i2v-task", default="i2v-14B")
    graph.add_argument("--wan-i2v-size", default=None)
    graph.add_argument("--wan-i2v-image-arg", default="--image")
    graph.add_argument("--wan-i2v-extra-args", default=None)
    graph.add_argument("--tool-catalog-path", default=None)
    graph.add_argument("--enable-open-world-tools", action="store_true")
    graph.add_argument("--open-world-max-candidates", type=int, default=6)
    graph.add_argument("--open-world-docker-bin", default="docker")
    graph.add_argument("--enable-mcp-tools", action="store_true")
    graph.add_argument("--mcp-servers-config", default=None)
    graph.add_argument("--enable-llm-mutation", action="store_true", help="Use an OpenAI-compatible LLM for open-ended bounded graph mutations")
    graph.add_argument("--disable-template-mutations", dest="template_mutations_enabled", action="store_false", help="Disable built-in mutation templates and use only LLM proposals")
    graph.add_argument("--graph-llm-model", default="openai/gpt-5.6-sol")
    graph.add_argument("--graph-llm-base-url", default="https://openrouter.ai/api/v1")
    graph.add_argument("--graph-llm-timeout-seconds", type=int, default=120)
    graph.add_argument("--graph-llm-temperature", type=float, default=0.2)
    graph.add_argument("--graph-llm-reasoning-effort", default="high")
    graph.add_argument("--graph-llm-max-candidates", type=int, default=3)
    graph.add_argument("--graph-llm-max-edits", type=int, default=6)
    graph.set_defaults(template_mutations_enabled=True)
    graph.add_argument("--auto-consolidate-foundation", action="store_true", help="Mine shared graph motifs and promote foundation skills after graph evolution")
    graph.add_argument("--min-support", type=int, default=2)
    graph.add_argument("--min-utility", type=float, default=0.02)
    graph.add_argument("--max-motif-size", type=int, default=4)
    graph.add_argument("--motif-cost-weight", type=float, default=0.08)
    graph.add_argument("--mdl-weight", type=float, default=0.02)
    graph.add_argument("--max-foundation-skills", type=int, default=5)
    graph.set_defaults(func=run_graph_evolve)

    consolidate = sub.add_parser("consolidate-foundation-skills", help="Mine reusable graph motifs and promote foundation video skills")
    consolidate.add_argument("--memory-dir", default="outputs/video_skill_graph_memory")
    consolidate.add_argument("--min-support", type=int, default=2)
    consolidate.add_argument("--min-utility", type=float, default=0.02)
    consolidate.add_argument("--max-motif-size", type=int, default=4)
    consolidate.add_argument("--motif-cost-weight", type=float, default=0.08)
    consolidate.add_argument("--mdl-weight", type=float, default=0.02)
    consolidate.add_argument("--max-foundation-skills", type=int, default=5)
    consolidate.set_defaults(func=run_consolidate_foundation_skills)

    route = sub.add_parser("route-graph", help="Route a new prompt through learned foundation and task graph skills")
    route.add_argument("--memory-dir", default="outputs/video_skill_graph_memory")
    route.add_argument("--task-id", default="route-demo-task")
    route.add_argument("--task-prompt", default=None)
    route.add_argument("--tasks", default=None)
    route.add_argument("--task-index", type=int, default=0)
    route.add_argument("--failure-types", default="")
    route.add_argument("--top-k", type=int, default=3)
    route.add_argument("--route-cost-weight", type=float, default=0.08)
    route.add_argument("--source-tool", default=None, help="Optional source tool for k-shortest routing on the global tool graph")
    route.add_argument("--target-tool", default=None, help="Optional target tool for k-shortest routing on the global tool graph")
    route.set_defaults(func=run_route_graph)

    harness = sub.add_parser("run-harness", help="Run configured benchmark suites and ablations for the paper harness")
    harness.add_argument("--config", default="configs/video_skill_graph.toml")
    harness.add_argument("--output-dir", default=None)
    harness.add_argument("--continue", dest="continue_mode", action="store_true", help="Resume from the saved frontier and exact sampler checkpoint")
    harness.set_defaults(func=run_harness)

    approvals = sub.add_parser("tool-approvals", help="List and interactively approve or reject open-world venv tools")
    approvals.add_argument("--config", default="configs/open_world_local_wan_harness.json")
    approvals.add_argument("--candidates-dir", default=None)
    approvals.add_argument("--approval-file", default=None)
    approvals.add_argument("--interactive", action="store_true")
    approvals.add_argument("--approve", action="append", default=[])
    approvals.add_argument("--reject", action="append", default=[])
    approvals.set_defaults(func=manage_tool_approvals)

    preflight = sub.add_parser("tool-preflight", help="Inspect typed executable and discoverable tool capabilities without generating video")
    preflight.add_argument("--config", default="configs/local_wan_harness.json")
    preflight.set_defaults(func=run_tool_preflight)

    prepare = sub.add_parser(
        "prepare-tools",
        help="Install and verify repository tools into a reusable store before graph evolution",
    )
    prepare.add_argument("--config", default="configs/open_world_local_wan_mini15_harness.json")
    prepare.add_argument("--capabilities", required=True)
    prepare.add_argument("--store-dir", default="outputs/prepared_tool_store")
    prepare.add_argument(
        "--allow-partial",
        action="store_true",
        help="Publish successful tools even when a required capability could not be prepared",
    )
    prepare.set_defaults(func=run_prepare_tools)

    summarize = sub.add_parser("summarize-results", help="Summarize a completed graph harness output directory")
    summarize.add_argument("--output-dir", default="outputs/harness_local_fake")
    summarize.set_defaults(func=run_summarize_results)

    programs = sub.add_parser("programs", help="List all evolved graph programs and frontier members")
    programs.add_argument("--run-dir", required=True, help="Suite/ablation run directory containing evolution/")
    programs.set_defaults(func=list_programs)

    program_diff = sub.add_parser("program-diff", help="Compare graph composition and metrics between two programs")
    program_diff.add_argument("--run-dir", required=True)
    program_diff.add_argument("--left", default="base")
    program_diff.add_argument("--right", default="best")
    program_diff.set_defaults(func=diff_programs)

    logs = sub.add_parser("evolution-logs", help="Show recent self-evolution iterations")
    logs.add_argument("--run-dir", required=True)
    logs.add_argument("--last", type=int, default=5)
    logs.set_defaults(func=show_evolution_logs)

    evaluate = sub.add_parser("eval-best", help="Evaluate the current best graph program on validation or test data")
    evaluate.add_argument("--config", default="configs/video_skill_graph.toml")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.add_argument("--suite", default=None)
    evaluate.add_argument("--split", choices=["validation", "test"], default="test")
    evaluate.set_defaults(func=evaluate_best_program)

    external = sub.add_parser(
        "eval-external",
        help="Evaluate base and frozen best programs on an external benchmark in an isolated workspace",
    )
    external.add_argument("--config", required=True)
    external.add_argument("--source-run-dir", required=True)
    external.add_argument("--output-dir", required=True)
    external.add_argument("--suite", default=None)
    external.add_argument("--programs", default="base,best")
    external.add_argument("--continue", dest="continue_mode", action="store_true")
    external.add_argument(
        "--require-specialized-routing",
        action="store_true",
        help="Fail when frozen best never routes differently from base",
    )
    external.set_defaults(func=evaluate_external_benchmark)

    ls = sub.add_parser("list-skills", help="List learned skills")
    ls.add_argument("--memory-dir", default="outputs/skill_memory")
    ls.set_defaults(func=list_skills)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
