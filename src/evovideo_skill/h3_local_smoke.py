"""Three real local H3 calls to verify both variants and artifact passing."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from evovideo_skill.evaluators import EvaluatorSuite
from evovideo_skill.graph_executor import ArtifactPassingGraphExecutor
from evovideo_skill.graph_skill import GraphEdge, GraphNode, ToolPathGraph
from evovideo_skill.h3_api import register_h3_tools
from evovideo_skill.harness import HarnessConfig
from evovideo_skill.models import VideoPlan, VideoTask
from evovideo_skill.runtime import build_h3_local_client, with_env_overrides
from evovideo_skill.tools import ToolRegistry


def smoke_graph() -> ToolPathGraph:
    return ToolPathGraph("h3-local-smoke", "h3-local-smoke", "Verify local variants; not a quality benchmark", ["generation"], [
        GraphNode("draft", "tool", "h3_t2va"),
        GraphNode("keyframe", "tool", "h3_frame_extract", {"position": "last", "role": "first_frame"}),
        GraphNode("conditioned", "tool", "h3_fl2va"),
        GraphNode("references", "tool", "h3_reference_pack", {"bindings": [
            {"source": "conditioned", "kind": "video", "role": "reference_video", "semantic_role": "subject and motion"}]}),
        GraphNode("reference_generation", "tool", "h3_ref2va"),
    ], [GraphEdge("draft-frame", "draft", "keyframe"), GraphEdge("frame-conditioned", "keyframe", "conditioned"),
        GraphEdge("conditioned-references", "conditioned", "references"), GraphEdge("references-generation", "references", "reference_generation")])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/h3_local_graph_harness.json")
    parser.add_argument("--output-dir", default="outputs/h3_local_smoke")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    settings = with_env_overrides(HarnessConfig.from_file(args.config).runtime)
    if settings.provider != "local-h3":
        raise ValueError("Smoke test requires provider=local-h3")
    settings.video_output_dir = str(Path(args.output_dir).expanduser().resolve())
    settings.h3_max_api_calls = 3
    client = build_h3_local_client(settings)
    health = client.check_health()
    registry = ToolRegistry()
    register_h3_tools(registry, client)
    task = VideoTask("h3-local-smoke", "A person wearing a red coat walks slowly through a quiet park. "
                     "Keep the same face and clothing. Natural footsteps and gentle ambient sound.",
                     duration_seconds=5, metadata={"generation_seed": args.seed, "replicate_label": args.seed})
    plan = VideoPlan(task.task_id, {}, ["walk slowly"], [], [], [], task.prompt)
    result = ArtifactPassingGraphExecutor(registry, EvaluatorSuite()).execute(task, plan, smoke_graph())
    report = {"status": "passed", "quality_gain_evaluated": False, "server_health": health,
              "graph": asdict(smoke_graph()), "artifacts": {key: value.metadata for key, value in result.node_artifacts.items()},
              "final_video": result.artifact.metadata["local_video_path"]}
    path = Path(settings.video_output_dir) / "smoke_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Local H3 smoke passed (3 real generation calls, no VLM/planner calls): {path}")
    print(f"Video: {report['final_video']}")


if __name__ == "__main__":
    main()
