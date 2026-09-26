#!/usr/bin/env python3
from __future__ import annotations

import argparse

from evovideo_skill.external_benchmarks import (
    VBENCH_DIMENSIONS,
    build_storybench_suite,
    build_vbench_suite,
    write_suite,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official external benchmark tasks")
    sub = parser.add_subparsers(dest="benchmark", required=True)

    vbench = sub.add_parser("vbench")
    vbench.add_argument("--info", required=True, help="Path to VBench_full_info.json")
    vbench.add_argument("--output", required=True)
    vbench.add_argument(
        "--dimensions",
        default=",".join(VBENCH_DIMENSIONS),
        help="Comma-separated official dimensions",
    )
    vbench.add_argument("--samples-per-prompt", type=int, default=5)
    vbench.add_argument("--temporal-flickering-samples", type=int, default=25)
    vbench.add_argument("--limit-prompts-per-dimension", type=int, default=None)
    vbench.add_argument("--seed-offset", type=int, default=20260901)

    story = sub.add_parser("storybench")
    story.add_argument("--tasks", required=True, help="Official StoryBench task JSON")
    story.add_argument("--output", required=True)
    story.add_argument("--task-mode", default="story_gen", choices=["story_gen"])
    story.add_argument("--samples-per-prompt", type=int, default=4)
    story.add_argument("--limit", type=int, default=None)
    story.add_argument("--seed-offset", type=int, default=20261901)

    args = parser.parse_args()
    if args.benchmark == "vbench":
        payload = build_vbench_suite(
            args.info,
            dimensions=[value.strip() for value in args.dimensions.split(",") if value.strip()],
            samples_per_prompt=args.samples_per_prompt,
            temporal_flickering_samples=args.temporal_flickering_samples,
            limit_prompts_per_dimension=args.limit_prompts_per_dimension,
            seed_offset=args.seed_offset,
        )
    else:
        payload = build_storybench_suite(
            args.tasks,
            task_mode=args.task_mode,
            samples_per_prompt=args.samples_per_prompt,
            limit=args.limit,
            seed_offset=args.seed_offset,
        )
    output = write_suite(payload, args.output)
    metadata = payload["metadata"]
    print(f"Prepared {args.benchmark}: {output}")
    print(f"  tasks={len(payload['tasks'])} metadata={metadata}")


if __name__ == "__main__":
    main()
