#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> Any:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _storybench_npz(source: Path, target: Path, *, width: int = 160, height: int = 96) -> None:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised in the official metric environment.
        raise SystemExit("StoryBench export requires numpy") from exc
    if not shutil.which("ffmpeg"):
        raise SystemExit("StoryBench export requires ffmpeg on PATH")
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(source),
        "-vf",
        f"fps=8,scale={width}:{height}:flags=lanczos",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg conversion failed for {source}: {result.stderr.decode('utf-8', errors='replace')}"
        )
    frame_bytes = width * height * 3
    if not result.stdout or len(result.stdout) % frame_bytes:
        raise RuntimeError(f"ffmpeg returned an invalid RGB byte stream for {source}")
    video = np.frombuffer(result.stdout, dtype=np.uint8).reshape((-1, height, width, 3))
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, video=video)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export frozen graph videos for official evaluators")
    parser.add_argument("--benchmark", choices=["vbench", "storybench"], required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--program", default="best", help="base, best, or an exact program name")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--copy", action="store_true", help="Copy VBench videos instead of symlinking")
    args = parser.parse_args()

    suite = _load(args.tasks)
    report = _load(args.report)
    program = report.get("frozen_program") if args.program == "best" else args.program
    evaluations = report.get("evaluations", {})
    if program not in evaluations:
        raise SystemExit(f"Program {program!r} is not present in {args.report}")
    task_by_id = {item["task_id"]: item for item in suite.get("tasks", [])}
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "export_manifest.jsonl"
    exported = 0
    skipped = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for rollout in evaluations[program].get("rollouts", []):
            task = task_by_id.get(rollout.get("task_id"))
            artifact = rollout.get("artifact_path")
            if not task or not artifact:
                skipped += 1
                continue
            source = Path(str(artifact)).expanduser().resolve()
            if not source.is_file():
                skipped += 1
                continue
            metadata = task.get("metadata", {})
            filename = str(metadata.get("official_filename") or f"{task['task_id']}.mp4")
            target = output / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if args.benchmark == "storybench":
                _storybench_npz(source, target)
            elif args.copy:
                shutil.copy2(source, target)
            else:
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(source)
            record = {
                "benchmark": args.benchmark,
                "program": program,
                "task_id": task["task_id"],
                "prompt": task["prompt"],
                "sample_index": metadata.get("official_sample_index"),
                "source": str(source),
                "target": str(target),
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            exported += 1
    print(f"Export complete: benchmark={args.benchmark} program={program}")
    print(f"  exported={exported} skipped={skipped}")
    print(f"  output_dir={output}")
    print(f"  manifest={manifest_path}")


if __name__ == "__main__":
    main()
