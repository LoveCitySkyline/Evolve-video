"""Optional task-declared audio verifier, kept independent from graph planning."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from evovideo_skill.api_tools import VideoApiError
from evovideo_skill.models import VideoArtifact, VideoTask


class H3MultimodalEvaluator:
    def __init__(self, visual_evaluator: Any, command: list[str], output_dir: str, timeout_seconds: int = 120):
        if not command or not all(isinstance(token, str) and token for token in command):
            raise ValueError("H3 audio verifier command must be a nonempty JSON argv array")
        self.visual_evaluator = visual_evaluator
        self.command = command
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.timeout_seconds = timeout_seconds
        self.model = visual_evaluator.model
        self.require_complete = getattr(visual_evaluator, "require_complete", False)

    def evaluate(self, task: VideoTask, artifact: VideoArtifact) -> dict[str, Any]:
        result = self.visual_evaluator.evaluate(task, artifact)
        criteria = task.metadata.get("h3_audio_criteria", [])
        if not criteria:
            return result
        rubric = task.metadata.get("evaluation", {})
        if not isinstance(criteria, list) or any(name not in rubric for name in criteria):
            raise VideoApiError("h3_audio_criteria must name predeclared evaluation criteria")
        digest = hashlib.sha256(f"{task.task_id}:{artifact.artifact_id}".encode()).hexdigest()[:20]
        folder = self.output_dir / digest
        folder.mkdir(parents=True, exist_ok=True)
        request_path, response_path = folder / "request.json", folder / "response.json"
        request_path.write_text(json.dumps({
            "task_id": task.task_id, "prompt": task.prompt,
            "candidate_video": artifact.metadata.get("local_video_path"),
            "references": task.metadata.get("h3_references", []),
            "criteria": {name: rubric[name] for name in criteria},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        response_path.unlink(missing_ok=True)
        try:
            with (folder / "verifier.log").open("w") as log:
                subprocess.run([*self.command, "--request", str(request_path), "--response", str(response_path)],
                               stdout=log, stderr=subprocess.STDOUT, timeout=self.timeout_seconds, check=True)
            audio = json.loads(response_path.read_text())
            scores, evidence = audio["criterion_scores"], audio["criterion_evidence"]
            if not audio.get("verifier"):
                raise ValueError("audio verifier must identify its model/version in verifier")
            for name in criteria:
                score = float(scores[name])
                if not math.isfinite(score) or not 0 <= score <= 1 or not evidence.get(name):
                    raise ValueError(f"missing/invalid observed audio evidence for {name}")
                result.setdefault("criterion_scores", {})[name] = score
                result.setdefault("criterion_evidence", {})[name] = str(evidence[name])
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
            raise VideoApiError(f"H3 audio verifier failed; log={folder / 'verifier.log'}: {exc}") from exc
        self.visual_evaluator._normalize_benchmark_result(task, result)
        result.update(audio_evidence_available=True, audio_verifier=audio["verifier"],
                      audio_verifier_record=str(response_path))
        return result
