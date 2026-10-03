"""Bounded, auditable tail trimming of generated local-H3 story clips.

Fixed benchmark inputs are never edited. Every baseline/candidate story generation
uses the same rule before frame extraction, reference reuse, or concatenation.
"""
from pathlib import Path
import hashlib
import json
import math
import uuid

from evovideo_skill.h3_media import _command, _probe

POLICY = "local-h3-story-tail-trim-v1"
MAX_OVERRUN_SECONDS = .75


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def align_story_output(source: Path, seconds: int, root: Path) -> tuple[Path, dict]:
    if type(seconds) is not int or not 4 <= seconds <= 15:
        raise ValueError("story output alignment requires a native 4..15 second request")
    source, root = source.resolve(), root.resolve()
    info = _probe(source, "video")
    actual = info["duration_seconds"]  # primary video stream, not AAC/container tail
    delta = actual-seconds
    if not math.isfinite(actual) or delta < -.001 or delta > MAX_OVERRUN_SECONDS+.001:
        raise ValueError(f"Generated story clip is {actual:.3f}s for a {seconds}s request; "
                         f"alignment only trims overruns up to {MAX_OVERRUN_SECONDS}s per native call. "
                         "Short or substantially overlong clips require backend diagnosis; no padding or time warping is applied.")
    source_hash = sha256(source)
    record = {"policy": POLICY, "source_path": str(source), "source_sha256": source_hash,
              "requested_seconds": seconds, "source_video_seconds": actual,
              "max_overrun_seconds": MAX_OVERRUN_SECONDS,
              "operation": "trim_tail" if delta > .001 else "none",
              "removed_tail_seconds": max(0., delta),
              "qualification": "Tail content can be lost. Evaluate the actual retained clip, including final state; original output is preserved. No reference input is altered."}
    if delta <= .001:
        return source, {**record, "output_path": str(source), "output_sha256": source_hash,
                        "output_video_seconds": actual}
    root.mkdir(parents=True, exist_ok=True)
    output = root / f"aligned-story-{source_hash}-{seconds}s-v1.mp4"
    manifest = output.with_suffix(".alignment.json")
    if output.exists() and manifest.exists():
        saved = json.loads(manifest.read_text())
        if (saved.get("source_sha256") == source_hash and saved.get("policy") == POLICY
                and saved.get("requested_seconds") == seconds
                and saved.get("output_sha256") == sha256(output)):
            return output, {**saved, "source_path": str(source), "output_path": str(output)}
        raise ValueError(f"Aligned story output checksum/protocol changed: {output}; preserve evidence and use a new output directory")
    temporary = root / f".align-{uuid.uuid4().hex}.mp4"
    try:
        _command(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(source),
                  "-t", str(seconds), "-map", "0:v:0", "-map", "0:a?",
                  "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                  "-movflags", "+faststart", str(temporary)])
        aligned = _probe(temporary, "video")
        if (abs(aligned["duration_seconds"]-seconds) > 1/24+.001
                or len(aligned["audio_streams"]) != len(info["audio_streams"])):
            raise ValueError("Story alignment failed duration or audio-track preservation check")
        temporary.replace(output)
        record.update(output_path=str(output), output_sha256=sha256(output),
                      output_video_seconds=aligned["duration_seconds"])
        tmp_manifest = manifest.with_suffix(".tmp")
        tmp_manifest.write_text(json.dumps(record, indent=2)+"\n")
        tmp_manifest.replace(manifest)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[H3 output] aligned generated story clip {actual:.3f}s -> {seconds}s "
          f"(trim_tail); original={source}", flush=True)
    return output, record
