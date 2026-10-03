"""Materialized, explicitly routed media preparation for H3 consumers."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import urllib.parse
import urllib.request
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from evovideo_skill.models import VideoArtifact, VideoPlan, VideoTask
from evovideo_skill.tool_onboarding import ToolSpec
from evovideo_skill.tools import ToolExecutionContext, ToolRegistry, VideoTool
from evovideo_skill.video_processing import VideoProcessor


_TIMEOUT = 120
_SET = "h3_reference_set"
_ROLES = {
    "image": {"reference_image", "first_frame", "last_frame"},
    "video": {"reference_video"},
    "audio": {"reference_audio"},
}


def _command(argv: list[str]) -> str:
    if not shutil.which(argv[0]):
        raise RuntimeError(f"{argv[0]} is required for H3 media preparation")
    try:
        return subprocess.run(
            argv, check=True, capture_output=True, text=True, timeout=_TIMEOUT,
        ).stdout
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{argv[0]} timed out after {_TIMEOUT}s") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{argv[0]} failed: {exc.stderr[-3000:]}") from exc


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _uri_identity(uri: Any) -> str:
    uri = _identifier(uri, "uri")
    parsed = urllib.parse.urlparse(uri)
    if not parsed.scheme:
        return Path(uri).expanduser().resolve().as_uri()
    if parsed.scheme == "file" and parsed.netloc in {"", "localhost"}:
        return Path(urllib.request.url2pathname(parsed.path)).expanduser().resolve().as_uri()
    return uri


def _role(kind: Any, role: Any) -> None:
    if not isinstance(kind, str) or kind not in _ROLES:
        raise ValueError(f"unsupported reference kind: {kind!r}")
    if not isinstance(role, str) or role not in _ROLES[kind]:
        raise ValueError(f"role {role!r} is invalid for {kind}")


def _probe(path: Path, kind: str) -> dict[str, Any]:
    data = json.loads(_command([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path),
    ]))
    streams = data.get("streams", [])
    video = [s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    fmt = data.get("format", {})
    format_name = fmt.get("format_name", "")
    still = any(name == "image2" or name.endswith("_pipe") for name in format_name.split(","))
    if (kind == "image" and (not video or not still)) or (kind == "video" and (not video or still)) or (kind == "audio" and (not audio or video)):
        raise ValueError(f"file is not a {kind}: {path}")
    info: dict[str, Any] = {"has_audio": bool(audio), "audio_streams": audio}
    if video:
        if video[0].get("width", 0) <= 0 or video[0].get("height", 0) <= 0:
            raise ValueError(f"media has no usable image dimensions: {path}")
        info.update(width=video[0]["width"], height=video[0]["height"])
    if kind != "image":
        primary = video[0] if kind == "video" else audio[0]
        try:
            duration = float(primary.get("duration") or fmt.get("duration", 0))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"media has no usable duration: {path}") from exc
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"media has no positive duration: {path}")
        info["duration_seconds"] = duration
    # Probing headers alone can succeed for corrupt files, including fake PNGs.
    stream = "a" if kind == "audio" else "v"
    _command(["ffmpeg", "-v", "error", "-nostdin", "-xerror", "-i", str(path),
              "-map", f"0:{stream}:0", f"-frames:{stream}", "1", "-f", "null", "-"])
    return info


class _H3MediaTool(VideoTool):
    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = Path(output_dir or "outputs/h3_media").expanduser().resolve()

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        return self.run_with_context(task, plan, ToolExecutionContext(self.name, {}))

    def _output(self, suffix: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir / f"{self.name}-{uuid.uuid4().hex}{suffix}"

    def _local(self, uri: Any) -> Path:
        uri = _identifier(uri, "uri")
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme in {"http", "https"}:
            suffix = Path(parsed.path).suffix
            path = self._output(suffix if suffix and len(suffix) <= 10 else ".media")
            try:
                with urllib.request.urlopen(uri, timeout=_TIMEOUT) as response, path.open("wb") as output:
                    shutil.copyfileobj(response, output)
            except Exception:
                path.unlink(missing_ok=True)
                raise
        elif parsed.scheme == "file":
            if parsed.netloc not in {"", "localhost"}:
                raise ValueError("file URI must refer to the local host")
            path = Path(urllib.request.url2pathname(parsed.path)).expanduser().resolve()
        elif not parsed.scheme:
            path = Path(uri).expanduser().resolve()
        else:
            raise ValueError(f"unsupported media URI scheme: {parsed.scheme}")
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"media file does not exist or is empty: {path}")
        return path

    def _references(self, values: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not isinstance(values, list) or not values:
            raise ValueError("h3_references must be a nonempty list")
        references, media_info = [], {}
        for value in values:
            if not isinstance(value, dict):
                raise ValueError("each reference must be a dictionary")
            ref = dict(value)
            ident = _identifier(ref.get("id"), "reference id")
            if ident in media_info:
                raise ValueError(f"duplicate reference id: {ident}")
            _role(ref.get("kind"), ref.get("role"))
            if "semantic_role" in ref:
                _identifier(ref["semantic_role"], "semantic_role")
            if ref.get("duration_seconds") is not None and _number(ref["duration_seconds"], "duration_seconds") <= 0:
                raise ValueError("duration_seconds must be positive")
            path = self._local(ref.get("uri"))
            info = _probe(path, ref["kind"])
            ref["uri"] = str(path)
            if "duration_seconds" in info:
                ref["duration_seconds"] = info["duration_seconds"]
            else:
                ref.pop("duration_seconds", None)
            references.append(ref)
            media_info[ident] = info
        return references, media_info

    def _artifact(self, task: VideoTask, context: ToolExecutionContext, metadata: dict[str, Any], frames: list[dict[str, Any]] | None = None, sources: list[str] | None = None) -> VideoArtifact:
        sources = list(context.input_artifacts) if sources is None else sources
        parents = [context.input_artifacts[source] for source in sources]
        return VideoArtifact(
            artifact_id=uuid.uuid4().hex, task_id=task.task_id, prompt=task.prompt, mode=task.mode,
            tool_chain=[name for parent in parents for name in parent.tool_chain] + [self.name],
            frames=frames or [], metadata={
                **metadata, "materialized": True, "source_nodes": sources,
                "source_artifact_ids": [parent.artifact_id for parent in parents],
                "source_lineage": [{"source_node": source, "artifact_id": parent.artifact_id,
                                    "parents": parent.metadata.get("source_lineage", [])}
                                   for source, parent in zip(sources, parents)],
                "upstream_conditioning_consumed": bool(parents),
            },
        )

    def _set(self, task: VideoTask, context: ToolExecutionContext, refs: Any, **metadata: Any) -> VideoArtifact:
        refs, info = self._references(refs)
        return self._artifact(task, context, {
            "artifact_type": _SET, "h3_references": refs, "reference_media_info": info, **metadata,
        })

    @staticmethod
    def _single(context: ToolExecutionContext, kind: str) -> VideoArtifact:
        if len(context.input_artifacts) != 1:
            raise ValueError(f"exactly one upstream {kind} artifact is required")
        artifact = next(iter(context.input_artifacts.values()))
        if artifact.metadata.get("artifact_type") != kind:
            raise ValueError(f"expected upstream artifact_type={kind}")
        return artifact

    def _media(self, artifact: VideoArtifact, kind: str) -> tuple[Path, dict[str, Any]]:
        if artifact.metadata.get("artifact_type") != kind:
            raise ValueError(f"expected upstream artifact_type={kind}")
        meta = artifact.metadata
        uri = next((meta[key] for key in (f"local_{kind}_path", f"reference_{kind}", f"{kind}_path", f"{kind}_url", "uri") if meta.get(key)), None)
        path = self._local(uri)
        return path, _probe(path, kind)

    def _video(self, path: Path, task: VideoTask, plan: VideoPlan) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        info = _probe(path, "video")
        processed = VideoProcessor(self.output_dir, timeout_seconds=_TIMEOUT).process(
            path.as_uri(), path.stem, task.prompt, {"plan": plan.__dict__},
        )
        if not processed.frames:
            raise RuntimeError("transformed video yielded no sampled frames")
        return {**info, "local_video_path": processed.local_video_path,
                "reference_video": processed.local_video_path,
                "sampled_frame_paths": processed.sampled_frame_paths}, processed.frames


class H3ReferenceBankTool(_H3MediaTool):
    name = "h3_reference_bank"

    def run_with_context(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext) -> VideoArtifact:
        if context.input_artifacts:
            raise ValueError("h3_reference_bank is a task source and accepts no upstream parents")
        values = task.metadata.get("h3_references", [])
        if not isinstance(values, list):
            raise ValueError("task metadata h3_references must be a list")
        values = list(values)
        if task.reference_video:
            source_uri = _uri_identity(task.reference_video)
            source_present = False
            for ref in values:
                if not isinstance(ref, dict):
                    continue
                same_uri = _uri_identity(ref.get("uri")) == source_uri
                if ref.get("id") == "source-video" and not same_uri:
                    raise ValueError("source-video collides with a different task.reference_video URI")
                source_present = source_present or same_uri
            if not source_present:
                values.append({"id": "source-video", "kind": "video", "uri": task.reference_video, "role": "reference_video"})
        return self._set(task, context, values)


class H3ReferenceSelectTool(_H3MediaTool):
    name = "h3_reference_select"

    def run_with_context(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext) -> VideoArtifact:
        upstream = self._single(context, _SET)
        refs, _ = self._references(upstream.metadata.get("h3_references"))
        ids = context.node_config.get("reference_ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError("reference_ids must be an explicit nonempty list")
        for ident in ids:
            _identifier(ident, "reference_ids entry")
        if len(set(ids)) != len(ids):
            raise ValueError("reference_ids must not contain duplicates")
        lookup = {ref["id"]: ref for ref in refs}
        if any(ident not in lookup for ident in ids):
            raise ValueError("reference_ids contains an unknown id")
        roles = context.node_config.get("roles", {})
        if not isinstance(roles, dict) or any(ident not in ids for ident in roles):
            raise ValueError("roles must map selected reference ids to API roles")
        semantic_roles = context.node_config.get("semantic_roles", {})
        if not isinstance(semantic_roles, dict) or any(ident not in ids for ident in semantic_roles):
            raise ValueError("semantic_roles must map selected reference ids to nonempty strings")
        for value in semantic_roles.values():
            _identifier(value, "semantic_roles value")
        selected = [{**lookup[ident], "role": roles.get(ident, lookup[ident]["role"])} for ident in ids]
        for ref in selected:
            if ref["id"] in semantic_roles:
                ref["semantic_role"] = semantic_roles[ref["id"]]
        return self._set(task, context, selected, selected_reference_ids=list(ids))


class H3ReferencePackTool(_H3MediaTool):
    name = "h3_reference_pack"

    def run_with_context(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext) -> VideoArtifact:
        bindings = context.node_config.get("bindings")
        if not isinstance(bindings, list) or not bindings:
            raise ValueError("bindings must be an explicit nonempty ordered list")
        refs, sources = [], []
        for binding in bindings:
            if not isinstance(binding, dict):
                raise ValueError("each binding must be a dictionary")
            source = _identifier(binding.get("source"), "binding source")
            if source not in context.input_artifacts:
                raise ValueError(f"binding source is not an upstream parent: {source}")
            kind, role = binding.get("kind"), binding.get("role")
            _role(kind, role)
            upstream = context.input_artifacts[source]
            if upstream.metadata.get("materialization_mode") == "semantic_storyboard":
                raise ValueError("Symbolic storyboard is not a physical scene/identity anchor. "
                                 "Use h3_frame_extract on a real draft, fixed task references, "
                                 "or a verified image generation tool.")
            if upstream.metadata.get("artifact_type") == _SET:
                candidates, _ = self._references(upstream.metadata.get("h3_references"))
                if "reference_id" in binding:
                    ident = _identifier(binding["reference_id"], "reference_id")
                    candidates = [ref for ref in candidates if ref["id"] == ident]
                if len(candidates) != 1:
                    raise ValueError("set binding requires an unambiguous reference_id")
                ref = dict(candidates[0])
                if ref["kind"] != kind:
                    raise ValueError("binding kind does not match selected reference")
            else:
                path, info = self._media(upstream, kind)
                ref = {"id": binding.get("reference_id", source), "kind": kind, "uri": str(path)}
                if "duration_seconds" in info:
                    ref["duration_seconds"] = info["duration_seconds"]
            ref["role"] = role
            if "semantic_role" in binding:
                ref["semantic_role"] = binding["semantic_role"]
            refs.append(ref)
            if source not in sources:
                sources.append(source)
        refs, info = self._references(refs)
        metadata = {"artifact_type": _SET, "h3_references": refs,
                    "reference_media_info": info, "bindings": [dict(b) for b in bindings]}
        repair_instructions = list(dict.fromkeys(
            str(context.input_artifacts[source].metadata["repair_instruction"])
            for source in sources
            if context.input_artifacts[source].task_id == task.task_id
            and context.input_artifacts[source].metadata.get("repair_instruction")
        ))
        if repair_instructions:
            metadata["repair_instruction"] = "\n".join(repair_instructions)
        localizations = [context.input_artifacts[source].metadata["failure_localization"]
                         for source in sources if context.input_artifacts[source].task_id == task.task_id
                         and context.input_artifacts[source].metadata.get("failure_localization")]
        if localizations:
            if any(item != localizations[0] for item in localizations):
                raise ValueError("Reference pack has conflicting repair timelines")
            metadata["failure_localization"] = deepcopy(localizations[0])
        return self._artifact(task, context, metadata, sources=sources)


class H3FrameExtractTool(_H3MediaTool):
    name = "h3_frame_extract"

    def run_with_context(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext) -> VideoArtifact:
        upstream = self._single(context, "video")
        path, info = self._media(upstream, "video")
        config = context.node_config
        position = config.get("position")
        if not isinstance(position, str) or position not in {"first", "last", "time"}:
            raise ValueError("position must be first, last, or time")
        role = config.get("role", {"first": "first_frame", "last": "last_frame", "time": "reference_image"}[position])
        _role("image", role)
        args = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(path)]
        timestamp = None
        if position == "time":
            timestamp = _number(config.get("time_seconds"), "time_seconds")
            if not 0 <= timestamp < info["duration_seconds"]:
                raise ValueError("time_seconds must be within the video duration")
            args += ["-ss", str(timestamp)]
        elif "time_seconds" in config:
            raise ValueError("time_seconds is only valid for position=time")
        elif position == "last":
            count = json.loads(_command(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)]))
            frames = int(count["streams"][0]["nb_read_frames"])
            if frames <= 0:
                raise ValueError("video has no decodable frames")
            args += ["-vf", f"select=eq(n\\,{frames - 1})"]
        output = self._output(".png")
        _command(args + ["-map", "0:v:0", "-frames:v", "1", "-fps_mode", "vfr", str(output)])
        if not output.is_file():
            raise RuntimeError("requested frame could not be extracted")
        _probe(output, "image")
        return self._artifact(task, context, {
            "artifact_type": "image", "local_image_path": str(output), "reference_image": str(output),
            "uri": str(output), "role": role, "h3_role": role, "semantic_role": role, "position": position,
            "time_seconds": timestamp, "source_video": str(path),
        }, frames=[{"index": 0, "frame_path": str(output)}])


class H3AudioExtractTool(_H3MediaTool):
    name = "h3_audio_extract"

    def run_with_context(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext) -> VideoArtifact:
        upstream = self._single(context, "video")
        path, info = self._media(upstream, "video")
        if not info["has_audio"]:
            raise ValueError("cannot extract an audio reference from a silent video")
        start = _number(context.node_config.get("start_seconds", 0), "start_seconds")
        end = _number(context.node_config.get("end_seconds", info["duration_seconds"]), "end_seconds")
        if not 0 <= start < end <= info["duration_seconds"]:
            raise ValueError("audio bounds must satisfy 0 <= start_seconds < end_seconds <= duration")
        output = self._output(".wav")
        _command(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(path),
                  "-ss", str(start), "-t", str(end - start), "-map", "0:a:0", "-vn",
                  "-ar", "32000", "-ac", "2", "-c:a", "pcm_s16le", str(output)])
        audio = _probe(output, "audio")
        return self._artifact(task, context, {
            "artifact_type": "audio", "local_audio_path": str(output), "reference_audio": str(output),
            "uri": str(output), "duration_seconds": audio["duration_seconds"],
            "source_video": str(path), "start_seconds": start, "end_seconds": end,
        })


class H3ReferenceTrimTool(_H3MediaTool):
    name = "h3_reference_trim"

    def run_with_context(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext) -> VideoArtifact:
        upstream = self._single(context, _SET)
        refs, info = self._references(upstream.metadata.get("h3_references"))
        config = context.node_config
        ident = _identifier(config.get("reference_id"), "reference_id")
        selected = next((ref for ref in refs if ref["id"] == ident), None)
        if selected is None or selected["kind"] not in {"video", "audio"}:
            raise ValueError("reference_id must select a video or audio reference")
        start = _number(config.get("start_seconds"), "start_seconds")
        end = _number(config.get("end_seconds"), "end_seconds")
        if not 0 <= start < end <= info[ident]["duration_seconds"]:
            raise ValueError("trim bounds must satisfy 0 <= start_seconds < end_seconds <= duration")
        video = selected["kind"] == "video"
        output = self._output(".mp4" if video else ".wav")
        args = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", selected["uri"], "-ss", str(start), "-t", str(end - start)]
        if video:
            args += ["-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart"]
        else:
            args += ["-map", "0:a", "-c:a", "pcm_s16le"]
        _command(args + [str(output)])
        selected["uri"] = str(output)
        metadata, frames = {}, []
        if video:
            metadata, frames = self._video(output, task, plan)
        result = self._set(task, context, refs, trim={"reference_id": ident, "start_seconds": start, "end_seconds": end})
        result.frames = frames
        result.metadata["trimmed_media"] = metadata or _probe(output, "audio")
        return result


class H3AVConcatTool(_H3MediaTool):
    name = "h3_av_concat"

    def run_with_context(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext) -> VideoArtifact:
        sources = context.node_config.get("source_nodes")
        if not isinstance(sources, list) or not sources:
            raise ValueError("source_nodes must be an explicit nonempty ordered list")
        inputs = []
        for source in sources:
            _identifier(source, "source_nodes entry")
            if source not in context.input_artifacts:
                raise ValueError(f"source node is not an upstream parent: {source}")
            inputs.append(self._media(context.input_artifacts[source], "video"))
        width = max(2, inputs[0][1]["width"] // 2 * 2)
        height = max(2, inputs[0][1]["height"] // 2 * 2)
        audio_count = max(1, max(len(info["audio_streams"]) for _, info in inputs))
        args = ["ffmpeg", "-v", "error", "-nostdin", "-y"]
        for path, _ in inputs:
            args += ["-i", str(path)]
        filters, concat, lineage = [], [], []
        for index, (path, info) in enumerate(inputs):
            duration = info["duration_seconds"]
            filters.append(f"[{index}:v:0]setpts=PTS-STARTPTS,scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24,format=yuv420p,trim=duration={duration}[v{index}]")
            concat.append(f"[v{index}]")
            for track in range(audio_count):
                label = f"a{index}_{track}"
                if track < len(info["audio_streams"]):
                    # Keep offsets relative to the video, including delayed audio starts.
                    filters.append(f"[{index}:a:{track}]aresample=32000:async=1:first_pts=0,aformat=sample_fmts=fltp:channel_layouts=stereo,apad,atrim=duration={duration},asetpts=PTS-STARTPTS[{label}]")
                else:
                    filters.append(f"anullsrc=r=32000:cl=stereo,atrim=duration={duration},asetpts=PTS-STARTPTS[{label}]")
                concat.append(f"[{label}]")
            lineage.append({"source_node": sources[index], "uri": str(path), "duration_seconds": duration,
                            "has_audio": info["has_audio"], "silent_tracks_added": audio_count - len(info["audio_streams"])})
            alignment = context.input_artifacts[sources[index]].metadata.get("h3_output_alignment")
            if alignment:
                lineage[-1]["h3_output_alignment"] = deepcopy(alignment)
        outputs = "[video]" + "".join(f"[audio{track}]" for track in range(audio_count))
        filters.append("".join(concat) + f"concat=n={len(inputs)}:v=1:a={audio_count}" + outputs)
        args += ["-filter_complex", ";".join(filters), "-map", "[video]"]
        for track in range(audio_count):
            args += ["-map", f"[audio{track}]"]
        output = self._output(".mp4")
        _command(args + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(output)])
        metadata, frames = self._video(output, task, plan)
        return self._artifact(task, context, {"artifact_type": "video", **metadata, "concat_sources": lineage}, frames, sources)


def register_h3_media_tools(registry: ToolRegistry, output_dir: str | Path | None = None) -> None:
    """Register preparation tools only; generation consumers are registered separately."""
    definitions = [
        (H3ReferenceBankTool, (), _SET, (), "Config: {}. Materialize task.metadata.h3_references dictionaries {id,kind:image|video|audio,uri,role,semantic_role?,duration_seconds?}; absent/null durations are probed. Expose task.reference_video as source-video unless its URI is already included; reject source-video bound to a different URI."),
        (H3ReferenceSelectTool, (_SET,), _SET, (), "Select/reorder one set. Config: {reference_ids:[id,...] (required, unique, nonempty),roles?:{id:API_role},semantic_roles?:{id:nonempty_string}}. Both maps target selected ids only. API roles must match kinds; semantic_roles independently reassigns reference meaning."),
        (H3ReferencePackTool, ("image", "video", "audio", _SET), _SET, (), "Pack explicit upstream media only. Config: {bindings:[{source:upstream_node_id,kind:image|video|audio,role:API_role,semantic_role?:string,reference_id?:id},...]}. Order is preserved; reference_id selects within a set (required if ambiguous), or names a direct media reference (defaults to source node id; explicit stable reference_id preferred). Output ids must be unique."),
        (H3FrameExtractTool, ("video",), "image", ("reference_image",), "Extract an actual frame from one video. Config: {position:first|last|time,time_seconds:number (required only for time),role?:reference_image|first_frame|last_frame}. Time must be >=0 and <duration; default role follows position."),
        (H3AudioExtractTool, ("video",), "audio", ("reference_audio",), "Extract the first real soundtrack as 32kHz stereo WAV. Config: {start_seconds?:number,end_seconds?:number}; defaults to full video, rejects silent inputs and invalid time bounds."),
        (H3ReferenceTrimTool, (_SET,), _SET, (), "Trim one video/audio reference, retaining other set entries and video audio tracks. Audio trims produce local WAV pcm_s16le. Config: {reference_id:id,start_seconds:number,end_seconds:number}; 0<=start<end<=duration."),
        (H3AVConcatTool, ("video",), "video", ("reference_video", "reference_image"), "Normalize and concatenate parent videos with soundtracks in explicit order. Config: {source_nodes:[upstream_node_id,...] (required, nonempty)}. Repeated nodes repeat clips. Normalize to first input dimensions, 24fps, 32kHz stereo; add silence only for missing audio tracks."),
    ]
    for tool_class, inputs, output, bindings, description in definitions:
        registry.register(tool_class(output_dir), ToolSpec(
            name=tool_class.name, capability=tool_class.name, input_types=inputs, output_type=output,
            output_bindings=bindings, backend="builtin", provenance="builtin", verified=True,
            consumes_upstream=bool(inputs), estimated_cost=0.1, description=description +
            " API roles: image=reference_image|first_frame|last_frame; video=reference_video; audio=reference_audio.",
            output_contract={"artifact_type": output, "materialized": True},
        ))
