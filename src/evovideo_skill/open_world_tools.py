from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import re
import signal
import ssl
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from evovideo_skill.tool_onboarding import (
    CapabilityRequest,
    CommandToolManifest,
    ToolSpec,
    materialize_smoke_command,
    requires_local_ml_runtime,
    smoke_command_digest,
)
from evovideo_skill.structured_json import StructuredJSONError, parse_json_object


class OpenWorldToolError(RuntimeError):
    pass


ALLOWED_LOCAL_SECRET_NAMES = frozenset(
    {
        "HF_TOKEN",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
        "GITHUB_TOKEN",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "GIT_SSL_CAINFO",
    }
)


def _external_secret_names_from_text(text: str) -> list[str]:
    names: set[str] = set()
    suffixes = ("API_KEY", "ACCESS_KEY", "SECRET_KEY", "SECRET", "TOKEN")
    for match in re.finditer(r"\b([A-Z][A-Z0-9_]{2,})\b", text):
        name = match.group(1)
        if not name.endswith(suffixes):
            continue
        if name in ALLOWED_LOCAL_SECRET_NAMES or name.startswith(("HF_", "HUGGINGFACE_", "GITHUB_")):
            continue
        names.add(name)
    return sorted(names)


def _external_secret_references(synthesized: "SynthesizedTool") -> list[str]:
    """Inspect the executable plan without treating optional README integrations as requirements."""
    text_parts = [
        json.dumps(synthesized.manifest.to_dict(), sort_keys=True),
        json.dumps(synthesized.install_commands, sort_keys=True),
        json.dumps(synthesized.environment, sort_keys=True),
        *synthesized.adapter_files.values(),
    ]
    return _external_secret_names_from_text("\n".join(text_parts))


def _looks_like_hosted_inference_wrapper(candidate: "ExternalToolCandidate") -> bool:
    text = f"{candidate.name} {candidate.description}".lower()
    providers = (
        "muapi", "replicate", "fal.ai", "fal-ai", "runway api", "kling api",
        "seedance api", "volcengine api", "dashscope api",
    )
    wrappers = ("api", "sdk", "client", "wrapper", "hosted", "cloud inference")
    return any(provider in text for provider in providers) and any(marker in text for marker in wrappers)


@dataclass
class DiscoveryConfig:
    github_token: str | None = None
    huggingface_token: str | None = None
    max_candidates: int = 6
    timeout_seconds: int = 30
    allowed_sources: tuple[str, ...] = ("github", "huggingface")
    allowed_licenses: tuple[str, ...] = (
        "apache-2.0",
        "mit",
        "bsd-3-clause",
        "bsd-2-clause",
        "openrail",
        "creativeml-openrail-m",
    )


@dataclass
class ExternalToolCandidate:
    candidate_id: str
    source: str
    name: str
    url: str
    revision: str
    description: str = ""
    license: str | None = None
    stars_or_likes: int = 0
    downloads: int = 0
    updated_at: str | None = None
    tags: list[str] = field(default_factory=list)
    documentation: str = ""
    score: float = 0.0
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InternetToolDiscoverer:
    """Search allowlisted machine-readable sources for a missing capability."""

    CAPABILITY_QUERIES = {
        "image_conditioned_video_generation": (
            "image-to-video inference diffusion",
            "i2v video generation inference",
            "CogVideoX image to video inference",
            "Wan2.1 image to video inference",
            "Stable Video Diffusion image to video inference",
        ),
        "keyframe_conditioned_video_generation": (
            "keyframe conditioned image to video inference",
            "keyframe guided video generation diffusion inference",
            "first last frame video generation inference",
        ),
        "character_keyframe_video_generation": (
            "identity preserving keyframe video generation inference",
            "character consistent image to video inference",
            "reference character keyframe video diffusion",
        ),
        "prompt_based_shot_planning": (
            "video storyboard shot planning Python",
            "screenplay shot list scene breakdown Python",
            "text to multi-shot storyboard planning",
        ),
        "multi_shot_identity_conditioned_generation": (
            "multi-shot video character consistency inference",
            "identity conditioned video generation",
            "ConsisID identity preserving video generation inference",
            "video personalization character consistency inference",
        ),
        "motion_conditioned_video_generation": (
            "controllable video generation motion trajectory inference",
            "motion guided image to video inference",
            "camera object motion control video diffusion inference",
            "Wan Move controllable video generation inference",
            "MotionCtrl controllable video generation inference",
        ),
        "audio_conditioned_video_generation": (
            "audio conditioned video generation local inference",
            "audio driven video synthesis inference",
            "audio synchronized video generation diffusion inference",
            "LatentSync audio video synchronization inference",
            "MuseTalk audio driven video inference",
        ),
        "region_video_editing": (
            "mask guided video editing inference",
            "region video editing diffusion",
        ),
        "global_video_editing": (
            "instruction guided video editing inference",
            "text guided video editing diffusion",
        ),
        "video_style_transfer": (
            "video-to-video style transfer inference",
            "temporal video stylization inference",
        ),
        "temporal_deflickering": (
            "video temporal deflicker inference",
            "video flicker removal neural inference",
        ),
        "failed_segment_repair": (
            "video inpainting segment repair diffusion",
            "video completion inference mask",
        ),
    }
    NEGATIVE_REPOSITORY_TERMS = (
        "awesome-",
        "free-for-dev",
        "agents",
        "mcp-server",
        "roadmap",
        "interview",
        "resources-list",
    )
    EXCLUDED_REPOSITORIES = {
        "anil-matcha/open-generative-ai",
    }
    TRUSTED_FRAMEWORK_REPOSITORIES = (
        "huggingface/diffusers",
        "zai-org/cogvideo",
        "modeltc/lightx2v",
        "pku-yuangroup/consisid",
        "tencentarc/motionctrl",
        "bytedance/latentsync",
        "tmelyralab/musetalk",
        "wan-video/wan2.1",
        "sczhou/propainter",
        "chenyanglei/all-in-one-deflicker",
        "williamyang1991/rerender_a_video",
        "omerbt/tokenflow",
        "rehglab/rave",
    )
    CAPABILITY_REPOSITORY_PRIORS = {
        "image_conditioned_video_generation": (
            "wan-video/wan2.1",
            "zai-org/cogvideo",
            "huggingface/diffusers",
            "modeltc/lightx2v",
        ),
        "multi_shot_identity_conditioned_generation": (
            "pku-yuangroup/consisid",
            "zai-org/cogvideo",
            "wan-video/wan2.1",
        ),
        "motion_conditioned_video_generation": (
            "tencentarc/motionctrl",
            "wan-video/wan2.1",
        ),
        "audio_conditioned_video_generation": (
            "bytedance/latentsync",
            "tmelyralab/musetalk",
        ),
        "global_video_editing": (
            "omerbt/tokenflow",
            "williamyang1991/rerender_a_video",
        ),
        "region_video_editing": ("sczhou/propainter",),
        "failed_segment_repair": ("sczhou/propainter",),
        "temporal_deflickering": ("chenyanglei/all-in-one-deflicker",),
        "video_style_transfer": (
            "williamyang1991/rerender_a_video",
            "omerbt/tokenflow",
            "rehglab/rave",
        ),
    }
    VIDEO_TERMS = ("video", "i2v", "v2v", "temporal", "diffusion")
    CAPABILITY_TERMS = {
        "image_conditioned_video_generation": ("image", "i2v", "condition"),
        "keyframe_conditioned_video_generation": ("keyframe", "first frame", "last frame", "i2v"),
        "character_keyframe_video_generation": ("character consistency", "identity", "reference image", "keyframe"),
        "prompt_based_shot_planning": ("shot planning", "shot list", "storyboard", "scene breakdown", "multi-shot"),
        "multi_shot_identity_conditioned_generation": ("identity", "character", "multi-shot", "multishot"),
        "motion_conditioned_video_generation": ("motion", "trajectory", "control", "camera"),
        "audio_conditioned_video_generation": ("audio", "synchronization", "audio-driven", "lip sync"),
        "region_video_editing": ("mask", "region", "object edit", "inpaint"),
        "global_video_editing": ("video edit", "instruction", "text-guided", "text guided"),
        "video_style_transfer": ("style", "stylization", "video-to-video", "v2v"),
        "temporal_deflickering": ("deflicker", "flicker", "temporal consistency"),
        "failed_segment_repair": ("inpaint", "completion", "repair", "restoration"),
    }

    def __init__(
        self,
        config: DiscoveryConfig,
        http_get: Callable[[str, dict[str, str]], Any] | None = None,
    ):
        self.config = config
        self.http_get = http_get or self._http_get
        self.last_rejections: list[str] = []

    def search(self, request: CapabilityRequest) -> list[ExternalToolCandidate]:
        raw_queries = self.CAPABILITY_QUERIES.get(
            request.capability,
            (request.capability.replace("_", " "),),
        )
        queries = (raw_queries,) if isinstance(raw_queries, str) else raw_queries
        candidates: list[ExternalToolCandidate] = []
        self.last_rejections = []
        candidates.extend(self._repository_prior_candidates(request))
        for query in queries:
            if "github" in self.config.allowed_sources:
                candidates.extend(self._search_github(query))
            if "huggingface" in self.config.allowed_sources:
                candidates.extend(self._search_huggingface(query))
        candidates = list({candidate.candidate_id: candidate for candidate in candidates}.values())
        filtered = []
        for candidate in candidates:
            metadata = " ".join([candidate.name, candidate.description, *candidate.tags]).lower()
            if self._has_negative_repository_metadata(candidate, metadata):
                reason = "repository name/description matches a non-model list or agent project"
                self.last_rejections.append(f"{candidate.name}: metadata filter: {reason}")
                continue
            candidate.score = self._score(candidate, request)
            filtered.append(candidate)
        candidates = filtered
        candidates.sort(key=lambda item: (item.score, item.stars_or_likes, item.downloads), reverse=True)
        enriched = []
        pool_size = max(self.config.max_candidates * 8, self.config.max_candidates)
        for candidate in candidates[:pool_size]:
            candidate.documentation = self._fetch_documentation(candidate)
            accepted, reason = self._domain_candidate(candidate, request)
            if not accepted:
                self.last_rejections.append(f"{candidate.name}: documentation filter: {reason}")
                continue
            executable, reason = self._executable_evidence(candidate, request)
            if not executable:
                self.last_rejections.append(f"{candidate.name}: executable evidence filter: {reason}")
                continue
            candidate.score += 0.25
            candidate.evidence.append("executable_inference_evidence:1.000")
            enriched.append(candidate)
        enriched.sort(key=lambda item: (item.score, item.stars_or_likes, item.downloads), reverse=True)
        return enriched[: self.config.max_candidates]

    def _repository_prior_candidates(
        self,
        request: CapabilityRequest,
    ) -> list[ExternalToolCandidate]:
        """Resolve known research repositories directly instead of hoping search ranks them."""
        if "github" not in self.config.allowed_sources:
            return []
        repositories = self._preferred_repositories(request)
        if request.capability != "video_style_transfer":
            return []
        candidates: list[ExternalToolCandidate] = []
        for repository in repositories:
            try:
                payload = self.http_get(
                    f"https://api.github.com/repos/{repository}",
                    self._github_headers(),
                )
            except (OpenWorldToolError, TimeoutError, OSError) as exc:
                self.last_rejections.append(
                    f"{repository}: trusted repository lookup unavailable: {exc}"
                )
                continue
            if not isinstance(payload, dict):
                continue
            name = str(payload.get("full_name") or repository)
            revision = str(payload.get("default_branch") or "")
            if not revision:
                self.last_rejections.append(
                    f"{repository}: trusted repository has no default revision"
                )
                continue
            candidates.append(
                ExternalToolCandidate(
                    candidate_id=self._id("github", name, revision),
                    source="github",
                    name=name,
                    url=str(payload.get("clone_url") or f"https://github.com/{name}.git"),
                    revision=revision,
                    description=str(payload.get("description") or ""),
                    license=str((payload.get("license") or {}).get("spdx_id") or "").lower() or None,
                    stars_or_likes=int(payload.get("stargazers_count") or 0),
                    updated_at=payload.get("updated_at"),
                    tags=[str(payload.get("language") or "").lower()],
                    evidence=[
                        "trusted_capability_repository:1.000",
                        f"github_stars:{int(payload.get('stargazers_count') or 0)}",
                    ],
                )
            )
        return candidates

    def _preferred_repositories(self, request: CapabilityRequest) -> tuple[str, ...]:
        repositories = self.CAPABILITY_REPOSITORY_PRIORS.get(request.capability, ())
        if request.capability != "video_style_transfer":
            return repositories
        configured = os.environ.get("OPEN_WORLD_STYLE_REPOSITORIES", "")
        if not configured.strip():
            return repositories
        return tuple(
            item.strip()
            for item in configured.split(",")
            if item.strip()
        )

    def _search_github(self, query: str) -> list[ExternalToolCandidate]:
        encoded = urllib.parse.quote(f"{query} in:name,description,readme")
        result_limit = min(50, max(10, self.config.max_candidates * 5))
        url = f"https://api.github.com/search/repositories?q={encoded}&sort=stars&order=desc&per_page={result_limit}"
        payload = self.http_get(url, self._github_headers())
        items = payload.get("items", []) if isinstance(payload, dict) else []
        candidates = []
        for item in items:
            revision = str(item.get("default_branch") or "")
            full_name = str(item.get("full_name") or "")
            if not full_name or not revision:
                continue
            candidates.append(
                ExternalToolCandidate(
                    candidate_id=self._id("github", full_name, revision),
                    source="github",
                    name=full_name,
                    url=str(item.get("clone_url") or f"https://github.com/{full_name}.git"),
                    revision=revision,
                    description=str(item.get("description") or ""),
                    license=str((item.get("license") or {}).get("spdx_id") or "").lower() or None,
                    stars_or_likes=int(item.get("stargazers_count") or 0),
                    updated_at=item.get("updated_at"),
                    tags=[str(item.get("language") or "").lower()],
                    evidence=[f"github_stars:{int(item.get('stargazers_count') or 0)}"],
                )
            )
        return candidates

    def _search_huggingface(self, query: str) -> list[ExternalToolCandidate]:
        params = urllib.parse.urlencode(
            {"search": query, "limit": self.config.max_candidates, "full": "true", "sort": "downloads", "direction": "-1"}
        )
        payload = self.http_get(f"https://huggingface.co/api/models?{params}", self._hf_headers())
        items = payload if isinstance(payload, list) else []
        candidates = []
        for item in items:
            model_id = str(item.get("modelId") or item.get("id") or "")
            revision = str(item.get("sha") or "")
            if not model_id or not revision:
                continue
            card = item.get("cardData") or {}
            candidates.append(
                ExternalToolCandidate(
                    candidate_id=self._id("huggingface", model_id, revision),
                    source="huggingface",
                    name=model_id,
                    url=f"https://huggingface.co/{model_id}",
                    revision=revision,
                    description=str(item.get("description") or ""),
                    license=str(card.get("license") or "").lower() or None,
                    stars_or_likes=int(item.get("likes") or 0),
                    downloads=int(item.get("downloads") or 0),
                    updated_at=item.get("lastModified"),
                    tags=[str(tag).lower() for tag in item.get("tags", [])],
                    evidence=[
                        f"hf_downloads:{int(item.get('downloads') or 0)}",
                        f"hf_likes:{int(item.get('likes') or 0)}",
                    ],
                )
            )
        return candidates

    def _fetch_documentation(self, candidate: ExternalToolCandidate) -> str:
        try:
            if candidate.source == "github":
                commit_url = f"https://api.github.com/repos/{candidate.name}/commits/{urllib.parse.quote(candidate.revision)}"
                commit = self.http_get(commit_url, self._github_headers())
                if isinstance(commit, dict) and commit.get("sha"):
                    candidate.revision = str(commit["sha"])
                    candidate.candidate_id = self._id(candidate.source, candidate.name, candidate.revision)
                if not candidate.license:
                    try:
                        license_url = (
                            f"https://api.github.com/repos/{candidate.name}/license"
                            f"?ref={urllib.parse.quote(candidate.revision)}"
                        )
                        license_payload = self.http_get(license_url, self._github_headers())
                        spdx = str((license_payload.get("license") or {}).get("spdx_id") or "").lower()
                        candidate.license = spdx if spdx and spdx != "noassertion" else None
                    except OpenWorldToolError:
                        candidate.evidence.append("github_license_endpoint_unavailable")
                url = f"https://api.github.com/repos/{candidate.name}/readme?ref={urllib.parse.quote(candidate.revision)}"
                payload = self.http_get(url, self._github_headers())
                if isinstance(payload, dict) and payload.get("content"):
                    return base64.b64decode(str(payload["content"])).decode("utf-8", errors="replace")[:30000]
            if candidate.source == "huggingface":
                url = f"https://huggingface.co/{candidate.name}/resolve/{candidate.revision}/README.md"
                payload = self.http_get(url, self._hf_headers())
                return str(payload)[:30000]
        except (OpenWorldToolError, ValueError):
            return ""
        return ""

    def _score(self, candidate: ExternalToolCandidate, request: CapabilityRequest) -> float:
        text = " ".join([
            candidate.name,
            candidate.description,
            *candidate.tags,
            candidate.documentation[:12000],
        ]).lower()
        tokens = [token for token in request.capability.split("_") if len(token) > 2]
        semantic = sum(token in text for token in tokens) / max(1, len(tokens))
        popularity = min(1.0, (candidate.stars_or_likes / 2000.0) + (candidate.downloads / 500000.0))
        license_score = 1.0 if candidate.license in self.config.allowed_licenses else 0.0
        fixed_revision = 1.0 if len(candidate.revision) >= 7 else 0.3
        source_prior = 0.1 if candidate.source == "github" else 0.0
        preferred = tuple(item.lower() for item in self._preferred_repositories(request))
        normalized_name = candidate.name.lower()
        repository_prior = 0.0
        if normalized_name in preferred:
            repository_prior = 1.5 - 0.1 * preferred.index(normalized_name)
        candidate.evidence.extend(
            [
                f"semantic:{semantic:.3f}",
                f"license:{license_score:.3f}",
                f"fixed_revision:{fixed_revision:.3f}",
                f"repository_prior:{repository_prior:.3f}",
            ]
        )
        return (
            0.45 * semantic
            + 0.2 * popularity
            + 0.2 * license_score
            + 0.1 * fixed_revision
            + source_prior
            + repository_prior
        )

    def _domain_candidate(
        self,
        candidate: ExternalToolCandidate,
        request: CapabilityRequest,
    ) -> tuple[bool, str]:
        metadata = " ".join([
            candidate.name,
            candidate.description,
            *candidate.tags,
        ]).lower()
        text = f"{metadata} {candidate.documentation[:12000].lower()}"
        if self._has_negative_repository_metadata(candidate, metadata):
            return False, "repository name/description matches a non-model list or agent project"
        if not any(term in text for term in self.VIDEO_TERMS):
            return False, "no video-generation or temporal-model evidence in repository metadata"
        required = self.CAPABILITY_TERMS.get(request.capability, ())
        if required and not any(term in text for term in required):
            return False, f"metadata does not match capability terms {list(required)}"
        if not required:
            generic = {"video", "generation", "conditioned", "based", "tool"}
            inferred = [
                token for token in request.capability.lower().split("_")
                if len(token) > 3 and token not in generic
            ]
            matched = sum(token in text for token in inferred)
            minimum = min(2, len(inferred))
            if minimum and matched < minimum:
                return False, f"metadata matches only {matched}/{len(inferred)} inferred capability terms"
        if request.capability in {"region_video_editing", "failed_segment_repair"}:
            video_edit_evidence = (
                "video inpainting",
                "video completion",
                "video object removal",
                "masked video",
                "inference_propainter.py",
            )
            if not any(term in text for term in video_edit_evidence):
                return False, "no concrete video inpainting/editing inference evidence"
        return True, "video domain and capability metadata matched"

    def _has_negative_repository_metadata(
        self,
        candidate: ExternalToolCandidate,
        metadata: str | None = None,
    ) -> bool:
        normalized_name = candidate.name.lower()
        if normalized_name in self.EXCLUDED_REPOSITORIES:
            return True
        if normalized_name in self.TRUSTED_FRAMEWORK_REPOSITORIES:
            return False
        searchable = metadata if metadata is not None else " ".join(
            [candidate.name, candidate.description, *candidate.tags]
        ).lower()
        return any(term in searchable for term in self.NEGATIVE_REPOSITORY_TERMS)

    def _executable_evidence(
        self,
        candidate: ExternalToolCandidate,
        request: CapabilityRequest,
    ) -> tuple[bool, str]:
        documentation = candidate.documentation.lower()
        if not documentation.strip():
            return False, "README/model card is unavailable"
        preferred = tuple(item.lower() for item in self._preferred_repositories(request))
        if candidate.name.lower() in preferred:
            candidate.evidence.append(
                "trusted repository entrypoint delegated to Codex tree inspection"
            )
            return True, "trusted capability repository will be inspected by Codex"
        inference_terms = (
            "inference.py",
            "generate.py",
            "predict.py",
            "python -m",
            "from_pretrained",
            "diffusionpipeline",
            "image-to-video",
            "image_to_video",
        )
        if not any(term in documentation for term in inference_terms):
            return False, "documentation has no concrete Python inference entrypoint"
        return True, "documentation contains a Python inference entrypoint"

    def _http_get(self, url: str, headers: dict[str, str]) -> Any:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in {"api.github.com", "huggingface.co"}:
            raise OpenWorldToolError(f"blocked discovery URL: {url}")
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read()
                content_type = response.headers.get("Content-Type", "")
                if "json" in content_type or raw[:1] in {b"{", b"["}:
                    return json.loads(raw.decode("utf-8"))
                return raw.decode("utf-8", errors="replace")
        except (TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise OpenWorldToolError(f"discovery request failed for {url}: {exc}") from exc

    def _github_headers(self) -> dict[str, str]:
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "evovideo-open-world-tool-agent"}
        if self.config.github_token:
            headers["Authorization"] = f"Bearer {self.config.github_token}"
        return headers

    def _hf_headers(self) -> dict[str, str]:
        headers = {"User-Agent": "evovideo-open-world-tool-agent"}
        if self.config.huggingface_token:
            headers["Authorization"] = f"Bearer {self.config.huggingface_token}"
        return headers

    @staticmethod
    def _id(source: str, name: str, revision: str) -> str:
        return hashlib.sha256(f"{source}:{name}:{revision}".encode("utf-8")).hexdigest()[:16]


@dataclass
class ToolSynthesisConfig:
    model: str = "openai/gpt-5.6-sol"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str | None = None
    timeout_seconds: int = 120
    temperature: float = 0.1
    max_output_tokens: int = 6144
    reasoning_effort: str | None = "high"
    repair_attempts: int = 2
    allowed_base_images: tuple[str, ...] = (
        "python:3.11-slim",
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
    )


@dataclass
class SynthesizedTool:
    candidate: ExternalToolCandidate
    manifest: CommandToolManifest
    base_image: str
    install_commands: list[list[str]]
    source_subdir: str = "."
    adapter_files: dict[str, str] = field(default_factory=dict)
    rationale: str = ""
    risks: list[str] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "manifest": self.manifest.public_dict(),
            "base_image": self.base_image,
            "install_commands": self.install_commands,
            "source_subdir": self.source_subdir,
            "adapter_files": dict(self.adapter_files),
            "rationale": self.rationale,
            "risks": self.risks,
            "environment": dict(self.environment),
        }


class OpenAICompatibleToolSynthesizer:
    """Turn repository evidence into a declarative, reviewable tool adapter."""

    def __init__(
        self,
        config: ToolSynthesisConfig,
        request_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        self.config = config
        self.request_fn = request_fn

    def synthesize(
        self,
        request: CapabilityRequest,
        candidate: ExternalToolCandidate,
        deployment_feedback: list[str] | None = None,
        repository_files: list[str] | None = None,
        repository_context: str | None = None,
    ) -> SynthesizedTool:
        payload = self._request_payload(
            request,
            candidate,
            deployment_feedback=deployment_feedback,
            repository_files=repository_files,
            repository_context=repository_context,
        )
        current_payload = payload
        repair_errors: list[str] = []
        data: dict[str, Any] | None = None
        for attempt in range(self.config.repair_attempts + 1):
            response = self.request_fn(current_payload) if self.request_fn else self._post(current_payload)
            try:
                parsed = self._parse_response(response)
            except OpenWorldToolError as error:
                repair_errors.append(str(error))
            else:
                contract_errors = self._contract_errors(parsed)
                if not contract_errors:
                    data = parsed
                    break
                repair_errors.extend(contract_errors)
            if attempt >= self.config.repair_attempts:
                raise OpenWorldToolError(
                    "tool synthesizer repair failed after retries; " + "; ".join(repair_errors[-6:])
                )
            current_payload = self._repair_payload(payload, repair_errors[-3:])
        if data is None:
            raise OpenWorldToolError("tool synthesizer returned no structured adapter")
        manifest_payload = dict(data.get("manifest") or {})
        manifest_payload.setdefault("name", self._safe_name(f"discovered_{candidate.name.split('/')[-1]}"))
        manifest_payload["capability"] = request.capability
        manifest_payload.setdefault("input_types", request.required_input_types or ["video"])
        manifest_payload.setdefault("output_type", "video")
        manifest_payload["backend"] = "docker"
        manifest_payload["verified"] = False
        manifest_payload["provenance"] = f"internet:{candidate.source}:{candidate.name}@{candidate.revision}"
        manifest_payload.setdefault("consumes_upstream", bool(manifest_payload["input_types"]))
        manifest_payload.setdefault("command", [])
        manifest_payload.setdefault("output_arg", "")
        manifest_payload.setdefault("smoke_test_command", data.get("smoke_test_command", []))
        manifest = CommandToolManifest.from_dict(manifest_payload)
        install_commands = self._commands(data.get("install_commands", []))
        base_image = self._normalize_base_image(data.get("base_image"))
        return SynthesizedTool(
            candidate=candidate,
            manifest=manifest,
            base_image=base_image,
            install_commands=install_commands,
            source_subdir=str(data.get("source_subdir") or "."),
            adapter_files={
                str(path): str(content)
                for path, content in (data.get("adapter_files") or {}).items()
            } if isinstance(data.get("adapter_files", {}), dict) else {},
            rationale=str(data.get("rationale") or ""),
            risks=[str(item) for item in data.get("risks", [])],
            environment=dict(data.get("environment") or {}),
        )

    def _request_payload(
        self,
        request: CapabilityRequest,
        candidate: ExternalToolCandidate,
        deployment_feedback: list[str] | None = None,
        repository_files: list[str] | None = None,
        repository_context: str | None = None,
    ) -> dict[str, Any]:
        system = (
            "You synthesize reproducible adapters for video-agent tools. Use only commands explicitly supported by the supplied "
            "repository documentation. Return JSON only. Never request privileged containers, host mounts, curl-pipe-shell, "
            "background services, or access to secrets. The runtime command must write {output_video} and use declared "
            "{reference_image}, {reference_video}, and {prompt} placeholders where applicable. Never invent requirements.txt, "
            "inference.py, or another path: every repository-relative path in commands must occur in repository_files when that "
            "field is supplied. Infer semantic artifact contracts from the documented CLI/function signature: distinguish first "
            "frame, last frame, identity reference, keyframe sequence, mask, pose, and source video; include formats, transport, "
            "resolution, fps, frame count, materialization, and required bindings when known. When repairing a failed deployment, "
            "directly address deployment_feedback. Declare the required Python minor and whether micromamba is needed for "
            "legacy Torch/CUDA stacks; venv cannot change the Python interpreter. The smoke test must execute a lightweight "
            "GPU operation through custom CUDA extensions used by the runtime, not only import modules."
        )
        schema = {
            "base_image": self.config.allowed_base_images[0],
            "allowed_base_images": list(self.config.allowed_base_images),
            "source_subdir": ".",
            "install_commands": [["python", "-m", "pip", "install", "-r", "requirements.txt"]],
            "adapter_files": {
                "evovideo_adapter.py": "optional small repository-specific adapter source"
            },
            "environment": {"manager": "auto", "python": "3.10", "gpu_smoke": True},
            "manifest": {
                "name": "snake_case_tool_name",
                "input_types": ["video"],
                "output_type": "video",
                "input_contracts": [{
                    "artifact_type": "video",
                    "semantic_role": "source_video",
                    "formats": ["mp4"],
                    "transport": ["local_path"],
                    "materialized": True,
                    "required_bindings": ["reference_video"],
                }],
                "output_contract": {
                    "artifact_type": "video",
                    "formats": ["mp4"],
                    "transport": ["local_path"],
                    "materialized": True,
                },
                "estimated_cost": 2.0,
                "consumes_upstream": True,
                "command": ["python", "inference.py", "--input", "{reference_video}", "--prompt", "{prompt}", "--output", "{output_video}"],
                "input_bindings": {"video": "reference_video"},
            },
            "smoke_test_command": ["python", "inference.py", "--help"],
            "rationale": "evidence-based adapter explanation",
            "risks": [],
        }
        user = {
            "capability_request": asdict(request),
            "repository": candidate.to_dict(),
            "documentation": candidate.documentation[:12000],
            "repository_files": list(repository_files or [])[:600],
            "repository_entrypoint_context": str(repository_context or "")[:16000],
            "deployment_feedback": list(deployment_feedback or [])[-8:],
            "output_schema": schema,
        }
        payload = {
            "model": self.config.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
        }
        if self.config.reasoning_effort and _is_openai_reasoning_model(self.config.model):
            payload["reasoning_effort"] = self.config.reasoning_effort
            payload["max_completion_tokens"] = self.config.max_output_tokens
        else:
            payload["temperature"] = self.config.temperature
            payload["max_tokens"] = self.config.max_output_tokens
        return payload

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.config.api_key:
            raise OpenWorldToolError(
                "tool synthesis requires GRAPH_LLM_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY, "
                "or DASHSCOPE_API_KEY"
            )
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise OpenWorldToolError(f"tool synthesis request failed: {exc}") from exc

    @staticmethod
    def _parse_response(response: dict[str, Any]) -> dict[str, Any]:
        try:
            choice = response["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenWorldToolError(f"invalid tool synthesis response: {response}") from exc
        content = message.get("content") if isinstance(message, dict) else None
        if not content and isinstance(message, dict):
            tool_calls = message.get("tool_calls") or []
            if tool_calls and isinstance(tool_calls[0], dict):
                content = (tool_calls[0].get("function") or {}).get("arguments")
            if not content:
                content = (message.get("function_call") or {}).get("arguments")
        if not content and isinstance(response, dict):
            content = response.get("output_text")
        if not content:
            finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
            refusal = message.get("refusal") if isinstance(message, dict) else None
            raise OpenWorldToolError(
                f"tool synthesizer returned empty structured content "
                f"(finish_reason={finish_reason!r}, refusal={refusal!r})"
            )
        try:
            return parse_json_object(content)
        except StructuredJSONError as exc:
            raise OpenWorldToolError(f"tool synthesizer returned invalid JSON: {str(content)[:1000]}") from exc

    def _normalize_base_image(self, raw: Any) -> str:
        candidates = raw if isinstance(raw, list) else [raw]
        for item in candidates:
            value = str(item or "").strip()
            if value in self.config.allowed_base_images:
                return value
        return self.config.allowed_base_images[0]

    @staticmethod
    def _repair_payload(payload: dict[str, Any], errors: list[str]) -> dict[str, Any]:
        repaired = json.loads(json.dumps(payload))
        if any("finish_reason='length'" in error for error in errors):
            token_key = "max_completion_tokens" if "max_completion_tokens" in repaired else "max_tokens"
            current = int(repaired.get(token_key, 4096))
            cap = max(current, int(os.environ.get("OPEN_WORLD_SYNTHESIS_LENGTH_RETRY_MAX_TOKENS", "16384")))
            repaired[token_key] = min(cap, current * 2)
            if "reasoning_effort" in repaired:
                repaired["reasoning_effort"] = "medium"
            try:
                user = json.loads(repaired["messages"][1]["content"])
            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                user = None
            if isinstance(user, dict):
                user["documentation"] = str(user.get("documentation", ""))[:6000]
                user["repository_files"] = list(user.get("repository_files", []))[:300]
                user["repository_entrypoint_context"] = str(
                    user.get("repository_entrypoint_context", "")
                )[:8000]
                repaired["messages"][1]["content"] = json.dumps(user, ensure_ascii=False)
        repaired["messages"].append(
            {
                "role": "user",
                "content": (
                    "Your previous response was invalid or truncated. Return one compact, complete JSON object only. "
                    "Do not repeat repository documentation or capability_request. base_image must be one string from "
                    "allowed_base_images, not an array. Keep commands as token arrays. Fix these validation errors: "
                    + "; ".join(errors)
                ),
            }
        )
        return repaired

    @staticmethod
    def _contract_errors(data: dict[str, Any]) -> list[str]:
        manifest = data.get("manifest")
        if not isinstance(manifest, dict):
            return ["manifest must be an object"]
        command = manifest.get("command")
        errors: list[str] = []
        if not isinstance(command, list) or not command:
            errors.append("manifest.command must be a non-empty token array")
        elif not any("{output_video}" in str(token) for token in command):
            errors.append("manifest.command must bind {output_video}")
        if not isinstance(data.get("install_commands", []), list):
            errors.append("install_commands must be a list")
        input_contracts = manifest.get("input_contracts", [])
        if input_contracts and (
            not isinstance(input_contracts, list)
            or any(not isinstance(item, dict) or not item.get("artifact_type") for item in input_contracts)
        ):
            errors.append("manifest.input_contracts must be a list of objects with artifact_type")
        output_contract = manifest.get("output_contract", {})
        if output_contract and (
            not isinstance(output_contract, dict) or not output_contract.get("artifact_type")
        ):
            errors.append("manifest.output_contract must be an object with artifact_type")
        adapter_files = data.get("adapter_files", {})
        if adapter_files and (
            not isinstance(adapter_files, dict)
            or any(not isinstance(path, str) or not isinstance(content, str) for path, content in adapter_files.items())
        ):
            errors.append("adapter_files must map relative string paths to string contents")
        return errors

    @staticmethod
    def _commands(raw: Any) -> list[list[str]]:
        if not isinstance(raw, list):
            raise OpenWorldToolError("install_commands must be a list")
        commands = []
        for item in raw:
            if isinstance(item, str):
                commands.append(shlex.split(item))
            elif isinstance(item, list):
                commands.append([str(token) for token in item])
            else:
                raise OpenWorldToolError("each install command must be a string or token list")
        return commands

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")[:80]


@dataclass
class ToolSecurityPolicy:
    allowed_licenses: tuple[str, ...]
    allowed_base_images: tuple[str, ...]
    max_install_commands: int = 8
    max_adapter_files: int = 8
    max_adapter_file_chars: int = 100_000

    FORBIDDEN_TOKENS = {
        "sudo",
        "--privileged",
        "--pid=host",
        "--network=host",
        "/var/run/docker.sock",
        "curl",
        "wget",
        "nc",
        "netcat",
        "ssh",
    }
    FORBIDDEN_EXECUTABLES = {"sh", "bash", "zsh", "fish", "dash", "cmd", "powershell", "pwsh"}

    def validate(self, synthesized: SynthesizedTool) -> list[str]:
        errors: list[str] = []
        candidate = synthesized.candidate
        if candidate.license not in self.allowed_licenses:
            errors.append(f"license {candidate.license!r} is not allowlisted")
        if len(candidate.revision) < 7:
            errors.append("candidate is not pinned to a revision")
        if synthesized.base_image not in self.allowed_base_images:
            errors.append(f"base image {synthesized.base_image!r} is not allowlisted")
        if len(synthesized.install_commands) > self.max_install_commands:
            errors.append("too many install commands")
        if len(synthesized.adapter_files) > self.max_adapter_files:
            errors.append("too many generated adapter files")
        for raw_path, content in synthesized.adapter_files.items():
            path = Path(raw_path)
            if path.is_absolute() or ".." in path.parts or not raw_path.strip():
                errors.append(f"adapter file escapes source_subdir: {raw_path!r}")
            if path.suffix.lower() not in {".py", ".json", ".yaml", ".yml", ".toml"}:
                errors.append(f"adapter file type is not allowlisted: {raw_path!r}")
            if len(content) > self.max_adapter_file_chars:
                errors.append(f"adapter file is too large: {raw_path!r}")
        for command in [*synthesized.install_commands, synthesized.manifest.command, synthesized.manifest.smoke_test_command]:
            if not command:
                continue
            lowered = {str(token).lower() for token in command}
            forbidden = lowered & self.FORBIDDEN_TOKENS
            if forbidden:
                errors.append(f"forbidden command tokens: {sorted(forbidden)}")
            executable = Path(str(command[0])).name.lower()
            if executable in self.FORBIDDEN_EXECUTABLES:
                errors.append(f"shell interpreters are not allowed: {executable}")
        if not synthesized.manifest.command:
            errors.append("runtime command is empty")
        if not any("{output_video}" in token for token in synthesized.manifest.command):
            errors.append("runtime command does not bind {output_video}")
        if not synthesized.manifest.smoke_test_command:
            errors.append("smoke test command is required for an open-world tool")
        if os.environ.get("OPEN_WORLD_ALLOW_EXTERNAL_API_TOOLS", "0") != "1":
            secret_names = _external_secret_references(synthesized)
            if secret_names:
                errors.append(
                    "local open-world tools must not depend on external inference API credentials: "
                    + ", ".join(secret_names)
                )
            if _looks_like_hosted_inference_wrapper(candidate):
                errors.append(
                    "local open-world candidate appears to be a hosted inference API wrapper rather than local model code"
                )
        runtime = synthesized.manifest.command
        smoke = synthesized.manifest.smoke_test_command
        if len(runtime) > 1 and Path(runtime[0]).name.lower() in {"python", "python3"}:
            runtime_entrypoint = runtime[1]
            if runtime_entrypoint == "-c":
                errors.append("inline python -c runtime commands are not admissible; generate a reviewable adapter file")
        return errors


@dataclass
class SandboxBuildConfig:
    root_dir: str = "outputs/open_world_tools"
    docker_bin: str = "docker"
    build_timeout_seconds: int = 1800
    smoke_timeout_seconds: int = 300
    memory_limit: str = "32g"
    pids_limit: int = 512
    gpu_enabled: bool = True
    allow_huggingface_model_clone: bool = False
    git_ca_info: str | None = None


@dataclass
class SandboxBuildResult:
    status: str
    manifest: CommandToolManifest | None
    image: str | None
    workspace: str
    evidence: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "image": self.image,
            "workspace": self.workspace,
            "evidence": self.evidence,
        }


def _missing_declared_paths(synthesized: SynthesizedTool, cwd: Path) -> list[str]:
    """Reject hallucinated repository paths before package installation or model loading."""
    missing: list[str] = []
    commands = [
        *synthesized.install_commands,
        synthesized.manifest.command,
        synthesized.manifest.smoke_test_command,
    ]
    for command in commands:
        if not command:
            continue
        executable = Path(str(command[0])).name.lower()
        for index, token in enumerate(command[:-1]):
            if token not in {"-r", "--requirement"}:
                continue
            declared = str(command[index + 1])
            if "{" not in declared and not (cwd / declared).exists():
                missing.append(declared)
        if executable in {"python", "python3"} and len(command) > 1:
            entrypoint = str(command[1])
            if (
                entrypoint not in {"-m", "-c"}
                and "{" not in entrypoint
                and (entrypoint.endswith(".py") or "/" in entrypoint)
                and not (cwd / entrypoint).exists()
            ):
                missing.append(entrypoint)
    return list(dict.fromkeys(missing))


def _materialize_adapter_files(synthesized: SynthesizedTool, cwd: Path) -> None:
    """Write bounded agent-generated adapters inside the selected repository subdirectory."""
    root = cwd.resolve()
    for raw_path, content in synthesized.adapter_files.items():
        path = (root / raw_path).resolve()
        if not path.is_relative_to(root):
            raise OpenWorldToolError(f"adapter file escapes source_subdir: {raw_path!r}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _manifest_paths_available(manifest: CommandToolManifest) -> bool:
    cwd = Path(manifest.cwd).expanduser() if manifest.cwd else None
    if cwd is None or not cwd.is_dir() or not manifest.command:
        return False
    executable = Path(manifest.command[0]).expanduser()
    if not executable.is_file():
        return False
    for command in (manifest.command, manifest.smoke_test_command):
        if len(command) < 2 or Path(command[0]).name.lower() not in {"python", "python3"}:
            continue
        entrypoint = command[1]
        if entrypoint in {"-m", "-c"} or "{" in entrypoint:
            continue
        if entrypoint.endswith(".py") or "/" in entrypoint:
            path = Path(entrypoint).expanduser()
            if not path.is_absolute():
                path = cwd / path
            if not path.is_file():
                return False
    return True


class DockerSandboxBuilder:
    """Build untrusted external tools inside a hardened, versioned Docker image."""

    def __init__(
        self,
        config: SandboxBuildConfig,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ):
        self.config = config
        self.runner = runner or subprocess.run

    def build(self, synthesized: SynthesizedTool) -> SandboxBuildResult:
        workspace = Path(self.config.root_dir) / synthesized.candidate.candidate_id
        source_dir = workspace / "source"
        workspace.mkdir(parents=True, exist_ok=True)
        evidence: list[str] = []
        if shutil.which(self.config.docker_bin) is None and self.runner is subprocess.run:
            return SandboxBuildResult("blocked", None, None, str(workspace), ["docker executable is unavailable"])
        try:
            self._materialize_source(synthesized.candidate, source_dir)
            cwd = (source_dir / synthesized.source_subdir).resolve()
            if not cwd.is_relative_to(source_dir.resolve()) or not cwd.exists():
                raise OpenWorldToolError(f"invalid or missing source_subdir: {synthesized.source_subdir}")
            _materialize_adapter_files(synthesized, cwd)
            missing_paths = _missing_declared_paths(synthesized, cwd)
            if missing_paths:
                raise OpenWorldToolError(
                    "adapter references repository paths that do not exist: " + ", ".join(missing_paths)
                )
            dockerfile = self._write_dockerfile(workspace, synthesized)
            image = f"evovideo-tool-{synthesized.candidate.candidate_id}:latest"
            build = self._run(
                [self.config.docker_bin, "build", "--pull=false", "-f", str(dockerfile), "-t", image, str(workspace)],
                timeout=self.config.build_timeout_seconds,
            )
            if build.returncode != 0:
                return SandboxBuildResult("blocked", None, None, str(workspace), [f"docker build failed: {(build.stderr or build.stdout)[-1500:]}"])
            evidence.append("docker image build passed")
            smoke = synthesized.manifest.smoke_test_command
            if smoke:
                command = self._hardened_run_prefix(image) + smoke
                result = self._run(command, timeout=self.config.smoke_timeout_seconds)
                if result.returncode != 0:
                    return SandboxBuildResult("blocked", None, image, str(workspace), [*evidence, f"container smoke test failed: {(result.stderr or result.stdout)[-1500:]}"])
                evidence.append("offline container smoke test passed")
            spec = ToolSpec(**{**synthesized.manifest.spec.__dict__, "verified": True, "backend": "docker"})
            manifest = synthesized.manifest
            manifest.spec = spec
            manifest.container_image = image
            manifest.smoke_test_command = []
            return SandboxBuildResult("ready", manifest, image, str(workspace), evidence)
        except (OSError, subprocess.TimeoutExpired, OpenWorldToolError) as exc:
            return SandboxBuildResult("blocked", None, None, str(workspace), [str(exc)])

    def image_available(self, image: str) -> bool:
        if not image:
            return False
        if shutil.which(self.config.docker_bin) is None and self.runner is subprocess.run:
            return False
        try:
            result = self._run([self.config.docker_bin, "image", "inspect", image], timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0

    def manifest_available(self, manifest: CommandToolManifest) -> bool:
        return bool(manifest.container_image and self.image_available(manifest.container_image))

    def _materialize_source(self, candidate: ExternalToolCandidate, source_dir: Path) -> None:
        revision_marker = source_dir / ".evovideo_revision"
        if source_dir.exists() and revision_marker.exists():
            if revision_marker.read_text(encoding="utf-8").strip() == candidate.revision:
                return
        if source_dir.exists():
            shutil.rmtree(source_dir)
        if candidate.source == "github":
            clone = self._run(self._git_command("clone", "--filter=blob:none", "--no-checkout", candidate.url, str(source_dir)), timeout=600)
            if clone.returncode != 0:
                raise OpenWorldToolError(self._clone_error("git clone", clone))
            checkout = self._run(self._git_command("-C", str(source_dir), "checkout", "--detach", candidate.revision), timeout=300)
            if checkout.returncode != 0:
                raise OpenWorldToolError(f"git checkout failed: {(checkout.stderr or checkout.stdout)[-1000:]}")
            revision_marker.write_text(candidate.revision + "\n", encoding="utf-8")
            return
        if candidate.source == "huggingface":
            if not self.config.allow_huggingface_model_clone:
                raise OpenWorldToolError(
                    "direct Hugging Face model cloning is disabled; use a GitHub inference repository or explicitly enable model cloning"
                )
            url = f"https://huggingface.co/{candidate.name}"
            clone = self._run(self._git_command("clone", "--filter=blob:none", url, str(source_dir)), timeout=600)
            if clone.returncode != 0:
                raise OpenWorldToolError(self._clone_error("Hugging Face clone", clone))
            checkout = self._run(self._git_command("-C", str(source_dir), "checkout", "--detach", candidate.revision), timeout=300)
            if checkout.returncode != 0:
                raise OpenWorldToolError(f"Hugging Face checkout failed: {(checkout.stderr or checkout.stdout)[-1000:]}")
            revision_marker.write_text(candidate.revision + "\n", encoding="utf-8")
            return
        raise OpenWorldToolError(f"unsupported source {candidate.source!r}")

    def _git_command(self, *args: str) -> list[str]:
        command = ["git"]
        if self.config.git_ca_info:
            command.extend(["-c", f"http.sslCAInfo={self.config.git_ca_info}"])
        command.extend(args)
        return command

    def _clone_error(self, operation: str, result: subprocess.CompletedProcess[str]) -> str:
        detail = (result.stderr or result.stdout)[-1000:]
        ca = self.config.git_ca_info or "system default"
        return (
            f"{operation} failed using CA bundle {ca}: {detail}. "
            "Set OPEN_WORLD_GIT_CAINFO to a valid PEM CA bundle; TLS verification is never disabled."
        )

    def _write_dockerfile(self, workspace: Path, synthesized: SynthesizedTool) -> Path:
        lines = [
            f"FROM {synthesized.base_image}",
            "WORKDIR /tool",
            "COPY source/ /tool/",
        ]
        if synthesized.source_subdir not in {"", "."}:
            safe_subdir = synthesized.source_subdir.strip("/")
            if ".." in Path(safe_subdir).parts:
                raise OpenWorldToolError("source_subdir cannot escape the repository")
            lines.append(f"WORKDIR /tool/{safe_subdir}")
        for command in synthesized.install_commands:
            lines.append("RUN " + " ".join(shlex.quote(token) for token in command))
        lines.extend(["ENV PYTHONUNBUFFERED=1", "ENTRYPOINT []"])
        path = workspace / "Dockerfile.generated"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _hardened_run_prefix(self, image: str) -> list[str]:
        command = [
            self.config.docker_bin,
            "run",
            "--rm",
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.config.pids_limit),
            "--memory",
            self.config.memory_limit,
        ]
        if self.config.gpu_enabled:
            command.extend(["--gpus", "all"])
        command.append(image)
        return command

    def _run(self, command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        return self.runner(command, capture_output=True, text=True, timeout=timeout, check=False)


@dataclass
class VenvBuildConfig:
    root_dir: str = "outputs/open_world_tools"
    python_bin: str = sys.executable
    build_timeout_seconds: int = 1800
    total_timeout_seconds: int = 1800
    idle_timeout_seconds: int = 300
    heartbeat_seconds: int = 30
    smoke_timeout_seconds: int = 300
    require_approval: bool = True
    approval_file: str | None = None
    allowed_repositories: tuple[str, ...] = ()
    allow_huggingface_model_clone: bool = False
    git_ca_info: str | None = None
    retry_without_proxy: bool = True
    environment_manager: str = "venv"
    micromamba_bin: str = "micromamba"
    python_version: str | None = None
    gpu_smoke_required: bool = False
    require_pinned_model_revision: bool = False
    require_model_load_smoke: bool = False


@dataclass(frozen=True)
class HuggingFaceModelReference:
    model_id: str
    revision: str | None
    source: str


class ToolApprovalStore:
    """Persist explicit allow/deny decisions for host-executed tool candidates."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()

    def decisions(self) -> dict[str, list[str]]:
        if not self.path.exists():
            return {"approved": [], "rejected": []}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"approved": [], "rejected": []}
        if isinstance(payload, list):
            return {"approved": [str(item) for item in payload], "rejected": []}
        if not isinstance(payload, dict):
            return {"approved": [], "rejected": []}
        return {
            "approved": [str(item) for item in payload.get("approved", [])],
            "rejected": [str(item) for item in payload.get("rejected", payload.get("denied", []))],
        }

    def status(self, candidate: ExternalToolCandidate) -> str:
        identifiers = {candidate.candidate_id, candidate.name, candidate.url}
        decisions = self.decisions()
        if identifiers.intersection(decisions["approved"]):
            return "approved"
        if identifiers.intersection(decisions["rejected"]):
            return "rejected"
        return "pending"

    def set_decision(self, candidate_id: str, approved: bool) -> None:
        candidate_id = str(candidate_id).strip()
        if not candidate_id:
            raise ValueError("candidate id cannot be empty")
        decisions = self.decisions()
        decisions["approved"] = [item for item in decisions["approved"] if item != candidate_id]
        decisions["rejected"] = [item for item in decisions["rejected"] if item != candidate_id]
        decisions["approved" if approved else "rejected"].append(candidate_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(decisions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_tool_plans(candidates_dir: str | Path) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for path in sorted(Path(candidates_dir).expanduser().glob("*/tool_plan.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("candidate"), dict):
            continue
        payload["plan_path"] = str(path.resolve())
        plans.append(payload)
    return plans


class SearchOnlyToolBuilder:
    """Persist synthesized adapters for review without installing or executing code."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)

    def build(self, synthesized: SynthesizedTool) -> SandboxBuildResult:
        workspace = self.root_dir / synthesized.candidate.candidate_id
        workspace.mkdir(parents=True, exist_ok=True)
        self._write_plan(workspace, synthesized, "search-only")
        return SandboxBuildResult(
            "pending",
            None,
            None,
            str(workspace),
            [f"search-only candidate saved for approval: {workspace / 'tool_plan.json'}"],
        )

    @staticmethod
    def _write_plan(workspace: Path, synthesized: SynthesizedTool, backend: str) -> None:
        payload = {
            "backend": backend,
            "candidate": synthesized.candidate.to_dict(),
            "manifest": synthesized.manifest.to_dict(),
            "install_commands": synthesized.install_commands,
            "source_subdir": synthesized.source_subdir,
            "adapter_files": synthesized.adapter_files,
            "rationale": synthesized.rationale,
            "risks": synthesized.risks,
            "environment": synthesized.environment,
        }
        (workspace / "tool_plan.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def manifest_available(manifest: CommandToolManifest) -> bool:
        del manifest
        return False


class VenvSandboxBuilder(DockerSandboxBuilder):
    """Install an approved pinned repository into a dedicated Python venv.

    A venv isolates Python dependencies, not the host filesystem or network.
    Automatic execution therefore requires an explicit opt-in.
    """

    def __init__(
        self,
        config: VenvBuildConfig,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        input_fn: Callable[[str], str] | None = None,
        output_fn: Callable[[str], None] | None = None,
    ):
        super().__init__(
            SandboxBuildConfig(
                root_dir=config.root_dir,
                build_timeout_seconds=config.build_timeout_seconds,
                smoke_timeout_seconds=config.smoke_timeout_seconds,
                allow_huggingface_model_clone=config.allow_huggingface_model_clone,
                git_ca_info=config.git_ca_info,
            ),
            runner=runner,
        )
        self.venv_config = config
        self.input_fn = input_fn or input
        self.output_fn = output_fn or print
        self._active_workspace: Path | None = None
        self._build_started: float | None = None
        self._active_stage = "idle"
        self._command_counter = 0

    def build(self, synthesized: SynthesizedTool) -> SandboxBuildResult:
        candidate = synthesized.candidate
        workspace = Path(self.venv_config.root_dir) / candidate.candidate_id
        source_dir = workspace / "source"
        venv_dir = workspace / ".venv"
        workspace.mkdir(parents=True, exist_ok=True)
        requested_python = self._requested_python_version(synthesized)
        requested_manager = self._requested_environment_manager(synthesized)
        SearchOnlyToolBuilder._write_plan(workspace, synthesized, requested_manager)
        approval_status = self._approval_status(candidate)
        if approval_status == "pending" and self._interactive_approval_enabled():
            approval_status = self._prompt_for_approval(synthesized, workspace)
        if approval_status == "rejected":
            return SandboxBuildResult(
                "blocked",
                None,
                None,
                str(workspace),
                [f"venv execution was rejected for candidate_id={candidate.candidate_id} repository={candidate.name}"],
            )
        if approval_status != "approved":
            return SandboxBuildResult(
                "pending",
                None,
                None,
                str(workspace),
                [
                    f"venv execution requires approval for candidate_id={candidate.candidate_id} "
                    f"repository={candidate.name}; review {workspace / 'tool_plan.json'}, then add the candidate id or "
                    "repository to OPEN_WORLD_VENV_APPROVAL_FILE, or run: "
                    f"PYTHONPATH=src python -m evovideo_skill.cli tool-approvals --candidates-dir "
                    f"{workspace.parent} --approve {candidate.candidate_id}"
                ],
            )
        evidence = [
            "local environment candidate explicitly approved",
            f"environment request: manager={requested_manager}, python={requested_python or 'host-compatible'}",
        ]
        self._active_workspace = workspace
        self._build_started = time.monotonic()
        self._command_counter = 0
        self._set_build_stage(workspace, "source", candidate=candidate.name)
        try:
            self._materialize_source(candidate, source_dir)
            cwd = (source_dir / synthesized.source_subdir).resolve()
            if not cwd.is_relative_to(source_dir.resolve()) or not cwd.exists():
                return self._blocked(workspace, f"invalid or missing source_subdir: {synthesized.source_subdir}")
            _materialize_adapter_files(synthesized, cwd)
            missing_paths = _missing_declared_paths(synthesized, cwd)
            if missing_paths:
                return self._blocked(
                    workspace,
                    "adapter references repository paths that do not exist: " + ", ".join(missing_paths),
                )
            model_references = self._model_references(synthesized, cwd)
            if self.venv_config.require_pinned_model_revision:
                model_evidence, model_error = self._validate_model_references(model_references)
                evidence.extend(model_evidence)
                if model_error:
                    return self._blocked(workspace, model_error)
            existing_python = venv_dir / "bin" / "python"
            if existing_python.exists():
                self._set_build_stage(workspace, "validate_environment", candidate=candidate.name)
                existing_manager = self._environment_type(venv_dir)
                valid, detail = self._validate_environment(
                    existing_python, venv_dir, existing_manager, requested_python
                )
                if not valid:
                    evidence.append(f"discarded incompatible existing environment: {detail}")
                    shutil.rmtree(venv_dir, ignore_errors=True)
            if not (venv_dir / "bin" / "python").exists():
                self._set_build_stage(workspace, "create_environment", candidate=candidate.name)
                python_used, diagnostics = self._create_environment(
                    venv_dir, requested_manager, requested_python
                )
                evidence.extend(diagnostics)
                if python_used is None:
                    return self._blocked(
                        workspace,
                        "isolated environment creation failed: " + " | ".join(diagnostics)[-4000:],
                    )
                evidence.append(f"isolated environment created with {python_used}")
            actual_manager = self._environment_type(venv_dir)
            # Keep the venv launcher path intact. Resolving its symlink executes
            # the base interpreter outside the venv and can write system packages.
            python_bin = (venv_dir / "bin" / "python").absolute()
            pip_bin = (venv_dir / "bin" / "pip").absolute()
            self._set_build_stage(workspace, "validate_environment", candidate=candidate.name)
            isolated, isolation_detail = self._validate_environment(
                python_bin, venv_dir, actual_manager, requested_python
            )
            if not isolated:
                return self._blocked(workspace, f"environment isolation validation failed: {isolation_detail}")
            evidence.append(f"{actual_manager} isolation verified: {isolation_detail}")
            install_env = self._install_env(venv_dir, actual_manager)
            flattened_install = " ".join(
                token for command in synthesized.install_commands for token in command
            ).lower()
            if re.search(r"torch(?:vision)?==[^ ]*\+cu\d+", flattened_install):
                torch_index = "https://download.pytorch.org/whl/torch_stable.html"
                install_env.setdefault("PIP_FIND_LINKS", torch_index)
                evidence.append(f"enabled official PyTorch legacy wheel index: {torch_index}")
            install_fingerprint = self._install_fingerprint(synthesized)
            if self._install_cache_hit(workspace, install_fingerprint):
                evidence.append("venv dependency installation reused verified fingerprint cache")
                print(
                    f"[open-world build] cache-hit repo={candidate.name} stage=install_dependencies",
                    flush=True,
                )
            else:
                for index, raw_command in enumerate(synthesized.install_commands, start=1):
                    self._set_build_stage(
                        workspace,
                        "install_dependencies",
                        candidate=candidate.name,
                        command_index=index,
                        command_count=len(synthesized.install_commands),
                    )
                    command = self._venv_command(raw_command, python_bin, pip_bin)
                    result, retry_evidence = self._run_install(command, cwd, install_env)
                    evidence.extend(retry_evidence)
                    if result.returncode != 0:
                        output = result.stderr or result.stdout or "dependency command failed"
                        return self._blocked(
                            workspace,
                            self._failure_diagnostic("dependency install", output, requested_python),
                        )
                self._write_install_cache(workspace, install_fingerprint)
                evidence.append("venv dependency installation passed")
            smoke = synthesized.manifest.smoke_test_command
            verified_smoke: list[str] = []
            gpu_smoke_status = "not-requested"
            model_load_smoke_status = "not-requested"
            if smoke:
                self._set_build_stage(workspace, "smoke_test", candidate=candidate.name)
                verified_smoke = self._venv_command(smoke, python_bin, pip_bin)
                command = materialize_smoke_command(
                    verified_smoke,
                    workspace / "smoke_artifacts",
                )
                smoke_env = self._smoke_env(venv_dir, actual_manager)
                if self.venv_config.require_model_load_smoke and requires_local_ml_runtime(
                    synthesized.manifest.spec.capability,
                    synthesized.manifest.spec.output_type,
                ):
                    smoke_env["EVOVIDEO_REQUIRE_MODEL_LOAD_SMOKE"] = "1"
                result = self._run_command(
                    command,
                    cwd=str(cwd),
                    env=smoke_env,
                    timeout=self.venv_config.smoke_timeout_seconds,
                )
                if result.returncode != 0:
                    return self._blocked(
                        workspace,
                        self._failure_diagnostic(
                            "repository smoke test", result.stderr or result.stdout or "smoke failed", requested_python
                        ),
                    )
                evidence.append("venv smoke test passed")
                if smoke_env.get("EVOVIDEO_REQUIRE_MODEL_LOAD_SMOKE") == "1":
                    if not re.search(r"(?m)^EVOVIDEO_MODEL_LOAD_SMOKE_OK\s*$", result.stdout or ""):
                        return self._blocked(
                            workspace,
                            "repository smoke test did not prove that the exact local model pipeline loaded; "
                            "the adapter must print EVOVIDEO_MODEL_LOAD_SMOKE_OK on a line by itself only after "
                            "from_pretrained/checkpoint loading and device initialization succeed",
                        )
                    model_load_smoke_status = "passed"
                    evidence.append("exact local model pipeline load smoke passed")
            if self.venv_config.gpu_smoke_required or bool(synthesized.environment.get("gpu_smoke")):
                self._set_build_stage(workspace, "gpu_smoke_test", candidate=candidate.name)
                gpu_probe = self._gpu_smoke(python_bin, cwd, self._smoke_env(venv_dir, actual_manager))
                if gpu_probe.returncode == 43:
                    if (
                        os.environ.get("OPEN_WORLD_ALLOW_EXTERNAL_API_TOOLS", "0") != "1"
                        and requires_local_ml_runtime(
                            synthesized.manifest.spec.capability,
                            synthesized.manifest.spec.output_type,
                        )
                    ):
                        return self._blocked(
                            workspace,
                            "GPU smoke test found no Torch installation for a local pixel-generative tool; "
                            "the repository adapter is not executing a local model",
                        )
                    evidence.append("GPU smoke skipped because the tool environment does not install torch")
                    gpu_smoke_status = "skipped-no-torch"
                elif gpu_probe.returncode != 0:
                    return self._blocked(
                        workspace,
                        self._failure_diagnostic(
                            "GPU smoke test", gpu_probe.stderr or gpu_probe.stdout or "GPU probe failed", requested_python
                        ),
                    )
                else:
                    evidence.append((gpu_probe.stdout or "GPU tensor smoke test passed").strip()[-1000:])
                    gpu_smoke_status = "passed"
            manifest = synthesized.manifest
            manifest.command = self._venv_command(manifest.command, python_bin, pip_bin)
            manifest.cwd = str(cwd)
            manifest.container_image = None
            manifest.sanitize_env = True
            manifest.env["EVOVIDEO_GPU_SMOKE_STATUS"] = gpu_smoke_status
            manifest.env["EVOVIDEO_MODEL_LOAD_SMOKE_STATUS"] = model_load_smoke_status
            manifest.env["EVOVIDEO_BUILDER_SMOKE_STATUS"] = "passed" if verified_smoke else "not-requested"
            manifest.env["EVOVIDEO_BUILDER_SMOKE_COMMAND_SHA256"] = (
                smoke_command_digest(verified_smoke) if verified_smoke else ""
            )
            manifest.env["EVOVIDEO_HF_MODEL_REFERENCES"] = json.dumps(
                [asdict(reference) for reference in model_references],
                sort_keys=True,
            )
            # Preserve the exact verified probe so restored tools can be
            # revalidated instead of trusting a stale manifest indefinitely.
            manifest.smoke_test_command = verified_smoke
            manifest.spec = ToolSpec(
                **{
                    **manifest.spec.__dict__,
                    "verified": True,
                    "backend": actual_manager,
                    "provenance": f"{manifest.spec.provenance}:approved-{actual_manager}",
                }
            )
            (workspace / "verified_manifest.json").write_text(
                json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self._set_build_stage(workspace, "complete", candidate=candidate.name)
            return SandboxBuildResult("ready", manifest, None, str(workspace), evidence)
        except (OSError, subprocess.TimeoutExpired, OpenWorldToolError, ValueError) as exc:
            self._set_build_stage(
                workspace,
                "blocked",
                candidate=candidate.name,
                error=f"{type(exc).__name__}: {exc}",
            )
            return self._blocked(workspace, str(exc))
        finally:
            self._active_workspace = None
            self._build_started = None
            self._active_stage = "idle"

    def manifest_available(self, manifest: CommandToolManifest) -> bool:
        if manifest.spec.backend not in {"venv", "micromamba"} or not manifest.smoke_test_command:
            return False
        gpu_status = manifest.env.get("EVOVIDEO_GPU_SMOKE_STATUS")
        local_only = os.environ.get("OPEN_WORLD_ALLOW_EXTERNAL_API_TOOLS", "0") != "1"
        if (
            local_only
            and requires_local_ml_runtime(
                manifest.spec.capability,
                manifest.spec.output_type,
            )
            and gpu_status == "skipped-no-torch"
        ):
            return False
        if (
            local_only
            and self.venv_config.gpu_smoke_required
            and requires_local_ml_runtime(
                manifest.spec.capability,
                manifest.spec.output_type,
            )
            and gpu_status != "passed"
        ):
            return False
        if (
            self.venv_config.require_model_load_smoke
            and requires_local_ml_runtime(
                manifest.spec.capability,
                manifest.spec.output_type,
            )
            and manifest.env.get("EVOVIDEO_MODEL_LOAD_SMOKE_STATUS") != "passed"
        ):
            return False
        return _manifest_paths_available(manifest)

    @staticmethod
    def _model_references(
        synthesized: SynthesizedTool,
        cwd: Path,
    ) -> list[HuggingFaceModelReference]:
        paths: dict[Path, str] = {}
        for relative in synthesized.adapter_files:
            path = (cwd / relative).resolve()
            if path.is_relative_to(cwd.resolve()) and path.suffix == ".py" and path.is_file():
                paths[path] = relative
        for command in (synthesized.manifest.command, synthesized.manifest.smoke_test_command):
            for token in command:
                if not str(token).endswith(".py"):
                    continue
                path = (cwd / str(token)).resolve()
                if path.is_relative_to(cwd.resolve()) and path.is_file():
                    paths[path] = str(token)

        references: dict[tuple[str, str | None], HuggingFaceModelReference] = {}
        for path, source_name in paths.items():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            bindings: dict[str, str] = {}
            for node in tree.body:
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                value = node.value
                if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        bindings[target.id] = value.value

            def string_value(node: ast.AST | None) -> str | None:
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    return node.value
                if isinstance(node, ast.Name):
                    return bindings.get(node.id)
                return None

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                function_name = (
                    node.func.attr if isinstance(node.func, ast.Attribute)
                    else node.func.id if isinstance(node.func, ast.Name)
                    else ""
                )
                if function_name != "from_pretrained":
                    continue
                model_node = node.args[0] if node.args else None
                revision_node: ast.AST | None = None
                for keyword in node.keywords:
                    if keyword.arg in {"pretrained_model_name_or_path", "model_id"} and model_node is None:
                        model_node = keyword.value
                    elif keyword.arg == "revision":
                        revision_node = keyword.value
                model_id = string_value(model_node)
                revision = string_value(revision_node)
                if not model_id or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model_id):
                    continue
                reference = HuggingFaceModelReference(model_id, revision, source_name)
                references[(model_id, revision)] = reference
        return list(references.values())

    def _validate_model_references(
        self,
        references: list[HuggingFaceModelReference],
    ) -> tuple[list[str], str | None]:
        evidence: list[str] = []
        for reference in references:
            if not reference.revision:
                return evidence, (
                    f"Hugging Face model {reference.model_id!r} in {reference.source} is not pinned to an "
                    "immutable revision; pass revision=<commit sha> to from_pretrained"
                )
            if not re.fullmatch(r"[0-9a-fA-F]{7,64}", reference.revision):
                return evidence, (
                    f"Hugging Face model {reference.model_id!r} in {reference.source} uses mutable or invalid "
                    f"revision {reference.revision!r}; use the model repository commit SHA, not a branch, tag, "
                    "or the GitHub tool repository revision"
                )
            valid, detail = self._huggingface_revision_status(reference)
            if not valid:
                return evidence, (
                    f"Hugging Face model revision preflight failed for {reference.model_id}@"
                    f"{reference.revision} in {reference.source}: {detail}"
                )
            evidence.append(
                f"verified Hugging Face model revision: {reference.model_id}@{reference.revision}"
            )
        return evidence, None

    def _huggingface_revision_status(
        self,
        reference: HuggingFaceModelReference,
    ) -> tuple[bool, str]:
        model = urllib.parse.quote(reference.model_id, safe="/")
        revision = urllib.parse.quote(str(reference.revision), safe="")
        request = urllib.request.Request(
            f"https://huggingface.co/api/models/{model}/revision/{revision}",
            headers={"User-Agent": "evovideo-skill/1.0"},
        )
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        context = (
            ssl.create_default_context(cafile=self.venv_config.git_ca_info)
            if self.venv_config.git_ca_info
            else ssl.create_default_context()
        )
        try:
            with urllib.request.urlopen(request, timeout=30, context=context) as response:
                if 200 <= int(response.status) < 300:
                    return True, f"HTTP {response.status}"
                return False, f"HTTP {response.status}"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False, "revision does not exist (HTTP 404)"
            return False, f"Hugging Face returned HTTP {exc.code}"
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            return False, f"could not verify revision over TLS: {exc}"

    def _approved(self, candidate: ExternalToolCandidate) -> bool:
        return self._approval_status(candidate) == "approved"

    def _approval_status(self, candidate: ExternalToolCandidate) -> str:
        if not self.venv_config.require_approval:
            return "approved"
        if any(
            candidate.name == item or candidate.name.startswith(item.rstrip("/") + "/")
            for item in self.venv_config.allowed_repositories
        ):
            return "approved"
        path = Path(self.venv_config.approval_file).expanduser() if self.venv_config.approval_file else None
        if path is None:
            return "pending"
        return ToolApprovalStore(path).status(candidate)

    @staticmethod
    def _interactive_approval_enabled() -> bool:
        return os.environ.get("OPEN_WORLD_VENV_INTERACTIVE_APPROVAL", "0") == "1" and sys.stdin.isatty()

    def _prompt_for_approval(self, synthesized: SynthesizedTool, workspace: Path) -> str:
        candidate = synthesized.candidate
        manifest = synthesized.manifest
        self.output_fn("\nOpen-world tool approval required")
        self.output_fn(f"  candidate_id: {candidate.candidate_id}")
        self.output_fn(f"  repository: {candidate.name}")
        self.output_fn(f"  source: {candidate.url}")
        self.output_fn(f"  revision: {candidate.revision}")
        self.output_fn(f"  license: {candidate.license or 'unknown'}")
        self.output_fn(f"  capability: {manifest.spec.capability}")
        self.output_fn(f"  install: {synthesized.install_commands or ['<none>']}")
        self.output_fn(f"  command: {manifest.command}")
        self.output_fn(f"  risks: {synthesized.risks or ['venv does not isolate host filesystem or network']}")
        self.output_fn(f"  plan: {workspace / 'tool_plan.json'}")
        while True:
            answer = self.input_fn("Approve this repository for local venv execution? [y]es/[n]o/[s]kip: ").strip().lower()
            if answer in {"s", "skip", "", "q", "quit"}:
                return "pending"
            if answer in {"y", "yes"}:
                self._save_interactive_decision(candidate.candidate_id, True)
                return "approved"
            if answer in {"n", "no"}:
                self._save_interactive_decision(candidate.candidate_id, False)
                return "rejected"
            self.output_fn("Please enter y, n, or s.")

    def _save_interactive_decision(self, candidate_id: str, approved: bool) -> None:
        if not self.venv_config.approval_file:
            raise OpenWorldToolError("interactive approval requires OPEN_WORLD_VENV_APPROVAL_FILE")
        ToolApprovalStore(self.venv_config.approval_file).set_decision(candidate_id, approved)

    @staticmethod
    def _venv_command(raw: list[str], python_bin: Path, pip_bin: Path) -> list[str]:
        if not raw:
            raise OpenWorldToolError("venv command is empty")
        command = [str(item) for item in raw]
        executable = Path(command[0]).name.lower()
        if executable in {"python", "python3"}:
            command[0] = str(python_bin)
        elif executable in {"pip", "pip3"}:
            command[0] = str(pip_bin)
        else:
            raise OpenWorldToolError(
                f"venv backend only permits python/pip commands, received {command[0]!r}"
            )
        forbidden_install_locations = ("--user", "--prefix", "--target", "--root")
        for token in command[1:]:
            lowered = token.lower()
            if any(lowered == flag or lowered.startswith(flag + "=") for flag in forbidden_install_locations):
                raise OpenWorldToolError(
                    f"venv backend forbids package install location override {token!r}"
                )
        return command

    def _create_environment(
        self,
        venv_dir: Path,
        manager: str,
        requested_python: str | None,
    ) -> tuple[str | None, list[str]]:
        diagnostics: list[str] = []
        if manager in {"venv", "auto"}:
            python_used, venv_diagnostics = self._create_venv(venv_dir, requested_python)
            diagnostics.extend(venv_diagnostics)
            if python_used is not None:
                return python_used, diagnostics
            if manager == "venv":
                return None, diagnostics
            shutil.rmtree(venv_dir, ignore_errors=True)
            diagnostics.append("compatible host Python unavailable; falling back to micromamba")
        python_used, mamba_diagnostics = self._create_micromamba(venv_dir, requested_python)
        diagnostics.extend(mamba_diagnostics)
        return python_used, diagnostics

    def _create_venv(
        self,
        venv_dir: Path,
        requested_python: str | None = None,
    ) -> tuple[str | None, list[str]]:
        candidates = list(
            dict.fromkeys(
                item
                for item in (
                    shutil.which(f"python{requested_python}") if requested_python else None,
                    self.venv_config.python_bin,
                    sys.executable,
                    shutil.which("python3"),
                    shutil.which("python"),
                )
                if item
            )
        )
        diagnostics: list[str] = []
        for python_bin in candidates:
            try:
                probe = self._run_command(
                    [python_bin, "-c", "import ensurepip, venv"],
                    timeout=30,
                )
            except OSError as exc:
                diagnostics.append(f"{python_bin}: unavailable ({exc})")
                continue
            if probe.returncode != 0:
                detail = (probe.stderr or probe.stdout or "ensurepip/venv unavailable").strip().splitlines()[-1]
                diagnostics.append(f"{python_bin}: ensurepip preflight failed ({detail})")
                continue
            if requested_python:
                version = self._python_version(Path(python_bin))
                if version != requested_python:
                    diagnostics.append(
                        f"{python_bin}: Python {version or 'unknown'} does not satisfy requested {requested_python}"
                    )
                    continue
            created = self._run_command(
                [python_bin, "-m", "venv", str(venv_dir)],
                timeout=self.venv_config.build_timeout_seconds,
            )
            if created.returncode == 0 and (venv_dir / "bin" / "python").exists():
                return str(python_bin), diagnostics
            detail = (created.stderr or created.stdout or "venv did not create bin/python").strip()
            diagnostics.append(f"{python_bin}: creation failed ({detail[-1000:]})")
        diagnostics.append(
            "no Python with ensurepip+venv is available; install python3-venv or set "
            "OPEN_WORLD_VENV_PYTHON to a compatible interpreter; auto mode can use micromamba"
        )
        return None, diagnostics

    def _create_micromamba(
        self,
        venv_dir: Path,
        requested_python: str | None,
    ) -> tuple[str | None, list[str]]:
        micromamba = shutil.which(self.venv_config.micromamba_bin) or (
            self.venv_config.micromamba_bin
            if Path(self.venv_config.micromamba_bin).expanduser().is_file()
            else None
        )
        if not micromamba:
            return None, [
                "micromamba is unavailable; install it or set OPEN_WORLD_MICROMAMBA_BIN. "
                "Unlike venv, micromamba can create the Python version required by legacy repositories."
            ]
        python_version = requested_python or "3.11"
        env = self._install_env(venv_dir, "micromamba")
        env.setdefault("MAMBA_ROOT_PREFIX", str(Path(self.venv_config.root_dir) / ".micromamba"))
        ca_info = self.venv_config.git_ca_info
        if ca_info:
            venv_dir.parent.mkdir(parents=True, exist_ok=True)
            env["MAMBA_SSL_VERIFY"] = ca_info
            env["CONDA_SSL_VERIFY"] = ca_info
            condarc = venv_dir.parent / ".condarc.evovideo"
            condarc.write_text(
                "channels:\n  - conda-forge\nssl_verify: " + json.dumps(ca_info) + "\n",
                encoding="utf-8",
            )
            env["CONDARC"] = str(condarc)
        command = [
            str(micromamba), "create", "--yes", "--prefix", str(venv_dir),
            f"python={python_version}", "pip",
        ]
        created = self._run_command(
            command,
            env=env,
            timeout=self.venv_config.build_timeout_seconds,
        )
        if created.returncode == 0 and (venv_dir / "bin" / "python").exists():
            return f"micromamba:{python_version}", []
        output = created.stderr or created.stdout or "micromamba did not create bin/python"
        return None, [f"micromamba creation failed: {output[-2000:]}"]

    def _validate_environment(
        self,
        python_bin: Path,
        venv_dir: Path,
        manager: str = "venv",
        requested_python: str | None = None,
    ) -> tuple[bool, str]:
        probe_code = (
            "import json,site,sys;"
            "print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,"
            "'sites':site.getsitepackages()}))"
        )
        try:
            result = self._run_command(
                [str(python_bin), "-c", probe_code],
                timeout=30,
            )
        except OSError as exc:
            return False, f"probe failed: {exc}"
        if result.returncode != 0:
            return False, (result.stderr or result.stdout or "probe returned non-zero")[-1000:]
        try:
            payload = json.loads((result.stdout or "").strip().splitlines()[-1])
            prefix = Path(str(payload["prefix"])).resolve()
            base_prefix = Path(str(payload["base_prefix"])).resolve()
            sites = [Path(str(item)).resolve() for item in payload.get("sites", [])]
            root = venv_dir.resolve()
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return False, f"invalid isolation probe output: {exc}: {(result.stdout or '')[-500:]}"
        if manager == "venv" and prefix == base_prefix:
            return False, f"sys.prefix equals sys.base_prefix ({prefix})"
        if not prefix.is_relative_to(root):
            return False, f"sys.prefix {prefix} is outside {root}"
        if not sites or any(not site_path.is_relative_to(root) for site_path in sites):
            return False, f"site-packages escape venv: {[str(item) for item in sites]}"
        version = self._python_version(python_bin)
        if requested_python and version != requested_python:
            return False, f"Python {version or 'unknown'} does not satisfy requested {requested_python}"
        return True, f"prefix={prefix}, python={version or 'unknown'}, manager={manager}"

    def _validate_venv(self, python_bin: Path, venv_dir: Path) -> tuple[bool, str]:
        return self._validate_environment(python_bin, venv_dir, "venv")

    def _python_version(self, python_bin: Path) -> str | None:
        try:
            result = self._run_command(
                [str(python_bin), "-c", "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                timeout=30,
            )
        except OSError:
            return None
        if result.returncode != 0:
            return None
        value = (result.stdout or "").strip().splitlines()
        return value[-1] if value else None

    @staticmethod
    def _environment_type(venv_dir: Path) -> str:
        return "micromamba" if (venv_dir / "conda-meta").is_dir() else "venv"

    def _requested_environment_manager(self, synthesized: SynthesizedTool) -> str:
        configured = str(self.venv_config.environment_manager or "venv").strip().lower()
        proposed = str(synthesized.environment.get("manager") or "auto").strip().lower()
        requested = configured if configured in {"venv", "micromamba"} else proposed
        if requested not in {"auto", "venv", "micromamba"}:
            raise OpenWorldToolError(f"unsupported environment manager {requested!r}")
        return requested

    def _requested_python_version(self, synthesized: SynthesizedTool) -> str | None:
        raw = synthesized.environment.get("python") or self.venv_config.python_version
        if raw:
            match = re.search(r"(3\.\d+)", str(raw))
            if not match:
                raise OpenWorldToolError(f"invalid requested Python version {raw!r}; expected 3.x")
            return match.group(1)
        flattened = " ".join(token for command in synthesized.install_commands for token in command).lower()
        if re.search(r"torch(?:vision)?==(?:1\.1[012]|0\.1[123])", flattened):
            return "3.10"
        return None

    def _run_install(
        self,
        command: list[str],
        cwd: Path,
        env: dict[str, str],
    ) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        result = self._run_command(
            command,
            cwd=str(cwd),
            env=env,
            timeout=self.venv_config.build_timeout_seconds,
        )
        output = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
        proxy_failure = "proxyerror" in output or "tunnel connection failed" in output
        if result.returncode == 0 or not proxy_failure or not self.venv_config.retry_without_proxy:
            return result, []
        direct_env = dict(env)
        for key in (
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
        ):
            direct_env.pop(key, None)
        direct_env["NO_PROXY"] = "*"
        direct_env["no_proxy"] = "*"
        direct_env["PIP_PROXY"] = ""
        retried = self._run_command(
            command,
            cwd=str(cwd),
            env=direct_env,
            timeout=self.venv_config.build_timeout_seconds,
        )
        return retried, ["dependency install proxy failed; retried once without proxy"]

    def _run(self, command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        return self._run_command(command, timeout=timeout)

    def _run_command(
        self,
        command: list[str],
        *,
        timeout: int | float,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        effective_timeout = self._effective_timeout(float(timeout))
        if self.runner is not subprocess.run or self._active_workspace is None:
            return self.runner(
                command,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                check=False,
            )

        self._command_counter += 1
        log_dir = self._active_workspace / "build_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_stage = re.sub(r"[^a-z0-9_-]+", "-", self._active_stage.lower()).strip("-") or "command"
        log_path = log_dir / f"{self._command_counter:02d}-{safe_stage}.log"
        started = time.monotonic()
        last_progress = started
        last_size = 0
        print(
            f"[open-world build] start stage={self._active_stage} "
            f"timeout={effective_timeout:.0f}s log={log_path}",
            flush=True,
        )
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(f"command={json.dumps(command, ensure_ascii=False)}\n")
            handle.write(f"cwd={cwd or os.getcwd()}\n--- output ---\n")
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=os.name != "nt",
            )
            while True:
                elapsed = time.monotonic() - started
                if elapsed >= effective_timeout:
                    self._terminate_process(process)
                    raise subprocess.TimeoutExpired(command, effective_timeout)
                try:
                    returncode = process.wait(
                        timeout=min(float(self.venv_config.heartbeat_seconds), effective_timeout - elapsed)
                    )
                    break
                except subprocess.TimeoutExpired:
                    handle.flush()
                    size = log_path.stat().st_size
                    now = time.monotonic()
                    if size != last_size:
                        last_size = size
                        last_progress = now
                    idle = now - last_progress
                    if self.venv_config.idle_timeout_seconds > 0 and idle >= self.venv_config.idle_timeout_seconds:
                        self._terminate_process(process)
                        raise OpenWorldToolError(
                            f"stage {self._active_stage} produced no output for {idle:.0f}s; "
                            f"process terminated; log={log_path}"
                        )
                    print(
                        f"[open-world build] running stage={self._active_stage} pid={process.pid} "
                        f"elapsed={now - started:.0f}s idle={idle:.0f}s bytes={size} log={log_path}",
                        flush=True,
                    )
        output = log_path.read_text(encoding="utf-8", errors="replace")
        print(
            f"[open-world build] done stage={self._active_stage} code={returncode} "
            f"elapsed={time.monotonic() - started:.1f}s log={log_path}",
            flush=True,
        )
        return subprocess.CompletedProcess(command, returncode, output, "")

    def _effective_timeout(self, requested: float) -> float:
        if self._build_started is None:
            return max(0.1, requested)
        remaining = self.venv_config.total_timeout_seconds - (
            time.monotonic() - self._build_started
        )
        if remaining <= 0:
            raise subprocess.TimeoutExpired(
                ["open-world", self._active_stage],
                self.venv_config.total_timeout_seconds,
            )
        return max(0.1, min(requested, remaining))

    @staticmethod
    def _terminate_process(process: subprocess.Popen[Any]) -> None:
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:  # pragma: no cover
                process.terminate()
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            try:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:  # pragma: no cover
                    process.kill()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def _set_build_stage(self, workspace: Path, stage: str, **details: Any) -> None:
        self._active_stage = stage
        elapsed = (
            time.monotonic() - self._build_started
            if self._build_started is not None
            else 0.0
        )
        payload = {
            "status": stage,
            "elapsed_seconds": elapsed,
            "total_timeout_seconds": self.venv_config.total_timeout_seconds,
            **details,
        }
        temporary = workspace / "build_status.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(workspace / "build_status.json")
        print(
            f"[open-world build] stage={stage} elapsed={elapsed:.1f}s "
            f"workspace={workspace}",
            flush=True,
        )

    @staticmethod
    def _install_fingerprint(synthesized: SynthesizedTool) -> str:
        payload = {
            "revision": synthesized.candidate.revision,
            "source_subdir": synthesized.source_subdir,
            "install_commands": synthesized.install_commands,
            "base_image": synthesized.base_image,
            "environment": synthesized.environment,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _install_cache_hit(workspace: Path, fingerprint: str) -> bool:
        path = workspace / "install_state.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return payload.get("status") == "complete" and payload.get("fingerprint") == fingerprint

    @staticmethod
    def _write_install_cache(workspace: Path, fingerprint: str) -> None:
        path = workspace / "install_state.json"
        temporary = workspace / "install_state.json.tmp"
        temporary.write_text(
            json.dumps(
                {"status": "complete", "fingerprint": fingerprint, "completed_at": time.time()},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _install_env(self, venv_dir: Path, manager: str = "venv") -> dict[str, str]:
        allowed = {
            "HOME", "USER", "LANG", "LC_ALL", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
            "http_proxy", "https_proxy", "all_proxy", "no_proxy", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL",
            "PIP_TRUSTED_HOST", "PIP_CERT", "GIT_SSL_CAINFO", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
            "CURL_CA_BUNDLE", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN",
            "MAMBA_SSL_VERIFY", "CONDA_SSL_VERIFY", "CONDARC",
        }
        env = {key: value for key, value in os.environ.items() if key in allowed}
        if self.venv_config.git_ca_info:
            env["GIT_SSL_CAINFO"] = self.venv_config.git_ca_info
            env.setdefault("SSL_CERT_FILE", self.venv_config.git_ca_info)
            env.setdefault("REQUESTS_CA_BUNDLE", self.venv_config.git_ca_info)
            env.setdefault("PIP_CERT", self.venv_config.git_ca_info)
            env.setdefault("CURL_CA_BUNDLE", self.venv_config.git_ca_info)
        env["PATH"] = f"{venv_dir / 'bin'}:/usr/bin:/bin"
        env["VIRTUAL_ENV"] = str(venv_dir)
        if manager == "micromamba":
            env["CONDA_PREFIX"] = str(venv_dir)
        env["PYTHONNOUSERSITE"] = "1"
        env["PIP_CONFIG_FILE"] = os.devnull
        if manager == "venv":
            env["PIP_REQUIRE_VIRTUALENV"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        env["PIP_NO_INPUT"] = "1"
        env.setdefault("PIP_DEFAULT_TIMEOUT", os.environ.get("OPEN_WORLD_PIP_TIMEOUT_SECONDS", "60"))
        env.setdefault("PIP_RETRIES", os.environ.get("OPEN_WORLD_PIP_RETRIES", "2"))
        cache_root = Path(
            os.environ.get(
                "OPEN_WORLD_SHARED_CACHE_DIR",
                str(Path(self.venv_config.root_dir) / ".shared_cache"),
            )
        ).expanduser()
        (cache_root / "pip").mkdir(parents=True, exist_ok=True)
        (cache_root / "huggingface").mkdir(parents=True, exist_ok=True)
        (cache_root / "torch").mkdir(parents=True, exist_ok=True)
        env.setdefault("PIP_CACHE_DIR", str(cache_root / "pip"))
        env.setdefault("HF_HOME", str(cache_root / "huggingface"))
        env.setdefault("TORCH_HOME", str(cache_root / "torch"))
        env.setdefault("TORCH_EXTENSIONS_DIR", str(cache_root / "torch_extensions" / venv_dir.parent.name))
        compute_capability = self._host_compute_capability()
        if compute_capability:
            env.setdefault("TORCH_CUDA_ARCH_LIST", compute_capability)
        env["GIT_TERMINAL_PROMPT"] = "0"
        return env

    def _smoke_env(self, venv_dir: Path, manager: str = "venv") -> dict[str, str]:
        env = self._runtime_env(venv_dir, manager)
        if os.environ.get("OPEN_WORLD_SMOKE_OFFLINE", "1") == "1":
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
        return env

    def _runtime_env(self, venv_dir: Path, manager: str = "venv") -> dict[str, str]:
        allowed = {
            "HOME", "USER", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES", "GIT_SSL_CAINFO", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
            "CURL_CA_BUNDLE", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
            "http_proxy", "https_proxy", "all_proxy", "no_proxy", "HF_TOKEN",
            "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_HOME", "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME",
        }
        env = {key: value for key, value in os.environ.items() if key in allowed}
        if self.venv_config.git_ca_info:
            env["GIT_SSL_CAINFO"] = self.venv_config.git_ca_info
            env.setdefault("SSL_CERT_FILE", self.venv_config.git_ca_info)
            env.setdefault("REQUESTS_CA_BUNDLE", self.venv_config.git_ca_info)
            env.setdefault("CURL_CA_BUNDLE", self.venv_config.git_ca_info)
        env["PATH"] = f"{venv_dir / 'bin'}:/usr/bin:/bin"
        env["VIRTUAL_ENV"] = str(venv_dir)
        if manager == "micromamba":
            env["CONDA_PREFIX"] = str(venv_dir)
        env["PYTHONNOUSERSITE"] = "1"
        cache_root = Path(
            os.environ.get(
                "OPEN_WORLD_SHARED_CACHE_DIR",
                str(Path(self.venv_config.root_dir) / ".shared_cache"),
            )
        ).expanduser()
        (cache_root / "huggingface").mkdir(parents=True, exist_ok=True)
        (cache_root / "torch").mkdir(parents=True, exist_ok=True)
        env.setdefault("HF_HOME", str(cache_root / "huggingface"))
        env.setdefault("TORCH_HOME", str(cache_root / "torch"))
        return env

    def _gpu_smoke(
        self,
        python_bin: Path,
        cwd: Path,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        probe = (
            "import importlib.util,sys;"
            "sys.exit(43) if importlib.util.find_spec('torch') is None else None;"
            "import torch;"
            "print('torch='+str(torch.__version__)+' cuda='+str(torch.version.cuda));"
            "sys.exit(42) if not torch.cuda.is_available() else None;"
            "d=torch.device('cuda');a=torch.randn((32,32),device=d);"
            "b=torch.randn((32,32),device=d);c=a@b;torch.cuda.synchronize();"
            "print('gpu_smoke_pass device='+torch.cuda.get_device_name(0)+' capability='+str(torch.cuda.get_device_capability(0)))"
        )
        return self._run_command(
            [str(python_bin), "-c", probe],
            cwd=str(cwd),
            env=env,
            timeout=min(120, self.venv_config.smoke_timeout_seconds),
        )

    @staticmethod
    def _host_compute_capability() -> str | None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        values = [item.strip() for item in result.stdout.splitlines() if re.fullmatch(r"\d+\.\d+", item.strip())]
        return ";".join(dict.fromkeys(values)) or None

    @staticmethod
    def _failure_diagnostic(stage: str, output: str, requested_python: str | None) -> str:
        lowered = output.lower()
        hints: list[str] = []
        if "no matching distribution found" in lowered or "could not find a version" in lowered:
            hints.append(
                "resolve the package's supported Python/CUDA matrix; declare environment.python and use manager=micromamba"
            )
        if "no module named" in lowered:
            hints.append("add the missing package to install_commands with a compatible pinned version")
        if "cannot import name" in lowered or "undefined symbol" in lowered:
            hints.append("pin a mutually compatible dependency set from repository metadata instead of mixing latest packages")
        if any(marker in lowered for marker in ("no kernel image", "invalid device function", "was built for sm")):
            hints.append(
                "rebuild the custom CUDA extension inside this environment with TORCH_CUDA_ARCH_LIST matching the host GPU"
            )
        if "glibcxx" in lowered or "cxx11_abi" in lowered:
            hints.append("align compiler, PyTorch CUDA build, and native-extension C++ ABI")
        suffix = f"; requested_python={requested_python}" if requested_python else ""
        advice = "; repair_hints=" + " | ".join(hints) if hints else ""
        return f"{stage} failed{suffix}{advice}; output={output[-5000:]}"

    @staticmethod
    def _blocked(workspace: Path, reason: str) -> SandboxBuildResult:
        return SandboxBuildResult("blocked", None, None, str(workspace), [reason])


class OpenWorldToolAcquirer:
    """Discover, synthesize, secure, sandbox, and return one executable manifest."""

    def __init__(
        self,
        discoverer: InternetToolDiscoverer,
        synthesizer: OpenAICompatibleToolSynthesizer,
        security: ToolSecurityPolicy,
        builder: Any,
        report_dir: str | Path,
    ):
        self.discoverer = discoverer
        self.synthesizer = synthesizer
        self.security = security
        self.builder = builder
        self.report_dir = Path(report_dir)
        self.acquisition_report_path = self.report_dir / "open_world_tool_acquisition.json"
        self.events: list[dict[str, Any]] = self._load_events()
        self.cache_rejections: list[str] = []
        self.registration_path = self.report_dir / "registered_open_world_tools.json"

    def acquire(self, request: CapabilityRequest) -> tuple[CommandToolManifest | None, list[str]]:
        evidence: list[str] = []
        pending = False
        try:
            candidates = self.discoverer.search(request)
        except (OpenWorldToolError, TimeoutError, OSError) as exc:
            evidence = [f"internet discovery failed: {exc}"]
            self._record(request, [], None, "blocked", evidence)
            return None, evidence
        discovery_rejections = list(getattr(self.discoverer, "last_rejections", []))
        evidence.extend(f"discovery rejected: {item}" for item in discovery_rejections[:20])
        if not candidates:
            evidence.append("internet discovery returned no executable domain-matched candidates")
            self._record(request, [], None, "missing", evidence)
            return None, evidence
        for candidate in candidates:
            deployment_feedback: list[str] = []
            repository_files: list[str] = []
            repository_context = ""
            for deployment_attempt in range(2):
                try:
                    if deployment_attempt == 0:
                        synthesized = self.synthesizer.synthesize(request, candidate)
                    else:
                        synthesized = self.synthesizer.synthesize(
                            request,
                            candidate,
                            deployment_feedback=deployment_feedback,
                            repository_files=repository_files,
                            repository_context=repository_context,
                        )
                except (OpenWorldToolError, TypeError) as exc:
                    label = "repair synthesis" if deployment_attempt else "synthesis"
                    evidence.append(f"{candidate.name}: {label} failed: {exc}")
                    break
                errors = self.security.validate(synthesized)
                if errors:
                    evidence.extend(f"{candidate.name}: security: {error}" for error in errors)
                    break
                built = self.builder.build(synthesized)
                evidence.extend(f"{candidate.name}: {item}" for item in built.evidence)
                if built.status == "ready" and built.manifest is not None:
                    self._record(request, candidates, built, "ready", evidence)
                    return built.manifest, evidence
                pending = pending or built.status == "pending"
                if built.status != "blocked" or deployment_attempt > 0:
                    break
                deployment_feedback = list(built.evidence)
                repository_files = self._repository_files(built.workspace)
                repository_context = self._repository_context(built.workspace, repository_files)
                evidence.append(
                    f"{candidate.name}: retrying adapter synthesis with deployment failure and "
                    f"{len(repository_files)} repository paths"
                )
        self._record(request, candidates, None, "pending" if pending else "blocked", evidence)
        return None, evidence or ["all discovered candidates failed synthesis or sandbox validation"]

    @staticmethod
    def _repository_files(workspace: str, limit: int = 600) -> list[str]:
        source = Path(workspace) / "source"
        if not source.exists():
            return []
        paths: list[str] = []
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            if any(part in {".git", ".venv", "__pycache__"} for part in relative.parts):
                continue
            paths.append(relative.as_posix())
        priority_terms = ("inference", "video", "i2v", "image", "example", "requirements", "pyproject", "setup")
        paths.sort(
            key=lambda value: (
                -sum(term in value.lower() for term in priority_terms),
                value.count("/"),
                len(value),
                value,
            )
        )
        return paths[:limit]

    @staticmethod
    def _repository_context(
        workspace: str,
        ranked_files: list[str],
        max_chars: int = 16000,
    ) -> str:
        source = Path(workspace) / "source"
        allowed_suffixes = {".py", ".toml", ".yaml", ".yml", ".json", ".md"}
        sections: list[str] = []
        used = 0
        for relative in ranked_files:
            path = source / relative
            if path.suffix.lower() not in allowed_suffixes or not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lower = content.lower()
            if path.suffix.lower() == ".py" and not any(
                token in lower
                for token in ("argparse", "click.", "typer.", "def infer", "def generate", "pipeline")
            ):
                continue
            excerpt = content[:5000]
            section = f"\n--- {relative} ---\n{excerpt}"
            if used + len(section) > max_chars:
                remaining = max_chars - used
                if remaining > 200:
                    sections.append(section[:remaining])
                break
            sections.append(section)
            used += len(section)
            if len(sections) >= 6:
                break
        return "".join(sections)

    def record_registration(self, request: CapabilityRequest, manifest: CommandToolManifest) -> None:
        records = self._registration_records()
        records = [item for item in records if item.get("manifest", {}).get("name") != manifest.spec.name]
        records.append({"request": asdict(request), "manifest": manifest.to_dict()})
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.registration_path.write_text(
            json.dumps({"tools": records}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def record_rejection(
        self,
        request: CapabilityRequest,
        manifest: CommandToolManifest,
        evidence: list[str],
    ) -> None:
        """Remove a cached registration that failed the harness-level contract gate."""
        records = [
            item
            for item in self._registration_records()
            if item.get("manifest", {}).get("name") != manifest.spec.name
        ]
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.registration_path.write_text(
            json.dumps({"tools": records}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        event = {
            "request": asdict(request),
            "status": "rejected_after_build",
            "manifest": manifest.to_dict(),
            "evidence": list(evidence),
        }
        self.events.append(event)
        self._write_events()
        rejection_log = self.report_dir / "rejected_open_world_tools.jsonl"
        with rejection_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def cached_manifests(self) -> list[CommandToolManifest]:
        manifests: list[CommandToolManifest] = []
        self.cache_rejections = []
        for record in self._registration_records():
            raw = record.get("manifest")
            if not isinstance(raw, dict):
                continue
            try:
                manifest = CommandToolManifest.from_dict(raw)
            except (TypeError, ValueError):
                continue
            available = False
            if hasattr(self.builder, "manifest_available"):
                available = bool(self.builder.manifest_available(manifest))
            elif manifest.container_image and hasattr(self.builder, "image_available"):
                available = bool(self.builder.image_available(manifest.container_image))
            if available:
                manifests.append(manifest)
            else:
                self.cache_rejections.append(
                    f"cached tool {manifest.spec.name} failed executable, entrypoint, environment, or smoke-probe availability checks"
                )
        return manifests

    def _registration_records(self) -> list[dict[str, Any]]:
        if not self.registration_path.exists():
            return []
        try:
            payload = json.loads(self.registration_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        records = payload.get("tools", []) if isinstance(payload, dict) else []
        return [item for item in records if isinstance(item, dict)]

    def _load_events(self) -> list[dict[str, Any]]:
        if not self.acquisition_report_path.exists():
            return []
        try:
            payload = json.loads(self.acquisition_report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        events = payload.get("events", []) if isinstance(payload, dict) else []
        return [item for item in events if isinstance(item, dict)]

    def _write_events(self) -> None:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.acquisition_report_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({"events": self.events}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.acquisition_report_path)

    def _record(
        self,
        request: CapabilityRequest,
        candidates: list[ExternalToolCandidate],
        build: SandboxBuildResult | None,
        status: str,
        evidence: list[str],
    ) -> None:
        event = {
            "request": asdict(request),
            "status": status,
            "candidates": [candidate.to_dict() for candidate in candidates],
            "build": build.to_dict() if build else None,
            "evidence": evidence,
        }
        self.events.append(event)
        self._write_events()


def acquirer_from_env(
    report_dir: str | Path,
    model: str = "openai/gpt-5.6-sol",
    base_url: str | None = None,
    max_candidates: int = 6,
    docker_bin: str = "docker",
    base_images: tuple[str, ...] | None = None,
    sandbox_backend: str | None = None,
) -> OpenWorldToolAcquirer:
    openai_model = _is_openai_reasoning_model(model)
    resolved_base_url = base_url or os.environ.get("GRAPH_LLM_BASE_URL")
    if not resolved_base_url:
        resolved_base_url = (
            "https://openrouter.ai/api/v1"
            if "/" in model
            else (
                "https://api.openai.com/v1"
                if openai_model
                else "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        )
    resolved_key = os.environ.get("GRAPH_LLM_API_KEY")
    if "openrouter.ai" in resolved_base_url.lower():
        resolved_key = resolved_key or os.environ.get("OPENROUTER_API_KEY")
    elif openai_model:
        resolved_key = resolved_key or os.environ.get("OPENAI_API_KEY")
    else:
        resolved_key = resolved_key or os.environ.get("DASHSCOPE_API_KEY")
    synthesis = ToolSynthesisConfig(
        model=model,
        base_url=resolved_base_url,
        api_key=resolved_key,
        timeout_seconds=int(
            os.environ.get(
                "OPEN_WORLD_SYNTHESIS_TIMEOUT_SECONDS",
                os.environ.get("GRAPH_LLM_TIMEOUT_SECONDS", "120"),
            )
        ),
        max_output_tokens=int(os.environ.get("OPEN_WORLD_SYNTHESIS_MAX_OUTPUT_TOKENS", "6144")),
        repair_attempts=max(0, int(os.environ.get("OPEN_WORLD_SYNTHESIS_REPAIR_ATTEMPTS", "2"))),
        reasoning_effort=(os.environ.get("GRAPH_LLM_REASONING_EFFORT", "high") or None) if openai_model else None,
        allowed_base_images=base_images or ToolSynthesisConfig.allowed_base_images,
    )
    discovery = DiscoveryConfig(
        github_token=os.environ.get("GITHUB_TOKEN"),
        huggingface_token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        max_candidates=max_candidates,
        timeout_seconds=int(os.environ.get("OPEN_WORLD_DISCOVERY_TIMEOUT_SECONDS", "30")),
    )
    discoverer = InternetToolDiscoverer(discovery)
    backend = (sandbox_backend or os.environ.get("OPEN_WORLD_SANDBOX_BACKEND", "docker")).strip().lower()
    candidate_root = str(Path(report_dir) / "candidates")
    git_ca_info = _resolve_git_ca_info()
    if backend == "docker":
        builder: Any = DockerSandboxBuilder(
            SandboxBuildConfig(
                root_dir=candidate_root,
                docker_bin=docker_bin,
                allow_huggingface_model_clone=os.environ.get("OPEN_WORLD_ALLOW_HF_CLONE", "0") == "1",
                git_ca_info=git_ca_info,
            )
        )
    elif backend in {"venv", "micromamba", "auto"}:
        allowed = tuple(
            item.strip()
            for item in os.environ.get("OPEN_WORLD_VENV_ALLOWED_REPOSITORIES", "").split(",")
            if item.strip()
        )
        builder = VenvSandboxBuilder(
            VenvBuildConfig(
                root_dir=candidate_root,
                python_bin=os.environ.get("OPEN_WORLD_VENV_PYTHON", sys.executable),
                build_timeout_seconds=int(os.environ.get("OPEN_WORLD_VENV_BUILD_TIMEOUT_SECONDS", "1800")),
                total_timeout_seconds=int(os.environ.get("OPEN_WORLD_VENV_TOTAL_TIMEOUT_SECONDS", "1800")),
                idle_timeout_seconds=int(os.environ.get("OPEN_WORLD_VENV_IDLE_TIMEOUT_SECONDS", "300")),
                heartbeat_seconds=max(1, int(os.environ.get("OPEN_WORLD_VENV_HEARTBEAT_SECONDS", "30"))),
                smoke_timeout_seconds=int(os.environ.get("OPEN_WORLD_VENV_SMOKE_TIMEOUT_SECONDS", "300")),
                require_approval=os.environ.get("OPEN_WORLD_VENV_AUTO_APPROVE", "0") != "1",
                approval_file=os.environ.get("OPEN_WORLD_VENV_APPROVAL_FILE"),
                allowed_repositories=allowed,
                allow_huggingface_model_clone=os.environ.get("OPEN_WORLD_ALLOW_HF_CLONE", "0") == "1",
                git_ca_info=git_ca_info,
                retry_without_proxy=os.environ.get("OPEN_WORLD_VENV_RETRY_WITHOUT_PROXY", "1") == "1",
                environment_manager=backend,
                micromamba_bin=os.environ.get("OPEN_WORLD_MICROMAMBA_BIN", "micromamba"),
                python_version=os.environ.get("OPEN_WORLD_TOOL_PYTHON_VERSION") or None,
                gpu_smoke_required=os.environ.get("OPEN_WORLD_GPU_SMOKE_REQUIRED", "0") == "1",
                require_pinned_model_revision=os.environ.get(
                    "OPEN_WORLD_REQUIRE_PINNED_MODEL_REVISION", "1"
                ) == "1",
                require_model_load_smoke=os.environ.get(
                    "OPEN_WORLD_REQUIRE_MODEL_LOAD_SMOKE", "1"
                ) == "1",
            )
        )
    elif backend in {"search-only", "search_only"}:
        builder = SearchOnlyToolBuilder(candidate_root)
    else:
        raise OpenWorldToolError(
            f"unsupported open-world sandbox backend {backend!r}; use docker, auto, venv, micromamba, or search-only"
        )
    return OpenWorldToolAcquirer(
        discoverer=discoverer,
        synthesizer=OpenAICompatibleToolSynthesizer(synthesis),
        security=ToolSecurityPolicy(discovery.allowed_licenses, synthesis.allowed_base_images),
        builder=builder,
        report_dir=report_dir,
    )


def _resolve_git_ca_info() -> str | None:
    explicit = os.environ.get("OPEN_WORLD_GIT_CAINFO")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise OpenWorldToolError(f"OPEN_WORLD_GIT_CAINFO does not exist or is not a file: {path}")
        return str(path.resolve())
    git_config_ca = None
    try:
        configured = subprocess.run(
            ["git", "config", "--path", "--get", "http.sslCAInfo"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if configured.returncode == 0 and configured.stdout.strip():
            git_config_ca = configured.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    candidates = [
        os.environ.get("GIT_SSL_CAINFO"),
        os.environ.get("SSL_CERT_FILE"),
        os.environ.get("REQUESTS_CA_BUNDLE"),
        git_config_ca,
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/etc/ssl/cert.pem",
    ]
    try:
        import certifi

        candidates.append(certifi.where())
    except ImportError:
        pass
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().is_file():
            return str(Path(candidate).expanduser().resolve())
    return None


def _is_openai_reasoning_model(model: str) -> bool:
    model_name = model.rsplit("/", 1)[-1].lower()
    return model_name.startswith(("gpt-", "o1", "o3", "o4"))
