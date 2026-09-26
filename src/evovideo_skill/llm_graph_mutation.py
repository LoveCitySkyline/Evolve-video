from __future__ import annotations

import json
import os
import re
import hashlib
import urllib.error
import urllib.request
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Callable

from evovideo_skill.graph_evolver import (
    GraphPathCandidate, GraphPathRollout, h3_registry_active, validate_h3_node_configs,
)
from evovideo_skill.graph_algorithms import approximate_graph_edit_distance, wl_kernel_similarity
from evovideo_skill.h3_graph_contracts import H3_CAPABILITY_ALIASES
from evovideo_skill.graph_skill import BoundedGraphEdit, GraphEdge, GraphNode, ToolPathGraph
from evovideo_skill.tool_onboarding import CapabilityRequest, ToolOnboardingManager
from evovideo_skill.structured_json import StructuredJSONError, parse_json_object
from evovideo_skill.weighted_tool_graph import (
    H3_LOCAL_COMPARISON_PROTOCOL, H3_LOCAL_EVALUATION_PROTOCOL, h3_local_active,
)


class GraphMutationError(RuntimeError):
    pass


@dataclass
class GraphMutationConfig:
    model: str = "openai/gpt-5.6-sol"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str | None = None
    timeout_seconds: int = 120
    temperature: float = 0.2
    max_candidates: int = 3
    max_edits: int = 12
    max_output_tokens: int = 8192
    reasoning_effort: str | None = "high"
    repair_attempts: int = 2
    open_world_invention_enabled: bool = True
    invention_candidates: int = 6
    invention_realization_budget: int = 3
    invention_max_nodes: int = 8
    invention_novelty_threshold: float = 0.82


class OpenAICompatibleGraphMutationProposer:
    """Ask an OpenAI-compatible LLM for bounded, executable DAG mutations."""

    ALLOWED_OPS = {
        "add_node",
        "delete_node",
        "replace_node",
        "add_edge",
        "delete_edge",
        "tighten_trigger",
        "add_validator",
        "add_fallback",
        "change_threshold",
    }
    ALLOWED_VERIFIERS = {
        "identity_consistency",
        "clothing_color_consistency",
        "background_preservation",
        "target_edit_success",
        "prompt_action_alignment",
        "vbench_dimension_alignment",
    }

    def __init__(
        self,
        config: GraphMutationConfig,
        request_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        onboarding_manager: ToolOnboardingManager | None = None,
    ):
        self.config = config
        self.request_fn = request_fn
        self.onboarding_manager = onboarding_manager
        self.last_onboarding_results: list[dict[str, Any]] = []
        self.last_candidate_errors: list[dict[str, Any]] = []
        self.last_invention_audits: list[dict[str, Any]] = []
        self._onboarding_tool_aliases: dict[str, str] = {}

    def propose(
        self,
        rollouts: list[GraphPathRollout],
        available_tools: set[str],
        max_candidates: int | None = None,
        search_context: dict[str, Any] | None = None,
    ) -> list[GraphPathCandidate]:
        self.last_onboarding_results = []
        self.last_candidate_errors = []
        self.last_invention_audits = []
        self._onboarding_tool_aliases = {}
        failed = [rollout for rollout in rollouts if rollout.failure is not None]
        if not failed:
            return []
        limit = min(max_candidates or self.config.max_candidates, self.config.max_candidates)
        payload = self._request_payload(failed, available_tools, limit, search_context or {})
        current_payload = payload
        errors: list[str] = []
        data: dict[str, Any] | None = None
        for attempt in range(self.config.repair_attempts + 1):
            response = self.request_fn(current_payload) if self.request_fn else self._post(current_payload)
            try:
                content = self._response_content(response)
                data = self._parse_json(content)
                break
            except GraphMutationError as error:
                errors.append(str(error))
                if attempt >= self.config.repair_attempts:
                    raise GraphMutationError(
                        "LLM graph mutation JSON repair failed after retries; " + "; ".join(errors)
                    ) from error
                if "finish_reason='length'" in str(error):
                    current_payload = self._length_repair_payload(payload, attempt + 1, errors[-2:])
                else:
                    current_payload = self._repair_payload(payload, errors[-2:])
        if data is None:
            raise GraphMutationError("LLM graph mutation returned no structured response")
        raw_candidates = data.get("candidates", [])
        if not isinstance(raw_candidates, list):
            raise GraphMutationError("LLM mutation response field 'candidates' must be a list")
        raw_explorations = data.get("exploration_candidates", [])
        if not isinstance(raw_explorations, list):
            raise GraphMutationError(
                "LLM mutation response field 'exploration_candidates' must be a list"
            )
        selected_explorations = self._select_abstract_explorations(
            raw_explorations[: self.config.invention_candidates],
            failed,
            search_context or {},
        ) if self.config.open_world_invention_enabled else []
        onboarding_data = dict(data)
        onboarding_data["exploration_candidates"] = selected_explorations
        self._onboard_requested_capabilities(onboarding_data, available_tools)
        self._rewrite_onboarded_tool_names(raw_candidates, selected_explorations)
        if self.onboarding_manager is not None:
            available_tools = self.onboarding_manager.registry.available_names()

        parent_by_id = {rollout.graph.graph_id: rollout for rollout in failed}
        graph_library = {
            str(item.get("graph_id")): ToolPathGraph.from_dict(item)
            for item in (search_context or {}).get("historical_graphs", [])
            if isinstance(item, dict) and item.get("graph_id")
        }
        candidates: list[GraphPathCandidate] = []
        errors: list[str] = []
        for index, raw in enumerate(raw_candidates[:limit]):
            try:
                candidates.append(self._materialize(raw, parent_by_id, graph_library, available_tools, index))
            except (KeyError, TypeError, ValueError, GraphMutationError) as exc:
                errors.append(f"candidate[{index}]: {exc}")
                self.last_candidate_errors.append(
                    {
                        "candidate_index": index,
                        "candidate_name": raw.get("name") if isinstance(raw, dict) else None,
                        "rejection_reason": str(exc),
                    }
                )
        for index, raw in enumerate(selected_explorations):
            try:
                candidate = self._materialize_exploration(
                    raw,
                    parent_by_id,
                    available_tools,
                    index,
                )
                candidates.append(candidate)
                self.last_invention_audits.append({
                    "stage": "capability_graph_realization",
                    "status": "executable",
                    "candidate_name": candidate.graph.skill_name,
                    "mechanism_family": candidate.graph.stats.get("mechanism_family"),
                    "novelty": candidate.graph.stats.get("structural_novelty"),
                    "tools": candidate.graph.tool_names(),
                })
            except (KeyError, TypeError, ValueError, GraphMutationError) as exc:
                errors.append(f"exploration_candidate[{index}]: {exc}")
                self.last_invention_audits.append({
                    "stage": "capability_graph_realization",
                    "status": "rejected",
                    "candidate_name": raw.get("name") if isinstance(raw, dict) else None,
                    "rejection_reason": str(exc),
                })
        if (raw_candidates or selected_explorations) and not candidates:
            onboarding = self._onboarding_failure_summary()
            suffix = f"; onboarding: {onboarding}" if onboarding else ""
            raise GraphMutationError(
                "all LLM graph mutations were invalid: " + "; ".join(errors) + suffix
            )
        return candidates

    def _onboard_requested_capabilities(
        self,
        data: dict[str, Any],
        available_tools: set[str] | None = None,
    ) -> None:
        raw_requests = data.get("capability_requests", [])
        if not isinstance(raw_requests, list):
            raise GraphMutationError("LLM mutation response field 'capability_requests' must be a list")
        explicit = [
            CapabilityRequest.from_dict(item)
            for item in raw_requests
            if isinstance(item, dict)
        ]
        requests = self._complete_capability_requests(
            explicit,
            data.get("candidates", []),
            available_tools or set(),
        )
        requests.extend(
            self._exploration_capability_requests(
                data.get("exploration_candidates", []),
                available_tools or set(),
            )
        )
        requests = self._dedupe_requests(requests)
        native_results = []
        pending_requests = []
        for request in requests:
            resolved = self._realization_request(request, "intermediate", available_tools or set())
            registry = self.onboarding_manager.registry if self.onboarding_manager else None
            native = h3_registry_active(available_tools or set()) and (
                request.capability in H3_CAPABILITY_ALIASES or resolved.capability.startswith("h3_")
            )
            matches = [spec for spec in registry.find_by_capability(resolved.capability)
                       if spec.verified and spec.name in (available_tools or set())
                       and (not resolved.required_input_types or "any" in spec.input_types
                            or set(resolved.required_input_types).issubset(spec.input_types))] if native and registry else []
            if not matches:
                pending_requests.append(request)
                continue
            chosen = sorted(matches, key=lambda spec: spec.name)[0]
            if request.suggested_tool_name:
                self._onboarding_tool_aliases[request.suggested_tool_name] = chosen.name
            native_results.append({"request": asdict(request), "status": "already_registered",
                                   "tool_name": chosen.name, "evidence": ["Reused verified native H3 capability; no repository acquisition"]})
        requests = pending_requests
        if not requests:
            self.last_onboarding_results = native_results
            return
        if self.onboarding_manager is None:
            self.last_onboarding_results = [
                {"status": "disabled", "request": asdict(item)}
                for item in requests
            ]
            return
        grouped: dict[tuple[str, str | None, str, tuple[str, ...]], list[CapabilityRequest]] = {}
        for request in requests:
            grouped.setdefault(self._acquisition_family_key(request), []).append(request)

        capability_budget = max(
            1,
            int(os.environ.get("OPEN_WORLD_MAX_CAPABILITIES_PER_MUTATION", "2")),
        )
        reusable_groups: list[list[CapabilityRequest]] = []
        acquisition_groups: list[list[CapabilityRequest]] = []
        for group in grouped.values():
            request = self._merge_capability_family(group)
            existing = self.onboarding_manager.registry.find_by_capability(request.capability)
            compatible = [
                spec for spec in existing
                if self.onboarding_manager._satisfies_request(spec, request)
            ]
            reusable = bool(compatible) and request.capability in CapabilityRequest.HARNESS_CAPABILITIES
            (reusable_groups if reusable else acquisition_groups).append(group)
        acquisition_groups.sort(
            key=lambda group: self._capability_deployability(self._merge_capability_family(group)),
            reverse=True,
        )
        selected_groups = [*reusable_groups, *acquisition_groups[:capability_budget]]
        deferred_groups = acquisition_groups[capability_budget:]
        merged_requests = [self._merge_capability_family(group) for group in selected_groups]
        print(
            "[tool onboarding] "
            f"requests={len(requests)} capability_families={len(grouped)} "
            f"reused={len(reusable_groups)} selected_for_acquisition={min(len(acquisition_groups), capability_budget)} "
            f"deferred={len(deferred_groups)}"
        )
        family_results = self.onboarding_manager.onboard_requests(merged_requests)
        expanded_results: list[dict[str, Any]] = list(native_results)
        for group, result in zip(selected_groups, family_results):
            for request in group:
                payload = result.to_dict()
                payload["request"] = asdict(request)
                if len(group) > 1:
                    payload["evidence"] = [
                        *payload.get("evidence", []),
                        f"consolidated {len(group)} graph-node requests into one capability arena",
                    ]
                expanded_results.append(payload)
                if request.suggested_tool_name and result.tool_name:
                    self._onboarding_tool_aliases[request.suggested_tool_name] = result.tool_name
        for group in deferred_groups:
            for request in group:
                expanded_results.append({
                    "request": asdict(request),
                    "status": "deferred",
                    "tool_name": None,
                    "evidence": [
                        "per-mutation capability acquisition budget exhausted; request is eligible in a later evolution round"
                    ],
                })
        self.last_onboarding_results = expanded_results

    def _capability_deployability(self, request: CapabilityRequest) -> tuple[int, int, int]:
        acquisition, _ = request.acquisition_request()
        repository_native = acquisition.capability in CapabilityRequest.REPOSITORY_CAPABILITIES
        physical_inputs = set(acquisition.required_input_types)
        standard_contract = physical_inputs.issubset({"image", "video"})
        named_catalog_match = any(
            item.get("capability") == acquisition.capability
            for item in self.onboarding_manager.discoverable_tools()
        ) if self.onboarding_manager is not None else False
        return (
            1 if named_catalog_match else 0,
            1 if repository_native else 0,
            1 if standard_contract else 0,
        )

    @staticmethod
    def _dedupe_requests(requests: list[CapabilityRequest]) -> list[CapabilityRequest]:
        deduped: list[CapabilityRequest] = []
        seen: set[tuple[str, str | None]] = set()
        for request in requests:
            key = (request.capability, request.suggested_tool_name)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(request)
        return deduped

    @staticmethod
    def _acquisition_family_key(
        request: CapabilityRequest,
    ) -> tuple[str, str | None, str, tuple[str, ...]]:
        acquisition, _ = request.acquisition_request()
        return (
            request.capability,
            acquisition.preferred_backend,
            acquisition.capability,
            tuple(sorted(acquisition.required_input_types)),
        )

    @staticmethod
    def _merge_capability_family(requests: list[CapabilityRequest]) -> CapabilityRequest:
        representative = requests[0]
        required = list(dict.fromkeys(
            artifact_type
            for request in requests
            for artifact_type in request.required_input_types
        ))
        reasons = list(dict.fromkeys(request.reason for request in requests if request.reason))
        return CapabilityRequest(
            capability=representative.capability,
            preferred_backend=representative.preferred_backend,
            suggested_tool_name=representative.suggested_tool_name,
            reason=" | ".join(reasons),
            required_input_types=required,
        )

    def _rewrite_onboarded_tool_names(
        self,
        raw_candidates: list[Any],
        exploration_candidates: list[Any],
    ) -> None:
        """Resolve graph-local invented names to the shared capability arena tool."""
        if not self._onboarding_tool_aliases:
            return
        for candidate in raw_candidates:
            if not isinstance(candidate, dict):
                continue
            for edit in candidate.get("edits", []):
                if not isinstance(edit, dict) or not isinstance(edit.get("payload"), dict):
                    continue
                name = str(edit["payload"].get("name") or "")
                if name in self._onboarding_tool_aliases:
                    edit["payload"]["name"] = self._onboarding_tool_aliases[name]
        for candidate in exploration_candidates:
            if not isinstance(candidate, dict):
                continue
            for node in candidate.get("nodes", []):
                if not isinstance(node, dict):
                    continue
                name = str(node.get("suggested_tool_name") or "")
                if name in self._onboarding_tool_aliases:
                    node["suggested_tool_name"] = self._onboarding_tool_aliases[name]

    def _realization_request(
        self, request: CapabilityRequest, declared_output: str, available_tools: set[str],
    ) -> CapabilityRequest:
        registry = self.onboarding_manager.registry if self.onboarding_manager else None
        aliases = H3_CAPABILITY_ALIASES if h3_registry_active(available_tools) else {}
        native = aliases.get(request.capability, request.capability)
        if registry is not None and (
            native in CapabilityRequest.HARNESS_CAPABILITIES or native.startswith("h3_")
            or request.capability in aliases
        ) and registry.find_by_capability(native):
            return CapabilityRequest(
                native, request.preferred_backend, request.suggested_tool_name,
                request.reason, list(request.required_input_types),
            )
        if declared_output not in {"video", "any", "intermediate"}:
            return request
        return request.acquisition_request()[0]

    def _exploration_capability_requests(
        self,
        raw_candidates: Any,
        available_tools: set[str],
    ) -> list[CapabilityRequest]:
        if not isinstance(raw_candidates, list):
            return []
        registry = self.onboarding_manager.registry if self.onboarding_manager is not None else None
        requests: list[CapabilityRequest] = []
        for candidate_index, candidate in enumerate(raw_candidates):
            if not isinstance(candidate, dict):
                continue
            candidate_name = self._safe_name(
                str(candidate.get("name") or f"invention_{candidate_index + 1}")
            )
            for node_index, node in enumerate(candidate.get("nodes", [])):
                if not isinstance(node, dict):
                    continue
                capability = self._safe_name(str(node.get("capability") or ""))
                node_id = self._safe_name(str(node.get("node_id") or f"node_{node_index + 1}"))
                policy = str(node.get("realization_policy") or "auto").lower()
                tool_name = str(node.get("tool_name") or "").strip()
                if tool_name in available_tools:
                    continue
                required = [str(item) for item in node.get("input_types", []) if str(item).lower() != "any"]
                normalized_request = CapabilityRequest(
                    capability=capability,
                    required_input_types=required,
                )
                capability = normalized_request.capability
                required = list(normalized_request.required_input_types)
                node["capability"] = capability
                node["input_types"] = required
                declared_output = str(node.get("output_type") or "intermediate")
                acquisition_request = self._realization_request(normalized_request, declared_output, available_tools)
                realization_inputs = list(acquisition_request.required_input_types)
                existing = (
                    registry.find_by_capability(acquisition_request.capability)
                    if registry is not None else []
                )
                if capability in CapabilityRequest.HARNESS_CAPABILITIES:
                    # A bridge performs one concrete conversion. Contract repair
                    # can compose several bridges for a compound abstract node.
                    compatible = [
                        item for item in existing
                        if not required
                        or bool(set(required).intersection(item.input_types))
                        or "any" in item.input_types
                    ]
                else:
                    compatible = [
                        item for item in existing
                        if not realization_inputs
                        or set(realization_inputs).issubset(set(item.input_types))
                        or "any" in item.input_types
                    ]
                compatible = [item for item in compatible if item.verified and item.name in available_tools
                              and (declared_output in {"any", "intermediate"} or item.output_type == declared_output)]
                if compatible and (
                    capability in CapabilityRequest.HARNESS_CAPABILITIES
                    or acquisition_request.capability.startswith("h3_")
                    or (h3_registry_active(available_tools) and capability in H3_CAPABILITY_ALIASES)
                    or policy not in {"search_new", "compare"}
                ):
                    continue
                suggested = str(node.get("suggested_tool_name") or "").strip()
                if not suggested:
                    digest = hashlib.sha256(
                        f"{candidate_name}:{node_id}:{capability}".encode()
                    ).hexdigest()[:6]
                    suggested = self._safe_name(f"ow_{capability}_{digest}")
                    node["suggested_tool_name"] = suggested
                requests.append(CapabilityRequest(
                    capability=capability,
                    suggested_tool_name=suggested,
                    required_input_types=required,
                    reason=(
                        f"Open-world capability graph {candidate_name} requires {capability}. "
                        f"Mechanism: {candidate.get('hypothesis') or candidate.get('mechanism_family') or 'novel path'}"
                    ),
                ))
        return requests

    def _complete_capability_requests(
        self,
        explicit: list[CapabilityRequest],
        raw_candidates: Any,
        available_tools: set[str],
    ) -> list[CapabilityRequest]:
        """Ensure every invented candidate tool has one onboarding request."""
        requests = [self._normalize_explicit_request(request) for request in explicit]
        covered = {
            request.suggested_tool_name
            for request in requests
            if request.suggested_tool_name
        }
        for tool_name in self._candidate_tool_names(raw_candidates):
            if tool_name in available_tools or tool_name in covered:
                continue
            requests.append(self._infer_capability_request(tool_name))
            covered.add(tool_name)
        deduped: list[CapabilityRequest] = []
        seen: set[tuple[str, str | None]] = set()
        for request in requests:
            key = (request.capability, request.suggested_tool_name)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(request)
        return deduped

    @classmethod
    def _normalize_explicit_request(
        cls,
        request: CapabilityRequest,
    ) -> CapabilityRequest:
        """Repair LLM-authored compound capabilities using the graph tool name."""
        if (not request.suggested_tool_name or request.capability in H3_CAPABILITY_ALIASES
                or request.capability.startswith("h3_")):
            return request
        inferred = cls._infer_capability_request(request.suggested_tool_name)
        canonical = {
            "image_conditioned_video_generation",
            "keyframe_conditioned_video_generation",
            "multi_shot_identity_conditioned_generation",
            "motion_conditioned_video_generation",
            "audio_conditioned_video_generation",
            "global_video_editing",
            "region_video_editing",
            "video_style_transfer",
            "temporal_deflickering",
            "failed_segment_repair",
        }
        explicit_physical = set(request.required_input_types).difference(
            CapabilityRequest.PROMPT_COMPILED_INPUT_TYPES
        )
        image_named = inferred.capability == "image_conditioned_video_generation"
        overconstrained_image_request = image_named and bool(
            explicit_physical.difference(CapabilityRequest.IMAGE_SEMANTIC_INPUT_TYPES)
        )
        if request.capability in canonical and not overconstrained_image_request:
            return request
        return CapabilityRequest(
            capability=inferred.capability,
            preferred_backend=request.preferred_backend,
            suggested_tool_name=request.suggested_tool_name,
            reason=(
                f"{request.reason} "
                "The harness normalized this compound LLM request to a repository-native primitive."
            ).strip(),
            required_input_types=inferred.required_input_types,
        )

    @staticmethod
    def _candidate_tool_names(raw_candidates: Any) -> list[str]:
        names: list[str] = []
        if not isinstance(raw_candidates, list):
            return names
        for candidate in raw_candidates:
            if not isinstance(candidate, dict):
                continue
            for edit in candidate.get("edits", []):
                if not isinstance(edit, dict) or edit.get("op") not in {"add_node", "replace_node"}:
                    continue
                payload = edit.get("payload") or {}
                if not isinstance(payload, dict) or payload.get("node_type") != "tool":
                    continue
                name = str(payload.get("name") or "").strip()
                if name and name not in names:
                    names.append(name)
        return names

    @staticmethod
    def _infer_capability_request(tool_name: str) -> CapabilityRequest:
        normalized = tool_name.lower()
        required: list[str] = []
        # Identity/keyframe/character-conditioned tools are deployed as I2V
        # primitives even when the LLM calls them an "editor". The graph can
        # deterministically extract/materialize a reference image from a draft
        # video; requiring one repository to natively accept video + identity +
        # every planning artifact makes acquisition needlessly impossible.
        if "audio" in normalized and "video" in normalized:
            capability = "audio_conditioned_video_generation"
            required.append("audio")
        elif any(token in normalized for token in ("i2v", "image", "keyframe", "identity", "character")):
            capability = "image_conditioned_video_generation"
        elif "video" in normalized and any(
            token in normalized for token in ("editor", "editing", "repair", "refiner")
        ):
            capability = "failed_segment_repair" if "repair" in normalized else "global_video_editing"
            required.append("video")
        elif any(token in normalized for token in ("motion", "trajectory", "pose", "structure")):
            capability = "motion_conditioned_video_generation"
        elif "style" in normalized:
            capability = "video_style_transfer"
            required.append("video")
        elif "deflicker" in normalized or "flicker" in normalized:
            capability = "temporal_deflickering"
            required.append("video")
        else:
            capability = normalized.removeprefix("local_")
        if "identity" in normalized:
            required.append("identity_reference")
        if "keyframe" in normalized:
            required.append("keyframes")
        if "character" in normalized:
            required.append("character_sheet")
        if "temporal" in normalized:
            required.append("temporal_plan")
        if "tracked" in normalized or "region" in normalized:
            required.append("tracked_regions")
        # Generic words such as "motion editor" describe the desired outcome,
        # not proof that a structure map node feeds the tool. Only explicit
        # structure/pose/trajectory names request a native motion artifact.
        if any(token in normalized for token in ("structure", "pose", "trajectory")):
            required.append("structure_motion_map")
        if (
            capability == "image_conditioned_video_generation"
            and not set(required).intersection(CapabilityRequest.IMAGE_SEMANTIC_INPUT_TYPES)
        ):
            required.append("image")
        return CapabilityRequest(
            capability=capability,
            suggested_tool_name=tool_name,
            required_input_types=list(dict.fromkeys(required)),
            reason=(
                "Automatically inferred from a missing tool node referenced by an LLM candidate; "
                "the LLM omitted a matching capability_request."
            ),
        )

    def _onboarding_failure_summary(self) -> str:
        summaries: list[str] = []
        for result in self.last_onboarding_results:
            if result.get("status") in {"registered", "already_registered"}:
                continue
            request = result.get("request") or {}
            name = request.get("suggested_tool_name") or request.get("capability") or "unknown"
            evidence = result.get("evidence") or []
            generic = (
                "no configured MCP fallback",
                "escalating to configured cloud MCP fallback",
                "produced no executable tool",
            )
            informative = [
                str(item) for item in evidence
                if not any(marker.lower() in str(item).lower() for marker in generic)
            ]
            detail = informative[-1] if informative else (str(evidence[-1]) if evidence else "no evidence")
            summaries.append(f"{name}={result.get('status', 'unknown')} ({detail[:240]})")
        return "; ".join(summaries)

    def _select_abstract_explorations(
        self,
        raw_candidates: list[Any],
        failed: list[GraphPathRollout],
        search_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Screen mechanism hypotheses before any repository installation is attempted."""
        references = [self._capability_view(rollout.graph) for rollout in failed]
        references.extend(
            self._capability_view(ToolPathGraph.from_dict(item))
            for item in search_context.get("historical_graphs", [])
            if isinstance(item, dict) and item.get("nodes")
        )
        screened: list[tuple[float, str, dict[str, Any], ToolPathGraph]] = []
        for index, raw in enumerate(raw_candidates):
            try:
                if not isinstance(raw, dict):
                    raise GraphMutationError("abstract exploration candidate must be an object")
                graph = self._abstract_capability_graph(raw, index)
                manager = self.onboarding_manager
                if (manager is not None and h3_registry_active(manager.registry.available_names())
                        and manager.mcp_acquirer is None and manager.open_world_acquirer is None):
                    # In a fixed native registry, infeasible ideas must not consume
                    # the slots that could realize a valid reference-based path.
                    self._materialize_exploration(
                        raw, {r.graph.graph_id: r for r in failed}, manager.registry.available_names(), index,
                    )
                similarities = [wl_kernel_similarity(graph, reference) for reference in references]
                max_similarity = max(similarities, default=0.0)
                novelty = max(0.0, 1.0 - max_similarity)
                edit_distance = min(
                    (approximate_graph_edit_distance(graph, reference) for reference in references),
                    default=float(len(graph.nodes) + len(graph.edges)),
                )
                family = self._safe_name(
                    str(raw.get("mechanism_family") or raw.get("name") or f"mechanism_{index + 1}")
                )
                raw["_invention_screening"] = {
                    "structural_novelty": novelty,
                    "max_wl_similarity_to_memory": max_similarity,
                    "approximate_edit_distance_to_nearest_memory": edit_distance,
                    "mechanism_family": family,
                }
                screened.append((novelty, family, raw, graph))
                self.last_invention_audits.append({
                    "stage": "capability_graph_screening",
                    "status": "screened",
                    "candidate_name": raw.get("name"),
                    **raw["_invention_screening"],
                })
            except (KeyError, TypeError, ValueError, GraphMutationError) as exc:
                self.last_invention_audits.append({
                    "stage": "capability_graph_screening",
                    "status": "rejected",
                    "candidate_name": raw.get("name") if isinstance(raw, dict) else None,
                    "rejection_reason": str(exc),
                })
        screened.sort(key=lambda item: (item[0], len(item[3].nodes)), reverse=True)
        selected: list[dict[str, Any]] = []
        selected_graphs: list[ToolPathGraph] = []
        families: set[str] = set()
        for novelty, family, raw, graph in screened:
            if len(selected) >= self.config.invention_realization_budget:
                self.last_invention_audits.append({
                    "stage": "capability_graph_screening",
                    "status": "deferred",
                    "candidate_name": raw.get("name"),
                    "reason": "realization budget exhausted",
                })
                continue
            peer_similarity = max(
                (wl_kernel_similarity(graph, other) for other in selected_graphs),
                default=0.0,
            )
            if family in families or peer_similarity >= self.config.invention_novelty_threshold:
                self.last_invention_audits.append({
                    "stage": "capability_graph_screening",
                    "status": "rejected",
                    "candidate_name": raw.get("name"),
                    "rejection_reason": (
                        "duplicate mechanism family" if family in families
                        else f"WL similarity {peer_similarity:.3f} exceeds diversity threshold"
                    ),
                })
                continue
            selected.append(raw)
            selected_graphs.append(graph)
            families.add(family)
            self.last_invention_audits.append({
                "stage": "capability_graph_screening",
                "status": "selected_for_realization",
                "candidate_name": raw.get("name"),
                "mechanism_family": family,
                "structural_novelty": novelty,
            })
        return selected

    def _abstract_capability_graph(self, raw: dict[str, Any], index: int) -> ToolPathGraph:
        raw_nodes = raw.get("nodes", [])
        raw_edges = raw.get("edges", [])
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise GraphMutationError("abstract capability graph requires non-empty nodes")
        if len(raw_nodes) > self.config.invention_max_nodes:
            raise GraphMutationError(
                f"abstract graph exceeds invention_max_nodes={self.config.invention_max_nodes}"
            )
        if not isinstance(raw_edges, list):
            raise GraphMutationError("abstract capability graph edges must be a list")
        nodes: list[GraphNode] = []
        for node_index, item in enumerate(raw_nodes):
            if not isinstance(item, dict):
                raise GraphMutationError("abstract capability node must be an object")
            node_id = self._safe_name(str(item.get("node_id") or f"capability_{node_index + 1}"))
            required = [str(value) for value in item.get("input_types", []) if str(value).lower() != "any"]
            request = CapabilityRequest(
                capability=self._safe_name(str(item.get("capability") or "")),
                required_input_types=required,
            )
            capability = request.capability
            item["capability"] = capability
            item["input_types"] = list(request.required_input_types)
            nodes.append(GraphNode(node_id, "tool", capability, {
                "input_types": list(request.required_input_types),
                "output_type": str(item.get("output_type") or "intermediate"),
                "realization_policy": str(item.get("realization_policy") or "auto"),
            }))
        edges: list[GraphEdge] = []
        for edge_index, item in enumerate(raw_edges):
            if not isinstance(item, dict):
                continue
            raw_condition = str(item.get("condition") or "always")
            condition = self._canonical_abstract_condition(raw_condition)
            config = dict(item.get("config") or {})
            if condition != raw_condition.strip().lower():
                config["original_semantic_condition"] = raw_condition
                item["condition"] = condition
                item["config"] = config
            edges.append(GraphEdge(
                str(item.get("edge_id") or f"abstract_edge_{edge_index + 1}"),
                str(item.get("source") or ""),
                str(item.get("target") or ""),
                condition,
                config,
            ))
        graph = ToolPathGraph(
            graph_id=self._safe_name(str(raw.get("name") or f"abstract_invention_{index + 1}")),
            skill_name=self._safe_name(str(raw.get("name") or f"abstract_invention_{index + 1}")),
            description=str(raw.get("hypothesis") or raw.get("description") or "Open-world capability hypothesis"),
            triggers=[str(item) for item in raw.get("triggers", [])],
            nodes=nodes,
            edges=edges,
            validators=[str(item) for item in raw.get("validators", [])],
        )
        self._validate_abstract_graph(graph)
        return graph

    @classmethod
    def _canonical_abstract_condition(cls, value: str) -> str:
        condition = (value or "always").strip().lower()
        if cls._supported_edge_condition(condition):
            return condition
        # Abstract capability graphs do not materialize verifier nodes. Natural
        # language failure guards therefore describe the hypothesis, while the
        # executable DAG edge remains an unconditional data dependency.
        if (
            " failure" in condition
            or "_failure" in condition
            or " or " in condition
            or " and " in condition
            or any(metric in condition for metric in cls.ALLOWED_VERIFIERS)
        ):
            return "always"
        # Bare labels such as ``motion_mismatch`` and ``camera_control`` are
        # routing annotations, not predicates understood by the executor. Task
        # routing has already happened before this capability DAG is executed.
        semantic_guard = bool(re.fullmatch(r"[a-z0-9_]+", condition))
        return "always" if semantic_guard else condition

    def _validate_abstract_graph(self, graph: ToolPathGraph) -> None:
        node_ids = [node.node_id for node in graph.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise GraphMutationError("abstract graph contains duplicate node ids")
        known = set(node_ids)
        outgoing = {node_id: [] for node_id in known}
        indegree = {node_id: 0 for node_id in known}
        for edge in graph.edges:
            if edge.source not in known or edge.target not in known:
                raise GraphMutationError(
                    f"abstract graph has dangling edge {edge.source}->{edge.target}"
                )
            if not self._supported_edge_condition(edge.condition):
                raise GraphMutationError(f"unsupported abstract edge condition {edge.condition!r}")
            outgoing[edge.source].append(edge.target)
            indegree[edge.target] += 1
        ready = [node_id for node_id, degree in indegree.items() if degree == 0]
        visited = 0
        while ready:
            source = ready.pop()
            visited += 1
            for target in outgoing[source]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if visited != len(known):
            raise GraphMutationError("abstract capability graph contains a cycle")
        terminal = [node for node in graph.nodes if not outgoing[node.node_id]]
        if not any(node.config.get("output_type") == "video" for node in terminal):
            raise GraphMutationError("abstract graph must have a terminal capability with output_type=video")

    def _capability_view(self, graph: ToolPathGraph) -> ToolPathGraph:
        view = ToolPathGraph.from_dict(graph.to_dict())
        registry = self.onboarding_manager.registry if self.onboarding_manager is not None else None
        for node in view.nodes:
            if node.node_type != "tool" or registry is None or not registry.has(node.name):
                continue
            node.name = str(registry.spec(node.name).capability)
        return view

    def _materialize_exploration(
        self,
        raw: dict[str, Any],
        parent_by_id: dict[str, GraphPathRollout],
        available_tools: set[str],
        index: int,
    ) -> GraphPathCandidate:
        parent_id = str(raw.get("parent_graph_id") or "")
        if parent_id not in parent_by_id:
            if len(parent_by_id) == 1:
                parent_id = next(iter(parent_by_id))
            else:
                raise GraphMutationError(f"unknown exploration parent_graph_id {parent_id!r}")
        abstract = self._abstract_capability_graph(raw, index)
        registry = self.onboarding_manager.registry if self.onboarding_manager is not None else None
        if registry is None:
            raise GraphMutationError("capability graph realization requires a tool registry")
        realized_nodes: list[GraphNode] = []
        edits: list[BoundedGraphEdit] = []
        raw_by_id = {
            self._safe_name(str(item.get("node_id") or f"capability_{node_index + 1}")): item
            for node_index, item in enumerate(raw.get("nodes", []))
            if isinstance(item, dict)
        }
        realization: list[dict[str, Any]] = []
        for abstract_node in abstract.nodes:
            item = raw_by_id[abstract_node.node_id]
            capability = abstract_node.name
            required = [str(value) for value in item.get("input_types", [])]
            request = CapabilityRequest(
                capability=capability,
                required_input_types=required,
            )
            declared_output = str(item.get("output_type") or "intermediate")
            realization_request = self._realization_request(request, declared_output, available_tools)
            realization_inputs = list(realization_request.required_input_types)
            policy = str(item.get("realization_policy") or "auto").lower()
            exact_names = [
                str(item.get("tool_name") or "").strip(),
                str(item.get("suggested_tool_name") or "").strip(),
            ]
            exact = next(
                (name for name in exact_names if name and name in available_tools and registry.has(name)),
                None,
            )
            capability_matches = registry.find_by_capability(realization_request.capability)
            if capability in CapabilityRequest.HARNESS_CAPABILITIES:
                matches = [
                    spec for spec in capability_matches
                    if not required
                    or bool(set(required).intersection(spec.input_types))
                    or "any" in spec.input_types
                ]
            else:
                matches = [
                    spec for spec in capability_matches
                    if not realization_inputs
                    or set(realization_inputs).issubset(set(spec.input_types))
                    or "any" in spec.input_types
                ]
            concrete_output = declared_output not in {"any", "intermediate"}
            if exact is not None:
                exact_spec = registry.spec(exact)
                if (not exact_spec.verified
                        or (concrete_output and str(exact_spec.output_type) != declared_output)
                        or (required and "any" not in exact_spec.input_types
                            and not set(required).issubset(set(exact_spec.input_types)))):
                    exact = None
            matches = [spec for spec in matches if spec.verified and spec.name in available_tools]
            if concrete_output:
                matches = [spec for spec in matches if str(spec.output_type) == declared_output]
            if exact is not None:
                spec = registry.spec(exact)
            elif matches:
                matches.sort(
                    key=lambda spec: (
                        len(set(required).intersection(spec.input_types)),
                        len(required) > 1 and "compose" in spec.name,
                        policy in {"search_new", "compare"} and "tool-arena" in str(spec.provenance),
                        spec.backend != "builtin",
                        -float(spec.estimated_cost),
                    ),
                    reverse=True,
                )
                spec = matches[0]
            else:
                raise GraphMutationError(
                    f"capability {capability!r} has no verified realization for inputs {required} "
                    f"and output {declared_output!r}; do not replace preparation with generation"
                )
            if declared_output not in {"any", "intermediate", str(spec.output_type)}:
                raise GraphMutationError(
                    f"realized tool {spec.name!r} outputs {spec.output_type}, not declared {declared_output}"
                )
            config = {
                "cost": float(spec.estimated_cost),
                "backend": spec.backend,
                "provenance": spec.provenance,
                "abstract_capability": capability,
                "realization_policy": policy,
            }
            if h3_registry_active(available_tools) and (spec.name.startswith("h3_") or spec.name == "mock_text_to_video"):
                native_config = item.get("config", {})
                if not isinstance(native_config, dict):
                    raise GraphMutationError("H3 exploration node config must be an object")
                config = dict(native_config)
                if capability == "terminal_frame_extraction":
                    if config.get("position", "last") != "last":
                        raise GraphMutationError("terminal_frame_extraction requires position=last")
                    config.setdefault("position", "last")
                elif capability == "first_frame_extraction":
                    if config.get("position", "first") != "first":
                        raise GraphMutationError("first_frame_extraction requires position=first")
                    config.setdefault("position", "first")
            realized_nodes.append(GraphNode(abstract_node.node_id, "tool", spec.name, config))
            realization.append({
                "node_id": abstract_node.node_id,
                "capability": capability,
                "tool": spec.name,
                "backend": spec.backend,
                "provenance": spec.provenance,
            })
            edits.append(BoundedGraphEdit(
                "add_node",
                payload={
                    "node_id": abstract_node.node_id,
                    "node_type": "tool",
                    "name": spec.name,
                    "config": config,
                },
                reason=f"realize abstract capability {capability} with verified tool {spec.name}",
            ))
        name = self._safe_name(str(raw.get("name") or f"open_world_invention_{index + 1}"))
        graph = ToolPathGraph(
            graph_id=name,
            skill_name=name,
            description=str(raw.get("hypothesis") or raw.get("description") or "Open-world invented tool path"),
            triggers=list(dict.fromkeys([
                *[str(item) for item in raw.get("triggers", [])],
                *[item.value for item in parent_by_id[parent_id].failure.failure_types],
            ])),
            nodes=realized_nodes,
            edges=[GraphEdge(**edge.__dict__) for edge in abstract.edges],
            validators=[
                str(item) for item in raw.get("validators", [])
                if str(item) in self.ALLOWED_VERIFIERS
            ],
        )
        edits.extend(BoundedGraphEdit(
            "add_edge",
            payload=edge.__dict__,
            reason="connect realized open-world capability nodes",
        ) for edge in graph.edges)
        screening = dict(raw.get("_invention_screening") or {})
        graph.stats.update({
            "proposal_source": "open_world_invention",
            "parent_graph_id": parent_id,
            "mechanism_family": screening.get("mechanism_family") or raw.get("mechanism_family"),
            "structural_novelty": float(screening.get("structural_novelty", 0.0)),
            "novelty_rationale": str(raw.get("novelty_rationale") or ""),
            "causal_hypothesis": str(raw.get("hypothesis") or ""),
            "abstract_capability_graph": {
                key: value for key, value in raw.items() if not str(key).startswith("_")
            },
            "capability_realization": realization,
        })
        self._validate_graph(graph, available_tools, local=h3_local_active(registry))
        outgoing = {node.node_id: 0 for node in graph.nodes}
        for edge in graph.edges:
            outgoing[edge.source] += 1
        terminal_tools = [node for node in graph.nodes if outgoing[node.node_id] == 0]
        if not any(registry.spec(node.name).output_type == "video" for node in terminal_tools):
            raise GraphMutationError("realized exploration graph has no terminal video tool")
        parent_task = parent_by_id[parent_id].task
        task_metadata = parent_task.metadata or {}
        task_family = str(
            task_metadata.get("task_family")
            or task_metadata.get("task_class")
            or task_metadata.get("category")
            or ""
        ).lower()
        if task_family == "audio_video_sync":
            audio_video_tools = [
                node.name
                for node in realized_nodes
                if registry.spec(node.name).output_type == "video"
                and ("audio" in set(registry.spec(node.name).input_types)
                     or (h3_registry_active(available_tools) and node.name == "h3_ref2va"
                         and "h3_reference_set" in set(registry.spec(node.name).input_types)))
            ]
            if not audio_video_tools:
                raise GraphMutationError(
                    "audio_video_sync exploration must include a video-producing tool that "
                    "declares and consumes a physical audio artifact; plain T2V/I2V realization is invalid"
                )
        failure_types = sorted(
            item.value for item in parent_by_id[parent_id].failure.failure_types
        )
        return GraphPathCandidate(
            graph,
            edits,
            str(raw.get("hypothesis") or "Test an open-world capability-graph hypothesis."),
            failure_types,
        )

    def _materialize(
        self,
        raw: dict[str, Any],
        parent_by_id: dict[str, GraphPathRollout],
        graph_library: dict[str, ToolPathGraph],
        available_tools: set[str],
        index: int,
    ) -> GraphPathCandidate:
        if not isinstance(raw, dict):
            raise GraphMutationError("candidate must be an object")
        parent_id = str(raw.get("parent_graph_id", ""))
        repaired_parent_id: str | None = None
        if parent_id not in parent_by_id and parent_id in graph_library and parent_by_id:
            repaired_parent_id = parent_id
            parent_id = next(iter(parent_by_id))
        if parent_id not in parent_by_id:
            raise GraphMutationError(f"unknown parent_graph_id {parent_id!r}")
        raw_edits = raw.get("edits", [])
        merge_graph_ids = [str(item) for item in raw.get("merge_graph_ids", [])]
        if repaired_parent_id and repaired_parent_id not in merge_graph_ids:
            merge_graph_ids.insert(0, repaired_parent_id)
        if not isinstance(raw_edits, list) or (not raw_edits and not merge_graph_ids):
            raise GraphMutationError("candidate must contain at least one edit or merge_graph_ids")
        if len(raw_edits) > self.config.max_edits:
            raise GraphMutationError(f"candidate exceeds max_edits={self.config.max_edits}")

        parent_graph = parent_by_id[parent_id].graph
        if merge_graph_ids:
            if h3_registry_active(available_tools):
                raise GraphMutationError(
                    "H3 composition requires explicit node/edge edits; merge_graph_ids loses per-call configs and ordered bindings"
                )
            donors = []
            for graph_id in merge_graph_ids:
                donor = graph_library.get(graph_id)
                if donor is None:
                    raise GraphMutationError(f"unknown historical graph {graph_id!r}")
                donors.append(donor)
            from evovideo_skill.graph_composition import GraphCompositionError, merge_tool_paths

            try:
                cost_by_tool = {
                    node.name: float(node.config.get("cost", 0.25))
                    for source in [parent_graph, *donors]
                    for node in source.nodes
                    if node.node_type == "tool"
                }
                graph = merge_tool_paths(
                    [parent_graph.tool_names(), *[donor.tool_names() for donor in donors]],
                    name=str(raw.get("name") or f"llm_merge_{index + 1}"),
                    triggers=[*parent_graph.triggers, *[trigger for donor in donors for trigger in donor.triggers]],
                    cost_by_tool=cost_by_tool,
                )
            except GraphCompositionError as exc:
                raise GraphMutationError(str(exc)) from exc
            graph.stats["merged_graph_ids"] = merge_graph_ids
        else:
            graph = ToolPathGraph.from_dict(parent_graph.to_dict())
        edits: list[BoundedGraphEdit] = []
        node_aliases: dict[str, str] = {}
        automatic_repairs: list[str] = []
        parsed_edits: list[BoundedGraphEdit] = []
        for edit_index, raw_edit in enumerate(raw_edits):
            parsed = self._parse_edit(raw_edit, available_tools, edit_index)
            if parsed is None:
                automatic_repairs.append(
                    f"dropped incomplete no-op edit at index {edit_index} instead of rejecting the candidate"
                )
                continue
            parsed_edits.append(parsed)
        # LLMs commonly emit edges before their nodes. Materialize structural edits
        # first so JSON ordering does not change the meaning of a proposed DAG.
        ordered_edits = [edit for edit in parsed_edits if edit.op != "add_edge"] + [
            edit for edit in parsed_edits if edit.op == "add_edge"
        ]
        if ordered_edits != parsed_edits:
            automatic_repairs.append("deferred add_edge edits until all node edits were materialized")
        for edit_index, edit in enumerate(ordered_edits):
            edit, injected = self._repair_edit_ids(
                edit, graph, node_aliases, edit_index, automatic_repairs, available_tools
            )
            for injected_edit in injected:
                graph = graph.apply_edit(injected_edit)
                edits.append(injected_edit)
            if edit is None:
                continue
            graph = graph.apply_edit(edit)
            edits.append(edit)
            if edit.op == "replace_node" and edit.target:
                replacement_id = str(edit.payload.get("node_id") or edit.target)
                if replacement_id != edit.target:
                    node_aliases[edit.target] = replacement_id
                    automatic_repairs.append(
                        f"redirected references from replaced node {edit.target} to {replacement_id}"
                    )
        name = self._safe_name(str(raw.get("name") or f"llm_mutation_{index + 1}"))
        graph.graph_id = name
        graph.skill_name = name
        graph.description = str(raw.get("description") or raw.get("reason") or "LLM-proposed graph mutation")
        for trigger in raw.get("triggers", []):
            trigger = str(trigger).strip()
            if trigger and trigger not in graph.triggers:
                graph.triggers.append(trigger)
        graph.stats.update({"proposal_source": "llm", "parent_graph_id": parent_id})
        if repaired_parent_id:
            automatic_repairs.append(
                f"converted historical parent {repaired_parent_id} into merge_graph_ids"
            )
        if automatic_repairs:
            graph.stats["automatic_repairs"] = automatic_repairs
        self._normalize_semantic_edge_conditions(graph, automatic_repairs)
        self._dedupe_physical_input_edges(graph, automatic_repairs, available_tools)
        if automatic_repairs:
            graph.stats["automatic_repairs"] = automatic_repairs
        self._apply_registered_tool_metadata(graph)
        self._validate_graph(graph, available_tools,
                             local=h3_local_active(getattr(self.onboarding_manager, "registry", None)))
        failure_types = sorted(
            {
                item.value
                for item in parent_by_id[parent_id].failure.failure_types
            }
        )
        return GraphPathCandidate(
            graph=graph,
            edits=edits,
            reason=str(raw.get("reason") or "LLM proposed a bounded tool-path mutation."),
            parent_failure_types=failure_types,
        )

    def _repair_edit_ids(
        self,
        edit: BoundedGraphEdit,
        graph: ToolPathGraph,
        aliases: dict[str, str],
        edit_index: int,
        repairs: list[str],
        available_tools: set[str],
    ) -> tuple[BoundedGraphEdit | None, list[BoundedGraphEdit]]:
        payload = dict(edit.payload)
        if edit.op == "add_edge":
            injected: list[BoundedGraphEdit] = []
            for endpoint in ("source", "target"):
                original = str(payload.get(endpoint, ""))
                resolved = aliases.get(original, original)
                node_ids = {node.node_id for node in graph.nodes}
                tool_name = None
                if resolved not in node_ids:
                    endpoint_names = self._endpoint_name_candidates(resolved)
                    name_matches = [node.node_id for node in graph.nodes if node.name in endpoint_names]
                    if not name_matches:
                        name_matches = self._semantic_node_matches(resolved, graph)
                    if len(name_matches) == 1:
                        resolved = name_matches[0]
                        repairs.append(f"resolved edge endpoint {original} by registered node name")
                    else:
                        tool_name = next((name for name in endpoint_names if name in available_tools), None)
                    if resolved not in node_ids and not name_matches and tool_name is not None:
                        node_id = self._unique_node_id(self._safe_name(tool_name), node_ids)
                        injected_edit = BoundedGraphEdit(
                            op="add_node",
                            payload={"node_id": node_id, "node_type": "tool", "name": tool_name, "config": {}},
                            reason="Automatically materialized a registered tool referenced by an LLM edge.",
                        )
                        injected.append(injected_edit)
                        graph.nodes.append(GraphNode(**injected_edit.payload))
                        aliases[original] = node_id
                        resolved = node_id
                        repairs.append(f"materialized registered tool node {original} for a dangling edge endpoint")
                payload[endpoint] = resolved
            # The caller applies injected edits. Undo temporary nodes used only for
            # resolving both endpoints in this edge.
            injected_ids = {item.payload["node_id"] for item in injected}
            graph.nodes = [node for node in graph.nodes if node.node_id not in injected_ids]
            edit.payload = payload
            return edit, injected
        if edit.op not in {"add_node", "replace_node"}:
            return edit, []
        requested_id = str(payload.get("node_id", ""))
        if not requested_id:
            requested_id = f"llm_node_{edit_index:02d}"
            payload["node_id"] = requested_id
        existing_ids = {node.node_id for node in graph.nodes}
        collision = requested_id in existing_ids and not (
            edit.op == "replace_node" and requested_id == edit.target
        )
        if not collision:
            edit.payload = payload
            return edit, []
        existing = graph.node(requested_id)
        if edit.op == "add_node" and existing.node_type == payload.get("node_type") and existing.name == payload.get("name"):
            aliases[requested_id] = requested_id
            repairs.append(f"reused existing node {requested_id} instead of adding a duplicate")
            return None, []
        suffix = 1
        candidate_id = f"{requested_id}__llm_{suffix}"
        while candidate_id in existing_ids:
            suffix += 1
            candidate_id = f"{requested_id}__llm_{suffix}"
        payload["node_id"] = candidate_id
        aliases[requested_id] = candidate_id
        repairs.append(f"renamed colliding node {requested_id} to {candidate_id}")
        edit.payload = payload
        return edit, []

    @staticmethod
    def _unique_node_id(base: str, existing: set[str]) -> str:
        candidate = base or "llm_tool"
        suffix = 1
        while candidate in existing:
            candidate = f"{base or 'llm_tool'}__llm_{suffix}"
            suffix += 1
        return candidate

    @staticmethod
    def _supported_edge_condition(condition: str) -> bool:
        normalized = (condition or "always").strip().lower()
        return normalized in {"always", "identity_reference_ready", "motif_sequence"} or bool(
            re.fullmatch(r"[a-z0-9_]+\s*(?:<=|>=|<|>)\s*(?:threshold|[0-9.]+)", normalized)
        )

    @classmethod
    def _normalize_semantic_edge_conditions(
        cls,
        graph: ToolPathGraph,
        repairs: list[str],
    ) -> None:
        """Turn non-executable data-availability prose into DAG dependencies."""
        for edge in graph.edges:
            condition = (edge.condition or "always").strip().lower()
            if cls._supported_edge_condition(condition):
                continue
            semantic_guard = bool(re.fullmatch(r"[a-z0-9_]+", condition))
            if not semantic_guard:
                continue
            edge.config = {**edge.config, "original_semantic_condition": condition}
            edge.condition = "always"
            repairs.append(
                f"normalized semantic edge guard {condition!r} to an executable data-dependency edge"
            )

    @staticmethod
    def _endpoint_name_candidates(value: str) -> list[str]:
        candidates = [value]
        stripped = value
        for prefix in ("tool_", "node_"):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
                candidates.append(stripped)
        for suffix in ("_node", "_tool", "_repair", "_refined", "_updated", "_new", "_v2"):
            if stripped.endswith(suffix):
                candidates.append(stripped[: -len(suffix)])
        return list(dict.fromkeys(item for item in candidates if item))

    @classmethod
    def _semantic_node_matches(cls, value: str, graph: ToolPathGraph) -> list[str]:
        normalized = cls._canonical_node_ref(value)
        if normalized in {"trigger", "generation", "trigger_generation"}:
            triggers = [node.node_id for node in graph.nodes if node.node_type == "trigger"]
            return triggers if len(triggers) == 1 else []
        matches = [
            node.node_id
            for node in graph.nodes
            if cls._canonical_node_ref(node.node_id) == normalized
            or cls._canonical_node_ref(node.name) == normalized
        ]
        return list(dict.fromkeys(matches))

    @staticmethod
    def _canonical_node_ref(value: str) -> str:
        normalized = re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")
        changed = True
        while changed:
            changed = False
            for prefix in ("tool_", "node_"):
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
                    changed = True
            for suffix in (
                "_node", "_tool", "_generator", "_repair", "_refined", "_updated", "_new", "_v2"
            ):
                if normalized.endswith(suffix):
                    normalized = normalized[: -len(suffix)]
                    changed = True
        return normalized

    def _apply_registered_tool_metadata(self, graph: ToolPathGraph) -> None:
        if self.onboarding_manager is None:
            return
        registry = self.onboarding_manager.registry
        for node in graph.nodes:
            if node.node_type != "tool" or not registry.has(node.name):
                continue
            spec = registry.spec(node.name)
            if h3_registry_active(registry.available_names()) and (
                node.name.startswith("h3_") or node.name == "mock_text_to_video"
            ):
                continue
            node.config["cost"] = float(spec.estimated_cost)
            node.config["backend"] = spec.backend
            node.config["provenance"] = spec.provenance

    def _parse_edit(
        self,
        raw: dict[str, Any],
        available_tools: set[str],
        edit_index: int = 0,
    ) -> BoundedGraphEdit | None:
        if not isinstance(raw, dict):
            raise GraphMutationError("edit must be an object")
        op = str(raw.get("op", ""))
        if op not in self.ALLOWED_OPS:
            raise GraphMutationError(f"unsupported edit operation {op!r}")
        raw_payload = raw.get("payload") or {}
        if not isinstance(raw_payload, dict):
            raise GraphMutationError("edit payload must be an object")
        payload = dict(raw_payload)
        if op in {"add_node", "replace_node"}:
            nested = payload.pop("node", None)
            if isinstance(nested, dict):
                for key, value in nested.items():
                    payload.setdefault(key, value)
            payload = {
                key: value for key, value in payload.items()
                if key in {"node_id", "node_type", "name", "config"}
            }
            payload.setdefault("config", {})
            if not payload.get("node_type") and payload.get("name"):
                payload["node_type"] = (
                    "verifier"
                    if str(payload["name"]) in self.ALLOWED_VERIFIERS
                    else "tool"
                )
        elif op == "add_edge":
            nested = payload.pop("edge", None)
            if isinstance(nested, dict):
                for key, value in nested.items():
                    payload.setdefault(key, value)
            payload = {
                key: value for key, value in payload.items()
                if key in {"edge_id", "source", "target", "condition", "config"}
            }
            payload.setdefault("condition", "always")
            payload.setdefault("config", {})
        if op == "add_edge" and not payload.get("edge_id"):
            source = self._safe_name(str(payload.get("source") or "source"))
            target = self._safe_name(str(payload.get("target") or "target"))
            payload["edge_id"] = f"llm_edge_{edit_index:02d}_{source}_{target}"
        if op in {"add_node", "replace_node"} and payload.get("node_type") == "tool":
            tool_name = str(payload.get("name", ""))
            if tool_name not in available_tools:
                raise GraphMutationError(f"tool {tool_name!r} is not registered")
        if op in {"add_node", "replace_node"} and payload.get("node_type") == "verifier":
            verifier = str(payload.get("name", ""))
            if verifier not in self.ALLOWED_VERIFIERS:
                raise GraphMutationError(f"verifier {verifier!r} is not registered")
        if op == "add_validator":
            validator = str(payload.get("validator") or raw.get("target") or "").strip()
            if not validator:
                validator = self._infer_validator(str(raw.get("reason") or "")) or ""
            # An empty add_validator carries no executable behavior. Treat it as
            # a repairable no-op so one malformed optional edit cannot discard an
            # otherwise valid and expensive graph proposal.
            if not validator:
                return None
            if validator not in self.ALLOWED_VERIFIERS:
                raise GraphMutationError(f"verifier {validator!r} is not registered")
            payload["validator"] = validator
        if op == "change_threshold":
            target = str(raw.get("target") or payload.get("validator") or "")
            if target in self.ALLOWED_VERIFIERS:
                payload["validator"] = target
            try:
                payload["threshold"] = float(payload["threshold"])
            except (KeyError, TypeError, ValueError) as exc:
                raise GraphMutationError("change_threshold requires a numeric threshold") from exc
        return BoundedGraphEdit(
            op=op,
            target=str(raw["target"]) if raw.get("target") is not None else None,
            payload=payload,
            reason=str(raw.get("reason", "")),
        )

    @classmethod
    def _infer_validator(cls, text: str) -> str | None:
        normalized = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
        aliases = {
            "identity": "identity_consistency",
            "face": "identity_consistency",
            "clothing": "clothing_color_consistency",
            "coat": "clothing_color_consistency",
            "background": "background_preservation",
            "edit": "target_edit_success",
            "action": "prompt_action_alignment",
            "motion": "prompt_action_alignment",
            "prompt": "prompt_action_alignment",
            "vbench": "vbench_dimension_alignment",
        }
        for verifier in cls.ALLOWED_VERIFIERS:
            if verifier in normalized:
                return verifier
        return next((verifier for token, verifier in aliases.items() if token in normalized), None)

    def _dedupe_physical_input_edges(
        self,
        graph: ToolPathGraph,
        repairs: list[str],
        available_tools: set[str] | None = None,
    ) -> None:
        """Compose duplicate visual controls instead of silently discarding one."""
        grouped: dict[tuple[str, str], list[Any]] = {}
        h3_active = h3_registry_active(
            available_tools if available_tools is not None else
            self.onboarding_manager.registry.available_names() if self.onboarding_manager is not None else set()
        )
        for edge in graph.edges:
            if h3_active and graph.node(edge.target).name.startswith("h3_"):
                continue
            binding = str(edge.config.get("binding") or "")
            if binding not in {"reference_image", "reference_video"}:
                continue
            grouped.setdefault((edge.target, binding), []).append(edge)
        remove_ids: set[str] = set()
        for (target, binding), edges in grouped.items():
            if len(edges) <= 1:
                continue
            registry = self.onboarding_manager.registry if self.onboarding_manager is not None else None
            if binding == "reference_image" and registry is not None and registry.has(
                "bridge_compose_reference_images"
            ):
                existing_ids = {node.node_id for node in graph.nodes}
                compose_id = self._unique_node_id(
                    self._safe_name(f"compose_reference_images_{target}"),
                    existing_ids,
                )
                graph.nodes.append(GraphNode(
                    compose_id,
                    "tool",
                    "bridge_compose_reference_images",
                    {"cost": 0.10, "contract_alignment": True},
                ))
                for edge in edges:
                    edge.target = compose_id
                    edge.config = {
                        key: value for key, value in edge.config.items()
                        if key != "binding"
                    }
                graph.edges.append(GraphEdge(
                    f"e_{compose_id}_{target}",
                    compose_id,
                    target,
                    "always",
                    {"binding": "reference_image"},
                ))
                repairs.append(
                    f"composed {len(edges)} reference_image producers for {target} through {compose_id}"
                )
                continue
            ranked = sorted(
                edges,
                key=lambda edge: (
                    graph.node(edge.source).name.startswith("bridge_"),
                    bool(graph.node(edge.source).config.get("contract_alignment")),
                    graph.nodes.index(graph.node(edge.source)),
                ),
                reverse=True,
            )
            keep = ranked[0]
            remove_ids.update(edge.edge_id for edge in ranked[1:])
            repairs.append(
                f"deduplicated {binding} inputs for {target}; kept materialized edge {keep.edge_id}"
            )
        if remove_ids:
            graph.edges = [edge for edge in graph.edges if edge.edge_id not in remove_ids]

    @staticmethod
    def _validate_graph(graph: ToolPathGraph, available_tools: set[str], *, local: bool = False) -> None:
        node_ids = [node.node_id for node in graph.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise GraphMutationError("graph contains duplicate node ids")
        nodes = set(node_ids)
        if not any(node.node_type == "tool" and node.name in available_tools for node in graph.nodes):
            raise GraphMutationError("graph contains no executable tool")
        for node in graph.nodes:
            if node.node_type == "tool" and node.name not in available_tools:
                raise GraphMutationError(f"graph requires unavailable tool {node.name!r}")
            if node.node_type == "verifier" and node.name not in OpenAICompatibleGraphMutationProposer.ALLOWED_VERIFIERS:
                raise GraphMutationError(f"graph requires unavailable verifier {node.name!r}")
        indegree = {node_id: 0 for node_id in nodes}
        outgoing = {node_id: [] for node_id in nodes}
        for edge in graph.edges:
            if edge.source not in nodes or edge.target not in nodes:
                raise GraphMutationError(f"dangling edge {edge.edge_id}: {edge.source}->{edge.target}")
            indegree[edge.target] += 1
            outgoing[edge.source].append(edge.target)
            condition = (edge.condition or "always").strip().lower()
            if not OpenAICompatibleGraphMutationProposer._supported_edge_condition(condition):
                raise GraphMutationError(f"unsupported edge condition {edge.condition!r}")
        ready = [node_id for node_id, degree in indegree.items() if degree == 0]
        visited = 0
        while ready:
            source = ready.pop()
            visited += 1
            for target in outgoing[source]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if visited != len(nodes):
            raise GraphMutationError("graph mutation creates a cycle")
        try:
            validate_h3_node_configs(graph, available_tools, local=local)
        except (TypeError, ValueError) as exc:
            raise GraphMutationError(str(exc)) from exc

    def _request_payload(
        self,
        failed: list[GraphPathRollout],
        available_tools: set[str],
        limit: int,
        search_context: dict[str, Any],
    ) -> dict[str, Any]:
        h3_active = h3_registry_active(available_tools)
        h3_local = h3_local_active(getattr(self.onboarding_manager, "registry", None)) or any(
            rollout.task.metadata.get("evaluation_protocol") == H3_LOCAL_EVALUATION_PROTOCOL
            or rollout.artifact.metadata.get("provider") == "local-h3" for rollout in failed
        )
        cases = []
        for rollout in failed:
            cases.append(
                {
                    "task": {
                        "task_id": rollout.task.task_id,
                        "prompt": rollout.task.prompt,
                        "mode": rollout.task.mode.value,
                        "metadata": rollout.task.metadata,
                    },
                    "parent_graph": rollout.graph.to_dict(),
                    "score": rollout.score,
                    "reward": (
                        rollout.reward.to_dict()
                        if rollout.reward is not None
                        else {
                            "score": rollout.evaluation.score,
                            "objective": "video_quality",
                        }
                    ),
                    "failed_metrics": [asdict(metric) for metric in rollout.evaluation.failed_metrics],
                    "diagnosis": asdict(rollout.failure),
                    "state": rollout.state.to_dict(),
                }
            )
        schema = {
            "capability_requests": [
                {
                    "capability": "a missing capability from the task failure",
                    "preferred_backend": "optional backend from discoverable_tools",
                    "suggested_tool_name": "exact tool name from discoverable_tools",
                    "required_input_types": ["video_or_reference_artifact_type"],
                    "reason": "why registered tools are insufficient",
                }
            ],
            "candidates": [
                {
                    "name": "snake_case_unique_name",
                    "parent_graph_id": "one graph_id from failure_cases",
                    "merge_graph_ids": ["optional graph ids from historical_graphs"],
                    "reason": "causal justification",
                    "description": "what the path executes",
                    "triggers": ["failure_or_prompt_trigger"],
                    "edits": [
                        {
                            "op": "add_node|delete_node|replace_node|add_edge|delete_edge|tighten_trigger|add_validator|add_fallback|change_threshold",
                            "target": "optional existing node or edge id",
                            "payload": {
                                "validator": "required for add_validator; one registered verifier name",
                                "threshold": "required numeric value for change_threshold",
                            },
                            "reason": "why this bounded edit addresses the failure",
                        }
                    ],
                }
            ],
            "exploration_candidates": [
                {
                    "name": "unique_open_world_mechanism_name",
                    "parent_graph_id": "one graph_id from failure_cases",
                    "mechanism_family": "causally distinct family such as reference_bank or post_generation_restoration",
                    "hypothesis": "why this mechanism can solve the observed failure",
                    "novelty_rationale": "how it differs from registered tools and historical graphs",
                    "triggers": ["task_or_failure_trigger"],
                    "nodes": [
                        {
                            "node_id": "capability_node_id",
                            "capability": "atomic functional capability, independent of repository names",
                            "input_types": ["typed upstream artifacts"],
                            "output_type": "typed artifact; every path needs a terminal video output",
                            "realization_policy": "auto|reuse|search_new|compare",
                            "suggested_tool_name": "optional unique tool name for search_new/compare",
                            "description": "node behavior",
                        }
                    ],
                    "edges": [
                        {
                            "edge_id": "unique_edge",
                            "source": "source node_id",
                            "target": "target node_id",
                            "condition": "always or supported verifier condition",
                        }
                    ],
                    "validators": ["registered verifier names"],
                }
            ],
        }
        system = (
            "You are the top-level tool-path scientist for a self-evolving video agent. "
            "Run both exploitation and genuine open-world exploration. In candidates, propose causally motivated, "
            "bounded mutations of supplied parent DAGs. In exploration_candidates, ignore the current graph topology "
            "and first invent several mechanism-level capability DAGs from scratch. Capability DAG nodes describe "
            "functions and typed artifacts, never repository names. Explore causally different solution families rather "
            "than cosmetic reorderings, including reference-memory, generate-then-restore, structured control, iterative "
            "verification, personalization, and other evidence-backed mechanisms when appropriate. "
            "Inside bounded candidates, use registered tools directly. If a required capability is absent, request it in capability_requests when it "
            "appears in discoverable_tools or open_world_acquisition_enabled is true. Request atomic repository-native "
            "primitives, not a fictional tool that combines video, identity, tracking, structure, and temporal planning. "
            + ("The active H3 registry and h3_native_planner grammar are authoritative. Use registered H3 tools "
               "directly, including repeated calls with distinct node ids. Native reference sets carry physical "
               "image/video/audio references; do not replace them with symbolic prompt metadata. "
               if h3_active else
            "Repository-deployable pixel capabilities are exactly text_to_video, image_conditioned_video_generation, "
            "multi_shot_identity_conditioned_generation, motion_conditioned_video_generation, global_video_editing, "
            "region_video_editing, video_style_transfer, temporal_deflickering, and failed_segment_repair. Harness-native "
            "structural capabilities are temporal_planning, scene_splitting, character_sheet_generation, keyframe_generation, "
            "identity_reference_extraction, structure_motion_extraction, object_tracking, and artifact_contract_bridge. "
            "Before requesting a repository for an unfamiliar compound name, decompose it into these atomic capabilities; "
            "keep only a genuinely irreducible new pixel mechanism as an open-world capability. "
            "Temporal_plan and shot_plan are harness planning contexts compiled into {prompt}; pair them with a normal I2V "
            "or V2V primitive. Use image_conditioned_video_generation for identity_reference/keyframes/character_sheet and "
            "global_video_editing for video plus temporal_plan. Compose extraction, tracking, bridges, generation, and repair "
            "as separate graph nodes. ")
            + "Set suggested_tool_name to a unique "
            "snake_case name and use exactly that same name in the bounded candidate graph. Otherwise never invent an "
            "unlisted concrete tool name. Abstract exploration capabilities are explicitly exempt: invent functional "
            "capabilities there, and let the harness realize them after structural screening. "
            "Preserve useful parent structure, connect every new execution node, avoid cycles, and prefer testable repair "
            "paths over prompt-only changes. Every candidate must obey the supplied graph invariants and end at a tool whose "
            "output_type is video. Analysis artifacts only matter when a downstream video generator/editor consumes them. "
            + ("H3 permits mode/root replacement, multiple native generations, reference mutations, global "
               "reference-conditioned regeneration, and ordered audio/video concatenation. The baseline alias "
               "mock_text_to_video uses direct native conditioning from the same task references available to "
               "candidates; it needs no reference bank parent. "
               + ("Local H3 runs at 768P with integer task generation seeds: compare matched task/seed replicates. "
                  "Require at least three distinct replicates; seed control does not guarantee statistical gain. "
                  if h3_local else "Cloud H3 runs at 2K and has no API seed control: evaluation_seed is only "
                  "a replicate label, never a controllable paired provider seed. ")
               if h3_active else
            "The parent T2V node is the fixed Wan2.1 draft generator and the scientific baseline. For generation-task "
            "identity, motion, object, or omission failures, prefer minimum-scope repair: verify the Wan draft, localize "
            "failed segments, extract materialized boundary frames, regenerate/edit only those spans, then use "
            "segment_stitcher to restore them into the untouched draft. Do not treat whole-clip I2V or global V2V "
            "regeneration as localized repair. Preserve every healthy span unless the task explicitly requests global "
            "style transfer. Use failed_segment_localizer, boundary_frame_extractor, and segment_stitcher directly when "
            "registered; these are harness-native execution tools, not repository capabilities. ")
            + "For add_validator, payload.validator is mandatory and must be one of the registered verifier names; omit the "
            "entire edit when no validator is needed. Never emit an empty add_validator payload. "
            "Acquire missing capabilities locally first: trusted catalog, then GitHub/Hugging Face repository discovery and "
            "sandboxed adapter deployment. Treat MCP video APIs as expensive cloud fallbacks only after local acquisition is "
            "exhausted. Set preferred_backend=mcp only when cloud execution is explicitly required. "
            "Return JSON only."
        )
        user = {
            "goal": (
                f"Propose at most {limit} bounded exploitation mutations and at most "
                f"{self.config.invention_candidates} causally distinct open-world capability graphs. "
                f"Only {self.config.invention_realization_budget} capability graphs will be realized, so maximize diversity."
            ),
            "registered_tools": sorted(available_tools),
            "registered_tool_specs": (
                self.onboarding_manager.registry.manifests()
                if self.onboarding_manager is not None
                else [{"name": name} for name in sorted(available_tools)]
            ),
            "discoverable_tools": (
                self.onboarding_manager.discoverable_tools()
                if self.onboarding_manager is not None
                else []
            ),
            "open_world_acquisition_enabled": bool(
                self.onboarding_manager is not None
                and (
                    self.onboarding_manager.mcp_acquirer is not None
                    or self.onboarding_manager.open_world_acquirer is not None
                )
            ),
            "tool_acquisition_priority": [
                "verified_local_registry",
                "trusted_local_catalog",
                "github_huggingface_local_deployment",
                "cloud_mcp_fallback",
            ],
            "capability_ontology": {
                "repository_deployable": sorted(CapabilityRequest.REPOSITORY_CAPABILITIES),
                "harness_native": sorted(CapabilityRequest.HARNESS_CAPABILITIES),
                "decomposition_examples": {
                    "identity_reference_acquisition": "identity_reference_extraction",
                    "shot_and_action_decomposition": "scene_splitting + temporal_planning",
                    "identity_conditioned_temporal_video_generation": (
                        "identity_reference_extraction + image_conditioned_video_generation"
                    ),
                    "temporal_plan_conditioned_video_editing": (
                        "temporal_planning + global_video_editing"
                    ),
                },
            },
            "registered_verifiers": sorted(self.ALLOWED_VERIFIERS),
            "graph_invariants": [
                "Every edge source output_type must match one of the target input_types; 'any' matches any artifact.",
                "Respect registered input_contracts and output_contract: semantic role, format, transport, resolution, fps, frame count, materialization, and required bindings.",
                "When contracts are convertible, use registered artifact_contract_bridge tools or allow the harness to insert a deterministic bridge; never claim symbolic metadata is a local image/video.",
                "When materializing keyframes or character sheets, preserve a draft/source video edge so the bridge can use real sampled pixels; a text-only storyboard is only a last-resort control artifact.",
                "A tool with consumes_upstream=true must have at least one compatible incoming artifact edge.",
                "Do not connect an edge into a source-only tool with empty input_types.",
                "Each terminal execution path must end in a registered tool with output_type=video.",
                "Do not leave temporal plans, keyframes, trackers, or verifiers as terminal outputs.",
            "A temporal plan may feed T2V only when its spec lists temporal_plan; keyframes require a real I2V/TI2V consumer.",
                "Never feed identity_reference, keyframes, character_sheet, or shot_plan to T2V unless its typed input_types explicitly accept it.",
                "Never claim an incompatible artifact can be serialized into prompt metadata; request a typed consuming tool instead.",
                "parent_graph_id must come only from failure_cases. Historical graph ids belong only in merge_graph_ids.",
                "add_node node_id values must not duplicate any node id already present in the parent or merged graph.",
                "When a required generator/editor is absent, emit capability_requests and use the identical suggested_tool_name in edits.",
                "Audio-conditioned video generation must consume a materialized audio artifact; ordinary T2V/I2V tools cannot impersonate it.",
                "Wan T2V remains the draft root. A downstream I2V/editor intended as repair must consume localized evidence and its output must flow through segment_stitcher before final scoring.",
                "For identity, motion, persistence, or omission failures, never replace the whole healthy Wan draft when failed_segments cover only part of the timeline.",
                "Connect the verified Wan video and segment_plan to boundary_frame_extractor; connect the original Wan video, segment_plan, and repaired video to segment_stitcher.",
            ],
            "allowed_edit_ops": sorted(self.ALLOWED_OPS),
            "max_edits_per_candidate": self.config.max_edits,
            "open_world_graph_invention": {
                "enabled": self.config.open_world_invention_enabled,
                "max_candidates": self.config.invention_candidates,
                "realization_budget": self.config.invention_realization_budget,
                "max_nodes_per_graph": self.config.invention_max_nodes,
                "wl_similarity_rejection_threshold": self.config.invention_novelty_threshold,
                "artifact_types": [
                    "any", "video", "image", "identity_reference", "character_sheet", "keyframes",
                    "shot_plan", "temporal_plan", "segment_plan", "structure_motion_map", "tracked_regions",
                ],
                "rules": [
                    "Do not copy the parent graph and rename nodes.",
                    "Each mechanism_family must represent a different causal intervention.",
                    "Use realization_policy=search_new or compare when a genuinely new executable backend is needed.",
                    "Novel non-video analysis capabilities must be decomposed into available atomic analysis primitives; open-world command deployment currently targets pixel-changing video tools.",
                    "Every abstract path must terminate in output_type=video.",
                ],
            },
            "failure_cases": cases,
            "weighted_search_context": {
                key: value for key, value in search_context.items() if key != "historical_graphs"
            },
            "historical_graphs": search_context.get("historical_graphs", []),
            "composition_guidance": (
                "Use merge_graph_ids to combine complementary historical paths. Shared tools become canonical shared nodes. "
                "Prefer high-UCB paths for exploration and high-advantage stable paths for exploitation."
            ),
            "output_schema": schema,
        }
        if h3_active:
            if not user["open_world_acquisition_enabled"]:
                user["graph_invariants"].append(
                    "Tool installation is disabled in this run. Realize every exploration with registered_tool_specs. "
                    "Do not spend realization slots on unavailable state-modeling, tracking or editing backends. "
                    "Diversify executable native paths: boundary FL2VA, reference-frame Ref2VA, "
                    "reference-role/selection changes, or explicitly staged native calls when task durations permit."
                )
            schema["exploration_candidates"][0]["nodes"][0]["config"] = {"prompt": "optional native per-node instruction"}
            user["graph_invariants"] = [
                rule for rule in user["graph_invariants"] if "Wan" not in rule
            ]
            user["h3_native_planner"] = self._h3_native_planner(available_tools, local=h3_local)
            user["weighted_search_context"] = deepcopy(user["weighted_search_context"])
            for context in user["weighted_search_context"].get("task_conditioned", {}).values():
                context["credit_assignment"] = (
                    "Observed quality delta against matched task/seed replicates for local H3; "
                    "seed control does not guarantee statistical gain." if h3_local else
                    "Observed quality delta against the same task/replicate label baseline, "
                    "not matched provider randomness. H3 API seeds are not controllable."
                )
                for arena in context.get("tool_arenas", {}).values():
                    arena["comparison_protocol"] = (H3_LOCAL_COMPARISON_PROTOCOL if h3_local else
                                                    "same graph and task/replicate labels, not matched random seeds")
            user["graph_invariants"].extend(user["h3_native_planner"]["rules"])
            user["open_world_graph_invention"]["artifact_types"].extend(["audio", "h3_reference_set"])
            user["composition_guidance"] = (
                "Use explicit bounded node/edge edits for H3 composition. Distinct calls to the same tool must "
                "keep distinct node ids and per-node configs. merge_graph_ids canonicalizes tool names and "
                "cannot preserve repeated H3 calls or ordered reference bindings."
            )
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

    @staticmethod
    def _h3_native_planner(available_tools: set[str] | None = None, *, local: bool = False) -> dict[str, Any]:
        grammar = {
            "backend": "local-h3" if local else "minimax-h3",
            "resolution": "768P" if local else "2K",
            "provider_seed_control": local,
            "reference_schema": {
                "source": "task.metadata.h3_references",
                "example": {"id": "actor", "kind": "image", "uri": "task-provided URI",
                            "role": "reference_image", "semantic_role": "lead character"},
                "fields": "id, kind:image|video|audio, uri, role:reference_image|reference_video|reference_audio|first_frame|last_frame; optional semantic_role, duration_seconds",
            },
            "rules": [
                "Use only tools present in registered_tool_specs; runtime input/output contracts remain authoritative.",
                "mock_text_to_video is the H3 direct compatibility alias: frame-role references select fl2va, other supplied references select ref2va, no references select t2va. It needs no task bank parent.",
                "keyframe_generator is symbolic planning, NOT a text-to-image model. bridge_materialize_image may draw a semantic storyboard, not a photorealistic scene anchor. For physical references generate a draft with H3 then use h3_frame_extract, or use fixed task references or a verified real image generator. Do not pass a symbolic storyboard as a physical identity/state anchor.",
                "For localized repair, pass failure_localization through boundary extraction and reference packing, select conditioning_strategy=localized_repair, and render ONLY the target interval. The stitcher aligns the entire segment to that interval. A full-video regeneration instead uses corresponding original timeline positions. One repair call repairs one interval; repeat explicitly for additional intervals.",
                "For task duration above 15 seconds, the direct baseline independently generates the explicitly declared h3_shots and concatenates them. Candidate DAGs may choose native mechanisms freely; allocate per-call durations so the final video matches task.duration_seconds within the executor's codec-aware 0.5--0.75 second tolerance, and retain audio when h3_audio_criteria requires it.",
                "For segmented fixed source videos, task.h3_shots[index].reference_ids identifies the source interval. The direct alias selects it automatically for shot_index; explicit h3_ref2va must select the appropriate IDs from its upstream reference bank. Never feed all source segments together if their total duration exceeds 15 seconds. Do not turn a continuous-video requirement into permission for camera cuts; verify continuity across the concat boundary.",
                "Audio reference input requires at least one image or video reference in the same Ref2VA call. Fixed audio-task appearance images are supplied in h3_references. Audio must stay paired with visual references; generated media is not a replacement for the original task audio during verification.",
                "Explicit h3_t2va is text only, accepts optional temporal_plan input only, and never consumes task references implicitly.",
                "h3_fl2va requires an upstream image or h3_reference_set with first_frame/last_frame image roles; h3_ref2va requires an upstream h3_reference_set of image/video/audio references.",
                "Never mix first_frame/last_frame with reference_image/reference_video/reference_audio in one generation call. FL2VA uses only endpoints; Ref2VA uses only ordinary references. To combine a boundary image and an identity image in Ref2VA, explicitly pack BOTH as reference_image and describe their semantic roles. This is soft reference conditioning, NOT a guaranteed first-frame boundary. The harness will not silently change roles or drop references.",
                "All native generators output video; duration_seconds is an integer 4..15 per call. Optional reference_ids selects references in the given order.",
                "Preparation is not generation: reference_set_packing/reference_packing/physical_reference_packaging map to h3_reference_pack and output h3_reference_set, never video. Use h3_reference_pack explicitly where possible. semantic_role belongs inside each bindings entry, not the node config root.",
                "physical_frame_extraction maps to h3_frame_extract with an explicit position; terminal_frame_extraction means position=last. Use actual pixels, never an abstract state description, for these nodes.",
                "Missing cuts/shot coverage and identity drift are different failures. For missing shots consider separately rendered shot branches and explicit h3_av_concat rather than only a global reference rerender. Check per-call minimum duration and final duration first; do not replace a continuous-shot task with hard cuts or invent undeclared shot indices.",
                "A healthy boundary frame may hide the subject during occlusion. For identity-preserving repair consider a clearly visible earlier identity frame as a separate reference, or global reference regeneration. Do not assume the boundary frame establishes identity.",
                "Omit duration_seconds on a whole-video terminal generator: it must inherit the CURRENT task duration, not a fixed example duration. Explicit durations are for intermediate segments/shots; their final composition must match the task duration. A single native call cannot implement a task longer than 15 seconds.",
                "Reusable nodes must derive all entities, objects and actions from the CURRENT task. config.prompt is discovery-task-specific prose: the harness binds it to the discovery prompt hash and omits it for other tasks. Never hardcode a chef, tomato, toy car or another discovery entity into a reusable intervention.",
                "Use config.conditioning_strategy=ordered_actions|preserve_state|preserve_identity|localized_repair for transferable instructions. These strategies resolve against each current task. Do not emit prompt_task_hashes; the harness supplies their scope.",
                "When diagnosis.intervention.near_miss_repair exists, mutate its parent graph to preserve measured gains and fix regressed criteria. Consider real reference-frame conditioning for identity regressions. A rejected near-miss is not an accepted path; revalidate against the accepted baseline.",
                "failed_segment_localizer obtains current-video verifier evidence at runtime. Its diagnosed intervals belong to that video, not to every task in the family. If the failure spans a causal sequence, consider native whole-video reference regeneration instead of blindly splicing a short clip.",
                "Optional shot_index is a nonnegative integer selecting a declared task.metadata.h3_shots[index] object {prompt,duration_seconds?}. Task h3_global_constraints supplies optional shared context; config.prompt adds a stage note. Use reusable shot-index chains only when the task declares those indices; otherwise use explicit stage prompts. Do not invent shots or reference assets for text-only tasks.",
                ("Local H3 at 768P applies task.metadata.generation_seed as the integer replicate seed. "
                 "Do not put seed/generation_seed/random_seed in node configs or override matched evaluation seeds. "
                 "Use matched_task_seed_replicates with at least three distinct replicates; no statistical gain is guaranteed."
                 if local else "No provider seeds are supported. Do not put seed/generation_seed/random_seed in node configs. evaluation_seed labels independent replicates only; do not claim paired API seed control."),
                "h3_reference_bank is a source (no artifact inputs) returning h3_reference_set from task.metadata.h3_references; use only supplied ids/URIs, never invent reference assets.",
                "h3_reference_select takes/returns h3_reference_set; reference_ids is a required ordered list, with optional roles mapping selected id to API role and semantic_roles mapping selected id to a semantic description.",
                "h3_frame_extract takes video and returns image; position is first|last|time, only time uses/requires time_seconds. Optional role is first_frame|last_frame|reference_image; by default first maps to first_frame, last to last_frame, time to reference_image.",
                "h3_reference_pack takes image/video/audio/h3_reference_set and returns h3_reference_set. bindings is an explicit ordered list of {source: upstream node id, kind, role, semantic_role?, reference_id?}; add an incoming edge for every source.",
                "h3_reference_trim takes/returns h3_reference_set; choose reference_id and 0 <= start_seconds < end_seconds for a video/audio reference.",
                "h3_av_concat takes videos and outputs video; source_nodes must explicitly order its incoming video node ids. Repeated ids intentionally repeat clips; set(source_nodes) must equal the parent video ids. Edge order alone is not binding order.",
                "video_concatenation/audio_video_concatenation mean h3_av_concat, not audio-conditioned generation; concat preserves existing tracks and requires video inputs, not an extra audio input. Allocate segment durations for every routed task; a fixed 6+4 second split is invalid for a 9 second task. Duration mismatches are rejected before any generation, not fixed by trimming or speeding up videos.",
                "Root/mode replacement, multiple H3 calls, global reference regeneration, select/reorder/re-role/trim/pack mutations and concat are allowed; no Wan draft preservation or segment_stitcher requirement applies.",
            ],
            "node_examples": [
                {"node_id": "bank", "node_type": "tool", "name": "h3_reference_bank", "config": {}},
                {"node_id": "selected", "node_type": "tool", "name": "h3_reference_select", "config": {"reference_ids": ["actor", "voice"], "roles": {"actor": "reference_image", "voice": "reference_audio"}, "semantic_roles": {"actor": "lead character identity", "voice": "lead character dialogue"}}},
                {"node_id": "trimmed", "node_type": "tool", "name": "h3_reference_trim", "config": {"reference_id": "voice", "start_seconds": 0, "end_seconds": 4}},
                {"node_id": "shot_a", "node_type": "tool", "name": "h3_ref2va", "config": {"reference_ids": ["actor", "voice"], "duration_seconds": 6, "prompt": "First shot: the character speaks."}},
                {"node_id": "boundary", "node_type": "tool", "name": "h3_frame_extract", "config": {"position": "last", "role": "first_frame"}},
                {"node_id": "packed", "node_type": "tool", "name": "h3_reference_pack", "config": {"bindings": [{"source": "boundary", "kind": "image", "role": "first_frame", "reference_id": "continuity", "semantic_role": "continuation frame"}]}},
                {"node_id": "shot_b", "node_type": "tool", "name": "h3_fl2va", "config": {"duration_seconds": 4, "reference_ids": ["continuity"], "prompt": "Continue the motion in the second shot."}},
                {"node_id": "final", "node_type": "tool", "name": "h3_av_concat", "config": {"source_nodes": ["shot_a", "shot_b"]}},
                {"node_id": "text_only_alternative", "node_type": "tool", "name": "h3_t2va", "config": {"conditioning_strategy": "ordered_actions"}},
            ],
            "example_edges": [["bank", "selected"], ["selected", "trimmed"], ["trimmed", "shot_a"],
                              ["shot_a", "boundary"], ["boundary", "packed"], ["packed", "shot_b"],
                              ["shot_a", "final"], ["shot_b", "final"]],
            "reusable_shot_example": {
                "task_metadata": {"h3_shots": [{"prompt": "Walk into the room.", "duration_seconds": 6}, {"prompt": "Turn and wave.", "duration_seconds": 4}], "h3_global_constraints": "Same character and clothing throughout."},
                "node_configs": {"shot_a": {"shot_index": 0}, "shot_b": {"shot_index": 1, "prompt": "Continue from the preceding boundary frame."}},
                "note": "Optional alternative configs; keep reference selection/bindings when the chosen native mode needs them. Explicit node duration_seconds overrides the shot duration.",
            },
            "examples_note": "Illustrative ids must be replaced with actual task references. text_only_alternative is a separate path, not another terminal of the concat example.",
        }
        if available_tools is not None and "h3_audio_extract" in available_tools:
            grammar["rules"].append(
                "h3_audio_extract takes video and produces materialized WAV audio, with optional start_seconds/end_seconds. "
                "Feed that audio to h3_reference_pack with kind=audio, role=reference_audio for reusable soundtrack conditioning."
            )
            grammar["optional_node_examples"] = [{
                "node_id": "soundtrack", "node_type": "tool", "name": "h3_audio_extract",
                "config": {"start_seconds": 0, "end_seconds": 4},
            }]
        return grammar

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.config.api_key:
            raise GraphMutationError(
                "LLM graph mutation is enabled but no API key is configured; set GRAPH_LLM_API_KEY, "
                "OPENROUTER_API_KEY, OPENAI_API_KEY, or DASHSCOPE_API_KEY"
            )
        request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise GraphMutationError(f"LLM mutation request failed ({exc.code}): {detail[:2000]}") from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            raise GraphMutationError(f"LLM mutation request failed: {exc}") from exc

    @staticmethod
    def _response_content(response: dict[str, Any]) -> Any:
        try:
            choice = response["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GraphMutationError(f"invalid chat completion response: {response}") from exc
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            content = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        if not content and isinstance(message, dict):
            tool_calls = message.get("tool_calls") or []
            if tool_calls and isinstance(tool_calls[0], dict):
                content = (tool_calls[0].get("function") or {}).get("arguments")
            if not content:
                content = (message.get("function_call") or {}).get("arguments")
        if not content:
            content = response.get("output_text") if isinstance(response, dict) else None
        if not content:
            finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
            refusal = message.get("refusal") if isinstance(message, dict) else None
            raise GraphMutationError(
                f"LLM returned empty structured content (finish_reason={finish_reason!r}, refusal={refusal!r})"
            )
        return content

    @staticmethod
    def _parse_json(content: Any) -> dict[str, Any]:
        try:
            return parse_json_object(content)
        except StructuredJSONError as exc:
            raise GraphMutationError(f"LLM did not return valid JSON: {str(content)[:1000]}") from exc

    @staticmethod
    def _repair_payload(payload: dict[str, Any], errors: list[str] | None = None) -> dict[str, Any]:
        repaired = json.loads(json.dumps(payload))
        repaired["messages"].append(
            {
                "role": "user",
                "content": (
                    "Your previous response was invalid or truncated. Return one compact, complete JSON object only, "
                    "with capability_requests and candidates. Do not repeat the input, analysis, schemas, failure cases, "
                    "or repository documentation. Keep at most the requested number of candidates."
                    + (" Previous errors: " + "; ".join(errors or []) if errors else "")
                ),
            }
        )
        return repaired

    @classmethod
    def _length_repair_payload(
        cls,
        payload: dict[str, Any],
        attempt: int,
        errors: list[str] | None = None,
    ) -> dict[str, Any]:
        """Recover from reasoning models consuming the completion budget before emitting JSON."""
        repaired = json.loads(json.dumps(payload))
        token_key = "max_completion_tokens" if "max_completion_tokens" in repaired else "max_tokens"
        current = int(repaired.get(token_key, 8192))
        cap = max(current, int(os.environ.get("GRAPH_LLM_LENGTH_RETRY_MAX_TOKENS", "32768")))
        repaired[token_key] = min(cap, current * (2 ** max(1, attempt)))
        if "reasoning_effort" in repaired:
            repaired["reasoning_effort"] = "medium" if attempt == 1 else "low"
        try:
            user = json.loads(repaired["messages"][1]["content"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            user = None
        if isinstance(user, dict):
            user["historical_graphs"] = list(user.get("historical_graphs", []))[:3]
            user["discoverable_tools"] = list(user.get("discoverable_tools", []))[:30]
            for spec in user.get("registered_tool_specs", []):
                if isinstance(spec, dict) and spec.get("description"):
                    spec["description"] = str(spec["description"])[:240]
            for case in user.get("failure_cases", []):
                if not isinstance(case, dict):
                    continue
                state = case.get("state")
                if isinstance(state, dict):
                    case["state"] = {
                        key: state[key]
                        for key in ("verifier_scores", "failures", "budget_used")
                        if key in state
                    }
            repaired["messages"][1]["content"] = json.dumps(user, ensure_ascii=False)
        repaired["messages"].append(
            {
                "role": "user",
                "content": (
                    "The previous completion exhausted its token budget before emitting JSON. "
                    "Return a compact JSON object immediately, with no prose and at most two candidates. "
                    "Keep each candidate to the minimum bounded edits needed."
                    + (" Previous errors: " + "; ".join(errors or []) if errors else "")
                ),
            }
        )
        return repaired

    @staticmethod
    def _safe_name(value: str) -> str:
        name = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip()).strip("_").lower()
        if not name:
            raise GraphMutationError("candidate name is empty")
        return name[:80]


def mutation_config_from_env(
    model: str = "openai/gpt-5.6-sol",
    base_url: str | None = None,
    timeout_seconds: int = 120,
    temperature: float = 0.2,
    max_candidates: int = 3,
    max_edits: int = 12,
    max_output_tokens: int = 8192,
    reasoning_effort: str | None = "high",
    open_world_invention_enabled: bool = True,
    invention_candidates: int = 6,
    invention_realization_budget: int = 3,
    invention_max_nodes: int = 8,
    invention_novelty_threshold: float = 0.82,
) -> GraphMutationConfig:
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
    return GraphMutationConfig(
        model=model,
        base_url=resolved_base_url,
        api_key=resolved_key,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_candidates=max_candidates,
        max_edits=max_edits,
        max_output_tokens=int(os.environ.get("GRAPH_LLM_MAX_OUTPUT_TOKENS", str(max_output_tokens))),
        repair_attempts=max(0, int(os.environ.get("GRAPH_LLM_REPAIR_ATTEMPTS", "2"))),
        reasoning_effort=(
            os.environ.get("GRAPH_LLM_REASONING_EFFORT", reasoning_effort or "") or None
            if openai_model
            else None
        ),
        open_world_invention_enabled=(
            os.environ.get("OPEN_WORLD_GRAPH_INVENTION", "1" if open_world_invention_enabled else "0") == "1"
        ),
        invention_candidates=max(
            0, int(os.environ.get("OPEN_WORLD_GRAPH_IDEAS", str(invention_candidates)))
        ),
        invention_realization_budget=max(
            0,
            int(os.environ.get(
                "OPEN_WORLD_GRAPH_REALIZATION_BUDGET",
                str(invention_realization_budget),
            )),
        ),
        invention_max_nodes=max(
            2, int(os.environ.get("OPEN_WORLD_GRAPH_MAX_NODES", str(invention_max_nodes)))
        ),
        invention_novelty_threshold=float(os.environ.get(
            "OPEN_WORLD_GRAPH_WL_THRESHOLD",
            str(invention_novelty_threshold),
        )),
    )


def _is_openai_reasoning_model(model: str) -> bool:
    model_name = model.rsplit("/", 1)[-1].lower()
    return model_name.startswith(("gpt-", "o1", "o3", "o4"))
