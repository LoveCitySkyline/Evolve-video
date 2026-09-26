from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evovideo_skill.graph_algorithms import approximate_graph_edit_distance, louvain_style_communities, wl_kernel_similarity
from evovideo_skill.graph_mining import GlobalToolGraph, GraphMotifMiner, ToolMotif
from evovideo_skill.graph_skill import GraphEdge, GraphNode, GraphSkillMemory, ToolPathGraph
from evovideo_skill.models import SkillValidationReport
from evovideo_skill.skill_memory import SkillMemory


@dataclass
class FoundationSkill:
    skill_name: str
    motif: ToolMotif
    graph: ToolPathGraph

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "motif": self.motif.to_dict(),
            "graph": self.graph.to_dict(),
        }


@dataclass
class FoundationSkillReport:
    global_graph: GlobalToolGraph
    motifs: list[ToolMotif]
    selected_motifs: list[ToolMotif]
    foundation_skills: list[FoundationSkill]
    centrality: dict[str, dict[str, float]]
    communities: dict[str, int]
    graph_similarities: list[dict[str, Any]]
    report_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_graph": self.global_graph.to_dict(),
            "motifs": [item.to_dict() for item in self.motifs],
            "selected_motifs": [item.to_dict() for item in self.selected_motifs],
            "foundation_skills": [item.to_dict() for item in self.foundation_skills],
            "centrality": self.centrality,
            "communities": self.communities,
            "graph_similarities": self.graph_similarities,
            "report_path": self.report_path,
        }


class FoundationSkillConsolidator:
    """Promote reusable graph motifs into foundation video skills."""

    def __init__(
        self,
        graph_memory: GraphSkillMemory,
        skill_memory: SkillMemory,
        miner: GraphMotifMiner | None = None,
        max_foundation_skills: int = 5,
    ):
        self.graph_memory = graph_memory
        self.skill_memory = skill_memory
        self.miner = miner or GraphMotifMiner()
        self.max_foundation_skills = max_foundation_skills

    def consolidate(self) -> FoundationSkillReport:
        graphs = self.graph_memory.list_graphs()
        experiences = self.graph_memory.list_experiences()
        evidence_backed_graphs = self.miner.evidence_backed_graphs(graphs, experiences)
        global_graph = self.miner.build_global_tool_graph(evidence_backed_graphs, experiences)
        motifs = self.miner.mine(evidence_backed_graphs, experiences)
        selected = self._select_non_redundant_motifs(motifs)[: self.max_foundation_skills]
        foundation_skills = [self._materialize_foundation_skill(motif, evidence_backed_graphs) for motif in selected]
        centrality = self.miner.centrality(global_graph)
        communities = louvain_style_communities(global_graph)
        graph_similarities = self._graph_similarities(evidence_backed_graphs)
        report_path = self._write_report(global_graph, motifs, selected, foundation_skills, centrality, communities, graph_similarities)
        return FoundationSkillReport(global_graph, motifs, selected, foundation_skills, centrality, communities, graph_similarities, str(report_path))

    def _select_non_redundant_motifs(self, motifs: list[ToolMotif]) -> list[ToolMotif]:
        selected: list[ToolMotif] = []
        selected_tool_sets: list[set[str]] = []
        for motif in motifs:
            tools = set(motif.tools)
            redundant = False
            for existing in selected_tool_sets:
                if tools <= existing:
                    redundant = True
                    break
            if redundant:
                continue
            selected.append(motif)
            selected_tool_sets.append(tools)
        return selected

    def _materialize_foundation_skill(self, motif: ToolMotif, graphs: list[ToolPathGraph]) -> FoundationSkill:
        skill_name = f"foundation_{motif.motif_id}_skill".replace("-", "_")
        source_graphs = [graph for graph in graphs if graph.skill_name in set(motif.source_graph_ids)]
        triggers = self._foundation_triggers(motif, source_graphs)
        nodes = [
            GraphNode(
                node_id=f"foundation_tool_{idx}",
                node_type="tool",
                name=tool,
                config={"cost": motif.estimated_cost / max(1, len(motif.tools)), "foundation_role": True},
            )
            for idx, tool in enumerate(motif.tools)
        ]
        edges = [
            GraphEdge(
                edge_id=f"foundation_edge_{idx}_{idx + 1}",
                source=nodes[idx].node_id,
                target=nodes[idx + 1].node_id,
                condition="motif_sequence",
            )
            for idx in range(len(nodes) - 1)
        ]
        validators = sorted({validator for graph in source_graphs for validator in graph.validators})
        graph = ToolPathGraph(
            graph_id=skill_name,
            skill_name=skill_name,
            description=(
                "Foundation skill consolidated from reusable high-utility tool motif: "
                + " -> ".join(motif.tools)
            ),
            triggers=triggers,
            nodes=nodes,
            edges=edges,
            validators=validators,
            fallbacks=sorted({fallback for graph in source_graphs for fallback in graph.fallbacks}),
            stats={
                "foundation_skill": True,
                "motif_id": motif.motif_id,
                "support": motif.support,
                "source_graph_ids": motif.source_graph_ids,
                "avg_gain": motif.avg_gain,
                "candidate_score": motif.avg_score,
                "quality_gain": motif.avg_gain,
                "estimated_cost": motif.estimated_cost,
                "stability": motif.stability,
                "utility": motif.utility,
                "mdl_gain": motif.mdl_gain,
                "accepted": True,
            },
        )
        self.graph_memory.upsert_graph(graph)
        skill = graph.to_skill_card()
        skill.version = "foundation-1.0"
        skill.procedure = [
            "retrieve this foundation skill when the prompt or failure state matches its consolidated motif",
            "expand the foundation node into its executable tool sequence",
            "reuse cached intermediate artifacts when the same motif appears inside a larger task-specific path",
            "run the inherited verifiers before composing this motif with downstream tools",
        ]
        skill.anti_patterns = [
            "do not promote single-case motifs without enough support or utility",
            "do not duplicate a larger accepted foundation motif with a strict subset unless it is cheaper and more stable",
        ]
        skill.validation = SkillValidationReport(
            skill_name=skill.skill_name,
            validation_task_ids=motif.validation_task_ids,
            baseline_score=max(0.0, motif.avg_score - motif.avg_gain),
            candidate_score=motif.avg_score,
            baseline_pass_rate=0.0,
            candidate_pass_rate=(
                sum(float(graph.stats.get("pass_rate", 0.0)) for graph in source_graphs)
                / max(1, len(source_graphs))
            ),
            quality_gain=motif.avg_gain,
            estimated_cost=motif.estimated_cost,
            accepted=True,
            evidence=motif.evidence + [f"mdl_gain={motif.mdl_gain:.3f}", f"utility={motif.utility:.3f}"],
        )
        self.skill_memory.upsert(skill)
        return FoundationSkill(skill_name, motif, graph)

    @staticmethod
    def _foundation_triggers(motif: ToolMotif, source_graphs: list[ToolPathGraph]) -> list[str]:
        triggers = []
        for graph in source_graphs:
            triggers.extend(graph.triggers)
        triggers.extend(motif.tools)
        tool_text = " ".join(motif.tools)
        if "image_to_video" in tool_text or "i2v" in tool_text:
            triggers.extend(["identity", "visual anchor", "consistent character"])
        if "style" in tool_text:
            triggers.extend(["style transfer", "anime", "appearance style"])
        if "region" in tool_text or "tracker" in tool_text:
            triggers.extend(["region edit", "only target", "background preservation"])
        if "keyframe" in tool_text or "segment" in tool_text:
            triggers.extend(["motion", "action order", "segment repair"])
        return sorted(dict.fromkeys(item for item in triggers if item))

    def _write_report(
        self,
        global_graph: GlobalToolGraph,
        motifs: list[ToolMotif],
        selected: list[ToolMotif],
        foundation_skills: list[FoundationSkill],
        centrality: dict[str, dict[str, float]],
        communities: dict[str, int],
        graph_similarities: list[dict[str, Any]],
    ) -> Path:
        report_dir = self.graph_memory.graph_dir / "foundation"
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / "foundation_report.json"
        payload = {
            "global_graph": global_graph.to_dict(),
            "motifs": [item.to_dict() for item in motifs],
            "selected_motifs": [item.to_dict() for item in selected],
            "foundation_skills": [item.to_dict() for item in foundation_skills],
            "centrality": centrality,
            "communities": communities,
            "graph_similarities": graph_similarities,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        return path

    @staticmethod
    def _graph_similarities(graphs: list[ToolPathGraph]) -> list[dict[str, Any]]:
        accepted = [graph for graph in graphs if graph.stats.get("accepted", True)]
        similarities: list[dict[str, Any]] = []
        for idx, left in enumerate(accepted):
            for right in accepted[idx + 1 :]:
                similarities.append(
                    {
                        "left": left.skill_name,
                        "right": right.skill_name,
                        "wl_similarity": wl_kernel_similarity(left, right),
                        "approx_graph_edit_distance": approximate_graph_edit_distance(left, right),
                    }
                )
        return sorted(similarities, key=lambda item: (-float(item["wl_similarity"]), float(item["approx_graph_edit_distance"])))
