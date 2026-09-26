from __future__ import annotations

import html
import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evovideo_skill.graph_skill import ToolPathGraph
from evovideo_skill.models import utc_now


class EvolutionGraphArchive:
    """Append-only graph snapshots, execution traces, and self-contained exports."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.snapshot_dir = self.root / "snapshots"
        self.manifest_path = self.root / "manifest.jsonl"
        self.execution_path = self.root / "executions.jsonl"
        self.candidate_audit_path = self.root / "candidate_audits.jsonl"
        self.tool_arena_path = self.root / "tool_arena_rankings.json"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._sequence = self._next_sequence()

    def reset(self) -> None:
        for path in self.snapshot_dir.glob("*.json"):
            path.unlink()
        for path in (
            self.manifest_path,
            self.execution_path,
            self.candidate_audit_path,
            self.tool_arena_path,
        ):
            if path.exists():
                path.unlink()
        self._sequence = 1

    def record_graph(
        self,
        graph: ToolPathGraph,
        *,
        stage: str,
        status: str,
        iteration: int,
        program: str | None = None,
        parent_program: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sequence = self._sequence
        self._sequence += 1
        snapshot_id = f"{sequence:06d}_{_safe(graph.skill_name)}"
        snapshot_path = self.snapshot_dir / f"{snapshot_id}.json"
        event = {
            "snapshot_id": snapshot_id,
            "sequence": sequence,
            "stage": stage,
            "status": status,
            "iteration": iteration,
            "program": program,
            "parent_program": parent_program,
            "graph_id": graph.graph_id,
            "skill_name": graph.skill_name,
            "tools": graph.tool_names(),
            "snapshot_path": str(snapshot_path.relative_to(self.root)),
            "metadata": metadata or {},
            "created_at": utc_now(),
        }
        _write_json(snapshot_path, {"event": event, "graph": graph.to_dict()})
        _append_jsonl(self.manifest_path, event)
        return event

    def record_execution(
        self,
        *,
        task_id: str,
        graph_id: str,
        score: float,
        passed: bool,
        tool_chain: list[str],
        estimated_cost: float,
        cache_hit: bool,
        failure_types: list[str],
        metric_scores: dict[str, float],
        active_metric_names: list[str] | None = None,
        evaluation_seed: int | None = None,
        seed_controlled: bool | None = None,
        execution_error: str | None = None,
        failed_tool_name: str | None = None,
        artifact_path: str | None = None,
        runtime_log: str | None = None,
        tool_provenance: dict[str, dict[str, Any]] | None = None,
        reward_objective: str = "video_quality",
        reward_components: dict[str, float] | None = None,
        reward_weights: dict[str, float] | None = None,
        missing_reward_metrics: list[str] | None = None,
        verifier_evidence: dict[str, Any] | None = None,
    ) -> None:
        _append_jsonl(
            self.execution_path,
            {
                "task_id": task_id,
                "graph_id": graph_id,
                "score": score,
                "passed": passed,
                "tool_chain": tool_chain,
                "estimated_cost": estimated_cost,
                "cache_hit": cache_hit,
                "failure_types": failure_types,
                "metric_scores": metric_scores,
                "active_metric_names": active_metric_names or [],
                "evaluation_seed": evaluation_seed,
                "seed_controlled": seed_controlled,
                "execution_error": execution_error,
                "failed_tool_name": failed_tool_name,
                "artifact_path": artifact_path,
                "runtime_log": runtime_log,
                "tool_provenance": tool_provenance or {},
                "reward_objective": reward_objective,
                "reward_components": reward_components or {},
                "reward_weights": reward_weights or {},
                "missing_reward_metrics": missing_reward_metrics or [],
                "verifier_evidence": verifier_evidence or {},
                "created_at": utc_now(),
            },
        )

    def record_candidate_audit(self, payload: dict[str, Any]) -> None:
        _append_jsonl(
            self.candidate_audit_path,
            {**payload, "created_at": utc_now()},
        )

    def export(
        self,
        *,
        programs: list[Any],
        feedback: list[Any],
        weighted_categories: dict[str, Any],
        frontier: list[str],
    ) -> dict[str, str]:
        snapshots = _read_jsonl(self.manifest_path)
        executions = _read_jsonl(self.execution_path)
        candidate_audits = _read_jsonl(self.candidate_audit_path)
        graph_payloads = []
        for event in snapshots:
            path = self.root / event["snapshot_path"]
            if path.exists():
                graph_payloads.append(json.loads(path.read_text(encoding="utf-8")))
        payload = {
            "version": 1,
            "created_at": utc_now(),
            "frontier": frontier,
            "programs": [item.to_dict() for item in programs],
            "feedback": [asdict(item) for item in feedback],
            "snapshots": graph_payloads,
            "executions": executions,
            "candidate_audits": candidate_audits,
            "weighted_task_graphs": weighted_categories,
            "tool_arenas": {
                category: details.get("tool_arenas", {})
                for category, details in weighted_categories.items()
                if details.get("tool_arenas")
            },
        }
        json_path = self.root / "evolution_graph.json"
        dot_path = self.root / "evolution_graph.dot"
        graphml_path = self.root / "evolution_graph.graphml"
        html_path = self.root / "index.html"
        _write_json(json_path, payload)
        _write_json(
            self.tool_arena_path,
            {
                "created_at": payload["created_at"],
                "task_classes": payload["tool_arenas"],
                "path_rankings": {
                    category: details.get("top_paths", [])
                    for category, details in weighted_categories.items()
                },
            },
        )
        dot_path.write_text(_to_dot(payload), encoding="utf-8")
        graphml_path.write_text(_to_graphml(payload), encoding="utf-8")
        html_path.write_text(_interactive_html(payload), encoding="utf-8")
        return {
            "directory": str(self.root),
            "html": str(html_path),
            "json": str(json_path),
            "dot": str(dot_path),
            "graphml": str(graphml_path),
            "manifest": str(self.manifest_path),
            "executions": str(self.execution_path),
            "candidate_audits": str(self.candidate_audit_path),
            "tool_arena_rankings": str(self.tool_arena_path),
        }

    def export_video_comparisons(
        self,
        tasks: list[Any],
        *,
        top_k: int = 5,
    ) -> dict[str, str]:
        """Persist reproducible Wan-baseline/optimized pairs ranked by reward gain."""
        executions = _read_jsonl(self.execution_path)
        task_index = {str(task.task_id): task for task in tasks}
        materialized = [
            item for item in executions
            if not item.get("execution_error")
            and item.get("artifact_path")
            and Path(str(item["artifact_path"])).expanduser().is_file()
        ]
        baselines = [item for item in materialized if self._is_wan_baseline(item)]
        candidates = [item for item in materialized if not self._is_wan_baseline(item)]
        pairs: list[dict[str, Any]] = []
        seen_candidates: set[tuple[str, str]] = set()
        for candidate in candidates:
            task_id = str(candidate.get("task_id") or "")
            candidate_path = str(Path(str(candidate["artifact_path"])).expanduser().resolve())
            candidate_key = (task_id, candidate_path)
            if candidate_key in seen_candidates:
                continue
            seen_candidates.add(candidate_key)
            compatible = [item for item in baselines if str(item.get("task_id")) == task_id]
            same_seed = [
                item for item in compatible
                if item.get("evaluation_seed") == candidate.get("evaluation_seed")
            ]
            compatible = same_seed or compatible
            if not compatible:
                continue
            baseline = min(
                compatible,
                key=lambda item: (
                    str(item.get("created_at") or ""),
                    str(item.get("artifact_path") or ""),
                ),
            )
            baseline_score = float(baseline.get("score", 0.0))
            candidate_score = float(candidate.get("score", 0.0))
            metric_names = set(baseline.get("metric_scores", {})) | set(candidate.get("metric_scores", {}))
            metric_deltas = {
                name: float(candidate.get("metric_scores", {}).get(name, 0.0))
                - float(baseline.get("metric_scores", {}).get(name, 0.0))
                for name in sorted(metric_names)
            }
            pairs.append({
                "task_id": task_id,
                "evaluation_seed": candidate.get("evaluation_seed"),
                "reward_gain": candidate_score - baseline_score,
                "positive_gain": candidate_score > baseline_score,
                "baseline": baseline,
                "optimized": candidate,
                "metric_deltas": metric_deltas,
            })

        pairs.sort(
            key=lambda item: (
                float(item["reward_gain"]),
                float(item["optimized"].get("score", 0.0)),
                item["task_id"],
            ),
            reverse=True,
        )
        best_per_task: list[dict[str, Any]] = []
        seen_tasks: set[str] = set()
        for pair in pairs:
            if pair["task_id"] in seen_tasks:
                continue
            seen_tasks.add(pair["task_id"])
            best_per_task.append(pair)
        selected = best_per_task[: max(0, int(top_k))]

        output_dir = self.root / "comparison_videos"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        exported: list[dict[str, Any]] = []
        for rank, pair in enumerate(selected, start=1):
            task_id = pair["task_id"]
            pair_dir = output_dir / f"{rank:02d}_{_safe(task_id)}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            baseline_source = Path(str(pair["baseline"]["artifact_path"])).expanduser().resolve()
            optimized_source = Path(str(pair["optimized"]["artifact_path"])).expanduser().resolve()
            baseline_target = pair_dir / "wan_baseline.mp4"
            optimized_target = pair_dir / "optimized.mp4"
            shutil.copy2(baseline_source, baseline_target)
            shutil.copy2(optimized_source, optimized_target)
            task = task_index.get(task_id)
            source_target = None
            reference = str(getattr(task, "reference_video", "") or "").strip()
            if reference and Path(reference).expanduser().is_file():
                source_target = pair_dir / "task_reference.mp4"
                shutil.copy2(Path(reference).expanduser().resolve(), source_target)
            metadata = {
                **pair,
                "rank": rank,
                "prompt": str(getattr(task, "prompt", "")),
                "task_family": str((getattr(task, "metadata", {}) or {}).get("task_family") or ""),
                "selection_policy": "highest task-conditioned reward gain per task; top-k across tasks",
                "exported_files": {
                    "wan_baseline": str(baseline_target.relative_to(output_dir)),
                    "optimized": str(optimized_target.relative_to(output_dir)),
                    "task_reference": (
                        str(source_target.relative_to(output_dir)) if source_target else None
                    ),
                },
            }
            _write_json(pair_dir / "metrics.json", metadata)
            exported.append(metadata)

        manifest = {
            "version": 1,
            "created_at": utc_now(),
            "selection_policy": "automatic top-k by paired reward gain; no manual cherry-picking",
            "pair_count": len(exported),
            "pairs": exported,
            "all_ranked_pairs": pairs,
        }
        manifest_path = output_dir / "comparison_manifest.json"
        html_path = output_dir / "index.html"
        _write_json(manifest_path, manifest)
        html_path.write_text(_comparison_html(exported), encoding="utf-8")
        print(
            f"[comparison export] pairs={len(exported)} html={html_path.resolve()}",
            flush=True,
        )
        return {
            "comparison_directory": str(output_dir),
            "comparison_manifest": str(manifest_path),
            "comparison_html": str(html_path),
        }

    @staticmethod
    def _is_wan_baseline(execution: dict[str, Any]) -> bool:
        tools = [str(item) for item in execution.get("tool_chain", [])]
        graph_id = str(execution.get("graph_id") or "").lower()
        return tools == ["mock_text_to_video"] or graph_id in {
            "baseline_t2v_graph",
            "baseline",
        }

    def _next_sequence(self) -> int:
        events = _read_jsonl(self.manifest_path)
        return max((int(item.get("sequence", 0)) for item in events), default=0) + 1


def _to_dot(payload: dict[str, Any]) -> str:
    lines = ["digraph ToolChainEvolution {", "  rankdir=LR;", "  node [fontname=Helvetica];"]
    for program in payload["programs"]:
        name = _dot(program["name"])
        status = program.get("status", "candidate")
        color = {"frontier": "#16a34a", "rejected": "#dc2626", "archived": "#64748b"}.get(status, "#2563eb")
        lines.append(f'  "p:{name}" [label="{name}", shape=box, color="{color}"];')
        if program.get("parent"):
            lines.append(f'  "p:{_dot(program["parent"])}" -> "p:{name}" [label="mutate"];')
        for graph_id in program.get("graph_ids", []):
            lines.append(f'  "g:{_dot(graph_id)}" [label="{_dot(graph_id)}", shape=ellipse];')
            lines.append(f'  "p:{name}" -> "g:{_dot(graph_id)}" [style=dashed, label="uses"];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def _to_graphml(payload: dict[str, Any]) -> str:
    nodes: dict[str, tuple[str, str]] = {}
    edges: list[tuple[str, str, str]] = []
    for program in payload["programs"]:
        pid = f'p:{program["name"]}'
        nodes[pid] = (program["name"], "program")
        if program.get("parent"):
            parent = f'p:{program["parent"]}'
            nodes.setdefault(parent, (program["parent"], "program"))
            edges.append((parent, pid, "mutation"))
        for graph_id in program.get("graph_ids", []):
            gid = f"g:{graph_id}"
            nodes[gid] = (graph_id, "tool_chain_graph")
            edges.append((pid, gid, "uses"))
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '<key id="label" for="node" attr.name="label" attr.type="string"/>',
        '<key id="kind" for="all" attr.name="kind" attr.type="string"/>',
        '<graph id="tool-chain-evolution" edgedefault="directed">',
    ]
    for node_id, (label, kind) in nodes.items():
        lines.append(f'<node id="{html.escape(node_id, quote=True)}"><data key="label">{html.escape(label)}</data><data key="kind">{kind}</data></node>')
    for index, (source, target, kind) in enumerate(edges):
        lines.append(f'<edge id="e{index}" source="{html.escape(source, quote=True)}" target="{html.escape(target, quote=True)}"><data key="kind">{kind}</data></edge>')
    lines.extend(["</graph>", "</graphml>"])
    return "\n".join(lines) + "\n"


def _interactive_html(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return _HTML_TEMPLATE.replace("__EVOLUTION_DATA__", encoded)


def _safe(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "graph"


def _comparison_html(pairs: list[dict[str, Any]]) -> str:
    rows = []
    for pair in pairs:
        files = pair["exported_files"]
        source = (
            f'<div><h3>Task reference</h3><video controls preload="metadata" src="{html.escape(files["task_reference"])}"></video></div>'
            if files.get("task_reference") else ""
        )
        rows.append(
            f'<section><h2>{pair["rank"]}. {html.escape(pair["task_id"])}</h2>'
            f'<p>Reward gain: <strong>{float(pair["reward_gain"]):+.3f}</strong> · '
            f'Seed: {html.escape(str(pair.get("evaluation_seed")))}</p>'
            f'<p>{html.escape(pair.get("prompt", ""))}</p><div class="videos">{source}'
            f'<div><h3>Wan baseline</h3><video controls preload="metadata" src="{html.escape(files["wan_baseline"])}"></video></div>'
            f'<div><h3>Optimized tool path</h3><video controls preload="metadata" src="{html.escape(files["optimized"])}"></video></div>'
            f'</div><pre>{html.escape(json.dumps(pair["metric_deltas"], indent=2, ensure_ascii=False))}</pre></section>'
        )
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Wan vs Evolved Tool Paths</title><style>body{font:14px/1.5 system-ui;margin:24px;color:#17202a}'
        'section{border-top:1px solid #d0d5dd;padding:20px 0}.videos{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}'
        'video{width:100%;max-height:420px;background:#111}pre{background:#f2f4f7;padding:12px;overflow:auto}</style></head><body>'
        '<h1>Wan baseline vs evolved tool paths</h1><p>Pairs are selected automatically by paired task-conditioned reward gain.</p>'
        + "".join(rows)
        + '</body></html>'
    )


def _dot(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tool Chain Graph Evolution</title>
<style>
:root{color-scheme:light;--ink:#17202a;--muted:#667085;--line:#d0d5dd;--paper:#f7f8fa;--blue:#2563eb;--green:#15803d;--red:#b42318;--amber:#b54708}*{box-sizing:border-box}body{margin:0;font:14px/1.45 Inter,ui-sans-serif,system-ui;color:var(--ink);background:var(--paper)}header{height:62px;padding:0 22px;display:flex;align-items:center;gap:18px;border-bottom:1px solid var(--line);background:#fff}h1{font-size:18px;margin:0}header span{color:var(--muted)}main{display:grid;grid-template-columns:minmax(0,1fr) 330px;height:calc(100vh - 62px)}section{min-width:0;display:flex;flex-direction:column}.toolbar{padding:10px 14px;background:#fff;border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:center}.toolbar button,.toolbar select{height:34px;border:1px solid var(--line);background:#fff;padding:0 11px;font:inherit}.toolbar button.active{color:#fff;background:var(--ink);border-color:var(--ink)}#canvas{width:100%;height:100%;background:#fff}.edge{stroke:#98a2b3;stroke-width:1.4;fill:none}.edge.accepted{stroke:var(--green);stroke-width:2.2}.node circle,.node rect{stroke-width:2}.node text{font-size:12px;pointer-events:none}.node{cursor:pointer}.panel{border-left:1px solid var(--line);background:#fff;overflow:auto;padding:18px}.panel h2{font-size:15px;margin:0 0 12px}.stats{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:18px}.stat{border:1px solid var(--line);padding:9px}.stat b{display:block;font-size:18px}.stat span{color:var(--muted);font-size:12px}pre{white-space:pre-wrap;word-break:break-word;background:#f2f4f7;border:1px solid var(--line);padding:10px;font-size:11px}.legend{margin-left:auto;color:var(--muted);font-size:12px}@media(max-width:850px){main{grid-template-columns:1fr}.panel{display:none}}
</style></head>
<body><header><h1>Tool Chain Graph Evolution</h1><span id="subtitle"></span></header>
<main><section><div class="toolbar"><button id="evolutionBtn" class="active">Evolution lineage</button><button id="snapshotsBtn">Snapshot paths</button><button id="auditsBtn">Candidate audits</button><button id="toolsBtn">Weighted tool graph</button><button id="arenaBtn">Tool arena</button><select id="category"></select><span class="legend">点击节点查看完整记录</span></div><svg id="canvas" viewBox="0 0 1100 700"></svg></section><aside class="panel"><h2>Experiment summary</h2><div id="stats" class="stats"></div><h2>Selection</h2><pre id="details">Select a node.</pre></aside></main>
<script>const DATA=__EVOLUTION_DATA__;const svg=document.querySelector('#canvas'),details=document.querySelector('#details'),cat=document.querySelector('#category');const NS='http://www.w3.org/2000/svg';let mode='evolution';
const categories=Object.keys(DATA.weighted_task_graphs||{});cat.innerHTML=categories.map(x=>`<option>${x}</option>`).join('');cat.style.display='none';
document.querySelector('#subtitle').textContent=`${DATA.programs.length} programs · ${DATA.snapshots.length} graph snapshots · ${DATA.executions.length} executions`;
document.querySelector('#stats').innerHTML=[['Programs',DATA.programs.length],['Snapshots',DATA.snapshots.length],['Executions',DATA.executions.length],['Candidate audits',(DATA.candidate_audits||[]).length],['Task classes',categories.length]].map(([k,v])=>`<div class="stat"><b>${v}</b><span>${k}</span></div>`).join('');
function el(name,attrs={}){const n=document.createElementNS(NS,name);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);return n}function clear(){svg.innerHTML=''}function inspect(x){details.textContent=JSON.stringify(x,null,2)}
function evolution(){clear();const programs=DATA.programs.slice().sort((a,b)=>a.iteration-b.iteration);const byIter={};programs.forEach(p=>(byIter[p.iteration]??=[]).push(p));const maxIt=Math.max(0,...programs.map(p=>p.iteration)),maxRows=Math.max(1,...Object.values(byIter).map(x=>x.length));svg.setAttribute('viewBox',`0 0 ${Math.max(1100,220+maxIt*210)} ${Math.max(700,170+maxRows*100)}`);const pos={};Object.entries(byIter).forEach(([it,ps])=>ps.forEach((p,i)=>pos[p.name]={x:100+Number(it)*210,y:100+i*100}));programs.forEach(p=>{if(p.parent&&pos[p.parent]){const a=pos[p.parent],b=pos[p.name],line=el('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,class:`edge ${p.status==='frontier'?'accepted':''}`});svg.append(line)}});programs.forEach(p=>{const q=pos[p.name],g=el('g',{class:'node'}),fill=p.status==='frontier'?'#dcfce7':p.status==='rejected'?'#fee4e2':p.status==='archived'?'#f2f4f7':'#dbeafe',stroke=p.status==='frontier'?'#15803d':p.status==='rejected'?'#b42318':'#2563eb';g.append(el('rect',{x:q.x-72,y:q.y-25,width:144,height:50,rx:5,fill,stroke}));const t=el('text',{x:q.x,y:q.y-3,'text-anchor':'middle'});t.textContent=p.name;g.append(t);const m=el('text',{x:q.x,y:q.y+14,'text-anchor':'middle',fill:'#667085'});m.textContent=p.metrics?`Q ${p.metrics.quality.toFixed(3)} · $ ${p.metrics.estimated_cost.toFixed(2)}`:p.status;g.append(m);g.onclick=()=>inspect(p);svg.append(g)})}
function snapshots(){clear();const rows=DATA.snapshots,toolCount=Math.max(1,...rows.map(x=>(x.event.tools||[]).length));svg.setAttribute('viewBox',`0 0 ${Math.max(1100,300+toolCount*170)} ${Math.max(700,100+rows.length*92)}`);rows.forEach((item,row)=>{const y=70+row*92,e=item.event,tools=e.tools||[],color=e.status==='accepted'||e.status==='promoted'||e.status==='frontier'?'#15803d':e.status==='rejected'||e.status==='dominated'?'#b42318':'#2563eb';const label=el('text',{x:18,y:y-10,fill:color});label.textContent=`${e.sequence}. ${e.stage} · ${e.status}`;svg.append(label);const sub=el('text',{x:18,y:y+10,fill:'#667085'});sub.textContent=`iter ${e.iteration} · ${e.skill_name}`;svg.append(sub);tools.forEach((tool,i)=>{const x=290+i*170;if(i){svg.append(el('line',{x1:x-98,y1:y,x2:x-72,y2:y,class:'edge'}))}const g=el('g',{class:'node'});g.append(el('rect',{x:x-72,y:y-22,width:144,height:44,rx:5,fill:'#fff',stroke:color}));const t=el('text',{x,y:y+4,'text-anchor':'middle'});t.textContent=tool.length>20?tool.slice(0,18)+'…':tool;g.append(t);g.onclick=()=>inspect({event:e,tool,graph:item.graph});svg.append(g)});if(!tools.length){const empty=el('text',{x:290,y:y+4,fill:'#667085'});empty.textContent='No executable tool nodes';svg.append(empty)}})}
function audits(){clear();const rows=DATA.candidate_audits||[];svg.setAttribute('viewBox',`0 0 1100 ${Math.max(700,100+rows.length*72)}`);if(!rows.length){inspect({message:'No candidate preflight audits were recorded.'});return}rows.forEach((item,row)=>{const y=55+row*72,status=item.status||item.stage,color=['executable','ready','selected_for_realization','screened'].includes(status)?'#15803d':status==='deferred'?'#b54708':'#b42318',g=el('g',{class:'node'});g.append(el('rect',{x:24,y:y-22,width:1030,height:48,rx:5,fill:'#fff',stroke:color}));const title=el('text',{x:42,y:y-3,fill:color});title.textContent=`iter ${item.iteration??'-'} · ${item.graph_id||item.candidate_name||item.stage} · ${status}`;g.append(title);const sub=el('text',{x:42,y:y+16,fill:'#667085'});sub.textContent=(item.rejection_reason||item.reason||item.mechanism_family||'').slice(0,140);g.append(sub);g.onclick=()=>inspect(item);svg.append(g)})}
function tools(){clear();svg.setAttribute('viewBox','0 0 1100 700');const name=cat.value||categories[0],d=DATA.weighted_task_graphs[name]||{},nodes=(d.important_nodes||[]).slice(),edges=(d.important_edges||[]).filter(e=>nodes.some(n=>n.tool===e.source)&&nodes.some(n=>n.tool===e.target));if(!nodes.length){inspect({task_class:name,message:'No weighted tools recorded.'});return}const cx=550,cy=340,r=Math.min(260,45*nodes.length),pos={};nodes.forEach((n,i)=>pos[n.tool]={x:cx+r*Math.cos(2*Math.PI*i/nodes.length),y:cy+r*Math.sin(2*Math.PI*i/nodes.length)});edges.forEach(e=>{const a=pos[e.source],b=pos[e.target];if(a&&b)svg.append(el('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,class:'edge'}))});nodes.forEach(n=>{const q=pos[n.tool],g=el('g',{class:'node'}),size=18+Math.min(24,Math.log1p(n.uses||1)*7);g.append(el('circle',{cx:q.x,cy:q.y,r:size,fill:'#e0f2fe',stroke:'#0369a1'}));const t=el('text',{x:q.x,y:q.y+size+17,'text-anchor':'middle'});t.textContent=n.tool;g.append(t);g.onclick=()=>inspect(n);svg.append(g)});inspect({task_class:name,total_rollouts:d.total_rollouts,baseline_quality:d.baseline_quality,top_paths:d.top_paths})}
function arena(){clear();const name=cat.value||categories[0],arenas=((DATA.weighted_task_graphs[name]||{}).tool_arenas)||{},rows=Object.entries(arenas);svg.setAttribute('viewBox',`0 0 1100 ${Math.max(700,120+rows.reduce((n,[,a])=>n+a.rankings.length*68+54,0))}`);if(!rows.length){inspect({task_class:name,message:'No multi-repository tool arena has evaluation evidence yet.'});return}let y=48;rows.forEach(([cap,a])=>{const title=el('text',{x:28,y,fill:'#17202a'});title.textContent=`${cap} · winner: ${a.winner||'pending'}`;svg.append(title);y+=34;a.rankings.forEach((item,index)=>{const winner=item.tool===a.winner,g=el('g',{class:'node'});g.append(el('rect',{x:28,y:y-22,width:1020,height:48,rx:5,fill:winner?'#dcfce7':'#fff',stroke:winner?'#15803d':'#98a2b3'}));const t=el('text',{x:46,y:y-3,fill:winner?'#15803d':'#17202a'});t.textContent=`${index+1}. ${item.tool}`;g.append(t);const s=el('text',{x:380,y:y-3,fill:'#667085'});s.textContent=item.evaluated?`utility ${item.arena_utility.toFixed(3)} · quality ${item.mean_quality.toFixed(3)} · uses ${item.uses}`:'not evaluated';g.append(s);g.onclick=()=>inspect({task_class:name,capability:cap,...item,comparison_protocol:a.comparison_protocol});svg.append(g);y+=68});y+=24})}
function setMode(next){mode=next;document.querySelector('#evolutionBtn').classList.toggle('active',next==='evolution');document.querySelector('#snapshotsBtn').classList.toggle('active',next==='snapshots');document.querySelector('#auditsBtn').classList.toggle('active',next==='audits');document.querySelector('#toolsBtn').classList.toggle('active',next==='tools');document.querySelector('#arenaBtn').classList.toggle('active',next==='arena');cat.style.display=next==='tools'||next==='arena'?'block':'none';next==='arena'?arena():next==='tools'?tools():next==='audits'?audits():next==='snapshots'?snapshots():evolution()}document.querySelector('#evolutionBtn').onclick=()=>setMode('evolution');document.querySelector('#snapshotsBtn').onclick=()=>setMode('snapshots');document.querySelector('#auditsBtn').onclick=()=>setMode('audits');document.querySelector('#toolsBtn').onclick=()=>setMode('tools');document.querySelector('#arenaBtn').onclick=()=>setMode('arena');cat.onchange=()=>mode==='arena'?arena():tools();evolution();
</script></body></html>'''
