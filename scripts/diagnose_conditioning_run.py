"""Summarize saved planner/verifier failures without calling models or changing a run."""
import argparse
from collections import Counter
import json
from pathlib import Path


def graph_summary(value):
    if not isinstance(value, dict):
        return {"invalid_graph_type": type(value).__name__}
    nodes = [n for n in value.get("nodes", []) if isinstance(n, dict)]
    edges = [e for e in value.get("edges", []) if isinstance(e, dict)]
    sources = {e.get("source") for e in edges}
    return {"node_types": dict(Counter(n.get("node_type") for n in nodes)),
            "terminal_nodes": [{k: n.get(k) for k in ("node_id", "node_type", "name")}
                               for n in nodes if n.get("node_id") not in sources]}


def proposal_summary(raw):
    if not isinstance(raw, dict):
        return {"invalid_proposal_type": type(raw).__name__}
    result = {"top_level_keys": list(raw)}
    if "anchor" in raw:
        result["anchor"] = graph_summary(raw["anchor"])
    if "graph" in raw:
        result["graph"] = graph_summary(raw["graph"])
    factors = raw.get("factors", [])
    if isinstance(factors, list):
        result["factors"] = [{"id": item.get("id"), "graph": graph_summary(item.get("graph")),
                              "strategy_keys": list(item["strategy"]) if isinstance(item.get("strategy"), dict)
                              else {"invalid_type": type(item.get("strategy")).__name__}}
                             for item in factors if isinstance(item, dict)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir
    if not root.is_dir():
        parser.error(f"run directory not found: {root}")
    output = {"run_dir": str(root.resolve()), "proposals": [], "verifier_responses": []}
    for path in sorted((root / "proposals").glob("*.json")):
        data = json.loads(path.read_text())
        output["proposals"].append({"file": str(path), "operation": data.get("operation"),
            "status": data.get("status"), "error": data.get("error"),
            "validation_failures": [{"error": failure.get("error"), "proposal": proposal_summary(failure.get("raw"))}
                                    for failure in data.get("validation_failures", [])],
            "final_proposal": proposal_summary(data.get("raw")),
            "invalid_factors": data.get("active_selection", {}).get("invalid_factors"),
            "rejected_pairs": data.get("active_selection", {}).get("audit", {}).get("rejected_pairs")})
    for folder in sorted((root / "verifier" / "runtime" / "judgments").glob("*")):
        if not folder.is_dir() or (folder / "result.json").exists():
            continue
        for path in sorted(folder.glob("*.raw.json")):
            data = json.loads(path.read_text())
            criteria = data.get("criteria") if isinstance(data, dict) else None
            output["verifier_responses"].append({"file": str(path),
                "top_level_keys": list(data) if isinstance(data, dict) else None,
                "criterion_keys": list(criteria) if isinstance(criteria, dict) else None})
        for path in sorted(folder.glob("*.format-*.json")):
            output["verifier_responses"].append({"file": str(path), **json.loads(path.read_text())})
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
