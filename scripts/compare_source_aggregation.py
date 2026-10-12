#!/usr/bin/env python3
"""Compare saved v35 wire arithmetic with v36 offline; never publish new scores."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from evovideo_skill.owned_observations import VERSION, SUPPORTED_VERSIONS
from evovideo_skill.review_observations import compile_plan, project_response
from evovideo_skill.scoped_judgment import project
from evovideo_skill.conditioning_verifier import parse_judgment
from evovideo_skill.verifier_identity import contract, assess
from evovideo_skill.verifier_facts import fact_conflicts


def compare(request, raw):
    old_protocol = request.get('output_contract', {}).get('response_protocol')
    if old_protocol not in SUPPORTED_VERSIONS:
        return {'status': 'skipped_non_owned_wire'}
    manifest, public = request['evidence_manifest'], request['original_task']
    spans = manifest['windows']
    projected, audits = [], {}
    for protocol in (old_protocol, VERSION):
        plan = compile_plan(request['criteria'], public, spans, manifest, protocol=protocol)
        wire, audit = project_response(raw, plan, manifest, with_audit=True)
        rows = parse_judgment(project(wire, plan['rules'], manifest), plan['rules'], spans, manifest)
        identity = assess(raw, contract(public, plan['rules'], manifest))
        if identity['blocked_criteria'] or fact_conflicts({k: [v] for k, v in rows.items()}):
            return {'status': 'evidence_gate_unresolved', 'changes': {}}
        projected.append(rows)
        audits = audit
    before, after = projected
    changes = {k: {'before': {'status': before[k]['status'], 'score': before[k]['score']},
                   'after': {'status': v['status'], 'score': v['score']},
                   'aggregation_bounds': audits.get(k)}
               for k, v in after.items() if (v['status'], v['score']) != (before[k]['status'], before[k]['score'])}
    return {'status': 'algebra_changes' if changes else 'same_algebra', 'changes': changes,
            'remaining_unknown_criteria': [k for k, v in after.items() if v['status'] == 'unobserved']}


def run(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError('run directory does not exist')
    records = []
    for path in sorted((root / 'verifier').glob('*/judgments/**/*.request-*.json')):
        match = re.fullmatch(r'(.+)\.request-([01])\.json', path.name)
        if not match:
            continue
        response = path.with_name(match[1] + ('.raw.json' if match[2] == '0' else '.correction-1.raw.json'))
        row = {'request': str(path.relative_to(root)), 'response': str(response.relative_to(root))}
        try:
            req_bytes, raw_bytes = path.read_bytes(), response.read_bytes()
            result = compare(json.loads(req_bytes), json.loads(raw_bytes))
            if result['status'] == 'skipped_non_owned_wire':
                continue
            row.update(result, request_sha256=hashlib.sha256(req_bytes).hexdigest(),
                       response_sha256=hashlib.sha256(raw_bytes).hexdigest())
        except (ValueError, KeyError, TypeError, OSError) as exc:
            row.update(status='invalid_saved_attempt', error=str(exc))
        records.append(row)
    return {'purpose': 'offline_counterfactual_algebra_not_visual_validation', 'model_calls': 0,
            'current_observation_protocol': VERSION, 'attempt_counts': dict(Counter(r['status'] for r in records)),
            'records': records,
            'qualification': 'Saved attempts are not independent evaluations. No media/model calls or file writes. '
                'Old model claims remain unverified, including possible predicate confusion. Algebraic changes '
                'are not new valid quality scores and must not be written into old evaluations.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
