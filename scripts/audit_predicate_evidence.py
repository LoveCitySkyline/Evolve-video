#!/usr/bin/env python3
"""Replay saved owned responses through the current guard, without API/media writes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from evovideo_skill.owned_observations import VERSION, SUPPORTED_VERSIONS
from evovideo_skill.isolated_observations import VERSION as ISOLATED_VERSION
from evovideo_skill.predicate_evidence import VERSION as GUARD_VERSION, QUALIFICATION, eligible
from evovideo_skill.review_observations import compile_plan, project_response
from evovideo_skill.scoped_judgment import project
from evovideo_skill.conditioning_verifier import parse_judgment
from evovideo_skill.verifier_identity import assess, contract


def audit(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError('probe directory does not exist: ' + str(root))
    records = []
    for path in sorted(root.rglob('*.raw.json')):
        attempt = 1 if '.correction-1.' in path.name else 0
        request_path = path.with_name(path.name.split('.')[0] + f'.request-{attempt}.json')
        if not request_path.exists():
            continue
        record = {'response': str(path.relative_to(root))}
        try:
            req_bytes, raw_bytes = request_path.read_bytes(), path.read_bytes()
            request, raw = json.loads(req_bytes), json.loads(raw_bytes)
            old = request.get('output_contract', {}).get('response_protocol')
            if old not in SUPPORTED_VERSIONS:
                continue
            record.update(source_protocol=old, request_sha256=hashlib.sha256(req_bytes).hexdigest(),
                          response_sha256=hashlib.sha256(raw_bytes).hexdigest())
            manifest, public = request['evidence_manifest'], request['original_task']
            plan = compile_plan(request['criteria'], public, manifest['windows'], manifest,
                protocol=old if old == ISOLATED_VERSION else VERSION)
            record['guard_scope'] = {
                'eligible_facts': [key for key, definition in plan['definitions'].items() if eligible(definition)],
                'excluded_contextual_facts': [key for key, definition in plan['definitions'].items() if not eligible(definition)],
                'qualification': 'Contextual facts retain normal validation; this narrow guard does not grade them.'}
            normalized = project_response(raw, plan, manifest)
            rows = parse_judgment(project(normalized, plan['rules'], manifest),
                                  plan['rules'], manifest['windows'], manifest)
            identity = assess(raw, contract(public, plan['rules'], manifest))
            held = {key: fact['predicate_evidence_audit'] for row in rows.values()
                    for key, fact in row.get('fact_observations', {}).items()
                    if fact.get('predicate_evidence_audit')}
            record.update(status='identity_unresolved' if identity['blocked_criteria'] else
                          'claims_withheld' if held else 'no_pattern_detected',
                          withheld_facts=held,
                          unknown_criteria=[name for name, row in rows.items() if row['status'] == 'unobserved'],
                          identity_blocked_criteria=identity['blocked_criteria'])
        except (ValueError, KeyError, TypeError, OSError) as exc:
            record.update(status='invalid_saved_attempt', error=str(exc))
        records.append(record)
    if not records:
        raise ValueError('no saved owned responses with matching requests found; no checks performed')
    return {'purpose': 'offline_rationale_guard_replay_not_visual_validation',
            'protocol': GUARD_VERSION, 'observation_protocol': VERSION, 'model_calls': 0,
            'attempt_counts': dict(Counter(r['status'] for r in records)), 'records': records,
            'qualification': QUALIFICATION + ' Old scores are not modified. Attempts are not independent videos.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.probe_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
