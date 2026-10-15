#!/usr/bin/env python3
"""Batch replay saved verifier responses against current validators; zero model calls.

This checks JSON, citations, source identity bindings and internal consistency.
It cannot establish visual truth or validate that a provider saw the media.
Raw files and saved runs are never rewritten. Legacy failures are migration
findings, not claims that a historical run used the current protocol.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from evovideo_skill.conditioning_verifier import parse_judgment, VERIFIER_PROTOCOL_VERSION, normalize_criterion_envelope
from evovideo_skill.review_observations import compile_plan, project_response, VERSION, expand_field_patch, CreditPatchAbstention
from evovideo_skill.scoped_judgment import is_scoped, project
from evovideo_skill.verifier_facts import with_fact_contract, fact_conflicts
from evovideo_skill.verifier_identity import contract, assess
from evovideo_skill.owned_observations import SUPPORTED_VERSIONS
from evovideo_skill.verifier_review import validate_checks


def replay(request, raw, require_review=False):
    try:
        raw = expand_field_patch(raw, request)
    except CreditPatchAbstention as exc:
        return {'status': 'score_unresolved', 'error': str(exc),
                'unknown_criteria': sorted(request['criteria']), 'criteria': sorted(request['criteria'])}
    rules = with_fact_contract(request['criteria'], request.get('original_task', {}))
    manifest = request['evidence_manifest']
    spans = manifest['windows']
    protocol = request.get('output_contract', {}).get('response_protocol')
    from evovideo_skill import observation_identity, isolated_observations
    identity_projection = protocol == isolated_observations.VERSION
    identity_spec = None
    if protocol in {VERSION, *SUPPORTED_VERSIONS}:
        plan = compile_plan(rules, request.get('original_task', {}), spans, manifest, protocol=protocol)
        normalized = project_response(raw, plan, manifest)
        rules = plan['rules']
        if identity_projection:
            identity_spec = observation_identity.contract(request.get('original_task', {}), plan, manifest)
            identity_audit = assess(raw, identity_spec)
            if identity_spec and identity_audit['actor_issues']:
                masked = observation_identity.mask(raw, plan, identity_spec, identity_audit)
                normalized = project_response(masked, plan, manifest)
                observation_identity.withhold_scores(normalized, plan, identity_audit)
    elif protocol and protocol.startswith('shared-observation-'):
        raise ValueError('unsupported saved shared observation protocol: ' + protocol)
    else:
        normalized = raw if is_scoped(rules) else normalize_criterion_envelope(raw, rules)
        values = normalized.get('criteria', {}) if isinstance(normalized, dict) else {}
        if (require_review or request.get('atomic_review') or
                any('atomic_checks' in f.get('required', []) for f in
                    request.get('output_contract', {}).get('fields', {}).values()) or
                (isinstance(values, dict) and any(isinstance(r, dict) and 'atomic_checks' in r for r in values.values()))):
            normalized = validate_checks(normalized, rules)
    if is_scoped(rules):
        normalized = project(normalized, rules, manifest)
    # identity_bindings is an input gate, not a criterion.
    if isinstance(normalized, dict) and 'identity_bindings' in normalized:
        normalized = {k: v for k, v in normalized.items() if k != 'identity_bindings'}
    rows = parse_judgment(normalized, rules, spans, manifest)
    identity_required = (bool(request.get('identity_contract')) or protocol in {VERSION, *SUPPORTED_VERSIONS})
    identity = (identity_audit if identity_projection else
                assess(raw, contract(request.get('original_task', {}), rules, manifest) if identity_required else None))
    conflicts = fact_conflicts({k: [v] for k, v in rows.items()})
    unknown = sorted(k for k, v in rows.items() if v['status'] == 'unobserved')
    if identity['blocked_criteria']:
        state = 'identity_unresolved'
    elif conflicts:
        state = 'evidence_conflict'
    elif unknown:
        state = 'evidence_unresolved'
    else:
        state = 'contract_valid'
    legacy_ambiguous = sorted(k for k, v in raw.get('facts', {}).items()
        if isinstance(v, dict) and v.get('basis') == 'not_visible') if identity_projection else []
    return {'status': state, 'unknown_criteria': unknown,
            'legacy_ambiguous_questions': legacy_ambiguous,
            'identity_check': 'required' if identity_required else 'not_requested_by_saved_contract',
            'identity_blocked_criteria': identity['blocked_criteria'],
            'conflicting_fact_ids': sorted(conflicts), 'criteria': sorted(rows)}


def audit(roots):
    records, seen = [], set()
    for root in map(Path, roots):
        if not root.is_dir():
            raise ValueError('run directory does not exist: ' + str(root))
        for path in sorted(root.rglob('*.request-*.json')):
            match = re.fullmatch(r'(.+)\.request-([01])\.json', path.name)
            if not match or path.resolve() in seen:
                continue
            seen.add(path.resolve())
            raw_path = path.with_name(match[1] + ('.raw.json' if match[2] == '0' else '.correction-1.raw.json'))
            record = {'request': str(path), 'response': str(raw_path), 'attempt': int(match[2])}
            try:
                data = path.read_bytes()
                request = json.loads(data)
                # Ignore non-verifier request records in a pipeline output tree.
                if not isinstance(request, dict) or not {'criteria', 'evidence_manifest'} <= request.keys():
                    continue
                record['request_sha256'] = hashlib.sha256(data).hexdigest()
                record['saved_protocol'] = request.get('output_contract', {}).get('response_protocol', 'legacy')
                record['stage'] = 'automatic_review' if path.name.startswith('confirmation-') else 'initial'
                if not raw_path.is_file():
                    record.update(status='missing_response', error='No saved response for this request; no retry performed.')
                else:
                    raw_data = raw_path.read_bytes()
                    record['response_sha256'] = hashlib.sha256(raw_data).hexdigest()
                    record.update(replay(request, json.loads(raw_data), require_review=record['stage'] == 'automatic_review'))
            except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
                record.update(status='invalid_response', error=str(exc))
            records.append(record)
    return {'purpose': 'offline_current_protocol_replay_not_visual_validation',
            'verifier_protocol': VERIFIER_PROTOCOL_VERSION, 'model_calls': 0,
            'summary': dict(sorted(Counter(r['status'] for r in records).items())),
            'requests': len(records), 'responses': records,
            'qualification': 'Every saved attempt is reported, including superseded failures. '
                'This is not a final-run acceptance rate. No source files, scores or media were changed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, action='append', required=True,
                        help='May be repeated to audit multiple saved runs in one pass.')
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
