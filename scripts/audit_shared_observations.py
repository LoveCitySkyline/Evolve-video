#!/usr/bin/env python3
"""Revalidate saved shared-observation responses offline; never call a model or update scores."""
import argparse
import hashlib
import json
from pathlib import Path

from evovideo_skill.conditioning_verifier import VERIFIER_PROTOCOL_VERSION, parse_judgment
from evovideo_skill.review_observations import VERSION, compile_plan, project_response, _validate_observation, expand_field_patch
from evovideo_skill.scoped_judgment import project
from evovideo_skill.owned_observations import VERSION as OWNED_VERSION, expand_observations
from evovideo_skill.verifier_identity import assess, contract


def audit(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError('judgment directory does not exist: ' + str(root))
    records = []
    for path in sorted(root.glob('auto_review/*/confirmation-*.raw.json')):
        attempt = 1 if '.correction-1.' in path.name else 0
        request_path = path.with_name(path.name.split('.')[0] + f'.request-{attempt}.json')
        record = {'file': str(path), 'request': str(request_path)}
        records.append(record)
        try:
            data, request_data = path.read_bytes(), request_path.read_bytes()
            raw, request = json.loads(data), json.loads(request_data)
            record.update(raw_sha256=hashlib.sha256(data).hexdigest(),
                          request_sha256=hashlib.sha256(request_data).hexdigest())
            protocol = request.get('output_contract', {}).get('response_protocol')
            if protocol not in {VERSION, OWNED_VERSION}:
                record.update(status='unsupported_protocol')
                continue
            manifest = request['evidence_manifest']
            raw = expand_field_patch(raw, request)
            spans = manifest['windows']
            plan = compile_plan(request['criteria'], request['original_task'], spans, manifest, protocol=protocol)
            validation_raw = expand_observations(raw, plan) if protocol == OWNED_VERSION else raw
            fact_errors = []
            for key, definition in plan['definitions'].items():
                try:
                    _validate_observation(key, definition, validation_raw['facts'][key], plan, manifest)
                except (ValueError, KeyError, TypeError) as exc:
                    fact_errors.append({'fact': key, 'error': str(exc)})
            record['fact_errors'] = fact_errors
            projected = project(project_response(raw, plan, manifest), plan['rules'], manifest)
            parsed = parse_judgment(projected, plan['rules'], spans, manifest)
            identity = assess(raw, contract(request['original_task'], plan['rules'], manifest))
            if identity['blocked_criteria']:
                raise ValueError('identity gate unresolved: ' + str(identity['blocked_criteria']))
            record.update(status='contract_valid', unknown_criteria=[
                key for key, row in parsed.items() if row['status'] == 'unobserved'])
        except (ValueError, KeyError, TypeError, OSError) as exc:
            record.update(status=getattr(exc, 'category', 'invalid_response'), error=str(exc))
    if not records:
        raise ValueError('no auto_review/*/confirmation-*.raw.json found')
    return {'purpose': 'offline_contract_regression_not_visual_validation',
            'protocol': VERIFIER_PROTOCOL_VERSION, 'model_calls': 0,
            'qualification': 'Each saved response/authorized missing-field patch is checked independently. No score selection or publication, '
                'media-integrity verification or visual-truth validation is performed.',
            'responses': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--judgment-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.judgment_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
