#!/usr/bin/env python3
"""Read-only, offline audit of saved automatic-review responses. No model calls."""
import argparse
import hashlib
import json
from pathlib import Path
from evovideo_skill.verifier_review import review_rules, validate_checks, ReviewEvidenceConflict, VERSION


def audit(probe_dir):
    root = Path(probe_dir)
    plan = json.loads((root / 'group_protocol.json').read_text())
    source = Path(plan['source_judgment']) / f"group-{plan['group']:03d}-repeat-{plan.get('source_repeat', 0)}.request-0.json"
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != plan['source_request_sha256']:
        raise ValueError('source request has changed; cannot audit against a different task')
    request = json.loads(data)
    criteria = review_rules(plan['criteria'], request.get('original_task', {}))
    records = []
    for path in sorted((root / 'review' / 'auto_review').glob('*/confirmation-*.raw.json')):
        item = {'file': str(path)}
        try:
            raw = json.loads(path.read_text())
            attempt = 1 if '.correction-1.' in path.name else 0
            actual_path = path.with_name(path.name.split('.')[0] + f'.request-{attempt}.json')
            actual = json.loads(actual_path.read_text()) if actual_path.exists() else {}
            from evovideo_skill.review_observations import VERSION as OBSERVATION_PROTOCOL
            from evovideo_skill.owned_observations import VERSION as OWNED_VERSION
            wire_protocol = actual.get('output_contract', {}).get('response_protocol')
            shared = wire_protocol in {OBSERVATION_PROTOCOL, OWNED_VERSION}
            if shared:
                from evovideo_skill.review_observations import compile_plan, project_response, expand_field_patch
                from evovideo_skill.scoped_judgment import project
                from evovideo_skill.conditioning_verifier import parse_judgment
                from evovideo_skill.verifier_identity import assess, contract
                if actual.get('original_task') != request.get('original_task'):
                    raise ValueError('review source task differs from original request')
                manifest = actual['evidence_manifest']
                raw = expand_field_patch(raw, actual)
                spans = manifest['windows']
                compiled = compile_plan(actual['criteria'], actual['original_task'], spans, manifest, protocol=wire_protocol)
                projected = project_response(raw, compiled, manifest)
                audited_facts = {key: fact for row in projected['criteria'].values()
                                 for key, fact in row.get('fact_observations', {}).items()}
                item['unknown_facts'] = sorted(key for key, fact in audited_facts.items() if fact['value'] == 'unknown')
                item['uncertainty_context_facts'] = sorted(key for key, fact in audited_facts.items()
                                                          if fact.get('uncertainty_context'))
                parse_judgment(project(projected, compiled['rules'], manifest), compiled['rules'], spans, manifest)
                identity = assess(raw, contract(actual['original_task'], compiled['rules'], manifest))
                if identity['blocked_criteria']:
                    raise ValueError('shared observation identity gate is unresolved')
            else:
                validate_checks(raw, criteria)
        except ReviewEvidenceConflict as exc:
            item.update(status='evidence_conflict', issues=exc.issues, format_issues=exc.format_issues)
        except (ValueError, KeyError, TypeError) as exc:
            item.update(status=getattr(exc, 'category', 'invalid_response'), error=str(exc))
        else:
            item.update(status='shared_observations_consistent' if shared else 'atomic_checks_consistent',
                qualification=('Projection, citation and identity contract checks only; not visual validation.'
                    if shared else 'Atomic checks only; no full citation, identity, fact or visual validation.'))
        records.append(item)
    if not records:
        raise ValueError('no saved confirmation raw responses found in this probe directory')
    return {'purpose': 'offline_atomic_evidence_audit_not_visual_validation', 'protocol': VERSION,
            'model_calls': 0, 'responses': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.probe_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
