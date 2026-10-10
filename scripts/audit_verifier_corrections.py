"""Offline audit of saved scoped responses against the current protocol; no API calls."""
import argparse
import json
from pathlib import Path
import re

from evovideo_skill.conditioning_verifier import VERIFIER_PROTOCOL_VERSION, parse_judgment
from evovideo_skill.scoped_judgment import is_scoped, project
from evovideo_skill.verifier_facts import with_fact_contract, correction_semantic_changes


def audit_response(request, raw, correction=None):
    criteria = with_fact_contract(request['criteria'], request.get('original_task', {}))
    if not is_scoped(criteria):
        return {'skipped': 'This audit covers fixed-window criteria only.'}
    manifest = request['evidence_manifest']
    # Saved requests include the actual host evidence catalog. Do not synthesize
    # frames, substitute citations, or use this audit as visual ground truth.
    spans = manifest.get('windows', [])
    rows = raw.get('criteria') if isinstance(raw, dict) else None
    errors, scores, changes, categories = {}, {}, {}, {}
    if not isinstance(rows, dict) or set(rows) != set(criteria) or set(raw) != {'criteria'}:
        return {'original_valid': False, 'errors': {'response': 'Requested criterion keys differ from response.'},
            'error_categories': {'response': 'response_format'}}
    for name, rule in criteria.items():
        try:
            projected = project({'criteria': {name: rows[name]}}, {name: rule}, manifest)
            parsed = parse_judgment(projected, {name: rule}, spans, manifest)
            scores[name] = parsed[name]['score']
        except (ValueError, KeyError, TypeError) as exc:
            errors[name] = str(exc)
            categories[name] = getattr(exc, 'category', 'response_format')
        corrected_rows = correction.get('criteria') if isinstance(correction, dict) else None
        if rule.get('fact_contract') and isinstance(corrected_rows, dict) and name in corrected_rows:
            drift = correction_semantic_changes(rows[name], corrected_rows[name])
            if drift:
                changes[name] = drift
    return {'original_valid': not errors, 'original_scores': scores, 'errors': errors,
        'error_categories': categories,
        'correction_semantic_changes': changes,
        'qualification': 'Original scores are parsed model claims, not verified truth. '
            'If the original response is valid, its old format correction is unnecessary. '
            'Changed judgments require separate bounded review when correction is needed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--judgment-dir', required=True, type=Path)
    parser.add_argument('--group', type=int, action='append', help='Repeat to audit multiple groups; default: all scoped groups')
    args = parser.parse_args()
    reports = []
    for path in sorted(args.judgment_dir.glob('group-*-repeat-*.raw.json')):
        match = re.fullmatch(r'group-(\d+)-repeat-(\d+)\.raw\.json', path.name)
        if not match or args.group is not None and int(match[1]) not in args.group:
            continue
        stem = path.name.removesuffix('.raw.json')
        request_path = path.with_name(stem + '.request-0.json')
        correction_path = path.with_name(stem + '.correction-1.raw.json')
        try:
            report = audit_response(json.loads(request_path.read_text()), json.loads(path.read_text()),
                json.loads(correction_path.read_text()) if correction_path.exists() else None)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report = {'audit_error': str(exc)}
        reports.append({'group': int(match[1]), 'repeat': int(match[2]), 'source': str(path), **report})
    if not reports:
        parser.error('No matching saved group responses; check --judgment-dir and --group.')
    print(json.dumps({'purpose': 'offline_protocol_audit_only_not_visual_recheck',
        'protocol': VERIFIER_PROTOCOL_VERSION, 'model_calls': 0,
        'summary': {'original_valid_groups': sum(r.get('original_valid') is True for r in reports),
            'original_invalid_groups': sum(r.get('original_valid') is False for r in reports),
            'audit_errors': sum('audit_error' in r for r in reports),
            'criteria_errors_by_category': {category: sum(list(r.get('error_categories', {}).values()).count(category)
                for r in reports) for category in ('citation_contract', 'semantic_conflict', 'response_format')},
            'criteria_with_correction_changes': sum(len(r.get('correction_semantic_changes', {})) for r in reports)},
        'qualification': 'Uses saved request manifests; does not read images, verify media bytes or change old results.',
        'reports': reports}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
