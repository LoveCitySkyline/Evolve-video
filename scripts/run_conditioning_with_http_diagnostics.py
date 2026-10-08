#!/usr/bin/env python3
"""Resume the normal runner with bounded, redacted verifier error diagnostics.

Only observes terminal request failures. Does not change requests, retries,
scores, caches, or protocol hashes, and does not make extra API calls.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
BODY_LIMIT = 16384


def redact(value):
    text = str(value)
    for name, secret in os.environ.items():
        if secret and any(word in name.upper() for word in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')):
            text = text.replace(secret, '[REDACTED]')
    text = re.sub(r'(?i)Bearer\s+\S+', 'Bearer [REDACTED]', text)
    text = re.sub(r'(?i)data:[^\s\"\']+', '[MEDIA REDACTED]', text)
    text = re.sub(r'https?://[^\s\"\']+', '[URL REDACTED]', text)
    text = re.sub(r'\b[A-Za-z0-9+/=_-]{96,}\b', '[LONG VALUE REDACTED]', text)
    return text[:2000]


def error_details(exc):
    """Read only whitelisted provider fields; never dump HTML or request bodies."""
    seen = set()
    cause = exc
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, urllib.error.HTTPError):
            row = {'error_type': type(cause).__name__, 'http_code': cause.code}
            try:
                body = cause.read(BODY_LIMIT + 1)
                row['response_truncated'] = len(body) > BODY_LIMIT
                if row['response_truncated']:
                    row['body_status'] = 'too_large_to_parse_safely'
                    return row
                payload = json.loads(body)
            except (ValueError, OSError, TypeError):
                row['body_status'] = 'empty_unreadable_or_non_json'
                return row
            if not isinstance(payload, dict):
                row['body_status'] = 'non_object_json'
                return row
            detail = payload.get('error', payload)
            if isinstance(detail, dict):
                row['provider_error'] = {k: redact(detail[k]) for k in
                    ('code', 'message', 'type', 'param', 'status')
                    if k in detail and isinstance(detail[k], (str, int))}
            # Request identifiers help the provider locate the failed request.
            for key in ('request_id', 'requestId'):
                if isinstance(payload.get(key), str):
                    row[key] = redact(payload[key])
            row['body_status'] = 'json_fields_only'
            return row
        if isinstance(cause, urllib.error.URLError):
            return {'error_type': type(cause).__name__, 'http_code': None,
                    'reason': redact(cause.reason)}
        cause = cause.__cause__ or cause.__context__
    return {'error_type': type(exc).__name__, 'http_code': None,
            'body_status': 'no_http_or_url_cause'}


@contextmanager
def diagnose_requests(verifier_class, destination):
    original = verifier_class.request

    def request(self, prompt, evidence, operation):
        try:
            return original(self, prompt, evidence, operation)
        except Exception as exc:
            # Diagnostics must never replace or swallow the runner's exception.
            try:
                row = {'time_utc': datetime.now(timezone.utc).isoformat(),
                       'operation': redact(operation), 'model': redact(self.model),
                       **error_details(exc)}
                print('[verifier HTTP diagnostic] ' + json.dumps(row, ensure_ascii=False),
                      file=sys.stderr, flush=True)
                with destination.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            except Exception:
                print('[verifier HTTP diagnostic] Unable to save diagnostics; '
                      'original request failure preserved.', file=sys.stderr, flush=True)
            raise

    verifier_class.request = request
    try:
        yield
    finally:
        verifier_class.request = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--output-dir', required=True)
    args, _ = parser.parse_known_args(argv)
    root = Path(args.output_dir).resolve()
    if not root.is_dir() or not (root / 'checkpoint.json').is_file():
        parser.error('diagnostic resume requires an existing run directory with checkpoint.json')
    forwarded = list(sys.argv[1:] if argv is None else argv)
    if '--continue' not in forwarded:
        parser.error('pass --continue with the SAME original configuration')
    from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier
    from evovideo_skill.conditioning_runner import main as run
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *forwarded]
        with diagnose_requests(ConditioningVideoVerifier, root / 'http_diagnostics.jsonl'):
            run()
    finally:
        sys.argv = old_argv


if __name__ == '__main__':
    main()
