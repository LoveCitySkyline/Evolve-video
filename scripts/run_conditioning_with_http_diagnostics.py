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
from evovideo_skill.provider_diagnostics import BODY_LIMIT, redact, error_details

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
