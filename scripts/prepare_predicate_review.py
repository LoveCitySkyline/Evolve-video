#!/usr/bin/env python3
"""One command to audit, deduplicate and package saved predicate evidence offline."""
import argparse
import json
from pathlib import Path
import shlex
import tempfile
from audit_predicate_evidence import audit
from evovideo_skill.predicate_review_pack import export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, help='New/empty sibling directory; allocated automatically if omitted')
    args = parser.parse_args()
    try:
        result_audit = audit(args.run_dir)
        output = args.output_dir or Path(tempfile.mkdtemp(
            prefix=args.run_dir.name + '_predicate_review_', dir=args.run_dir.resolve().parent))
        result = export(args.run_dir, output, result_audit)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    command = 'PYTHONPATH=src python scripts/summarize_predicate_labels.py --review-dir ' + shlex.quote(str(output))
    print(json.dumps({'review_dir': str(output), 'report': str(output / 'index.html'),
                     'after_saving_human_labels_command': command, **result}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
