#!/usr/bin/env python3
"""Validate exported human labels and summarize selected development facts offline."""
import argparse
import json
from pathlib import Path
from evovideo_skill.predicate_review_pack import summarize_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-dir', type=Path, required=True)
    parser.add_argument('--labels', type=Path, help='Defaults to human-labels.json inside review-dir')
    args = parser.parse_args()
    try:
        result = summarize_labels(args.review_dir, args.labels or args.review_dir / 'human-labels.json')
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
