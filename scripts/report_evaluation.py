"""Summarize saved quality, constraints and costs without generating or judging media."""
import argparse
import json
from pathlib import Path
from evovideo_skill.evaluation_report import export_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    if not args.run_dir.is_dir():
        parser.error('run directory does not exist')
    report = export_report(args.run_dir)
    print(json.dumps({'report': str(args.run_dir / 'evaluation_report.html'),
                      'outputs': report['output_counts'], 'read_issues': report['read_issues'],
                      'model_calls': 0}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
