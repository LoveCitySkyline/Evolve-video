"""Exact bounds for declared operators on scores in [0, 1]. No visual inference."""
import math

VERSION = 'bounded-source-aggregation-v1'


def bounds(value):
    if value is None:
        return (0.0, 1.0)
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('aggregate score must be a finite number in [0,1] or None')
    return (float(value), float(value))


def combine(intervals, operator):
    intervals = list(intervals)
    if not intervals:
        raise ValueError('aggregation requires nonempty declared components')
    for lo, hi in intervals:
        bounds(lo)
        bounds(hi)
        if lo is None or hi is None or lo > hi:
            raise ValueError('invalid score interval')
    if operator == 'minimum':
        return (min(lo for lo, _ in intervals), min(hi for _, hi in intervals))
    if operator == 'mean':
        return (sum(lo for lo, _ in intervals) / len(intervals),
                sum(hi for _, hi in intervals) / len(intervals))
    raise ValueError('unsupported source aggregation operator')


def point(interval):
    # No tolerance: a nonzero interval is uncertainty, however small.
    return interval[0] if interval[0] == interval[1] else None


def integrity_blocked(row):
    return bool(row.get('fact_conflicts') or row.get('scope_issues')
                or row.get('identity_gate', {}).get('status') == 'blocked'
                or row.get('processing_status') in {'invalid_response', 'identity_unresolved', 'disputed'})


def observation_bounds(row):
    # Never promote stale scores carried by an unknown or quarantined row.
    if integrity_blocked(row) or row.get('status') != 'observed':
        return (0.0, 1.0)
    return bounds(row.get('score'))


def audit(interval, operator, unknown, decisive=()):
    return {'protocol': VERSION, 'operator': operator, 'lower': interval[0], 'upper': interval[1],
            'score_determined': point(interval) is not None, 'unknown_components': list(unknown),
            'decisive_zero_components': list(decisive),
            'qualification': 'Host algebra on validated observations, not new visual evidence. '
                             'Unknown children remain unknown; a determined parent does not complete its children.'}
