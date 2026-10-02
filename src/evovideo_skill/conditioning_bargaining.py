"""Discrete constrained bargaining; fixed references, no H3 gradient training.

KS here means max-min normalized attainment with Nash tie-breaking, not an
axiomatic KS solution on a convex bargaining set. Missing evidence is an error.
"""
from copy import deepcopy
import math
import statistics

DEFAULTS = {"enabled": False, "method": "ks", "metric_source": "criteria", "metrics": [],
    "floor": .0, "aspiration": 1., "overrides": {}, "max_call_ratio": 4.,
    "max_second_ratio": 4., "ideal_call_ratio": 1., "ideal_second_ratio": 1.,
    "min_gain": .01, "min_validation_gain": .01, "se_multiplier": 1.,
    "stopping_enabled": True, "stop_patience": 2, "stop_min_rounds": 2}


def options(config):
    raw = config.get("bargaining", {})
    if not isinstance(raw, dict) or set(raw) - DEFAULTS.keys():
        raise ValueError("invalid bargaining settings")
    o = deepcopy(DEFAULTS) | deepcopy(raw)
    if type(o['enabled']) is not bool or o['method'] not in {'ks', 'nash'}:
        raise ValueError('invalid bargaining enabled/method')
    if type(o['stopping_enabled']) is not bool:
        raise ValueError('bargaining.stopping_enabled must be boolean')
    if o['metric_source'] not in {'criteria', 'metrics'}:
        raise ValueError('bargaining metric_source must be criteria or metrics')
    if (not isinstance(o['metrics'], list) or any(not isinstance(k, str) or not k or k == 'cost' for k in o['metrics'])
            or len(o['metrics']) != len(set(o['metrics']))):
        raise ValueError('invalid bargaining metrics')
    for key in ('floor', 'aspiration', 'max_call_ratio', 'max_second_ratio', 'ideal_call_ratio',
                'ideal_second_ratio', 'min_gain', 'min_validation_gain', 'se_multiplier'):
        if type(o[key]) not in (int, float) or not math.isfinite(o[key]) or o[key] < 0:
            raise ValueError(f'invalid bargaining.{key}')
    if not 0 <= o['floor'] < o['aspiration'] <= 1:
        raise ValueError('bargaining requires 0 <= floor < aspiration <= 1')
    for unit in ('call', 'second'):
        if not 0 < o[f'ideal_{unit}_ratio'] < o[f'max_{unit}_ratio']:
            raise ValueError('bargaining cost ideal must be positive and below maximum')
    if not isinstance(o['overrides'], dict):
        raise ValueError('bargaining overrides must be a metric map')
    for key, value in o['overrides'].items():
        if (not isinstance(key, str) or not isinstance(value, dict) or set(value) != {'floor', 'aspiration'}
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in value.values())
                or not 0 <= value['floor'] < value['aspiration'] <= 1):
            raise ValueError('invalid bargaining metric reference')
    for key in ('stop_patience', 'stop_min_rounds'):
        if type(o[key]) is not int or o[key] < 1:
            raise ValueError(f'invalid bargaining.{key}')
    return o


def vector(record, o):
    if o['metric_source'] == 'criteria':
        values = record.get('criterion_scores', {})
    else:
        from evovideo_skill.conditioning_memory import metric_vector
        values = metric_vector(record)
    names = o['metrics'] or sorted(values)
    if not names or 'cost' in names or any(k not in values or type(values[k]) not in (int, float)
                        or not math.isfinite(values[k]) or not 0 <= values[k] <= 1 for k in names):
        raise ValueError('bargaining requires complete finite measured objectives; missing is not zero')
    return {k: values[k] for k in names}


def from_values(values, cost, o):
    ratios, violations = {}, []
    for key, value in values.items():
        ref = o['overrides'].get(key, o)
        ratios[key] = (value - ref['floor']) / (ref['aspiration'] - ref['floor'])
        if value < ref['floor']:
            violations.append(key + ':below_floor')
    # Correlated call/seconds proxies form ONE conservative cost participant.
    cost_ratios = [(o[f'max_{unit}_ratio'] - cost[key]) /
                   (o[f'max_{unit}_ratio'] - o[f'ideal_{unit}_ratio'])
                   for unit, key in (('call', 'normalized_calls'), ('second', 'normalized_seconds'))]
    ratios['cost'] = min(cost_ratios)
    if ratios['cost'] < 0:
        violations.append('cost:over_budget')
    # Aspiration is a target, not a reward for unlimited overachievement.
    attainment = {k: min(1., v) for k, v in ratios.items()}
    nash = statistics.mean(math.log(max(1e-12, v)) for v in attainment.values())
    primary = min(attainment.values()) if o['method'] == 'ks' else nash
    return {'values': values, 'attainment': attainment, 'score': primary, 'nash_log': nash,
            'feasible': not violations, 'violations': violations,
            'cost': {k: cost[k] for k in ('calls', 'generated_seconds', 'normalized_calls', 'normalized_seconds')},
            'bottlenecks': [k for k, v in attainment.items() if abs(v-min(attainment.values())) < 1e-9]}


def profile(records, o):
    if not records:
        raise ValueError('empty bargaining evidence')
    vectors = [vector(r, o) for r in records]
    if any(v.keys() != vectors[0].keys() for v in vectors):
        raise ValueError('bargaining objective dimensions changed')
    for r in records:
        c = r.get('generation_cost', {})
        if any(k not in c or type(c[k]) not in (int, float) or not math.isfinite(c[k]) or c[k] < 0
               for k in ('calls', 'generated_seconds', 'normalized_calls', 'normalized_seconds')):
            raise ValueError('bargaining requires complete cost evidence')
    means = {k: statistics.mean(v[k] for v in vectors) for k in vectors[0]}
    se = {k: statistics.stdev(v[k] for v in vectors)/math.sqrt(len(vectors)) if len(vectors)>1 else 0.
          for k in means}
    conservative = {k: max(0., value-o['se_multiplier']*se[k]) for k, value in means.items()}
    costs = {k: statistics.mean(r['generation_cost'][k] for r in records)
             for k in ('calls', 'generated_seconds', 'normalized_calls', 'normalized_seconds')}
    return {**from_values(conservative, costs, o), 'means': means, 'standard_errors': se,
            'uncertainty_note': 'Seed SE penalty is a heuristic, not calibrated coverage or verifier-bias correction.'}


def annotate(effect, before, after):
    all_records = [*before, *after]
    present = ['bargaining_objective' in r for r in all_records]
    if not any(present):
        return effect
    if not all(present):
        raise ValueError('mixed missing bargaining objective')
    o = before[0]['bargaining_objective']
    if any(r['bargaining_objective'] != o for r in all_records):
        raise ValueError('bargaining objective changed between observations')
    a, b = profile(before, o), profile(after, o)
    if a['values'].keys() != b['values'].keys():
        raise ValueError('bargaining dimensions differ between paired observations')
    gains = [profile([y], o)['score']-profile([x], o)['score'] for x, y in zip(before, after)]
    effect['bargaining'] = {'objective': o, 'before': a, 'after': b, 'gain': b['score']-a['score'],
        'seed_gains': gains, 'gain_std': statistics.stdev(gains) if len(gains)>1 else None,
        'metric_deltas': {k: b['means'][k]-a['means'][k] for k in a['means']}}
    return effect


def supported(effect, config):
    o = options(config)
    e = effect.get('bargaining')
    if e is None or e['objective'] != o:
        raise ValueError('bargaining selection requires matching objective evidence')
    # Preserve the existing average-quality and per-metric safeguards.
    from evovideo_skill.conditioning_cost import cost_options
    if effect['gain'] < -cost_options(config)['max_quality_drop'] or any(
            v < -config['max_metric_regression'] for v in effect['metric_deltas'].values()):
        return False
    se = (e['gain_std'] or 0)/math.sqrt(len(e['seed_gains']))
    return (e['after']['feasible'] and e['gain'] > o['min_gain'] and
            sum(v > 0 for v in e['seed_gains'])/len(e['seed_gains']) >= config.get('min_positive_seed_fraction', 0) and
            statistics.mean(e['seed_gains'])-config.get('gain_se_multiplier', 0)*se > 0)


def frontier(cells, o):
    points = [{'cell': key, **profile(records, o)} for key, records in cells.items()]
    for p in points:
        p['dominated_by'] = []
        for other in points:
            if other['values'].keys() != p['values'].keys():
                raise ValueError('cannot compare different bargaining objective sets')
            a = [*other['values'].values(), -other['cost']['calls'], -other['cost']['generated_seconds']]
            b = [*p['values'].values(), -p['cost']['calls'], -p['cost']['generated_seconds']]
            if other['feasible'] and all(x >= y for x,y in zip(a,b)) and any(x > y for x,y in zip(a,b)):
                p['dominated_by'].append(other['cell'])
        p['pareto'] = p['feasible'] and not p['dominated_by']
    return points


def acquisition(records, prediction, joint_cost, parent_equals_anchor, config):
    o = options(config)
    parent = profile(records, o)
    # Per-dimension empirical means; unknown terms retain an exploration radius.
    terms = prediction['terms']
    unknown = [k for k in parent['means'] if any(k not in t['metric_means'] for t in terms)]
    radius = prediction['uncertainty'] * config.get('active_graph_search', {}).get('exploration_beta', .5)
    values = {k: min(1., max(0., v + sum(t['metric_means'].get(k, 0.) for t in terms) + radius))
              for k,v in parent['means'].items()}
    if not parent_equals_anchor:
        # Never treat an unmeasured scaffold change as a zero quality effect.
        values = {k: 1. for k in values}
        unknown = list(values)
    optimistic = from_values(values, joint_cost, o)
    expense = prediction['cost_tradeoff']['experiment_penalty']
    return {'acquisition': optimistic['score']-parent['score']-expense,
        'parent': parent, 'optimistic': optimistic, 'unknown_objectives': unknown,
        'anchor_quality_unknown': not parent_equals_anchor,
        'experiment_penalty': expense,
        'qualification': 'Uncalibrated optimistic scheduling heuristic, not measured utility or expected information value.'}


def update_stopping(state, improved, config):
    o = options(config)
    state = deepcopy(state or {'rounds': 0, 'no_improvement': 0})
    state['rounds'] += 1
    state['no_improvement'] = 0 if improved else state['no_improvement']+1
    state['stop'] = o['stopping_enabled'] and state['rounds'] >= o['stop_min_rounds'] and state['no_improvement'] >= o['stop_patience']
    state['reason'] = 'measured_no_improvement_patience' if state['stop'] else None
    state['qualification'] = 'Budget-saving patience heuristic; not proof that all unexplored paths are inferior.'
    return state
