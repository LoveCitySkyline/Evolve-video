"""Fixed decision axes; detailed criteria remain the acceptance/protection contract.

An aggregate and its source children must not each get a bargaining vote.
These axes are a declared scalarization, not calibrated latent quality factors.
"""
import math
import re
import statistics

VERSION = 'story-quality-axes-v1'
FIXED = {
    'identity': ('identity_consistency_score', 'clothing_color_score', 'identity_continuity'),
    'scene': ('background_preservation_score', 'scene_geometry'),
    'motion': ('motion_coherence',),
    'visual': ('visual_quality',),
    # Host action alignment already derives from source events/obligations.
    'action': ('action_alignment_score',),
}


def build(scores, criteria):
    declared = set(criteria)
    groups = {axis: {name: [name] for name in names if name in declared}
              for axis, names in FIXED.items()}
    action_fallback = not groups['action']
    if action_fallback:
        for name in sorted(declared):
            match = re.fullmatch(r'(story\.s\d+)\.event\.(.+)', name)
            if match:
                prefix = match[1] + '.obligation.' + match[2] + '.'
                groups['action'][name] = [name, *sorted(k for k in declared if k.startswith(prefix))]
    state, flow = {}, []
    for name in sorted(declared):
        match = re.fullmatch(r'story\.s(\d+)\.(pre|post|invariant)\..+', name)
        if match:
            state.setdefault(match[1], {}).setdefault(match[2], []).append(name)
        if re.fullmatch(r'story\.s\d+\.state_flow', name):
            flow.append(name)
    if flow:
        groups['motion']['physical_continuity'] = flow
    if state:
        groups['state'] = {shot + ':' + phase: names for shot, phases in state.items()
                           for phase, names in phases.items()}
    axes, used = {}, set()
    for axis, components in groups.items():
        if not components:
            continue
        members = sorted({name for names in components.values() for name in names})
        used.update(members)
        missing = [name for name in members if type(scores.get(name)) not in (int, float)
                   or not math.isfinite(scores[name]) or not 0 <= scores[name] <= 1]
        value = None
        if not missing:
            values = {k: statistics.mean(scores[n] for n in names) for k, names in components.items()}
            if axis == 'action' and action_fallback:
                value = statistics.mean(min(scores[n] for n in names) for names in components.values())
            elif axis == 'state':
                # Equal shot weight, then equal pre/post/invariant phase weight.
                per_shot = {}
                for key, v in values.items():
                    per_shot.setdefault(key.split(':')[0], []).append(v)
                value = statistics.mean(statistics.mean(v) for v in per_shot.values())
            else:
                value = min(values.values())
        axes['axis.' + axis] = {'score': value, 'status': 'measured' if value is not None else 'unknown',
            'members': members, 'components': components, 'missing': missing,
            'aggregation': 'mean_of_event_conjunctions' if axis == 'action' and action_fallback else
                           'equal_shot_then_phase_then_criterion_mean' if axis == 'state'
                           else 'minimum_of_component_means'}
    return {'protocol': VERSION, 'axes': axes,
        'diagnostic_only': sorted(declared - used),
        'qualification': 'Fixed family weights for decision making. Original criteria and mandatory '
            'checks remain active. No missing member is treated as zero or dropped.'}


def values(record):
    packet = record.get('quality_axes', {})
    if packet.get('protocol') != VERSION:
        raise ValueError('bargaining axes require the matching quality-axis protocol')
    return {key: row['score'] for key, row in packet['axes'].items()}


def from_evaluation(evaluation, fallback_rules):
    """Include measured generic axes even when they are outside the task rubric."""
    meta = evaluation.get('verification_metadata', {})
    rules = meta.get('effective_criterion_contracts') or fallback_rules
    from evovideo_skill.isolated_observations import VERSION as OBSERVATION_VERSION
    legacy_flow = [key for key in rules if key.endswith('.state_flow')
                   and meta.get('shared_observation_protocol') != OBSERVATION_VERSION]
    # Old state_flow mixed ideal task progression with physics. Displaying an
    # old run must not relabel those scores as newly measured continuity.
    if legacy_flow:
        rules = {key: rule for key, rule in rules.items() if key not in legacy_flow}
    scores = {k: evaluation[k] for k in rules if k in evaluation}
    scores.update(evaluation.get('criterion_scores', {}))
    unresolved = set(meta.get('unobserved_criteria', [])) | set(meta.get('disagreement_criteria', []))
    for key, state in meta.get('criterion_processing_states', {}).items():
        if state != 'measured':
            unresolved.add(key)
    for key in unresolved:
        scores[key] = None
    packet = build(scores, rules)
    packet['measurement_protocol'] = meta.get('shared_observation_protocol')
    packet['excluded_legacy_state_flow'] = legacy_flow
    return packet
