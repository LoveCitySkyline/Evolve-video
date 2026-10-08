"""Versioned evidence contracts, never keyword-based rewriting of model scores."""
from copy import deepcopy
import math
import re


def previous_boundary_evidence(definition, spans, manifest=None):
    """Only host-supplied preceding last-frame media may extend a citation scope.

    Never take allowed times from a model response or from task-defined times.
    The media router checks that the corresponding image is in the request.
    """
    if not definition.get('requires_previous_boundary'):
        return None
    index = definition.get('story_shot_index')
    if type(index) is not int or index <= 0:
        return None
    view = (manifest or {}).get('evaluation_view', {})
    frame = view.get('previous_boundary_context', {})
    previous = next((s for s in spans if s['segment_id'] == index - 1), None)
    timestamp = frame.get('source_timestamp_seconds')
    if (view.get('kind') != 'fixed_window_clip' or view.get('segment_id') != index
            or frame.get('boundary') != 'last' or frame.get('segment_id') != index - 1
            or not frame.get('media_label') or previous is None
            or type(timestamp) not in (int, float) or not math.isfinite(timestamp)
            or not previous['start_seconds'] <= timestamp < previous['end_seconds']):
        return None
    return {'source_segment_id': index - 1, 'source_timestamp_seconds': timestamp,
            'display_timestamp_seconds': round(timestamp, 6), 'media_label': frame['media_label'],
            'role': 'previous_boundary_context_only'}


def grounding_output_contract(criteria, spans, manifest=None):
    """Put every conditional field in the actual request contract, not only prose."""
    result = {}
    assessments = {
        'physical-motion-v1': {
            'required_fields': ['basis', 'outcome', 'defects'],
            'basis': 'physical_motion', 'defects': 'array of specific visible physical defects',
            'outcomes': {'coherent': 'observed, score=1, defects=[]',
                         'defective': 'observed, score<1, nonempty defects',
                         'unknown': 'unobserved, score=null'}},
        'state-equality-v1': {
            'required_fields': ['outcome'],
            'outcomes': {'satisfied': 'observed, score=1', 'violated': 'observed, score=0',
                         'unknown': 'unobserved, score=null'}},
        'required-action-v1': {
            'required_fields': ['outcome', 'matched', 'unmet'],
            'matched': 'array of correctly performed SOURCE requirements, not intent or available props',
            'unmet': 'array of unmet SOURCE requirements',
            'outcomes': {'complete': 'observed, score=1, matched nonempty, unmet empty',
                         'absent': 'observed, score=0, matched empty, unmet nonempty',
                         'partial': 'observed, 0<score<1, matched and unmet both nonempty',
                         'unknown': 'unobserved, score=null'}}}
    for name, rule in criteria.items():
        if not isinstance(rule, dict):
            continue
        contract = {}
        index = rule.get('story_shot_index')
        target_spans = [span for span in spans if index is None or span['segment_id'] == index]
        if rule.get('temporal_grounding'):
            contract['evidence_times_seconds'] = {
                'location': f'criteria[{name!r}].segments[*].evidence_times_seconds',
                'type': 'array of finite JSON numbers, NOT strings, ranges or objects',
                'required_for': 'Every applicable segment object, including unobserved segments',
                'observed': 'Nonempty array of original-video timestamps actually supporting that segment judgment',
                'unobserved': 'Empty array; explain the actual evidence limitation. A missing JSON field alone is NOT missing video evidence.',
                'windows': [{**span, 'interval': '[start_seconds, end_seconds)',
                             'end_is_exclusive': True} for span in target_spans],
                'instruction': 'Do not substitute interval endpoints for evidence. Cite actual supplied evidence. '
                    'Convert clip-local seconds using evaluation_view.source_time_offset_seconds. '
                    'Keep source boundary timestamps at supplied precision; do not round a last frame to the excluded endpoint. '
                    'Never clamp, invent, or move an out-of-window event to make it fit. '
                    'Include this field even when the same timestamp appears in evidence prose.'}
            context = previous_boundary_evidence(rule, spans, manifest)
            if context:
                contract['evidence_times_seconds']['allowed_context_evidence'] = [context]
                contract['evidence_times_seconds']['instruction'] += (
                    ' This criterion additionally permits the listed previous-boundary image timestamp '
                    '(exact source precision or its six-decimal displayed value) for entry-state comparison. '
                    'It is context, not an action inside this window. An observed judgment must still cite '
                    'actual current-window evidence. No other earlier timestamps are allowed.')
        kind = rule.get('judgment_contract')
        if kind:
            contract['assessment'] = {'contract': kind,
                'location': 'Top-level criterion AND each applicable target segment',
                **deepcopy(assessments[kind])}
            if index is not None:
                contract['same_window_consistency'] = 'Top-level and target segment status/score must match'
        if contract:
            result[name] = contract
    return result


def grounded_criteria(task, criteria):
    result = deepcopy(criteria)
    if not task.metadata.get("story_contract"):
        return result
    for name, rule in result.items():
        if name in {"motion_coherence", "action_alignment_score"}:
            rule["fixed_window_coverage"] = True
            rule["temporal_grounding"] = "original-timestamps-v1"
        if name == "motion_coherence":
            rule["judgment_contract"] = "physical-motion-v1"
        if "story_shot_index" in rule:
            rule["temporal_grounding"] = "original-timestamps-v1"
            if re.match(r"^story\.s\d+\.(pre|post)\.", name) and not name.endswith(".pre.initial_setup"):
                rule["judgment_contract"] = "state-equality-v1"
            elif re.match(r"^story\.s\d+\.(event|obligation)\.", name):
                rule["judgment_contract"] = "required-action-v1"
    return result


GROUNDING_INSTRUCTIONS = """
For temporal_grounding=original-timestamps-v1, each observed segment must include
evidence_times_seconds: a nonempty list of ORIGINAL-video timestamps supporting
its evidence, inside that segment's [start_seconds,end_seconds) interval. Convert
clip-local timestamps using the supplied offset. Unknown segments use an empty
list and explain the visibility/sampling/decoding limitation. Do not invent a
timestamp or treat an unexpected action as a missing time window. A full-video
group also receives actual first/last images for every fixed window; these prove
the supplied boundary observations, not all intervening actions.
Exception ONLY for requires_previous_boundary: output_contract lists the actual
previous last-frame image as allowed_context_evidence. You may cite that image's
source timestamp (or its six-decimal displayed timestamp) in evidence_times_seconds
for entry-state comparison, alongside actual current-window evidence. Describe it
as preceding context, never as an action inside the target window. This does not
authorize arbitrary timestamps in the previous window, other clips or references.

For judgment_contract=physical-motion-v1, every top-level and applicable segment
judgment must include assessment {basis:'physical_motion', outcome:'coherent'|
'defective'|'unknown', defects:[string,...]}. Judge only visible physical motion,
contacts and continuity. Actor assignment, narrative order and missing required
actions belong to action/story metrics, never this metric. Physically plausible
stillness is not a physics violation. Coherent means no visible physical defect
in the supplied evidence: observed, score=1, defects=[]. Defective means observed,
score<1, with specific visible physical defects and supporting timestamps. Unknown
means unobserved, score=null. Do not claim full-rate coherence from sparse frames.

For judgment_contract=state-equality-v1, top-level and target segment must include
assessment {outcome:'satisfied'|'violated'|'unknown'}. This is the exact declared
boundary fact, not a graded action. Satisfied requires observed score=1; violated
requires observed score=0; unknown requires unobserved score=null. An open empty
destination earns no partial credit when the required object is visibly elsewhere.
For a compound fact, all declared attributes must hold. Occlusion or unresolved
identity remains unknown. Use the supplied first frame for pre and last for post.

For judgment_contract=required-action-v1, top-level and target segment must include
assessment {outcome:'complete'|'partial'|'absent'|'unknown', matched:[string,...],
unmet:[string,...]}. Complete requires observed score=1 and no unmet requirements.
Absent means the specified core action was visibly not performed (including a
wrong actor or reverse direction with no correct component): observed score=0.
Partial allows 0<score<1 ONLY with specific correctly executed requirements in
matched AND actual unmet requirements in unmet. Intent, available props, preserved
identity or the mere absence of unrelated errors do not earn action credit.
Unknown requires unobserved score=null. Describe actors using frozen appearance
definitions; do not relabel them by possession. For compound events identify which
required components occurred; do not invent stricter prerequisites than the source.
All assessment fields describe the SAME visible evidence as status, score and text.
For a single scoped window, top-level and target-segment status and score must
match: they assess the same requirement on the same evidence.
If they conflict, reconcile honestly; never change evidence to pass the contract.
"""


def validate_grounding(name, definition, item, spans, manifest=None):
    if not isinstance(definition, dict):
        return
    kind = definition.get("judgment_contract")
    index = definition.get("story_shot_index")
    targets = [s for s in item["segments"] if index is None or s["segment_id"] == index]
    if definition.get("temporal_grounding"):
        by_id = {s["segment_id"]: s for s in spans}
        context = previous_boundary_evidence(definition, spans, manifest)
        context_times = {context['source_timestamp_seconds'], context['display_timestamp_seconds']} if context else set()
        problems = []
        for row in targets:
            row.pop('context_evidence_citations', None)  # Only the host may supply this audit annotation.
            if row["status"] == "not_applicable":
                continue
            times = row.get("evidence_times_seconds")
            span = by_id[row["segment_id"]]
            if 'evidence_times_seconds' not in row:
                reason = 'missing_field'
            elif not isinstance(times, list):
                reason = 'expected_array'
            elif row['status'] == 'observed' and not times:
                reason = 'observed_requires_nonempty_array'
            elif any(type(t) not in (int, float) or not math.isfinite(t) for t in times):
                reason = 'expected_finite_numeric_timestamps'
            elif any(not span['start_seconds'] <= t < span['end_seconds'] and t not in context_times for t in times):
                reason = 'timestamp_out_of_window'
            elif row['status'] == 'observed' and not any(span['start_seconds'] <= t < span['end_seconds'] for t in times):
                reason = 'previous_context_alone_cannot_support_current_window'
            else:
                if context:
                    # Keep the model's original values and explicitly identify the
                    # host-backed cross-window citations in the parsed audit.
                    row['context_evidence_citations'] = [
                        {**context, 'cited_timestamp_seconds': t} for t in times if t in context_times]
                continue
            problems.append(f"INSIDE segment {row['segment_id']} "
                f"[{span['start_seconds']}, {span['end_seconds']}): {reason}; "
                f"received={repr(times)[:240]}; status={row['status']}; "
                f"allowed_previous_boundary={sorted(context_times)}")
        if problems:
            raise ValueError(f"{name}: evidence_times_seconds at each applicable segment: " + '; '.join(problems) +
                '. Add missing fields from the SAME actual evidence; never invent/clamp timestamps or treat missing JSON as missing video.')
    if not kind:
        return
    if index is not None and any((row['status'], row['score']) != (item['status'], item['score'])
                                 for row in targets):
        raise ValueError(f"{name}: scoped assessment requires matching top-level and target-segment status/score")
    def strings(values):
        return isinstance(values, list) and all(isinstance(v, str) and v.strip() for v in values)
    for row in [item, *targets]:
        if row["status"] == "not_applicable":
            continue
        assessment = row.get("assessment")
        if not isinstance(assessment, dict):
            raise ValueError(f"{name}: {kind} requires structured assessment")
        outcome, score = assessment.get("outcome"), row["score"]
        expected_status = "unobserved" if outcome == "unknown" else "observed"
        valid = row["status"] == expected_status
        if kind == "physical-motion-v1":
            defects = assessment.get("defects")
            valid = valid and assessment.get("basis") == "physical_motion" and strings(defects)
            valid = valid and (outcome == "unknown" and score is None or
                              outcome == "coherent" and score == 1 and not defects or
                              outcome == "defective" and score is not None and score < 1 and bool(defects))
        elif kind == "state-equality-v1":
            valid = valid and outcome in {"satisfied", "violated", "unknown"}
            valid = valid and score == {"satisfied": 1, "violated": 0, "unknown": None}.get(outcome)
        elif kind == "required-action-v1":
            matched, unmet = assessment.get("matched"), assessment.get("unmet")
            valid = valid and strings(matched) and strings(unmet)
            valid = valid and (outcome == "unknown" and score is None or
                outcome == "absent" and score == 0 and bool(unmet) and not matched or
                outcome == "complete" and score == 1 and bool(matched) and not unmet or
                outcome == "partial" and score is not None and 0 < score < 1 and bool(matched) and bool(unmet))
        else:
            raise ValueError(f"unknown judgment contract: {kind}")
        if not valid:
            raise ValueError(f"{name}: {kind} assessment conflicts with status/score or uses an "
                             "invalid basis; reassess the SAME evidence, do not invent observations")
