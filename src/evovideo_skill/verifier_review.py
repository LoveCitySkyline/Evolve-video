"""Bounded blind evidence acquisition. Unresolved judgments remain abstentions."""
from copy import deepcopy
import json
import math
import os
import tempfile
import subprocess
from pathlib import Path
import time
import uuid
from evovideo_skill.h3_api import portable_interprocess_lock
from evovideo_skill.research_protocol import write_json
from evovideo_skill.research_subgraphs import stable_hash

VERSION = 'bounded-evidence-review-v6.2'
DEFAULTS = dict(enabled=False, max_calls_per_criterion=4, max_calls_per_video=8,
                max_calls_per_run=200, max_seconds_per_video=600, fps=8, max_width=1536,
                secondary_model=None)
CHECKS = {
    'referents': 'Can required actors/objects be identified visually, without inferring identity from desired actions?',
    'visibility': 'Is the predicate observable, rather than occluded or hidden in an unsampled gap?',
    'predicate': 'Does the EXACT requirement hold? Decompose all explicit clauses into components; add no requirements.',
    'temporal_scope': 'Does evidence establish this predicate in the required window, using the true pre/post boundary?',
}
REVIEW_SYSTEM = '''
This request is a bounded evidence review. In addition to the ordinary judgment
fields, EACH requested criterion MUST contain atomic_checks, with exactly these
four named objects: referents, visibility, predicate, temporal_scope. This is part
of the output schema, not optional commentary. Each object requires status
(supported|contradicted|unknown) and evidence. predicate additionally requires a
nonempty components list; each component has source_quote (an exact substring of
this criterion's description, or the EXACT complete primary proposition listed
in predicate_sources), status and evidence. Check all explicit clauses.
The predicate status is their conjunction: any contradicted => contradicted;
otherwise any unknown => unknown; otherwise supported. Unknown prerequisites
(referents, visibility or temporal_scope) require an unknown judgment. Never
invent observations or improve a score merely to fill missing JSON fields.
For assessment_patch_only corrections, return ONLY the requested patches instead;
the host retains and revalidates the previously supplied atomic_checks.
'''


def predicate_sources(rule):
    """Only this criterion and its host-compiled primary proposition are sources."""
    contract = rule.get('fact_contract', {})
    primary = contract.get('facts', {}).get(contract.get('primary_fact'), {})
    return {'description': rule.get('description', ''),
            'primary_proposition': primary.get('proposition')}


def valid_source_quote(quote, rule):
    sources = predicate_sources(rule)
    return isinstance(quote, str) and bool(quote.strip()) and (
        quote in sources['description'] or quote == sources['primary_proposition'])


def add_review_contract(payload):
    """Put the extension in the same required fields the base judge follows."""
    contract = payload['output_contract']
    if contract.get('response_mode') == 'assessment_patch_only':
        return
    check_schema = {'status': 'supported|contradicted|unknown', 'evidence': 'visible support or limitation'}
    schema = {key: deepcopy(check_schema) for key in CHECKS}
    schema['predicate']['components'] = [{'source_quote': 'exact substring of description OR exact full primary_proposition in predicate_sources',
        **deepcopy(check_schema)}]
    for name in payload['criteria']:
        fields = contract.setdefault('fields', {}).setdefault(name, {})
        required = fields.setdefault('required', [])
        if 'atomic_checks' not in required:
            required.append('atomic_checks')
        fields['atomic_checks'] = deepcopy(schema)
        contract.setdefault('predicate_sources', {})[name] = predicate_sources(payload['criteria'][name])

class ReviewBudgetExhausted(RuntimeError):
    pass


def options(value=None):
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError('unknown auto_review options')
    out = {**DEFAULTS, **value}
    if type(out['enabled']) is not bool:
        raise ValueError('auto_review.enabled must be boolean')
    for key in ('max_calls_per_criterion', 'max_calls_per_video', 'max_calls_per_run',
                'max_seconds_per_video', 'max_width'):
        if type(out[key]) is not int or out[key] <= 0:
            raise ValueError('invalid auto_review.' + key)
    if type(out['fps']) not in (int, float) or not math.isfinite(out['fps']) or not 0 < out['fps'] <= 30:
        raise ValueError('invalid auto_review.fps')
    if out['secondary_model'] is not None and (not isinstance(out['secondary_model'], str) or not out['secondary_model'].strip()):
        raise ValueError('auto_review.secondary_model must be a model ID at the same configured provider')
    return out


def uncertain(rows, threshold):
    from evovideo_skill.verifier_facts import fact_conflicts
    if any(r.get('fact_conflicts') for r in rows) or fact_conflicts({'criterion': rows}):
        return True
    if not rows or any(r.get('status') == 'unobserved' or r.get('scope_issues') for r in rows):
        return True
    if len({r.get('status') for r in rows}) != 1:
        return True
    if rows[0].get('status') == 'not_applicable':
        return False
    if max(r['score'] for r in rows) - min(r['score'] for r in rows) > threshold:
        return True
    segments = {}
    for row in rows:
        for s in row.get('segments', []):
            segments.setdefault(s['segment_id'], []).append(s)
    return any(any(s.get('status') == 'unobserved' for s in ss) or
               len({s['status'] for s in ss}) > 1 or
               (all(s['status'] == 'observed' for s in ss) and
                max(s['score'] for s in ss) - min(s['score'] for s in ss) > threshold)
               for ss in segments.values())


class ReviewLedger:
    """Reserve before sending; crashes cannot reset API budgets on resume."""
    def __init__(self, root, config, candidate, criterion):
        self.path = Path(root) / 'auto_review_budget.json'
        self.config, self.candidate, self.criterion = config, candidate, criterion

    def reserve(self, ticket=None):
        ticket = ticket or uuid.uuid4().hex
        with portable_interprocess_lock(self.path.with_suffix('.lock'), timeout_seconds=30):
            data = json.loads(self.path.read_text()) if self.path.exists() else {'calls': 0, 'videos': {}}
            video = data['videos'].setdefault(self.candidate, {'calls': 0, 'reserved_seconds': 0, 'criteria': {}})
            if ticket in video.get('reservations', {}):
                raise ValueError('duplicate review reservation')
            count = video['criteria'].get(self.criterion, 0)
            remaining = self.config['max_seconds_per_video'] - video['reserved_seconds']
            if (data['calls'] >= self.config['max_calls_per_run'] or video['calls'] >= self.config['max_calls_per_video']
                    or count >= self.config['max_calls_per_criterion'] or remaining <= 0):
                raise ReviewBudgetExhausted('automatic review budget exhausted; no score assigned')
            timeout = min(180, remaining)
            data['calls'] += 1
            video['calls'] += 1
            video['criteria'][self.criterion] = count + 1
            video['reserved_seconds'] += timeout
            video.setdefault('reservations', {})[ticket] = {'seconds': timeout, 'status': 'pending',
                'criterion': self.criterion}
            write_json(self.path, data)
            return timeout

    def settle(self, ticket, elapsed_seconds):
        """Return unused wall-time reservation, never refund a model-call credit.

        A crashed process leaves its full pending reservation charged. Duplicate
        settlement is idempotent and cannot refund other workers' time.
        """
        if not isinstance(elapsed_seconds, (int, float)) or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError('invalid review elapsed time')
        with portable_interprocess_lock(self.path.with_suffix('.lock'), timeout_seconds=30):
            data = json.loads(self.path.read_text())
            video = data['videos'][self.candidate]
            reservation = video.get('reservations', {}).get(ticket)
            if reservation is None or reservation['criterion'] != self.criterion:
                raise ValueError('unknown review reservation')
            if reservation['status'] == 'settled':
                return
            # reserved_seconds includes settled actual time plus pending reservations.
            # Charge measured overruns too; subsequent requests then stop.
            video['reserved_seconds'] += elapsed_seconds - reservation['seconds']
            video['spent_seconds'] = video.get('spent_seconds', 0) + elapsed_seconds
            reservation.update(status='settled', actual_seconds=elapsed_seconds)
            write_json(self.path, data)


def validate_checks(raw, criteria):
    """Validate explicit support, never infer facts from prose or confidence."""
    clean = deepcopy(raw)
    rows = clean.get('criteria', {}) if isinstance(clean, dict) else {}
    if not isinstance(rows, dict) or set(rows) != set(criteria):
        raise ValueError('review must return exactly the requested criteria')
    for name, rule in criteria.items():
        row = rows[name]
        if not isinstance(row, dict):
            raise ValueError('review judgment must be an object')
        checks = row.pop('atomic_checks', None)
        if not isinstance(checks, dict) or set(checks) != set(CHECKS):
            raise ValueError('review requires all atomic_checks')
        for key, check in checks.items():
            if (not isinstance(check, dict) or check.get('status') not in {'supported', 'contradicted', 'unknown'}
                    or not isinstance(check.get('evidence'), str) or not check['evidence'].strip()):
                raise ValueError('invalid atomic check ' + key)
        components = checks['predicate'].get('components')
        if not isinstance(components, list) or not components:
            raise ValueError('predicate needs source-backed components')
        for part in components:
            if (not isinstance(part, dict) or not valid_source_quote(part.get('source_quote'), rule)
                    or part.get('status') not in {'supported', 'contradicted', 'unknown'}
                    or not isinstance(part.get('evidence'), str) or not part['evidence'].strip()):
                raise ValueError(f'each component needs an exact criterion quote (description substring or '
                    f'complete primary proposition), status and evidence; allowed_sources={predicate_sources(rule)}')
        statuses = [c['status'] for c in components]
        predicate = 'contradicted' if 'contradicted' in statuses else 'unknown' if 'unknown' in statuses else 'supported'
        if checks['predicate']['status'] != predicate:
            raise ValueError('predicate contradicts component conjunction')
        observable = all(checks[k]['status'] == 'supported' for k in ('referents', 'visibility', 'temporal_scope'))
        conclusion = predicate if observable else 'unknown'
        assessment = row.get('assessment', {})
        if not isinstance(assessment, dict):
            raise ValueError('review assessment must be an object')
        outcome = assessment.get('outcome')
        if rule.get('judgment_contract') == 'state-equality-v1':
            if outcome != {'supported': 'satisfied', 'contradicted': 'violated', 'unknown': 'unknown'}[conclusion]:
                raise ValueError('state judgment contradicts atomic checks')
        else:
            unknown = outcome == 'unknown' if rule.get('judgment_contract') else row.get('status') == 'unobserved'
            if unknown != (conclusion == 'unknown'):
                raise ValueError('observability contradicts atomic checks')
            score = row.get('score')
            if conclusion == 'contradicted' and (outcome in {'complete', 'coherent', 'satisfied'} or score == 1):
                raise ValueError('violated component cannot receive full credit')
            if conclusion == 'supported' and (outcome in {'absent', 'defective', 'violated'} or score == 0):
                raise ValueError('supported components contradict negative judgment')
    return clean


def compress_review_samples(reviewer, evidence, manifest, width_cap):
    """Keep every timestamped sample; JPEG transport leaves PNG anchors intact."""
    view = manifest.get('evaluation_view', {})
    frames = view.get('sampled_frames', [])
    if not frames:
        return evidence, manifest
    source_paths = [reviewer.root / 'media' / frame['media_file'] for frame in frames]
    source_dir = source_paths[0].parent
    expected = [f'{i + 1:06d}.png' for i in range(len(frames))]
    if [p.name for p in source_paths] != expected or any(p.parent != source_dir for p in source_paths):
        raise ValueError('review sample source is not the complete ordered PNG sequence')
    for path, frame in zip(source_paths, frames):
        if stable_hash(path.read_bytes().hex()) != frame['image_hash']:
            raise ValueError('review sample changed before transport encoding')
    encoding = f'jpeg-q2-yuvj444p-v1-w{width_cap}'
    folder = reviewer.root / 'media' / ('review-samples-' + stable_hash([
        [f['image_hash'] for f in frames], encoding]))
    names = [f'{i + 1:06d}.jpg' for i in range(len(frames))]
    if not folder.exists():
        with tempfile.TemporaryDirectory(dir=folder.parent) as tmp:
            target = Path(tmp) / 'frames'
            target.mkdir()
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-start_number', '1',
                '-i', str(source_dir / '%06d.png'), '-frames:v', str(len(frames)),
                '-vf', f"scale='min({width_cap},iw)':-2",
                '-fps_mode', 'passthrough', '-c:v', 'mjpeg', '-q:v', '2',
                '-pix_fmt', 'yuvj444p', str(target / '%06d.jpg')],
                check=True, capture_output=True, timeout=120)
            if sorted(p.name for p in target.iterdir()) != names:
                raise ValueError('review JPEG encoding lost sample frames')
            target.replace(folder)
    if sorted(p.name for p in folder.iterdir()) != names:
        raise ValueError('review JPEG cache is incomplete')
    evidence, manifest = list(evidence), deepcopy(manifest)
    replacements = {}
    for frame, name in zip(manifest['evaluation_view']['sampled_frames'], names):
        target = folder / name
        if not target.read_bytes().startswith(b'\xff\xd8\xff'):
            raise ValueError('review sample transport is not JPEG')
        medium = reviewer.media(target, 'image')
        replacements[frame['media_label']] = medium
        frame.update(original_image_hash=frame['image_hash'], original_media_file=frame['media_file'],
            image_hash=medium['source_hash'], media_file=str(target.relative_to(reviewer.root / 'media')),
            transport_encoding=encoding)
    view = manifest['evaluation_view']
    view['sample_transport'] = {'encoding': encoding, 'frame_count': len(frames), 'width_cap': width_cap,
        'qualification': 'Lossy image encoding with an explicit width cap; unchanged timestamps, no dropped samples. '
                         'Original PNG samples retained. Boundary frames, crops and references unchanged.'}
    return [(label, replacements.get(label, medium)) for label, medium in evidence], manifest


def fit_review_samples(reviewer, evidence, manifest):
    """Budget for all media plus JSON overhead; final serialized body is checked too."""
    limit = reviewer.profile['max_request_bytes']
    widths = list(dict.fromkeys(min(reviewer.profile['max_width'], cap)
                               for cap in (reviewer.profile['max_width'], 1152, 768)))
    for width in widths:
        packed, packed_manifest = compress_review_samples(reviewer, evidence, manifest, width)
        total = sum(len(m['data']) for _, m in packed)
        if total <= int(limit * .85):
            break
    view = packed_manifest['evaluation_view']
    view['sample_transport'].update(base64_media_bytes=total, max_request_bytes=limit,
        json_headroom_fraction=.15, final_serialized_size_check=True)
    print(f"[conditioning verifier] review transport samples={len(view['sampled_frames'])} "
          f"width_cap={width} base64_media_bytes={total} request_limit={limit}", flush=True)
    # Even the minimum representation may be too large (e.g. huge references).
    # Preserve the evidence and let the exact request guard abstain, never truncate it.
    return packed, packed_manifest


def add_boundary_crops(reviewer, evidence, manifest):
    """Deterministic overlapping tiles retain the full frame and exact provenance."""
    from evovideo_skill.h3_api import probe_media
    view = manifest.get('evaluation_view', {})
    if view.get('kind') != 'fixed_window_clip':
        return evidence, manifest
    evidence, manifest = reviewer.fixed_window_input(evidence, manifest)
    view = manifest['evaluation_view']
    view['evidence_crops'] = []
    for frame in view.get('boundary_frames', []):
        source = reviewer.root / 'media' / frame['media_file']
        stream = probe_media(str(source))['streams'][0]
        width, height = stream['width'], stream['height']
        cw, ch = max(1, int(width * .6)), max(1, int(height * .6))
        for index, (x, y) in enumerate(((0, 0), (width-cw, 0), (0, height-ch), (width-cw, height-ch))):
            bbox = [x, y, cw, ch]
            digest = stable_hash([frame['image_hash'], bbox, VERSION])
            target = reviewer.root / 'media' / f'review-crop-{digest}.png'
            if not target.exists():
                fd, temporary = tempfile.mkstemp(suffix='.png', dir=target.parent)
                os.close(fd)
                try:
                    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(source),
                        '-vf', f'crop={cw}:{ch}:{x}:{y}', '-frames:v', '1', temporary],
                        check=True, capture_output=True, timeout=30)
                    Path(temporary).replace(target)
                finally:
                    Path(temporary).unlink(missing_ok=True)
            label = (f"CANDIDATE CROP {index}, {frame['boundary']} boundary, segment_id={frame['segment_id']}; "
                f"original-video timestamp={frame['source_timestamp_seconds']}; crop xywh={bbox} "
                f"in supplied boundary frame {width}x{height}; SAME frame, full image also supplied")
            medium = reviewer.media(target, 'image')
            record = {**deepcopy(frame), 'parent_image_hash': frame['image_hash'],
                'image_hash': medium['source_hash'], 'media_file': target.name, 'media_label': label,
                'crop_xywh': bbox, 'parent_dimensions': [width, height],
                'evidence_id': f"s{frame['segment_id']}:{frame['boundary']}:crop{index}"}
            view['evidence_crops'].append(record)
            evidence.append((label, medium))
    return fit_review_samples(reviewer, evidence, manifest)


def review_group(owner, task, artifact, public, subset, rows, folder, group, digest, expected_source_hash=None):
    """Two blind valid confirmations on enhanced evidence, or retain uncertainty."""
    from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier, VerifierFormatError, VerifierEvidenceError
    from evovideo_skill.api_tools import VideoApiError
    config = options(owner.profile.get('auto_review'))
    source_hash = stable_hash(Path(artifact.metadata['local_video_path']).read_bytes().hex())
    if expected_source_hash is not None and source_hash != expected_source_hash:
        raise VerifierEvidenceError('candidate source changed before automatic review')
    audit = {}
    for name, initial in rows.items():
        if not uncertain(initial, owner.profile['disagreement_threshold']):
            continue
        if any(row.get('identity_gate', {}).get('status') == 'blocked' for row in initial):
            audit[name] = {'status': 'abstained', 'observations': [],
                'errors': ['shared identity reassessment unresolved; no per-criterion retry or score assigned'],
                'review_scope': 'shared_identity', 'additional_model_calls': 0}
            continue
        key = stable_hash([group, name])[:16]
        root = folder / 'auto_review' / key
        saved = root / 'decision.json'
        if saved.exists():
            decision = json.loads(saved.read_text())
            audit[name] = decision
            if decision['status'] == 'resolved':
                rows[name] = decision['observations']
            continue
        root.mkdir(parents=True, exist_ok=True)
        print(f'[conditioning verifier] auto-review criterion={name} group={group} budgeted=true', flush=True)
        started = time.monotonic()
        results, failures = [], []
        rule = {name: subset[name]}
        ledger_root = owner.root.parent if owner.root.name in {'runtime', 'final'} else owner.root
        ledger = ReviewLedger(ledger_root, config, source_hash,
                              stable_hash([owner.root.name, name, subset[name]]))

        class Reviewer(ConditioningVideoVerifier):
            response_contract_instructions = REVIEW_SYSTEM

            def validate_response_contract(self, raw, criteria):
                return validate_checks(raw, criteria)

            def request(self, prompt, evidence, operation):
                payload = json.loads(prompt)
                add_review_contract(payload)
                payload['atomic_review'] = {'checks': CHECKS, 'location': 'atomic_checks inside each criterion',
                    'schema': {'each_check': {'status': 'supported|contradicted|unknown', 'evidence': 'visible support or limitation'},
                        'predicate.components': [{'source_quote': 'exact description substring or full primary proposition from predicate_sources',
                            'status': 'supported|contradicted|unknown', 'evidence': 'support for this clause'}]},
                    'instruction': 'Independently assess ALL explicit requirements including conjunctions. '
                        'Unknown identity or occlusion does not establish a violation. Box-like appearance alone '
                        'does not establish an object function. Never infer identity from the target. '
                        'No new numerical distance, object count or action requirement may be invented. '
                        'Components must cover ALL explicit clauses; unknown prerequisites mean unknown judgment. '
                        'Crops supplement the full image; never infer absence outside a crop. '
                        'For physical motion judge physics only, not desired narrative.'}
                # Physical-domain sanitization already happened in _observe_group.
                timeout = getattr(self, 'prepaid_timeout', None)
                if timeout is not None:
                    self.prepaid_timeout = None
                    timeout -= time.monotonic() - self.acquisition_started
                    if timeout < 1:
                        raise ReviewBudgetExhausted('evidence acquisition exhausted reserved review time')
                    ticket = self.prepaid_ticket
                    request_started = self.acquisition_started
                else:
                    ticket = uuid.uuid4().hex
                    timeout = ledger.reserve(ticket)
                    request_started = time.monotonic()
                self.prepaid_ticket = None
                self.profile['timeout_seconds'] = min(self.profile['timeout_seconds'], max(1, int(timeout)))
                try:
                    raw = super().request(json.dumps(payload, ensure_ascii=False), evidence, operation)
                except Exception:
                    ledger.settle(ticket, time.monotonic() - request_started)
                    raise
                else:
                    ledger.settle(ticket, time.monotonic() - request_started)
                write_json(root / (stable_hash(operation) + '.atomic.raw.json'), raw)
                return raw

        for repeat in range(2):
            p = {**owner.profile, 'auto_review': {**config, 'enabled': False}, 'max_attempts': 1,
                 'max_width': max(owner.profile['max_width'], config['max_width']),
                 'fps': max(owner.profile['fps'], config['fps'])}
            if repeat and config['secondary_model']:
                p['model'] = config['secondary_model']
            reviewer = Reviewer(p, owner.root)
            parsed_cache = root / f'confirmation-{repeat}.parsed.json'
            if parsed_cache.exists():
                results.append(json.loads(parsed_cache.read_text())[name])
                continue
            try:
                ticket = uuid.uuid4().hex
                reviewer.prepaid_timeout = ledger.reserve(ticket)
                reviewer.prepaid_ticket = ticket
                reviewer.acquisition_started = time.monotonic()
                evidence, manifest = reviewer.evidence(task, artifact)
                if manifest['candidate_hash'] != source_hash:
                    raise VerifierEvidenceError('candidate source changed during automatic review')
                media, manifest = reviewer.group_evidence(evidence, manifest, rule)
                media, manifest = add_boundary_crops(reviewer, media, manifest)
                payload = {'original_task': public, 'criteria': rule, 'evidence_manifest': manifest}
                parsed = reviewer._observe_group(root / f'confirmation-{repeat}.json', payload, media,
                    f'{digest[:12]}/review/{key}/{repeat}', rule, manifest['windows'])
                write_json(parsed_cache, parsed)
                results.append(parsed[name])
            except ReviewBudgetExhausted as exc:
                failures.append(str(exc))
                break
            except VerifierFormatError as exc:
                failures.append(str(exc))
                break
            except (ValueError, VideoApiError) as exc:
                if str(exc).startswith(('verifier media exceeds fixed size budget',
                        'verifier request exceeds fixed byte budget', 'verifier image count exceeds')):
                    failures.append(str(exc))
                    break
                raise
            finally:
                if getattr(reviewer, 'prepaid_ticket', None) is not None:
                    ledger.settle(reviewer.prepaid_ticket, time.monotonic() - reviewer.acquisition_started)
                    reviewer.prepaid_ticket = None
        resolved = len(results) == 2 and not uncertain(results, owner.profile['disagreement_threshold'])
        decision = {'protocol': VERSION, 'criterion': name, 'status': 'resolved' if resolved else 'abstained',
            'initial_observations': initial, 'observations': results, 'errors': failures,
            'wall_seconds': time.monotonic() - started,
            'models': [owner.model, config['secondary_model'] or owner.model],
            'independent_model': bool(config['secondary_model'] and config['secondary_model'] != owner.model),
            'qualification': 'Two blind reviews of enhanced source evidence; agreement is not proof of correctness.'}
        write_json(saved, decision)
        print(f"[conditioning verifier] auto-review criterion={name} status={decision['status']} errors={json.dumps(failures, ensure_ascii=False)} audit={saved}", flush=True)
        audit[name] = decision
        if resolved:
            rows[name] = results
    return audit
