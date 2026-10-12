"""Offline, blinded development review packets; never change experiment scores."""
from copy import deepcopy
from collections import Counter
from hashlib import sha256
from html import escape
import json
from pathlib import Path

from evovideo_skill.research_subgraphs import stable_hash

VERSION = 'predicate-review-pack-v1.1'


def _request_path(root, response):
    path = (root / response).resolve()
    if not path.is_relative_to(root):
        raise ValueError('response path escapes source run')
    attempt = 1 if '.correction-1.' in path.name else 0
    return path.with_name(path.name.split('.')[0] + f'.request-{attempt}.json')


def build_cases(root, audit):
    """Deduplicate requests, not observations into a majority-vote truth."""
    root = Path(root).resolve()
    records = audit['records']
    by_response = {r['response']: r for r in records}
    superseded, invalid, cases = [], [], {}
    for record in records:
        response = record['response']
        correction = response[:-len('.raw.json')] + '.correction-1.raw.json'
        replacement = by_response.get(correction) if '.correction-1.' not in response else None
        if replacement and replacement['status'] not in {'invalid_saved_attempt', 'identity_unresolved'}:
            superseded.append({'response': response, 'correction': correction,
                               'status': record['status']})
            continue
        if record['status'] in {'invalid_saved_attempt', 'identity_unresolved'}:
            invalid.append({'response': response, 'status': record['status'], 'error': record.get('error'),
                            'identity_blocked_criteria': record.get('identity_blocked_criteria')})
            continue
        if not record.get('withheld_facts'):
            continue
        request_path = _request_path(root, response)
        request_bytes = request_path.read_bytes()
        if sha256(request_bytes).hexdigest() != record['request_sha256']:
            raise ValueError('request changed since audit: ' + response)
        response_bytes = (root / response).read_bytes()
        if sha256(response_bytes).hexdigest() != record['response_sha256']:
            raise ValueError('response changed since audit: ' + response)
        request = json.loads(request_bytes)
        public, manifest = request['original_task'], request['evidence_manifest']
        candidate = manifest.get('candidate_hash')
        # Absent media identity must never merge unrelated judgments.
        identity = candidate or 'unverified:' + str(request_path.relative_to(root))
        for key, held in record['withheld_facts'].items():
            definition = held['source']
            case_id = stable_hash([identity, public, key, definition])[:24]
            case = cases.setdefault(case_id, {'case_id': case_id, 'task_id': public.get('task_id'),
                'candidate_hash': candidate, 'fact_id': key, 'requirement': deepcopy(definition),
                'original_task_prompt': public.get('prompt', ''),
                'identity_contract': deepcopy(request.get('identity_contract', {})),
                'variants': [], 'issue_codes': set()})
            case['issue_codes'].update(i['code'] for i in held['issues'])
            case['variants'].append({'response': response, 'request': str(request_path.relative_to(root)),
                'request_sha256': record['request_sha256'], 'response_sha256': record['response_sha256'],
                'view': deepcopy(manifest.get('evaluation_view', {})),
                'original_observation': deepcopy(held['original_observation'])})
    for case in cases.values():
        case['issue_codes'] = sorted(case['issue_codes'])
    return {'cases': list(cases.values()), 'superseded_attempts': superseded,
            'unresolved_attempts': invalid}


def _copy_media(root, output, record, copied):
    """Only hash-verified cached candidate media under this run's media roots."""
    filename = record.get('media_file')
    expected = record.get('image_hash') or record.get('sampled_media_hash')
    if not isinstance(filename, str) or not filename or not isinstance(expected, str) or not expected:
        return {'status': 'missing_media_metadata', 'media_file': filename,
                'missing_fields': [key for key, value in [('media_file', filename), ('media_hash', expected)]
                                   if not isinstance(value, str) or not value]}
    relative_path = Path(filename)
    if relative_path.is_absolute() or '..' in relative_path.parts or '\\' in filename:
        return {'status': 'unsafe_media_path', 'media_file': filename}
    allowed = [root / 'verifier' / 'media', *sorted((root / 'verifier').glob('*/media'))]
    found = []
    for folder in allowed:
        folder = folder.resolve()
        path = (folder / relative_path).resolve()
        if folder.is_relative_to(root.resolve()) and path.is_relative_to(folder) and path.is_file():
            data = path.read_bytes()
            if stable_hash(data.hex()) == expected:
                found.append((path, data))
    if not found:
        return {'status': 'media_missing_or_hash_mismatch', 'media_file': filename}
    path, data = found[0]
    suffix = path.suffix.lower()
    if suffix not in {'.png', '.jpg', '.jpeg', '.webp', '.mp4'}:
        return {'status': 'unsupported_media_type'}
    digest = sha256(data).hexdigest()
    relative = 'media/' + digest + suffix
    if relative not in copied:
        (output / relative).write_bytes(data)
        copied[relative] = {'sha256': digest, 'bytes': len(data), 'source_media_hash': expected}
    return {'status': 'verified', 'file': relative, 'sha256': digest,
            'original_seconds': record.get('source_timestamp_seconds'),
            'label': record.get('media_label', '')}


def export(root, output, audit):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('output must be separate from source run')
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory must be empty; existing labels are never overwritten')
    packet = build_cases(root, audit)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'media').mkdir(exist_ok=True)
    copied, problems = {}, []
    blind = []
    for case in packet['cases']:
        row = {k: deepcopy(case[k]) for k in ('case_id', 'task_id', 'candidate_hash', 'fact_id',
                                            'requirement', 'original_task_prompt', 'identity_contract')}
        row['views'] = []
        seen = set()
        for variant in case['variants']:
            view = variant['view']
            view_id = stable_hash(view)
            if view_id in seen:
                continue
            seen.add(view_id)
            rendered = {'segment_id': view.get('segment_id'),
                        'start_seconds': view.get('start_seconds'), 'end_seconds': view.get('end_seconds'),
                        'frames': [], 'clip': None}
            frames = [*view.get('boundary_frames', []), *view.get('sampled_frames', []),
                      *view.get('evidence_crops', [])]
            for frame in frames:
                media = _copy_media(root, output, frame, copied)
                rendered['frames'].append(media)
                if media['status'] != 'verified':
                    problems.append({'case_id': case['case_id'], **media})
            if view.get('sampled_media_file'):
                rendered['clip'] = _copy_media(root, output, {
                    'media_file': view['sampled_media_file'],
                    'sampled_media_hash': view.get('sampled_media_hash'),
                    'media_label': view.get('media_label', '')}, copied)
                if rendered['clip']['status'] != 'verified':
                    problems.append({'case_id': case['case_id'], **rendered['clip']})
            if not frames and not rendered['clip']:
                problems.append({'case_id': case['case_id'], 'status': 'no_cached_evidence'})
            row['views'].append(rendered)
        blind.append(row)
    labels = {'purpose': 'development_labels_not_heldout_gain', 'protocol': VERSION,
              'packet_fingerprint': stable_hash(blind),
              'labels': [{'case_id': row['case_id'], 'label': None, 'evidence': '',
                          'reviewer': ''} for row in blind],
              'allowed_labels': ['supported', 'contradicted', 'unknown'],
              'instruction': 'Judge the exact source proposition from media. Do not copy model judgments. '
                             'Unjudgeable evidence is unknown. Labels never overwrite saved experiment scores.'}
    (output / 'labels-template.json').write_text(json.dumps(labels, ensure_ascii=False, indent=2))
    (output / 'cases.json').write_text(json.dumps(blind, ensure_ascii=False, indent=2))
    # Model judgments stay outside the blind interface.
    (output / 'model-audit.json').write_text(json.dumps(packet, ensure_ascii=False, indent=2))
    report = {'protocol': VERSION, 'purpose': 'offline_development_evidence_review_not_visual_validation',
        'model_calls': 0, 'source_attempt_counts': audit['attempt_counts'],
        'unique_fact_cases': len(blind),
        'unique_candidate_hashes': len({r['candidate_hash'] for r in blind if r['candidate_hash']}),
        'superseded_attempts': len(packet['superseded_attempts']),
        'unresolved_attempts': len(packet['unresolved_attempts']), 'media_issues': problems,
        'media_issue_count': len(problems),
        'media_issue_counts': dict(Counter(item['status'] for item in problems)),
        'verified_media_files': len(copied),
        'media': copied, 'human_labels_complete': False, 'packet_fingerprint': labels['packet_fingerprint'],
        'qualification': 'Cases are review candidates, not confirmed model errors. Repeated responses do not '
                         'create independent videos. Old judgments are unchanged. Unknown remains a valid label.'}
    (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (output / 'index.html').write_text(render(blind, labels, report), encoding='utf-8')
    return {k: v for k, v in report.items() if k != 'media'}


def summarize_labels(packet_dir, label_file):
    """Summarize human-confirmed development facts, never write verifier scores."""
    packet_dir = Path(packet_dir)
    cases = json.loads((packet_dir / 'cases.json').read_text())
    packet = json.loads((packet_dir / 'model-audit.json').read_text())
    labels = json.loads(Path(label_file).read_text())
    if labels.get('packet_fingerprint') != stable_hash(cases):
        raise ValueError('labels belong to different or changed evidence packet')
    # Verify the actual copied bytes as well as the case metadata. Missing media
    # already flagged at export remains unknown; verified media may not change.
    for case in cases:
        for view in case['views']:
            for medium in [*view['frames'], view.get('clip')]:
                if not medium or medium['status'] != 'verified':
                    continue
                path = (packet_dir / medium['file']).resolve()
                if not path.is_relative_to(packet_dir.resolve()) or not path.is_file():
                    raise ValueError('review packet media missing or outside packet: ' + medium['file'])
                if sha256(path.read_bytes()).hexdigest() != medium['sha256']:
                    raise ValueError('review packet media changed: ' + medium['file'])
    index = {row['case_id']: row for row in cases}
    source = {row['case_id']: row for row in packet['cases']}
    if set(source) != set(index):
        raise ValueError('model audit case IDs differ from evidence packet')
    rows = labels.get('labels')
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise ValueError('labels must be a list of case records')
    ids = [row.get('case_id') for row in rows]
    if any(not isinstance(k, str) for k in ids) or len(set(ids)) != len(ids) or set(ids) != set(index):
        raise ValueError('labels must contain every packet case ID exactly once; unreviewed labels stay null')
    results = []
    for row in rows:
        label = row.get('label')
        if label is not None and label not in ('supported', 'contradicted', 'unknown'):
            raise ValueError('invalid human label')
        if label is not None and any(not isinstance(row.get(k), str) or not row[k].strip()
                                     for k in ('reviewer', 'evidence')):
            raise ValueError('each human label needs reviewer and visible evidence or uncertainty explanation')
        case = source[row['case_id']]
        proposals = sorted({v['original_observation']['basis'] for v in case['variants']})
        claims = {'visible_support': 'supported', 'visible_counterexample': 'contradicted'}
        if label is None:
            status = 'unlabeled'
        elif label == 'unknown':
            status = 'human_unknown'
        elif len(proposals) != 1 or proposals[0] not in claims:
            status = 'mixed_original_claims'
        else:
            status = 'withheld_claim_matches_label' if claims[proposals[0]] == label else 'withheld_claim_disagrees_with_label'
        results.append({'case_id': row['case_id'], 'fact_id': case['fact_id'], 'human_label': label,
                        'original_claims': proposals, 'status': status})
    from collections import Counter
    return {'purpose': 'selected_development_fact_labels_not_general_accuracy', 'model_calls': 0,
        'packet_fingerprint': labels['packet_fingerprint'], 'cases': len(rows),
        'counts': dict(Counter(r['status'] for r in results)), 'results': results,
        'complete': bool(rows) and all(r['human_label'] is not None for r in results),
        'qualification': 'Selected withheld cases are not representative accuracy data. A matching label '
            'does not validate the original rationale. No votes, score updates or heldout gain are inferred.'}


def render(cases, labels, report):
    cards = []
    for row in cases:
        views = []
        for view in row['views']:
            media = []
            clip = view.get('clip')
            if clip and clip['status'] == 'verified':
                media.append(f'<video controls preload="metadata" src="{escape(clip["file"], quote=True)}"></video>')
            for frame in view['frames']:
                if frame['status'] != 'verified':
                    continue
                media.append(f'<figure><img loading="lazy" src="{escape(frame["file"], quote=True)}">'
                    f'<figcaption>{escape(str(frame["original_seconds"]))}s · {escape(frame["label"])}</figcaption></figure>')
            views.append(f'<h3>镜头 {escape(str(view["segment_id"]))} · 原视频 '
                f'{escape(str(view["start_seconds"]))}–{escape(str(view["end_seconds"]))} 秒</h3>'
                '<div class="media">' + ''.join(media) + '</div>')
        cards.append(f'<article data-id="{escape(row["case_id"], quote=True)}">'
            f'<h2>{escape(str(row["task_id"]))} · {escape(row["fact_id"])}</h2>'
            f'<p>视频标识：{escape(str(row["candidate_hash"])[:16])} · 复核项：{escape(row["case_id"])}</p>'
            f'<p class="target">{escape(row["requirement"]["proposition"])}</p>'
            f'<p>阶段：{escape(row["requirement"]["phase"])}。只判断该阶段的明确命题。</p>'
            '<details><summary>源任务与身份定义</summary><pre>' + escape(row['original_task_prompt']) + '\n'
            + escape(json.dumps(row['identity_contract'], ensure_ascii=False, indent=2)) + '</pre></details>'
            + ''.join(views) + '<label>人工标签 <select class="label"><option value="">未标注</option>'
            '<option value="supported">符合</option><option value="contradicted">不符合</option>'
            '<option value="unknown">证据不足 / 看不清</option></select></label>'
            '<textarea class="evidence" placeholder="说明可见事实、原视频时间，以及看不清的原因"></textarea></article>')
    # No model verdicts, confidence or rationale appear in this HTML or its JS.
    encoded = json.dumps(labels, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>事实证据复核</title>
<style>body{font:16px system-ui;margin:2rem auto;max-width:1100px;padding:0 1rem;color:#172332;background:#f4f6f8}
article{background:white;padding:1.5rem;margin:1.5rem 0;border-radius:12px}.target{font-size:1.2rem;font-weight:600}
.media{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;width:250px}img{width:100%}video{width:100%;max-height:500px}
figcaption{font-size:12px;overflow-wrap:anywhere}textarea{display:block;width:95%;min-height:65px;margin-top:12px}
pre{white-space:pre-wrap;overflow-wrap:anywhere}button,select{padding:10px}.warning{background:#fff0cd;padding:1rem}</style>
<h1>事实证据复核</h1><p>一次标注独立事实，复用到开发回归。模型评分与理由已从此页面隐藏。</p>
<p>连接状态不等于张紧状态；操作机构不等于目标对象转动。无法看清时选择“证据不足”。
图片及裁片均来自候选视频，不是目标参考图。视频播放器显示裁片时间，图片标注为原视频时间。</p>
<p class="warning">''' + f'待核对事实 {len(cases)} 项；媒体缺失或校验问题 {len(report["media_issues"])} 项。' + '''
不要用缺失画面推断失败。页面不会自动保存，关闭前请导出标签。标签不会修改旧评分。</p>
<label>标注人 <input id="reviewer"></label><button id="download">导出人工标签 JSON</button>
''' + ''.join(cards) + '''<script>
const template = ''' + encoded + ''';
document.getElementById('download').onclick = () => {
  const result = structuredClone(template);
  for (const entry of result.labels) {
    const card = document.querySelector('[data-id="' + entry.case_id + '"]');
    entry.label = card.querySelector('.label').value || null;
    entry.evidence = card.querySelector('.evidence').value;
    entry.reviewer = document.getElementById('reviewer').value;
    if (entry.label && (!entry.evidence.trim() || !entry.reviewer.trim())) {
      alert('已选择标签的项目需要填写标注人和证据说明；看不清时请写明原因。'); return;
    }
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(result,null,2)], {type:'application/json'}));
  a.download = 'human-labels.json'; a.click(); setTimeout(() => URL.revokeObjectURL(a.href),1000);
};</script></html>'''
