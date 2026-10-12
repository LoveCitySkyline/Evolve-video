"""Review-packet provenance and offline workflow tests, not visual calibration."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from evovideo_skill.predicate_review_pack import build_cases, export, _copy_media, summarize_labels
from evovideo_skill.research_subgraphs import stable_hash
from evovideo_skill.review_observations import wire_contract
import test_predicate_evidence as fixtures


class PredicateReviewPackTests(unittest.TestCase):
    def setup_run(self, root, duplicate=True):
        fx = fixtures.PredicateEvidenceTests()
        public, plan, manifest, raw = fx.fixture('shared-observation-review-v3')
        public['task_id'] = 'unit'; public['prompt'] = '<script>alert("source")</script>'
        manifest['candidate_hash'] = 'candidate-one'
        view = manifest['evaluation_view']
        view.update(start_seconds=12., end_seconds=18.)
        media = root/'verifier/final/media'; media.mkdir(parents=True)
        for i, frame in enumerate([*view['boundary_frames'], *view['sampled_frames']]):
            data = f'fixture image {i}'.encode()
            frame['media_file'] = f'frame{i}.png'
            frame['image_hash'] = stable_hash(data.hex())
            (media/frame['media_file']).write_bytes(data)
        request = {'original_task': public, 'evidence_manifest': manifest,
                   'criteria': plan['rules'], 'output_contract': wire_contract(plan)}
        folder = root/'verifier/final/judgments/one'; folder.mkdir(parents=True)
        pairs = [(folder/'group-001-repeat-0', request, raw)]
        if duplicate:
            review = folder/'auto_review/one'; review.mkdir(parents=True)
            pairs.append((review/'confirmation-0', request, raw))
        for path, req, response in pairs:
            Path(str(path)+'.request-0.json').write_text(json.dumps(req))
            Path(str(path)+'.raw.json').write_text(json.dumps(response))
        script = Path(__file__).parents[1]/'scripts/audit_predicate_evidence.py'
        spec = importlib.util.spec_from_file_location('audit_pack_fixture', script)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module, folder

    def test_duplicate_attempts_become_one_fact_case_not_votes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            module, _ = self.setup_run(root)
            result = build_cases(root, module.audit(root))
            self.assertEqual(len(result['cases']), 4)
            self.assertTrue(all(len(c['variants']) == 2 for c in result['cases']))
            self.assertEqual(len({c['candidate_hash'] for c in result['cases']}), 1)

    def test_different_media_or_requirements_never_share_labels(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            module, folder = self.setup_run(root)
            path = folder/'auto_review/one/confirmation-0.request-0.json'
            request = json.loads(path.read_text())
            request['evidence_manifest']['candidate_hash'] = 'other-candidate'
            path.write_text(json.dumps(request))
            result = build_cases(root, module.audit(root))
            self.assertEqual(len(result['cases']), 8)
            request['evidence_manifest']['candidate_hash'] = 'candidate-one'
            request['original_task']['prompt'] = 'A different source task'
            path.write_text(json.dumps(request))
            self.assertEqual(len(build_cases(root, module.audit(root))['cases']), 8)

    def test_explicit_valid_correction_supersedes_initial_format_error_only(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            module, folder = self.setup_run(root, duplicate=False)
            request = folder/'group-001-repeat-0.request-0.json'
            raw = folder/'group-001-repeat-0.raw.json'
            Path(str(request).replace('request-0', 'request-1')).write_bytes(request.read_bytes())
            Path(str(raw).replace('.raw.json', '.correction-1.raw.json')).write_bytes(raw.read_bytes())
            bad = json.loads(raw.read_text()); bad['facts']['s2:post:drive.state']['evidence_refs']=['s9:first']
            raw.write_text(json.dumps(bad))
            result = build_cases(root, module.audit(root))
            self.assertEqual(len(result['superseded_attempts']), 1)
            self.assertEqual(result['unresolved_attempts'], [])
            self.assertEqual(len(result['cases']), 4)
            Path(str(raw).replace('.raw.json', '.correction-1.raw.json')).write_text(json.dumps(bad))
            result = build_cases(root, module.audit(root))
            self.assertEqual(len(result['unresolved_attempts']), 2)
            self.assertEqual(result['superseded_attempts'], [])

    def test_export_blinds_model_answers_verifies_media_and_preserves_old_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)/'source'; root.mkdir()
            output = Path(tmp)/'pack'
            module, _ = self.setup_run(root)
            before = {str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            result = export(root, output, module.audit(root))
            self.assertEqual(result['unique_fact_cases'], 4)
            self.assertEqual(result['unique_candidate_hashes'], 1)
            self.assertFalse(result['media_issues'])
            self.assertEqual(len(list((output/'media').iterdir())), 4)
            html = (output/'index.html').read_text()
            self.assertNotIn('therefore not slack', html)
            self.assertNotIn('original_observation', html)
            self.assertNotIn('<script>alert', html)
            labels = json.loads((output/'labels-template.json').read_text())
            self.assertTrue(all(r['label'] is None for r in labels['labels']))
            self.assertIn('therefore not slack', (output/'model-audit.json').read_text())
            self.assertEqual(before, {str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})
            with self.assertRaisesRegex(ValueError, 'empty'):
                export(root, output, module.audit(root))

    def test_missing_media_not_replaced_and_no_outside_reads_or_source_output(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)/'source'; root.mkdir()
            module, _ = self.setup_run(root)
            media = root/'verifier/final/media/frame0.png'; media.write_bytes(b'changed')
            output = Path(tmp)/'pack'
            result = export(root, output, module.audit(root))
            self.assertTrue(result['media_issues'])
            self.assertEqual(len(list((output/'media').iterdir())), 3)
            self.assertEqual(_copy_media(root, output, {'media_file':'../secret.png','image_hash':'x'}, {})['status'],
                             'missing_media_metadata')
            with self.assertRaisesRegex(ValueError, 'separate'):
                export(root, root/'pack', module.audit(root))

    def test_changed_request_and_response_rejected(self):
        for suffix in ('.raw.json', '.request-0.json'):
            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                module, folder = self.setup_run(root)
                result = module.audit(root)
                path = folder/('group-001-repeat-0'+suffix)
                path.write_text(path.read_text()+' ')
                with self.assertRaisesRegex(ValueError, 'changed since audit'):
                    build_cases(root, result)

    def test_labels_are_bound_to_packet_unknown_is_not_zero_and_cases_are_not_votes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)/'source'; root.mkdir()
            output = Path(tmp)/'pack'
            module, _ = self.setup_run(root)
            export(root, output, module.audit(root))
            labels = json.loads((output/'labels-template.json').read_text())
            self.assertEqual(summarize_labels(output, output/'labels-template.json')['counts'], {'unlabeled': 4})
            for i, label in enumerate(labels['labels']):
                label.update(label='unknown' if i == 0 else 'supported', reviewer='fixture-human', evidence='fixture-only label')
            path = output/'human-labels.json'; path.write_text(json.dumps(labels))
            before = (output/'model-audit.json').read_bytes()
            result = summarize_labels(output, path)
            self.assertTrue(result['complete'])
            self.assertEqual(result['cases'], 4)
            self.assertEqual(sum(result['counts'].values()), 4)
            self.assertEqual(result['counts']['human_unknown'], 1)
            self.assertEqual(before, (output/'model-audit.json').read_bytes())
            labels['labels'][0]['reviewer'] = ''; path.write_text(json.dumps(labels))
            with self.assertRaisesRegex(ValueError, 'reviewer'):
                summarize_labels(output, path)
            labels['packet_fingerprint'] = 'wrong'; path.write_text(json.dumps(labels))
            with self.assertRaisesRegex(ValueError, 'different or changed'):
                summarize_labels(output, path)

    def test_duplicate_case_labels_and_modified_media_manifest_are_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)/'source'; root.mkdir()
            output = Path(tmp)/'pack'
            module, _ = self.setup_run(root)
            export(root, output, module.audit(root))
            labels = json.loads((output/'labels-template.json').read_text())
            labels['labels'][1] = deepcopy(labels['labels'][0])
            path = output/'bad-labels.json'; path.write_text(json.dumps(labels))
            with self.assertRaisesRegex(ValueError, 'exactly once'):
                summarize_labels(output, path)
            cases = json.loads((output/'cases.json').read_text())
            cases[0]['requirement']['proposition'] = 'different target'
            (output/'cases.json').write_text(json.dumps(cases))
            with self.assertRaisesRegex(ValueError, 'different or changed'):
                summarize_labels(output, output/'labels-template.json')

    def test_one_command_cli_allocates_output_without_extra_arguments(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)/'source'; root.mkdir()
            self.setup_run(root)
            repo = Path(__file__).parents[1]
            result = subprocess.run([sys.executable, str(repo/'scripts/prepare_predicate_review.py'),
                                     '--run-dir', str(root)], env={**os.environ, 'PYTHONPATH': str(repo/'src')},
                                    capture_output=True, text=True, check=True)
            summary = json.loads(result.stdout)
            self.assertEqual(summary['model_calls'], 0)
            self.assertEqual(summary['unique_fact_cases'], 4)
            self.assertTrue(Path(summary['report']).is_file())
            self.assertIn('summarize_predicate_labels.py --review-dir', summary['after_saving_human_labels_command'])

    def test_label_import_rechecks_actual_media_bytes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)/'source'; root.mkdir()
            output = Path(tmp)/'pack'
            module, _ = self.setup_run(root)
            export(root, output, module.audit(root))
            medium = next((output/'media').iterdir())
            medium.write_bytes(b'different picture')
            with self.assertRaisesRegex(ValueError, 'media changed'):
                summarize_labels(output, output/'labels-template.json')
            medium.unlink()
            with self.assertRaisesRegex(ValueError, 'media missing'):
                summarize_labels(output, output/'labels-template.json')


if __name__ == '__main__':
    unittest.main()
