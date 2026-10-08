import importlib.util
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('http_diagnostics',
    ROOT / 'scripts/run_conditioning_with_http_diagnostics.py')
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


class HttpDiagnosticsTests(unittest.TestCase):
    def test_http_400_cause_keeps_provider_reason_and_redacts_sensitive_fields(self):
        body = {'error': {'code': 'InvalidParameter', 'message':
            'Video rejected: secret-value https://signed.test/a?token=other data:video/mp4;base64,AAAA',
            'request_body': 'must not be logged'}, 'request_id': 'r-123'}
        error = urllib.error.HTTPError('https://endpoint.test', 400, 'bad', {},
                                      io.BytesIO(json.dumps(body).encode()))
        outer = RuntimeError('wrapped')
        outer.__cause__ = error
        with patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'secret-value'}):
            result = script.error_details(outer)
        self.assertEqual(result['http_code'], 400)
        self.assertEqual(result['provider_error']['code'], 'InvalidParameter')
        self.assertEqual(result['request_id'], 'r-123')
        serialized = json.dumps(result)
        for private in ('secret-value', 'signed.test', 'AAAA', 'must not be logged'):
            self.assertNotIn(private, serialized)

    def test_body_reads_are_bounded_and_html_not_logged(self):
        for body, status in ((b'x' * (script.BODY_LIMIT + 10), 'too_large_to_parse_safely'),
                             (b'<html>private proxy page</html>', 'empty_unreadable_or_non_json')):
            stream = io.BytesIO(body)
            error = urllib.error.HTTPError('url', 400, 'bad', {}, stream)
            result = script.error_details(error)
            self.assertEqual(result['body_status'], status)
            self.assertLessEqual(stream.tell(), script.BODY_LIMIT + 1)
            self.assertNotIn('private', json.dumps(result))

    def test_success_identity_and_failure_identity_preserved_without_retry(self):
        class Fake:
            model = 'test-model'
            calls = 0
            def request(self, *args):
                self.calls += 1
                if isinstance(self.result, Exception):
                    raise self.result
                return self.result
        instance = Fake()
        original = Fake.request
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'http.jsonl'
            with script.diagnose_requests(Fake, path), patch('sys.stderr', io.StringIO()):
                instance.result = {'criteria': {'failed': 0}}
                self.assertIs(instance.request('prompt', ['evidence'], 'op'), instance.result)
                self.assertFalse(path.exists())
                instance.result = urllib.error.URLError('connection refused')
                with self.assertRaises(urllib.error.URLError) as caught:
                    instance.request('prompt', ['evidence'], 'op')
                self.assertIs(caught.exception, instance.result)
                self.assertEqual(instance.calls, 2)
                self.assertEqual(json.loads(path.read_text())['reason'], 'connection refused')
            self.assertIs(Fake.request, original)

    def test_real_verifier_400_gets_one_call_and_unchanged_failure(self):
        from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier
        from evovideo_skill.api_tools import VideoApiError
        verifier = object.__new__(ConditioningVideoVerifier)
        verifier.model = 'test-model'
        verifier.profile = dict(api_key_env='DIAGNOSTIC_KEY', transport='dashscope_video',
            model='test-model', base_url='https://example.test/v1', fps=2,
            max_request_bytes=100000, max_attempts=2, timeout_seconds=180)
        with TemporaryDirectory() as tmp:
            verifier.root = Path(tmp)
            destination = verifier.root / 'http.jsonl'
            body = b'{"error":{"code":"InvalidParameter","message":"bad video"}}'
            error = urllib.error.HTTPError('url', 400, 'bad', {}, io.BytesIO(body))
            with patch.dict(os.environ, {'DIAGNOSTIC_KEY': 'fixture-key'}), \
                    patch('urllib.request.urlopen', side_effect=error) as call, \
                    patch('sys.stderr', io.StringIO()), script.diagnose_requests(ConditioningVideoVerifier, destination):
                with self.assertRaises(VideoApiError) as caught:
                    verifier.request('prompt', [], 'job/0/0')
                self.assertIs(caught.exception.__cause__, error)
                self.assertEqual(call.call_count, 1)
                request = call.call_args.args[0]
                payload = json.loads(request.data)
                self.assertEqual(payload['messages'][1]['content'], [{'type': 'text', 'text': 'prompt'}])
                self.assertEqual(call.call_args.kwargs['timeout'], 180)
                self.assertEqual(json.loads(destination.read_text())['provider_error']['message'], 'bad video')

    def test_main_forwards_arguments_and_restores_runner_and_argv(self):
        from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'checkpoint.json').write_text('{}')
            args = ['--config', 'configs/example.json', '--output-dir', tmp,
                    '--continue', '--phase', 'all', '--candidate-review-policy', 'skip-experiment']
            previous = script.sys.argv
            original = ConditioningVideoVerifier.request
            def run():
                self.assertEqual(script.sys.argv[1:], args)
                self.assertIsNot(ConditioningVideoVerifier.request, original)
            with patch('evovideo_skill.conditioning_runner.main', side_effect=run) as runner:
                script.main(args)
                runner.assert_called_once_with()
            self.assertIs(script.sys.argv, previous)
            self.assertIs(ConditioningVideoVerifier.request, original)
