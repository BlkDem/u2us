import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import torch

from micronmt.checkpoint import tokenizer_metadata
from micronmt.model import MicroTransformer
from micronmt.serve import build_server, resolve_direction
from micronmt.tokenizer import SimpleTokenizer


def _checkpoint(directory):
    tokenizer = SimpleTokenizer.from_texts(['hello world привет мир'])
    config = dict(d_model=8, nhead=2, encoder_layers=1, decoder_layers=1,
                  dim_feedforward=16, dropout=0.0, max_len=12)
    model = MicroTransformer(tokenizer.vocab_size, **config).eval()
    path = Path(directory) / 'model.pt'
    torch.save(dict(model=model.state_dict(), config=config,
                    tokenizer=tokenizer_metadata(tokenizer)), path)
    return path


class ServeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = _checkpoint(cls.tmp.name)
        cls.server = build_server(str(path), host='127.0.0.1', port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = 'http://127.0.0.1:%d' % cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.tmp.cleanup()

    def post(self, path, payload, headers=None, base=None):
        request = urllib.request.Request(
            (base or self.base) + path, data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json', **(headers or {})}, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, response.read().decode('utf-8')
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode('utf-8')

    def get(self, path, headers=None, base=None):
        request = urllib.request.Request((base or self.base) + path, headers=headers or {})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, response.read().decode('utf-8')
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode('utf-8')

    def test_health(self):
        status, body = self.get('/health')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {'status': 'ok'})

    def test_models(self):
        status, body = self.get('/v1/models')
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload['object'], 'list')
        ids = {model['id'] for model in payload['data']}
        self.assertIn('micronmt', ids)
        self.assertIn('ru2en', ids)
        self.assertIn('en2ru', ids)

    def test_chat_completion(self):
        status, body = self.post('/v1/chat/completions', {
            'model': 'ru2en',
            'messages': [{'role': 'user', 'content': 'привет'}],
        })
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload['object'], 'chat.completion')
        self.assertEqual(payload['choices'][0]['message']['role'], 'assistant')
        self.assertIsInstance(payload['choices'][0]['message']['content'], str)
        self.assertEqual(payload['choices'][0]['finish_reason'], 'stop')
        self.assertEqual(payload['usage']['total_tokens'],
                         payload['usage']['prompt_tokens'] + payload['usage']['completion_tokens'])

    def test_chat_completion_content_parts(self):
        status, body = self.post('/v1/chat/completions', {
            'model': 'micronmt',
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'hello'}]}],
        })
        self.assertEqual(status, 200)
        self.assertIsInstance(json.loads(body)['choices'][0]['message']['content'], str)

    def test_chat_completion_stream(self):
        status, body = self.post('/v1/chat/completions', {
            'model': 'en2ru',
            'stream': True,
            'messages': [{'role': 'user', 'content': 'hello world'}],
        })
        self.assertEqual(status, 200)
        events = [line for line in body.splitlines() if line.startswith('data: ')]
        self.assertGreaterEqual(len(events), 3)
        self.assertEqual(events[-1], 'data: [DONE]')
        for event in events[:-1]:
            payload = json.loads(event[len('data: '):])
            self.assertEqual(payload['object'], 'chat.completion.chunk')

    def test_legacy_completion(self):
        status, body = self.post('/v1/completions', {'model': 'ru2en', 'prompt': 'мир'})
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload['object'], 'text_completion')
        self.assertIsInstance(payload['choices'][0]['text'], str)

    def test_validation_errors(self):
        status, body = self.post('/v1/chat/completions', {'model': 'ru2en', 'messages': []})
        self.assertEqual(status, 400)
        self.assertIn('error', json.loads(body))
        status, body = self.post('/v1/completions', {'prompt': ''})
        self.assertEqual(status, 400)
        status, body = self.post('/v1/completions', {'prompt': 'x'}, headers={})
        self.assertEqual(status, 200)

    def test_unknown_route(self):
        status, _ = self.get('/v1/unknown')
        self.assertEqual(status, 404)

    def test_api_key(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = _checkpoint(tmp.name)
        server = build_server(str(path), host='127.0.0.1', port=0, api_key='secret')
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = 'http://127.0.0.1:%d' % server.server_address[1]
        status, _ = self.get('/v1/models', base=base)
        self.assertEqual(status, 401)
        status, _ = self.get('/v1/models', headers={'Authorization': 'Bearer secret'}, base=base)
        self.assertEqual(status, 200)
        status, _ = self.get('/health', base=base)
        self.assertEqual(status, 200)

    def test_resolve_direction(self):
        self.assertEqual(resolve_direction({}, 'ru2en'), 'ru')
        self.assertEqual(resolve_direction({}, 'en2ru'), 'en')
        self.assertEqual(resolve_direction({}, 'micronmt-en-ru'), 'en')
        self.assertEqual(resolve_direction({}, 'micronmt', default_source='en'), 'en')
        self.assertEqual(resolve_direction({'source': 'en'}, 'ru2en'), 'en')
        self.assertEqual(resolve_direction({'metadata': {'source': 'ru'}}, 'en2ru'), 'ru')


if __name__ == '__main__':
    unittest.main()
