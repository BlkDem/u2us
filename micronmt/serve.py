import argparse
import hmac
import json
import os
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .api import Translator, load_terminology

MAX_BODY_BYTES = 1 << 20
CREATED = int(time.time())


def _json_error(status, message, error_type='invalid_request_error', code=None):
    return status, {'error': {'message': message, 'type': error_type, 'code': code}}


def _direction_from_name(name):
    if not name:
        return None
    normalized = re.sub(r'[^a-z0-9]+', '-', str(name).lower().replace('2', '-')).strip('-')
    if 'ru-en' in normalized:
        return 'ru'
    if 'en-ru' in normalized:
        return 'en'
    return None


def resolve_direction(payload, model, default_source='ru'):
    for key in ('source', 'source_language', 'source_lang'):
        value = payload.get(key)
        if value in ('ru', 'en'):
            return value
    metadata = payload.get('metadata')
    if isinstance(metadata, dict):
        for key in ('source', 'source_language', 'source_lang'):
            value = metadata.get(key)
            if value in ('ru', 'en'):
                return value
    direction = _direction_from_name(model)
    if direction:
        return direction
    return default_source


def _content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get('text')
                if isinstance(text, str):
                    parts.append(text)
        return '\n'.join(parts)
    return ''


_DECODING_KEYS = ('beam_size', 'length_penalty', 'no_repeat_ngram_size', 'repetition_penalty')


def _decoding_overrides(payload):
    decoding = payload.get('decoding')
    merged = dict(decoding) if isinstance(decoding, dict) else {}
    for key in _DECODING_KEYS:
        if key in payload:
            merged[key] = payload[key]
    overrides = {}
    if 'beam_size' in merged:
        value = merged['beam_size']
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError('beam_size must be an integer >= 1')
        overrides['beam_size'] = value
    if 'no_repeat_ngram_size' in merged:
        value = merged['no_repeat_ngram_size']
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError('no_repeat_ngram_size must be an integer >= 0')
        overrides['no_repeat_ngram_size'] = value
    if 'length_penalty' in merged:
        value = merged['length_penalty']
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ValueError('length_penalty must be a number >= 0')
        overrides['length_penalty'] = float(value)
    if 'repetition_penalty' in merged:
        value = merged['repetition_penalty']
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 1:
            raise ValueError('repetition_penalty must be a number >= 1')
        overrides['repetition_penalty'] = float(value)
    return overrides


def _chat_prompt(messages):
    if not isinstance(messages, list) or not messages:
        return None
    chosen = None
    for message in messages:
        if isinstance(message, dict) and message.get('role') == 'user':
            text = _content_to_text(message.get('content')).strip()
            if text:
                chosen = text
    if chosen:
        return chosen
    for message in reversed(messages):
        if isinstance(message, dict):
            text = _content_to_text(message.get('content')).strip()
            if text:
                return text
    return None


class TranslationServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, translator, model_name='micronmt', api_key=None, default_source='ru'):
        super().__init__(address, TranslationHandler)
        self.translator = translator
        self.model_name = model_name
        self.api_key = api_key
        self.default_source = default_source
        self.lock = threading.Lock()


class TranslationHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'micronmt/0.1'

    def log_message(self, fmt, *args):
        print('%s - %s' % (self.address_string(), fmt % args), flush=True)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path in ('/health', '/healthz'):
            self._send_json(200, {'status': 'ok'})
            return
        if not self._authorized():
            return
        if path == '/v1/models':
            self._send_json(200, {'object': 'list', 'data': self._model_list()})
            return
        if path.startswith('/v1/models/'):
            model_id = path[len('/v1/models/'):]
            for entry in self._model_list():
                if entry['id'] == model_id:
                    self._send_json(200, entry)
                    return
        self._send_json(*_json_error(404, 'Unknown route: ' + path, 'not_found_error'))

    def do_POST(self):
        path = self.path.split('?', 1)[0]
        if not self._authorized():
            return
        if path not in ('/v1/chat/completions', '/v1/completions'):
            self._send_json(*_json_error(404, 'Unknown route: ' + path, 'not_found_error'))
            return
        payload = self._read_json()
        if payload is None:
            return
        try:
            if path == '/v1/chat/completions':
                status, body = self._chat_completion(payload)
            else:
                status, body = self._completion(payload)
        except ValueError as error:
            status, body = _json_error(400, str(error))
        except Exception as error:  # pragma: no cover - defensive
            status, body = _json_error(500, str(error), 'server_error')
        if body is None:
            return
        self._send_json(status, body)

    def _authorized(self):
        expected = self.server.api_key
        if not expected:
            return True
        header = self.headers.get('Authorization', '')
        prefix = 'Bearer '
        token = header[len(prefix):] if header.startswith(prefix) else ''
        if token and hmac.compare_digest(token, expected):
            return True
        self.close_connection = True
        self._send_json(*_json_error(401, 'Missing or invalid API key', 'authentication_error', 'invalid_api_key'))
        return False

    def _read_json(self):
        raw = self.headers.get('Content-Length')
        try:
            length = int(raw) if raw else 0
        except ValueError:
            self._send_json(*_json_error(400, 'Invalid Content-Length'))
            return None
        if length > MAX_BODY_BYTES:
            self._send_json(*_json_error(413, 'Request body too large'))
            return None
        body = self.rfile.read(length) if length else b''
        if not body:
            return {}
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as error:
            self._send_json(*_json_error(400, 'Invalid JSON: ' + str(error)))
            return None
        if not isinstance(payload, dict):
            self._send_json(*_json_error(400, 'JSON body must be an object'))
            return None
        return payload

    def _model_list(self):
        base = self.server.model_name
        return [
            {'id': base, 'object': 'model', 'created': CREATED, 'owned_by': 'micronmt'},
            {'id': 'ru2en', 'object': 'model', 'created': CREATED, 'owned_by': 'micronmt'},
            {'id': 'en2ru', 'object': 'model', 'created': CREATED, 'owned_by': 'micronmt'},
        ]

    def _translate(self, text, payload):
        model = payload.get('model') or self.server.model_name
        source = resolve_direction(payload, model, self.server.default_source)
        overrides = _decoding_overrides(payload)
        with self.server.lock:
            result = self.server.translator.translate(text, source=source, **overrides)
        return model, result

    def _chat_completion(self, payload):
        prompt = _chat_prompt(payload.get('messages'))
        if not prompt:
            raise ValueError('messages must contain at least one non-empty user message')
        model, result = self._translate(prompt, payload)
        created = int(time.time())
        completion_id = 'chatcmpl-' + uuid.uuid4().hex
        usage = self._usage(prompt, result['translation'])
        if payload.get('stream'):
            self._stream_chat(completion_id, created, model, result['translation'], usage)
            return None
        return 200, {
            'id': completion_id,
            'object': 'chat.completion',
            'created': created,
            'model': model,
            'choices': [{
                'index': 0,
                'message': {'role': 'assistant', 'content': result['translation']},
                'finish_reason': 'stop',
            }],
            'usage': usage,
        }

    def _completion(self, payload):
        prompt = payload.get('prompt')
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ''
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('prompt must be a non-empty string')
        model, result = self._translate(prompt, payload)
        created = int(time.time())
        completion_id = 'cmpl-' + uuid.uuid4().hex
        usage = self._usage(prompt, result['translation'])
        if payload.get('stream'):
            self._stream_completion(completion_id, created, model, result['translation'], usage)
            return None
        return 200, {
            'id': completion_id,
            'object': 'text_completion',
            'created': created,
            'model': model,
            'choices': [{'index': 0, 'text': result['translation'], 'finish_reason': 'stop'}],
            'usage': usage,
        }

    def _usage(self, prompt, translation):
        tokenizer = self.server.translator.tokenizer
        prompt_tokens = len(tokenizer.encode(prompt, self.server.translator.max_len))
        completion_tokens = len(tokenizer.encode(translation, self.server.translator.max_len))
        return {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
        }

    def _start_stream(self):
        self.send_response(200)
        self._cors_headers()
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True

    def _event(self, data):
        if isinstance(data, str):
            payload = 'data: ' + data + '\n\n'
        else:
            payload = 'data: ' + json.dumps(data, ensure_ascii=False) + '\n\n'
        self.wfile.write(payload.encode('utf-8'))
        self.wfile.flush()

    def _stream_chat(self, completion_id, created, model, translation, usage):
        self._start_stream()
        base = {'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model}
        self._event({**base, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})
        self._event({**base, 'choices': [{'index': 0, 'delta': {'content': translation}, 'finish_reason': None}]})
        self._event({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': usage})
        self._event('[DONE]')

    def _stream_completion(self, completion_id, created, model, translation, usage):
        self._start_stream()
        base = {'id': completion_id, 'object': 'text_completion', 'created': created, 'model': model}
        self._event({**base, 'choices': [{'index': 0, 'text': translation, 'finish_reason': None}]})
        self._event({**base, 'choices': [{'index': 0, 'text': '', 'finish_reason': 'stop'}], 'usage': usage})
        self._event('[DONE]')

    def _cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Authorization, Content-Type')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')

    def _send_json(self, status, body):
        data = json.dumps(body, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self._cors_headers()
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()


def build_server(checkpoint, host='127.0.0.1', port=8000, device='cpu', model_name='micronmt',
                 default_source='ru', max_len=None, beam_size=1, length_penalty=0.6,
                 no_repeat_ngram_size=0, repetition_penalty=1.0, terminology=None, api_key=None):
    translator = Translator(checkpoint, device=device, max_len=max_len, terminology=terminology,
                            beam_size=beam_size, length_penalty=length_penalty,
                            no_repeat_ngram_size=no_repeat_ngram_size,
                            repetition_penalty=repetition_penalty)
    return TranslationServer((host, port), translator, model_name=model_name,
                             api_key=api_key, default_source=default_source)


def main():
    parser = argparse.ArgumentParser(description='OpenAI-compatible translation HTTP API')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model-name', default='micronmt')
    parser.add_argument('--source', choices=['ru', 'en'], default='ru',
                        help='default source language when the model name does not specify a direction')
    parser.add_argument('--max-len', type=int)
    parser.add_argument('--beam-size', type=int, default=1)
    parser.add_argument('--length-penalty', type=float, default=0.6)
    parser.add_argument('--no-repeat-ngram-size', type=int, default=0)
    parser.add_argument('--repetition-penalty', type=float, default=1.0)
    parser.add_argument('--terms', help='JSON terminology file (non-destructive control hints)')
    parser.add_argument('--api-key', default=os.environ.get('MICRONMT_API_KEY'),
                        help='require Authorization: Bearer <key> (env MICRONMT_API_KEY)')
    args = parser.parse_args()

    terminology = load_terminology(args.terms) if args.terms else None
    server = build_server(args.checkpoint, host=args.host, port=args.port, device=args.device,
                          model_name=args.model_name, default_source=args.source,
                          max_len=args.max_len, beam_size=args.beam_size,
                          length_penalty=args.length_penalty,
                          no_repeat_ngram_size=args.no_repeat_ngram_size,
                          repetition_penalty=args.repetition_penalty,
                          terminology=terminology, api_key=args.api_key)
    host, port = server.server_address[:2]
    print(f'micronmt OpenAI-compatible API listening on http://{host}:{port}', flush=True)
    print(f'  GET  /v1/models', flush=True)
    print(f'  POST /v1/chat/completions', flush=True)
    print(f'  POST /v1/completions', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
