"""Linux image gate: real nginx, original-method CSRF, sessions and fail-closed."""
import argparse
import base64
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
sys.dont_write_bytecode = True

def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

class Backend(BaseHTTPRequestHandler):
    def do_GET(self):
        self.respond(b'private synthetic media')
    def do_POST(self):
        self.respond(json.dumps({key: self.headers.get(key) for key in ['Host', 'Origin', 'X-Forwarded-Proto']}).encode())
    def respond(self, body):
        assert not self.headers.get('Authorization'), 'Gateway leaked Authorization'
        assert not self.headers.get('Cookie'), 'Gateway leaked Cookie'
        assert not self.headers.get('X-Studio-Method'), 'Gateway leaked auth context'
        self.send_response(200); self.end_headers(); self.wfile.write(body)
    def log_message(self, *args): pass

def run(toolkit):
    # Missing binary/module is a failed prerequisite, never a simulated PASS.
    version = subprocess.run(['nginx', '-V'], capture_output=True, text=True, check=True)
    assert '--with-http_auth_request_module' in version.stderr, 'nginx auth_request module unavailable'
    spec = importlib.util.spec_from_file_location('access', toolkit / 'scripts/dataset_studio_access.py')
    access = importlib.util.module_from_spec(spec); spec.loader.exec_module(access)
    now = [1800000000]
    protocol = access.Access('synthetic-gateway-password-' + '0' * 32, lambda: now[0])
    basic = 'Basic ' + base64.b64encode(b'studio:' + protocol.secret).decode()
    backend_port, gateway_port, auth_port = port(), port(), port()
    backend = HTTPServer(('127.0.0.1', backend_port), Backend)
    threading.Thread(target=backend.serve_forever, daemon=True).start()
    service = access.server(protocol, auth_port)
    threading.Thread(target=service.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = Path(__file__).with_name('nginx.conf').read_text()
        config = config.replace('listen 8675;', f'listen 127.0.0.1:{gateway_port};')
        config = config.replace('127.0.0.1:8676', f'127.0.0.1:{backend_port}').replace('127.0.0.1:8677', f'127.0.0.1:{auth_port}')
        config = config.replace('/run/dataset-studio-nginx.pid', str(root / 'nginx.pid'))
        file = root / 'nginx.conf'; file.write_text(config)
        subprocess.run(['nginx', '-t', '-c', str(file)], capture_output=True, check=True)
        process = subprocess.Popen(['nginx', '-c', str(file), '-g', 'daemon off;'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        def request(path, method='GET', headers=None, body=None):
            c = http.client.HTTPConnection('127.0.0.1', gateway_port, timeout=5)
            c.request(method, path, body=body, headers={'Host': 'studio.example.test', 'X-Forwarded-Proto': 'https', **(headers or {})})
            r = c.getresponse(); data = r.read(); result = (r.status, dict(r.getheaders()), data); c.close()
            return result
        def expect(code, *args, **kwargs):
            result = request(*args, **kwargs)
            assert result[0] == code, f'Expected {code}, got {result[0]}'
            return result
        try:
            for attempt in range(50):
                try:
                    with socket.create_connection(('127.0.0.1', gateway_port), timeout=.1): break
                except OSError:
                    assert process.poll() is None, 'nginx exited before gate'
                    time.sleep(.1)
            for route in ['/', '/datasets', '/api/dataset-studio', '/api/img/private.png', '/api/files/private.bin']:
                denied = expect(401, route)
                assert 'WWW-Authenticate' in denied[1]
                expect(401, route, headers={'Authorization': 'Basic ' + base64.b64encode(b'studio:wrong').decode()})
                expect(200, route, headers={'Authorization': basic, 'Cookie': 'unrelated=private'})
            landing = expect(200, '/access')
            assert b'history.replaceState' in landing[2] and protocol.link.encode() not in landing[2]
            assert landing[1]['Referrer-Policy'] == 'no-referrer' and landing[1]['Cache-Control'] == 'no-store'
            assert "default-src 'none'" in landing[1]['Content-Security-Policy']
            expect(404, '/_studio_auth', headers={'Authorization': basic})
            expect(405, '/access/session')
            exchange_headers = {'Origin': 'https://studio.example.test', 'Content-Type': 'application/json'}
            expect(403, '/access/session', 'POST', {'Content-Type': 'application/json'}, json.dumps({'token': protocol.link}))
            expect(403, '/access/session', 'POST', {**exchange_headers, 'Origin': 'https://evil.example'}, '{}')
            expect(401, '/access/session', 'POST', exchange_headers, json.dumps({'token': 'f' * 64}))
            expect(413, '/access/session', 'POST', exchange_headers, 'x' * 257)
            exchange = expect(204, '/access/session', 'POST', exchange_headers, json.dumps({'token': protocol.link}))
            assert exchange[1]['Location'] == '/datasets'
            value = exchange[1]['Set-Cookie']; cookie = value.split(';')[0]
            for flag in ['__Host-dataset-studio=', 'Max-Age=2592000', 'Path=/', 'Secure', 'HttpOnly', 'SameSite=Strict']:
                assert flag in value
            assert 'Domain=' not in value
            for route in ['/datasets', '/api/dataset-studio', '/api/img/private.png', '/api/files/private.bin']:
                expect(200, route, headers={'Cookie': cookie + '; unrelated=private'})
            expect(401, '/datasets', headers={'Cookie': cookie[:-1] + ('a' if cookie[-1] != 'a' else 'b')})
            assert request('/datasets', headers={'Cookie': cookie + '; ' + cookie})[0] >= 400
            expect(401, '/datasets', headers={'Cookie': cookie, 'X-Forwarded-Proto': 'http'})
            expect(403, '/api/dataset-studio', 'POST', {'Cookie': cookie}, '{}')
            expect(403, '/api/dataset-studio', 'POST', {'Cookie': cookie, 'Origin': 'https://evil.example'}, '{}')
            forwarded = expect(200, '/api/dataset-studio', 'POST', {'Cookie': cookie, 'Origin': 'https://studio.example.test'}, '{}')
            assert json.loads(forwarded[2]) == {'Host': 'studio.example.test', 'Origin': 'https://studio.example.test', 'X-Forwarded-Proto': 'https'}
            expect(200, '/api/dataset-studio', 'POST', {'Authorization': basic}, '{}')
            expect(403, '/api/dataset-studio', 'POST', {'Authorization': basic, 'Origin': 'https://evil.example'}, '{}')
            expect(403, '/api/dataset-studio', 'POST', {'Cookie': cookie, 'X-Studio-Method': 'GET', 'X-Studio-Host': 'evil.example', 'X-Studio-Proto': 'https'}, '{}')
            expect(400, '/datasets', headers={'Cookie': cookie, 'X-Forwarded-Proto': 'https,http'})
            expect(403, '/api/settings', 'POST', {'Cookie': cookie, 'Origin': 'https://studio.example.test'}, '{}')
            expect(403, '/api/settings', 'POST', {'Authorization': basic}, '{}')
            now[0] += access.LIFETIME
            expect(401, '/datasets', headers={'Cookie': cookie})
            now[0] -= access.LIFETIME
            # Real service death denies both valid Basic and signed sessions.
            service.shutdown(); service.server_close()
            assert request('/datasets', headers={'Cookie': cookie})[0] >= 500
            assert request('/api/img/private.png', headers={'Authorization': basic})[0] >= 500
            service = access.server(protocol, auth_port)
            threading.Thread(target=service.serve_forever, daemon=True).start()
            expect(200, '/datasets', headers={'Cookie': cookie})
            print('Real Linux nginx gate PASS: private routes, secure session, original-method CSRF, Basic compatibility, credential stripping, restart and fail-closed')
        finally:
            process.terminate(); process.wait(timeout=5)
            service.shutdown(); service.server_close(); backend.shutdown(); backend.server_close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--toolkit', type=Path, default=Path(__file__).resolve().parents[2])
    run(parser.parse_args().toolkit)
