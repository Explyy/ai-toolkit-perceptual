"""Linux image smoke: real nginx authentication includes legacy media routes."""
import base64
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request

def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

class Backend(BaseHTTPRequestHandler):
    def do_GET(self):
        assert not self.headers.get('Authorization'), 'Gateway leaked browser credentials'
        self.send_response(200); self.end_headers(); self.wfile.write(b'private synthetic media')
    def do_POST(self):
        assert not self.headers.get('Authorization'), 'Gateway leaked browser credentials'
        headers = {key: self.headers.get(key) for key in ['Host', 'Origin', 'X-Forwarded-Proto']}
        self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(headers).encode())
    def log_message(self, *args): pass

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    password = 'synthetic-gateway-password-0000000000000'
    hashed = subprocess.run(['openssl', 'passwd', '-6', '-stdin'], input=password + '\n', text=True, capture_output=True, check=True).stdout.strip()
    auth = root / 'auth'; auth.write_text('studio:' + hashed + '\n')
    backend_port, gateway_port = port(), port()
    backend = HTTPServer(('127.0.0.1', backend_port), Backend)
    threading.Thread(target=backend.serve_forever, daemon=True).start()
    config = Path(__file__).with_name('nginx.conf').read_text()
    config = config.replace('listen 8675;', f'listen 127.0.0.1:{gateway_port};').replace('127.0.0.1:8676', f'127.0.0.1:{backend_port}')
    config = config.replace('/run/dataset-studio-auth', str(auth)).replace('/run/dataset-studio-nginx.pid', str(root / 'nginx.pid'))
    file = root / 'nginx.conf'; file.write_text(config)
    process = subprocess.Popen(['nginx', '-c', str(file), '-g', 'daemon off;'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for attempt in range(50):
            try:
                with socket.create_connection(('127.0.0.1', gateway_port), timeout=.1): break
            except OSError: time.sleep(.1)
        token = base64.b64encode(('studio:' + password).encode()).decode()
        for route in ['/', '/api/dataset-studio', '/api/img/private.png', '/api/files/private.bin']:
            request = urllib.request.Request(f'http://127.0.0.1:{gateway_port}' + route)
            try:
                urllib.request.urlopen(request, timeout=3)
                raise AssertionError('Unauthenticated route allowed: ' + route)
            except urllib.error.HTTPError as e: assert e.code == 401
            request.add_header('Authorization', 'Basic ' + base64.b64encode(b'studio:wrong-password').decode())
            try:
                urllib.request.urlopen(request, timeout=3)
                raise AssertionError('Incorrect password allowed')
            except urllib.error.HTTPError as e: assert e.code == 401
            request.add_header('Authorization', 'Basic ' + token)
            with urllib.request.urlopen(request, timeout=3) as response: assert response.status == 200
        request = urllib.request.Request(f'http://127.0.0.1:{gateway_port}/api/settings', data=b'{}', headers={'Authorization': 'Basic ' + token})
        try:
            urllib.request.urlopen(request, timeout=3)
            raise AssertionError('Cloud settings mutation allowed')
        except urllib.error.HTTPError as e: assert e.code == 403
        request = urllib.request.Request(f'http://127.0.0.1:{gateway_port}/api/dataset-studio', data=b'{}',
                    headers={'Authorization': 'Basic ' + token, 'Host': 'studio.example.test',
                             'Origin': 'https://studio.example.test', 'X-Forwarded-Proto': 'https'})
        with urllib.request.urlopen(request, timeout=3) as response:
            assert json.load(response) == {'Host': 'studio.example.test', 'Origin': 'https://studio.example.test', 'X-Forwarded-Proto': 'https'}
        request.add_header('X-Forwarded-Proto', 'https,http')
        try:
            urllib.request.urlopen(request, timeout=3)
            raise AssertionError('Ambiguous forwarded protocol allowed')
        except urllib.error.HTTPError as e: assert e.code == 400
        print('Gateway smoke passed: all UI/API/media require authentication; server secrets stay private; cloud settings writes denied')
    finally:
        process.terminate(); process.wait(timeout=5); backend.shutdown()
