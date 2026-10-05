#!/usr/bin/env python3
"""Loopback-only Studio authentication; credentials are never logged."""
import argparse
import base64
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import sys
import time
from urllib.parse import urlsplit

COOKIE = '__Host-dataset-studio'
LIFETIME = 30 * 24 * 60 * 60
MAX_BODY = 256
STYLE = 'body{margin:0;background:#0a0a0a;color:#ededed;font:16px system-ui;display:grid;place-items:center;min-height:100vh}main{max-width:32rem;padding:2rem}h1{font-size:1.5rem}p{color:#a3a3a3;line-height:1.6}'
SCRIPT = """(()=>{let busy=false;async function exchange(){if(busy)return;const token=location.hash.slice(1);history.replaceState(null,'','/access');const message=document.getElementById('message');if(!/^[a-f0-9]{64}$/.test(token)){message.textContent='Apri il link privato completo per accedere.';return;}busy=true;message.textContent='Accesso in corso…';try{const response=await fetch('/access/session',{method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json'},body:JSON.stringify({token})});if(!response.ok)throw new Error();location.replace('/datasets');}catch{message.textContent='Accesso non riuscito. Riapri il link privato completo.';}finally{busy=false;if(location.hash)exchange();}}addEventListener('hashchange',exchange);exchange();})();"""
LANDING = ('<!doctype html><html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="referrer" content="no-referrer"><title>Accesso privato · Dataset Studio</title><style>' + STYLE + '</style></head><body><main><h1>Accesso privato allo Studio</h1><p id="message" role="status">Accesso in corso…</p></main><script>' + SCRIPT + '</script></body></html>').encode()
def source_hash(source):
    return base64.b64encode(hashlib.sha256(source.encode()).digest()).decode()
CSP = ("default-src 'none'; script-src 'sha256-" + source_hash(SCRIPT) + "'; style-src 'sha256-" + source_hash(STYLE) + "'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")

class Access:
    def __init__(self, secret, clock=time.time):
        if not isinstance(secret, str) or not re.fullmatch(r'[A-Za-z0-9_-]{32,256}', secret):
            raise ValueError('Invalid private authentication configuration')
        self.secret = secret.encode()
        self.clock = clock
        self.link = hmac.digest(self.secret, b'dataset-studio/private-link/v1', 'sha256').hex()
        self.key = hmac.digest(self.secret, b'dataset-studio/session-key/v1', 'sha256')

    def accepts_link(self, token):
        return isinstance(token, str) and bool(re.fullmatch(r'[a-f0-9]{64}', token)) and hmac.compare_digest(token, self.link)

    def session(self):
        issued = int(self.clock())
        value = f'v1.{issued}.{issued + LIFETIME}.{secrets.token_hex(16)}'
        return value + '.' + hmac.digest(self.key, b'dataset-studio/session/v1\n' + value.encode(), 'sha256').hex()

    def accepts_session(self, value):
        if not isinstance(value, str):
            return False
        match = re.fullmatch(r'v1\.([1-9][0-9]{0,11})\.([1-9][0-9]{0,11})\.([a-f0-9]{32})\.([a-f0-9]{64})', value)
        if not match:
            return False
        issued, expires = int(match[1]), int(match[2])
        now = int(self.clock())
        if issued > now or expires != issued + LIFETIME or now >= expires:
            return False
        payload = value.rsplit('.', 1)[0]
        return hmac.compare_digest(match[4], hmac.digest(self.key, b'dataset-studio/session/v1\n' + payload.encode(), 'sha256').hex())

    def accepts_basic(self, header):
        if not isinstance(header, str) or not header.startswith('Basic ') or len(header) > 1024:
            return False
        try:
            raw = base64.b64decode(header[6:], validate=True)
            username, password = raw.split(b':', 1)
            return hmac.compare_digest(username, b'studio') and hmac.compare_digest(password, self.secret)
        except (ValueError, TypeError):
            return False

def handler(access):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def one(self, name):
            values = self.headers.get_all(name, [])
            if len(values) > 1:
                raise ValueError('Duplicate header')
            return values[0] if values else ''

        def reply(self, status, body=b'', headers=None):
            self.send_response(status)
            for key, value in {'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer',
                               'X-Content-Type-Options': 'nosniff', 'Content-Security-Policy': CSP,
                               'Content-Length': str(len(body)), **(headers or {})}.items():
                self.send_header(key, value)
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(body)

        def context(self):
            method = self.one('X-Studio-Method')
            host = self.one('X-Studio-Host')
            proto = self.one('X-Studio-Proto')
            if not re.fullmatch(r'[A-Z]{1,16}', method) or not re.fullmatch(r'[A-Za-z0-9.-]+(?::[0-9]{1,5})?', host) or proto not in ('http', 'https'):
                raise ValueError('Invalid gateway context')
            return method, host, proto

        def authorize(self):
            method, host, proto = self.context()
            origin = self.one('Origin')
            if origin and origin != proto + '://' + host:
                return self.reply(403)
            cookie = self.one('Cookie')
            if len(cookie) > 8192:
                raise ValueError('Cookie too long')
            values = []
            for part in cookie.split(';'):
                name, sep, value = part.strip().partition('=')
                if name == COOKIE:
                    if not sep:
                        raise ValueError('Malformed session')
                    values.append(value)
            if len(values) > 1:
                raise ValueError('Duplicate session')
            basic = access.accepts_basic(self.one('Authorization'))
            session = len(values) == 1 and access.accepts_session(values[0]) and proto == 'https'
            if not (basic or session):
                return self.reply(401, headers={'WWW-Authenticate': 'Basic realm="Dataset Studio"'})
            if not basic and method not in ('GET', 'HEAD') and origin != 'https://' + host:
                return self.reply(403)
            return self.reply(204)

        def exchange(self):
            method, host, proto = self.context()
            if self.command != 'POST' or method != 'POST':
                return self.reply(405, headers={'Allow': 'POST'})
            if proto != 'https' or self.one('Origin') != 'https://' + host:
                return self.reply(403)
            if self.one('Transfer-Encoding'):
                raise ValueError('Unsupported transfer encoding')
            length = self.one('Content-Length')
            if not re.fullmatch(r'[1-9][0-9]{0,2}', length):
                return self.reply(413)
            length = int(length)
            if length > MAX_BODY:
                return self.reply(413)
            if self.one('Content-Type').split(';', 1)[0].strip() != 'application/json':
                return self.reply(415)
            def pairs(values):
                result = {}
                for key, value in values:
                    if key in result:
                        raise ValueError('Duplicate JSON key')
                    result[key] = value
                return result
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError('Incomplete request')
            value = json.loads(body, object_pairs_hook=pairs)
            if not isinstance(value, dict) or set(value) != {'token'} or not access.accepts_link(value['token']):
                return self.reply(401)
            cookie = COOKIE + '=' + access.session() + '; Max-Age=' + str(LIFETIME) + '; Path=/; Secure; HttpOnly; SameSite=Strict'
            return self.reply(204, headers={'Set-Cookie': cookie, 'Location': '/datasets'})

        def dispatch(self):
            self.connection.settimeout(5)
            try:
                if self.path == '/access' and self.command in ('GET', 'HEAD'):
                    return self.reply(200, LANDING, {'Content-Type': 'text/html; charset=utf-8'})
                if self.path == '/access/session':
                    return self.exchange()
                if self.path == '/_studio_auth' and self.command == 'GET':
                    return self.authorize()
                return self.reply(404)
            except (ValueError, KeyError, TypeError, UnicodeError):
                self.reply(400)
            except (OSError, TimeoutError):
                # No credentials, bodies, headers or exception messages are logged.
                self.close_connection = True

        do_GET = dispatch
        do_HEAD = dispatch
        do_POST = dispatch
        do_PUT = dispatch
        do_PATCH = dispatch
        do_DELETE = dispatch
        do_OPTIONS = dispatch
    return Handler

class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        # Never log request context, even for an unexpected protocol exception.
        pass

def server(access, port=8677):
    return QuietServer(('127.0.0.1', port), handler(access))

def write_link(access, origin, destination):
    parsed = urlsplit(origin)
    if not re.fullmatch(r'https://[A-Za-z0-9.-]+(?::[0-9]{1,5})?/?', origin) or parsed.port is not None and parsed.port > 65535 or parsed.scheme != 'https' or not parsed.netloc or parsed.username or parsed.password or parsed.path not in ('', '/') or parsed.query or parsed.fragment:
        raise ValueError('Use a clean HTTPS service origin')
    destination = Path(destination).absolute()
    for parent in [destination, *destination.parents]:
        if parent.is_symlink():
            raise ValueError('Private delivery path contains a symlink')
    with open(destination, 'x', opener=lambda p, flags: os.open(p, flags, 0o600)) as output:
        output.write(origin.rstrip('/') + '/access#' + access.link + '\n')
    return destination

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['serve', 'link'])
    parser.add_argument('--origin')
    parser.add_argument('--output')
    args = parser.parse_args()
    access = Access(os.environ.get('AI_TOOLKIT_AUTH'))
    if args.action == 'serve':
        with server(access) as service:
            service.serve_forever()
    else:
        if not args.origin or not args.output:
            raise ValueError('Private delivery destination required')
        write_link(access, args.origin, args.output)

if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError):
        print('Studio access refused: invalid private configuration or unavailable service.', file=sys.stderr)
        sys.exit(1)
