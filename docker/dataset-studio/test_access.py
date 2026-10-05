"""Actual loopback HTTP protocol tests; only synthetic credentials, no data."""
import base64
import http.client
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('access', REPO / 'scripts/dataset_studio_access.py')
access = importlib.util.module_from_spec(spec); spec.loader.exec_module(access)

class AccessTests(unittest.TestCase):
    def setUp(self):
        self.now = 1800000000
        self.protocol = access.Access('synthetic-access-password-' + '0' * 32, lambda: self.now)
        self.service = access.server(self.protocol, 0)
        self.thread = threading.Thread(target=self.service.serve_forever, daemon=True); self.thread.start()
        self.addCleanup(self.service.server_close); self.addCleanup(self.service.shutdown)
        self.context = {'X-Studio-Host': 'studio.example.test', 'X-Studio-Proto': 'https', 'X-Studio-Method': 'POST'}
        self.basic = 'Basic ' + base64.b64encode(b'studio:' + self.protocol.secret).decode()

    def request(self, path, method='GET', body=None, headers=None):
        c = http.client.HTTPConnection('127.0.0.1', self.service.server_port, timeout=3)
        c.request(method, path, body=body, headers=headers or {})
        r = c.getresponse(); data = r.read(); result = (r.status, dict(r.getheaders()), data); c.close()
        return result

    def exchange(self, token=None, **headers):
        return self.request('/access/session', 'POST', json.dumps({'token': self.protocol.link if token is None else token}),
                            {**self.context, 'Origin': 'https://studio.example.test', 'Content-Type': 'application/json', **headers})

    def authorize(self, cookie='', method='GET', **headers):
        return self.request('/_studio_auth', headers={**self.context, 'X-Studio-Method': method, 'Cookie': cookie, **headers})[0]

    def test_real_http_exchange_secure_cookie_and_refresh(self):
        status, headers, _ = self.exchange()
        self.assertEqual(status, 204); self.assertEqual(headers['Location'], '/datasets')
        value = headers['Set-Cookie']; cookie = value.split(';')[0]
        self.assertTrue(cookie.startswith(access.COOKIE + '='))
        for flag in ['Max-Age=2592000', 'Path=/', 'Secure', 'HttpOnly', 'SameSite=Strict']:
            self.assertIn(flag, value)
        self.assertNotIn('Domain=', value)
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertEqual(headers['Referrer-Policy'], 'no-referrer')
        self.assertEqual(self.authorize(cookie), 204)
        self.assertEqual(self.authorize(cookie, method='HEAD'), 204)
        self.assertEqual(self.authorize(), 401)
        self.assertEqual(self.exchange('f' * 64)[0], 401)

    def test_expiry_future_tampering_duplicates_and_rotation_fail_closed(self):
        cookie = access.COOKIE + '=' + self.protocol.session()
        self.assertEqual(self.authorize(cookie), 204)
        self.assertEqual(self.authorize(cookie + '; ' + cookie), 400)
        self.assertEqual(self.authorize(cookie[:-1] + ('a' if cookie[-1] != 'a' else 'b')), 401)
        self.now += access.LIFETIME
        self.assertEqual(self.authorize(cookie), 401)
        future = self.protocol.session(); self.now -= 1
        self.assertEqual(self.authorize(access.COOKIE + '=' + future), 401)
        restarted = access.Access(self.protocol.secret.decode(), lambda: 1800000000)
        self.assertEqual(restarted.link, self.protocol.link)
        self.assertTrue(restarted.accepts_session(cookie.split('=', 1)[1]))
        changed = access.Access('another-synthetic-secret-' + '1' * 32, lambda: 1800000000)
        self.assertFalse(changed.accepts_link(self.protocol.link))
        self.assertFalse(changed.accepts_session(cookie.split('=', 1)[1]))
        self.assertNotEqual(self.protocol.link, self.protocol.key.hex())

    def test_cookie_mutations_require_origin_basic_preserves_nonbrowser_operators(self):
        cookie = access.COOKIE + '=' + self.protocol.session()
        self.assertEqual(self.authorize(cookie, method='POST'), 403)
        self.assertEqual(self.authorize(cookie, method='POST', Origin='https://evil.example'), 403)
        self.assertEqual(self.authorize(cookie, method='POST', Origin='https://studio.example.test'), 204)
        self.assertEqual(self.authorize(method='POST', Authorization=self.basic), 204)
        self.assertEqual(self.authorize(method='POST', Authorization=self.basic, Origin='https://evil.example'), 403)
        self.assertEqual(self.authorize(method='POST', Authorization=self.basic, Origin='https://studio.example.test'), 204)
        self.assertEqual(self.authorize(cookie, method='POST', **{'X-Forwarded-Proto': 'https', 'X-Forwarded-Host': 'evil.example'}), 403)
        self.assertEqual(self.authorize(cookie, **{'X-Studio-Proto': 'http'}), 401)

    def test_exchange_limits_origin_methods_content_and_duplicate_payload(self):
        self.assertEqual(self.exchange(Origin='')[0], 403)
        self.assertEqual(self.exchange(Origin='https://evil.example')[0], 403)
        self.assertEqual(self.exchange(**{'X-Studio-Proto': 'http'})[0], 403)
        self.assertEqual(self.request('/access/session', headers=self.context)[0], 405)
        headers = {**self.context, 'Origin': 'https://studio.example.test', 'Content-Type': 'application/json'}
        for body, code in [('x' * 257, 413), ('{"token":"x","token":"x"}', 400), ('{}', 401), ('not JSON', 400)]:
            self.assertEqual(self.request('/access/session', 'POST', body, headers)[0], code)
        self.assertEqual(self.exchange(**{'Content-Type': 'text/plain'})[0], 415)
        self.assertEqual(self.request('/access/session?token=not-allowed', 'POST', '{}', headers)[0], 404)

    def test_actual_landing_script_hash_navigation_cleans_before_post_and_serializes(self):
        # Execute the shipped JS, with only browser primitives simulated.
        harness = r"""
const vm = require('node:vm'), assert = require('node:assert/strict');
let hash='', events={}, pending=[], calls=[], redirects=[], message={textContent:''};
const ctx={location:{get hash(){return hash},replace:x=>redirects.push(x)},
 history:{replaceState:(_,__,path)=>{assert.equal(path,'/access');hash=''}},
 document:{getElementById:()=>message},addEventListener:(event,fn)=>events[event]=fn,
 fetch:(url,options)=>{assert.equal(hash,'');assert.equal(url,'/access/session');
 assert.equal(options.method,'POST');assert.equal(options.credentials,'same-origin');
 calls.push(JSON.parse(options.body).token);return new Promise(resolve=>pending.push(resolve));}};
vm.runInNewContext(require('node:fs').readFileSync(0,'utf8'),ctx);
(async()=>{assert.equal(calls.length,0);assert.match(message.textContent,/Apri il link/);
 hash='#'+'a'.repeat(64);events.hashchange();assert.equal(calls.length,1);assert.equal(hash,'');
 hash='#'+'b'.repeat(64);events.hashchange();assert.equal(calls.length,1);
 pending.shift()({ok:false});await new Promise(resolve=>setImmediate(resolve));
 assert.deepEqual(calls,['a'.repeat(64),'b'.repeat(64)]);assert.equal(hash,'');
 pending.shift()({ok:true});await new Promise(resolve=>setImmediate(resolve));
 assert.deepEqual(redirects,['/datasets']);})();
"""
        subprocess.run(['node', '-e', harness], input=access.SCRIPT, text=True, check=True, capture_output=True)

    def test_public_landing_has_no_private_metadata_and_delivery_is_exclusive(self):
        status, headers, body = self.request('/access')
        self.assertEqual(status, 200); self.assertIn(b'history.replaceState', body)
        self.assertIn("connect-src 'self'", headers['Content-Security-Policy'])
        self.assertNotIn(self.protocol.link.encode(), body)
        self.assertNotIn(self.protocol.secret, body)
        self.assertEqual(self.request('/access?token=not-allowed')[0], 404)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory).resolve() / 'private-link.txt'
            access.write_link(self.protocol, 'https://studio.example.test', file)
            self.assertEqual(stat.S_IMODE(file.stat().st_mode), 0o600)
            self.assertEqual(file.read_text(), 'https://studio.example.test/access#' + self.protocol.link + '\n')
            with self.assertRaises(FileExistsError): access.write_link(self.protocol, 'https://studio.example.test', file)
            for origin in ['http://studio.example.test', 'https://user:password@studio.example.test', 'https://studio.example.test?token=x', 'https://studio.example.test/path', 'https://studio.example.test#token', 'https://studio.example.test:99999', 'https://studio.example.test\n']:
                with self.assertRaises(ValueError): access.write_link(self.protocol, origin, Path(directory) / 'wrong')
            link = Path(directory) / 'symlink'; link.symlink_to(file)
            with self.assertRaises(ValueError): access.write_link(self.protocol, 'https://studio.example.test', link)

if __name__ == '__main__': unittest.main()
