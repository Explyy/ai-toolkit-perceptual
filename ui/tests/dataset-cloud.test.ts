import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import YAML from 'yaml';

test('cloud startup guards and persistent native DB recovery (real Prisma)', () => {
  execFileSync('python3', ['../docker/dataset-studio/test_startup.py'], { cwd: process.cwd(), stdio: 'pipe' });
});

test('gateway session protocol uses real HTTP and keeps Next private', () => {
  execFileSync('python3', ['../docker/dataset-studio/test_access.py'], { cwd: process.cwd(), stdio: 'pipe' });
  const gateway = fs.readFileSync('../docker/dataset-studio/nginx.conf', 'utf8');
  assert.match(gateway, /auth_request \/_studio_auth/);
  assert.match(gateway, /location = \/_studio_auth \{\s*internal;/);
  assert.match(gateway, /proxy_pass http:\/\/127\.0\.0\.1:8676/);
  assert.match(gateway, /proxy_set_header Authorization ""/);
  assert.match(gateway, /proxy_set_header Cookie ""/);
  assert.match(gateway, /if \(\$readonly_settings\) \{ return 403; \}/);
});

test('workflow login and image publication require the exact reviewed release ref', () => {
  const workflow = YAML.parse(fs.readFileSync('../.github/workflows/dataset-studio.yml', 'utf8'));
  const steps = workflow.jobs.release.steps;
  const login = steps.find((step: any) => step.uses?.startsWith('docker/login-action@'));
  const build = steps.find((step: any) => step.uses?.startsWith('docker/build-push-action@'));
  const expressions = [login.if, build.with.push];
  for (const expression of expressions) {
    // Evaluate the workflow's actual restricted boolean expression against an
    // independent event/ref matrix, rather than duplicating its predicate.
    const condition = expression.replace(/^\$\{\{\s*|\s*\}\}$/g, '');
    const allowed = new Function('github', 'inputs', `return Boolean(${condition})`);
    for (const [ref, event, publish, expected] of [
      ['refs/heads/codex/dataset-studio-cloud-live', 'push', false, true],
      ['refs/heads/codex/dataset-studio-cloud-live', 'workflow_dispatch', true, true],
      ['refs/heads/codex/dataset-studio-cloud-live', 'workflow_dispatch', false, false],
      ['refs/heads/main', 'workflow_dispatch', true, false],
      ['refs/heads/other', 'push', true, false],
      ['refs/tags/codex/dataset-studio-cloud-live', 'workflow_dispatch', true, false],
      ['refs/heads/codex/dataset-studio-cloud-live', 'pull_request', true, false],
    ]) assert.equal(allowed({ ref, event_name: event }, { publish }), expected, `${event} ${ref} publish=${publish}`);
  }
});

test('native SQLite addon actually opens, queries and closes using the running Node binary', () => {
  const output = execFileSync(process.execPath, ['../docker/dataset-studio/check_sqlite.cjs', process.cwd()],
    { cwd: process.cwd(), encoding: 'utf8' });
  const receipt = JSON.parse(output);
  assert.equal(receipt.node, process.versions.node);
  assert.equal(receipt.platform, process.platform);
  assert.equal(receipt.architecture, process.arch);
  assert.equal(receipt.openQueryClose, 'passed');
});
