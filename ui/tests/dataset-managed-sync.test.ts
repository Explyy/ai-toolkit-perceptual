import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import sharp from 'sharp';
import { StudioStore, hash } from '../src/datasetStudio/store';
import { stableJSON } from '../src/datasetStudio/domain';
import { Hub } from '../src/datasetStudio/hf';
import {
  recordLocal,
  readOutbox,
  drainManaged,
  managedProjection,
  managedRoot,
  validateProjection,
  syncStatus,
} from '../src/datasetStudio/managedSync';
import { folders, materialize } from '../src/datasetStudio/catalog';
import { defaultCaptionModel } from '../src/datasetStudio/captionModels';
function server() {
  let serial = 1,
    head = serial.toString(16).padStart(40, '0'),
    commits = 0,
    unknown = false,
    unknownBlob = false,
    reject = false;
  const versions = new Map<string, Map<string, Buffer>>([[head, new Map()]]),
    uploads: string[] = [],
    payloads: number[] = [];
  const oid = (b: Buffer) =>
    createHash('sha1')
      .update('blob ' + b.length + '\0')
      .update(b)
      .digest('hex');
  const transport = (async (url: any, init: any = {}) => {
    const u = new URL(String(url)),
      p = decodeURIComponent(u.pathname);
    if (p === '/api/whoami-v2') return Response.json({ type: 'user', name: 'daverave' });
    if (p === '/api/datasets') {
      if (!u.searchParams.has('cursor'))
        return Response.json(
          [
            { id: 'daverave/Personal', private: true },
            { id: 'other/foreign', private: true },
          ],
          {
            headers: {
              link: '<https://huggingface.co/api/datasets?author=daverave&limit=100&full=true&cursor=2>; rel="next"',
            },
          },
        );
      return Response.json([
        { id: 'daverave/Another', private: true },
        { id: 'daverave/Public', private: false },
      ]);
    }
    if (p.includes('/tree/')) {
      const tail = p.split('/tree/')[1],
        rev = tail.slice(0, 40),
        folder = tail.slice(41),
        prefix = folder ? folder + '/' : '';
      return Response.json(
        [...versions.get(rev)!.entries()]
          .filter(
            ([key]) =>
              key.startsWith(prefix) &&
              (u.searchParams.get('recursive') === 'true' || !key.slice(prefix.length).includes('/')),
          )
          .map(([key, b]) => ({ type: 'file', path: key, size: b.length, oid: oid(b) })),
      );
    }
    if (p.endsWith('/preupload/main')) {
      const x = JSON.parse(init.body);
      uploads.push(...x.files.map((y: any) => y.path));
      return Response.json({ files: x.files.map((y: any) => ({ ...y, uploadMode: 'regular', shouldIgnore: false })) });
    }
    if (p.endsWith('/commit/main')) {
      commits++;
      payloads.push(Buffer.byteLength(String(init.body)));
      const ops = String(init.body)
        .trim()
        .split('\n')
        .map(x => JSON.parse(x));
      if (reject) {
        reject = false;
        bump();
        return new Response('', { status: 409 });
      }
      if (ops[0].value.parentCommit !== head) return new Response('', { status: 409 });
      const next = new Map(versions.get(head));
      for (const x of ops.slice(1)) next.set(x.value.path, Buffer.from(x.value.content, 'base64'));
      head = (++serial).toString(16).padStart(40, '0');
      versions.set(head, next);
      if ((unknown && ops.slice(1).some((x: any) => x.value.path.endsWith('/current.json'))) || unknownBlob) {
        unknown = false;
        unknownBlob = false;
        throw new Error('connection lost after durable commit');
      }
      return Response.json({ commitOid: head });
    }
    if (p.includes('/paths-info/')) {
      const rev = p.split('/paths-info/')[1],
        key = JSON.parse(init.body).paths[0],
        b = versions.get(rev)?.get(key);
      return Response.json(b ? [{ path: key, size: b.length, oid: oid(b) }] : []);
    }
    if (p.includes('/resolve/')) {
      const tail = p.split('/resolve/')[1],
        b = versions.get(tail.slice(0, 40))?.get(tail.slice(41));
      return b ? new Response(new Uint8Array(b)) : new Response('', { status: 404 });
    }
    if (p.startsWith('/api/datasets/')) return Response.json({ private: true, sha: head });
    throw new Error('Unexpected fixture endpoint ' + p);
  }) as typeof fetch;
  function bump() {
    head = (++serial).toString(16).padStart(40, '0');
    versions.set(head, new Map(versions.get([...versions.keys()].at(-1)!)!));
  }
  return {
    hub: new Hub('fixture-token', transport),
    transport,
    uploads,
    payloads,
    get commits() {
      return commits;
    },
    get head() {
      return head;
    },
    get files() {
      return versions.get(head)!;
    },
    loseResponse() {
      unknown = true;
    },
    loseBlobResponse() {
      unknownBlob = true;
    },
    rejectParent() {
      reject = true;
    },
    bump,
  };
}
async function fixture() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'studio-managed-')),
    datasets = path.join(root, 'datasets'),
    data = path.join(root, 'data');
  await fs.mkdir(path.join(datasets, 'native'), { recursive: true });
  for (let i = 0; i < 3; i++) {
    const b = await sharp({
      create: { width: 128, height: 128, channels: 3, background: ['#ff0000', '#00ff00', '#0000ff'][i] },
    })
      .png()
      .toBuffer();
    await fs.writeFile(path.join(datasets, 'native', i + '.png'), b);
    await fs.writeFile(path.join(datasets, 'native', i + '.txt'), 'original ' + i);
  }
  const st = await new StudioStore(data, datasets, 'native').init();
  await st.read();
  return { st, root, data, datasets, cleanup: () => fs.rm(root, { recursive: true, force: true }) };
}
test('private owner catalog paginates, skips public/foreign repos and rejects hostile pagination', async () => {
  const h = server();
  assert.deepEqual(await h.hub.repositories(), [{ repo: 'daverave/Personal' }, { repo: 'daverave/Another' }]);
  await assert.rejects(h.hub.owned('other/foreign'), /token-owner/);
  const malicious = new Hub('fixture', (async () =>
    Response.json([], { headers: { link: '<https://evil.example/api/datasets>; rel="next"' } })) as typeof fetch);
  await assert.rejects(malicious.pages('/api/datasets'), /Untrusted/);
});
test('original pixels, saved captions/drafts/preferences/manual review restore; caption edit uploads no unchanged pixels', async () => {
  const f = await fixture(),
    h = server();
  try {
    let s = await f.st.read();
    s = await f.st.edit(s.revision, [s.images[0].id], { caption: 'manual caption', excluded: 1, category: 'face' });
    s = await f.st.captionSettings(s.revision, {
      key: defaultCaptionModel.key,
      instructions: 'Describe shape and color',
    });
    s = await f.st.captionDraft(
      s.revision,
      s.images[1].id,
      { caption: 'unfinished saved draft', baseRevision: s.images[1].revision },
      0,
    );
    await drainManaged(f.st, h.hub, Date.now() + 10000);
    assert.equal((await readOutbox(f.st))!.phase, 'synced');
    const pixelUploads = h.uploads.filter(x => x.includes('/blobs/')).length;
    assert.equal(pixelUploads, 3);
    s = await f.st.edit((await f.st.read()).revision, [s.images[0].id], { caption: 'second caption' });
    await drainManaged(f.st, h.hub, Date.now() + 10000);
    assert.equal(h.uploads.filter(x => x.includes('/blobs/')).length, pixelUploads);
    const projection = managedProjection(await f.st.read()),
      serialized = stableJSON(projection);
    assert.doesNotMatch(serialized, /embedding|keypoints|depth|job_config|datasetRoot|AITK|HF_TOKEN/);
    const entries = await folders(h.hub, 'daverave/Personal');
    assert.equal(entries.length, 1);
    const imported = await materialize(h.hub, f.data, f.datasets, entries[0]),
      restored = await new StudioStore(f.data, f.datasets, imported.name).init();
    const reopened = await restored.read();
    assert.deepEqual(
      reopened.images.map(x => x.sha),
      s.images.map(x => x.sha),
    );
    assert.equal(reopened.images[0].caption, 'second caption');
    assert.equal(reopened.images[0].excluded, 1);
    assert.equal(reopened.images[1].captionDraft?.caption, 'unfinished saved draft');
    assert.deepEqual(reopened.captionPreferences, s.captionPreferences);
    assert.equal(
      managedProjection(reopened).dataset,
      'native',
      'collision-safe folder does not change logical managed title',
    );
    assert.equal((await readOutbox(restored))!.desired, (await readOutbox(restored))!.baseDigest);
    const duplicate = await materialize(h.hub, f.data, f.datasets, entries[0]);
    assert.equal(duplicate.name, imported.name);
  } finally {
    await f.cleanup();
  }
});
test('newer local edit during capture/upload is never dropped or reported synced; restart drains pending state', async () => {
  const f = await fixture(),
    h = server();
  try {
    const read = f.st.read.bind(f.st);
    let once = true;
    f.st.read = async () => {
      const captured = await read();
      if (once) {
        once = false;
        await f.st.edit(captured.revision, [captured.images[0].id], { caption: 'newer edit' });
      }
      return captured;
    };
    await drainManaged(f.st, h.hub, Date.now() + 10000);
    let o = (await readOutbox(f.st))!;
    assert.equal(o.phase, 'pending');
    assert.notEqual(o.desired, o.baseDigest);
    const restarted = await new StudioStore(f.data, f.datasets, 'native').init();
    await drainManaged(restarted, h.hub, Date.now() + 10000);
    o = (await readOutbox(restarted))!;
    assert.equal(o.phase, 'synced');
    assert.equal(o.desired, hash(stableJSON(managedProjection(await restarted.read()))));
  } finally {
    await f.cleanup();
  }
});
test('unknown commit reconciles exact readback without a second POST; unrelated parent rejects rebase; real divergence stays conflict', async () => {
  const f = await fixture(),
    h = server();
  try {
    h.loseResponse();
    await drainManaged(f.st, h.hub, Date.now() + 10000);
    assert.equal((await readOutbox(f.st))!.phase, 'commit_started');
    assert.equal(h.commits, 4);
    const restarted = await new StudioStore(f.data, f.datasets, 'native').init();
    await drainManaged(restarted, h.hub, Date.now() + 10000);
    assert.equal(h.commits, 4);
    assert.equal((await readOutbox(f.st))!.phase, 'synced');
    let s = await f.st.read();
    await f.st.edit(s.revision, [s.images[0].id], { caption: 'after unrelated campaign' });
    h.rejectParent();
    await drainManaged(f.st, h.hub, Date.now() + 10000);
    assert.equal((await readOutbox(f.st))!.phase, 'synced');
    assert.equal(h.commits, 6);
    s = await f.st.read();
    await f.st.edit(s.revision, [s.images[0].id], { caption: 'local divergence' });
    h.files.set(
      managedRoot(hash('native')) + '/current.json',
      Buffer.from(stableJSON({ schema: 1, key: hash('native'), digest: 'f'.repeat(64) })),
    );
    await drainManaged(f.st, h.hub, Date.now() + 10000);
    assert.equal((await readOutbox(f.st))!.phase, 'conflict');
    assert.equal(h.commits, 6);
    assert.equal((await f.st.read()).images[0].caption, 'local divergence');
  } finally {
    await f.cleanup();
  }
});
test('raw root image without TXT imports with empty caption, preserves SHA and reports missing caption', async () => {
  const f = await fixture(),
    h = server();
  try {
    const bytes = await fs.readFile(path.join(f.st.datasetRoot, '0.png'));
    h.files.set('original.png', bytes);
    const entries = await folders(h.hub, 'daverave/Personal');
    assert.equal(entries[0].folder, '');
    const imported = await materialize(h.hub, f.data, f.datasets, entries[0]);
    assert.equal(imported.missingCaptions, 1);
    const st = await new StudioStore(f.data, f.datasets, imported.name).init(),
      s = await st.read();
    assert.equal(s.images[0].sha, hash(bytes));
    assert.equal(s.images[0].caption, '');
    await assert.rejects(materialize(h.hub, f.data, f.datasets, { ...entries[0], folder: '../foreign' }), /path/i);
    const bad = managedProjection(await f.st.read());
    bad.images[0].relative = '../foreign.png';
    assert.throws(() => validateProjection(bad), /path/i);
  } finally {
    await f.cleanup();
  }
});
test('debounce and concurrent service claims prevent duplicate commits; an edit during network await stays pending', async () => {
  const f = await fixture(),
    h = server();
  try {
    await drainManaged(f.st, h.hub);
    assert.equal(h.commits, 0, 'debounced edits do not upload immediately');
    let entered!: () => void;
    const started = new Promise<void>(resolve => (entered = resolve));
    let once = true;
    const transport = (async (url: any, init: any) => {
      if (String(url).endsWith('/preupload/main') && once) {
        once = false;
        entered();
        const s = await f.st.read();
        await f.st.edit(s.revision, [s.images[0].id], { caption: 'edited during upload' });
        await new Promise(resolve => setTimeout(resolve, 180));
      }
      return h.transport(url, init);
    }) as typeof fetch;
    const first = drainManaged(f.st, new Hub('fixture', transport), Date.now() + 10000);
    await started;
    await assert.rejects(drainManaged(f.st, h.hub, Date.now() + 10000), /operation active/);
    await first;
    assert.equal(h.commits, 4);
    assert.equal((await readOutbox(f.st))!.phase, 'pending');
    await drainManaged(f.st, h.hub, Date.now() + 10000);
    assert.equal((await readOutbox(f.st))!.phase, 'synced');
    assert.equal((await f.st.read()).images[0].caption, 'edited during upload');
  } finally {
    await f.cleanup();
  }
});

test('lost per-original commit reconciles SHA after restart before any subsequent POST; outbox contains no pixel operations', async () => {
  const f = await fixture(),
    h = server();
  try {
    h.loseBlobResponse();
    await drainManaged(f.st, h.hub, Date.now() + 10000);
    let o = (await readOutbox(f.st))!;
    assert.equal(o.active?.blob?.phase, 'commit_started');
    assert.equal(h.commits, 1);
    const raw = await fs.readFile(path.join(f.st.folder, 'managed-outbox.json'), 'utf8');
    assert.ok(Buffer.byteLength(raw) < 20000);
    assert.doesNotMatch(raw, /"operations"|"encoding"|"content"/);
    const restarted = await new StudioStore(f.data, f.datasets, 'native').init();
    await drainManaged(restarted, h.hub, Date.now() + 10000);
    o = (await readOutbox(f.st))!;
    assert.equal(o.phase, 'synced');
    assert.equal(h.commits, 4, 'three unique originals plus one final pointer, no repeated first blob');
    assert.equal(h.uploads.filter(x => x.includes('/blobs/')).length, 3);
  } finally {
    await f.cleanup();
  }
});
test('separate Node processes use the same durable HF claim rather than duplicate blob/pointer commits', async () => {
  const f = await fixture(),
    h = server();
  const { createServer } = await import('node:http');
  const { spawn } = await import('node:child_process');
  let release!: () => void, entered!: () => void;
  const gate = new Promise<void>(resolve => (release = resolve)),
    started = new Promise<void>(resolve => (entered = resolve));
  let first = true;
  const httpServer = createServer(async (req, res) => {
    try {
      const chunks: Buffer[] = [];
      for await (const x of req) chunks.push(Buffer.from(x));
      if (req.url?.endsWith('/preupload/main') && first) {
        first = false;
        entered();
        await gate;
      }
      const response = await h.transport('https://huggingface.co' + req.url, {
        method: req.method,
        body: chunks.length ? Buffer.concat(chunks).toString() : undefined,
      });
      res.writeHead(response.status, Object.fromEntries(response.headers.entries()));
      res.end(Buffer.from(await response.arrayBuffer()));
    } catch {
      res.writeHead(500);
      res.end();
    }
  });
  await new Promise<void>(resolve => httpServer.listen(0, '127.0.0.1', resolve));
  const address = httpServer.address() as any;
  const script = path.join(f.root, 'drain.cjs');
  await fs.writeFile(
    script,
    `
const {StudioStore}=require(${JSON.stringify(path.resolve('src/datasetStudio/store.ts'))});
const {Hub}=require(${JSON.stringify(path.resolve('src/datasetStudio/hf.ts'))});
const {drainManaged}=require(${JSON.stringify(path.resolve('src/datasetStudio/managedSync.ts'))});
(async()=>{const st=await new StudioStore(${JSON.stringify(f.data)},${JSON.stringify(f.datasets)},'native').init();
await drainManaged(st,new Hub('fixture',(url,init)=>fetch('http://127.0.0.1:${address.port}'+new URL(url).pathname+new URL(url).search,init)),Date.now()+10000);
})().catch(e=>{process.stderr.write(e.message);process.exitCode=1});`,
  );
  const child = () => {
    const p = spawn(process.execPath, ['--import', path.resolve('node_modules/tsx/dist/loader.mjs'), script], {
      env: { ...process.env, TSX_TSCONFIG_PATH: path.resolve('tsconfig.json') },
    });
    let output = '';
    p.stderr.on('data', x => (output += x));
    const done = new Promise<{ code: number | null; output: string }>(resolve =>
      p.on('close', code => resolve({ code, output })),
    );
    return { p, done };
  };
  let a: ReturnType<typeof child> | undefined, b: ReturnType<typeof child> | undefined;
  try {
    a = child();
    await started;
    b = child();
    const denied = await b.done;
    assert.equal(denied.code, 1);
    assert.match(denied.output, /operation active/);
    release();
    assert.equal((await a.done).code, 0);
    assert.equal(h.commits, 4);
    assert.equal((await readOutbox(f.st))!.phase, 'synced');
  } finally {
    release();
    a?.p.kill();
    b?.p.kill();
    await new Promise<void>(resolve => httpServer.close(() => resolve()));
    await f.cleanup();
  }
});

test('raw UTF-8/spaced image names and folders preserve exact original bytes without weakening path confinement', async () => {
  const f = await fixture(),
    h = server();
  try {
    const bytes = await fs.readFile(path.join(f.st.datasetRoot, '0.png'));
    h.files.set('foto personali/ritratto è 1.png', bytes);
    const entry = (await folders(h.hub, 'daverave/Personal'))[0];
    assert.equal(entry.folder, 'foto personali');
    const imported = await materialize(h.hub, f.data, f.datasets, entry),
      st = await new StudioStore(f.data, f.datasets, imported.name).init();
    const s = await st.read();
    assert.equal(s.images[0].filename, 'ritratto è 1.png');
    assert.equal(s.images[0].sha, hash(bytes));
  } finally {
    await f.cleanup();
  }
});

test('unknown original not yet visible blocks later POSTs across restart; local bytes never enter outbox', async () => {
  const f = await fixture(),
    h = server();
  try {
    let lost = false;
    const transport = (async (url: any, init: any) => {
      if (String(url).endsWith('/commit/main')) {
        lost = true;
        throw new Error('unknown network outcome');
      }
      return h.transport(url, init);
    }) as typeof fetch;
    await drainManaged(f.st, new Hub('fixture', transport), Date.now() + 10000);
    assert.equal(lost, true);
    assert.equal((await readOutbox(f.st))!.active?.blob?.phase, 'commit_started');
    const restarted = await new StudioStore(f.data, f.datasets, 'native').init();
    await drainManaged(restarted, h.hub, Date.now() + 10000);
    assert.equal(h.commits, 0, 'unconfirmed first blob prevents every subsequent commit');
    assert.equal((await readOutbox(restarted))!.active?.blob?.phase, 'commit_started');
    assert.match((await readOutbox(restarted))!.reason!, /attesa/);
    const raw = await fs.readFile(path.join(f.st.folder, 'managed-outbox.json'), 'utf8');
    assert.doesNotMatch(raw, /"operations"|"encoding"|"content"/);
  } finally {
    await f.cleanup();
  }
});

test('large originals remain separate bounded commits and private legacy model paths stay local', async () => {
  const f = await fixture(),
    h = server();
  try {
    const file = path.join(f.st.datasetRoot, '0.png');
    await fs.appendFile(file, Buffer.alloc(2 * 1024 * 1024));
    let s = await f.st.read();
    s = await f.st.mutate(s.revision, current => {
      current.settings.model = '/workspace/private/model.safetensors';
    });
    await drainManaged(f.st, h.hub, Date.now() + 10000);
    assert.equal((await readOutbox(f.st))!.phase, 'synced');
    assert.equal(h.commits, 4);
    assert.ok(Math.max(...h.payloads) < 34 * 1024 * 1024);
    assert.ok(h.payloads[0] > 2 * 1024 * 1024, 'fixture really exercised a large pixel payload');
    const raw = await fs.readFile(path.join(f.st.folder, 'managed-outbox.json'), 'utf8');
    assert.ok(Buffer.byteLength(raw) < 20000);
    assert.doesNotMatch(raw, /private\/model|"operations"|"content"/);
    assert.equal(managedProjection(await f.st.read()).settings.model, '');
    assert.equal((await f.st.read()).settings.model, '/workspace/private/model.safetensors');
  } finally {
    await f.cleanup();
  }
});
