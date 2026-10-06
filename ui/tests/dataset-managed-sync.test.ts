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
import { folders, materialize, nativeCatalog } from '../src/datasetStudio/catalog';
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
    if (p.includes('/commits/')) {
      const rev = p.split('/commits/')[1],
        ids = [...versions.keys()].slice(0, [...versions.keys()].indexOf(rev) + 1).reverse();
      const offset = Number(u.searchParams.get('cursor') ?? 0),
        page = ids.slice(offset, offset + 50);
      return Response.json(
        page.map(id => ({ id })),
        {
          headers:
            offset + 50 < ids.length
              ? { link: '<https://huggingface.co' + u.pathname + '?cursor=' + (offset + 50) + '>; rel="next"' }
              : {},
        },
      );
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
async function fixture(name = 'native') {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'studio-managed-')),
    datasets = path.join(root, 'datasets'),
    data = path.join(root, 'data');
  await fs.mkdir(path.join(datasets, name), { recursive: true });
  for (let i = 0; i < 3; i++) {
    const b = await sharp({
      create: { width: 128, height: 128, channels: 3, background: ['#ff0000', '#00ff00', '#0000ff'][i] },
    })
      .png()
      .toBuffer();
    await fs.writeFile(path.join(datasets, name, i + '.png'), b);
    await fs.writeFile(path.join(datasets, name, i + '.txt'), 'original ' + i);
  }
  const st = await new StudioStore(data, datasets, name).init();
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
test('catalog uses canonical pinned root URLs and preserves nested encoding across guarded pagination', async () => {
  const pinned = 'a'.repeat(40),
    root = 'https://huggingface.co/api/datasets/daverave/Personal/tree/' + pinned;
  for (const sample of [
    { folder: '', recursive: false, url: root + '?recursive=false&limit=1000' },
    { folder: '', recursive: true, url: root + '?recursive=true&limit=1000' },
    {
      folder: 'àrea spaced/深度',
      recursive: true,
      url: root + '/%C3%A0rea%20spaced/%E6%B7%B1%E5%BA%A6?recursive=true&limit=1000',
    },
  ]) {
    const calls: string[] = [];
    const hub = new Hub('fixture-token', (async (url: any, init: any) => {
      calls.push(String(url));
      assert.equal(init.redirect, 'error');
      assert.equal(init.headers.Authorization, 'Bearer fixture-token');
      assert.equal(String(url), sample.url + (calls.length === 2 ? '&cursor=second' : ''));
      return Response.json([{ path: calls.length + '.png' }], {
        headers: calls.length === 1 ? { link: '<' + sample.url + '&cursor=second>; rel="next"' } : {},
      });
    }) as typeof fetch);
    assert.deepEqual(await hub.entries('daverave/Personal', pinned, sample.folder, sample.recursive), [
      { path: '1.png' },
      { path: '2.png' },
    ]);
    assert.equal(calls.length, 2);
  }
});
test('canonical root catalog still refuses foreign, changed-scope and repeated pagination before following it', async () => {
  const pinned = 'a'.repeat(40),
    endpoint = 'https://huggingface.co/api/datasets/daverave/Personal/tree/' + pinned + '?recursive=true&limit=1000';
  for (const sample of [
    { next: endpoint.replace('huggingface.co', 'evil.example'), error: /Untrusted/ },
    { next: endpoint.replace(pinned, 'b'.repeat(40)), error: /Untrusted/ },
    { next: endpoint.replace('recursive=true', 'recursive=false'), error: /scope changed/ },
    { next: endpoint.replace('limit=1000', 'limit=100'), error: /scope changed/ },
    { next: endpoint, error: /Repeated/ },
  ]) {
    let calls = 0;
    const hub = new Hub('fixture-token', (async (url: any, init: any) => {
      calls++;
      assert.equal(String(url), endpoint);
      assert.equal(init.redirect, 'error');
      assert.equal(init.headers.Authorization, 'Bearer fixture-token');
      return Response.json([], { headers: { link: '<' + sample.next + '>; rel="next"' } });
    }) as typeof fetch);
    await assert.rejects(hub.entries('daverave/Personal', pinned, '', true), sample.error);
    assert.equal(calls, 1);
  }
});
test('logical title sync changes metadata only and restores with the original canonical managed key', async () => {
  const f = await fixture('shinra'), h = server();
  try {
    let s = await f.st.read();
    assert.equal(managedProjection(s).dataset, 'shinra', 'legacy state has a canonical title fallback');
    assert.equal((await nativeCatalog(f.datasets, f.data))[0].title, 'shinra');
    s = await f.st.edit(s.revision, [s.images[0].id], { excluded: 1, caption: 'Saved caption' });
    s = await f.st.captionDraft(s.revision, s.images[1].id,
      { caption: 'Protected draft', baseRevision: s.images[1].revision }, 0);
    await drainManaged(f.st, h.hub, Date.now() + 10000);
    const prior = (await readOutbox(f.st))!, key = prior.key,
      images = structuredClone(s.images), uploadCount = h.uploads.length;
    s = await f.st.title(s.revision, 'eleonora ghostwell');
    assert.equal(s.dataset, 'shinra');
    assert.equal(managedProjection(s).key, key);
    assert.equal(managedProjection(s).dataset, 'eleonora ghostwell');
    for (const invalid of ['', ' padded ', 'unsafe\nname', 'unsafe\u0080name', 'unsafe\u0085name', 'unsafe\u009fname'])
      assert.throws(() => validateProjection({ ...managedProjection(s), dataset: invalid }), /title/);
    assert.deepEqual(s.images, images);
    const pending = (await readOutbox(f.st))!;
    assert.equal(pending.baseDigest, prior.baseDigest);
    assert.notEqual(pending.desired, prior.desired);
    await drainManaged(f.st, h.hub, Date.now() + 10000);
    assert.equal(h.uploads.slice(uploadCount).filter(x => x.includes('/blobs/')).length, 0);
    assert.equal((await readOutbox(f.st))!.phase, 'synced');
    const entry = (await folders(h.hub, 'daverave/Personal')).find(x => x.kind === 'managed' && x.key === key)!;
    const imported = await materialize(h.hub, f.data, f.datasets, entry);
    const st = await new StudioStore(f.data, f.datasets, imported.name).init(), restored = await st.read();
    assert.notEqual(restored.dataset, 'shinra', 'restored folder remains collision-safe');
    assert.equal(restored.displayTitle, 'eleonora ghostwell');
    assert.equal(managedProjection(restored).key, key);
    assert.equal(managedProjection(restored).dataset, 'eleonora ghostwell');
    assert.deepEqual(restored.images.map(x => [x.sha, x.caption, x.captionDraft, x.excluded]),
      images.map(x => [x.sha, x.caption, x.captionDraft, x.excluded]));
    const local = await nativeCatalog(f.datasets, f.data);
    assert.equal(local.find(x => x.name === 'shinra')!.title, 'eleonora ghostwell');
    assert.equal(local.find(x => x.name === imported.name)!.title, 'eleonora ghostwell');
    const priorBinding = structuredClone(restored.managedBinding);
    const retitled = await st.title(restored.revision, 'Second logical title');
    assert.deepEqual(retitled.managedBinding, priorBinding);
    assert.equal(managedProjection(retitled).key, key);
    assert.equal(managedProjection(retitled).dataset, 'Second logical title');
  } finally { await f.cleanup(); }
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

test('unapplied unknown original recovers once at the exact parent after restart; local bytes never enter outbox', async () => {
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
    assert.equal(h.commits, 4, 'one CAS recovery plus remaining originals and final pointer');
    const recovered = (await readOutbox(restarted))!;
    assert.equal(recovered.phase, 'synced');
    const proof = Object.values(recovered.blobRecoveries!)[0];
    assert.equal(proof.original.parent, proof.parent);
    await drainManaged(restarted, h.hub, Date.now() + 10000);
    assert.equal(h.commits, 4, 'repeated drain never replays again');
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

test('known429 rejection persists status/retry deadline across restart and sends no commit before Retry-After', async () => {
  const f = await fixture(),
    h = server();
  let rejected = false,
    posts = 0;
  const now = Math.floor(Date.now() / 1000) * 1000 + 10000;
  let clockTime = now;
  const clock = () => clockTime;
  try {
    const transport = (async (url: any, init: any) => {
      if (String(url).endsWith('/commit/main')) {
        posts++;
        if (!rejected) {
          rejected = true;
          clockTime = now + 150000;
          return new Response('', {
            status: 429,
            headers: { 'Retry-After': new Date(clockTime + 90000).toUTCString() },
          });
        }
      }
      return h.transport(url, init);
    }) as typeof fetch;
    const hub = new Hub('fixture', transport);
    await drainManaged(f.st, hub, clock);
    let o = (await readOutbox(f.st))!;
    assert.equal(o.phase, 'error');
    assert.equal(o.lastRemoteFailure?.status, 429);
    assert.equal(o.retryAt, now + 240000);
    assert.equal(o.active?.blob?.phase, 'uploading');
    const restarted = await new StudioStore(f.data, f.datasets, 'native').init();
    assert.equal(o.lastRemoteFailure?.at, now + 150000);
    clockTime = now + 239999;
    await drainManaged(restarted, hub, clock);
    assert.equal(posts, 1);
    clockTime = now + 240000;
    await drainManaged(restarted, hub, clock);
    o = (await readOutbox(restarted))!;
    assert.equal(o.phase, 'synced');
    assert.equal(o.lastRemoteFailure?.status, 429);
    assert.equal(posts, 5);
    assert.equal(Object.keys(o.blobRecoveries ?? {}).length, 0);
  } finally {
    await f.cleanup();
  }
});
test('original and once-only recovery share identical parent/content; concurrent original wins CAS with one application', async () => {
  const f = await fixture(),
    h = server();
  let pending: any,
    attempts = 0;
  try {
    const transport = (async (url: any, init: any) => {
      if (String(url).endsWith('/commit/main')) {
        attempts++;
        if (!pending) {
          pending = { url, init };
          throw Error('original response unavailable');
        }
        if (attempts === 2) {
          assert.equal(init.body, pending.init.body);
          await h.transport(pending.url, pending.init);
        }
      }
      return h.transport(url, init);
    }) as typeof fetch;
    const hub = new Hub('fixture', transport);
    await drainManaged(f.st, hub, Date.now() + 10000);
    const original = (await readOutbox(f.st))!.active!.blob!;
    const restarted = await new StudioStore(f.data, f.datasets, 'native').init();
    await drainManaged(restarted, hub, Date.now() + 10000);
    const o = (await readOutbox(restarted))!;
    assert.equal(o.phase, 'synced');
    assert.deepEqual(o.blobRecoveries![original.sha].original, original);
    assert.equal(h.commits, 5, 'original apply, rejected recovery, two remaining originals and pointer');
    assert.equal(h.files.size, 5);
    await drainManaged(restarted, hub, Date.now() + 10000);
    assert.equal(h.commits, 5);
  } finally {
    await f.cleanup();
  }
});
test('three original unknowns at one parent recover sequentially through bounded complete historical absence proof', async () => {
  const all = await Promise.all(['a', 'b', 'c'].map(fixture)),
    h = server();
  try {
    const lost = new Hub('fixture', (async (url: any, init: any) => {
      if (String(url).endsWith('/commit/main')) throw Error('unapplied original');
      return h.transport(url, init);
    }) as typeof fetch);
    for (const f of all) await drainManaged(f.st, lost, Date.now() + 10000);
    const parent = h.head;
    for (const f of all) {
      await drainManaged(f.st, h.hub, Date.now() + 10000);
      const o = (await readOutbox(f.st))!;
      assert.equal(o.phase, 'synced');
      assert.equal(Object.values(o.blobRecoveries!)[0].original.parent, parent);
    }
    assert.equal(h.commits, 12);
    assert.ok(Object.values((await readOutbox(all[1].st))!.blobRecoveries!)[0].commits.length > 1);
    assert.ok(Object.values((await readOutbox(all[2].st))!.blobRecoveries!)[0].commits.length > 5);
  } finally {
    for (const f of all) await f.cleanup();
  }
});
test('historical blob existence followed by deletion is a conflict; missing parent, bounds and readerrors never prove absence', async () => {
  const f = await fixture(),
    h = server();
  let pending: any;
  try {
    const lost = new Hub('fixture', (async (url: any, init: any) => {
      if (String(url).endsWith('/commit/main')) {
        pending = { url, init };
        throw Error('unknown');
      }
      return h.transport(url, init);
    }) as typeof fetch);
    await drainManaged(f.st, lost, Date.now() + 10000);
    const original = (await readOutbox(f.st))!.active!.blob!;
    await h.transport(pending.url, pending.init);
    h.bump();
    h.files.delete(original.path);
    const posts = h.commits;
    await drainManaged(f.st, h.hub, Date.now() + 10000);
    let o = (await readOutbox(f.st))!;
    assert.equal(o.phase, 'conflict');
    assert.match(o.reason!, /history/);
    assert.deepEqual(o.active!.blob, original);
    assert.equal(h.commits, posts);
    await assert.rejects(h.hub.historyTo('daverave/Personal', h.head, 'f'.repeat(40)), /missing/);
    for (let i = 0; i < 201; i++) h.bump();
    await assert.rejects(h.hub.historyTo('daverave/Personal', h.head, original.parent), /200/);
  } finally {
    await f.cleanup();
  }
  const g = await fixture('read-error'),
    s = server();
  try {
    const lost = new Hub('fixture', (async (url: any, init: any) => {
      if (String(url).endsWith('/commit/main')) throw Error('unknown');
      return s.transport(url, init);
    }) as typeof fetch);
    await drainManaged(g.st, lost, Date.now() + 10000);
    s.bump();
    const blob = (await readOutbox(g.st))!.active!.blob!,
      head = s.head;
    const errors = new Hub('fixture', (async (url: any, init: any) => {
      if (String(url).endsWith('/paths-info/' + blob.parent) && JSON.parse(init.body).paths[0] === blob.path)
        return new Response('', { status: 503 });
      return s.transport(url, init);
    }) as typeof fetch);
    await drainManaged(g.st, errors, Date.now() + 10000);
    const o = (await readOutbox(g.st))!;
    assert.equal(s.commits, 0);
    assert.equal(s.head, head);
    assert.deepEqual(o.active!.blob, blob);
    assert.equal(o.lastRemoteFailure?.status, 503);
    assert.equal(Object.keys(o.blobRecoveries ?? {}).length, 0);
  } finally {
    await g.cleanup();
  }
});
test('a second unknown recovery response remains bounded after restart; no third POST or metadata reset', async () => {
  const f = await fixture(),
    h = server();
  let calls = 0;
  try {
    const lost = new Hub('fixture', (async (url: any, init: any) => {
      if (String(url).endsWith('/commit/main')) {
        calls++;
        throw Error('unknown');
      }
      return h.transport(url, init);
    }) as typeof fetch);
    await drainManaged(f.st, lost, Date.now() + 10000);
    await drainManaged(f.st, lost, Date.now() + 10000);
    assert.equal(calls, 2);
    const old = (await readOutbox(f.st))!;
    const restarted = await new StudioStore(f.data, f.datasets, 'native').init();
    await drainManaged(restarted, lost, Date.now() + 10000);
    assert.equal(calls, 2);
    assert.deepEqual((await readOutbox(restarted))!.blobRecoveries, old.blobRecoveries);
    assert.equal((await readOutbox(restarted))!.active!.blob!.phase, 'commit_started');
  } finally {
    await f.cleanup();
  }
});

test('confirmed429 during unknown recovery retains audit and cooldown, then revalidates proof after restart', async () => {
  const f = await fixture(),
    h = server();
  let calls = 0;
  const now = Date.now() + 10000;
  let clockTime = now;
  const clock = () => clockTime;
  try {
    const transport = (async (url: any, init: any) => {
      if (String(url).endsWith('/commit/main')) {
        calls++;
        if (calls === 1) throw Error('original unavailable');
        if (calls === 2) {
          clockTime = now + 150000;
          return new Response('', { status: 429, headers: { 'Retry-After': '90' } });
        }
      }
      return h.transport(url, init);
    }) as typeof fetch;
    const hub = new Hub('fixture', transport);
    await drainManaged(f.st, hub, clock);
    const original = (await readOutbox(f.st))!.active!.blob!;
    await drainManaged(f.st, hub, clock);
    let o = (await readOutbox(f.st))!;
    assert.equal(o.lastRemoteFailure?.status, 429);
    assert.equal(o.lastRemoteFailure?.at, now + 150000);
    assert.equal(o.retryAt, now + 240000);
    assert.equal(o.blobRecoveries![original.sha].attempts![0].retryAt, now + 240000);
    assert.equal(o.blobRecoveries![original.sha].attempts![0].responseAt, now + 150000);
    assert.equal(o.blobRecoveries![original.sha].retryableReject, true);
    assert.equal(calls, 2);
    const restarted = await new StudioStore(f.data, f.datasets, 'native').init();
    clockTime = now + 239999;
    await drainManaged(restarted, hub, clock);
    assert.equal(calls, 2);
    clockTime = now + 240000;
    await drainManaged(restarted, hub, clock);
    o = (await readOutbox(restarted))!;
    assert.equal(o.phase, 'synced');
    assert.equal(calls, 6);
    const proof = o.blobRecoveries![original.sha];
    assert.deepEqual(proof.original, original);
    assert.equal(proof.attempts!.length, 2);
    assert.equal(proof.attempts![0].status, 429);
    assert.equal(proof.retryableReject, undefined);
  } finally {
    await f.cleanup();
  }
});

// Hold a real SQLite edit lock at a receipt boundary; release before catch.
function contendReceipt(st: StudioStore, when: (o: NonNullable<Awaited<ReturnType<typeof readOutbox>>>) => boolean) {
  const locked = st.locked.bind(st);
  let injected = false;
  st.locked = async action => {
    const o = await readOutbox(st);
    if (!injected && o && when(o)) {
      injected = true;
      let entered!: () => void, release!: () => void;
      const started = new Promise<void>(r => (entered = r)),
        until = new Promise<void>(r => (release = r));
      const holder = locked(async () => {
        entered();
        await until;
      });
      await started;
      try {
        return await locked(action);
      } finally {
        release();
        await holder;
      }
    }
    return locked(action);
  };
  return () => injected;
}
async function legacyBusy(f: Awaited<ReturnType<typeof fixture>>, h: ReturnType<typeof server>) {
  await drainManaged(f.st, h.hub, Date.now() + 10000);
  const s = await f.st.read();
  await f.st.edit(s.revision, [s.images[1].id], { excluded: 1 });
  const projection = managedProjection(await f.st.read()),
    o = (await readOutbox(f.st))!;
  o.active = { digest: hash(stableJSON(projection)), projection, parent: h.head, baseDigest: o.baseDigest };
  o.phase = 'conflict';
  o.reason = 'Dataset operation active; refresh';
  await fs.writeFile(path.join(f.st.folder, 'managed-outbox.json'), stableJSON(o));
  return o;
}

test('real edit-lock contention before and after metadata POST defers without false conflict or duplicate commit', async () => {
  for (const boundary of ['before', 'after']) {
    const f = await fixture(),
      h = server();
    try {
      await drainManaged(f.st, h.hub, Date.now() + 10000);
      const s = await f.st.read();
      await f.st.edit(s.revision, [s.images[0].id], { caption: 'durable curation' });
      const initialPosts = h.commits;
      const injected = contendReceipt(
        f.st,
        o =>
          !!o.active &&
          (boundary === 'before' ? o.phase === 'uploading' : o.phase === 'commit_started' && h.commits > initialPosts),
      );
      await drainManaged(f.st, h.hub, Date.now() + 10000);
      assert.equal(injected(), true);
      const deferred = (await readOutbox(f.st))!;
      assert.equal(deferred.phase, boundary === 'before' ? 'uploading' : 'commit_started');
      assert.ok(deferred.active);
      const restart = await new StudioStore(f.data, f.datasets, 'native').init();
      await drainManaged(restart, h.hub, Date.now() + 20000);
      assert.equal((await readOutbox(restart))!.phase, 'synced');
      assert.equal(h.commits, initialPosts + 1);
      assert.equal((await restart.read()).images[0].caption, 'durable curation');
    } finally {
      await f.cleanup();
    }
  }
});

test('legacy busy recovers one exact-parent operation then latest caption, draft and selection without reverting local state', async () => {
  const f = await fixture(),
    h = server();
  try {
    const old = await legacyBusy(f, h),
      activeBytes = stableJSON(old.active);
    let s = await f.st.read();
    s = await f.st.edit(s.revision, [s.images[0].id], { caption: 'latest caption', excluded: 1 });
    await f.st.captionDraft(
      s.revision,
      s.images[2].id,
      { caption: 'protected latest draft', baseRevision: s.images[2].revision },
      0,
    );
    let durable = await f.st.raw(),
      latestDigest = hash(stableJSON(managedProjection(durable)));
    const posts = h.commits;
    let editedDuringUpload = false;
    const hub = new Hub('fixture', (async (url: any, init: any) => {
      if (!editedDuringUpload && String(url).endsWith('/preupload/main')) {
        editedDuringUpload = true;
        const current = await f.st.read();
        await f.st.edit(current.revision, [current.images[0].id], {
          caption: 'caption typed during upload',
          excluded: 1,
        });
        durable = await f.st.raw();
        latestDigest = hash(stableJSON(managedProjection(durable)));
      }
      return h.transport(url, init);
    }) as typeof fetch);
    const restarted = await new StudioStore(f.data, f.datasets, 'native').init();
    await drainManaged(restarted, hub, Date.now() + 10000);
    const pending = (await readOutbox(restarted))!;
    assert.equal(pending.baseDigest, old.active!.digest);
    assert.equal(pending.desired, latestDigest);
    assert.equal(pending.phase, 'pending');
    assert.equal(editedDuringUpload, true);
    assert.equal(h.commits, posts + 1);
    assert.deepEqual(await restarted.raw(), durable);
    assert.equal(stableJSON(old.active), activeBytes);
    assert.equal(Object.values(pending.metadataRecoveries!)[0].parent, old.active!.parent);
    await drainManaged(restarted, h.hub, Date.now() + 20000);
    assert.equal((await readOutbox(restarted))!.baseDigest, latestDigest);
    assert.equal((await readOutbox(restarted))!.phase, 'synced');
    assert.deepEqual(await restarted.raw(), durable);
    const remote = JSON.parse(h.files.get(managedRoot(old.key) + '/versions/' + latestDigest + '.json')!.toString());
    assert.equal(remote.images[0].caption, 'caption typed during upload');
    assert.equal(remote.images[0].excluded, 1);
    assert.equal(remote.images[2].captionDraft.caption, 'protected latest draft');
  } finally {
    await f.cleanup();
  }
});

test('old and recovery CAS race applies one operation; applied/lost response adopts without further POST', async () => {
  for (const scenario of ['old-wins', 'lost-response', 'already-applied', 'busy-after-recovery']) {
    const f = await fixture(),
      h = server();
    try {
      const old = await legacyBusy(f, h),
        initialHead = h.head;
      let first = true,
        recoveryPosts = 0;
      const hub = new Hub('fixture', (async (url: any, init: any) => {
        if (String(url).endsWith('/commit/main')) {
          recoveryPosts++;
          if (first && scenario === 'old-wins') {
            first = false;
            await h.transport(url, init);
          }
          if (scenario === 'lost-response') h.loseResponse();
        }
        return h.transport(url, init);
      }) as typeof fetch);
      if (scenario === 'already-applied') {
        h.files.set(
          managedRoot(old.key) + '/current.json',
          Buffer.from(stableJSON({ schema: 1, key: old.key, digest: old.active!.digest })),
        );
        h.files.set(
          managedRoot(old.key) + '/versions/' + old.active!.digest + '.json',
          Buffer.from(stableJSON(old.active!.projection)),
        );
      }
      if (scenario === 'busy-after-recovery') contendReceipt(f.st, o => !!o.metadataRecoveries && recoveryPosts === 1);
      await drainManaged(f.st, hub, Date.now() + 10000);
      const restart = await new StudioStore(f.data, f.datasets, 'native').init();
      await drainManaged(restart, hub, Date.now() + 30000);
      assert.equal((await readOutbox(restart))!.phase, 'synced');
      assert.equal(recoveryPosts, scenario === 'already-applied' ? 0 : 1);
      if (scenario !== 'already-applied') assert.equal(BigInt('0x' + h.head), BigInt('0x' + initialHead) + BigInt(1));
      assert.equal((await readOutbox(restart))!.baseDigest, old.active!.digest);
    } finally {
      await f.cleanup();
    }
  }
});

test('unknown recovery never replays after restart; changed HEAD, divergence and invalid identity refuse', async () => {
  for (const scenario of [
    'unknown',
    'missing-parent',
    'diverged',
    'tampered',
    'source-changed',
    'bad-receipt',
    'invalid-path',
    'remote-read-failure',
  ]) {
    const f = await fixture(),
      h = server();
    try {
      const old = await legacyBusy(f, h);
      if (scenario === 'missing-parent') {
        old.active!.parent = 'f'.repeat(40);
        await fs.writeFile(path.join(f.st.folder, 'managed-outbox.json'), stableJSON(old));
        h.bump();
      }
      if (scenario === 'diverged')
        h.files.set(
          managedRoot(old.key) + '/current.json',
          Buffer.from(stableJSON({ schema: 1, key: old.key, digest: 'f'.repeat(64) })),
        );
      if (scenario === 'tampered') {
        old.active!.projection.images[0].caption = 'changed without digest';
        await fs.writeFile(path.join(f.st.folder, 'managed-outbox.json'), stableJSON(old));
      }
      if (scenario === 'invalid-path') {
        old.active!.projection.images[0].relative = '../foreign.png';
        old.active!.digest = hash(stableJSON(old.active!.projection));
        await fs.writeFile(path.join(f.st.folder, 'managed-outbox.json'), stableJSON(old));
      }
      if (scenario === 'bad-receipt') {
        old.metadataRecoveries = {
          [hash(old.active!.parent + ':' + old.active!.digest)]: {
            digest: 'f'.repeat(64),
            parent: old.active!.parent,
            attemptedAt: 0,
          },
        };
        await fs.writeFile(path.join(f.st.folder, 'managed-outbox.json'), stableJSON(old));
      }
      if (scenario === 'source-changed')
        await fs.writeFile(
          path.join(f.st.datasetRoot, '0.png'),
          await fs.readFile(path.join(f.st.datasetRoot, '1.png')),
        );
      let posts = 0;
      const hub = new Hub('fixture', (async (url: any, init: any) => {
        if (scenario === 'remote-read-failure') return new Response('', { status: 503 });
        if (String(url).endsWith('/commit/main')) {
          posts++;
          throw Error('unknown outcome without application');
        }
        return h.transport(url, init);
      }) as typeof fetch);
      await drainManaged(f.st, hub, Date.now() + 10000);
      const after = (await readOutbox(f.st))!;
      if (scenario === 'unknown') {
        assert.equal(after.phase, 'commit_started');
        assert.equal(Object.keys(after.metadataRecoveries!).length, 1);
      } else assert.equal(after.phase, 'conflict');
      await drainManaged(await new StudioStore(f.data, f.datasets, 'native').init(), hub, Date.now() + 20000);
      assert.equal(posts, scenario === 'unknown' ? 1 : 0);
      assert.ok((await readOutbox(f.st))!.active);
    } finally {
      await f.cleanup();
    }
  }
});

test('legacy busy with an uncertain blob resumes verified readback or one bounded recovery without requiring unuploaded originals', async () => {
  for (const applied of [true, false]) {
    const f = await fixture(),
      h = server();
    try {
      let first = true;
      const lost = new Hub('fixture', (async (url: any, init: any) => {
        if (first && String(url).endsWith('/commit/main')) {
          first = false;
          if (applied) await h.transport(url, init);
          throw Error('uncertain original');
        }
        return h.transport(url, init);
      }) as typeof fetch);
      await drainManaged(f.st, lost, Date.now() + 10000);
      const old = (await readOutbox(f.st))!;
      assert.equal(old.active!.blob!.phase, 'commit_started');
      const original = structuredClone(old.active!.blob!);
      old.phase = 'conflict';
      old.reason = 'Dataset operation active; refresh';
      await fs.writeFile(path.join(f.st.folder, 'managed-outbox.json'), stableJSON(old));
      const s = await f.st.read();
      await f.st.edit(s.revision, [s.images[0].id], { caption: 'newer saved caption', excluded: 1 });
      const durable = await f.st.raw();
      h.bump();
      const posts = h.commits;
      await drainManaged(f.st, h.hub, Date.now() + 20000);
      const pending = (await readOutbox(f.st))!;
      assert.equal(pending.phase, 'pending');
      assert.equal(pending.baseDigest, old.active!.digest);
      assert.equal(h.commits - posts, applied ? 3 : 4);
      if (applied) assert.equal(pending.blobRecoveries?.[original.sha], undefined);
      else assert.deepEqual(pending.blobRecoveries![original.sha].original, original);
      assert.deepEqual(await f.st.raw(), durable);
      await drainManaged(f.st, h.hub, Date.now() + 30000);
      assert.equal((await readOutbox(f.st))!.phase, 'synced');
      assert.deepEqual(await f.st.raw(), durable);
    } finally {
      await f.cleanup();
    }
  }
});

test('unknown metadata recovers at a proven descendant once, retaining original parent and latest curation', async () => {
  const f = await fixture(),
    h = server();
  try {
    const old = await legacyBusy(f, h),
      original = structuredClone(old.active!);
    h.bump();
    h.bump();
    const proven = h.head,
      posts = h.commits;
    let typed = false;
    const hub = new Hub('fixture', (async (url: any, init: any) => {
      if (!typed && String(url).endsWith('/preupload/main')) {
        typed = true;
        const s = await f.st.read();
        await f.st.edit(s.revision, [s.images[0].id], { caption: 'latest during historical proof', excluded: 1 });
      }
      return h.transport(url, init);
    }) as typeof fetch);
    await drainManaged(f.st, hub, Date.now() + 10000);
    const pending = (await readOutbox(f.st))!;
    assert.equal(pending.phase, 'pending');
    assert.equal(pending.baseDigest, original.digest);
    assert.equal(h.commits, posts + 1);
    const receipt = Object.values(pending.metadataRecoveries!)[0] as any;
    assert.equal(receipt.parent, proven);
    assert.equal(receipt.originalParent, original.parent);
    assert.deepEqual(receipt.commits, [
      proven,
      (BigInt('0x' + proven) - BigInt(1)).toString(16).padStart(40, '0'),
      original.parent,
    ]);
    assert.match(receipt.proofDigest, /^[0-9a-f]{64}$/);
    const durable = await f.st.raw();
    const beforeLate = h.head;
    await assert.rejects(
      h.hub.commit('daverave/Personal', original.parent, [], original.digest),
      (e: any) => e.remoteStatus === 409,
    );
    assert.equal(h.head, beforeLate);
    await drainManaged(await new StudioStore(f.data, f.datasets, 'native').init(), h.hub, Date.now() + 20000);
    assert.equal((await readOutbox(f.st))!.phase, 'synced');
    assert.deepEqual(await f.st.raw(), durable);
  } finally {
    await f.cleanup();
  }
});

test('historical metadata, pointer or base changes refuse even when current bytes look restored; incomplete proof never posts', async () => {
  for (const scenario of [
    'version-deleted',
    'pointer-restored',
    'base-restored',
    'cap',
    'foreign',
    'non404',
    'head-during-proof',
  ]) {
    const f = await fixture(),
      h = server();
    try {
      const old = await legacyBusy(f, h),
        active = structuredClone(old.active!),
        prefix = managedRoot(old.key);
      h.bump();
      if (scenario === 'version-deleted')
        h.files.set(prefix + '/versions/' + active.digest + '.json', Buffer.from(stableJSON(active.projection)));
      if (scenario === 'pointer-restored')
        h.files.set(
          prefix + '/current.json',
          Buffer.from(stableJSON({ schema: 1, key: old.key, digest: 'f'.repeat(64) })),
        );
      const basePath = prefix + '/versions/' + active.baseDigest + '.json',
        base = Buffer.from(h.files.get(basePath)!);
      if (scenario === 'base-restored') h.files.set(basePath, Buffer.from('corrupted historic base'));
      h.bump();
      h.files.delete(prefix + '/versions/' + active.digest + '.json');
      h.files.set(
        prefix + '/current.json',
        Buffer.from(stableJSON({ schema: 1, key: old.key, digest: active.baseDigest })),
      );
      h.files.set(basePath, base);
      if (scenario === 'cap') for (let i = 0; i < 201; i++) h.bump();
      const durable = await f.st.raw(),
        posts = h.commits;
      let bumped = false;
      const hub = new Hub('fixture', (async (url: any, init: any) => {
        if (scenario === 'foreign' && String(url).includes('/commits/'))
          return Response.json([{ id: h.head }], { headers: { link: '<https://foreign.example/next>; rel="next"' } });
        if (
          scenario === 'non404' &&
          String(url).includes('/paths-info/') &&
          JSON.parse(init.body).paths[0] === prefix + '/versions/' + active.digest + '.json'
        )
          return new Response('', { status: 503 });
        if (scenario === 'head-during-proof' && !bumped && String(url).endsWith('/preupload/main')) {
          bumped = true;
          h.bump();
        }
        return h.transport(url, init);
      }) as typeof fetch);
      await drainManaged(f.st, hub, Date.now() + 10000);
      const after = (await readOutbox(f.st))!;
      assert.equal(h.commits, posts, scenario);
      assert.equal(stableJSON(after.active), stableJSON(active), scenario);
      assert.equal(after.metadataRecoveries, undefined, scenario);
      assert.deepEqual(await f.st.raw(), durable, scenario);
      assert.equal(
        after.phase,
        ['non404', 'head-during-proof'].includes(scenario) ? 'commit_started' : 'conflict',
        scenario,
      );
    } finally {
      await f.cleanup();
    }
  }
});

test('descendant recovery loss or receipt contention consumes one durable attempt across restart and further head advances', async () => {
  for (const scenario of ['unapplied', 'lost-applied', 'busy-after']) {
    const f = await fixture(),
      h = server();
    try {
      const old = await legacyBusy(f, h);
      h.bump();
      const parent = h.head;
      let posts = 0;
      const hub = new Hub('fixture', (async (url: any, init: any) => {
        if (String(url).endsWith('/commit/main')) {
          posts++;
          const pending = (await readOutbox(f.st))!;
          assert.equal(stableJSON(pending.active), stableJSON(old.active));
          assert.equal(Object.values(pending.metadataRecoveries!)[0].parent, parent);
          if (scenario === 'unapplied') throw Error('uncertain response, not applied');
          if (scenario === 'lost-applied') h.loseResponse();
        }
        return h.transport(url, init);
      }) as typeof fetch);
      if (scenario === 'busy-after') contendReceipt(f.st, o => !!o.metadataRecoveries && posts === 1);
      await drainManaged(f.st, hub, Date.now() + 10000);
      h.bump();
      const restarted = await new StudioStore(f.data, f.datasets, 'native').init();
      await drainManaged(restarted, hub, Date.now() + 30000);
      assert.equal(posts, 1);
      const after = (await readOutbox(restarted))!;
      assert.equal(Object.keys(after.metadataRecoveries!).length, 1);
      assert.equal(after.phase, scenario === 'unapplied' ? 'commit_started' : 'synced');
      if (scenario === 'unapplied') assert.equal(stableJSON(after.active), stableJSON(old.active));
    } finally {
      await f.cleanup();
    }
  }
});

test('malformed legacy blob evidence refuses without changing its uncertainty or local curation', async () => {
  for (const malformed of ['path', 'sha', 'size', 'parent', 'phase', 'source']) {
    const f = await fixture(),
      h = server();
    try {
      const lost = new Hub('fixture', (async (url: any, init: any) => {
        if (String(url).endsWith('/commit/main')) throw Error('unknown blob');
        return h.transport(url, init);
      }) as typeof fetch);
      await drainManaged(f.st, lost, Date.now() + 10000);
      const old = (await readOutbox(f.st))!;
      old.phase = 'conflict';
      old.reason = 'Dataset operation active; refresh';
      const blob = old.active!.blob!;
      if (malformed === 'path') blob.path = '../foreign';
      if (malformed === 'sha') blob.sha = 'f'.repeat(64);
      if (malformed === 'size') blob.size++;
      if (malformed === 'parent') blob.parent = 'bad';
      if (malformed === 'phase') (blob as any).phase = 'done';
      if (malformed === 'source')
        await fs.writeFile(
          path.join(f.st.datasetRoot, '0.png'),
          await fs.readFile(path.join(f.st.datasetRoot, '1.png')),
        );
      await fs.writeFile(path.join(f.st.folder, 'managed-outbox.json'), stableJSON(old));
      const posts = h.commits;
      await drainManaged(f.st, h.hub, Date.now() + 10000);
      const after = (await readOutbox(f.st))!;
      assert.equal(after.phase, 'conflict');
      assert.equal(h.commits, posts);
      assert.equal(stableJSON(after.active), stableJSON(old.active));
    } finally {
      await f.cleanup();
    }
  }
});

async function committedReadback429(f: Awaited<ReturnType<typeof fixture>>, h: ReturnType<typeof server>) {
  await drainManaged(f.st, h.hub, Date.now() + 10000);
  const state = await f.st.read();
  await f.st.edit(state.revision, [state.images[0].id], { caption: 'latest committed curation', excluded: 1 });
  const posts = h.commits,
    at = Date.now() + 20000;
  let rejected = false;
  const hub = new Hub('fixture', (async (url: any, init: any) => {
    if (
      !rejected &&
      h.commits === posts + 1 &&
      String(url).includes('/paths-info/') &&
      JSON.parse(init.body).paths[0].endsWith('/current.json')
    ) {
      rejected = true;
      return new Response('', { status: 429, headers: { 'Retry-After': '90' } });
    }
    return h.transport(url, init);
  }) as typeof fetch);
  await drainManaged(f.st, hub, at);
  assert.equal(rejected, true, '429 occurs only after the ordinary metadata commit is durable');
  const outbox = (await readOutbox(f.st))!;
  assert.equal(outbox.active!.revision, h.head);
  assert.equal(outbox.lastRemoteFailure!.status, 429);
  return { posts, at, outbox };
}

test('committed metadata readback429 retains verifying and adopts after cooldown/restart with exactly one commit', async () => {
  const f = await fixture(),
    h = server();
  try {
    const { posts, outbox } = await committedReadback429(f, h);
    assert.equal(outbox.phase, 'verifying');
    const before = await f.st.raw();
    let requests = 0;
    const hub = new Hub('fixture', (async (url: any, init: any) => {
      requests++;
      return h.transport(url, init);
    }) as typeof fetch);
    const restarted = await new StudioStore(f.data, f.datasets, 'native').init();
    await drainManaged(restarted, hub, outbox.retryAt! - 1);
    assert.equal(requests, 0, 'readback cooldown survives restart');
    await drainManaged(restarted, hub, outbox.retryAt! + 1);
    const done = (await readOutbox(restarted))!;
    assert.equal(done.phase, 'synced');
    assert.equal(done.baseDigest, outbox.active!.digest);
    assert.equal(done.active, undefined);
    assert.equal(done.retryAt, undefined);
    assert.equal(h.commits, posts + 1);
    assert.deepEqual(await restarted.raw(), before);
  } finally {
    await f.cleanup();
  }
});

test('persisted committed error and exact429 false conflict adopt read-only at an unrelated new head, preserving newer curation', async () => {
  for (const phase of ['error', 'conflict'] as const) {
    const f = await fixture(),
      h = server();
    try {
      const { posts, outbox } = await committedReadback429(f, h);
      outbox.phase = phase;
      outbox.reason =
        phase === 'conflict' ? 'Remote dataset divergence; both versions preserved' : 'HF request failed (429).';
      await fs.writeFile(path.join(f.st.folder, 'managed-outbox.json'), stableJSON(outbox));
      h.bump(); // Another dataset advances the repo without changing this pointer.
      const reads = new Set<string>();
      let writes = 0,
        edited = false,
        durable = await f.st.raw();
      const hub = new Hub('fixture', (async (url: any, init: any) => {
        if (/\/(preupload|commit)\/main$/.test(String(url))) writes++;
        if (String(url).includes('/resolve/')) reads.add(String(url).split('/resolve/')[1].slice(0, 40));
        if (!edited && String(url).includes('/resolve/') && String(url).endsWith('/current.json')) {
          edited = true;
          let s = await f.st.read();
          s = await f.st.edit(s.revision, [s.images[1].id], {
            caption: 'newer during committed readback',
            excluded: 1,
          });
          await f.st.captionDraft(
            s.revision,
            s.images[2].id,
            { caption: 'protected readback draft', baseRevision: s.images[2].revision },
            0,
          );
          durable = await f.st.raw();
        }
        return h.transport(url, init);
      }) as typeof fetch);
      const restarted = await new StudioStore(f.data, f.datasets, 'native').init();
      await drainManaged(restarted, hub, outbox.retryAt! + 1);
      const adopted = (await readOutbox(restarted))!;
      assert.equal(writes, 0, 'adoption performs no preupload/upload/commit');
      assert.equal(h.commits, posts + 1);
      assert.ok(reads.has(outbox.active!.revision!));
      assert.ok(reads.has(h.head));
      assert.equal(adopted.phase, 'pending');
      assert.equal(adopted.baseDigest, outbox.active!.digest);
      assert.equal(adopted.desired, hash(stableJSON(managedProjection(durable))));
      assert.equal(adopted.remoteRevision, h.head);
      assert.equal(adopted.active, undefined);
      assert.equal(adopted.reason, undefined);
      assert.equal(adopted.retryAt, undefined);
      assert.deepEqual(await restarted.raw(), durable);
      await drainManaged(restarted, h.hub, outbox.retryAt! + 10000);
      assert.equal((await readOutbox(restarted))!.phase, 'synced');
      assert.deepEqual(await restarted.raw(), durable);
    } finally {
      await f.cleanup();
    }
  }
});

test('committed adoption refuses divergent pointer/original, malformed projection/receipt/binding and changing ownership or source', async () => {
  for (const scenario of [
    'current-pointer',
    'current-original',
    'recorded-metadata',
    'projection',
    'revision',
    'blob',
    'binding',
    'changed-receipt',
    'changed-source',
  ]) {
    const f = await fixture(),
      h = server();
    try {
      const { posts, outbox } = await committedReadback429(f, h);
      const active = outbox.active!;
      outbox.phase = 'error';
      h.bump();
      if (scenario === 'current-pointer')
        h.files.set(
          managedRoot(outbox.key) + '/current.json',
          Buffer.from(stableJSON({ schema: 1, key: outbox.key, digest: 'a'.repeat(64) })),
        );
      if (scenario === 'current-original')
        h.files.set(
          managedRoot(outbox.key) + '/blobs/' + active.projection.images[0].sha,
          Buffer.from('foreign changed original'),
        );
      if (scenario === 'projection') active.projection.images[0].caption = 'tampered projection';
      if (scenario === 'revision') active.revision = 'not-a-revision';
      if (scenario === 'blob')
        active.blob = {
          path: managedRoot(outbox.key) + '/blobs/' + active.projection.images[0].sha,
          sha: active.projection.images[0].sha,
          size: active.projection.images[0].size,
          parent: active.parent,
          phase: 'commit_started',
        };
      await fs.writeFile(path.join(f.st.folder, 'managed-outbox.json'), stableJSON(outbox));
      if (scenario === 'binding') {
        const s = await f.st.raw();
        s.managedBinding = {
          schema: 1,
          repo: 'daverave/Another',
          key: outbox.key,
          title: 'native',
          baseDigest: outbox.baseDigest,
        } as any;
        await fs.writeFile(path.join(f.st.folder, 'state.json'), stableJSON(s));
      }
      const retained = stableJSON(active),
        before = await f.st.raw();
      let writes = 0,
        injected = false;
      const hub = new Hub('fixture', (async (url: any, init: any) => {
        if (/\/(preupload|commit)\/main$/.test(String(url))) writes++;
        if (
          scenario === 'recorded-metadata' &&
          String(url).includes('/resolve/' + active.revision + '/') &&
          String(url).endsWith('/versions/' + active.digest + '.json')
        )
          return new Response('invalid immutable metadata');
        if (
          !injected &&
          ['changed-receipt', 'changed-source'].includes(scenario) &&
          String(url).includes('/resolve/' + h.head + '/') &&
          String(url).endsWith('/current.json')
        ) {
          injected = true;
          if (scenario === 'changed-receipt') {
            const x = (await readOutbox(f.st))!;
            x.active!.parent = 'f'.repeat(40);
            await fs.writeFile(path.join(f.st.folder, 'managed-outbox.json'), stableJSON(x));
          } else
            await fs.writeFile(
              path.join(f.datasets, 'native', '0.png'),
              await sharp({ create: { width: 128, height: 128, channels: 3, background: '#ffffff' } })
                .png()
                .toBuffer(),
            );
        }
        return h.transport(url, init);
      }) as typeof fetch);
      try {
        await drainManaged(f.st, hub, outbox.retryAt! + 1);
      } catch (e: any) {
        assert.equal(scenario, 'binding');
        assert.match(e.message, /binding changed/i);
      }
      const refused = (await readOutbox(f.st))!;
      assert.equal(writes, 0, scenario);
      assert.equal(h.commits, posts + 1, scenario);
      assert.equal(refused.baseDigest, outbox.baseDigest, scenario);
      assert.ok(refused.active, scenario);
      if (scenario !== 'changed-receipt') assert.equal(stableJSON(refused.active), retained, scenario);
      else assert.equal(refused.active.parent, 'f'.repeat(40));
      assert.deepEqual(await f.st.raw(), before, scenario);
      if (scenario !== 'binding') assert.equal(refused.phase, 'conflict', scenario);
    } finally {
      await f.cleanup();
    }
  }
});

test('an absent revision or unrelated conflict never qualifies for historical committed adoption', async () => {
  for (const scenario of ['no-revision', 'no429', 'different-reason']) {
    const f = await fixture(),
      h = server();
    try {
      const { posts, outbox } = await committedReadback429(f, h);
      outbox.phase = 'conflict';
      outbox.reason = 'Remote dataset divergence; both versions preserved';
      if (scenario === 'no-revision') delete outbox.active!.revision;
      if (scenario === 'no429') outbox.lastRemoteFailure!.status = 503;
      if (scenario === 'different-reason') outbox.reason = 'Actual remote writer conflict';
      await fs.writeFile(path.join(f.st.folder, 'managed-outbox.json'), stableJSON(outbox));
      let calls = 0;
      await drainManaged(
        f.st,
        new Hub('fixture', (async (url: any, init: any) => {
          calls++;
          return h.transport(url, init);
        }) as typeof fetch),
        outbox.retryAt! + 1,
      );
      assert.equal(calls, 0);
      assert.equal(h.commits, posts + 1);
      assert.equal(stableJSON(await readOutbox(f.st)), stableJSON(outbox));
    } finally {
      await f.cleanup();
    }
  }
});

test('repeated transient committed readback and real final lock contention preserve the receipt without mutation fallthrough', async () => {
  const f = await fixture(),
    h = server();
  try {
    const { posts, outbox } = await committedReadback429(f, h);
    const retained = stableJSON(outbox.active);
    let clock = outbox.retryAt! + 1,
      writes = 0;
    for (const failure of ['429', '503', 'network']) {
      const hub = new Hub('fixture', (async (url: any, init: any) => {
        if (/\/(preupload|commit)\/main$/.test(String(url))) writes++;
        if (String(url).includes('/paths-info/') && JSON.parse(init.body).paths[0].endsWith('/current.json')) {
          if (failure === 'network') throw new Error('fixture connection lost');
          return new Response('', { status: Number(failure), headers: { 'Retry-After': '90' } });
        }
        return h.transport(url, init);
      }) as typeof fetch);
      await drainManaged(f.st, hub, clock);
      const pending = (await readOutbox(f.st))!;
      assert.equal(pending.phase, 'verifying', failure);
      assert.equal(stableJSON(pending.active), retained);
      assert.equal(h.commits, posts + 1);
      assert.equal(writes, 0);
      clock = Math.max(clock + 1, (pending.retryAt ?? clock) + 1);
    }
    let proved = false;
    const injected = contendReceipt(f.st, () => proved);
    const hub = new Hub('fixture', (async (url: any, init: any) => {
      if (String(url).includes('/resolve/') && String(url).endsWith('/current.json')) proved = true;
      return h.transport(url, init);
    }) as typeof fetch);
    await drainManaged(f.st, hub, clock);
    assert.equal(injected(), true, 'real edit-lock contention occurs at adoption, after network proof');
    assert.equal(stableJSON((await readOutbox(f.st))!.active), retained);
    assert.equal((await readOutbox(f.st))!.phase, 'verifying');
    assert.equal(h.commits, posts + 1);
    const restarted = await new StudioStore(f.data, f.datasets, 'native').init();
    await drainManaged(restarted, h.hub, clock + 1);
    assert.equal((await readOutbox(restarted))!.phase, 'synced');
    assert.equal(h.commits, posts + 1);
  } finally {
    await f.cleanup();
  }
});
