import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import sharp from 'sharp';
import { StudioStore, hash, contained, validateTraining, captionPath } from '../src/datasetStudio/store';
import { select, bucket, stableJSON, analyzePixels, settings, Problem } from '../src/datasetStudio/domain';
import { createJob, reconcileJob, applyCaption, captionConfig } from '../src/datasetStudio/jobs';
import { sync, remoteFiles } from '../src/datasetStudio/sync';
import { sameOrigin, body } from '../src/datasetStudio/http';
const training = () => ({
  job: 'extension',
  config: {
    name: 'reviewed draft',
    process: [
      {
        type: 'diffusion_trainer',
        model: { arch: 'krea2', name_or_path: 'krea/Krea-2-Raw' },
        network: { type: 'lokr', linear: 32 },
        train: { steps: 4250, lr: 0.0003 },
        datasets: [{ caption_dropout_rate: 0.05 }],
      },
    ],
  },
});
async function fixture() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'native-studio-test-'));
  const datasets = path.join(root, 'datasets');
  await fs.mkdir(path.join(datasets, 'Subject A'), { recursive: true });
  for (const [i, size] of [
    [1200, 1200],
    [1024, 1536],
    [1536, 1024],
  ].entries()) {
    const dir = path.join(datasets, 'Subject A'),
      name = String(i + 1);
    await fs.writeFile(
      path.join(dir, name + '.png'),
      await sharp({
        create: { width: size[0], height: size[1], channels: 3, background: ['#895936', '#b19c75', '#d46e6f'][i] },
      })
        .png()
        .toBuffer(),
    );
    await fs.writeFile(path.join(dir, name + '.txt'), 'Original caption ' + name);
  }
  const st = await new StudioStore(path.join(root, 'data'), datasets, 'Subject A').init();
  return { root, st, cleanup: () => fs.rm(root, { recursive: true, force: true }) };
}
async function categorized(st: StudioStore) {
  let s = await st.read();
  for (const [i, category] of ['face', 'body', 'variety'].entries())
    s = await st.edit(s.revision, [s.images[i].id], { category });
  return s;
}
async function exported(st: StudioStore) {
  let s = await categorized(st);
  s = await st.prepareExport(s.revision);
  const id = s.snapshots[0].id;
  for (let i = 0; i < 3; i++) s = await st.exportFile(s.revision, id, i);
  s = await st.finishExport(s.revision, id);
  return s;
}
function dbFixture() {
  const rows = new Map<string, any>();
  let writes = 0;
  return {
    rows,
    get writes() {
      return writes;
    },
    job: {
      async findUnique({ where }: any) {
        return [...rows.values()].find(x => (where.name ? x.name === where.name : x.id === where.id)) ?? null;
      },
      async create({ data }: any) {
        writes++;
        const row = { ...data, id: 'job-' + writes, step: 0, total_steps: data.total_steps };
        rows.set(row.name, row);
        return row;
      },
    },
  };
}
test('CAS bulk exclusion/restore and manual caption overrides survive reload without changing originals', async () => {
  const f = await fixture();
  try {
    let s = await f.st.read();
    const originals = await Promise.all(s.images.map(x => f.st.source(x)));
    const captions = await Promise.all(
      s.images.map(x => fs.readFile(path.join(f.st.datasetRoot, x.filename.replace('.png', '.txt')))),
    );
    s = await f.st.edit(
      s.revision,
      s.images.slice(0, 2).map(x => x.id),
      { excluded: 1 },
    );
    await assert.rejects(f.st.edit(s.revision - 1, [s.images[0].id], { excluded: 0 }), /Stale/);
    s = await f.st.edit(s.revision, [s.images[0].id], {
      caption: 'Manual wins',
      tags: ['portrait'],
      category: 'face',
      baseRevision: s.images[0].revision,
    });
    assert.equal((await f.st.read()).images[0].caption, 'Manual wins');
    await assert.rejects(
      f.st.edit(s.revision, [s.images[0].id], { caption: 'Stale draft', baseRevision: 0 }),
      /draft retained/,
    );
    s = await f.st.edit(s.revision, [s.images[0].id], { excluded: 0, discarded: 0 });
    assert.equal(s.images[1].excluded, 1);
    for (const [i, x] of s.images.entries()) {
      assert.deepEqual(await f.st.source(x), originals[i]);
      assert.deepEqual(await fs.readFile(path.join(f.st.datasetRoot, x.filename.replace('.png', '.txt'))), captions[i]);
    }
  } finally {
    await f.cleanup();
  }
});
test('dataset and file realpath containment reject sibling-prefix, traversal and symlink escapes', async () => {
  const f = await fixture();
  try {
    assert.throws(
      () => new StudioStore(path.join(f.root, 'data'), path.join(f.root, 'datasets'), '../escape'),
      /Invalid/,
    );
    const outside = path.join(f.root, 'outside');
    await fs.writeFile(outside, 'private');
    await fs.symlink(outside, path.join(f.st.datasetRoot, 'escape.png'));
    await assert.rejects(f.st.read(), /Symlink/);
    await assert.rejects(contained(f.st.datasetRoot, f.st.datasetRoot + '-other/file', true), /outside/);
  } finally {
    await f.cleanup();
  }
});
test('logical title is CAS saved without renaming canonical storage or altering protected curation', async () => {
  const f = await fixture();
  try {
    let s = await exported(f.st);
    s = await f.st.approve(s.revision, {
      approve: true, name: 'Original approval', subject: 'Subject A', trigger: 'token',
      settings: s.settings, captioner: 'Qwen3VLCaptioner', captionModel: 'Qwen/Qwen3-VL-2B-Instruct',
      training: training(),
    });
    s = await f.st.edit(s.revision, [s.images[0].id], { excluded: 1 });
    s = await f.st.captionDraft(s.revision, s.images[1].id,
      { caption: 'Protected unsaved caption', baseRevision: s.images[1].revision }, 0);
    const before = structuredClone(s), root = f.st.datasetRoot, folder = f.st.folder;
    const files = await Promise.all(s.images.map(x => f.st.source(x)));
    s = await f.st.title(s.revision, 'eleonora ghostwell');
    assert.equal(s.displayTitle, 'eleonora ghostwell');
    assert.equal(s.dataset, 'Subject A');
    assert.equal(s.revision, before.revision + 1);
    const { displayTitle, revision, ...rest } = s;
    const { revision: oldRevision, ...original } = before;
    assert.deepEqual(rest, original);
    const reopened = await new StudioStore(path.join(f.root, 'data'), path.join(f.root, 'datasets'), 'Subject A').init();
    assert.equal(reopened.datasetRoot, root);
    assert.equal(reopened.folder, folder);
    assert.equal((await reopened.read()).displayTitle, 'eleonora ghostwell');
    await reopened.verifySnapshot(s.snapshots[0]);
    for (const [i, image] of s.images.entries()) assert.deepEqual(await reopened.source(image), files[i]);
    await assert.rejects(reopened.title(oldRevision, 'Stale tab title'), /Stale/);
    for (const invalid of ['', '  ', ' padded ', 'bad\nline', 'bad\tline', 'bad\x7f', 'bad\u0080', 'bad\u0085', 'bad\u009f', '.'.repeat(129), 123])
      await assert.rejects(reopened.title(s.revision, invalid), /title|text/i);
    assert.deepEqual(await reopened.raw(), s);
  } finally { await f.cleanup(); }
});
test('templates require approval and native caption/model+architecture pairs; later approvals preserve prior values', async () => {
  const f = await fixture();
  try {
    let s = await f.st.read();
    const input = {
      approve: true,
      name: 'Subject A',
      subject: 'subject',
      trigger: 'token',
      previous: null,
      settings: s.settings,
      captioner: 'Qwen3VLCaptioner',
      captionModel: 'Qwen/Qwen3-VL-2B-Instruct',
      training: training(),
    };
    await assert.rejects(f.st.approve(s.revision, { ...input, approve: false }), /Approve/);
    assert.throws(
      () =>
        validateTraining({
          ...training(),
          config: {
            process: [{ ...training().config.process[0], model: { arch: 'invented', name_or_path: 'anything' } }],
          },
        }),
      /catalog/,
    );
    assert.throws(() => captionConfig('Qwen3VLCaptioner', 'invented', 'instructions', 'scope', 'draft'), /catalog/);
    s = await f.st.approve(s.revision, input);
    const original = stableJSON(s.templates[0]);
    s = await f.st.approve(s.revision, {
      ...input,
      previous: s.templates[0].id,
      name: 'Subject A v2',
      training: {
        ...training(),
        config: {
          ...training().config,
          process: [{ ...training().config.process[0], train: { steps: 5000, lr: 0.001 } }],
        },
      },
    });
    assert.equal(stableJSON(s.templates[0]), original);
    assert.notEqual(s.templates[0].id, s.templates[1].id);
  } finally {
    await f.cleanup();
  }
});
test('native bucket boundaries and deterministic quota/pin conflicts are explicit', async () => {
  assert.deepEqual(bucket(33, 37, 512), { width: 32, height: 40 });
  assert.deepEqual(bucket(1536, 1024, 512), { width: 624, height: 416 });
  const f = await fixture();
  try {
    let s = await categorized(f.st);
    assert.equal(select(s.images, 3).complete, true);
    s = await f.st.edit(s.revision, [s.images[0].id], { excluded: 1 });
    assert.equal(select(s.images, 3).complete, false);
    const exported = await f.st.prepareExport(s.revision);
    assert.deepEqual(
      exported.snapshots[0].source.map(x => x.id),
      s.images
        .filter(x => !x.excluded)
        .sort((a, b) => a.sha.localeCompare(b.sha))
        .map(x => x.id),
    );
  } finally {
    await f.cleanup();
  }
});
test('real JPEG/TXT export reads back immutable SHA+size, raw manifest digest and separate optional test', async () => {
  const f = await fixture();
  try {
    let s = await exported(f.st),
      v = s.snapshots[0];
    assert.equal(v.files.length, 6);
    for (const file of v.files) {
      const b = await fs.readFile(await f.st.file(v.id, file.path));
      assert.equal(hash(b), file.sha);
      assert.equal(b.length, file.size);
      if (file.path.endsWith('.jpg')) {
        const m = await sharp(b).metadata();
        assert.equal(m.width, file.width);
        assert.equal(m.height, file.height);
      }
    }
    const raw = await fs.readFile(await f.st.file(v.id, 'manifest.json'));
    assert.equal(hash(raw), v.digest);
    const old = raw;
    await assert.rejects(f.st.exportFile(s.revision, v.id, 0), /not building/);
    s = await f.st.mutate(s.revision, s => {
      s.settings = { ...s.settings, count: 2, testId: s.images[2].id, testPrompt: 'Optional comparison' };
    });
    s = await f.st.prepareExport(s.revision);
    const next = s.snapshots.at(-1)!;
    for (let i = 0; i < 2; i++) s = await f.st.exportFile(s.revision, next.id, i);
    s = await f.st.finishExport(s.revision, next.id);
    assert(s.snapshots.at(-1)!.files.some(f => f.path === 'test/prompt.txt'));
    assert.deepEqual(await fs.readFile(await f.st.file(v.id, 'manifest.json')), old);
  } finally {
    await f.cleanup();
  }
});
test('manual export ignores proposal count/categories/pins and preserves committed captions, drafts and membership', async () => {
  const f = await fixture();
  try {
    let s = await f.st.read();
    const first = s.images[0].id;
    s = await f.st.edit(s.revision, [first], {
      caption: 'Committed manual caption',
      baseRevision: s.images[0].revision,
    });
    s = await f.st.captionDraft(
      s.revision,
      first,
      { caption: 'Protected unsaved typing', baseRevision: s.images[0].revision },
      0,
    );
    s = await f.st.edit(s.revision, [s.images[2].id], { excluded: 1, pinned: 1 });
    s = await f.st.mutate(s.revision, state => {
      state.settings.count = 30;
    });
    assert.equal(select(s.images, 30).complete, false);
    const curation = stableJSON(s.images),
      target = s.settings.count;
    const before = await Promise.all(s.images.map(x => f.st.source(x)));
    s = await f.st.prepareExport(s.revision);
    const snapshot = s.snapshots.at(-1)!;
    assert.equal(snapshot.source.length, 2);
    assert.deepEqual(
      snapshot.source.map(x => x.id),
      s.images
        .filter(x => !x.excluded)
        .sort((a, b) => a.sha.localeCompare(b.sha))
        .map(x => x.id),
    );
    assert.equal(snapshot.manifest.settings.count, target);
    assert.equal(snapshot.source.find(x => x.id === first)!.caption, 'Committed manual caption');
    for (let i = 0; i < snapshot.source.length; i++) s = await f.st.exportFile(s.revision, snapshot.id, i);
    s = await f.st.finishExport(s.revision, snapshot.id);
    const reopened = await f.st.read();
    assert.equal(stableJSON(reopened.images), curation);
    assert.equal(reopened.settings.count, 30);
    const captionFile =
      'training/' + String(snapshot.source.findIndex(x => x.id === first) + 1).padStart(6, '0') + '.txt';
    assert.equal(await fs.readFile(await f.st.file(snapshot.id, captionFile), 'utf8'), 'Committed manual caption');
    for (const [i, image] of reopened.images.entries()) assert.deepEqual(await f.st.source(image), before[i]);
  } finally {
    await f.cleanup();
  }
});
test('manual export excludes only an explicit test reference and refuses empty or missing sources without snapshots', async () => {
  const f = await fixture();
  try {
    let s = await f.st.read();
    s = await f.st.mutate(s.revision, state => {
      state.settings.testId = state.images[1].id;
    });
    s = await f.st.prepareExport(s.revision);
    assert.equal(s.snapshots[0].source.length, 2);
    assert(!s.snapshots[0].source.some(x => x.id === s.settings.testId));
    const existing = stableJSON(s.snapshots);
    s = await f.st.edit(
      s.revision,
      s.images.filter(x => x.id !== s.settings.testId).map(x => x.id),
      { excluded: 1 },
    );
    await assert.rejects(f.st.prepareExport(s.revision), /almeno un/);
    assert.equal(stableJSON((await f.st.raw()).snapshots), existing);
    for (const image of s.images) await fs.unlink(path.join(f.st.datasetRoot, image.relative));
    await assert.rejects(f.st.prepareExport(s.revision), /Stale/);
    s = await f.st.read();
    assert.equal(s.images.length, 0);
    await assert.rejects(f.st.prepareExport(s.revision), /almeno un/);
    assert.equal(stableJSON((await f.st.raw()).snapshots), existing);
  } finally {
    await f.cleanup();
  }
});
test('pixel quality includes original resolution and stores actual source provenance', async () => {
  const f = await fixture();
  try {
    let s = await f.st.read();
    s = await f.st.analyze(s.revision, s.images[0].id);
    assert(s.images[0].analysis?.version);
    assert.equal(s.images[0].analysis?.resolution, 1);
    assert.equal((await f.st.read()).images[0].analysis?.quality, s.images[0].analysis?.quality);
  } finally {
    await f.cleanup();
  }
});
test('native job drafts bind exact bucket version/approval, stay stopped and reconcile a lost response without duplicate', async () => {
  const f = await fixture();
  try {
    let s = await exported(f.st),
      db = dbFixture();
    s = await f.st.approve(s.revision, {
      approve: true,
      name: 'A',
      subject: 'A',
      trigger: 'A',
      settings: s.settings,
      captioner: 'Qwen3VLCaptioner',
      captionModel: 'Qwen/Qwen3-VL-2B-Instruct',
      training: training(),
    });
    const input = { kind: 'train', version: s.snapshots[0].id, template: s.templates[0].id, gpu: '0' };
    s = await createJob(f.st, s.revision, input, db, path.join(f.root, 'output'), path.join(f.root, 'sqlite.db'));
    assert.equal(db.writes, 1);
    const job = [...db.rows.values()][0];
    assert.equal(job.status, 'stopped');
    assert.equal(JSON.parse(job.job_config).config.process[0].train.steps, 4250);
    assert.equal(JSON.parse(job.job_config).config.process[0].datasets[0].caption_dropout_rate, 0.05);
    assert(JSON.parse(job.job_config).config.process[0].datasets[0].folder_path.includes(input.version));
    s = await createJob(f.st, s.revision, input, db, path.join(f.root, 'output'), path.join(f.root, 'sqlite.db'));
    assert.equal(db.writes, 1);
    const link = s.jobs[0];
    const wrong = { ...job, job_ref: 'borrowed' };
    db.rows.set(job.name, wrong);
    await assert.rejects(reconcileJob(db, link, '0', f.st.datasetRoot), /identity/);
  } finally {
    await f.cleanup();
  }
});
test('caption scope has no baseline TXT; incomplete/stopped native output cannot apply, manual edits survive full output', async () => {
  const f = await fixture();
  try {
    let s = await f.st.read(),
      db = dbFixture();
    s = await f.st.edit(s.revision, [s.images[0].id], { excluded: 1 });
    const input = {
      kind: 'caption',
      scope: 'included',
      ids: s.images.slice(1).map(x => x.id),
      captioner: 'Qwen3VLCaptioner',
      model: 'Qwen/Qwen3-VL-2B-Instruct',
      instructions: 'Visible only',
      gpu: '0',
    };
    s = await createJob(f.st, s.revision, input, db, 'output', 'sqlite');
    const link = s.jobs[0],
      row = [...db.rows.values()][0];
    assert.equal((await fs.readdir(link.folder!)).filter(x => x.endsWith('.txt')).length, 0);
    await assert.rejects(applyCaption(f.st, s.revision, link.name, true, db), /complete/);
    row.status = 'completed';
    row.step = 2;
    row.total_steps = 2;
    await fs.writeFile(path.join(link.folder!, '000001.txt'), 'Generated1');
    await assert.rejects(applyCaption(f.st, s.revision, link.name, true, db));
    await fs.writeFile(path.join(link.folder!, '000002.txt'), 'Generated2');
    s = await f.st.edit(s.revision, [s.images[1].id], { caption: 'Concurrent manual' });
    await assert.rejects(applyCaption(f.st, s.revision, link.name, true, db), /manual caption/);
    s = await f.st.edit(s.revision, [s.images[1].id], { caption: link.scope![0].caption });
    s = await applyCaption(f.st, s.revision, link.name, true, db);
    assert.equal((await f.st.read()).images[1].caption_source, 'native caption job ' + row.id);
  } finally {
    await f.cleanup();
  }
});
test('same-origin and actual streamed body limits reject unsafe mutations', async () => {
  assert.throws(
    () => sameOrigin(new Request('http://localhost/api', { method: 'POST', headers: { Origin: 'http://evil.test' } })),
    /Same-origin/,
  );
  await assert.rejects(
    body(
      new Request('http://localhost/api', {
        method: 'POST',
        headers: { Origin: 'http://localhost' },
        body: 'x'.repeat(1500001),
      }),
    ),
    /too large/,
  );
});
test('portable JSON omits optional undefined fields and preserves golden numeric manifests', () => {
  const raw = stableJSON({ small: 9.574142e-7, optional: undefined });
  assert.deepEqual(JSON.parse(raw), { small: 9.574142e-7 });
  assert.equal(hash(raw), hash(Buffer.from(raw)));
});

test('Next normalized internal URL uses exact actual HTTP Host; forwarded host cannot authorize foreign Origin', () => {
  sameOrigin(
    new Request('http://localhost:5175/api', {
      method: 'POST',
      headers: { Host: '127.0.0.1:5175', Origin: 'http://127.0.0.1:5175', 'x-forwarded-proto': 'http' },
    }),
  );
  assert.throws(
    () =>
      sameOrigin(
        new Request('http://localhost/api', {
          method: 'POST',
          headers: { Host: 'localhost', Origin: 'http://evil.test', 'x-forwarded-host': 'evil.test' },
        }),
      ),
    /Same-origin/,
  );
});
test('parallel media reads use atomic state without contending for scan lock', async () => {
  const f = await fixture();
  try {
    const s = await f.st.read();
    await f.st.locked(async () => {
      const states = await Promise.all([f.st.raw(), f.st.raw(), f.st.raw()]);
      assert(states.every(x => x.revision === s.revision));
      const bytes = await Promise.all(states.map((x, i) => f.st.source(x.images[i])));
      assert(bytes.every(x => x.length > 0));
    });
  } finally {
    await f.cleanup();
  }
});
function fakeHub(mode: 'ok' | 'unknown' | 'conflict' = 'ok') {
  const remote = new Map<string, Buffer>();
  let head = 'a'.repeat(40),
    commits = 0;
  const transport = (async (url: any, init: any = {}) => {
    const u = new URL(String(url));
    if (u.pathname.endsWith('/preupload/main')) {
      const x = JSON.parse(init.body);
      return Response.json({
        files: x.files.map((f: any) => ({ path: f.path, uploadMode: 'regular', shouldIgnore: false })),
      });
    }
    if (u.pathname.endsWith('/commit/main')) {
      commits++;
      if (mode === 'conflict') return new Response('', { status: 409 });
      const raw = await new Response(init.body).text();
      for (const line of raw.trim().split('\n')) {
        const op = JSON.parse(line);
        if (op.key === 'file') remote.set(op.value.path, Buffer.from(op.value.content, 'base64'));
      }
      head = 'b'.repeat(40);
      return mode === 'unknown' ? new Response('', { status: 502 }) : Response.json({ commitOid: head });
    }
    if (u.pathname.includes('/paths-info/')) {
      const key = JSON.parse(init.body).paths[0],
        bytes = remote.get(key);
      if (!bytes) return Response.json([]);
      return Response.json([{ path: key, size: bytes.length, lfs: { oid: hash(bytes), size: bytes.length } }]);
    }
    if (u.pathname.includes('/resolve/')) {
      const key = u.pathname.split('/resolve/')[1].split('/').slice(1).join('/');
      return remote.has(key) ? new Response(new Uint8Array(remote.get(key)!)) : new Response('', { status: 404 });
    }
    return Response.json({ private: true, sha: head });
  }) as typeof fetch;
  return {
    transport,
    remote,
    get commits() {
      return commits;
    },
  };
}
async function staged(st: StudioStore, h: ReturnType<typeof fakeHub>) {
  let s = await exported(st),
    id = s.snapshots[0].id;
  s = await sync(st, s.revision, { action: 'hfStart', id, repo: 'owner/private' }, 'fixture-token', h.transport);
  for (let index = 0; index < 7; index++)
    s = await sync(st, s.revision, { action: 'hfUpload', id, index }, 'fixture-token', h.transport);
  return { s, id };
}
test('HF immutable raw manifest and each ordered pair require actual SHA/size readback at pinned commit', async () => {
  const f = await fixture(),
    h = fakeHub();
  try {
    let { s, id } = await staged(f.st, h);
    s = await sync(f.st, s.revision, { action: 'hfCommit', id }, 'fixture-token', h.transport);
    assert.equal(h.commits, 1);
    assert.equal(s.snapshots[0].hf.phase, 'verifying');
    for (let index = 0; index < 7; index++)
      s = await sync(f.st, s.revision, { action: 'hfVerify', id, index }, 'fixture-token', h.transport);
    assert.equal(s.snapshots[0].hf.phase, 'verified');
    assert.equal(s.snapshots[0].hf.revision, 'b'.repeat(40));
    assert(s.snapshots[0].hf.folder.startsWith('datasets/Training_Studio_'));
    assert.equal(h.remote.size, 7);
  } finally {
    await f.cleanup();
  }
});
test('HF parent conflict and unknown commit persist; retry never sends replacement commit; corrupted readback fails', async () => {
  for (const mode of ['unknown', 'conflict'] as const) {
    const f = await fixture(),
      h = fakeHub(mode);
    try {
      let { s, id } = await staged(f.st, h);
      await assert.rejects(sync(f.st, s.revision, { action: 'hfCommit', id }, 'fixture-token', h.transport));
      s = await f.st.read();
      assert.equal(s.snapshots[0].hf.phase, mode);
      await assert.rejects(sync(f.st, s.revision, { action: 'hfCommit', id }, 'fixture-token', h.transport));
      assert.equal(h.commits, 1);
      if (mode === 'unknown') {
        s = await sync(f.st, s.revision, { action: 'hfReconcile', id }, 'fixture-token', h.transport);
        const file = remoteFiles(s.snapshots[0], s.snapshots[0].hf)[0];
        h.remote.set(file.remote, Buffer.from('corrupted'));
        await assert.rejects(
          sync(f.st, s.revision, { action: 'hfVerify', id, index: 0 }, 'fixture-token', h.transport),
        );
        assert.equal((await f.st.read()).snapshots[0].hf.phase, 'verifying');
      }
    } finally {
      await f.cleanup();
    }
  }
});
test('unknown native create reconciles actual saved row without second create', async () => {
  const db = dbFixture(),
    original = db.job.create;
  db.job.create = async (input: any) => {
    await original(input);
    throw new Error('response lost');
  };
  const link: any = { name: 'studio-test', kind: 'train', config: training() };
  const row = await reconcileJob(db, link, '0', 'fixture-dataset');
  assert.equal(row.name, link.name);
  assert.equal(db.writes, 1);
});

test('approval records immutable categories/tags/membership and rejects invalid local caption catalog pairs in storage', async () => {
  const f = await fixture();
  try {
    let s = await categorized(f.st);
    s = await f.st.edit(s.revision, [s.images[0].id], { tags: ['face'], pinned: 1 });
    const input = {
      approve: true,
      name: 'A',
      subject: 'A',
      trigger: 'A',
      settings: s.settings,
      captioner: 'Qwen3VLCaptioner',
      captionModel: 'Qwen/Qwen3-VL-2B-Instruct',
      training: training(),
    };
    await assert.rejects(f.st.approve(s.revision, { ...input, captionModel: 'invented' }), /catalog/);
    s = await f.st.approve(s.revision, input);
    const curation = stableJSON(s.templates[0].curation);
    s = await f.st.edit(s.revision, [s.images[0].id], { category: 'body', tags: [], excluded: 1, pinned: 0 });
    assert.equal(stableJSON((await f.st.read()).templates[0].curation), curation);
  } finally {
    await f.cleanup();
  }
});

test('unapproved template draft persists separately across store reload without fabricating approval', async () => {
  const f = await fixture();
  try {
    let s = await f.st.read();
    const draft = {
      name: 'Mady draft',
      subject: 'Mady',
      trigger: 'mady',
      training: '{incomplete json draft',
      captioner: 'Qwen3VLCaptioner',
      captionModel: 'Qwen/Qwen3-VL-2B-Instruct',
      instructions: 'Visible only',
      previous: null,
    };
    await assert.rejects(f.st.draft(s.revision, { ...draft, captioner: 'invented' }), /catalog/);
    s = await f.st.draft(s.revision, draft);
    const reloaded = await new StudioStore(
      path.join(f.root, 'data'),
      path.join(f.root, 'datasets'),
      'Subject A',
    ).init();
    assert.deepEqual((await reloaded.read()).templateDraft, draft);
    assert.equal(s.templates.length, 0);
  } finally {
    await f.cleanup();
  }
});

test('corrupted completed snapshot is refused before any native job creation', async () => {
  const f = await fixture();
  try {
    let s = await exported(f.st);
    s = await f.st.approve(s.revision, {
      approve: true,
      name: 'A',
      subject: 'A',
      trigger: 'A',
      settings: s.settings,
      captioner: 'Qwen3VLCaptioner',
      captionModel: 'Qwen/Qwen3-VL-2B-Instruct',
      training: training(),
    });
    await fs.writeFile(await f.st.file(s.snapshots[0].id, 'training/000001.txt'), 'tampered');
    const db = dbFixture();
    await assert.rejects(
      createJob(
        f.st,
        s.revision,
        { kind: 'train', version: s.snapshots[0].id, template: s.templates[0].id },
        db,
        'out',
        'sqlite',
      ),
      /Snapshot file changed/,
    );
    assert.equal(db.writes, 0);
  } finally {
    await f.cleanup();
  }
});

test('caption extension cannot overwrite any original image/media or traverse paths', () => {
  assert.equal(captionPath('/dataset/image.jpg', 'txt'), '/dataset/image.txt');
  for (const ext of ['jpg', 'png', 'mp4', 'wav', '../escape', 'a/b'])
    assert.throws(() => captionPath('/dataset/image.jpg', ext));
});
test('JPEG EXIF orientation is recorded and exported to matching displayed bucket dimensions', async () => {
  const f = await fixture();
  try {
    for (const name of await fs.readdir(f.st.datasetRoot)) await fs.unlink(path.join(f.st.datasetRoot, name));
    const bytes = await sharp({ create: { width: 40, height: 80, channels: 3, background: '#785643' } })
      .jpeg()
      .withMetadata({ orientation: 6 })
      .toBuffer();
    await fs.writeFile(path.join(f.st.datasetRoot, 'phone.jpg'), bytes);
    let s = await f.st.read();
    assert.equal(s.images[0].width, 80);
    assert.equal(s.images[0].height, 40);
    s = await f.st.edit(s.revision, [s.images[0].id], { category: 'face' });
    s = await f.st.mutate(s.revision, s => {
      s.settings.count = 1;
    });
    s = await f.st.prepareExport(s.revision);
    s = await f.st.exportFile(s.revision, s.snapshots[0].id, 0);
    s = await f.st.finishExport(s.revision, s.snapshots[0].id);
    const meta = await sharp(await f.st.file(s.snapshots[0].id, 'training/000001.jpg')).metadata();
    assert.equal(meta.width, 80);
    assert.equal(meta.height, 40);
    assert.deepEqual(await f.st.source(s.images[0]), bytes);
  } finally {
    await f.cleanup();
  }
});
test('native bearer protection applies to caption mutations while established media GET stays public', async () => {
  const { middleware } = await import('../src/middleware');
  const { NextRequest } = await import('next/server');
  const before = process.env.AI_TOOLKIT_AUTH;
  process.env.AI_TOOLKIT_AUTH = 'synthetic-test-bearer';
  try {
    assert.equal(middleware(new NextRequest('http://localhost/api/img/caption', { method: 'POST' })).status, 401);
    assert.equal(middleware(new NextRequest('http://localhost/api/dataset-studio')).status, 401);
    assert.equal(
      middleware(new NextRequest('http://localhost/api/img/file.jpg')).headers.get('x-middleware-next'),
      '1',
    );
    assert.equal(
      middleware(
        new NextRequest('http://localhost/api/img/caption', {
          method: 'POST',
          headers: { Authorization: 'Bearer synthetic-test-bearer' },
        }),
      ).headers.get('x-middleware-next'),
      '1',
    );
  } finally {
    if (before === undefined) delete process.env.AI_TOOLKIT_AUTH;
    else process.env.AI_TOOLKIT_AUTH = before;
  }
});
test('actual caption route respects Host origin, rejects media extension/escape and saves only contained TXT', async () => {
  const f = await fixture(),
    before = process.env.DATASET_STUDIO_DATASETS_ROOT;
  process.env.DATASET_STUDIO_DATASETS_ROOT = path.join(f.root, 'datasets');
  try {
    const { POST } = await import('../src/app/api/img/caption/route');
    const file = path.join(f.st.datasetRoot, '1.png'),
      original = await fs.readFile(file);
    const request = (input: any) =>
      new Request('http://localhost:5175/api/img/caption', {
        method: 'POST',
        headers: { Host: '127.0.0.1:5175', Origin: 'http://127.0.0.1:5175' },
        body: JSON.stringify(input),
      });
    assert.equal((await POST(request({ imgPath: file, caption: 'Native saved', ext: 'txt' }))).status, 200);
    assert.equal(await fs.readFile(file.replace('.png', '.txt'), 'utf8'), 'Native saved');
    assert.equal((await POST(request({ imgPath: file, caption: 'erase image', ext: 'png' }))).status, 400);
    assert.deepEqual(await fs.readFile(file), original);
    assert.equal((await POST(request({ imgPath: path.join(f.root, 'outside.png'), caption: 'escape' }))).status, 403);
  } finally {
    if (before === undefined) delete process.env.DATASET_STUDIO_DATASETS_ROOT;
    else process.env.DATASET_STUDIO_DATASETS_ROOT = before;
    await f.cleanup();
  }
});
test('duplicate groups choose one candidate unless explicit pins, and pins exceeding quotas block completion', async () => {
  const f = await fixture();
  try {
    const s = await categorized(f.st),
      images = structuredClone(s.images);
    images[1].sha = images[0].sha;
    assert.equal(select(images, 3).complete, false);
    images[0].pinned = 1;
    images[1].pinned = 1;
    assert.equal(select(images, 3).complete, true);
    images[1].category = 'face';
    assert.equal(select(images, 3).complete, false);
    assert(select(images, 3).conflicts.some(x => x.includes('pins exceed')));
    assert.deepEqual(
      select([...s.images].reverse(), 3).selected.map(x => x.id),
      select(s.images, 3).selected.map(x => x.id),
    );
  } finally {
    await f.cleanup();
  }
});
test('actual paired upload writes new originals and rejects collisions before modifying existing files', async () => {
  const f = await fixture(),
    before = process.env.DATASET_STUDIO_DATASETS_ROOT;
  process.env.DATASET_STUDIO_DATASETS_ROOT = path.join(f.root, 'datasets');
  try {
    const { POST } = await import('../src/app/api/datasets/upload/route');
    const original = await fs.readFile(path.join(f.st.datasetRoot, '1.png'));
    const input = (collision = false) => {
      const form = new FormData();
      form.set('datasetName', 'Subject A');
      form.append('files', new File([new Uint8Array(original)], collision ? '1.png' : 'new.png'));
      form.append('files', new File(['New paired caption'], 'new.txt'));
      return new Request('http://localhost/upload', {
        method: 'POST',
        headers: { Host: '127.0.0.1:5175', Origin: 'http://127.0.0.1:5175' },
        body: form,
      });
    };
    assert.equal((await POST(input(true))).status, 409);
    await assert.rejects(fs.stat(path.join(f.st.datasetRoot, 'new.txt')));
    assert.deepEqual(await fs.readFile(path.join(f.st.datasetRoot, '1.png')), original);
    assert.equal((await POST(input())).status, 200);
    assert.equal(await fs.readFile(path.join(f.st.datasetRoot, 'new.txt'), 'utf8'), 'New paired caption');
    assert.deepEqual(await fs.readFile(path.join(f.st.datasetRoot, 'new.png')), original);
  } finally {
    if (before === undefined) delete process.env.DATASET_STUDIO_DATASETS_ROOT;
    else process.env.DATASET_STUDIO_DATASETS_ROOT = before;
    await f.cleanup();
  }
});

test('native training refuses a snapshot whose tier differs from frozen template approval before DB creation', async () => {
  const f = await fixture();
  try {
    let s = await exported(f.st);
    s = await f.st.approve(s.revision, {
      approve: true,
      name: 'A512',
      subject: 'A',
      trigger: 'A',
      settings: { ...s.settings, tier: 512 },
      captioner: 'Qwen3VLCaptioner',
      captionModel: 'Qwen/Qwen3-VL-2B-Instruct',
      training: training(),
    });
    const db = dbFixture();
    await assert.rejects(
      createJob(
        f.st,
        s.revision,
        { kind: 'train', version: s.snapshots[0].id, template: s.templates[0].id },
        db,
        'out',
        'sqlite',
      ),
      /differs from approved/,
    );
    assert.equal(db.writes, 0);
  } finally {
    await f.cleanup();
  }
});

test('actual native GIF/BMP upload preserves valid source bytes and all originals; either filename collision refuses the whole batch', async () => {
  const f = await fixture(),
    before = process.env.DATASET_STUDIO_DATASETS_ROOT;
  process.env.DATASET_STUDIO_DATASETS_ROOT = path.join(f.root, 'datasets');
  try {
    const { POST } = await import('../src/app/api/datasets/upload/route');
    const gif = Buffer.from('R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7', 'base64');
    assert.equal((await sharp(gif).metadata()).width, 1);
    // Complete 24-bit Windows BMP: 14-byte file header, 40-byte info header,
    // two padded 2-pixel rows (8 bytes each), bottom-up RGB pixels.
    const bmp = Buffer.alloc(70);
    bmp.write('BM');
    bmp.writeUInt32LE(70, 2);
    bmp.writeUInt32LE(54, 10);
    bmp.writeUInt32LE(40, 14);
    bmp.writeInt32LE(2, 18);
    bmp.writeInt32LE(2, 22);
    bmp.writeUInt16LE(1, 26);
    bmp.writeUInt16LE(24, 28);
    bmp.writeUInt32LE(16, 34);
    Buffer.from([0, 0, 255, 0, 255, 0, 0, 0, 255, 0, 0, 255, 255, 255, 0, 0]).copy(bmp, 54);
    const media = [
      { name: 'native.gif', type: 'image/gif', bytes: gif },
      { name: 'native.bmp', type: 'image/bmp', bytes: bmp },
    ];
    const originalNames = await fs.readdir(f.st.datasetRoot),
      originalBytes = await Promise.all(originalNames.map(name => fs.readFile(path.join(f.st.datasetRoot, name))));
    const submit = (files: File[]) => {
      const form = new FormData();
      form.set('datasetName', 'Subject A');
      for (const file of files) form.append('files', file);
      return POST(
        new Request('http://localhost/upload', {
          method: 'POST',
          headers: { Host: '127.0.0.1:5175', Origin: 'http://127.0.0.1:5175' },
          body: form,
        }),
      );
    };
    const response = await submit(media.map(x => new File([new Uint8Array(x.bytes)], x.name, { type: x.type })));
    assert.equal(response.status, 200);
    assert.deepEqual(
      (await response.json()).files,
      media.map(x => x.name),
    );
    for (const item of media) {
      assert.deepEqual(await fs.readFile(path.join(f.st.datasetRoot, item.name)), item.bytes);
      const marker = 'should-not-save-' + item.name + '.txt';
      const collision = await submit([
        new File(['No partial write'], marker),
        new File([new Uint8Array(item.bytes)], item.name, { type: item.type }),
      ]);
      assert.equal(collision.status, 409);
      await assert.rejects(fs.stat(path.join(f.st.datasetRoot, marker)));
      assert.deepEqual(await fs.readFile(path.join(f.st.datasetRoot, item.name)), item.bytes);
    }
    for (const [index, name] of originalNames.entries())
      assert.deepEqual(await fs.readFile(path.join(f.st.datasetRoot, name)), originalBytes[index]);
    assert.equal((await f.st.read()).images.length, 3, 'GIF/BMP remain native media; image Studio scope unchanged');
  } finally {
    if (before === undefined) delete process.env.DATASET_STUDIO_DATASETS_ROOT;
    else process.env.DATASET_STUDIO_DATASETS_ROOT = before;
    await f.cleanup();
  }
});

test('membership validates all targets before save: changed bytes, TXT and symlinks refuse the entire bulk', async () => {
  const f = await fixture();
  try {
    const s = await f.st.read(),
      ids = s.images.slice(0, 2).map(x => x.id);
    const stateFile = path.join(f.st.folder, 'state.json'),
      before = await fs.readFile(stateFile);
    const outboxFile = path.join(f.st.folder, 'managed-outbox.json'),
      outbox = await fs.readFile(outboxFile);
    const file = path.join(f.st.datasetRoot, s.images[1].relative),
      original = await fs.readFile(file);
    // Same length, different bytes: size/mtime shortcuts cannot accept this.
    const changed = Buffer.from(original);
    changed[changed.length - 1] ^= 1;
    await fs.writeFile(file, changed);
    await assert.rejects(f.st.edit(s.revision, ids, { excluded: 1 }), /image changed/i);
    assert.deepEqual(await fs.readFile(stateFile), before);
    assert.deepEqual(await fs.readFile(outboxFile), outbox);
    await fs.writeFile(file, original);
    const txt = file.replace('.png', '.txt'),
      caption = await fs.readFile(txt);
    await fs.writeFile(txt, 'Externally changed');
    await assert.rejects(f.st.edit(s.revision, ids, { excluded: 1 }), /caption changed/i);
    assert.deepEqual(await fs.readFile(stateFile), before);
    await fs.writeFile(txt, caption);
    await fs.unlink(file);
    await fs.symlink(path.join(f.st.datasetRoot, s.images[0].relative), file);
    await assert.rejects(f.st.edit(s.revision, ids, { excluded: 1 }), /symlink/i);
    assert.deepEqual(await fs.readFile(stateFile), before);
    await fs.unlink(file);
    await fs.writeFile(file, original);
    await fs.unlink(txt);
    await fs.symlink(path.join(f.st.datasetRoot, s.images[0].relative.replace('.png', '.txt')), txt);
    await assert.rejects(f.st.edit(s.revision, ids, { excluded: 1 }), /symlink/i);
    assert.deepEqual(await fs.readFile(stateFile), before);
  } finally {
    await f.cleanup();
  }
});
test('membership preserves protected captions, legacy tags and managed outbox; manual review/image CAS advance', async () => {
  const f = await fixture();
  try {
    let s = await f.st.read(),
      id = s.images[0].id;
    s = await f.st.edit(s.revision, [id], {
      caption: 'Manual caption',
      tags: ['legacy'],
      baseRevision: s.images[0].revision,
    });
    s = await f.st.captionDraft(
      s.revision,
      id,
      { caption: 'Later protected draft', baseRevision: s.images[0].revision },
      0,
    );
    const image = structuredClone(s.images[0]),
      oldOutbox = JSON.parse(await fs.readFile(path.join(f.st.folder, 'managed-outbox.json'), 'utf8'));
    s = await f.st.edit(s.revision, [id], { excluded: 1 });
    assert.equal(s.images[0].revision, image.revision + 1);
    assert.equal(s.images[0].reviewRevision, (image.reviewRevision ?? 0) + 1);
    assert.equal(s.images[0].caption, 'Manual caption');
    assert.equal(s.images[0].captionDraft?.caption, image.captionDraft!.caption);
    assert.equal(s.images[0].captionDraft?.baseRevision, s.images[0].revision);
    assert.equal(s.images[0].captionDraftRevision, image.captionDraftRevision! + 1);
    await assert.rejects(
      f.st.captionDraft(
        s.revision,
        id,
        { caption: 'Concurrent old tab', baseRevision: image.revision },
        image.captionDraftRevision!,
      ),
      /nessun overwrite/,
    );
    assert.deepEqual(s.images[0].tags, ['legacy']);
    const outbox = JSON.parse(await fs.readFile(path.join(f.st.folder, 'managed-outbox.json'), 'utf8'));
    assert.notEqual(outbox.desired, oldOutbox.desired);
    assert.equal(outbox.phase, 'pending');
    await assert.rejects(
      f.st.edit(s.revision, [id], { caption: 'Stale save', baseRevision: image.revision }),
      /draft retained/,
    );
    // Reopening restores a fresh protected draft that can save without another edit.
    s = await f.st.read();
    s = await f.st.edit(s.revision, [id], {
      caption: s.images[0].captionDraft!.caption,
      baseRevision: s.images[0].captionDraft!.baseRevision,
    });
    const loaded = await f.st.read();
    assert.equal(loaded.images[0].caption, 'Later protected draft');
    assert.equal(loaded.images[0].captionDraft, undefined);
    assert.deepEqual(loaded.images[0].tags, ['legacy']);
    s = await f.st.edit(loaded.revision, [id], { excluded: 0 });
    assert.equal((await f.st.read()).images[0].excluded, 0);
  } finally {
    await f.cleanup();
  }
});
test('targeted membership is not discovery; authoritative read still hashes and discovers changed untargeted originals', async () => {
  const f = await fixture();
  try {
    const s = await f.st.read(),
      unrelated = s.images[2];
    const source = path.join(f.st.datasetRoot, unrelated.relative);
    const bytes = await sharp({ create: { width: 1536, height: 1024, channels: 3, background: '#abcdef' } })
      .png()
      .toBuffer();
    await fs.writeFile(source, bytes);
    const saved = await f.st.edit(s.revision, [s.images[0].id], { excluded: 1 });
    assert.equal(saved.images[2].sha, unrelated.sha);
    const discovered = await f.st.read();
    assert.equal(discovered.images.find(x => x.filename === unrelated.filename)!.sha, hash(bytes));
    assert.notEqual(discovered.images.find(x => x.filename === unrelated.filename)!.id, unrelated.id);
    assert.equal(discovered.images[0].excluded, 1);
  } finally {
    await f.cleanup();
  }
});
test('membership never rebases an already stale caption draft', async () => {
  const f = await fixture();
  try {
    let s = await f.st.read(),
      id = s.images[0].id;
    s = await f.st.captionDraft(s.revision, id, { caption: 'Protected stale text', baseRevision: -1 }, 0);
    const draft = structuredClone(s.images[0].captionDraft),
      token = s.images[0].captionDraftRevision;
    s = await f.st.edit(s.revision, [id], { excluded: 1 });
    assert.deepEqual(s.images[0].captionDraft, draft);
    assert.equal(s.images[0].captionDraftRevision, token);
    await assert.rejects(
      f.st.edit(s.revision, [id], { caption: draft!.caption, baseRevision: draft!.baseRevision }),
      /draft retained/,
    );
  } finally {
    await f.cleanup();
  }
});
