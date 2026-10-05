import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import sharp from 'sharp';
import { StudioStore, hash, JobLink } from '../src/datasetStudio/store';
import { select, analyzePixels } from '../src/datasetStudio/domain';
import { inferredCategory, validateSignals, MODEL_ANALYZER, ModelSignals } from '../src/datasetStudio/analysisPolicy';
import {
  analysisConfiguration,
  prepareAnalysis,
  applyAnalysisResults,
  applyProposal,
  reconcilePendingAnalysis,
  analysisProcessState,
} from '../src/datasetStudio/analysisFlow';
import { ownedLock } from '../src/datasetStudio/ownerLock';
const sha = 'a'.repeat(64),
  config = 'c'.repeat(64);
function signals(kind: 'face' | 'body' | 'variety' | 'none' = 'body'): ModelSignals {
  const embedding = Array(512).fill(0);
  embedding[0] = 1;
  const points = Array.from(
    { length: 17 },
    (_, i) => [0.3 + i * 0.01, 0.1 + i * 0.04, 0.9] as [number, number, number],
  );
  return {
    version: MODEL_ANALYZER,
    sha,
    config,
    faces:
      kind === 'none'
        ? []
        : [{ box: kind === 'face' ? [0.2, 0.1, 0.7, 0.7] : [0.3, 0.1, 0.4, 0.2], confidence: 0.95, embedding }],
    persons:
      kind === 'none'
        ? []
        : [
            {
              box: [0.1, 0.05, 0.8, 0.95],
              confidence: 0.9,
              keypoints: kind === 'variety' ? points.map(p => [p[0], p[1], 0.1]) : points,
            },
          ],
    depth: { grid: Array(64).fill(0.5), map: 'depth.png', sha: 'd'.repeat(64), relative: true },
    runtime: { device: 'fixture (not inference)', torch: 'fixture', transformers: 'fixture', onnxruntime: 'fixture' },
  };
}
function image(i: number, kind: 'face' | 'body' | 'variety') {
  const pixels = analyzePixels(new Uint8ClampedArray(64 * 64 * 4).fill(120));
  return {
    id: String(i),
    project: 'test',
    sha: i.toString(16).padStart(64, '0'),
    filename: i + '.png',
    mime: 'image/png',
    size: 1,
    width: 512,
    height: 512,
    caption: 'original',
    caption_source: 'fixture',
    category: kind,
    pinned: 0,
    excluded: 0,
    discarded: 0,
    analysis: { ...pixels, quality: 70, model: signals(kind) },
    pose: null,
    revision: 0,
    created: 0,
  };
}
async function fixture() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'studio-analysis-'));
  const datasets = path.join(root, 'datasets'),
    data = path.join(root, 'data');
  await fs.mkdir(path.join(datasets, 'test'), { recursive: true });
  const original = await sharp({ create: { width: 128, height: 128, channels: 3, background: '#aa5577' } })
    .png()
    .toBuffer();
  await fs.writeFile(path.join(datasets, 'test', 'a.png'), original);
  await fs.writeFile(path.join(datasets, 'test', 'a.txt'), 'original caption');
  const st = await new StudioStore(data, datasets, 'test').init();
  await st.read();
  return { root, st, original, cleanup: () => fs.rm(root, { recursive: true, force: true }) };
}
test('model-shaped signals infer face/body/context automatically; absent humans remain explicit', () => {
  for (const kind of ['face', 'body', 'variety'] as const)
    assert.equal(inferredCategory(validateSignals(signals(kind), sha, config)).category, kind);
  assert.equal(inferredCategory(signals('none')).category, 'unclassified');
  assert.throws(() => validateSignals({ ...signals(), sha: 'b'.repeat(64) }, sha, config), /stale/);
  assert.throws(() => validateSignals({ ...signals(), depth: null }, sha, config));
  assert.throws(() =>
    validateSignals({ ...signals(), persons: [{ ...signals().persons[0], keypoints: [] }] }, sha, config),
  );
});
test('same identity preserves pose/depth diversity, thirds and stable ordering', () => {
  const rows = [
    image(1, 'face'),
    image(2, 'face'),
    image(3, 'body'),
    image(4, 'body'),
    image(5, 'variety'),
    image(6, 'variety'),
  ];
  for (const [i, row] of rows.entries()) {
    row.analysis.model!.depth.grid.fill(i * 0.15);
    row.analysis.model!.persons[0].keypoints = row.analysis.model!.persons[0].keypoints.map(k => [
      Math.min(1, k[0] + i * 0.05),
      k[1],
      k[2],
    ]);
  }
  const a = select(rows, 6),
    b = select([...rows].reverse(), 6);
  assert.equal(a.complete, true);
  assert.deepEqual(
    a.selected.map(x => x.id),
    b.selected.map(x => x.id),
  );
  assert.deepEqual(a.quotas, { face: 2, body: 2, variety: 2 });
  assert.equal(new Set(Object.values(a.groups)).size, 6);
  assert.equal(
    select(
      rows.filter(x => x.category !== 'variety'),
      6,
    ).deficits.variety,
    2,
  );
});
test('proposal preserves reviewed inclusion/category and exposes quota shortages', () => {
  const rows = [image(1, 'face'), image(2, 'body'), image(3, 'variety')];
  for (const r of rows) r.analysis.model!.config = config;
  const s: any = { images: rows, settings: { count: 3, testId: null }, revision: 0 };
  rows[0].excluded = 1;
  (rows[0] as any).reviewRevision = 1;
  applyProposal(s, config);
  assert.equal(rows[0].excluded, 1);
  assert.equal(s.analysisFlow.deficits.face, 1);
  assert.equal(s.analysisFlow.count, 2);
});
test('real Store refuses unavailable host without creating a native job or fabricated signals', async () => {
  const f = await fixture();
  try {
    const state = await prepareAnalysis(
      f.st,
      {
        job: {
          create: () => {
            throw Error('must not create');
          },
        },
      },
      '/none',
      { platform: 'darwin', gpu0: false, preview: true },
    );
    assert.equal(state.analysisFlow?.phase, 'unavailable');
    assert.equal(state.images[0].analysis, null);
    assert.equal(state.images[0].category, 'unclassified');
    assert.equal(state.images[0].caption, 'original caption');
    assert.equal(state.jobs.length, 0);
  } finally {
    await f.cleanup();
  }
});
test('actual result artifacts bind SHA/config/depth; scan and late apply preserve review/captions', async () => {
  const f = await fixture();
  try {
    const { config: cfg } = await analysisConfiguration();
    let state = await f.st.read(),
      img = state.images[0];
    const folder = path.join(f.st.folder, 'analysis', 'fixture');
    await fs.mkdir(path.join(folder, 'result-000000'), { recursive: true });
    const depth = await sharp({ create: { width: 8, height: 8, channels: 3, background: '#555555' } })
      .png()
      .toBuffer();
    const output = path.join(folder, 'result-000000');
    await fs.writeFile(path.join(output, 'depth.png'), depth);
    const model = { ...signals('face'), sha: img.sha, config: cfg, depth: { ...signals().depth, sha: hash(depth) } };
    await fs.writeFile(path.join(output, 'result.json'), JSON.stringify(model));
    const link: JobLink = {
      kind: 'analysis',
      state: 'linked',
      name: 'fixture',
      config: {},
      folder,
      scope: [structuredClone(img)],
    };
    state = await f.st.edit(state.revision, [img.id], { category: 'variety', excluded: 1 });
    state = await applyAnalysisResults(f.st, link, cfg);
    assert.equal(state.images[0].category, 'variety');
    assert.equal(state.images[0].excluded, 1);
    assert.equal(state.images[0].caption, 'original caption');
    const reloaded = await new StudioStore(path.dirname(f.st.root), path.dirname(f.st.datasetRoot), 'test').init();
    assert.deepEqual((await reloaded.read()).images[0].pose, model.persons);
    await fs.writeFile(path.join(output, 'depth.png'), Buffer.from('tampered'));
    await assert.rejects(() => applyAnalysisResults(f.st, link, cfg), /checksum/);
    await fs.writeFile(
      path.join(f.st.datasetRoot, 'a.png'),
      await sharp({ create: { width: 128, height: 128, channels: 3, background: '#118844' } })
        .png()
        .toBuffer(),
    );
    state = await f.st.read();
    assert.notEqual(state.images[0].id, img.id);
    assert.equal(state.images[0].analysis, null);
  } finally {
    await f.cleanup();
  }
});
test('SQLite lock blocks live competing process, releases on real SIGKILL, preserves state', async () => {
  const f = await fixture();
  let child: any;
  try {
    await ownedLock(f.st.folder, async () => {});
    const file = path.join(f.st.folder, '.operation-lock.db');
    child = spawn(
      process.execPath,
      [
        '-e',
        "const sqlite=require('sqlite3');const db=new sqlite.Database(process.argv[1]);db.exec('BEGIN IMMEDIATE',e=>{if(e)throw e;process.stdout.write('locked\\n');});setInterval(()=>{},1000)",
        file,
      ],
      { cwd: process.cwd(), stdio: ['ignore', 'pipe', 'pipe'] },
    );
    await once(child.stdout, 'data');
    await assert.rejects(() => ownedLock(f.st.folder, async () => {}), /operation active/);
    const ended = once(child, 'exit');
    child.kill('SIGKILL');
    await ended;
    await ownedLock(f.st.folder, async () => {
      assert.equal((await f.st.raw()).images[0].caption, 'original caption');
    });
    await fs.mkdir(path.join(f.st.folder, 'lock'));
    await assert.rejects(() => f.st.read(), /owner death unproven/);
  } finally {
    child?.kill('SIGKILL');
    await f.cleanup();
  }
});
test('operation-lock symlinks and result path escapes are refused', async () => {
  const f = await fixture();
  try {
    await fs.unlink(path.join(f.st.folder, '.operation-lock.db'));
    await fs.symlink('/tmp/foreign-lock', path.join(f.st.folder, '.operation-lock.db'));
    await assert.rejects(() => f.st.read(), /unsafe/);
  } finally {
    await f.cleanup();
  }
});

test('owned process receipt proves absence/reused PID, refuses live/foreign/unknown owners', async () => {
  const { provenAnalysisDeath } = await import('../src/datasetStudio/analysisFlow');
  const f = await fixture();
  try {
    const folder = path.join(f.st.folder, 'analysis', 'receipt');
    await fs.mkdir(folder, { recursive: true });
    const link: JobLink = { name: 'owned-analysis', kind: 'analysis', state: 'linked', config: {}, folder };
    const receipt = {
      pid: 12345,
      start: '999',
      host: 'same-host',
      boot: 'same-boot',
      namespace: 'pid:[fixture]',
      name: link.name,
    };
    await fs.writeFile(path.join(folder, 'runtime.json'), JSON.stringify(receipt));
    const stat = (start: string) =>
      '12345 (fixture) ' + Array.from({ length: 22 }, (_, i) => (i === 19 ? start : '0')).join(' ');
    const probe = {
      platform: 'linux',
      host: 'same-host',
      boot: async () => 'same-boot',
      namespace: async () => 'pid:[fixture]',
      stat: async () => stat('999'),
    };
    assert.equal(await provenAnalysisDeath(f.st, link, { pid: 12345 }, probe), false);
    assert.equal(
      await provenAnalysisDeath(f.st, link, { pid: 12345 }, { ...probe, stat: async () => stat('1000') }),
      true,
    );
    assert.equal(
      await provenAnalysisDeath(
        f.st,
        link,
        { pid: 12345 },
        {
          ...probe,
          stat: async () => {
            throw Object.assign(Error('absent'), { code: 'ENOENT' });
          },
        },
      ),
      true,
    );
    assert.equal(await provenAnalysisDeath(f.st, link, { pid: 12345 }, { ...probe, host: 'different-host' }), false);
    assert.equal(
      await provenAnalysisDeath(f.st, link, { pid: 12345 }, { ...probe, namespace: async () => 'pid:[different]' }),
      false,
    );
    assert.equal(
      await provenAnalysisDeath(f.st, link, { pid: 12345 }, { ...probe, boot: async () => 'different-boot' }),
      false,
    );
    assert.equal(
      await provenAnalysisDeath(
        f.st,
        link,
        { pid: 12345 },
        {
          ...probe,
          stat: async () => {
            throw Object.assign(Error('permission'), { code: 'EACCES' });
          },
        },
      ),
      false,
    );
  } finally {
    await f.cleanup();
  }
});

test('legacy manual inclusion migrates as reviewed; completed old-config analysis cannot satisfy new config', async () => {
  const f = await fixture();
  try {
    let s = await f.st.read();
    s.images[0].category = 'body';
    s.images[0].excluded = 1;
    delete s.images[0].reviewRevision;
    delete s.images[0].categorySource;
    await f.st.save(s);
    s = await f.st.read();
    assert.equal(s.images[0].reviewRevision, 1);
    assert.equal(s.images[0].categorySource, 'manual');
    s.images[0].analysis = { ...analyzePixels(new Uint8ClampedArray(64 * 64 * 4).fill(120)), model: signals('body') };
    assert.equal(applyProposal(s, 'new-config'), false);
    assert.equal(s.images[0].excluded, 1);
    // An old inclusion edit can leave default flags/category, with only revision
    // history distinguishing it from a fresh uncategorized upload.
    s.images[0].category = 'unclassified';
    s.images[0].excluded = 0;
    s.images[0].revision = (s.images[0].captionRevision ?? 0) + 1;
    delete s.images[0].reviewRevision;
    delete s.images[0].categorySource;
    await f.st.save(s);
    s = await f.st.read();
    assert.equal(s.images[0].reviewRevision, 1);
    assert.equal(s.images[0].categorySource, undefined);
  } finally {
    await f.cleanup();
  }
});

test('real native SQLite analysis queue persists one job, preserves paused caption queue and applies completed scope', async () => {
  const { PrismaClient } = await import('@prisma/client');
  const { execFile } = await import('node:child_process');
  const { promisify } = await import('node:util');
  const { reconcileAnalysis } = await import('../src/datasetStudio/analysisFlow');
  const f = await fixture();
  let db: any;
  try {
    const sqlite = path.join(f.root, 'native.db'),
      schema = path.join(f.root, 'schema.prisma');
    await fs.writeFile(sqlite, Buffer.alloc(0));
    await fs.writeFile(
      schema,
      (await fs.readFile(path.resolve('prisma/schema.prisma'), 'utf8')).replace(
        /url\s*=\s*"[^"]+"/,
        'url = "file:' + sqlite + '"',
      ),
    );
    await promisify(execFile)(
      process.execPath,
      [path.resolve('node_modules/prisma/build/index.js'), 'db', 'push', '--schema', schema, '--skip-generate'],
      { timeout: 30000 },
    );
    db = new PrismaClient({ datasourceUrl: 'file:' + sqlite });
    const { config: cfg } = await analysisConfiguration();
    let s = await f.st.read(),
      original = structuredClone(s.images[0]);
    const folder = path.join(f.st.folder, 'analysis', 'queue-fixture'),
      name = 'studio-analysis-fixture';
    const link: JobLink = {
      kind: 'analysis',
      name,
      state: 'intent',
      folder,
      scope: [original],
      config: {
        job: 'extension',
        config: {
          name,
          process: [
            { type: 'dataset_studio_analysis', request: path.join(folder, 'request.json'), sqlite_db_path: sqlite },
          ],
        },
      },
      automatic: {
        id: 'queue-fixture',
        digest: 'fixture',
        gpu: '0',
        phase: 'prepared',
        createdAt: new Date().toISOString(),
      },
    };
    s.jobs.push(link);
    await f.st.save(s);
    await db.queue.create({ data: { gpu_ids: '0', is_running: false } });
    const caption = await db.job.create({
      data: { name: 'unrelated-caption', gpu_ids: '0', job_config: '{}', job_type: 'caption', status: 'queued' },
    });
    const host = { platform: 'linux', gpu0: true, preview: false };
    s = await reconcileAnalysis(f.st, 'queue-fixture', db, host);
    assert.equal(s.analysisFlow?.phase, 'failed');
    assert.equal((await db.queue.findUnique({ where: { gpu_ids: '0' } })).is_running, false);
    assert.equal((await db.job.findUnique({ where: { id: caption.id } })).status, 'queued');
    await db.job.update({ where: { id: caption.id }, data: { status: 'completed' } });
    s = await reconcileAnalysis(f.st, 'queue-fixture', db, host, true);
    assert.equal(s.analysisFlow?.phase, 'active');
    await reconcileAnalysis(f.st, 'queue-fixture', db, host);
    assert.equal(await db.job.count({ where: { job_type: 'analysis' } }), 1);
    const managed = s.jobs.find(x => x.kind === 'analysis')!;
    // A stale missing receipt is unknown on every host, including Linux.
    await db.job.update({
      where: { id: managed.jobId },
      data: { status: 'running', pid: 12345, updated_at: new Date(Date.now() - 180000) },
    });
    s = await reconcileAnalysis(f.st, 'queue-fixture', db, host);
    assert.equal(s.jobs.find(x => x.kind === 'analysis')!.automatic!.phase, 'blocked');
    assert.equal((await db.job.findUnique({ where: { id: managed.jobId } })).status, 'running');
    assert.equal(await db.job.count({ where: { job_type: 'analysis' } }), 1);

    // The production background seam receives a native row before Python's receipt.
    const now = Date.now(),
      stat = '12345 (fixture) ' + Array.from({ length: 22 }, (_, i) => (i === 19 ? '999' : '0')).join(' ');
    const probe = {
      platform: 'linux',
      host: 'fixture-host',
      boot: async () => 'fixture-boot',
      namespace: async () => 'pid:[fixture]',
      stat: async () => stat,
      now: () => now,
    };
    await db.job.update({ where: { id: managed.jobId }, data: { updated_at: new Date(now) } });
    const nativeBefore = await db.job.findUnique({ where: { id: managed.jobId } }),
      queueBefore = await db.queue.findUnique({ where: { gpu_ids: '0' } });
    s = await reconcilePendingAnalysis(f.st, db, host, probe);
    assert.equal(s.jobs.find(x => x.kind === 'analysis')!.automatic!.phase, 'active');
    assert.match(s.jobs.find(x => x.kind === 'analysis')!.automatic!.reason!, /Avvio/);
    assert.deepEqual(await db.job.findUnique({ where: { id: managed.jobId } }), nativeBefore);
    assert.deepEqual(await db.queue.findUnique({ where: { gpu_ids: '0' } }), queueBefore);
    assert.equal(
      await analysisProcessState(f.st, managed, { ...nativeBefore, updated_at: new Date(now - 120001) }, probe),
      'unknown',
    );
    assert.equal(await analysisProcessState(f.st, managed, { ...nativeBefore, status: 'completed' }, probe), 'unknown');
    const receipt = {
      pid: 12345,
      start: '999',
      host: 'fixture-host',
      boot: 'fixture-boot',
      namespace: 'pid:[fixture]',
      name: managed.name,
    };
    await fs.writeFile(path.join(folder, 'runtime.json'), JSON.stringify({ ...receipt, host: 'foreign-host' }));
    s = await reconcilePendingAnalysis(f.st, db, host, probe);
    assert.equal(s.jobs.find(x => x.kind === 'analysis')!.automatic!.phase, 'blocked');
    await fs.writeFile(path.join(folder, 'runtime.json'), 'invalid-json');
    s = await reconcilePendingAnalysis(f.st, db, host, probe);
    assert.equal(s.jobs.find(x => x.kind === 'analysis')!.automatic!.phase, 'blocked');
    await fs.writeFile(path.join(folder, 'runtime.json'), JSON.stringify(receipt));
    s = await reconcilePendingAnalysis(f.st, db, host, probe);
    assert.equal(s.jobs.find(x => x.kind === 'analysis')!.automatic!.phase, 'active');
    assert.equal(s.jobs.find(x => x.kind === 'analysis')!.automatic!.reason, undefined);
    assert.deepEqual(await db.job.findUnique({ where: { id: managed.jobId } }), nativeBefore);
    assert.deepEqual(await db.queue.findUnique({ where: { gpu_ids: '0' } }), queueBefore);

    const output = path.join(folder, 'result-000000');
    await fs.mkdir(output, { recursive: true });
    const depth = await sharp({ create: { width: 8, height: 8, channels: 3, background: '#333333' } })
      .png()
      .toBuffer();
    await fs.writeFile(path.join(output, 'depth.png'), depth);
    await fs.writeFile(
      path.join(output, 'result.json'),
      JSON.stringify({
        ...signals('face'),
        sha: original.sha,
        config: cfg,
        depth: { ...signals().depth, sha: hash(depth) },
      }),
    );
    // Inclusion review while inference runs must preserve exclusion, while category still infers.
    s = await f.st.read();
    await f.st.edit(s.revision, [original.id], { excluded: 1 });
    await db.job.update({ where: { id: managed.jobId }, data: { status: 'completed', step: 1, total_steps: 1 } });
    s = await reconcilePendingAnalysis(f.st, db, host, probe);
    assert.equal(s.jobs.find(x => x.kind === 'analysis')!.automatic!.phase, 'applied');
    assert.equal(s.images[0].excluded, 1);
    assert.equal(s.images[0].category, 'face');
    assert.equal(s.images[0].caption, 'original caption');
    assert.equal(await db.job.count({ where: { job_type: 'analysis' } }), 1);
  } finally {
    await db?.$disconnect();
    await f.cleanup();
  }
});

test('export retains pixel/provenance only; raw vectors, keypoints and depth maps stay local', async () => {
  const f = await fixture();
  try {
    let s = await f.st.read();
    s.settings.count = 1;
    s.images[0].category = 'face';
    s.images[0].analysis = {
      ...analyzePixels(new Uint8ClampedArray(64 * 64 * 4).fill(120)),
      model: { ...signals('face'), sha: s.images[0].sha },
    };
    await f.st.save(s);
    s = await f.st.read();
    s = await f.st.prepareExport(s.revision);
    const exported = s.snapshots[0].manifest.source[0].analysis;
    assert.equal(exported.model.version, MODEL_ANALYZER);
    assert.equal(exported.model.faceCount, 1);
    assert.equal(exported.model.depthMapSha, 'd'.repeat(64));
    assert.equal(exported.model.faces, undefined);
    assert.equal(exported.model.persons, undefined);
    assert.equal(exported.model.depth, undefined);
    assert.equal(s.images[0].analysis!.model!.faces[0].embedding.length, 512);
  } finally {
    await f.cleanup();
  }
});

test('proposal membership CAS advances even when counts/deficits unchanged; pixel refresh keeps model cache', async () => {
  const rows = [image(1, 'face'), image(2, 'face')];
  for (const r of rows) r.analysis.model!.config = config;
  const s: any = { images: rows, settings: { count: 1, testId: null }, revision: 0 };
  applyProposal(s, config);
  const before = s.revision;
  rows[0].excluded = 1;
  rows[1].excluded = 0; // Simulate interrupted older automatic membership, no manual review.
  applyProposal(s, config);
  assert.ok(s.revision > before);
  const f = await fixture();
  try {
    let state = await f.st.read();
    state.images[0].analysis = { ...rows[0].analysis, model: { ...signals(), sha: state.images[0].sha } };
    await f.st.save(state);
    state = await f.st.read();
    state = await f.st.analyze(state.revision, state.images[0].id);
    assert.equal(state.images[0].analysis!.model!.version, MODEL_ANALYZER);
  } finally {
    await f.cleanup();
  }
});

// Isolate TOOLKIT_ROOT with a tiny verified artifact fixture. This exercises the
// real preparation path and Prisma's global name constraint without ML imports,
// model downloads, or changing the repository's pinned model manifest.
async function importIdentityRegression(scenario: 'fresh' | 'legacy' | 'command') {
  const f = await fixture();
  try {
    await fs.mkdir(path.join(f.root, 'toolkit', 'ui'), { recursive: true });
    const toolkit = await fs.realpath(path.join(f.root, 'toolkit'));
    await fs.symlink(path.resolve('node_modules'), path.join(toolkit, 'ui', 'node_modules'));
    const script = path.join(f.root, 'identity-regression.cjs');
    await fs.writeFile(
      script,
      String.raw`
const assert=require('node:assert/strict'), fs=require('node:fs/promises'), path=require('node:path');
const {execFileSync}=require('node:child_process');
const {PrismaClient}=require(${JSON.stringify(path.resolve('node_modules/@prisma/client'))});
const sharp=require(${JSON.stringify(path.resolve('node_modules/sharp'))});
const {StudioStore,hash}=require(${JSON.stringify(path.resolve('src/datasetStudio/store.ts'))});
const {stableJSON}=require(${JSON.stringify(path.resolve('src/datasetStudio/domain.ts'))});
const {MODEL_ANALYZER}=require(${JSON.stringify(path.resolve('src/datasetStudio/analysisPolicy.ts'))});
const {prepareAnalysis,executeAnalysisCommand,reconcileAnalysis,analysisConfiguration}=require(${JSON.stringify(path.resolve('src/datasetStudio/analysisFlow.ts'))});
const root=${JSON.stringify(toolkit)}, scenario=${JSON.stringify(scenario)};
const sqlite=path.join(root,'native.db'), host={platform:'linux',gpu0:true,preview:false};
let db;
(async()=>{
 const marker=Buffer.from('fixture!'), manifest={version:MODEL_ANALYZER,models:{fixture:{files:{'fixture.bin':{size:marker.length,sha256:hash(marker)}}}}};
 await fs.mkdir(path.join(root,'extensions_built_in/dataset_studio_analysis'),{recursive:true});
 await fs.writeFile(path.join(root,'extensions_built_in/dataset_studio_analysis/models.json'),JSON.stringify(manifest));
 await fs.mkdir(path.join(root,'cache/analysis-models/fixture'),{recursive:true});
 await fs.writeFile(path.join(root,'cache/analysis-models/fixture/fixture.bin'),marker);
 const schema=path.join(root,'schema.prisma');await fs.writeFile(sqlite,Buffer.alloc(0));
 await fs.writeFile(schema,(await fs.readFile(${JSON.stringify(path.resolve('prisma/schema.prisma'))},'utf8')).replace(/url\s*=\s*"[^"]+"/,'url = "file:'+sqlite+'"'));
 execFileSync(process.execPath,[${JSON.stringify(path.resolve('node_modules/prisma/build/index.js'))},'db','push','--schema',schema,'--skip-generate'],{timeout:30000,stdio:'pipe'});
 db=new PrismaClient({datasourceUrl:'file:'+sqlite});const {config}=await analysisConfiguration();
 const original=await fs.readFile(${JSON.stringify(path.join(f.st.datasetRoot, 'a.png'))});
 async function store(name,filename='a.png'){
  const folder=path.join(root,'datasets',name);await fs.mkdir(folder,{recursive:true});
  await fs.writeFile(path.join(folder,filename),original);await fs.writeFile(path.join(folder,filename.replace('.png','.txt')),'original caption');
  const st=await new StudioStore(path.join(root,'data'),path.join(root,'datasets'),name).init();await st.read();return st;
 }
 async function legacy(st){
  const state=await st.read(),scope=structuredClone(state.images),digest=hash(stableJSON({config,items:scope.map(x=>({id:x.id,sha:x.sha}))}));
  const folder=path.join(st.folder,'analysis',digest),name='studio-analysis-'+digest.slice(0,24);
  const link={kind:'analysis',state:'intent',name,folder,scope,config:{job:'extension',config:{name,process:[{type:'dataset_studio_analysis',request:path.join(folder,'request.json'),sqlite_db_path:sqlite}]}},automatic:{id:st.name+'-legacy',digest,gpu:'0',phase:'prepared',createdAt:new Date().toISOString()}};
  state.jobs.push(link);await st.save(state);return link;
 }
 const active=state=>state.jobs.find(x=>x.kind==='analysis'&&x.automatic.phase!=='dismissed');
 async function complete(st){
  const state=await st.read(),link=active(state),img=state.images[0],output=path.join(link.folder,'result-000000');
  await fs.mkdir(output,{recursive:true});const depth=await sharp({create:{width:8,height:8,channels:3,background:'#444444'}}).png().toBuffer();
  await fs.writeFile(path.join(output,'depth.png'),depth);const embedding=Array(512).fill(0);embedding[0]=1;
  await fs.writeFile(path.join(output,'result.json'),JSON.stringify({version:MODEL_ANALYZER,sha:img.sha,config,faces:[{box:[.2,.1,.7,.7],confidence:.95,embedding}],persons:[],depth:{grid:Array(64).fill(.5),map:'depth.png',sha:hash(depth),relative:true},runtime:{device:'fixture, no inference'}}));
  await db.job.update({where:{id:link.jobId},data:{status:'completed',step:1,total_steps:1}});
  return reconcileAnalysis(st,link.automatic.id,db,host);
 }
 const a=await store('alpha'),b=await store('beta');
 assert.equal((await a.read()).images[0].id,(await b.read()).images[0].id);assert.notEqual(a.datasetRoot,b.datasetRoot);
 let first;
 if(scenario==='legacy'){const link=await legacy(a);first=await reconcileAnalysis(a,link.automatic.id,db,host);}
 else if(scenario==='command'){
  let local=await a.read(); const draftTarget=6;
  local=await a.edit(local.revision,[local.images[0].id],{excluded:1});await a.read();
  assert.equal(await db.job.count(),0,'reads, local target draft and image click/edit must not enqueue');
  assert.equal(local.settings.count,3);assert.equal(draftTarget,6);
  first=await executeAnalysisCommand(a,db,sqlite,host,{revision:local.revision,count:6});
  assert.equal(first.settings.count,6);assert.equal(await db.job.count(),1);
  await executeAnalysisCommand(a,db,sqlite,host,{revision:local.revision,count:6});
  assert.equal(await db.job.count(),1,'duplicate explicit command deduplicates');
  await assert.rejects(executeAnalysisCommand(a,db,sqlite,host,{revision:(await a.read()).revision,count:9}),/Attendi/);
 } else first=await prepareAnalysis(a,db,sqlite,host);
 assert.equal(first.analysisFlow.phase,'active',first.analysisFlow.reason);first=await complete(a);
 if(scenario==='command'){first=await executeAnalysisCommand(a,db,sqlite,host,{revision:first.revision,count:3});assert.equal(await db.job.count(),1,'completed cache target change creates no GPU job');assert.equal(first.images[0].excluded,1,'manual exclusion survives proposal');}
 first=await a.edit(first.revision,[first.images[0].id],{category:'variety',excluded:1});
 const rowA=await db.job.findUnique({where:{id:active(first).jobId}}),stateA=await fs.readFile(path.join(a.folder,'state.json'));
 const resultA=await fs.readFile(path.join(active(first).folder,'result-000000/result.json'));
 let second,old,requestBefore;
 if(scenario==='legacy'){
  old=await legacy(b);second=await reconcileAnalysis(b,old.automatic.id,db,host);
  assert.equal(second.analysisFlow.phase,'failed');assert.match(second.analysisFlow.reason,/identity\/config differs/);
  requestBefore=await fs.readFile(path.join(old.folder,'request.json'));
  const noRetry=await prepareAnalysis(b,db,sqlite,host);assert.equal(noRetry.jobs.length,1);assert.equal(noRetry.jobs[0].automatic.phase,'failed');assert.equal(await db.job.count(),1);
 }
 second=await prepareAnalysis(b,db,sqlite,host,scenario==='legacy');
 assert.equal(second.analysisFlow.phase,'active','second independent import must get its own native job');
 const managed=active(second);assert.notEqual(managed.name,rowA.name);assert.notEqual(managed.jobId,rowA.id);
 assert.equal((await db.job.findUnique({where:{id:managed.jobId}})).job_ref,b.datasetRoot);
 assert.equal(managed.automatic.digest,active(first).automatic.digest,'content/config digest remains unchanged');
 if(old){
  const archived=second.jobs.find(x=>x.automatic.id===old.automatic.id);assert.equal(archived.automatic.phase,'dismissed');
  assert.equal(archived.name,old.name);assert.deepEqual(archived.config,old.config);assert.deepEqual(archived.scope,old.scope);
  assert.notEqual(managed.folder,old.folder);assert.ok(managed.folder.startsWith(old.folder+path.sep));
  assert.deepEqual(await fs.readFile(path.join(old.folder,'request.json')),requestBefore);
 }
 await prepareAnalysis(b,db,sqlite,host);await prepareAnalysis(b,db,sqlite,host,true);
 assert.equal(await db.job.count(),2,'same-store prepare/retry must deduplicate');
 assert.equal((await b.read()).jobs.filter(x=>x.automatic.phase!=='dismissed').length,1);
 await complete(b);
 assert.deepEqual(await db.job.findUnique({where:{id:rowA.id}}),rowA);
 assert.deepEqual(await fs.readFile(path.join(a.folder,'state.json')),stateA);
 assert.deepEqual(await fs.readFile(path.join(active(first).folder,'result-000000/result.json')),resultA);
 assert.equal(hash(await fs.readFile(path.join(a.datasetRoot,'a.png'))),hash(original));
 if(scenario==='legacy'){
  const own=await store('owned-legacy','own.png'),ownLink=await legacy(own);
  let ownState=await reconcileAnalysis(own,ownLink.automatic.id,db,host);const ownID=active(ownState).jobId;
  const ownRequest=await fs.readFile(path.join(ownLink.folder,'request.json'));
  ownState=await prepareAnalysis(own,db,sqlite,host,true);assert.equal(active(ownState).jobId,ownID);assert.equal(active(ownState).name,ownLink.name);assert.deepEqual(active(ownState).config,ownLink.config);
  assert.deepEqual(await fs.readFile(path.join(ownLink.folder,'request.json')),ownRequest);await complete(own);
  const malformed=await store('malformed'),bad=await legacy(malformed);await reconcileAnalysis(malformed,bad.automatic.id,db,host);
  const badRequest=path.join(bad.folder,'request.json'),tampered=JSON.parse(await fs.readFile(badRequest,'utf8'));tampered.dataset=a.datasetRoot;await fs.writeFile(badRequest,JSON.stringify(tampered));
  const badBytes=await fs.readFile(badRequest);let refused=await prepareAnalysis(malformed,db,sqlite,host,true);assert.equal(refused.jobs.length,1);assert.equal(refused.jobs[0].automatic.phase,'failed');assert.deepEqual(await fs.readFile(badRequest),badBytes);
  const linked=await store('linked-foreign'),foreign=await legacy(linked);const linkedState=await linked.read();linkedState.jobs[0].jobId=rowA.id;linkedState.jobs[0].state='linked';linkedState.jobs[0].automatic.phase='failed';await linked.save(linkedState);
  refused=await prepareAnalysis(linked,db,sqlite,host,true);assert.equal(refused.jobs.length,1);assert.equal(refused.jobs[0].automatic.phase,'failed');assert.equal(refused.jobs[0].jobId,rowA.id);
  const unknown=await store('unknown','unknown.png'),unknownLink=await legacy(unknown);
  await db.job.create({data:{name:unknownLink.name,gpu_ids:'0',job_type:'analysis',job_ref:null,job_config:JSON.stringify(unknownLink.config),status:'completed'}});
  await reconcileAnalysis(unknown,unknownLink.automatic.id,db,host);refused=await prepareAnalysis(unknown,db,sqlite,host,true);assert.equal(refused.jobs.length,1);assert.equal(refused.jobs[0].automatic.phase,'failed');
  assert.equal(await db.job.count(),4);assert.deepEqual(await db.job.findUnique({where:{id:rowA.id}}),rowA);
  assert.deepEqual(await fs.readFile(path.join(a.folder,'state.json')),stateA);assert.deepEqual(await fs.readFile(path.join(old.folder,'request.json')),requestBefore);
 }
 console.log('identity-regression '+scenario+' passed');
})().catch(e=>{console.error(e);process.exitCode=1}).finally(async()=>{await db?.$disconnect()});
`,
    );
    const { execFile } = await import('node:child_process');
    const { promisify } = await import('node:util');
    await promisify(execFile)(
      process.execPath,
      ['--import', path.resolve('node_modules/tsx/dist/loader.mjs'), script],
      {
        cwd: path.join(toolkit, 'ui'),
        env: {
          ...process.env,
          TSX_TSCONFIG_PATH: path.resolve('tsconfig.json'),
          DATASET_STUDIO_ROOT: toolkit,
          DATASET_STUDIO_ANALYSIS_ENABLED: '1',
        },
        timeout: 60000,
      },
    );
  } finally {
    await f.cleanup();
  }
}
test('independent identical imports use distinct native SQLite identities and preserve the first review', () =>
  importIdentityRegression('fresh'));
test('explicit retry recovers only a proven unlinked legacy collision; old own and unknown identities remain intact', () =>
  importIdentityRegression('legacy'));

test('explicit analysis command persists target once; reads/drafts/edits do not enqueue, duplicate/cache commands preserve review', () =>
  importIdentityRegression('command'));
