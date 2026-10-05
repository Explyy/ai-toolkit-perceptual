import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import sharp from 'sharp';
import { StudioStore, hash } from '../src/datasetStudio/store';
import {
  generateCaption,
  reconcileCaption,
  dismissCaption,
  needsCaptionReconcile,
} from '../src/datasetStudio/captionFlow';
import { defaultCaptionModel, preferences, captionModels } from '../src/datasetStudio/captionModels';
const host = { platform: 'linux', preview: false, gpu0: true };
const prefs = { key: defaultCaptionModel.key, instructions: 'Descrivi soltanto ciò che vedi.' };
function database() {
  const rows = new Map<string, any>(),
    queues = new Map<string, any>();
  let created = 0,
    queueWrites = 0,
    updates = 0;
  let failCommit = false,
    failCreate = false;
  const db: any = {
    rows,
    queues,
    get created() {
      return created;
    },
    get queueWrites() {
      return queueWrites;
    },
    get updates() {
      return updates;
    },
    set failCommit(x: boolean) {
      failCommit = x;
    },
    set failCreate(x: boolean) {
      failCreate = x;
    },
    job: {
      findUnique: async ({ where }: any) =>
        [...rows.values()].find(x => (where.name ? x.name === where.name : x.id === where.id)) ?? null,
      create: async ({ data }: any) => {
        const row = { ...data, id: 'caption-' + ++created, step: 0, queue_position: 0 };
        rows.set(row.id, row);
        if (failCreate) {
          failCreate = false;
          throw Error('lost create response');
        }
        return structuredClone(row);
      },
      findFirst: async ({ where }: any) =>
        [...rows.values()].find(
          x => x.gpu_ids === where.gpu_ids && x.id !== where.id.not && where.status.in.includes(x.status),
        ) ?? null,
      aggregate: async () => ({
        _max: { queue_position: Math.max(0, ...[...rows.values()].map(x => x.queue_position ?? 0)) },
      }),
      updateMany: async ({ where, data }: any) => {
        const row = rows.get(where.id);
        if (!row || row.status !== where.status) return { count: 0 };
        Object.assign(row, data);
        updates++;
        return { count: 1 };
      },
    },
    queue: {
      findUnique: async ({ where }: any) =>
        [...queues.values()].find(x => (where.gpu_ids ? x.gpu_ids === where.gpu_ids : x.id === where.id)) ?? null,
      create: async ({ data }: any) => {
        const row = { ...data, id: 'queue-' + data.gpu_ids };
        queues.set(row.id, row);
        queueWrites++;
        return row;
      },
      update: async ({ where, data }: any) => {
        const row = queues.get(where.id);
        Object.assign(row, data);
        queueWrites++;
        return row;
      },
    },
    $transaction: async (fn: any) => {
      const a = structuredClone(rows),
        b = structuredClone(queues);
      let out;
      try {
        out = await fn(db);
      } catch (e) {
        rows.clear();
        queues.clear();
        for (const [k, v] of a) rows.set(k, v);
        for (const [k, v] of b) queues.set(k, v);
        throw e;
      }
      if (failCommit) {
        failCommit = false;
        throw Error('lost committed response');
      }
      return out;
    },
  };
  return db;
}
async function fixture() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'caption-flow-'));
  const datasets = path.join(root, 'datasets');
  await fs.mkdir(path.join(datasets, 'Subject A'), { recursive: true });
  for (let i = 1; i <= 3; i++) {
    await fs.writeFile(
      path.join(datasets, 'Subject A', `${i}.png`),
      await sharp({ create: { width: 32, height: 48, channels: 3, background: { r: i * 50, g: 80, b: 150 } } })
        .png()
        .toBuffer(),
    );
    await fs.writeFile(path.join(datasets, 'Subject A', `${i}.txt`), 'Original ' + i);
  }
  const st = await new StudioStore(path.join(root, 'data'), datasets, 'Subject A').init();
  return { root, st, db: database(), cleanup: () => fs.rm(root, { recursive: true, force: true }) };
}
async function generate(f: any, id = randomUUID()) {
  const s = await f.st.read();
  return generateCaption(f.st, s.revision, { requestId: id, preferences: prefs }, f.db, '/synthetic/aitk.db', host);
}
async function outputs(f: any) {
  const s = await f.st.read(),
    link = s.jobs.find((x: any) => x.automatic)!;
  for (const [i] of link.scope.entries())
    await fs.writeFile(path.join(link.folder, String(i + 1).padStart(6, '0') + '.txt'), 'Generated ' + i);
  const row = f.db.rows.get(link.jobId);
  Object.assign(row, { status: 'completed', step: link.scope.length, total_steps: link.scope.length });
  return link;
}
function current(s: any) {
  return s.jobs.find((x: any) => x.automatic)!;
}

test('single authoritative grouped native model and persisted preferences are additive/CAS/bounded', async () => {
  const f = await fixture();
  try {
    assert.equal(defaultCaptionModel.captioner, 'Qwen3VLCaptioner');
    assert.ok(captionModels.every(x => JSON.parse(x.key)[1] === x.model));
    let s = await f.st.read();
    s = await f.st.captionSettings(s.revision, prefs, 0);
    assert.equal((await f.st.read()).captionPreferences!.instructions, prefs.instructions);
    assert.equal((await f.st.captionSettings(s.revision, prefs, 1)).revision, s.revision);
    await assert.rejects(f.st.captionSettings(s.revision, { ...prefs, instructions: 'changed' }, 0), /altrove/);
    assert.throws(() => preferences({ ...prefs, key: 'invented' }));
    assert.throws(() => preferences({ ...prefs, instructions: 'a'.repeat(8001) }));
  } finally {
    await f.cleanup();
  }
});

test('included unclassified scope creates and activates only one new caption job; originals/legacy jobs stay untouched', async () => {
  const f = await fixture();
  try {
    let s = await f.st.read();
    const before = await Promise.all(s.images.map((x: any) => f.st.source(x).then(hash)));
    s = await f.st.edit(s.revision, [s.images[1].id], { excluded: 1 });
    f.db.rows.set('old', { id: 'old', gpu_ids: '0', status: 'stopped', job_type: 'train', queue_position: 500 });
    s = await generate(f);
    const link = current(s);
    assert.equal(link.scope.length, 2);
    assert.equal(link.automatic.phase, 'active');
    assert.equal(f.db.created, 1);
    assert.equal(f.db.queues.get('queue-0').is_running, true);
    assert.equal(f.db.rows.get('old').status, 'stopped');
    assert.deepEqual(await Promise.all(s.images.map((x: any) => f.st.source(x).then(hash))), before);
    assert.equal(link.scope[0].category, 'unclassified');
    await assert.rejects(fs.stat(path.join(link.folder, '000001.txt')), { code: 'ENOENT' });
  } finally {
    await f.cleanup();
  }
});

for (const status of ['queued', 'running', 'stopping'])
  test('paused queue with other ' + status + ' work is not activated and recovers explicitly', async () => {
    const f = await fixture();
    try {
      f.db.queues.set('q', { id: 'q', gpu_ids: '0', is_running: false });
      f.db.rows.set('other', { id: 'other', gpu_ids: '0', status, queue_position: 25 });
      let s = await generate(f);
      const id = current(s).automatic.id;
      assert.equal(current(s).automatic.phase, 'blocked');
      assert.equal(f.db.queueWrites, 0);
      assert.equal(f.db.rows.get('other').status, status);
      assert.equal(f.db.rows.get(current(s).jobId).status, 'stopped');
      const revision = s.revision;
      s = await reconcileCaption(f.st, id, f.db, host);
      assert.equal(s.revision, revision);
      f.db.queues.get('q').is_running = true;
      s = await reconcileCaption(f.st, id, f.db, host, [], true);
      assert.equal(current(s).automatic.phase, 'active');
      assert.equal(f.db.queueWrites, 0);
      assert.equal(f.db.created, 1);
    } finally {
      await f.cleanup();
    }
  });

test('active native queue remains unchanged with other queued training and another GPU', async () => {
  const f = await fixture();
  try {
    f.db.queues.set('q', { id: 'q', gpu_ids: '0', is_running: true });
    f.db.queues.set('q1', { id: 'q1', gpu_ids: '1', is_running: false });
    f.db.rows.set('old', { id: 'old', gpu_ids: '0', status: 'queued', queue_position: 99 });
    const s = await generate(f);
    assert.equal(current(s).automatic.phase, 'active');
    assert.equal(f.db.queueWrites, 0);
    assert.equal(f.db.rows.get('old').status, 'queued');
    assert.equal(f.db.queues.get('q1').is_running, false);
  } finally {
    await f.cleanup();
  }
});

for (const blockedHost of [
  { ...host, platform: 'darwin' },
  { ...host, preview: true },
  { ...host, gpu0: false },
])
  test(
    'host guard rejects before any intent, file staging, or native job: ' + JSON.stringify(blockedHost),
    async () => {
      const f = await fixture();
      try {
        const s = await f.st.read();
        await assert.rejects(
          generateCaption(
            f.st,
            s.revision,
            { requestId: randomUUID(), preferences: prefs },
            f.db,
            '/synthetic/aitk.db',
            blockedHost,
          ),
          /non disponibile/,
        );
        assert.equal(f.db.created, 0);
        assert.equal((await f.st.read()).jobs.length, 0);
        await assert.rejects(fs.stat(path.join(f.st.folder, 'caption')), { code: 'ENOENT' });
      } finally {
        await f.cleanup();
      }
    },
  );

test('request replay, new request while pending, polling and completed replay do not duplicate/reset jobs', async () => {
  const f = await fixture();
  try {
    const id = randomUUID();
    let s = await generate(f, id);
    const row = f.db.rows.get(current(s).jobId);
    const position = row.queue_position;
    s = await generate(f, id);
    s = await generate(f);
    assert.equal(f.db.created, 1);
    assert.equal(row.queue_position, position);
    const rev = s.revision;
    s = await reconcileCaption(f.st, id, f.db, host);
    assert.equal(s.revision, rev);
    row.status = 'running';
    row.step = 1;
    s = await reconcileCaption(f.st, id, f.db, host);
    assert.equal(row.step, 1);
    assert.equal(f.db.updates, 1);
    const link = await outputs(f);
    s = await reconcileCaption(f.st, link.automatic.id, f.db, host);
    const appliedRev = s.revision;
    s = await reconcileCaption(f.st, id, f.db, host, [], true);
    assert.equal(s.revision, appliedRev);
    assert.equal(f.db.created, 1);
  } finally {
    await f.cleanup();
  }
});

test('unknown create/committed enqueue responses reconcile the exact persisted identity', async () => {
  const f = await fixture();
  try {
    f.db.failCreate = true;
    f.db.failCommit = true;
    let s = await generate(f);
    const id = current(s).automatic.id;
    assert.equal(current(s).automatic.phase, 'unknown');
    assert.equal(f.db.created, 1);
    assert.equal(f.db.rows.get(current(s).jobId).status, 'queued');
    const pos = f.db.rows.get(current(s).jobId).queue_position;
    s = await reconcileCaption(f.st, id, f.db, host, [], true);
    assert.equal(current(s).automatic.phase, 'active');
    assert.equal(f.db.created, 1);
    assert.equal(f.db.rows.get(current(s).jobId).queue_position, pos);
  } finally {
    await f.cleanup();
  }
});

test('full exact completion applies once after store reload with original TXT/bytes unchanged', async () => {
  const f = await fixture();
  try {
    let s = await generate(f);
    const originals = await Promise.all(s.images.map((x: any) => f.st.source(x).then(hash)));
    const link = await outputs(f);
    const reloaded = await new StudioStore(
      path.join(f.root, 'data'),
      path.join(f.root, 'datasets'),
      'Subject A',
    ).init();
    s = await reconcileCaption(reloaded, link.automatic.id, f.db, host);
    assert.equal(current(s).automatic.phase, 'applied');
    assert.ok(s.images.every((x: any) => x.caption.startsWith('Generated ')));
    assert.deepEqual(await Promise.all(s.images.map((x: any) => f.st.source(x).then(hash))), originals);
    assert.equal(await fs.readFile(path.join(f.st.datasetRoot, '1.txt'), 'utf8'), 'Original 1');
    const rev = s.revision;
    s = await reconcileCaption(reloaded, link.automatic.id, f.db, host);
    assert.equal(s.revision, rev);
    s = await generate(f);
    assert.equal(f.db.created, 2);
  } finally {
    await f.cleanup();
  }
});

for (const kind of ['partial', 'missing', 'empty', 'large'])
  test('incomplete ' + kind + ' completion is visible, atomic and not retried by polling', async () => {
    const f = await fixture();
    try {
      await generate(f);
      const link = await outputs(f);
      if (kind === 'partial') f.db.rows.get(link.jobId).step = 2;
      if (kind === 'missing') await fs.unlink(path.join(link.folder, '000003.txt'));
      if (kind === 'empty') await fs.writeFile(path.join(link.folder, '000003.txt'), '  ');
      if (kind === 'large') await fs.writeFile(path.join(link.folder, '000003.txt'), 'a'.repeat(64001));
      let s = await reconcileCaption(f.st, link.automatic.id, f.db, host);
      assert.equal(current(s).automatic.phase, 'failed');
      assert.ok(s.images.every((x: any) => x.caption.startsWith('Original')));
      assert.equal(needsCaptionReconcile(current(s)), false);
      const rev = s.revision;
      s = await reconcileCaption(f.st, link.automatic.id, f.db, host);
      assert.equal(s.revision, rev);
      await outputs(f);
      s = await reconcileCaption(f.st, link.automatic.id, f.db, host, [], true);
      assert.equal(current(s).automatic.phase, 'applied');
    } finally {
      await f.cleanup();
    }
  });

for (const mode of ['manual', 'undo', 'client-draft', 'durable-draft'])
  test('later ' + mode + ' prevents all generated overwrites', async () => {
    const f = await fixture();
    try {
      let s = await generate(f);
      const link = await outputs(f),
        id = s.images[0].id;
      let blocked: string[] = [];
      if (mode === 'client-draft') blocked = [id];
      else if (mode === 'durable-draft') {
        s = await f.st.captionDraft(
          s.revision,
          id,
          { caption: 'Unsaved draft', baseRevision: s.images[0].revision },
          0,
        );
        assert.equal((await f.st.read()).images[0].captionDraft!.caption, 'Unsaved draft');
        await assert.rejects(f.st.captionDraft(s.revision, id, null, 0), /altrove/);
      } else {
        s = await f.st.edit(s.revision, [id], { caption: 'Manual edit' });
        if (mode === 'undo') s = await f.st.edit(s.revision, [id], { caption: link.scope[0].caption });
      }
      s = await reconcileCaption(f.st, link.automatic.id, f.db, host, blocked);
      assert.equal(current(s).automatic.phase, 'conflict');
      assert.ok(s.images.every((x: any) => !x.caption.startsWith('Generated')));
      s = await dismissCaption(f.st, s.revision, link.automatic.id);
      assert.equal(current(s).automatic.phase, 'dismissed');
    } finally {
      await f.cleanup();
    }
  });

test('saving exact durable draft clears it; edit during save retains the newer draft and CAS rejects stale edit', async () => {
  const f = await fixture();
  try {
    let s = await f.st.read();
    const img = s.images[0];
    s = await f.st.captionDraft(s.revision, img.id, { caption: 'first', baseRevision: img.revision }, 0);
    s = await f.st.captionDraft(s.revision, img.id, { caption: 'second', baseRevision: img.revision }, 1);
    s = await f.st.edit(s.revision, [img.id], { caption: 'first', baseRevision: img.revision });
    assert.equal(s.images[0].captionDraft!.caption, 'second');
    await assert.rejects(
      f.st.edit(s.revision, [img.id], { caption: 'stale', baseRevision: img.revision }),
      /draft retained/,
    );
    s = await f.st.captionDraft(s.revision, img.id, { caption: 'second', baseRevision: s.images[0].revision }, 2);
    s = await f.st.edit(s.revision, [img.id], { caption: 'second', baseRevision: s.images[0].revision });
    assert.equal(s.images[0].captionDraft, undefined);
  } finally {
    await f.cleanup();
  }
});

for (const field of ['gpu_ids', 'job_ref', 'job_config'])
  test('changed native ' + field + ' refuses queue/apply identity', async () => {
    const f = await fixture();
    try {
      let s = await generate(f);
      const link = current(s),
        row = f.db.rows.get(link.jobId);
      row[field] = field === 'job_config' ? '{}' : 'different';
      const writes = f.db.updates;
      s = await reconcileCaption(f.st, link.automatic.id, f.db, host, [], true);
      assert.ok(['blocked', 'conflict'].includes(current(s).automatic.phase));
      assert.equal(f.db.updates, writes);
    } finally {
      await f.cleanup();
    }
  });

test('native stop after queue is terminal and never auto restarts; legacy link cannot be reconciled', async () => {
  const f = await fixture();
  try {
    let s = await generate(f);
    const link = current(s);
    f.db.rows.get(link.jobId).status = 'stopped';
    s = await reconcileCaption(f.st, link.automatic.id, f.db, host);
    assert.equal(current(s).automatic.phase, 'failed');
    s = await reconcileCaption(f.st, link.automatic.id, f.db, host, [], true);
    assert.equal(current(s).automatic.phase, 'failed');
    assert.equal(f.db.updates, 1);
    await assert.rejects(reconcileCaption(f.st, 'old-legacy-draft', f.db, host), /non trovata/);
  } finally {
    await f.cleanup();
  }
});

for (const tamper of ['original', 'staged', 'symlink'])
  test('changed ' + tamper + ' refuses generated readback', async () => {
    const f = await fixture();
    try {
      await generate(f);
      const link = await outputs(f);
      if (tamper === 'original')
        await fs.writeFile(
          path.join(f.st.datasetRoot, link.scope[0].filename),
          await sharp({ create: { width: 32, height: 48, channels: 3, background: '#ffff00' } })
            .png()
            .toBuffer(),
        );
      if (tamper === 'staged') await fs.writeFile(path.join(link.folder, '000001.png'), 'changed input');
      if (tamper === 'symlink') {
        await fs.unlink(path.join(link.folder, '000001.txt'));
        await fs.symlink(path.join(f.st.datasetRoot, '1.txt'), path.join(link.folder, '000001.txt'));
      }
      const s = await reconcileCaption(f.st, link.automatic.id, f.db, host);
      assert.notEqual(current(s).automatic.phase, 'applied');
      assert.ok(s.images.every((x: any) => !x.caption.startsWith('Generated')));
    } finally {
      await f.cleanup();
    }
  });

test('a persisted request cannot be rebound to new instructions and GET-style read never queues legacy or new work', async () => {
  const f = await fixture();
  try {
    const id = randomUUID();
    let s = await generate(f, id);
    const updates = f.db.updates,
      creates = f.db.created;
    await assert.rejects(
      generateCaption(
        f.st,
        s.revision,
        { requestId: id, preferences: { ...prefs, instructions: 'different' } },
        f.db,
        '/synthetic/aitk.db',
        host,
      ),
      /altri parametri/,
    );
    for (let i = 0; i < 3; i++) s = await f.st.read();
    assert.equal(f.db.updates, updates);
    assert.equal(f.db.created, creates);
    assert.equal(current(s).automatic.id, id);
  } finally {
    await f.cleanup();
  }
});
