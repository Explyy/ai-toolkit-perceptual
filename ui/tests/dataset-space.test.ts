import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import sharp from 'sharp';
import { PrismaClient } from '@prisma/client';
import { StudioStore, hash, JobLink } from '../src/datasetStudio/store';
import { stableJSON } from '../src/datasetStudio/domain';
import { MODEL_ANALYZER } from '../src/datasetStudio/analysisPolicy';
import { stageBudget } from '../src/datasetStudio/space';
import { reclaimPlan, reclaimInputs } from '../src/datasetStudio/reclamation';
async function fixture() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'studio-space-')),
    datasets = path.join(root, 'datasets');
  await fs.mkdir(path.join(datasets, 'own'), { recursive: true });
  const pixels = await sharp({ create: { width: 128, height: 128, channels: 3, background: '#aabbcc' } })
    .png()
    .toBuffer();
  await fs.writeFile(path.join(datasets, 'own', 'a.png'), pixels);
  await fs.writeFile(path.join(datasets, 'own', 'a.txt'), 'original');
  const st = await new StudioStore(path.join(root, 'data'), datasets, 'own').init(),
    s = await st.read(),
    img = s.images[0];
  const sqlite = path.join(root, 'native.db'),
    schema = path.join(root, 'schema.prisma');
  await fs.writeFile(sqlite, '');
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
  const db = new PrismaClient({ datasourceUrl: 'file:' + sqlite }),
    folder = path.join(st.folder, 'analysis', 'fixture');
  await fs.mkdir(path.join(folder, 'result-000000'), { recursive: true });
  const config = 'c'.repeat(64),
    depth = await sharp({ create: { width: 8, height: 8, channels: 3, background: '#555555' } })
      .png()
      .toBuffer();
  const signals = {
    version: MODEL_ANALYZER,
    sha: img.sha,
    config,
    faces: [],
    persons: [],
    depth: { grid: Array(64).fill(0.5), map: 'depth.png', sha: hash(depth), relative: true },
    runtime: { device: 'fixture, not inference' },
  };
  await fs.writeFile(path.join(folder, 'input-000000.png'), pixels);
  await fs.writeFile(path.join(folder, 'result-000000/depth.png'), depth);
  await fs.writeFile(path.join(folder, 'result-000000/result.json'), JSON.stringify(signals));
  const name = 'studio-analysis-space',
    cfg = {
      job: 'extension',
      config: {
        name,
        process: [
          { type: 'dataset_studio_analysis', request: path.join(folder, 'request.json'), sqlite_db_path: sqlite },
        ],
      },
    };
  await fs.writeFile(
    path.join(folder, 'request.json'),
    stableJSON({
      name,
      dataset: st.datasetRoot,
      config,
      items: [{ id: img.id, sha: img.sha, input: 'input-000000.png', output: 'result-000000' }],
    }),
  );
  const row = await db.job.create({
    data: {
      name,
      job_type: 'analysis',
      job_ref: st.datasetRoot,
      gpu_ids: '0',
      job_config: JSON.stringify(cfg),
      status: 'completed',
      step: 1,
      total_steps: 1,
    },
  });
  const link: JobLink = {
    name,
    kind: 'analysis',
    state: 'linked',
    jobId: row.id,
    folder,
    config: cfg,
    scope: [structuredClone(img)],
    automatic: { id: 'owned', digest: 'fixture', gpu: '0', phase: 'applied', createdAt: '' },
  };
  s.images[0].analysis = { model: signals } as any;
  s.jobs.push(link);
  const second = folder + '-duplicate';
  await fs.cp(folder, second, { recursive: true });
  const name2 = name + '-duplicate',
    cfg2 = {
      job: 'extension',
      config: {
        name: name2,
        process: [
          { type: 'dataset_studio_analysis', request: path.join(second, 'request.json'), sqlite_db_path: sqlite },
        ],
      },
    };
  const req = JSON.parse(await fs.readFile(path.join(second, 'request.json'), 'utf8'));
  req.name = name2;
  await fs.writeFile(path.join(second, 'request.json'), stableJSON(req));
  const row2 = await db.job.create({
    data: {
      name: name2,
      job_type: 'analysis',
      job_ref: st.datasetRoot,
      gpu_ids: '0',
      job_config: JSON.stringify(cfg2),
      status: 'completed',
      step: 1,
      total_steps: 1,
    },
  });
  s.jobs.push({
    ...structuredClone(link),
    name: name2,
    folder: second,
    jobId: row2.id,
    config: cfg2,
    automatic: { ...link.automatic!, id: 'duplicate' },
  });
  await st.save(s);
  return {
    st,
    db,
    link,
    pixels,
    config,
    cleanup: async () => {
      await db.$disconnect();
      await fs.rm(root, { recursive: true, force: true });
    },
  };
}
test('existing verified staging/results are charged zero; only missing bytes/output consume reserve', async () => {
  const f = await fixture();
  try {
    assert.equal((await stageBudget(f.st, f.link, f.config)).needed, 0);
    await fs.unlink(path.join(f.link.folder!, 'input-000000.png'));
    assert.equal((await stageBudget(f.st, f.link, f.config)).missingBytes, f.pixels.length);
    assert.equal((await stageBudget(f.st, f.link, f.config)).outputBytes, 0);
    await fs.unlink(path.join(f.link.folder!, 'result-000000/result.json'));
    assert.equal((await stageBudget(f.st, f.link, f.config)).outputBytes, 1100000);
  } finally {
    await f.cleanup();
  }
});
test('bounded derived-copy dedup preserves original/caption/request/results/native row and refuses stale revisions/active/hardlinked copies', async () => {
  const f = await fixture();
  try {
    const original = await fs.readFile(path.join(f.st.datasetRoot, 'a.png')),
      request = await fs.readFile(path.join(f.link.folder!, 'request.json')),
      result = await fs.readFile(path.join(f.link.folder!, 'result-000000/result.json')),
      row = await f.db.job.findUnique({ where: { id: f.link.jobId } });
    let plan = await reclaimPlan(f.st, f.db, 1);
    assert.equal(plan.files.length, 1);
    const s = await f.st.read();
    await f.st.edit(s.revision, [s.images[0].id], { excluded: 1 });
    await assert.rejects(reclaimInputs(f.st, f.db, plan), /obsoleto/);
    await f.db.job.update({ where: { id: f.link.jobId }, data: { status: 'running' } });
    assert.equal((await reclaimPlan(f.st, f.db, 1)).files.length, 0);
    await f.db.job.update({ where: { id: f.link.jobId }, data: { status: 'completed' } });
    const input = (await reclaimPlan(f.st, f.db, 1)).files[0].file,
      alias = path.join(f.st.folder, 'hardlink');
    await fs.link(input, alias);
    assert.equal((await reclaimPlan(f.st, f.db, 1)).files.length, 0);
    await fs.unlink(alias);
    plan = await reclaimPlan(f.st, f.db, 1);
    const beforeRow = await f.db.job.findUnique({ where: { id: f.link.jobId } });
    const receipt = await reclaimInputs(f.st, f.db, plan);
    assert.equal(receipt.replacedBytes, f.pixels.length);
    assert.equal((await reclaimPlan(f.st, f.db, 1)).files.length, 0);
    assert.deepEqual(await fs.readFile(input), f.pixels);
    assert.ok((await fs.stat(input)).nlink >= 2);
    assert.notEqual((await fs.stat(input)).ino, (await fs.stat(path.join(f.st.datasetRoot, 'a.png'))).ino);
    assert.deepEqual(await fs.readFile(path.join(f.st.datasetRoot, 'a.png')), original);
    assert.equal(await fs.readFile(path.join(f.st.datasetRoot, 'a.txt'), 'utf8'), 'original');
    assert.deepEqual(await fs.readFile(path.join(f.link.folder!, 'request.json')), request);
    assert.deepEqual(await fs.readFile(path.join(f.link.folder!, 'result-000000/result.json')), result);
    assert.deepEqual(await f.db.job.findUnique({ where: { id: f.link.jobId } }), beforeRow);
    assert.equal((await f.st.read()).images[0].excluded, 1);
    assert.equal(JSON.parse(await fs.readFile(receipt.receipt, 'utf8')).phase, 'finished');
  } finally {
    await f.cleanup();
  }
});

test('dedup dry-run does not persist newly discovered source images or change state/outbox/derived paths', async () => {
  const f = await fixture();
  try {
    const stateFile = path.join(f.st.folder, 'state.json'),
      outboxFile = path.join(f.st.folder, 'managed-outbox.json');
    const before = await Promise.all([fs.readFile(stateFile), fs.readFile(outboxFile)]);
    const source = path.join(f.st.datasetRoot, 'new-upload.png');
    await fs.writeFile(source, f.pixels);
    const copied = path.join(f.link.folder! + '-duplicate', 'input-000000.png');
    const stat = await fs.stat(copied);
    const plan = await reclaimPlan(f.st, f.db, 1);
    assert.equal(plan.files.length, 1);
    const after = await Promise.all([fs.readFile(stateFile), fs.readFile(outboxFile)]);
    assert.deepEqual(after, before, 'planning may not scan-save new originals or schedule sync');
    assert.equal((await fs.stat(copied)).ino, stat.ino);
    assert.deepEqual(await fs.readFile(source), f.pixels);
    await assert.rejects(fs.stat(path.join(f.st.folder, 'analysis', 'derived-input-pool')), /ENOENT/);
  } finally {
    await f.cleanup();
  }
});
