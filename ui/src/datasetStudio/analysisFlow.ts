import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import sharp from 'sharp';
import { randomUUID } from 'node:crypto';
import { StudioStore, State, JobLink, atomic, contained, hash } from './store';
import { ensure, stableJSON, select, analyzePixels } from './domain';
import { inferredCategory, validateSignals, MODEL_ANALYZER } from './analysisPolicy';
import { reconcileJob } from './jobs';
import type { CaptionHost } from './captionFlow';
import { stageBudget, requireBudget } from './space';
import { reclaimPlan, reclaimInputs } from './reclamation';
import { TOOLKIT_ROOT } from '@/paths';
export type AnalysisSummary = {
  config: string;
  phase: 'unavailable' | 'pending' | 'active' | 'failed' | 'complete';
  reason?: string;
  count?: number;
  deficits?: Record<string, number>;
  conflicts?: string[];
};
export async function analysisConfiguration() {
  const file = path.join(TOOLKIT_ROOT, 'extensions_built_in', 'dataset_studio_analysis', 'models.json');
  const manifest = JSON.parse(await fs.readFile(file, 'utf8'));
  ensure(manifest.version === MODEL_ANALYZER, 'Unsupported model manifest');
  return { manifest, config: hash(stableJSON(manifest)) };
}
async function modelPrerequisite(manifest: any) {
  ensure(
    process.env.DATASET_STUDIO_ANALYSIS_ENABLED === '1',
    'Analisi non disponibile: serve il cloud dedicato con modelli verificati.',
    403,
  );
  const base = process.env.DATASET_STUDIO_ROOT;
  ensure(base && path.isAbsolute(base), 'Storage cloud di analisi non configurato', 403);
  const root = await contained(base!, path.join(base!, 'cache', 'analysis-models'));
  for (const [key, model] of Object.entries(manifest.models) as [string, any][]) {
    for (const [name, expected] of Object.entries(model.files) as [string, any][]) {
      const file = await contained(root, path.join(root, key, name));
      ensure(
        (await fs.stat(file)).size === expected.size && hash(await fs.readFile(file)) === expected.sha256,
        'Modelli di analisi mancanti o checksum diverso: ' + key,
        403,
      );
    }
  }
}
function summarize(s: State, value: AnalysisSummary) {
  if (stableJSON(s.analysisFlow) !== stableJSON(value)) {
    s.analysisFlow = value;
    s.revision++;
    return true;
  }
  return false;
}
export function applyProposal(s: State, config: string) {
  const pending = s.images.filter(x => !x.discarded && x.analysis?.model?.config !== config);
  if (pending.length) return false;
  const candidates = s.images.map(x => ({ ...x, excluded: (x.reviewRevision ?? 0) > 0 ? x.excluded : 0 }));
  const proposal = select(candidates, s.settings.count, s.settings.testId),
    chosen = new Set(proposal.selected.map(x => x.id));
  let changed = false;
  for (const image of s.images)
    if (!image.discarded && (image.reviewRevision ?? 0) === 0) {
      const excluded = chosen.has(image.id) ? 0 : 1;
      if (image.excluded !== excluded) {
        image.excluded = excluded;
        changed = true;
      }
    }
  const summaryChanged = summarize(s, {
    config,
    phase: 'complete',
    count: proposal.selected.length,
    deficits: proposal.deficits,
    conflicts: proposal.conflicts,
  });
  if (changed && !summaryChanged) s.revision++;
  return summaryChanged || changed;
}
async function unlinkedLegacyCollision(
  st: StudioStore,
  link: JobLink,
  db: any,
  config: string,
  digest: string,
  missing: State['images'],
  sqlite: string,
) {
  // Only an explicit retry may archive a failed intent. Existing linked jobs,
  // unknown identities and immutable request artifacts are never rewritten.
  if (link.jobId || link.state !== 'intent' || link.automatic?.phase !== 'failed') return false;
  try {
    const name = 'studio-analysis-' + digest.slice(0, 24),
      folder = path.join(st.folder, 'analysis', digest);
    if (link.name !== name || link.folder !== folder || link.automatic.digest !== digest) return false;
    const expectedConfig = {
      job: 'extension',
      config: {
        name,
        process: [
          { type: 'dataset_studio_analysis', request: path.join(folder, 'request.json'), sqlite_db_path: sqlite },
        ],
      },
    };
    if (stableJSON(link.config) !== stableJSON(expectedConfig) || link.scope?.length !== missing.length) return false;
    if (
      !link.scope.every(
        (image, i) =>
          image.id === missing[i].id &&
          image.sha === missing[i].sha &&
          image.relative === missing[i].relative &&
          image.project === hash(st.name),
      )
    )
      return false;
    const items = link.scope.map((image, i) => ({
      id: image.id,
      sha: image.sha,
      input: 'input-' + String(i).padStart(6, '0') + path.extname(image.filename).toLowerCase(),
      output: 'result-' + String(i).padStart(6, '0'),
    }));
    const request = await contained(st.folder, path.join(folder, 'request.json'));
    if ((await fs.stat(request)).size > 1500000) return false;
    if (
      stableJSON(JSON.parse(await fs.readFile(request, 'utf8'))) !==
      stableJSON({ name, dataset: st.datasetRoot, config, items })
    )
      return false;
    for (const item of items) {
      const input = await contained(st.folder, path.join(folder, item.input));
      if (hash(await fs.readFile(input)) !== item.sha) return false;
    }
    const foreign = await db.job.findUnique({ where: { name } });
    return (
      !!foreign &&
      foreign.name === name &&
      foreign.job_type === 'analysis' &&
      foreign.gpu_ids === '0' &&
      typeof foreign.job_ref === 'string' &&
      path.isAbsolute(foreign.job_ref) &&
      foreign.job_ref !== st.datasetRoot &&
      (await fs.realpath(foreign.job_ref)) === foreign.job_ref
    );
  } catch {
    return false;
  }
}
// The sole user command boundary. Reads, target drafts and image edits never call this.
export async function executeAnalysisCommand(
  st: StudioStore,
  db: any,
  sqlite: string,
  host: CaptionHost,
  input: { revision: number; count?: number; retry?: boolean },
) {
  if (input.count !== undefined) {
    ensure(Number.isInteger(input.count) && input.count >= 1 && input.count <= 1000, 'Numero immagini non valido');
    const before = await st.read();
    if (before.settings.count !== input.count) {
      await st.mutate(input.revision, current => {
        ensure(
          !current.jobs.some(
            j => j.kind === 'analysis' && ['prepared', 'enqueue-intent', 'active'].includes(j.automatic?.phase ?? ''),
          ),
          'Attendi la fine dell’analisi prima di applicare un nuovo obiettivo',
          409,
        );
        current.settings.count = input.count!;
      });
    }
  }
  return prepareAnalysis(st, db, sqlite, host, input.retry === true);
}
export async function prepareAnalysis(st: StudioStore, db: any, sqlite: string, host: CaptionHost, retry = false) {
  const { manifest, config } = await analysisConfiguration();
  let s = await st.read();
  const missing = s.images
    .filter(x => !x.discarded && x.analysis?.model?.config !== config)
    .sort((a, b) => a.sha.localeCompare(b.sha) || a.id.localeCompare(b.id));
  if (!missing.length)
    return st.locked(async () => {
      const state = await st.scan(await st.raw());
      if (applyProposal(state, config)) await st.save(state);
      return state;
    });
  const digest = hash(stableJSON({ config, items: missing.map(x => ({ id: x.id, sha: x.sha })) }));
  const previous = s.jobs.find(
    x => x.kind === 'analysis' && x.automatic?.digest === digest && x.automatic.phase !== 'dismissed',
  );
  if (previous && !retry) return reconcileAnalysis(st, previous.automatic!.id, db, host);
  try {
    ensure(
      host.platform === 'linux' && host.gpu0 && !host.preview,
      'Analisi non disponibile qui: nessun modello viene eseguito sul Mac o nel preview.',
      403,
    );
    await modelPrerequisite(manifest);
  } catch (e: any) {
    return st.locked(async () => {
      const state = await st.scan(await st.raw());
      let changed = false;
      for (const x of state.images)
        if (x.analysis?.model?.config !== config) {
          const status = { config, phase: 'unavailable' as const, reason: 'Modelli o host dedicato non disponibili' };
          if (stableJSON(x.analysisStatus) !== stableJSON(status)) {
            x.analysisStatus = status;
            changed = true;
          }
        }
      if (
        summarize(state, {
          config,
          phase: 'unavailable',
          reason: e.code === 'ENOENT' ? 'Modelli verificati non ancora disponibili sul cloud.' : e.message,
        })
      )
        changed = true;
      if (changed) {
        state.revision++;
        await st.save(state);
      }
      return state;
    });
  }
  let link: JobLink;
  s = await st.locked(async () => {
    const state = await st.scan(await st.raw());
    const old = state.jobs.find(
      x => x.kind === 'analysis' && x.automatic?.digest === digest && x.automatic.phase !== 'dismissed',
    );
    if (old) {
      if (!retry || !(await unlinkedLegacyCollision(st, old, db, config, digest, missing, sqlite))) {
        link = old;
        return state;
      }
      old.automatic!.phase = 'dismissed';
      old.automatic!.reason = 'Previous intent collided with another dataset; immutable evidence preserved.';
    }
    const id = randomUUID(),
      folder = old ? path.join(st.folder, 'analysis', digest, 'retry-' + id) : path.join(st.folder, 'analysis', digest),
      name = 'studio-analysis-' + hash(st.datasetRoot).slice(0, 16) + '-' + digest.slice(0, 24);
    const cfg = {
      job: 'extension',
      config: {
        name,
        process: [
          { type: 'dataset_studio_analysis', request: path.join(folder, 'request.json'), sqlite_db_path: sqlite },
        ],
      },
    };
    link = {
      kind: 'analysis',
      name,
      state: 'intent',
      folder,
      scope: structuredClone(missing),
      config: cfg,
      automatic: { id, digest, gpu: '0', phase: 'prepared', createdAt: new Date().toISOString() },
    };
    state.jobs.push(link);
    for (const x of state.images) if (missing.some(m => m.id === x.id)) x.analysisStatus = { config, phase: 'pending' };
    summarize(state, { config, phase: 'pending' });
    state.revision++;
    await st.save(state);
    return state;
  });
  return reconcileAnalysis(st, link!.automatic!.id, db, host, retry);
}
function identity(row: any, link: JobLink, st: StudioStore) {
  ensure(
    row &&
      row.id === link.jobId &&
      row.name === link.name &&
      row.job_type === 'analysis' &&
      row.job_ref === st.datasetRoot &&
      row.gpu_ids === '0' &&
      stableJSON(JSON.parse(row.job_config)) === stableJSON(link.config),
    'Analysis job identity changed; no queue mutation',
    409,
  );
}
async function stage(st: StudioStore, link: JobLink, config: string, db: any) {
  let budget = await stageBudget(st, link, config);
  if (budget.deficit > 0) {
    const plan = await reclaimPlan(st, db, budget.deficit);
    if (plan.plannedBytes >= plan.quota && plan.files.length) {
      await reclaimInputs(st, db, plan);
      budget = await stageBudget(st, link, config);
    }
  }
  requireBudget(budget);
  await contained(st.folder, link.folder!, true);
  await fs.mkdir(link.folder!, { recursive: true });
  const items = [];
  for (const [i, image] of link.scope!.entries()) {
    const input = 'input-' + String(i).padStart(6, '0') + path.extname(image.filename).toLowerCase(),
      output = 'result-' + String(i).padStart(6, '0');
    const file = await contained(st.folder, path.join(link.folder!, input), true),
      bytes = await st.source(image);
    try {
      ensure(hash(await fs.readFile(file)) === image.sha, 'Staged analysis bytes changed', 409);
    } catch (e: any) {
      if (e.code !== 'ENOENT') throw e;
      await atomic(file, bytes);
    }
    items.push({ id: image.id, sha: image.sha, input, output });
  }
  const value = { name: link.name, dataset: st.datasetRoot, config, items };
  const request = await contained(st.folder, path.join(link.folder!, 'request.json'), true);
  try {
    ensure(
      stableJSON(JSON.parse(await fs.readFile(request, 'utf8'))) === stableJSON(value),
      'Analysis request changed',
      409,
    );
  } catch (e: any) {
    if (e.code !== 'ENOENT') throw e;
    await atomic(request, stableJSON(value));
  }
}
export type ProcessProbe = {
  platform: string;
  host: string;
  boot: () => Promise<string>;
  namespace: () => Promise<string>;
  stat: (pid: number) => Promise<string>;
  now?: () => number;
};
const processProbe: ProcessProbe = {
  platform: os.platform(),
  host: os.hostname(),
  boot: async () => (await fs.readFile('/proc/sys/kernel/random/boot_id', 'utf8')).trim(),
  namespace: () => fs.readlink('/proc/self/ns/pid'),
  stat: pid => fs.readFile('/proc/' + pid + '/stat', 'utf8'),
};
export async function analysisProcessState(
  st: StudioStore,
  link: JobLink,
  row: any,
  probe: ProcessProbe = processProbe,
): Promise<'alive' | 'dead' | 'unknown' | 'starting'> {
  if (probe.platform !== 'linux') return 'unknown';
  try {
    const file = await contained(st.folder, path.join(link.folder!, 'runtime.json'));
    const receipt = JSON.parse(await fs.readFile(file, 'utf8'));
    if (!row.pid || !Number.isInteger(row.pid)) return 'unknown';
    if (
      receipt.pid !== row.pid ||
      receipt.host !== probe.host ||
      receipt.boot !== (await probe.boot()) ||
      receipt.namespace !== (await probe.namespace()) ||
      receipt.name !== link.name
    )
      return 'unknown';
    try {
      const current = await probe.stat(row.pid),
        fields = current
          .slice(current.lastIndexOf(')') + 1)
          .trim()
          .split(/\s+/);
      if (fields[19] !== receipt.start || fields[0] === 'Z') return 'dead';
      return 'alive';
    } catch (e: any) {
      return e.code === 'ENOENT' ? 'dead' : 'unknown';
    }
  } catch (e: any) {
    // The native launcher marks its exactly owned row running before Python
    // can write runtime.json. This bounded wait makes no liveness/death claim.
    const launched = new Date(row.updated_at).getTime(),
      now = probe.now?.() ?? Date.now();
    if (
      e.code === 'ENOENT' &&
      row.status === 'running' &&
      Number.isFinite(launched) &&
      launched <= now &&
      now - launched <= 120000
    )
      return 'starting';
    return 'unknown';
  }
}
export async function provenAnalysisDeath(
  st: StudioStore,
  link: JobLink,
  row: any,
  probe: ProcessProbe = processProbe,
) {
  return (await analysisProcessState(st, link, row, probe)) === 'dead';
}
export async function applyAnalysisResults(st: StudioStore, link: JobLink, config: string) {
  const results = new Map<string, any>();
  for (const [i, image] of link.scope!.entries()) {
    const folder = path.join(link.folder!, 'result-' + String(i).padStart(6, '0'));
    const file = await contained(st.folder, path.join(folder, 'result.json'), true);
    try {
      const stat = await fs.stat(file);
      ensure(stat.size <= 1000000, 'Analysis result too large');
      const model = validateSignals(JSON.parse(await fs.readFile(file, 'utf8')), image.sha, config);
      const depth = await contained(st.folder, path.join(folder, 'depth.png'));
      ensure(hash(await fs.readFile(depth)) === model.depth.sha, 'Depth map checksum changed');
      const pixels = await sharp(await st.source(image))
        .rotate()
        .resize(64, 64, { fit: 'fill' })
        .ensureAlpha()
        .raw()
        .toBuffer();
      results.set(image.id, {
        ...analyzePixels(new Uint8ClampedArray(pixels), 64, 64, image.width, image.height),
        model,
      });
    } catch (e: any) {
      if (e.code !== 'ENOENT') throw e;
    }
  }
  return st.locked(async () => {
    const state = await st.scan(await st.raw());
    let changed = false;
    for (const image of state.images) {
      const result = results.get(image.id),
        original = link.scope!.find(x => x.id === image.id);
      if (!result || image.sha !== original?.sha || image.analysis?.model?.config === config) continue;
      image.analysis = result;
      image.pose = result.model.persons;
      image.analysisStatus = { config, phase: 'complete' };
      if (image.categorySource !== 'manual') {
        image.category = inferredCategory(result.model).category;
        image.categorySource = 'automatic';
      }
      changed = true;
    }
    if (applyProposal(state, config)) changed = true;
    if (changed) {
      state.revision++;
      await st.save(state);
    }
    return state;
  });
}
export async function reconcileAnalysis(
  st: StudioStore,
  id: string,
  db: any,
  host: CaptionHost,
  retry = false,
  probe: ProcessProbe = processProbe,
) {
  let state = await st.read(),
    link = state.jobs.find(x => x.kind === 'analysis' && x.automatic?.id === id);
  ensure(link?.automatic, 'Analysis request missing', 404);
  const { config } = await analysisConfiguration();
  if (['applied', 'dismissed'].includes(link.automatic.phase)) return state;
  if (['failed', 'unknown'].includes(link.automatic.phase) && !retry) return state;
  const setPhase = async (phase: NonNullable<JobLink['automatic']>['phase'], reason?: string) =>
    st.locked(async () => {
      const s = await st.scan(await st.raw()),
        current = s.jobs.find(x => x.automatic?.id === id)!;
      current.automatic!.phase = phase;
      current.automatic!.reason = reason;
      if (phase === 'active') current.automatic!.queued = true;
      if (phase !== 'applied')
        summarize(s, {
          config,
          phase: phase === 'active' ? 'active' : phase === 'prepared' ? 'pending' : 'failed',
          reason,
        });
      s.revision++;
      await st.save(s);
      return s;
    });
  try {
    if (!link.jobId) {
      await stage(st, link, config, db);
      const row = await reconcileJob(db, link, '0', st.datasetRoot);
      state = await st.locked(async () => {
        const s = await st.scan(await st.raw()),
          current = s.jobs.find(x => x.automatic?.id === id)!;
        current.jobId = row.id;
        current.state = 'linked';
        s.revision++;
        await st.save(s);
        return s;
      });
      link = state.jobs.find(x => x.automatic?.id === id)!;
    }
    let recovered = false;
    let row = await db.job.findUnique({ where: { id: link.jobId } });
    identity(row, link, st);
    state = await applyAnalysisResults(st, link, config);
    if (row.status === 'completed') {
      ensure(
        row.step === link.scope!.length && row.total_steps === link.scope!.length,
        'Analysis did not complete its full native scope',
        409,
      );
      ensure(
        link.scope!.every(x => state.images.find(y => y.id === x.id)?.analysis?.model?.config === config),
        'Incomplete analysis results; no completed claim',
        409,
      );
      return setPhase('applied');
    }
    if (['running', 'stopping'].includes(row.status)) {
      const processState = await analysisProcessState(st, link, row, probe);
      if (processState === 'alive') return setPhase('active');
      if (processState === 'starting')
        return setPhase('active', 'Avvio del processo di analisi: attendiamo la conferma, senza riavviare il lavoro.');
      if (processState === 'unknown')
        return setPhase(
          'blocked',
          'Proprietà del processo non verificabile in questo host/namespace: risultati conservati, nessuna ripresa automatica.',
        );
      ensure(!row.stop && !row.return_to_queue, 'Analysis explicitly stopped; completed cache retained', 409);
      const changed = await db.job.updateMany({
        where: { id: row.id, status: row.status, pid: row.pid },
        data: { status: 'stopped', pid: null, info: 'Owned analysis process exited; resuming missing cache' },
      });
      ensure(changed.count === 1, 'Analysis process state changed', 409);
      recovered = true;
      row = await db.job.findUnique({ where: { id: row.id } });
    }
    if (['error', 'failed'].includes(row.status)) {
      if (!retry) return setPhase('failed', 'Analisi non riuscita. Originali e risultati già completati conservati.');
      await db.job.updateMany({
        where: { id: row.id, status: row.status },
        data: { status: 'stopped', pid: null, stop: false },
      });
      row = await db.job.findUnique({ where: { id: row.id } });
    }
    if (row.status === 'stopped' && retry) {
      await db.job.updateMany({
        where: { id: row.id, status: 'stopped' },
        data: { stop: false, return_to_queue: false },
      });
      row = await db.job.findUnique({ where: { id: row.id } });
    }
    if (row.status === 'stopped' && link.automatic?.queued && !recovered && !retry)
      return setPhase('failed', 'Analisi fermata nei controlli nativi; nessun riavvio automatico.');
    ensure(host.platform === 'linux' && host.gpu0 && !host.preview, 'Dedicated GPU unavailable', 403);
    await db.$transaction(async (tx: any) => {
      const fresh = await tx.job.findUnique({ where: { id: row.id } });
      identity(fresh, link!, st);
      if (['running', 'stopping', 'completed'].includes(fresh.status)) return;
      const queue = await tx.queue.findUnique({ where: { gpu_ids: '0' } });
      if (!queue?.is_running)
        ensure(
          !(await tx.job.findFirst({
            where: { id: { not: row.id }, gpu_ids: '0', status: { in: ['queued', 'running', 'stopping'] } },
          })),
          'Coda in pausa con altri lavori: riprendila nei controlli nativi.',
          409,
        );
      ensure(['queued', 'stopped'].includes(fresh.status), 'Analysis is not queueable', 409);
      if (fresh.status === 'stopped') {
        ensure(!fresh.stop, 'Analysis was stopped explicitly; retry required', 409);
        const maximum = await tx.job.aggregate({ _max: { queue_position: true } });
        await tx.job.updateMany({
          where: { id: row.id, status: 'stopped' },
          data: {
            status: 'queued',
            queue_position: (maximum._max.queue_position ?? 0) + 1000,
            stop: false,
            return_to_queue: false,
          },
        });
      }
      if (!queue) await tx.queue.create({ data: { gpu_ids: '0', is_running: true } });
      else if (!queue.is_running) await tx.queue.update({ where: { id: queue.id }, data: { is_running: true } });
    });
    return setPhase('active');
  } catch (e: any) {
    return setPhase('failed', e.message ?? 'Analisi non confermata; nessun nuovo lavoro duplicato.');
  }
}

// Background evidence reconciliation follows only already persisted owned intents.
// Including blocked analysis links permits late runtime receipts/results to clear
// startup uncertainty after the browser closes; it never requests an explicit retry.
export async function reconcilePendingAnalysis(st: StudioStore, db: any, host: CaptionHost, probe?: ProcessProbe) {
  let state = await st.raw();
  for (const link of state.jobs) {
    if (
      link.kind !== 'analysis' ||
      !link.automatic ||
      ['applied', 'dismissed', 'failed', 'unknown', 'conflict'].includes(link.automatic.phase)
    )
      continue;
    state = await reconcileAnalysis(st, link.automatic.id, db, host, false, probe);
  }
  return state;
}
