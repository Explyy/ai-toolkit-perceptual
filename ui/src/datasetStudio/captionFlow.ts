import { stageBudget, requireBudget } from './space';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { StudioStore, JobLink, State, atomic, contained, hash } from './store';
import { captionConfig, reconcileJob } from './jobs';
import { captionModels, preferences } from './captionModels';
import { ensure, stableJSON, Problem } from './domain';
const exec = promisify(execFile);
export type CaptionHost = { platform: string; preview: boolean; gpu0: boolean };
export async function captionHost(): Promise<CaptionHost> {
  const platform = os.platform(),
    preview = process.env.DATASET_STUDIO_PREVIEW === '1';
  if (preview || platform !== 'linux') return { platform, preview, gpu0: false };
  try {
    const { stdout } = await exec('nvidia-smi', ['--query-gpu=index,uuid', '--format=csv,noheader'], {
      timeout: 5000,
      maxBuffer: 16000,
    });
    return { platform, preview, gpu0: stdout.split('\n').some(row => /^0\s*,\s*GPU-[a-f0-9-]+\s*$/i.test(row)) };
  } catch {
    return { platform, preview, gpu0: false };
  }
}
export function requireCaptionHost(host: CaptionHost) {
  ensure(
    !host.preview && host.platform === 'linux' && host.gpu0,
    'Generazione non disponibile qui: serve AI Toolkit su un host Linux con GPU NVIDIA. Nel preview e sul Mac non avviamo modelli.',
    403,
  );
}
const terminal = new Set(['applied', 'dismissed', 'conflict', 'failed', 'blocked', 'unknown']);
export function needsCaptionReconcile(link: JobLink) {
  return !!link.automatic && !terminal.has(link.automatic.phase);
}
function identity(row: any, link: JobLink, dataset: string) {
  ensure(
    row &&
      row.id === link.jobId &&
      row.name === link.name &&
      row.job_type === 'caption' &&
      row.job_ref === dataset &&
      row.gpu_ids === link.automatic!.gpu &&
      stableJSON(JSON.parse(row.job_config)) === stableJSON(link.config),
    'Identità del lavoro caption cambiata; nessun avvio o overwrite',
    409,
  );
}
async function phase(
  st: StudioStore,
  id: string,
  value: NonNullable<JobLink['automatic']>['phase'],
  reason?: string,
  extra: any = {},
) {
  return st.locked(async () => {
    const s = await st.scan(await st.raw()),
      link = s.jobs.find(x => x.automatic?.id === id);
    ensure(link?.automatic, 'Generazione non trovata', 404);
    if (['applied', 'dismissed'].includes(link.automatic.phase)) return s;
    const next = { ...link.automatic, phase: value, ...extra };
    if (reason) next.reason = reason;
    else delete next.reason;
    if (stableJSON(next) === stableJSON(link.automatic)) return s;
    link.automatic = next;
    s.revision++;
    await st.save(s);
    return s;
  });
}
async function linkJob(st: StudioStore, id: string, job: any) {
  return st.locked(async () => {
    const s = await st.scan(await st.raw()),
      link = s.jobs.find(x => x.automatic?.id === id)!;
    if (link.state === 'linked' && link.jobId === job.id) return s;
    link.state = 'linked';
    link.jobId = job.id;
    s.revision++;
    await st.save(s);
    return s;
  });
}
async function stage(st: StudioStore, link: JobLink) {
  requireBudget(await stageBudget(st, link));
  await contained(st.folder, link.folder!, true);
  await fs.mkdir(link.folder!, { recursive: true });
  for (const [i, img] of link.scope!.entries()) {
    const bytes = await st.source(img),
      file = await contained(
        st.folder,
        path.join(link.folder!, String(i + 1).padStart(6, '0') + path.extname(img.filename).toLowerCase()),
        true,
      );
    try {
      const current = await fs.readFile(file);
      ensure(hash(current) === img.sha, 'Copia caption cambiata; originali preservati', 409);
    } catch (e: any) {
      if (e.code !== 'ENOENT') throw e;
      await atomic(file, bytes);
    }
  }
}
export async function enqueueCaption(db: any, link: JobLink, dataset: string, host: CaptionHost) {
  requireCaptionHost(host);
  return db.$transaction(async (tx: any) => {
    const row = await tx.job.findUnique({ where: { id: link.jobId } });
    identity(row, link, dataset);
    const queue = await tx.queue.findUnique({ where: { gpu_ids: link.automatic!.gpu } });
    if (['running', 'stopping', 'completed', 'error', 'failed'].includes(row.status)) return row;
    if (row.status === 'stopped')
      ensure(!link.automatic!.queued, 'Il lavoro è stato fermato; non viene riavviato automaticamente', 409);
    else ensure(row.status === 'queued', 'Stato nativo non avviabile', 409);
    if (!queue?.is_running) {
      const other = await tx.job.findFirst({
        where: { gpu_ids: row.gpu_ids, id: { not: row.id }, status: { in: ['queued', 'running', 'stopping'] } },
      });
      ensure(
        !other,
        'La coda è in pausa e contiene altri lavori: non la avviamo. Riprendi la coda dai controlli nativi, poi Riconcilia.',
        409,
      );
    }
    if (row.status === 'stopped') {
      const maximum = await tx.job.aggregate({ _max: { queue_position: true } });
      const changed = await tx.job.updateMany({
        where: { id: row.id, status: 'stopped' },
        data: {
          status: 'queued',
          queue_position: (maximum._max.queue_position ?? 0) + 1000,
          stop: false,
          return_to_queue: false,
          info: 'Caption selezionate in coda',
        },
      });
      ensure(changed.count === 1, 'Stato coda cambiato: riconcilia prima di riprovare', 409);
    }
    if (!queue) await tx.queue.create({ data: { gpu_ids: row.gpu_ids, is_running: true } });
    else if (!queue.is_running) await tx.queue.update({ where: { id: queue.id }, data: { is_running: true } });
    return tx.job.findUnique({ where: { id: row.id } });
  });
}
export async function generateCaption(
  st: StudioStore,
  rev: number,
  input: any,
  db: any,
  sqlite: string,
  host: CaptionHost,
) {
  const pref = preferences(input.preferences);
  requireCaptionHost(host);
  ensure(
    typeof input.requestId === 'string' &&
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(input.requestId),
    'Richiesta caption non valida',
  );
  let link: JobLink | undefined;
  let s = await st.mutate(rev, s => {
    const model = captionModels.find(x => x.key === pref.key)!;
    const previous = s.jobs.find(x => x.automatic?.id === input.requestId);
    if (previous) {
      ensure(
        previous.config.config.process[0].type === model.captioner &&
          previous.config.config.process[0].caption.model_name_or_path === model.model &&
          previous.config.config.process[0].caption.caption_prompt === pref.instructions,
        'Richiesta già legata ad altri parametri',
        409,
      );
      link = previous;
      return;
    }
    const scope = s.images.filter(x => !x.excluded && !x.discarded);
    ensure(scope.length > 0 && scope.length <= 1000, 'Seleziona da1 a1000 immagini');
    const digest = hash(
      stableJSON({
        preferences: pref,
        scope: scope.map(x => ({ id: x.id, sha: x.sha, caption: x.caption, revision: x.captionRevision ?? 0 })),
      }),
    );
    const pending = s.jobs.find(
      x =>
        x.automatic?.digest === digest && !['applied', 'dismissed', 'conflict', 'failed'].includes(x.automatic.phase),
    );
    if (pending) {
      link = pending;
      return;
    }
    const folder = path.join(st.folder, 'caption', 'automatic', input.requestId),
      name = 'studio-caption-auto-' + hash(input.requestId).slice(0, 24),
      cfg = captionConfig(model.captioner, model.model, pref.instructions, folder, name);
    cfg.config.process[0].sqlite_db_path = sqlite;
    link = {
      name,
      kind: 'caption',
      state: 'intent',
      config: cfg,
      scope: structuredClone(scope),
      folder,
      automatic: { id: input.requestId, digest, gpu: '0', phase: 'prepared', createdAt: new Date().toISOString() },
    };
    if (input.preferencesRevision !== undefined)
      ensure(
        (s.captionPreferencesRevision ?? 0) === input.preferencesRevision,
        'Impostazioni modificate altrove: bozza conservata',
        409,
      );
    if (stableJSON(s.captionPreferences) !== stableJSON(pref)) {
      s.captionPreferences = pref;
      s.captionPreferencesRevision = (s.captionPreferencesRevision ?? 0) + 1;
    }
    s.jobs.push(link);
  });
  if (terminal.has(link!.automatic!.phase)) return s;
  return reconcileCaption(st, link!.automatic!.id, db, host, [], true);
}
async function applyCompleted(st: StudioStore, link: JobLink, row: any, blocked: string[], db: any) {
  ensure(
    row.step === link.scope!.length && row.total_steps === link.scope!.length,
    'La generazione non ha completato tutto lo scope',
    409,
  );
  const captions: string[] = [];
  for (const [i, img] of link.scope!.entries()) {
    await st.source(img);
    const copy = await contained(
      st.folder,
      path.join(link.folder!, String(i + 1).padStart(6, '0') + path.extname(img.filename).toLowerCase()),
    );
    ensure(hash(await fs.readFile(copy)) === img.sha, 'Input nativo cambiato; caption non applicate', 409);
    const file = await contained(st.folder, path.join(link.folder!, String(i + 1).padStart(6, '0') + '.txt'));
    const stat = await fs.stat(file);
    ensure(stat.size > 0 && stat.size <= 64000, 'Caption generata vuota o troppo grande', 409);
    const value = await fs.readFile(file, 'utf8');
    ensure(value.trim().length > 0, 'Caption generata vuota', 409);
    captions.push(value);
  }
  return st.locked(async () => {
    const s = await st.scan(await st.raw()),
      current = s.jobs.find(x => x.automatic?.id === link.automatic!.id)!;
    if (current.automatic!.phase === 'applied' || current.automatic!.phase === 'dismissed') return s;
    const fresh = await db.job.findUnique({ where: { id: link.jobId } });
    identity(fresh, link, st.datasetRoot);
    ensure(
      fresh.status === 'completed' && fresh.step === link.scope!.length && fresh.total_steps === link.scope!.length,
      'Completamento nativo cambiato',
      409,
    );
    ensure(
      !blocked.some(id => link.scope!.some(x => x.id === id)),
      'Una caption ha una bozza non salvata; nessun risultato sovrascritto',
      409,
    );
    for (const img of link.scope!) {
      const image = s.images.find(x => x.id === img.id);
      ensure(
        image &&
          !image.captionDraft &&
          image.caption === img.caption &&
          (image.captionRevision ?? 0) === (img.captionRevision ?? 0),
        'Una caption è stata modificata o ha una bozza dopo Genera; manteniamo le modifiche manuali',
        409,
      );
    }
    for (const [i, img] of link.scope!.entries()) {
      const image = s.images.find(x => x.id === img.id)!;
      image.caption = captions[i];
      image.captionOverride = true;
      image.caption_source = 'native caption job ' + row.id;
      image.captionRevision = (image.captionRevision ?? 0) + 1;
      image.revision++;
    }
    current.automatic!.phase = 'applied';
    current.automatic!.appliedAt = new Date().toISOString();
    delete current.automatic!.reason;
    s.revision++;
    await st.save(s);
    return s;
  });
}
export async function reconcileCaption(
  st: StudioStore,
  id: string,
  db: any,
  host: CaptionHost,
  blocked: string[] = [],
  explicit = false,
) {
  ensure(
    Array.isArray(blocked) &&
      blocked.length <= 1000 &&
      blocked.every(x => typeof x === 'string' && /^[0-9a-f]{64}$/.test(x)),
    'Bozze non valide',
  );
  let s = await st.read(),
    link = s.jobs.find(x => x.automatic?.id === id);
  ensure(link?.automatic && link.kind === 'caption', 'Generazione gestita non trovata', 404);
  if (link.automatic.phase === 'applied' || link.automatic.phase === 'dismissed') return s;
  if (terminal.has(link.automatic.phase) && !explicit) return s;
  let completed = false;
  try {
    if (!link.jobId) {
      requireCaptionHost(host);
      await stage(st, link);
      const row = await reconcileJob(db, link, link.automatic.gpu, st.datasetRoot);
      s = await linkJob(st, id, row);
      link = s.jobs.find(x => x.automatic?.id === id)!;
    }
    const row = await db.job.findUnique({ where: { id: link.jobId } });
    identity(row, link, st.datasetRoot);
    if (row.status === 'completed') {
      completed = true;
      return await applyCompleted(st, link, row, blocked, db);
    }
    if (['error', 'failed'].includes(row.status))
      return phase(
        st,
        id,
        'failed',
        'Il modello ha segnalato un errore. Le caption attuali e gli originali restano intatti.',
      );
    if (row.status === 'stopped' && link.automatic!.queued)
      return phase(st, id, 'failed', 'Generazione fermata nei controlli nativi; nessun riavvio automatico.');
    if (row.status === 'running' || row.status === 'stopping')
      return phase(st, id, 'active', undefined, { queued: true });
    if (row.status === 'queued' && link.automatic!.phase === 'active' && !explicit) {
      const queue = await db.queue.findUnique({ where: { gpu_ids: link.automatic!.gpu } });
      if (!queue?.is_running) return phase(st, id, 'blocked', 'Coda in pausa: Riconcilia quando vuoi riprenderla.');
      return s;
    }
    requireCaptionHost(host);
    await stage(st, link);
    s = await phase(st, id, 'enqueue-intent');
    link = s.jobs.find(x => x.automatic?.id === id)!;
    const result = await enqueueCaption(db, link, st.datasetRoot, host);
    return phase(st, id, 'active', undefined, { queued: result.status !== 'stopped' });
  } catch (e: any) {
    const message =
      e instanceof Problem
        ? e.message
        : completed
          ? 'Risultati incompleti o non leggibili: nessuna caption applicata. Riconcilia dopo aver verificato il lavoro nativo.'
          : 'Esito della coda non confermato: Riconcilia lo stesso lavoro, senza crearne un altro.';
    return phase(
      st,
      id,
      e instanceof Problem
        ? e.status === 403
          ? 'blocked'
          : e.message.includes('bozza') || e.message.includes('modificata') || e.message.includes('cambiata')
            ? 'conflict'
            : completed
              ? 'failed'
              : 'blocked'
        : completed
          ? 'failed'
          : 'unknown',
      message,
    );
  }
}
export async function dismissCaption(st: StudioStore, rev: number, id: string) {
  return st.mutate(rev, s => {
    const link = s.jobs.find(x => x.automatic?.id === id);
    ensure(link?.automatic && ['failed', 'conflict'].includes(link.automatic.phase), 'Nessun risultato da ignorare');
    link.automatic.phase = 'dismissed';
    link.automatic.reason = 'Risultati ignorati; caption manuali conservate.';
  });
}
