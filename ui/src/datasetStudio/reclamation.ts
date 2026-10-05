import fs from 'node:fs/promises';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { StudioStore, State, JobLink, atomic, contained, hash } from './store';
import { ensure, stableJSON } from './domain';
import { validateSignals } from './analysisPolicy';
import { freeBytes, inputName } from './space';
export type ReclaimCandidate = {
  jobId: string;
  intent: string;
  file: string;
  sha: string;
  size: number;
  dev: number;
  ino: number;
  requestSha: string;
  resultSha: string;
  depthSha: string;
  sourceSha: string;
  linkSha: string;
  config: string;
  blocks: number;
  nlink: number;
};
async function rowIdentity(st: StudioStore, link: JobLink, db: any) {
  ensure(
    link.kind === 'analysis' &&
      link.state === 'linked' &&
      link.jobId &&
      link.automatic?.phase === 'applied' &&
      link.folder &&
      link.scope?.length,
    'Copia non completata/applicata',
    409,
  );
  const row = await db.job.findUnique({ where: { id: link.jobId } });
  ensure(
    row?.status === 'completed' &&
      row.name === link.name &&
      row.job_type === 'analysis' &&
      row.job_ref === st.datasetRoot &&
      row.gpu_ids === '0' &&
      row.step === link.scope.length &&
      row.total_steps === link.scope.length &&
      !row.stop &&
      !row.return_to_queue &&
      stableJSON(JSON.parse(row.job_config)) === stableJSON(link.config),
    'Identità/stato nativo non reclamabile',
    409,
  );
  if (row.pid) {
    const { provenAnalysisDeath } = await import('./analysisFlow');
    ensure(await provenAnalysisDeath(st, link, row), 'Processo completato non ancora provato morto', 409);
  }
  const active = await db.job.findMany({ where: { status: { in: ['queued', 'running', 'stopping'] } } });
  ensure(
    !active.some((x: any) => x.job_ref === st.datasetRoot || x.job_config.includes(link.folder!)),
    'Consumatori attivi: nessuna rimozione',
    409,
  );
  return row;
}
async function candidates(st: StudioStore, s: State, db: any) {
  const output: ReclaimCandidate[] = [];
  for (const link of s.jobs) {
    if (link.kind !== 'analysis' || link.automatic?.phase !== 'applied') continue;
    try {
      const own: ReclaimCandidate[] = [];
      await rowIdentity(st, link, db);
      const requestFile = await contained(st.folder, path.join(link.folder!, 'request.json'));
      ensure((await fs.stat(requestFile)).size <= 1500000, 'Request troppo grande');
      const raw = await fs.readFile(requestFile),
        request = JSON.parse(raw.toString());
      const process = link.config?.config?.process;
      ensure(
        process?.length === 1 &&
          process[0].type === 'dataset_studio_analysis' &&
          process[0].request === requestFile &&
          link.config.config.name === link.name,
        'Scope processo non valido',
      );
      ensure(
        request.name === link.name &&
          request.dataset === st.datasetRoot &&
          /^[0-9a-f]{64}$/.test(request.config) &&
          request.items?.length === link.scope!.length,
        'Request non propria',
      );
      for (const [i, img] of link.scope!.entries()) {
        const item = request.items[i],
          current = s.images.find(x => x.id === img.id && x.sha === img.sha);
        ensure(current && current.analysis?.model?.config === request.config, 'Originale/cache corrente differente');
        ensure(
          item.id === img.id &&
            item.sha === img.sha &&
            item.input === inputName(link, i) &&
            item.output === 'result-' + String(i).padStart(6, '0'),
          'Scope immutabile differente',
        );
        await st.source(current);
        const result = await contained(st.folder, path.join(link.folder!, item.output, 'result.json'));
        ensure((await fs.stat(result)).size <= 1000000, 'Risultato troppo grande');
        const bytes = await fs.readFile(result),
          model = validateSignals(JSON.parse(bytes.toString()), img.sha, request.config);
        const depth = await contained(st.folder, path.join(link.folder!, item.output, 'depth.png'));
        ensure(hash(await fs.readFile(depth)) === model.depth.sha, 'Depth cambiata');
        const file = await contained(st.folder, path.join(link.folder!, item.input), true);
        let stat;
        try {
          stat = await fs.lstat(file);
        } catch (e: any) {
          if (e.code === 'ENOENT') continue;
          throw e;
        }
        ensure(
          stat.isFile() &&
            !stat.isSymbolicLink() &&
            stat.nlink >= 1 &&
            stat.uid === processUid() &&
            stat.size === img.size,
          'Copia non esclusiva/regolare/propria',
        );
        ensure(hash(await fs.readFile(file)) === img.sha, 'Copia SHA differente');
        const original = await fs.stat(await contained(st.datasetRoot, path.join(st.datasetRoot, current.relative)));
        ensure(stat.dev !== original.dev || stat.ino !== original.ino, 'Copia alias di originale modificabile');
        own.push({
          jobId: link.jobId!,
          intent: link.automatic!.id,
          file,
          sha: img.sha,
          size: stat.size,
          dev: stat.dev,
          ino: stat.ino,
          requestSha: hash(raw),
          resultSha: hash(bytes),
          depthSha: model.depth.sha,
          sourceSha: img.sha,
          linkSha: hash(stableJSON(link)),
          config: request.config,
          blocks: stat.blocks * 512,
          nlink: stat.nlink,
        });
      }
      output.push(...own);
    } catch {
      /* Unknown, failed, foreign or incomplete dependencies never qualify. */
    }
  }
  return output;
}
function processUid() {
  return process.getuid?.() ?? -1;
}
export type DedupFile = ReclaimCandidate & { canonical: ReclaimCandidate };
export async function reclaimPlan(st: StudioStore, db: any, quota: number) {
  ensure(Number.isSafeInteger(quota) && quota >= 0, 'Quota non valida');
  return st.locked(async () => {
    const s = await st.raw(),
      all = await candidates(st, s, db),
      first = new Map<string, ReclaimCandidate>();
    let eligibleBytes = 0,
      bytes = 0;
    const files: DedupFile[] = [];
    for (const x of all.sort((a, b) => a.file.localeCompare(b.file))) {
      const key = x.config + ':' + x.sha,
        canonical = first.get(key);
      if (!canonical) {
        first.set(key, x);
        continue;
      }
      if (x.dev !== canonical.dev || x.ino === canonical.ino || x.nlink !== 1) continue;
      eligibleBytes += x.blocks;
      if (bytes < quota) {
        files.push({ ...x, canonical });
        bytes += x.blocks;
      }
    }
    return {
      schema: 1,
      dataset: st.name,
      datasetRoot: st.datasetRoot,
      revision: s.revision,
      quota,
      eligibleBytes,
      plannedBytes: bytes,
      files,
    };
  });
}
// Every historical path remains byte-identical and restart-readable. A native
// read transaction blocks requeue while same-dataset detached derived copies
// are atomically replaced. The canonical pool NEVER links mutable originals.
export async function reclaimInputs(st: StudioStore, db: any, plan: Awaited<ReturnType<typeof reclaimPlan>>) {
  return st.locked<{ receipt: string; replacedBytes: number; beforeFree: number; afterFree: number }>(() =>
    db.$transaction(
      async (tx: any) => {
        const s = await st.raw();
        ensure(
          plan.datasetRoot === st.datasetRoot && plan.dataset === st.name && s.revision === plan.revision,
          'Piano storage obsoleto',
          409,
        );
        const fresh = await candidates(st, s, tx),
          byFile = new Map(fresh.map(x => [x.file, x]));
        ensure(plan.plannedBytes >= plan.quota, 'Copie duplicate completate insufficienti: nessuna sostituzione', 409);
        for (const { canonical, ...x } of plan.files) {
          ensure(
            stableJSON(byFile.get(x.file)) === stableJSON(x) &&
              stableJSON(byFile.get(canonical.file)) === stableJSON(canonical),
            'Prova copia cambiata: nessuna sostituzione',
            409,
          );
        }
        const dir = await contained(st.folder, path.join(st.folder, 'reclamation'), true);
        await fs.mkdir(dir, { recursive: true });
        const pool = await contained(st.folder, path.join(st.folder, 'analysis', 'derived-input-pool'), true);
        await fs.mkdir(pool, { recursive: true });
        const file = path.join(dir, randomUUID() + '.json'),
          receipt = {
            ...plan,
            beforeFree: await freeBytes(st.folder),
            phase: 'claimed',
            deduplicated: [] as DedupFile[],
            afterFree: 0,
          };
        await atomic(file, stableJSON(receipt));
        for (const x of plan.files) {
          for (const proof of [x, x.canonical]) {
            const link = s.jobs.find(j => j.automatic?.id === proof.intent)!;
            await rowIdentity(st, link, tx);
            const current = s.images.find(i => i.sha === proof.sourceSha && link.scope!.some(j => j.id === i.id))!;
            await st.source(current);
            const stat = await fs.lstat(await contained(st.folder, proof.file));
            ensure(
              stat.ino === proof.ino &&
                stat.dev === proof.dev &&
                stat.uid === processUid() &&
                hash(await fs.readFile(proof.file)) === proof.sha,
              'Copia cambiata prima della sostituzione',
              409,
            );
          }
          ensure((await fs.lstat(x.file)).nlink === 1, 'Copia già condivisa: ricalcola piano', 409);
          const blob = await contained(st.folder, path.join(pool, x.config + '-' + x.sha), true);
          try {
            await fs.link(x.canonical.file, blob);
          } catch (e: any) {
            if (e.code !== 'EEXIST') throw e;
          }
          const blobStat = await fs.lstat(blob),
            canonicalStat = await fs.lstat(x.canonical.file);
          ensure(
            blobStat.isFile() &&
              !blobStat.isSymbolicLink() &&
              blobStat.uid === processUid() &&
              blobStat.ino === canonicalStat.ino &&
              blobStat.dev === canonicalStat.dev &&
              hash(await fs.readFile(blob)) === x.sha,
            'Pool derived non verificato',
            409,
          );
          const temporary = x.file + '.dedup-' + randomUUID();
          try {
            await fs.link(blob, temporary); // Unsupported filesystem refuses safely, no raw-unlink fallback.
            await fs.rename(temporary, x.file);
          } finally {
            await fs.rm(temporary, { force: true });
          }
          ensure(hash(await fs.readFile(x.file)) === x.sha, 'Readback copia differente');
          receipt.deduplicated.push(x);
          receipt.afterFree = await freeBytes(st.folder);
          await atomic(file, stableJSON(receipt));
          if (receipt.afterFree - receipt.beforeFree >= plan.quota) break;
        }
        receipt.phase = 'finished';
        receipt.afterFree = await freeBytes(st.folder);
        await atomic(file, stableJSON(receipt));
        return {
          receipt: file,
          replacedBytes: receipt.deduplicated.reduce((n, x) => n + x.size, 0),
          beforeFree: receipt.beforeFree,
          afterFree: receipt.afterFree,
        };
      },
      { timeout: 120000, maxWait: 1000 },
    ),
  );
}
