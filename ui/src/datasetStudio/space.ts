import fs from 'node:fs/promises';
import path from 'node:path';
import { StudioStore, JobLink, contained, hash } from './store';
import { ensure } from './domain';
import { validateSignals } from './analysisPolicy';
export const ANALYSIS_OUTPUT_BYTES = 1100000; // validated JSON <=1MB +128x128 16-bit PNG, receipt/metadata allowance.
export function reserveBytes() {
  const n = Number(process.env.DATASET_STUDIO_MIN_FREE_BYTES ?? 24000000000);
  ensure(Number.isSafeInteger(n) && n >= 24000000000, 'Riserva storage non valida');
  return n;
}
export async function freeBytes(folder: string) {
  const x = await fs.statfs(folder);
  return x.bavail * x.bsize;
}
export function inputName(link: JobLink, index: number) {
  return (
    (link.kind === 'analysis' ? 'input-' + String(index).padStart(6, '0') : String(index + 1).padStart(6, '0')) +
    path.extname(link.scope![index].filename).toLowerCase()
  );
}
export async function stageBudget(st: StudioStore, link: JobLink, config?: string) {
  let missingBytes = 0,
    outputBytes = 0;
  for (const [i, img] of link.scope!.entries()) {
    await st.source(img);
    const file = await contained(st.folder, path.join(link.folder!, inputName(link, i)), true);
    try {
      const stat = await fs.lstat(file);
      ensure(stat.isFile() && !stat.isSymbolicLink(), 'Copia di lavoro non sicura', 409);
      ensure(stat.size === img.size && hash(await fs.readFile(file)) === img.sha, 'Copia di lavoro cambiata', 409);
    } catch (e: any) {
      if (e.code !== 'ENOENT') throw e;
      missingBytes += img.size;
    }
    if (link.kind === 'analysis') {
      try {
        const dir = path.join(link.folder!, 'result-' + String(i).padStart(6, '0'));
        const result = await contained(st.folder, path.join(dir, 'result.json'));
        ensure((await fs.stat(result)).size <= 1000000, 'Risultato troppo grande');
        const model = validateSignals(JSON.parse(await fs.readFile(result, 'utf8')), img.sha, config!);
        ensure(
          hash(await fs.readFile(await contained(st.folder, path.join(dir, 'depth.png')))) === model.depth.sha,
          'Cache depth cambiata',
        );
      } catch (e: any) {
        if (e.code !== 'ENOENT') throw e;
        outputBytes += ANALYSIS_OUTPUT_BYTES;
      }
    }
  }
  const free = await freeBytes(st.folder),
    reserve = reserveBytes(),
    needed = missingBytes + outputBytes;
  return { free, reserve, missingBytes, outputBytes, needed, deficit: Math.max(0, reserve + needed - free) };
}
export type SpaceBudget = Awaited<ReturnType<typeof stageBudget>>;
export function requireBudget(value: SpaceBudget) {
  ensure(
    value.deficit === 0,
    `Spazio insufficiente: liberi ${value.free} byte, copie mancanti ${value.missingBytes}, risultati ${value.outputBytes}, riserva ${value.reserve}; mancano ${value.deficit} byte. Originali conservati.`,
    409,
  );
}
