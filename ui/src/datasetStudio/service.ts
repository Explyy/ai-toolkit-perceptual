import fs from 'node:fs/promises';
import path from 'node:path';
import { getDataRoot, getDatasetsRoot, getHFToken } from '@/server/settings';
import prisma from '@/server/prisma';
import { StudioStore, contained } from './store';
import { ownedLock } from './ownerLock';
import { Hub } from './hf';
import { drainManaged } from './managedSync';
import { reconcileAnalysis } from './analysisFlow';
import { reconcileCaption, captionHost } from './captionFlow';
let running = false,
  timer: ReturnType<typeof setInterval> | undefined;
export async function serviceTick() {
  if (running) return;
  running = true;
  try {
    const data = await getDataRoot(),
      datasets = await getDatasetsRoot(),
      base = await fs.realpath(data),
      claim = await contained(base, path.join(base, 'dataset-studio', 'background-service'), true);
    await fs.mkdir(claim, { recursive: true });
    await ownedLock(claim, async () => {
      const token = await getHFToken(),
        host = await captionHost();
      for (const d of await fs.readdir(datasets, { withFileTypes: true }))
        if (d.isDirectory() && !d.name.startsWith('.')) {
          try {
            if (await fs.lstat(path.join(datasets, d.name, '.studio-materializing.json')).catch(() => null)) continue;
            const st = await new StudioStore(data, datasets, d.name).init();
            let s = await st.read();
            for (const link of s.jobs) {
              if (
                !link.automatic ||
                ['applied', 'dismissed', 'failed', 'unknown', 'conflict', 'blocked'].includes(link.automatic.phase)
              )
                continue;
              // Only persisted user-created intents are replayed. Never prepareAnalysis.
              if (link.kind === 'analysis') s = await reconcileAnalysis(st, link.automatic.id, prisma, host);
              if (link.kind === 'caption') s = await reconcileCaption(st, link.automatic.id, prisma, host);
            }
            if (token) await drainManaged(st, new Hub(token));
          } catch {
            /* Per-dataset durable status/owned receipts remain; retry next tick. No credentials/log dumps. */
          }
        }
    });
  } finally {
    running = false;
  }
}
export function startStudioService() {
  if (timer || process.env.DATASET_STUDIO_PREVIEW === '1' || !process.env.DATASET_STUDIO_ROOT) return;
  const tick = () => void serviceTick().catch(() => {});
  timer = setInterval(tick, 15000);
  timer.unref();
  tick();
}
