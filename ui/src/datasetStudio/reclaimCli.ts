// Standalone bridge for a reviewed release whose startup free-space guard must
// remain unchanged. Dry-run is default. Uses the same implementation as staging.
import fs from 'node:fs/promises';
import path from 'node:path';
import { PrismaClient } from '@prisma/client';
import { StudioStore, hash, atomic } from './store';
import { ensure, stableJSON } from './domain';
import { reclaimPlan, reclaimInputs } from './reclamation';
async function main() {
  const [data, datasets, name, sqlite, quotaText, receipt, approval] = process.argv.slice(2);
  ensure(
    [data, datasets, sqlite, receipt].every(x => x && path.isAbsolute(x)),
    'Absolute bound roots/DB/receipt required',
  );
  const quota = Number(quotaText);
  ensure(Number.isSafeInteger(quota) && quota > 0, 'Exact positive quota required');
  const db = new PrismaClient({ datasourceUrl: 'file:' + sqlite });
  try {
    const st = await new StudioStore(data, datasets, name).init(),
      plan = await reclaimPlan(st, db, quota),
      digest = hash(stableJSON(plan));
    if (approval) {
      ensure(approval === digest, 'Reviewed plan differs; no consolidation', 409);
      await atomic(receipt, stableJSON({ plan, digest, result: await reclaimInputs(st, db, plan) }));
    } else await atomic(receipt, stableJSON({ plan, digest }));
    console.log(
      JSON.stringify({
        phase: approval ? 'executed' : 'dry-run',
        digest,
        files: plan.files.length,
        plannedBytes: plan.plannedBytes,
        eligibleBytes: plan.eligibleBytes,
      }),
    );
  } finally {
    await db.$disconnect();
  }
}
main().catch(e => {
  console.error(e.message);
  process.exitCode = 1;
});
