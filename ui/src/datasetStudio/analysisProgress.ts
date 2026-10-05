import fs from 'node:fs/promises';
import path from 'node:path';
import type { State, StudioStore, JobLink } from './store';
import { contained } from './store';
import { select } from './domain';
export type AnalysisProgress = {
  phase: 'ready' | 'queued' | 'running' | 'waiting' | 'complete' | 'blocked' | 'unavailable';
  done: number;
  total: number;
  jobDone: number;
  jobTotal: number;
  elapsedSeconds?: number;
  etaSeconds?: number;
  currentFile?: string;
  detail?: string;
  tentative: string[];
};
export function measuredTiming(receipt: any, now = Date.now()) {
  if (!receipt || !Number.isFinite(receipt.startedAt) || receipt.startedAt > now) return {};
  const end = Number.isFinite(receipt.finishedAt) ? receipt.finishedAt : now;
  const elapsedSeconds = Math.max(0, (end - receipt.startedAt) / 1000);
  const samples = Array.isArray(receipt.samples) ? receipt.samples : [];
  if (
    !Number.isInteger(receipt.total) ||
    !Number.isInteger(receipt.done) ||
    receipt.done < 0 ||
    receipt.done > receipt.total ||
    samples.length < 2 ||
    samples.length > 32 ||
    samples.some((x: any) => !Number.isFinite(x) || x <= 0)
  )
    return { elapsedSeconds };
  const remaining = Math.max(0, receipt.total - receipt.done);
  return {
    elapsedSeconds,
    etaSeconds: (remaining * samples.reduce((n: number, x: number) => n + x, 0)) / samples.length,
  };
}
export function progressState(s: State, row: any, config: string, receipt?: any): AnalysisProgress {
  const images = s.images.filter(x => !x.discarded),
    valid = images.filter(x => x.analysis?.model?.config === config),
    link = s.jobs.filter(x => x.kind === 'analysis' && x.automatic?.phase !== 'dismissed').at(-1);
  let phase: AnalysisProgress['phase'] = 'ready';
  if (
    valid.length === images.length &&
    s.analysisFlow?.config === config &&
    s.analysisFlow?.phase === 'complete' &&
    (!link || ['applied', 'failed', 'blocked'].includes(link.automatic!.phase))
  )
    phase = 'complete';
  else if (s.analysisFlow?.phase === 'unavailable' && !link) phase = 'unavailable';
  else if (link && ['blocked', 'failed', 'unknown'].includes(link.automatic!.phase)) phase = 'blocked';
  else if (link && !['applied', 'dismissed'].includes(link.automatic!.phase))
    phase = row?.status === 'queued' ? 'queued' : row?.status === 'running' ? 'running' : 'waiting';
  const done = valid.length;
  return {
    phase,
    done,
    total: images.length,
    jobDone: Math.max(0, Math.min(link?.scope?.length ?? 0, row?.step ?? 0)),
    jobTotal: link?.scope?.length ?? 0,
    ...measuredTiming(receipt),
    currentFile:
      receipt && Number.isInteger(receipt.currentIndex) ? link?.scope?.[receipt.currentIndex]?.filename : undefined,
    detail: link?.automatic?.reason ?? (phase === 'unavailable' ? s.analysisFlow?.reason : undefined),
    tentative: phase !== 'complete' ? select(valid, s.settings.count, s.settings.testId).selected.map(x => x.id) : [],
  };
}
export async function analysisProgress(st: StudioStore, s: State, rows: any[], config: string) {
  const link = s.jobs.filter(x => x.kind === 'analysis' && x.automatic?.phase !== 'dismissed').at(-1);
  let receipt;
  if (link?.folder) {
    try {
      const file = await contained(st.folder, path.join(link.folder, 'progress.json'));
      if ((await fs.stat(file)).size <= 10000) {
        const value = JSON.parse(await fs.readFile(file, 'utf8'));
        if (
          value.name === link.name &&
          value.config === config &&
          value.total === link.scope?.length &&
          Number.isInteger(value.done) &&
          value.done >= 0 &&
          value.done <= value.total
        )
          receipt = value;
      }
    } catch {
      /* Missing/legacy progress never manufactures timing. */
    }
  }
  return progressState(
    s,
    rows.find(x => x.id === link?.jobId),
    config,
    receipt,
  );
}
