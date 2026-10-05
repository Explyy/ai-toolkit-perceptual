import test from 'node:test';
import assert from 'node:assert/strict';
import { measuredTiming, progressState } from '../src/datasetStudio/analysisProgress';
import type { State } from '../src/datasetStudio/store';

test('progress distinguishes persisted queue, real artifact completion and blocked recovery without invented ETA', () => {
  const s = {
    images: [{ id: 'a', filename: 'a.png', analysis: null, excluded: 1, reviewRevision: 1 }],
    settings: { count: 3 },
    jobs: [],
    analysisFlow: undefined,
  } as unknown as State;
  assert.equal(progressState(s, null, 'cfg').phase, 'ready');
  s.jobs.push({
    kind: 'analysis',
    name: 'own',
    state: 'linked',
    scope: [...s.images],
    jobId: 'id',
    config: {},
    automatic: { id: 'intent', digest: 'digest', gpu: '0', phase: 'active', createdAt: '' },
  });
  assert.equal(progressState(s, { status: 'queued', step: 0 }, 'cfg').phase, 'queued');
  const running = progressState(s, { status: 'running', step: 1 }, 'cfg', {
    startedAt: 1000,
    total: 1,
    done: 1,
    samples: [],
    currentIndex: 0,
  });
  assert.equal(running.done, 0, 'native step is not a verified artifact');
  assert.equal(running.jobDone, 1);
  assert.equal(running.currentFile, 'a.png');
  assert.equal(running.etaSeconds, undefined);
  assert.equal(s.images[0].excluded, 1);
  s.jobs[0].automatic!.phase = 'failed';
  s.jobs[0].automatic!.reason = 'Spazio riservato insufficiente';
  assert.equal(progressState(s, null, 'cfg').phase, 'blocked');
  assert.match(progressState(s, null, 'cfg').detail!, /Spazio/);
  s.images[0].analysis = { model: { config: 'cfg' } } as any;
  s.jobs[0].automatic!.phase = 'applied';
  s.analysisFlow = { phase: 'complete', config: 'cfg' };
  assert.equal(progressState(s, { status: 'completed', step: 1 }, 'cfg').phase, 'complete');
  assert.equal(s.images[0].excluded, 1);
});
test('ETA uses at least two measured inference samples and rejects malformed counts/times', () => {
  assert.deepEqual(measuredTiming({ startedAt: 1000, total: 10, done: 2, samples: [2, 4] }, 11000), {
    elapsedSeconds: 10,
    etaSeconds: 24,
  });
  assert.equal(measuredTiming({ startedAt: 1000, total: 10, done: 2, samples: [2] }, 11000).etaSeconds, undefined);
  assert.equal(measuredTiming({ startedAt: 1000, total: 10, done: 12, samples: [2, 4] }, 11000).etaSeconds, undefined);
  assert.deepEqual(measuredTiming({ startedAt: 12000, samples: [2, 4] }, 11000), {});
});
