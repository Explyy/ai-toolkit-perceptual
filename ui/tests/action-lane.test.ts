import test from 'node:test';
import assert from 'node:assert/strict';
import { ActionLane, LoadErrorOwner } from '../src/datasetStudio/actionLane';
const tick = () => new Promise<void>(resolve => setImmediate(resolve));
test('a successful current reload clears only its previous same-dataset load failure', () => {
  const errors = new LoadErrorOwner(),
    owner = {};
  assert.equal(errors.failed(errors.begin(owner), owner), true);
  assert.equal(errors.succeeded(errors.begin(owner), owner), true);
  assert.equal(errors.succeeded(errors.begin(owner), owner), false);
});
test('reload preserves earlier and later action/save errors and protected draft warnings', () => {
  const errors = new LoadErrorOwner(),
    owner = {};
  errors.action();
  assert.equal(errors.succeeded(errors.begin(owner), owner), false);
  assert.equal(errors.failed(errors.begin(owner), owner), true);
  const read = errors.begin(owner);
  errors.action();
  assert.equal(errors.succeeded(read, owner), false);
  assert.equal(errors.failed(read, owner), false);
});
test('older requests and dataset epochs cannot overwrite or clear newer load errors', () => {
  const errors = new LoadErrorOwner(),
    a = {},
    b = {},
    newA = {};
  const old = errors.begin(a),
    latest = errors.begin(a);
  assert.equal(errors.failed(latest, a), true);
  assert.equal(errors.current(old, a), false);
  assert.equal(errors.succeeded(old, a), false);
  assert.equal(errors.failed(old, a), false);
  assert.equal(errors.failed(errors.begin(a), b), false);
  const pending = errors.begin(a);
  assert.equal(errors.succeeded(pending, newA), false);
  assert.equal(errors.failed(pending, newA), false);
  errors.action();
  assert.equal(errors.succeeded(errors.begin(newA), newA), false);
});
test('the current manual reload cannot replace a newer accepted curation revision or clear its load error', () => {
  const errors = new LoadErrorOwner(),
    owner = {};
  assert.equal(errors.failed(errors.begin(owner), owner), true);
  const pending = errors.begin(owner);
  assert.equal(errors.accepts(pending, owner, 81, 82), false);
  // Workspace calls succeeded only after accepting a current, nonstale view.
  assert.equal(errors.accepts(pending, owner, 82, 82), true);
  assert.equal(errors.succeeded(pending, owner), true);
  assert.equal(errors.accepts(errors.begin(owner), owner, 81), true);
  assert.equal(errors.accepts(errors.begin(owner), {}, 83, 82), false);
});
function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>(r => {
    resolve = r;
  });
  return { promise, resolve };
}

test('an enabled selection waits for preference autosave and uses its confirmed latest revision without retry', async () => {
  const lane = new ActionLane();
  let revision = 10,
    included = true;
  const release = lane.tryBackground()!;
  const selection = lane.run(
    'exclude-bucket2',
    () => true,
    async () => {
      assert.equal(revision, 11);
      included = false;
      revision++;
    },
  );
  await tick();
  assert.equal(included, true);
  assert.equal(lane.tryBackground(), null);
  revision = 11;
  release();
  await selection;
  assert.equal(included, false);
  assert.equal(revision, 12);
  assert.equal(lane.busy, false);
});

for (const background of ['preferences', 'caption draft', 'automatic poll'])
  test('distinct user intents are FIFO and never dropped behind ' + background, async () => {
    const lane = new ActionLane(),
      release = lane.tryBackground()!,
      seen: string[] = [],
      first = deferred();
    const a = lane.run(
      'select-image1',
      () => true,
      async () => {
        seen.push('include1');
        await first.promise;
        seen.push('saved1');
      },
    );
    const b = lane.run(
      'exclude-image2',
      () => true,
      async () => {
        seen.push('exclude2');
      },
    );
    const c = lane.run(
      'bulk-visible',
      () => true,
      async () => {
        seen.push('bulk');
      },
    );
    assert.deepEqual(seen, []);
    release();
    await tick();
    assert.deepEqual(seen, ['include1']);
    assert.equal(lane.tryBackground(), null);
    first.resolve();
    await Promise.all([a, b, c]);
    assert.deepEqual(seen, ['include1', 'saved1', 'exclude2', 'bulk']);
    assert.equal(lane.busy, false);
  });

test('replayed pending Generate shares one intent and cannot create a duplicate native job', async () => {
  const lane = new ActionLane(),
    release = lane.tryBackground()!;
  let creates = 0;
  const first = lane.run(
    'datasetA:generate',
    () => true,
    async () => {
      creates++;
      return 'same-request-and-job';
    },
  );
  const replay = lane.run(
    'datasetA:generate',
    () => true,
    async () => {
      creates++;
      return 'duplicate';
    },
  );
  assert.equal(replay, first);
  release();
  assert.deepEqual(await Promise.all([first, replay]), ['same-request-and-job', 'same-request-and-job']);
  assert.equal(creates, 1);
  await tick();
  await lane.run(
    'datasetA:generate',
    () => true,
    async () => {
      creates++;
      return 'new explicit intent';
    },
  );
  assert.equal(creates, 2);
});

test('queued actions from the old dataset are cancelled, including A to B to A, and new dataset work proceeds', async () => {
  const lane = new ActionLane(),
    release = lane.tryBackground()!;
  let scope = { dataset: 'A' },
    writes = 0;
  const previous = scope;
  const old = lane.run(
    'A:epoch0:upload',
    () => scope === previous,
    async () => {
      writes++;
    },
  );
  scope = { dataset: 'B' };
  scope = { dataset: 'A' };
  const latest = scope;
  const fresh = lane.run(
    'A:epoch2:selection',
    () => scope === latest,
    async () => {
      writes += 10;
    },
  );
  release();
  assert.equal(await old, undefined);
  await fresh;
  assert.equal(writes, 10);
  assert.equal(lane.busy, false);
});

test('dataset changes after permission but before action execution are checked again', async () => {
  const lane = new ActionLane();
  let same = true,
    writes = 0;
  const intent = lane.run(
    'old-caption-save',
    () => same,
    async () => {
      writes++;
    },
  );
  same = false;
  assert.equal(await intent, undefined);
  assert.equal(writes, 0);
  assert.equal(lane.busy, false);
});

test('failed user work releases the exact lane once and allows queued recovery without background takeover', async () => {
  const lane = new ActionLane(),
    release = lane.tryBackground()!;
  let recovered = false;
  const failed = lane.run(
    'save',
    () => true,
    async () => {
      throw Error('stale CAS, retain draft');
    },
  );
  const recovery = lane.run(
    'refresh',
    () => true,
    async () => {
      recovered = true;
    },
  );
  release();
  release();
  await assert.rejects(failed, /retain draft/);
  await recovery;
  assert.equal(recovered, true);
  assert.equal(lane.busy, false);
  const next = lane.tryBackground();
  assert.ok(next);
  next();
  assert.equal(lane.busy, false);
});

test('an immediate foreground action runs with no background and does not overwrite a later draft during save', async () => {
  const lane = new ActionLane(),
    sent = deferred();
  let draft = 'first',
    persisted = '';
  const save = lane.run(
    'caption-save',
    () => true,
    async () => {
      const snapshot = draft;
      await sent.promise;
      persisted = snapshot;
      if (draft === snapshot) draft = '';
    },
  );
  await tick();
  draft = 'edited while saving';
  sent.resolve();
  await save;
  assert.equal(persisted, 'first');
  assert.equal(draft, 'edited while saving');
  assert.equal(lane.busy, false);
});
