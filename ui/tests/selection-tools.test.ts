import test from 'node:test';
import assert from 'node:assert/strict';
import {
  selectionRectangle,
  selectionHits,
  selectionCommit,
  SelectionGesture,
} from '../src/datasetStudio/selectionTools';
const scope = { dataset: 'A', epoch: 0 };
const targets = [
  { id: 'a', rect: { left: 10, top: 100, right: 90, bottom: 180 } },
  { id: 'b', rect: { left: 110, top: 100, right: 190, bottom: 180 } },
  { id: 'c', rect: { left: 210, top: 100, right: 290, bottom: 180 } },
];
function gesture(excluded: 0 | 1 = 0): SelectionGesture {
  return { scope, pointerId: 1, start: { x: 5, y: 95 }, end: { x: 200, y: 185 }, targets, excluded };
}
test('reverse rectangle crosses only the first two photos; both modes preserve the third', () => {
  const reverse = { ...gesture(), start: { x: 200, y: 185 }, end: { x: 5, y: 95 } };
  assert.deepEqual(selectionRectangle(reverse.start, reverse.end), { left: 5, top: 95, right: 200, bottom: 185 });
  for (const excluded of [0, 1] as const) {
    const before = new Map([
      ['a', 0],
      ['b', 1],
      ['c', 1],
    ]);
    const ids = selectionCommit({ ...reverse, excluded }, scope, ['a', 'b', 'c']);
    for (const id of ids) before.set(id, excluded);
    assert.deepEqual([...before.values()], [excluded, excluded, 1]);
  }
});
test('frozen filtered inventory cannot select hidden or newly visible photos', () => {
  const filtered = { ...gesture(), targets: [targets[0]] };
  assert.deepEqual(selectionCommit(filtered, scope, ['a']), ['a']);
  assert.deepEqual(selectionCommit(filtered, scope, ['a', 'b']), []);
  assert.deepEqual(selectionCommit(gesture(), scope, ['a']), []);
  assert.deepEqual(selectionCommit(gesture(), scope, ['b', 'a', 'c']), []);
});
test('cancel, pointer loss, dataset change including A-B-A and click-sized movement commit nothing', () => {
  assert.deepEqual(selectionCommit(null, scope, ['a', 'b', 'c']), []);
  assert.deepEqual(selectionCommit(gesture(), { dataset: 'A', epoch: 2 }, ['a', 'b', 'c']), []);
  assert.deepEqual(selectionCommit({ ...gesture(), end: { x: 7, y: 97 } }, scope, ['a', 'b', 'c']), []);
});
test('document coordinates freeze targets without admitting edge-only or off-photo hits', () => {
  assert.deepEqual(selectionHits({ ...gesture(), start: { x: 90, y: 100 }, end: { x: 110, y: 180 } }), []);
  const shifted = {
    ...gesture(),
    start: { x: 5, y: 1095 },
    end: { x: 200, y: 1185 },
    targets: targets.map(x => ({ ...x, rect: { ...x.rect, top: x.rect.top + 1000, bottom: x.rect.bottom + 1000 } })),
  };
  assert.deepEqual(selectionHits(shifted), ['a', 'b']);
});
