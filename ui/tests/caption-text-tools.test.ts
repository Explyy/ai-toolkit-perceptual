import test from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import sharp from 'sharp';
import { StudioStore } from '../src/datasetStudio/store';
import assert from 'node:assert/strict';
import {
  captionTextSources,
  previewCaptionText,
  captionPreviewCurrent,
  saveCaptionSnapshot,
  captionDraftAfterSave,
  CaptionSaveDraft,
} from '../src/datasetStudio/captionTextTools';
const images = [
  { id: 'a', filename: 'bucket1.png', caption: 'saved cat', revision: 3, excluded: 0 },
  { id: 'b', filename: 'bucket2.png', caption: 'cat cat', revision: 5, excluded: 1 },
  { id: 'c', filename: 'bucket3.png', caption: 'cat', revision: 7, excluded: 0 },
  { id: 'd', filename: 'discarded.png', caption: 'cat', revision: 1, discarded: 1 },
];
test('selected dataset scope uses current drafts; visible scope includes dim photos and respects draft search', () => {
  const drafts = { a: { caption: 'draft cat cat' } };
  const selected = captionTextSources(images, drafts, 'selected', 'bucket1');
  assert.deepEqual(
    selected.map(x => x.id),
    ['a', 'c'],
  );
  const preview = previewCaptionText(selected, 'cat', 'dog');
  assert.equal(preview.occurrences, 3);
  assert.equal(preview.changes[0].next, 'draft dog dog');
  assert.equal(images[0].caption, 'saved cat');
  assert.deepEqual(
    captionTextSources(images, drafts, 'visible', 'bucket2').map(x => x.id),
    ['b'],
  );
  assert.deepEqual(
    captionTextSources(images, drafts, 'visible', 'draft').map(x => x.id),
    ['a'],
  );
});
test('replacement is global literal including regex symbols, dollar tokens, Unicode and deletion', () => {
  const source = [{ id: 'a', caption: 'é🙂.* é🙂.*', revision: 1 }];
  const preview = previewCaptionText(source, 'é🙂.*', '$&$1');
  assert.equal(preview.occurrences, 2);
  assert.equal(preview.changes[0].next, '$&$1 $&$1');
  assert.equal(previewCaptionText(source, 'é🙂.*', '').changes[0].next, ' ');
  assert.equal(previewCaptionText([{ id: 'a', caption: 'aaaa', revision: 1 }], 'aa', 'b').changes[0].next, 'bb');
});
test('empty and unmatched find produce no changes, including empty captions', () => {
  const sources = [{ id: 'a', caption: '', revision: 1 }, images[0]];
  assert.equal(previewCaptionText(sources, '', 'new').changes.length, 0);
  assert.equal(previewCaptionText(sources, 'missing', 'new').changes.length, 0);
  assert.equal(captionPreviewCurrent(previewCaptionText(sources, '', 'new'), sources, '', 'new'), false);
});
test('manual edits, server revisions, changed scope or parameters stale the frozen preview', () => {
  const sources = captionTextSources(images, {}, 'selected', '');
  const preview = previewCaptionText(sources, 'cat', 'dog');
  assert.equal(captionPreviewCurrent(preview, sources, 'cat', 'dog'), true);
  sources[0].caption = 'late edit cat';
  assert.equal(preview.sources[0].caption, 'saved cat');
  assert.equal(captionPreviewCurrent(preview, sources, 'cat', 'dog'), false);
  const original = captionTextSources(images, {}, 'selected', '');
  assert.equal(captionPreviewCurrent(preview, [{ ...original[0], revision: 4 }, original[1]], 'cat', 'dog'), false);
  assert.equal(captionPreviewCurrent(preview, original.slice(1), 'cat', 'dog'), false);
  assert.equal(captionPreviewCurrent(preview, original, 'cat', '$&'), false);
  assert.equal(captionPreviewCurrent(preview, captionTextSources(images, {}, 'visible', ''), 'cat', 'dog'), false);
});

async function saveFixture() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'caption-save-regression-'));
  const datasets = path.join(root, 'datasets');
  await fs.mkdir(path.join(datasets, 'A'), { recursive: true });
  await fs.writeFile(
    path.join(datasets, 'A', 'photo.png'),
    await sharp({ create: { width: 8, height: 8, channels: 3, background: '#123456' } })
      .png()
      .toBuffer(),
  );
  await fs.writeFile(path.join(datasets, 'A', 'photo.txt'), 'geometria originale');
  const store = await new StudioStore(path.join(root, 'data'), datasets, 'A').init();
  let state = await store.read();
  const image = state.images[0];
  state = await store.captionDraft(
    state.revision,
    image.id,
    { caption: 'geometria Modifica successiva.', baseRevision: image.revision },
    0,
  );
  const calls: string[] = [];
  const request = async (action: string, payload: any) => {
    calls.push(action);
    if (action === 'captionDraft')
      state = await store.captionDraft(state.revision, payload.id, payload.draft, payload.draftRevision);
    else state = await store.edit(state.revision, payload.ids, payload.patch);
    return state;
  };
  const draft: CaptionSaveDraft = {
    caption: 'formaQA Modifica successiva.',
    revision: image.revision,
    draftRevision: 1,
  };
  return { root, store, request, draft, id: image.id, calls };
}
test('actual Store reload after immediate Save restores replacement, never the older protected draft', async () => {
  const f = await saveFixture();
  try {
    const saved = await saveCaptionSnapshot(f.id, f.draft, f.request, () => {});
    assert.deepEqual(f.calls, ['captionDraft', 'edit']);
    const reloaded = await f.store.read();
    assert.equal(reloaded.images[0].caption, 'formaQA Modifica successiva.');
    assert.equal(reloaded.images[0].captionDraft, undefined);
    assert.equal(captionDraftAfterSave(f.draft, f.draft, saved.images[0]), undefined);
  } finally {
    await fs.rm(f.root, { recursive: true, force: true });
  }
});
test('edit during protected snapshot save stays local and can protect/reload against the new image revision', async () => {
  const f = await saveFixture();
  try {
    let latest = { ...f.draft };
    const saved = await saveCaptionSnapshot(f.id, f.draft, f.request, draftRevision => {
      latest = { ...latest, caption: latest.caption + ' truly later', draftRevision, persisted: f.draft.caption };
    });
    const remaining = captionDraftAfterSave(latest, f.draft, saved.images[0])!;
    assert.equal(remaining.caption, 'formaQA Modifica successiva. truly later');
    assert.equal(remaining.revision, saved.images[0].revision);
    assert.equal(remaining.persisted, undefined);
    await f.request('captionDraft', {
      id: f.id,
      draft: { caption: remaining.caption, baseRevision: remaining.revision },
      draftRevision: remaining.draftRevision,
    });
    const reloaded = await f.store.read();
    assert.equal(reloaded.images[0].caption, f.draft.caption);
    assert.equal(reloaded.images[0].captionDraft?.caption, remaining.caption);
    assert.equal(reloaded.images[0].captionDraft?.baseRevision, reloaded.images[0].revision);
  } finally {
    await fs.rm(f.root, { recursive: true, force: true });
  }
});
test('another editor protected revision wins CAS; rejected protection never reaches final edit', async () => {
  const f = await saveFixture();
  try {
    const state = await f.store.read();
    await f.store.captionDraft(state.revision, f.id, { caption: 'other editor', baseRevision: f.draft.revision }, 1);
    // Read the current dataset revision, as polling would; the local protected-draft CAS remains stale.
    const request = async (action: string, payload: any) => {
      const current = await f.store.read();
      f.calls.push(action);
      return action === 'captionDraft'
        ? f.store.captionDraft(current.revision, payload.id, payload.draft, payload.draftRevision)
        : f.store.edit(current.revision, payload.ids, payload.patch);
    };
    await assert.rejects(
      saveCaptionSnapshot(f.id, f.draft, request, () => {}),
      /nessun overwrite/,
    );
    assert.deepEqual(f.calls, ['captionDraft']);
    const reloaded = await f.store.read();
    assert.equal(reloaded.images[0].caption, 'geometria originale');
    assert.equal(reloaded.images[0].captionDraft?.caption, 'other editor');
  } finally {
    await fs.rm(f.root, { recursive: true, force: true });
  }
});
