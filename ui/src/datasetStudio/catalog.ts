import fs from 'node:fs/promises';
import path from 'node:path';
import { Hub, HfFile, hubPath } from './hf';
import { StudioStore, atomic, contained, hash, datasetName } from './store';
import { ensure, repoId, revision, stableJSON, path as remotePath } from './domain';
import { validateProjection, managedRoot } from './managedSync';
import { ownedLock } from './ownerLock';
import { freeBytes, reserveBytes } from './space';
export type CatalogEntry = {
  repo: string;
  folder: string;
  revision: string;
  kind: 'pairs' | 'managed' | 'media';
  key?: string;
  unsupported?: number;
};
export async function nativeCatalog(datasets: string, data: string) {
  const root = await fs.realpath(datasets),
    rows = [];
  for (const d of await fs.readdir(root, { withFileTypes: true }))
    if (d.isDirectory() && !d.name.startsWith('.')) {
      const dir = await contained(root, path.join(root, d.name));
      datasetName(d.name);
      if (await fs.lstat(path.join(dir, '.studio-materializing.json')).catch(() => null)) continue;
      const st = await new StudioStore(data, root, d.name).init(),
        s = await st.raw();
      rows.push({ name: d.name, title: s.displayTitle ?? s.managedBinding?.title ?? s.dataset, source: s.sourceBinding, managed: s.managedBinding });
    }
  return rows;
}
export async function folders(hub: Hub, repo: string, rev?: string) {
  const info = await hub.owned(repo),
    pinned = rev ? revision(rev) : info.sha,
    rows = await hub.entries(repo, pinned, '', true),
    entries = new Map<string, CatalogEntry>();
  for (const x of rows) {
    if (x.type !== 'file') continue;
    if (/\.(png|jpe?g|webp)$/i.test(x.path)) {
      const parent = path.posix.dirname(x.path),
        folder = parent === '.' ? '' : parent;
      if (folder) hubPath(folder);
      entries.set(folder, { repo, folder, revision: pinned, kind: 'pairs' });
    }
    if (/\.(gif|bmp|avif|tiff?|mp4|avi|mov|mkv|wmv|m4v|flv|mp3|wav|flac|ogg)$/i.test(x.path)) {
      const parent = path.posix.dirname(x.path),
        folder = parent === '.' ? '' : parent;
      const old = entries.get(folder);
      if (old) old.unsupported = (old.unsupported ?? 0) + 1;
      else entries.set(folder, { repo, folder, revision: pinned, kind: 'media', unsupported: 1 });
    }
    const managed = String(x.path).match(/^dataset-studio\/managed\/([0-9a-f]{64})\/current\.json$/);
    if (managed)
      entries.set(managedRoot(managed[1]), {
        repo,
        folder: managedRoot(managed[1]),
        revision: pinned,
        kind: 'managed',
        key: managed[1],
      });
  }
  // Internal content-addressed blobs are managed objects, never native datasets.
  return [...entries.values()]
    .filter(x => x.kind === 'managed' || !x.folder.startsWith('dataset-studio/'))
    .sort((a, b) => a.folder.localeCompare(b.folder));
}
async function importRoot(data: string) {
  const root = await fs.realpath(data),
    dir = await contained(root, path.join(root, 'dataset-studio', 'catalog-imports'), true);
  await fs.mkdir(dir, { recursive: true });
  return dir;
}
export function catalogIdentity(entry: CatalogEntry) {
  return hash(
    stableJSON({
      repo: repoId(entry.repo),
      folder: entry.folder ? hubPath(entry.folder) : '',
      revision: revision(entry.revision),
      kind: entry.kind,
    }),
  );
}
export async function importStatus(data: string, id: string) {
  ensure(/^[0-9a-f]{64}$/.test(id), 'Invalid import identity');
  const dir = await importRoot(data);
  try {
    return JSON.parse(await fs.readFile(await contained(dir, path.join(dir, id + '.json')), 'utf8'));
  } catch (e: any) {
    if (e.code === 'ENOENT') return { phase: 'idle', done: 0 };
    throw e;
  }
}
export async function materialize(hub: Hub, data: string, datasets: string, entry: CatalogEntry) {
  repoId(entry.repo);
  if (entry.folder) hubPath(entry.folder);
  revision(entry.revision);
  ensure(['pairs', 'managed'].includes(entry.kind), 'Invalid import kind');
  await hub.owned(entry.repo);
  const id = catalogIdentity(entry),
    imports = await importRoot(data);
  return ownedLock(imports, async () => {
    const base = await fs.realpath(datasets),
      title = entry.folder.split('/').at(-1) || entry.repo.split('/').at(-1)!,
      name = datasetName(('HF-' + title.slice(0, 60) + '-' + id.slice(0, 16)).replace(/[^a-zA-Z0-9_.-]/g, '_'));
    const target = await contained(base, path.join(base, name), true),
      receipt = await contained(imports, path.join(imports, id + '.json'), true);
    let temporary = await contained(base, path.join(base, '.studio-import-' + id), true);
    let present = false;
    try {
      await fs.stat(target);
      present = true;
    } catch (e: any) {
      if (e.code !== 'ENOENT') throw e;
    }
    if (present) {
      const st = await new StudioStore(data, base, name).init(),
        s = await st.raw();
      const markerFile = await contained(target, path.join(target, '.studio-materializing.json'), true);
      let marker;
      try {
        marker = JSON.parse(await fs.readFile(markerFile, 'utf8'));
      } catch (e: any) {
        if (e.code !== 'ENOENT') throw e;
      }
      if (s.sourceBinding?.inventory === id) {
        if (marker) {
          ensure(marker.id === id && stableJSON(marker.entry) === stableJSON(entry), 'Unknown import marker', 409);
          await st.read();
          await fs.unlink(markerFile);
        }
        return { name, id, phase: 'complete' };
      }
      const old = JSON.parse(await fs.readFile(receipt, 'utf8'));
      ensure(
        marker?.id === id &&
          stableJSON(marker.entry) === stableJSON(entry) &&
          old.id === id &&
          old.name === name &&
          old.phase === 'downloading' &&
          s.jobs.length === 0 &&
          s.templates.length === 0 &&
          s.snapshots.length === 0 &&
          s.images.every(x => !x.captionOverride && !x.reviewRevision && !x.captionDraft),
        'Existing local dataset differs; no overwrite',
        409,
      );
      temporary = target; // explicit owned unfinished publication, never an existing user dataset
    }
    let projection: any,
      digest: string | undefined,
      files: Array<{ relative: string; sha?: string; size: number; remote: string; caption?: HfFile }>;
    if (entry.kind === 'managed') {
      const match = entry.folder.match(/^dataset-studio\/managed\/([0-9a-f]{64})$/);
      ensure(match && match[1] === entry.key, 'Invalid managed namespace');
      const pointer = JSON.parse(
        Buffer.from(await hub.download(entry.repo, entry.revision, entry.folder + '/current.json', 2000)).toString(),
      );
      ensure(
        pointer.schema === 1 && pointer.key === entry.key && /^[0-9a-f]{64}$/.test(pointer.digest),
        'Invalid managed pointer',
      );
      digest = pointer.digest;
      const bytes = await hub.download(
        entry.repo,
        entry.revision,
        entry.folder + '/versions/' + digest + '.json',
        16 * 1024 * 1024,
      );
      ensure(hash(bytes) === digest, 'Managed projection changed');
      projection = validateProjection(JSON.parse(Buffer.from(bytes).toString()));
      ensure(projection.key === entry.key, 'Managed key differs');
      files = projection.images.map((x: any) => ({
        relative: x.relative,
        sha: x.sha,
        size: x.size,
        remote: entry.folder + '/blobs/' + x.sha,
      }));
    } else {
      const values = (await hub.entries(entry.repo, entry.revision, entry.folder)).filter(x => x.type === 'file');
      const images = values.filter(x => /\.(png|jpe?g|webp)$/i.test(x.path)),
        paired = new Set(images.flatMap(x => [x.path, x.path.replace(/\.[^.]+$/, '.txt')]));
      const prefix = entry.folder ? entry.folder + '/' : '';
      ensure(images.length > 0 && images.length <= 2500, 'No bounded image sources');
      files = images.map(x => {
        ensure(x.path.startsWith(prefix) && !x.path.slice(prefix.length).includes('/'), 'Only direct image sources');
        return {
          relative: path.posix.basename(x.path),
          size: x.size,
          remote: x.path,
          caption: values.find(y => y.path === x.path.replace(/\.[^.]+$/, '.txt')),
        };
      });
    }
    ensure(
      files.length > 0 &&
        files.length <= 2500 &&
        files.every(x => Number.isSafeInteger(x.size) && x.size > 0 && x.size <= 24 * 1024 * 1024),
      'Import bounds exceeded',
    );
    await fs.mkdir(temporary, { recursive: true });
    let missing = 0;
    for (const x of files) {
      const f = await contained(temporary, path.join(temporary, x.relative), true);
      try {
        const bytes = await fs.readFile(f);
        ensure(bytes.length === x.size && (!x.sha || hash(bytes) === x.sha), 'Import staging changed');
      } catch (e: any) {
        if (e.code !== 'ENOENT') throw e;
        missing +=
          x.size +
          (x.caption?.size ??
            Buffer.byteLength(projection?.images.find((y: any) => y.relative === x.relative)?.caption ?? ''));
      }
    }
    const free = await freeBytes(base),
      reserve = reserveBytes();
    ensure(
      free >= reserve + missing,
      `Import sospeso: liberi ${free}, copie mancanti ${missing}, riserva ${reserve} byte. Nessun originale eliminato.`,
      409,
    );
    const progress = {
      schema: 1,
      id,
      entry,
      name,
      phase: 'downloading',
      done: 0,
      total: files.length,
      missingCaptions: projection ? 0 : files.filter(x => !x.caption).length,
    };
    await atomic(receipt, stableJSON(progress));
    for (const x of files) {
      const f = await contained(temporary, path.join(temporary, x.relative), true);
      await fs.mkdir(path.dirname(f), { recursive: true });
      let bytes: Uint8Array;
      try {
        bytes = await fs.readFile(f);
      } catch (e: any) {
        if (e.code !== 'ENOENT') throw e;
        bytes = await hub.download(entry.repo, entry.revision, x.remote, x.size + 1);
      }
      ensure(bytes.length === x.size && (!x.sha || hash(bytes) === x.sha), 'Imported original SHA differs');
      // Pinned Hub Git/LFS integrity is verified even for resumed pair staging.
      if (!x.sha) {
        const verified = await hub.download(entry.repo, entry.revision, x.remote, x.size + 1);
        ensure(hash(verified) === hash(bytes), 'Resumed pinned image differs');
      }
      await atomic(f, bytes);
      const caption = x.caption
        ? await hub.download(entry.repo, entry.revision, x.caption.path, 64000)
        : Buffer.from(projection?.images.find((y: any) => y.relative === x.relative)?.caption ?? '');
      ensure(caption.length < 64000, 'Caption exceeds64KB');
      await atomic(f.replace(/\.[^.]+$/, '.txt'), caption);
      progress.done++;
      await atomic(receipt, stableJSON(progress));
    }
    // Only the owned hidden import is published; existing local folders never replaced.
    if (!present) {
      ensure(
        !(await fs.lstat(target).catch((e: any) => {
          if (e.code === 'ENOENT') return null;
          throw e;
        })),
        'Local target appeared; no overwrite',
        409,
      );
      await atomic(path.join(temporary, '.studio-materializing.json'), stableJSON({ id, entry }));
      await fs.rename(temporary, target);
    }
    const st = await new StudioStore(data, base, name).init();
    let s = await st.read();
    s.sourceBinding = { repo: entry.repo, folder: entry.folder, revision: entry.revision, inventory: id };
    if (projection) {
      s.displayTitle = projection.dataset;
      s.settings = projection.settings;
      s.captionPreferences = projection.captionPreferences;
      s.managedBinding = { repo: entry.repo, key: projection.key, title: projection.dataset, baseDigest: digest };
      for (const image of s.images) {
        const restored = projection.images.find((x: any) => x.relative === image.relative && x.sha === image.sha);
        ensure(restored, 'Restored inventory differs');
        Object.assign(image, restored, {
          id: image.id,
          project: image.project,
          width: image.width,
          height: image.height,
          filename: image.filename,
          analysis: null,
          pose: null,
        });
      }
    }
    s.revision++;
    await st.locked(() => st.save(s));
    await fs.unlink(await contained(target, path.join(target, '.studio-materializing.json')));
    progress.phase = 'complete';
    await atomic(receipt, stableJSON(progress));
    return { name, id, phase: 'complete', missingCaptions: progress.missingCaptions };
  });
}
