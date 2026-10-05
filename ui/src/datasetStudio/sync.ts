import fs from 'node:fs/promises';
import path from 'node:path';
import { StudioStore, atomic, hash, contained, Snapshot } from './store';
import { ensure, repoId, revision, stableJSON, Problem, integer } from './domain';
import { Hub } from './hf';
export function remoteFiles(v: Snapshot, s: any) {
  return [
    ...v.files.map(f => ({
      ...f,
      remote: f.path.startsWith('training/') ? s.folder + '/' + f.path.slice(9) : s.metadata + '/' + f.path,
    })),
    {
      path: 'manifest.json',
      remote: s.metadata + '/manifest.json',
      sha: v.digest!,
      size: Buffer.byteLength(stableJSON(v.manifest)),
    },
  ];
}
export async function sync(st: StudioStore, rev: number, input: any, token: string, transport: typeof fetch = fetch) {
  const state = await st.read();
  ensure(state.revision === rev, 'Stale sync action', 409);
  const v = state.snapshots.find(x => x.id === input.id);
  ensure(v?.state === 'complete' && v.digest, 'Completed snapshot required');
  const hub = new Hub(token, transport);
  if (input.action === 'hfStart') {
    repoId(input.repo);
    if (v.hf) {
      ensure(v.hf.repo === input.repo, 'Version already bound to another Hub repo', 409);
      return state;
    }
    const info = await hub.info(input.repo);
    return st.mutate(rev, s => {
      s.snapshots.find(x => x.id === v.id)!.hf = {
        repo: input.repo,
        parent: info.sha,
        phase: 'uploading',
        folder: 'datasets/Training_Studio_' + hash(st.name).slice(0, 12) + '_' + v.digest,
        metadata: 'dataset-studio/native/' + hash(st.name) + '/versions/' + v.digest,
        uploaded: [],
        verified: [],
      };
    });
  }
  ensure(v.hf, 'Start sync first');
  const h = structuredClone(v.hf),
    files = remoteFiles(v, h);
  const save = () =>
    st.mutate(rev, s => {
      s.snapshots.find(x => x.id === v.id)!.hf = h;
    });
  if (input.action === 'hfUpload' || input.action === 'hfVerify') {
    const i = integer(input.index, 0, files.length - 1),
      f = files[i],
      verify = input.action === 'hfVerify',
      list = verify ? h.verified : h.uploaded;
    if (list.includes(i)) return state;
    if (verify) {
      ensure(h.phase === 'verifying' && h.revision, 'Unknown commit needs reconciliation first', 409);
      const bytes = await hub.download(h.repo, h.revision, f.remote, f.size + 1);
      ensure(bytes.length === f.size && hash(bytes) === f.sha, 'HF readback differs', 502);
      h.verified.push(i);
      if (h.verified.length === files.length) h.phase = 'verified';
    } else {
      ensure(h.phase === 'uploading', 'Sync no longer accepts uploads', 409);
      const file = await st.file(v.id, f.path),
        bytes = await fs.readFile(file);
      ensure(hash(bytes) === f.sha && bytes.length === f.size, 'Local snapshot changed', 409);
      const operation = await hub.upload(h.repo, f.remote, bytes);
      const dir = path.join(st.folder, 'versions', v.id, 'hf-operations');
      await contained(st.folder, dir, true);
      await fs.mkdir(dir, { recursive: true });
      await atomic(path.join(dir, i + '.json'), stableJSON(operation));
      h.uploaded.push(i);
    }
    return save();
  }
  if (input.action === 'hfCommit') {
    ensure(h.phase === 'uploading' && h.uploaded.length === files.length, 'Finish uploads before one commit', 409);
    h.phase = 'commit_started';
    const intended = await save();
    rev = intended.revision;
    async function* chunks() {
      yield Buffer.from(
        stableJSON({
          key: 'header',
          value: { summary: 'Dataset Studio native ' + v!.digest!.slice(0, 12), parentCommit: h.parent },
        }) + '\n',
      );
      for (let i = 0; i < files.length; i++) {
        const file = await contained(st.folder, path.join(st.folder, 'versions', v!.id, 'hf-operations', i + '.json'));
        yield await fs.readFile(file);
        yield Buffer.from('\n');
      }
    }
    const iterator = chunks();
    const stream = new ReadableStream({
      async pull(controller) {
        const x = await iterator.next();
        if (x.done) controller.close();
        else controller.enqueue(x.value);
      },
      async cancel() {
        await iterator.return(undefined);
      },
    });
    try {
      const result = await hub.request('/api/datasets/' + h.repo + '/commit/main', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-ndjson' },
        body: stream,
        duplex: 'half',
      } as RequestInit);
      h.revision = revision((await result.json()).commitOid);
      h.phase = 'verifying';
      return save();
    } catch (e) {
      h.phase = e instanceof Problem && e.status === 409 ? 'conflict' : 'unknown';
      await save();
      throw e;
    }
  }
  if (input.action === 'hfReconcile') {
    ensure(['unknown', 'commit_started', 'conflict'].includes(h.phase), 'Nothing unknown to reconcile');
    const info = await hub.info(h.repo);
    const f = files.at(-1)!;
    const bytes = await hub.download(h.repo, info.sha, f.remote, f.size + 1);
    ensure(hash(bytes) === v.digest, 'Manifest not confirmed; no new commit was sent', 409);
    h.revision = info.sha;
    h.phase = 'verifying';
    return save();
  }
  throw new Problem(400, 'Invalid sync action');
}
