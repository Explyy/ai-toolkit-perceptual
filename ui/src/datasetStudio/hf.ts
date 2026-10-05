// Ported from approved Dataset Studio v1; native file storage is separate.
import { ensure, repoId, path, revision, sha256, stableJSON, Problem } from './domain';
import { boundedBytes } from './images';
const base64 = (bytes: Uint8Array) => Buffer.from(bytes).toString('base64');
const ORIGIN = 'https://huggingface.co';
export type HfFile = {
  path: string;
  size: number;
  oid: string;
  lfs?: { oid: string; size: number };
};
export class Hub {
  constructor(
    private token: string | undefined,
    private transport: typeof fetch = fetch,
  ) {
    ensure(token, 'Connect Hugging Face on the server first', 503);
  }
  async request(endpoint: string, init: RequestInit = {}) {
    ensure(endpoint.startsWith('/') && !endpoint.startsWith('//'), 'Invalid Hub endpoint');
    const r = await this.transport(ORIGIN + endpoint, {
      ...init,
      redirect: 'error',
      headers: { Authorization: 'Bearer ' + this.token, ...init.headers },
      signal: AbortSignal.timeout(30000),
    });
    if (!r.ok)
      throw new Problem(
        r.status === 409 || r.status === 412 ? 409 : 502,
        r.status === 409 || r.status === 412
          ? 'Hugging Face changed since this export started. Reconcile before retrying.'
          : 'Hugging Face request failed (' + r.status + ').',
      );
    return r;
  }
  async info(repo: string, rev?: string) {
    repoId(repo);
    if (rev) revision(rev);
    const r = await this.request(`/api/datasets/${repo}${rev ? '/revision/' + rev : ''}`);
    const x = (await r.json()) as { sha: string; private: boolean };
    ensure(x.private === true, 'Choose a private Hugging Face dataset', 403);
    revision(x.sha);
    return x;
  }
  async tree(repo: string, rev: string, folder: string) {
    repoId(repo);
    revision(rev);
    path(folder);
    const r = await this.request(`/api/datasets/${repo}/tree/${rev}/${folder}?recursive=false&limit=1000`);
    ensure(!r.headers.get('link')?.includes('rel="next"'), 'Folder has too many files; split it into direct folders');
    const rows = (await r.json()) as Array<HfFile & { type: string }>;
    return rows.filter(x => x.type === 'file');
  }
  async metadata(repo: string, rev: string, remotePath: string) {
    repoId(repo);
    revision(rev);
    path(remotePath);
    const r = await this.request(`/api/datasets/${repo}/paths-info/${rev}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paths: [remotePath], expand: false }),
    });
    const rows = (await r.json()) as HfFile[];
    ensure(rows.length === 1 && rows[0].path === remotePath, 'Remote file not found', 404);
    return rows[0];
  }
  async download(repo: string, rev: string, remotePath: string, max?: number) {
    repoId(repo);
    revision(rev);
    path(remotePath);
    const m = await this.metadata(repo, rev, remotePath);
    ensure(m.size <= (max ?? 24 * 1024 * 1024), 'Remote file exceeds the import limit', 413);
    let url = `${ORIGIN}/datasets/${repo}/resolve/${rev}/${remotePath}`,
      r: Response | undefined;
    for (let hop = 0; hop < 6; hop++) {
      const u = new URL(url);
      ensure(
        u.protocol === 'https:' &&
          (u.hostname === 'huggingface.co' ||
            u.hostname.endsWith('.hf.co') ||
            u.hostname.endsWith('.amazonaws.com') ||
            u.hostname.endsWith('.xethub.hf.co')),
        'Untrusted Hub redirect',
        502,
      );
      r = await this.transport(url, {
        redirect: 'manual',
        headers: u.origin === ORIGIN ? { Authorization: 'Bearer ' + this.token } : {},
        signal: AbortSignal.timeout(45000),
      });
      if (![301, 302, 303, 307, 308].includes(r.status)) break;
      const location = r.headers.get('location');
      ensure(location, 'Invalid Hub redirect', 502);
      url = new URL(location, url).toString();
    }
    ensure(r, 'No Hub response', 502);
    const bytes = await boundedBytes(r, max);
    ensure(bytes.length === m.size, 'Remote size mismatch', 502);
    if (m.lfs) ensure((await sha256(bytes)) === m.lfs.oid, 'Remote SHA-256 mismatch', 502);
    else {
      const prefix = new TextEncoder().encode(`blob ${bytes.length}\0`),
        blob = new Uint8Array(prefix.length + bytes.length);
      blob.set(prefix);
      blob.set(bytes, prefix.length);
      const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-1', blob))]
        .map(x => x.toString(16).padStart(2, '0'))
        .join('');
      ensure(hash === m.oid, 'Remote Git object mismatch', 502);
    }
    return bytes;
  }
  async upload(repo: string, remotePath: string, bytes: Uint8Array) {
    repoId(repo);
    path(remotePath);
    const sha = await sha256(bytes);
    const p = await this.request(`/api/datasets/${repo}/preupload/main`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        files: [
          {
            path: remotePath,
            sample: base64(bytes.subarray(0, 512)),
            size: bytes.length,
          },
        ],
      }),
    });
    const info = (await p.json()) as {
      files: Array<{ path: string; uploadMode: string; shouldIgnore: boolean }>;
    };
    ensure(
      info.files.length === 1 && info.files[0].path === remotePath && !info.files[0].shouldIgnore,
      'Hub refused export file',
      502,
    );
    if (info.files[0].uploadMode === 'regular')
      return {
        key: 'file',
        value: { path: remotePath, encoding: 'base64', content: base64(bytes) },
      };
    ensure(info.files[0].uploadMode === 'lfs', 'Unsupported Hub upload mode', 502);
    const batch = await this.request(`/datasets/${repo}.git/info/lfs/objects/batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/vnd.git-lfs+json',
        Accept: 'application/vnd.git-lfs+json',
      },
      body: JSON.stringify({
        operation: 'upload',
        transfers: ['basic'],
        objects: [{ oid: sha, size: bytes.length }],
        hash_algo: 'sha256',
        ref: { name: 'main' },
      }),
    });
    const b = (await batch.json()) as {
      objects: Array<{
        oid: string;
        size: number;
        error?: unknown;
        actions?: {
          upload: { href: string; header?: Record<string, string> };
          verify?: { href: string };
        };
      }>;
    };
    const object = b.objects[0];
    ensure(
      b.objects.length === 1 && object.oid === sha && object.size === bytes.length && !object.error,
      'LFS upload rejected',
      502,
    );
    if (object.actions) {
      const u = new URL(object.actions.upload.href);
      ensure(
        u.protocol === 'https:' &&
          (u.hostname === 'huggingface.co' ||
            u.hostname.endsWith('.amazonaws.com') ||
            u.hostname.endsWith('.hf.co') ||
            u.hostname.endsWith('.xethub.hf.co')) &&
          !u.username &&
          !u.password,
        'Untrusted upload destination',
        502,
      );
      const uploaded = await this.transport(u.toString(), {
        method: 'PUT',
        body: new Uint8Array(bytes),
        redirect: 'error',
        signal: AbortSignal.timeout(60000),
      });
      ensure(uploaded.ok, 'LFS storage upload failed', 502);
      if (object.actions.verify) {
        const v = new URL(object.actions.verify.href);
        ensure(v.origin === ORIGIN, 'Untrusted verification destination');
        await this.request(v.pathname + v.search, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ oid: sha, size: bytes.length }),
        });
      }
    }
    return {
      key: 'lfsFile',
      value: { path: remotePath, algo: 'sha256', oid: sha, size: bytes.length },
    };
  }
  async commit(repo: string, parent: string, operations: unknown[], digest: string) {
    repoId(repo);
    revision(parent);
    const lines =
      [
        {
          key: 'header',
          value: {
            summary: 'Dataset Studio ' + digest.slice(0, 12),
            parentCommit: parent,
          },
        },
        ...operations,
      ]
        .map(stableJSON)
        .join('\n') + '\n';
    const r = await this.request(`/api/datasets/${repo}/commit/main`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-ndjson' },
      body: lines,
    });
    const x = (await r.json()) as { commitOid: string };
    return revision(x.commitOid);
  }
}
export function directPairs(files: HfFile[], folder: string) {
  path(folder);
  const map = new Map(files.map(x => [x.path, x])),
    pairs: Array<{ image: HfFile; caption: HfFile }> = [];
  for (const f of files) {
    ensure(
      f.path.startsWith(folder + '/') && !f.path.slice(folder.length + 1).includes('/'),
      'Only direct files are accepted',
    );
    if (/\.(png|jpe?g|webp)$/i.test(f.path)) {
      const name = f.path.replace(/\.[^.]+$/, '.txt'),
        caption = map.get(name);
      ensure(caption, 'Missing caption: ' + name);
      pairs.push({ image: f, caption });
    } else ensure(f.path.endsWith('.txt'), 'Unexpected source file: ' + f.path);
  }
  ensure(
    pairs.length > 0 && pairs.length <= 1000 && pairs.length * 2 === files.length,
    'Folder must contain image/TXT pairs only',
  );
  return pairs.sort((a, b) => a.image.path.localeCompare(b.image.path));
}
