import fs from 'node:fs/promises';
import path from 'node:path';
import type { StudioStore, State } from './store';
import { atomic, contained, hash } from './store';
import { ensure, stableJSON, settings, repoId, revision, integer, text, CATEGORIES } from './domain';
import { preferences } from './captionModels';
import { Hub } from './hf';
import { ownedLock, OperationBusy } from './ownerLock';
const LIMIT = 16 * 1024 * 1024,
  DEBOUNCE = 3000;
export type ManagedProjection = {
  schema: 1;
  key: string;
  dataset: string;
  settings: State['settings'];
  captionPreferences?: State['captionPreferences'];
  images: Array<any>;
};
export type Outbox = {
  schema: 1;
  repo: string;
  key: string;
  desired: string;
  changedAt: number;
  baseDigest?: string;
  remoteRevision?: string;
  phase: 'pending' | 'uploading' | 'commit_started' | 'verifying' | 'synced' | 'error' | 'conflict';
  reason?: string;
  retryAt?: number;
  lastRemoteFailure?: { status: number; at: number; retryAt: number };
  metadataRecoveries?: Record<
    string,
    {
      digest: string;
      parent: string;
      baseDigest?: string;
      attemptedAt: number;
      originalParent?: string;
      commits?: string[];
      proofDigest?: string;
    }
  >;
  blobRecoveries?: Record<
    string,
    {
      original: NonNullable<NonNullable<Outbox['active']>['blob']>;
      parent: string;
      commits: string[];
      proofDigest: string;
      attemptedAt: number;
      retryableReject?: boolean;
      attempts?: Array<{
        at: number;
        parent: string;
        proofDigest: string;
        status?: number;
        responseAt?: number;
        retryAt?: number;
      }>;
      revision?: string;
    }
  >;
  verifiedObjects?: Record<string, { size: number; oid: string }>;
  active?: {
    digest: string;
    projection: ManagedProjection;
    parent: string;
    baseDigest?: string;
    blob?: { path: string; sha: string; size: number; parent: string; phase: 'uploading' | 'commit_started' };
    revision?: string;
  };
};
const definiteReject = (e: any) => [400, 401, 403, 404, 409, 412, 413, 422, 429].includes(e.remoteStatus);
const root = (key: string) => 'dataset-studio/managed/' + key;
export function managedProjection(s: State): ManagedProjection {
  const key = s.managedBinding?.key ?? hash(s.dataset);
  ensure(/^[0-9a-f]{64}$/.test(key), 'Invalid managed identity');
  const value: ManagedProjection = {
    schema: 1,
    key,
    dataset: s.managedBinding?.title ?? s.dataset,
    settings: {
      ...structuredClone(s.settings),
      // Legacy caption settings may contain a host-only model path. Keep it
      // locally, but never publish absolute runtime paths as managed metadata.
      model: path.isAbsolute(s.settings.model) || /^[a-zA-Z]:[\\/]/.test(s.settings.model) ? '' : s.settings.model,
    },
    captionPreferences: s.captionPreferences,
    images: s.images.map(x => ({
      relative: x.relative,
      sha: x.sha,
      size: x.size,
      mime: x.mime,
      caption: x.caption,
      captionOverride: x.captionOverride,
      caption_source: x.caption_source,
      category: x.category,
      categorySource: x.categorySource,
      tags: x.tags,
      pinned: x.pinned,
      excluded: x.excluded,
      discarded: x.discarded,
      reviewRevision: x.reviewRevision ?? 0,
      revision: x.revision,
      captionRevision: x.captionRevision ?? 0,
      captionDraft: x.captionDraft,
      captionDraftRevision: x.captionDraftRevision ?? 0,
    })),
  };
  ensure(Buffer.byteLength(stableJSON(value)) <= LIMIT, 'Dataset metadata exceeds16MB sync limit', 413);
  return value;
}
async function outboxFile(st: StudioStore) {
  return contained(st.folder, path.join(st.folder, 'managed-outbox.json'), true);
}
export async function readOutbox(st: StudioStore): Promise<Outbox | undefined> {
  try {
    const f = await outboxFile(st);
    ensure((await fs.stat(f)).size <= LIMIT * 2 + 100000, 'Outbox too large');
    return JSON.parse(await fs.readFile(f, 'utf8'));
  } catch (e: any) {
    if (e.code === 'ENOENT') return undefined;
    throw e;
  }
}
async function save(st: StudioStore, o: Outbox) {
  ensure(Buffer.byteLength(stableJSON(o)) <= LIMIT * 2 + 100000, 'Outbox recovery evidence exceeds limit', 413);
  await atomic(await outboxFile(st), stableJSON(o));
}
// Invoked after local state durability, under the dataset lock. Metadata only;
// it never waits on the network or loses a newer local edit during an upload.
export async function recordLocal(st: StudioStore, s: State, now = Date.now()) {
  s = await st.raw(); // caller may have captured a view before another local edit
  if (!s.images.length) return;
  const projection = managedProjection(s),
    digest = hash(stableJSON(projection));
  let o = await readOutbox(st);
  if (o && (o.key !== projection.key || o.repo !== (s.managedBinding?.repo ?? 'daverave/Personal'))) {
    ensure(!o.active && !o.baseDigest, 'Managed binding changed during sync', 409);
    o = undefined;
  }
  if (o?.desired === digest) return;
  const next: Outbox = o ?? {
    schema: 1,
    repo: s.managedBinding?.repo ?? 'daverave/Personal',
    key: projection.key,
    desired: digest,
    changedAt: now,
    baseDigest: s.managedBinding?.baseDigest,
    phase: 'pending',
  };
  next.desired = digest;
  next.changedAt = now;
  if (!next.active && next.phase !== 'conflict') next.phase = 'pending';
  await save(st, next);
}
export async function syncStatus(st: StudioStore, s: State, configured: boolean) {
  await st.locked(() => recordLocal(st, s));
  const o = await readOutbox(st);
  return {
    phase: configured ? (o?.phase ?? 'synced') : 'local',
    pending: !!o && o.desired !== o.baseDigest,
    reason: o?.reason,
    revision: o?.remoteRevision,
  };
}
async function pointer(hub: Hub, repo: string, rev: string, key: string) {
  const bytes = await hub.optional(repo, rev, root(key) + '/current.json', 2000);
  if (!bytes) return undefined;
  const x = JSON.parse(Buffer.from(bytes).toString());
  ensure(x.schema === 1 && x.key === key && /^[0-9a-f]{64}$/.test(x.digest), 'Invalid managed pointer', 409);
  return x as { digest: string };
}
async function verifyBlob(hub: Hub, o: Outbox, rev: string, image: { sha: string; size: number }) {
  const file = root(o.key) + '/blobs/' + image.sha;
  let metadata;
  try {
    metadata = await hub.metadata(o.repo, rev, file);
  } catch (e: any) {
    if (e.remoteStatus === 404 || e.status === 404) return undefined;
    throw e;
  }
  const oid = metadata.lfs?.oid ?? metadata.oid,
    known = o.verifiedObjects?.[image.sha];
  ensure(metadata.size === image.size, 'Managed original size changed', 409);
  if (known?.size === image.size && known.oid === oid) return known;
  const bytes = await hub.download(o.repo, rev, file, image.size + 1);
  ensure(bytes.length === image.size && hash(bytes) === image.sha, 'Managed original readback differs', 409);
  return { size: image.size, oid };
}
async function readback(hub: Hub, o: Outbox, rev: string) {
  const a = o.active!,
    p = await pointer(hub, o.repo, rev, o.key);
  ensure(p?.digest === a.digest, 'Managed pointer readback differs', 409);
  const bytes = await hub.download(o.repo, rev, root(o.key) + '/versions/' + a.digest + '.json', LIMIT);
  ensure(hash(bytes) === a.digest, 'Managed metadata readback differs', 502);
  for (const image of a.projection.images) {
    ensure(await verifyBlob(hub, o, rev, image), 'Managed original missing', 502);
  }
}
// The service claim is separate from the edit lock and survives process crashes
// through SQLite locking. No browser lifetime or JS-only singleton is trusted.
export async function drainManaged(st: StudioStore, hub: Hub, clock: number | (() => number) = Date.now) {
  const observeTime = typeof clock === 'function' ? clock : () => clock,
    now = observeTime();
  const observedFailures = new WeakMap<object, number>();
  const responseTime = (e: any) => {
    if (!e || typeof e !== 'object') return observeTime();
    const at = observedFailures.get(e) ?? observeTime();
    observedFailures.set(e, at);
    return at;
  };
  const deadline = (e: any, at: number) =>
    at +
    Math.max(
      1000,
      Number.isFinite(e.retryAfterAt)
        ? e.retryAfterAt - at
        : (e.retryAfterMs ?? (e.remoteStatus === 409 || e.remoteStatus === 412 ? 5000 : 60000)),
    );
  const seed = await st.raw(),
    claim = await contained(
      st.root,
      path.join(
        st.root,
        'managed-service-' +
          hash((seed.managedBinding?.repo ?? 'daverave/Personal') + ':' + (seed.managedBinding?.key ?? hash(st.name))),
      ),
      true,
    );
  await fs.mkdir(claim, { recursive: true });
  return ownedLock(claim, async () => {
    const state = await st.read();
    await st.locked(() => recordLocal(st, state, now));
    let o = await readOutbox(st);
    const legacyBusy = o?.phase === 'conflict' && o.reason === 'Dataset operation active; refresh' && !!o.active;
    if (
      !o ||
      (o.retryAt ?? 0) > now ||
      (o.phase === 'conflict' && !legacyBusy) ||
      (!o.active && (o.desired === o.baseDigest || now - o.changedAt < DEBOUNCE))
    )
      return;
    const update = async (fn: (x: Outbox) => void | Promise<void>) =>
      st.locked(async () => {
        const latest = (await readOutbox(st))!;
        await fn(latest);
        await save(st, latest);
        o = latest;
      });
    try {
      const info = await hub.owned(o.repo);
      const validateActive = () => {
        const a = o!.active!;
        validateProjection(a.projection);
        ensure(
          hash(stableJSON(a.projection)) === a.digest &&
            a.projection.key === o!.key &&
            a.projection.key === managedProjection(state).key &&
            o!.repo === (state.managedBinding?.repo ?? 'daverave/Personal') &&
            a.baseDigest === o!.baseDigest &&
            /^[0-9a-f]{40}$/.test(a.parent),
          'Managed recovery identity differs',
          409,
        );
      };
      const verifyRetained = async (rev?: string) => {
        for (const image of o!.active!.projection.images) {
          const source = state.images.find(
            x => x.relative === image.relative && x.sha === image.sha && x.size === image.size,
          );
          ensure(source, 'Managed recovery original changed', 409);
          await st.source(source);
          if (rev) ensure(await verifyBlob(hub, o!, rev, image), 'Managed recovery original missing', 409);
        }
      };
      if (legacyBusy) {
        // The old untyped busy status lost the prior phase. It is never proof
        // that a metadata POST was not sent: retain the operation as uncertain.
        validateActive();
        const p = await pointer(hub, o.repo, info.sha, o.key);
        ensure(
          p?.digest === o.active!.digest || p?.digest === o.active!.baseDigest,
          'Remote dataset divergence; both versions preserved',
          409,
        );
        const blob = o.active!.blob;
        if (blob) {
          ensure(
            !o.active!.revision &&
              ['uploading', 'commit_started'].includes(blob.phase) &&
              /^[0-9a-f]{40}$/.test(blob.parent) &&
              /^[0-9a-f]{64}$/.test(blob.sha) &&
              Number.isSafeInteger(blob.size) &&
              blob.size > 0 &&
              blob.path === root(o.key) + '/blobs/' + blob.sha &&
              o.active!.projection.images.some(x => x.sha === blob.sha && x.size === blob.size),
            'Managed recovery blob identity differs',
            409,
          );
        }
        // A pending original can precede the other uploads. Its durable unknown
        // receipt goes through the existing blob readback/history path below.
        await verifyRetained(blob ? undefined : info.sha);
        await update(x => {
          x.phase = blob && p?.digest !== o!.active!.digest ? 'uploading' : 'commit_started';
          x.reason = 'Esito HF da verificare; curatela locale conservata.';
        });
      }
      if (o.active && ['commit_started', 'verifying'].includes(o.phase)) {
        const p = await pointer(hub, o.repo, info.sha, o.key);
        if (p?.digest === o.active.digest) {
          await readback(hub, o, o.active.revision ?? info.sha);
          const digest = o.active.digest;
          await update(x => {
            x.baseDigest = digest;
            x.remoteRevision = info.sha;
            delete x.active;
            delete x.reason;
            x.phase = x.desired === digest ? 'synced' : 'pending';
          });
          return;
        }
        if (o.phase === 'commit_started') {
          const active = o.active;
          ensure(p?.digest === active.baseDigest, 'Remote dataset divergence; both versions preserved', 409);
          validateActive();
          const attempts = o.metadataRecoveries ?? {};
          ensure(typeof attempts === 'object' && !Array.isArray(attempts), 'Invalid metadata recovery receipt', 409);
          for (const [key, attempt] of Object.entries(attempts)) {
            ensure(
              attempt &&
                /^[0-9a-f]{64}$/.test(attempt.digest) &&
                /^[0-9a-f]{40}$/.test(attempt.parent) &&
                key === hash(attempt.parent + ':' + attempt.digest) &&
                (attempt.baseDigest === undefined || /^[0-9a-f]{64}$/.test(attempt.baseDigest)) &&
                Number.isSafeInteger(attempt.attemptedAt) &&
                attempt.attemptedAt >= 0,
              'Invalid metadata recovery receipt',
              409,
            );
            if (
              attempt.originalParent !== undefined ||
              attempt.commits !== undefined ||
              attempt.proofDigest !== undefined
            ) {
              ensure(
                /^[0-9a-f]{40}$/.test(attempt.originalParent ?? '') &&
                  Array.isArray(attempt.commits) &&
                  attempt.commits.length > 0 &&
                  attempt.commits.length <= 200 &&
                  new Set(attempt.commits).size === attempt.commits.length &&
                  attempt.commits.every(x => /^[0-9a-f]{40}$/.test(x)) &&
                  attempt.commits[0] === attempt.parent &&
                  attempt.commits.at(-1) === attempt.originalParent &&
                  attempt.proofDigest ===
                    hash(
                      stableJSON({
                        digest: attempt.digest,
                        originalParent: attempt.originalParent,
                        parent: attempt.parent,
                        baseDigest: attempt.baseDigest,
                        commits: attempt.commits,
                      }),
                    ),
                'Invalid metadata historical proof',
                409,
              );
            }
          }
          // One exact-parent replay is safe even while the original request is
          // still in flight: parentCommit allows at most one of them to apply.
          // A proven descendant can fence the old POST, but only complete
          // historical absence and unchanged dataset base permit one new CAS.
          // Any previous recovery for this digest stays readback-only forever.
          if (
            !active.blob &&
            !active.revision &&
            !Object.values(o.metadataRecoveries ?? {}).some(x => x.digest === active.digest)
          ) {
            const commits =
              info.sha === active.parent ? [info.sha] : await hub.historyTo(o.repo, info.sha, active.parent);
            for (const rev of commits) {
              ensure(
                !(await hub.optional(o.repo, rev, root(o.key) + '/versions/' + active.digest + '.json', LIMIT)),
                'Managed metadata existed in history; both versions preserved',
                409,
              );
              ensure(
                (await pointer(hub, o.repo, rev, o.key))?.digest === active.baseDigest,
                'Managed pointer diverged in history; both versions preserved',
                409,
              );
              if (active.baseDigest) {
                const bytes = await hub.download(
                  o.repo,
                  rev,
                  root(o.key) + '/versions/' + active.baseDigest + '.json',
                  LIMIT,
                );
                ensure(hash(bytes) === active.baseDigest, 'Managed recovery base changed in history', 409);
              }
            }
            await verifyRetained(info.sha);
            const operations = [
              await hub.upload(
                o.repo,
                root(o.key) + '/versions/' + active.digest + '.json',
                Buffer.from(stableJSON(active.projection)),
              ),
              await hub.upload(
                o.repo,
                root(o.key) + '/current.json',
                Buffer.from(stableJSON({ schema: 1, key: o.key, digest: active.digest })),
              ),
            ];
            ensure(
              Buffer.byteLength(stableJSON(operations)) <= 23 * 1024 * 1024,
              'Metadata commit payload exceeds bounded limit',
              413,
            );
            const fresh = await hub.info(o.repo),
              current = await pointer(hub, o.repo, fresh.sha, o.key);
            if (fresh.sha === info.sha && current?.digest === active.baseDigest) {
              await update(async asyncGuard => {
                ensure(
                  asyncGuard.phase === 'commit_started' &&
                    stableJSON(asyncGuard.active) === stableJSON(active) &&
                    !Object.values(asyncGuard.metadataRecoveries ?? {}).some(x => x.digest === active.digest),
                  'Managed recovery receipt changed',
                  409,
                );
                const latestState = await st.raw();
                ensure(
                  managedProjection(latestState).key === asyncGuard.key &&
                    (latestState.managedBinding?.repo ?? 'daverave/Personal') === asyncGuard.repo,
                  'Managed recovery local binding changed',
                  409,
                );
                for (const image of active.projection.images) {
                  const source = latestState.images.find(
                    x => x.relative === image.relative && x.sha === image.sha && x.size === image.size,
                  );
                  ensure(source, 'Managed recovery original changed', 409);
                  await st.source(source);
                }
                (asyncGuard.metadataRecoveries ??= {})[hash(info.sha + ':' + active.digest)] = {
                  digest: active.digest,
                  parent: info.sha,
                  originalParent: active.parent,
                  commits,
                  proofDigest: hash(
                    stableJSON({
                      digest: active.digest,
                      originalParent: active.parent,
                      parent: info.sha,
                      baseDigest: active.baseDigest,
                      commits,
                    }),
                  ),
                  baseDigest: active.baseDigest,
                  attemptedAt: observeTime(),
                };
              });
              const committed = await hub.commit(o.repo, info.sha, operations, active.digest);
              await update(x => {
                x.active!.revision = committed;
                x.phase = 'verifying';
              });
              await readback(hub, o, committed);
              await update(x => {
                x.baseDigest = active.digest;
                x.remoteRevision = committed;
                delete x.active;
                delete x.reason;
                delete x.retryAt;
                x.phase = x.desired === active.digest ? 'synced' : 'pending';
              });
              return;
            }
          }
          await update(x => {
            x.reason = 'Esito HF non ancora confermato: riconciliazione in corso, nessun invio duplicato.';
          });
          return;
        }
        ensure(false, 'Managed pointer changed while verifying', 409);
      }
      if (!o.active) {
        const p = await pointer(hub, o.repo, info.sha, o.key);
        ensure(p?.digest === o.baseDigest, 'Remote dataset changed; both versions preserved', 409);
        const projection = managedProjection(state),
          digest = hash(stableJSON(projection));
        await update(x => {
          x.active = { digest, projection, parent: info.sha, baseDigest: x.baseDigest };
          x.phase = 'uploading';
          delete x.reason;
        });
      }
      const active = o.active!;
      if (active.blob?.phase === 'commit_started') {
        const blob = active.blob;
        let bytes = await hub.optional(o.repo, info.sha, blob.path, blob.size + 1);
        let verifiedRevision = info.sha;
        if (!bytes) {
          const previous = o.blobRecoveries?.[blob.sha];
          if (previous && !previous.retryableReject) {
            await update(x => {
              x.reason = 'Esito del recupero HF in attesa; nessun ulteriore invio.';
            });
            return;
          }
          const image = active.projection.images.find(x => x.sha === blob.sha && x.size === blob.size),
            source = image && state.images.find(x => x.sha === image.sha && x.relative === image.relative);
          ensure(
            image && source && blob.path === root(o.key) + '/blobs/' + blob.sha,
            'Unknown original identity differs',
            409,
          );
          ensure(
            (await pointer(hub, o.repo, info.sha, o.key))?.digest === active.baseDigest,
            'Remote dataset divergence; both versions preserved',
            409,
          );
          const sourceBytes = await st.source(source);
          ensure(
            sourceBytes.length === blob.size && hash(sourceBytes) === blob.sha,
            'Unknown original source differs',
            409,
          );
          const commits = info.sha === blob.parent ? [info.sha] : await hub.historyTo(o.repo, info.sha, blob.parent);
          for (const rev of commits) {
            if (rev === info.sha) continue;
            ensure(
              !(await hub.optional(o.repo, rev, blob.path, blob.size + 1)),
              'Original existed in history but is now missing; both versions preserved',
              409,
            );
          }
          const operation = await hub.upload(o.repo, blob.path, sourceBytes);
          ensure(
            Buffer.byteLength(stableJSON(operation)) <= 34 * 1024 * 1024,
            'Original recovery payload exceeds bound',
            413,
          );
          const fresh = await hub.info(o.repo);
          ensure(
            fresh.sha === info.sha && (await pointer(hub, o.repo, fresh.sha, o.key))?.digest === active.baseDigest,
            'HF changed during unknown-outcome proof; reconcile before recovery',
            409,
          );
          // Retain the original receipt and proof before exactly one CAS replay.
          // The old POST cannot apply after a proven descendant; at the same
          // parent, strict CAS permits at most one of original/recovery to apply.
          const proof = {
            original: structuredClone(blob),
            parent: info.sha,
            commits,
            proofDigest: hash(
              stableJSON({ blob, head: info.sha, commits, sourceSHA: blob.sha, base: active.baseDigest }),
            ),
            attemptedAt: observeTime(),
          };
          const attempts = [
            ...(previous?.attempts ?? []),
            { at: proof.attemptedAt, parent: info.sha, proofDigest: proof.proofDigest },
          ];
          await update(x => {
            (x.blobRecoveries ??= {})[blob.sha] = { ...proof, attempts };
          });
          try {
            verifiedRevision = await hub.commit(o.repo, info.sha, [operation], blob.sha);
            await update(x => {
              x.blobRecoveries![blob.sha].revision = verifiedRevision;
            });
          } catch (e: any) {
            const receivedAt = responseTime(e);
            if (e.remoteStatus === 429) {
              // Confirmed non-application does not spend the one uncertain
              // replay allowance. Retain its audit and freshly reprove safety
              // after the durable Retry-After deadline before another attempt.
              await update(x => {
                const proof = x.blobRecoveries![blob.sha];
                proof.retryableReject = true;
                const attempt = proof.attempts!.at(-1)!;
                attempt.status = 429;
                attempt.responseAt = receivedAt;
                attempt.retryAt = deadline(e, receivedAt);
              });
              throw e;
            }
            if (!definiteReject(e)) throw e;
            // A competing original may have won CAS. Adopt only exact readback;
            // no rebase/repeat of this already attempted recovery is permitted.
            const current = await hub.info(o.repo);
            bytes = await hub.optional(o.repo, current.sha, blob.path, blob.size + 1);
            verifiedRevision = current.sha;
            if (!bytes) throw e;
          }
          bytes ??= await hub.optional(o.repo, verifiedRevision, blob.path, blob.size + 1);
          ensure(bytes, 'Unknown original recovery not yet readable', 502);
        }
        ensure(bytes.length === blob.size && hash(bytes) === blob.sha, 'Unknown original commit readback differs', 409);
        const meta = await hub.metadata(o.repo, verifiedRevision, blob.path);
        await update(x => {
          (x.verifiedObjects ??= {})[blob.sha] = { size: blob.size, oid: meta.lfs?.oid ?? meta.oid };
          delete x.active!.blob;
          delete x.reason;
          delete x.retryAt;
        });
      }
      // Each original is one bounded immutable commit. Only identity/projection
      // receipts are persisted; pixel/base64 operations never enter the outbox.
      for (const image of active.projection.images) {
        const remote = root(o.key) + '/blobs/' + image.sha,
          current = await hub.info(o.repo);
        const existing = await verifyBlob(hub, o, current.sha, image);
        if (existing) {
          await update(x => {
            (x.verifiedObjects ??= {})[image.sha] = existing;
          });
          continue;
        }
        const source = state.images.find(x => x.sha === image.sha && x.relative === image.relative);
        if (!source)
          throw Object.assign(new Error('Local source changed; newer work retained'), { localSuperseded: true });
        const sourceBytes = await st.source(source).catch(e => {
          throw Object.assign(e, { localSuperseded: true });
        });
        const operation = await hub.upload(o.repo, remote, sourceBytes);
        ensure(
          Buffer.byteLength(stableJSON(operation)) <= 34 * 1024 * 1024,
          'Original commit payload exceeds bounded limit',
          413,
        );
        for (let attempt = 0; attempt < 2; attempt++) {
          const latest = await hub.info(o.repo),
            p = await pointer(hub, o.repo, latest.sha, o.key);
          ensure(p?.digest === active.baseDigest, 'Remote dataset divergence; both versions preserved', 409);
          await update(x => {
            x.active!.blob = {
              path: remote,
              sha: image.sha,
              size: image.size,
              parent: latest.sha,
              phase: 'commit_started',
            };
            x.phase = 'uploading';
          });
          let committed;
          try {
            committed = await hub.commit(o.repo, latest.sha, [operation], image.sha);
          } catch (e: any) {
            responseTime(e);
            if (e.remoteStatus === 409 || e.remoteStatus === 412) {
              await update(x => {
                x.active!.blob!.phase = 'uploading';
              });
              if (attempt === 0) continue;
            }
            if (definiteReject(e))
              await update(x => {
                x.active!.blob!.phase = 'uploading';
              });
            throw e;
          }
          const verified = await verifyBlob(hub, o, committed, image);
          ensure(verified, 'Original commit not readable', 502);
          await update(x => {
            (x.verifiedObjects ??= {})[image.sha] = verified;
            delete x.active!.blob;
          });
          break;
        }
      }
      const operations = [
        await hub.upload(
          o.repo,
          root(o.key) + '/versions/' + active.digest + '.json',
          Buffer.from(stableJSON(active.projection)),
        ),
        await hub.upload(
          o.repo,
          root(o.key) + '/current.json',
          Buffer.from(stableJSON({ schema: 1, key: o.key, digest: active.digest })),
        ),
      ];
      ensure(
        Buffer.byteLength(stableJSON(operations)) <= 23 * 1024 * 1024,
        'Metadata commit payload exceeds bounded limit',
        413,
      );
      for (let attempt = 0; attempt < 2; attempt++) {
        const current = await hub.info(o.repo),
          p = await pointer(hub, o.repo, current.sha, o.key);
        ensure(p?.digest === active.baseDigest, 'Remote dataset divergence; both versions preserved', 409);
        // The same pointer plus immutable blobs/metadata are proven at the new
        // repository parent. Other datasets/campaign commits may safely rebase.
        if (current.sha !== active.parent && active.baseDigest) {
          const baseBytes = await hub.download(
            o.repo,
            current.sha,
            root(o.key) + '/versions/' + active.baseDigest + '.json',
            LIMIT,
          );
          ensure(hash(baseBytes) === active.baseDigest, 'Managed base changed', 409);
          const base = JSON.parse(Buffer.from(baseBytes).toString());
          for (const image of base.images) {
            ensure(await verifyBlob(hub, o, current.sha, image), 'Managed base original changed', 409);
          }
        }
        await update(x => {
          x.active!.parent = current.sha;
          x.phase = 'commit_started';
        });
        let committed;
        try {
          committed = await hub.commit(o.repo, current.sha, operations, active.digest);
        } catch (e: any) {
          responseTime(e);
          if (e.remoteStatus === 409 || e.remoteStatus === 412) {
            await update(x => {
              x.phase = 'uploading';
            });
            if (attempt === 0) continue;
          }
          if (definiteReject(e))
            await update(x => {
              x.phase = 'uploading';
            });
          throw e;
        }
        await update(x => {
          x.active!.revision = committed;
          x.phase = 'verifying';
        });
        await readback(hub, o, committed);
        await update(x => {
          delete x.retryAt;
          x.baseDigest = active.digest;
          x.remoteRevision = committed;
          delete x.active;
          delete x.reason;
          x.phase = x.desired === active.digest ? 'synced' : 'pending';
        });
        break;
      }
    } catch (e: any) {
      // Contention never changes a semantic phase or erases an uncertain POST
      // receipt. A later service tick retries the same durable operation.
      if (e instanceof OperationBusy) return;
      const receivedAt = responseTime(e);
      await update(x => {
        if (Number.isInteger(e.remoteStatus)) {
          x.retryAt = deadline(e, receivedAt);
          x.lastRemoteFailure = { status: e.remoteStatus, at: receivedAt, retryAt: x.retryAt };
        }
        if (legacyBusy && x.phase === 'conflict') {
          // A failed validation/network read must not turn the legacy uncertain
          // operation into a normal uploading retry. Transient reads retain the
          // legacy marker; invalid local evidence stays an explicit conflict.
          if (!e.remoteStatus && [400, 403, 409, 413].includes(e.status)) x.reason = e.message;
          return;
        }
        if (x.phase === 'commit_started' || x.active?.blob?.phase === 'commit_started') {
          if (e.status === 409 && !e.remoteStatus) {
            x.phase = 'conflict';
            x.reason = e.message;
            return;
          }
          x.reason = e.remoteStatus
            ? 'HF (' + e.remoteStatus + '): conferma in attesa; nessun invio duplicato.'
            : 'Conferma HF in attesa; nessun reinvio automatico.';
          return;
        }
        if (e.localSuperseded && x.active && x.desired !== x.active.digest) {
          delete x.active;
          x.phase = 'pending';
          x.reason = 'Nuova revisione locale in attesa; originali già caricati immutabili conservati.';
          return;
        }
        x.phase = e.status === 409 && !e.remoteStatus ? 'conflict' : 'error';
        x.reason = e.message ?? 'Sincronizzazione temporaneamente non disponibile';
      });
    }
  });
}
export function validateProjection(value: any): ManagedProjection {
  ensure(
    value?.schema === 1 &&
      /^[0-9a-f]{64}$/.test(value.key) &&
      typeof value.dataset === 'string' &&
      Array.isArray(value.images) &&
      value.images.length > 0 &&
      value.images.length <= 2500,
    'Invalid managed dataset',
  );
  const seen = new Set<string>();
  const images = value.images.map((x: any) => {
    ensure(
      typeof x.relative === 'string' &&
        x.relative.length <= 500 &&
        !x.relative.startsWith('/') &&
        !x.relative.split('/').some((p: string) => !p || p === '.' || p === '..' || p.startsWith('.')) &&
        /\.(png|jpe?g|webp)$/i.test(x.relative) &&
        !/[\\\x00-\x1f]/.test(x.relative) &&
        !seen.has(x.relative),
      'Invalid managed image path',
    );
    seen.add(x.relative);
    ensure(/^[0-9a-f]{64}$/.test(x.sha), 'Invalid managed image SHA');
    integer(x.size, 1, 24 * 1024 * 1024);
    ensure([...CATEGORIES, 'unclassified'].includes(x.category), 'Invalid managed category');
    const row: any = {
      relative: x.relative,
      sha: x.sha,
      size: x.size,
      mime: text(x.mime, 80),
      caption: text(x.caption, 64000),
      captionOverride: !!x.captionOverride,
      caption_source: text(x.caption_source, 200),
      category: x.category,
      categorySource:
        x.categorySource === 'manual' ? 'manual' : x.categorySource === 'automatic' ? 'automatic' : undefined,
      tags: Array.isArray(x.tags) ? x.tags.map((y: any) => text(y, 60)).slice(0, 30) : [],
      pinned: integer(x.pinned, 0, 1),
      excluded: integer(x.excluded, 0, 1),
      discarded: integer(x.discarded, 0, 1),
      reviewRevision: integer(x.reviewRevision ?? 0, 0, Number.MAX_SAFE_INTEGER),
      revision: integer(x.revision ?? 0, 0, Number.MAX_SAFE_INTEGER),
      captionRevision: integer(x.captionRevision ?? 0, 0, Number.MAX_SAFE_INTEGER),
      captionDraftRevision: integer(x.captionDraftRevision ?? 0, 0, Number.MAX_SAFE_INTEGER),
    };
    if (x.captionDraft)
      row.captionDraft = {
        caption: text(x.captionDraft.caption, 64000),
        baseRevision: integer(x.captionDraft.baseRevision, 0, Number.MAX_SAFE_INTEGER),
        revision: integer(x.captionDraft.revision, 0, Number.MAX_SAFE_INTEGER),
      };
    return row;
  });
  return {
    schema: 1,
    key: value.key,
    dataset: text(value.dataset, 128),
    settings: settings(value.settings),
    captionPreferences: value.captionPreferences ? preferences(value.captionPreferences) : undefined,
    images,
  };
}
export { root as managedRoot };
