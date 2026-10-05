import { ActionLane } from './actionLane';
export const galleryCategories = ['face', 'body', 'variety', 'unclassified'] as const;
export type GalleryCategory = typeof galleryCategories[number];
export const categoryLabels: Record<GalleryCategory, string> = { face: 'Volti', body: 'Corpo', variety: 'Altre foto', unclassified: 'Da analizzare' };
export type MembershipFilter = 'all' | 'included' | 'excluded';
export type GalleryFilters = { search: string; membership: MembershipFilter; categories: readonly string[]; minQuality: number };
type GalleryImage = { id: string; filename: string; caption: string; excluded: number; discarded?: number; category: string; analysis?: { quality: number } | null };
export function galleryVisible<T extends GalleryImage>(images: T[], drafts: Record<string, {caption: string}>, filters: GalleryFilters): T[] {
  return images.filter(x => !x.discarded &&
    (x.filename + ' ' + (drafts[x.id]?.caption ?? x.caption)).toLowerCase().includes(filters.search.toLowerCase()) &&
    (filters.membership === 'all' || (filters.membership === 'included' ? !x.excluded : !!x.excluded)) &&
    filters.categories.includes(x.category) && (!filters.minQuality || !!x.analysis && x.analysis.quality >= filters.minQuality));
}
export function gallerySections<T extends GalleryImage>(images: T[]) {
  return [0, 1].flatMap(excluded => galleryCategories.flatMap(category => {
    const items = images.filter(x => Number(!!x.excluded) === excluded && x.category === category);
    return items.length ? [{ excluded: excluded as 0 | 1, category, items }] : [];
  }));
}
export function imageOpacity(excluded: number, membership: MembershipFilter) {
  return excluded && membership === 'all' ? 0.55 : 1;
}
export function membershipFailure(error: any) {
  const reason = String(error?.response?.data?.error ?? error?.message ?? '');
  if (/caption/i.test(reason)) return 'Caption aggiornata o non valida. Ricarica il dataset prima di salvare la selezione.';
  if (error?.response?.status === 409 || /changed|disappeared|stale/i.test(reason))
    return 'Dataset aggiornato altrove. Verifica e salva la selezione.';
  if (/symlink|symbolic|contain|path|regular|64KB/i.test(reason))
    return 'File non valido. Controlla il dataset e salva di nuovo la selezione.';
  return 'Selezione non confermata. Controlla la connessione e salva le modifiche.';
}
type MembershipState = { revision: number; images: {id: string; excluded: number}[] };
type Intent = { id: string; excluded: 0 | 1; sequence: number };
// One page-local owner. Late confirmed responses never erase newer local intent.
export class MembershipController<S extends MembershipState> {
  private desired = new Map<string, Intent>();
  private sequence = 0;
  private work?: Promise<unknown>;
  private uncertainRevision?: number;
  failed: unknown = null;
  constructor(private options: {
    lane: ActionLane; key: string; valid: () => boolean; current: () => S;
    post: (ids: string[], excluded: 0 | 1) => Promise<S>;
    reconcile: () => Promise<S>; changed: () => void;
  }) {}
  has(id: string) { return this.desired.has(id); }
  get pending() { return this.desired.size; }
  get saving() { return !!this.work; }
  overlay<T extends { id: string; excluded: number }>(images: T[]): T[] {
    return images.map(x => {
      const intent = this.desired.get(x.id);
      return intent && x.excluded !== intent.excluded ? { ...x, excluded: intent.excluded } : x;
    });
  }
  set(ids: string[], excluded: 0 | 1) {
    if (!this.options.valid()) return;
    for (const id of ids) this.desired.set(id, {id, excluded, sequence: ++this.sequence});
    this.options.changed();
    if (!this.failed) void this.flush(false);
  }
  retry() { return this.flush(true); }
  settled() { return this.work ?? Promise.resolve(); }
  private acknowledge(sent: Intent[]) {
    for (const intent of sent) if (this.desired.get(intent.id)?.sequence === intent.sequence) this.desired.delete(intent.id);
  }
  private flush(reconcile: boolean): Promise<unknown> {
    if (this.work) return this.work;
    if (!this.pending || !this.options.valid() || this.failed && !reconcile) return Promise.resolve();
    this.work = this.options.lane.run(this.options.key, this.options.valid, async () => {
      if (reconcile) {
        const state = await this.options.reconcile();
        if (!this.options.valid()) return;
        // Same revision cannot prove an old uncertain POST finished. Keep intent
        // and write a guarded CAS even if membership already looks identical.
        if (this.uncertainRevision !== undefined && state.revision > this.uncertainRevision) {
          for (const intent of this.desired.values())
            if (state.images.some(x => x.id === intent.id && x.excluded === intent.excluded)) this.desired.delete(intent.id);
        }
        this.failed = null;
      }
      while (this.pending && this.options.valid()) {
        for (const excluded of [0, 1] as const) {
          const sent = [...this.desired.values()].filter(x => x.excluded === excluded);
          if (!sent.length || !this.options.valid()) continue;
          this.uncertainRevision = this.options.current().revision;
          await this.options.post(sent.map(x => x.id), excluded);
          if (!this.options.valid()) return;
          this.acknowledge(sent);
          this.uncertainRevision = undefined;
          this.options.changed();
        }
      }
    }).catch(error => {
      if (this.options.valid()) { this.failed = error; this.options.changed(); }
    }).finally(() => {
      this.work = undefined;
      if (this.options.valid()) {
        this.options.changed();
        if (this.pending && !this.failed) void this.flush(false);
      }
    });
    return this.work;
  }
}

type DraftBase = { revision: number; draftRevision: number; persisted?: string; caption: string };
type CaptionImage = { id: string; revision: number; caption: string; captionRevision?: number; captionDraftRevision?: number;
  captionDraft?: {caption: string; baseRevision: number; revision: number} };
// A review response can advance a page-owned fresh draft, never turn an old
// caption base or another editor's protected token into permission to overwrite.
export function rebaseGalleryDrafts<D extends DraftBase>(drafts: Record<string,D>, ids: readonly string[], before: readonly CaptionImage[], after: readonly CaptionImage[], membershipOnly: boolean): Record<string,D> {
  return Object.fromEntries(Object.entries(drafts).map(([id,draft]) => {
    const prior = before.find(x => x.id === id), image = after.find(x => x.id === id);
    if (!ids.includes(id) || !prior || !image || draft.revision !== prior.revision ||
        draft.draftRevision !== (prior.captionDraftRevision ?? 0) || image.revision !== prior.revision + 1 ||
        image.caption !== prior.caption || (image.captionRevision ?? 0) !== (prior.captionRevision ?? 0)) return [id,draft];
    const oldToken = prior.captionDraftRevision ?? 0, token = image.captionDraftRevision ?? 0,
      oldProtected = prior.captionDraft, protectedDraft = image.captionDraft;
    const unchangedToken = token === oldToken && (!oldProtected && !protectedDraft ||
      !!oldProtected && !!protectedDraft && protectedDraft.caption === oldProtected.caption &&
      protectedDraft.baseRevision === oldProtected.baseRevision && protectedDraft.revision === oldProtected.revision);
    const ownAtomicRebase = membershipOnly && !!oldProtected && !!protectedDraft &&
      oldProtected.baseRevision === prior.revision && oldProtected.revision === oldToken &&
      token === oldToken + 1 && protectedDraft.revision === token &&
      protectedDraft.baseRevision === image.revision && protectedDraft.caption === oldProtected.caption;
    return [id, unchangedToken || ownAtomicRebase ? {...draft, revision:image.revision, draftRevision:token, persisted:undefined} : draft];
  }));
}
