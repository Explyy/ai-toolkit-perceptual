export type CaptionSource = { id: string; caption: string; revision: number };
export type CaptionTextPreview = {
  sources: CaptionSource[];
  find: string;
  replacement: string;
  changes: (CaptionSource & { next: string; occurrences: number })[];
  occurrences: number;
};
export function previewCaptionText(sources: CaptionSource[], find: string, replacement: string): CaptionTextPreview {
  const changes = find
    ? sources.flatMap(source => {
        const pieces = source.caption.split(find),
          occurrences = pieces.length - 1;
        return occurrences ? [{ ...source, next: pieces.join(replacement), occurrences }] : [];
      })
    : [];
  return {
    sources: sources.map(x => ({ ...x })),
    find,
    replacement,
    changes,
    occurrences: changes.reduce((total, x) => total + x.occurrences, 0),
  };
}
export function captionPreviewCurrent(
  preview: CaptionTextPreview,
  sources: CaptionSource[],
  find: string,
  replacement: string,
): boolean {
  return (
    !!find &&
    preview.find === find &&
    preview.replacement === replacement &&
    preview.sources.length === sources.length &&
    preview.sources.every(
      (x, i) => x.id === sources[i].id && x.caption === sources[i].caption && x.revision === sources[i].revision,
    )
  );
}

export function captionTextSources(
  images: (CaptionSource & { filename: string; excluded?: number; discarded?: number })[],
  drafts: Record<string, { caption: string }>,
  scope: 'selected' | 'visible',
  search: string,
): CaptionSource[] {
  return images
    .filter(x => !x.discarded)
    .map(x => ({ ...x, caption: drafts[x.id]?.caption ?? x.caption }))
    .filter(x =>
      scope === 'selected' ? !x.excluded : (x.filename + ' ' + x.caption).toLowerCase().includes(search.toLowerCase()),
    )
    .map(({ id, caption, revision }) => ({ id, caption, revision }));
}

export type CaptionSaveDraft = { caption: string; revision: number; draftRevision: number; persisted?: string };
// Final edit only clears the matching protected draft; protect this exact save snapshot first.
export async function saveCaptionSnapshot<T extends { images: { id: string; captionDraftRevision?: number }[] }>(
  id: string,
  draft: CaptionSaveDraft,
  request: (action: string, payload: any) => Promise<T>,
  protectedSnapshot: (draftRevision: number) => void,
): Promise<T> {
  const protectedState = await request('captionDraft', {
    id,
    draft: { caption: draft.caption, baseRevision: draft.revision },
    draftRevision: draft.draftRevision,
  });
  const image = protectedState.images.find(x => x.id === id);
  if (!image) throw new Error('Immagine cambiata: bozza conservata');
  protectedSnapshot(image.captionDraftRevision ?? 0);
  return request('edit', { ids: [id], patch: { caption: draft.caption, baseRevision: draft.revision } });
}
export function captionDraftAfterSave(
  current: CaptionSaveDraft | undefined,
  sent: CaptionSaveDraft,
  image: { revision: number; captionDraftRevision?: number },
): CaptionSaveDraft | undefined {
  if (!current || current.caption === sent.caption) return undefined;
  return { ...current, revision: image.revision, draftRevision: image.captionDraftRevision ?? 0, persisted: undefined };
}
