'use client';
import React, { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { apiClient } from '@/utils/api';
import { TopBar, MainContent } from '@/components/layout';
import AdvancedWorkspace from './AdvancedWorkspace';
import { ActionLane } from './actionLane';
import { captionModels, defaultCaptionModel, CaptionPreferences } from './captionModels';
import type { Image, State, JobLink } from './store';
import type { AnalysisProgress } from './analysisProgress';
import { selectionRectangle, selectionHits, selectionCommit, SelectionGesture } from './selectionTools';
import {
  previewCaptionText,
  captionPreviewCurrent,
  captionTextSources,
  CaptionTextPreview,
  saveCaptionSnapshot,
  captionDraftAfterSave,
} from './captionTextTools';
const button = 'rounded-lg border border-gray-600 bg-gray-800 px-4 py-2.5 disabled:opacity-40 hover:bg-gray-700';
const field = 'w-full min-w-0 rounded-lg border border-gray-600 bg-gray-950 p-3';
type View = State & {
  managedSync: {phase:string;pending:boolean;reason?:string;revision?:string};
  space: {free:number;reserve:number;missingBytes:number;outputBytes:number};
 analysisProgress: AnalysisProgress; jobsLive: any[]; preview: boolean; captionHostSupported: boolean; analysisEnabled: boolean };
type Draft = { caption: string; revision: number; draftRevision: number; persisted?: string };
const pollPhases = new Set(['prepared', 'enqueue-intent', 'active']);
function initialPreferences(s: State): CaptionPreferences {
  if (s.captionPreferences) return s.captionPreferences;
  const old = s.templateDraft,
    key = old ? JSON.stringify([old.captioner, old.captionModel]) : null;
  return {
    key: captionModels.some(x => x.key === key) ? key! : defaultCaptionModel.key,
    instructions: old?.instructions ?? s.settings.instructions,
  };
}
function PreviewImage({ dataset, image, height }: { dataset: string; image: Image; height: number }) {
  const holder = useRef<HTMLDivElement>(null),
    [url, setUrl] = useState(''),
    [failed, setFailed] = useState(false),
    [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let dead = false,
      created = '',
      started = false;
    const observer = new IntersectionObserver(
      entries => {
        if (started || !entries.some(x => x.isIntersecting)) return;
        started = true;
        observer.disconnect();
        setFailed(false);
        apiClient
          .get('/api/dataset-studio', { params: { dataset, image: image.id }, responseType: 'blob' })
          .then(r => {
            created = URL.createObjectURL(r.data);
            if (!dead) setUrl(created);
          })
          .catch(() => {
            if (!dead) setFailed(true);
          });
      },
      { rootMargin: '400px' },
    );
    if (holder.current) observer.observe(holder.current);
    return () => {
      dead = true;
      observer.disconnect();
      if (created) URL.revokeObjectURL(created);
    };
  }, [dataset, image.id, attempt]);
  return (
    <div ref={holder} style={{ height }} className="w-full flex items-center justify-center rounded bg-gray-950">
      {url ? (
        <img
          src={url}
          draggable={false}
          alt={image.filename}
          style={{
            opacity: image.excluded ? 0.35 : 1,
            width: '100%',
            height: '100%',
            maxWidth: '100%',
            maxHeight: height,
            minHeight: 0,
            objectFit: 'contain',
          }}
        />
      ) : failed ? (
        <button onClick={() => setAttempt(x => x + 1)}>Immagine non disponibile · Riprova</button>
      ) : (
        <span className="text-gray-400">Caricamento immagine…</span>
      )}
    </div>
  );
}
export default function Workspace({ dataset, onNativeView }: { dataset: string; onNativeView: () => void }) {
  const [state, setState] = useState<View | null>(null),
    [error, setError] = useState(''),
    [busy, setBusy] = useState(''),
    [search, setSearch] = useState(''),
    [target, setTarget] = useState('3'),
    [categoryFilter, setCategoryFilter] = useState('all'),
    [membershipFilter, setMembershipFilter] = useState('all'),
    [tagFilter, setTagFilter] = useState(''),
    [minQuality, setMinQuality] = useState(0),
    [tagDrafts, setTagDrafts] = useState<Record<string,string>>({}),
    [size, setSize] = useState(460),
    [advanced, setAdvanced] = useState(false),
    [advancedMounted, setAdvancedMounted] = useState(false),
    [drafts, setDrafts] = useState<Record<string, Draft>>({}),
    [saveStatus, setSaveStatus] = useState(''),
    [pref, setPref] = useState<CaptionPreferences>({ key: defaultCaptionModel.key, instructions: '' }),
    [prefStatus, setPrefStatus] = useState(''),
    [selectionMode, setSelectionMode] = useState<0 | 1>(0),
    [drag, setDrag] = useState<SelectionGesture | null>(null),
    [textToolsOpen, setTextToolsOpen] = useState(false),
    [findText, setFindText] = useState(''),
    [replacement, setReplacement] = useState(''),
    [textScope, setTextScope] = useState<'selected' | 'visible'>('selected'),
    [textPreview, setTextPreview] = useState<CaptionTextPreview | null>(null);
  const grid = useRef<HTMLDivElement>(null),
    gesture = useRef<SelectionGesture | null>(null),
    textPreviewScope = useRef<object | null>(null);
  const current = useRef<View | null>(null),
    activeDataset = useRef(dataset),
    initialized = useRef(''),
    lane = useRef(new ActionLane()),
    scope = useRef({ dataset, epoch: 0 }),
    userPending = useRef(0),
    intentSequence = useRef(0),
    draftRef = useRef(drafts),
    prefDirty = useRef(false),
    prefBase = useRef(0),
    prefError = useRef(false),
    generationId = useRef<string | null>(null),
    prefCurrent = useRef(pref),
    prefEpoch = useRef(0);
  prefCurrent.current = pref;
  activeDataset.current = dataset;
  if (scope.current.dataset !== dataset) scope.current = { dataset, epoch: scope.current.epoch + 1 };
  draftRef.current = drafts;
  function accept(s: View) {
    if (s.dataset !== activeDataset.current) return;
    current.current = s;
    setState(s);
    if (initialized.current !== dataset) {
      initialized.current = dataset;
      setTarget(String(s.settings.count));
      setPref(initialPreferences(s));
      setDrafts(
        Object.fromEntries(
          s.images
            .filter(x => x.captionDraft)
            .map(x => [
              x.id,
              {
                caption: x.captionDraft!.caption,
                revision: x.captionDraft!.baseRevision,
                draftRevision: x.captionDraftRevision ?? 0,
                persisted: x.captionDraft!.caption,
              },
            ]),
        ),
      );
      prefDirty.current = false;
      if (activeDataset.current !== dataset) return;
      prefBase.current = s.captionPreferencesRevision ?? 0;
    }
  }
  async function refresh() {
    const owner = scope.current;
    try {
      const r = await apiClient.get('/api/dataset-studio', { params: { dataset } });
      if (scope.current === owner) accept(r.data);
    } catch {
      if (scope.current !== owner) return;
      setError('Non riusciamo a leggere il dataset. Le bozze restano qui; riprova.');
    }
  }
  async function request(action: string, payload: any = {}) {
    const owner = scope.current;
    const s = current.current;
    if (!s || s.dataset !== dataset || activeDataset.current !== dataset)
      throw new Error('Dataset non caricato o cambiato');
    const r = await apiClient.post('/api/dataset-studio', { dataset, revision: s.revision, action, ...payload });
    if (scope.current !== owner) throw new Error('Dataset cambiato: risposta precedente ignorata');
    accept(r.data);
    if (action === 'edit' && payload.patch.caption === undefined && activeDataset.current === dataset) {
      setDrafts(old =>
        Object.fromEntries(
          Object.entries(old).map(([id, draft]) => {
            const image = r.data.images.find((x: Image) => x.id === id);
            return [
              id,
              payload.ids.includes(id) && image ? { ...draft, revision: image.revision, persisted: undefined } : draft,
            ];
          }),
        ),
      );
    }
    return r.data as View;
  }
  async function run(label: string, fn: () => Promise<any>, replayKey?: string) {
    const owner = scope.current;
    const valid = () => scope.current === owner && owner.dataset === dataset;
    userPending.current++;
    setBusy(label);
    setError('');
    try {
      return await lane.current.run(
        owner.epoch + ':' + dataset + ':' + (replayKey ?? 'intent-' + ++intentSequence.current),
        valid,
        fn,
      );
    } catch (e: any) {
      if (valid())
        setError(
          e.response?.data?.error ?? 'Operazione non confermata. Le caption e gli originali restano preservati.',
        );
    } finally {
      if (valid()) {
        userPending.current--;
        if (!userPending.current) setBusy('');
      }
    }
  }
  useEffect(() => {
    current.current = null;
    initialized.current = '';
    setState(null);
    setDrafts({});
    setTagDrafts({});
    cancelSelection();
    setTextPreview(null);
    setTextToolsOpen(false);
    setAdvanced(false);
    setAdvancedMounted(false);
    setError('');
    setBusy('');
    userPending.current = 0;
    prefDirty.current = false;
    prefError.current = false;
    generationId.current = null;
    draftFailure.current = false;
    void refresh();
  }, [dataset]);
  useEffect(() => {
    const timer = setInterval(async () => {
      if (lane.current.busy || !current.current) return;
      if (prefDirty.current && !prefError.current) {
        await savePreferences();
        return;
      }
      const release = lane.current.tryBackground(),
        owner = scope.current;
      if (!release) return;
      try {
        const automatic = current.current.jobs.filter(
          x =>
            x.automatic &&
            (pollPhases.has(x.automatic.phase) || (x.kind === 'analysis' && x.automatic.phase === 'blocked')),
        );
        for (const link of automatic) {
          if (scope.current !== owner) break;
          await request(link.kind === 'analysis' ? 'reconcileAnalysis' : 'reconcileCaption', {
            id: link.automatic!.id,
            blockedIds: Object.keys(draftRef.current),
          });
        }
        if (!advanced && scope.current === owner) await refresh();
      } catch {
        if (scope.current === owner)
          setError('Connessione interrotta: mostriamo gli ultimi dati confermati. Il lavoro già avviato continua sul server; aggiorniamo appena torna la connessione.');
      } finally {
        release();
      }
    }, 4000);
    return () => clearInterval(timer);
  }, [dataset, advanced]);
  useEffect(() => {
    const leave = (e: BeforeUnloadEvent) => {
      if (Object.keys(draftRef.current).length || prefDirty.current) {
        e.preventDefault();
        e.returnValue = '';
      }
    };
    window.addEventListener('beforeunload', leave);
    return () => window.removeEventListener('beforeunload', leave);
  }, []);
  async function savePreferences() {
    if (!prefDirty.current || prefError.current) return;
    const release = lane.current.tryBackground(),
      owner = scope.current;
    if (!release) return;
    setPrefStatus('Salvataggio impostazioni…');
    try {
      const epoch = prefEpoch.current;
      const s = await request('captionPreferences', {
        preferences: prefCurrent.current,
        preferencesRevision: prefBase.current,
      });
      if (scope.current !== owner) return;
      prefBase.current = s.captionPreferencesRevision ?? 0;
      prefDirty.current = prefEpoch.current !== epoch;
      setPrefStatus(prefDirty.current ? 'Modifiche da salvare' : 'Modello e istruzioni salvati');
      if (prefDirty.current) setTimeout(() => void savePreferences(), 0);
    } catch (e: any) {
      if (scope.current !== owner) return;
      prefError.current = true;
      setPrefStatus(e.response?.data?.error ?? 'Impostazioni non salvate: bozza conservata');
    } finally {
      release();
    }
  }
  useEffect(() => {
    if (!prefDirty.current || prefError.current) return;
    const timer = setTimeout(() => void savePreferences(), 700);
    return () => clearTimeout(timer);
  }, [pref.key, pref.instructions, busy]);
  function changePreferences(next: CaptionPreferences) {
    if (!prefDirty.current) prefBase.current = current.current?.captionPreferencesRevision ?? 0;
    prefDirty.current = true;
    prefEpoch.current++;
    prefError.current = false;
    setPrefStatus('Modifiche da salvare');
    setPref(next);
    generationId.current = null;
  }
  function changeCaption(image: Image, value: string) {
    const next = (old: Record<string, Draft>) => ({
      ...old,
      [image.id]: {
        caption: value,
        revision: old[image.id]?.revision ?? image.revision,
        draftRevision: old[image.id]?.draftRevision ?? image.captionDraftRevision ?? 0,
        persisted: old[image.id]?.persisted,
      },
    });
    draftRef.current = next(draftRef.current);
    setDrafts(next);
    draftFailure.current = false;
    setSaveStatus('Caption da salvare');
  }
  const draftFailure = useRef(false);
  useEffect(() => {
    if (draftFailure.current) return;
    const timer = setInterval(async () => {
      if (draftFailure.current) return;
      if (lane.current.busy) return;
      const entries = Object.entries(draftRef.current).filter(([, d]) => d.caption !== d.persisted);
      if (!entries.length) return;
      const release = lane.current.tryBackground(),
        owner = scope.current;
      if (!release) return;
      try {
        for (const [id, draft] of entries) {
          if (scope.current !== owner) break;
          const saved = await request('captionDraft', {
            id,
            draft: { caption: draft.caption, baseRevision: draft.revision },
            draftRevision: draft.draftRevision,
          });
          if (scope.current !== owner) return;
          const img = saved.images.find(x => x.id === id)!;
          const update = (old: Record<string, Draft>) =>
            old[id]
              ? { ...old, [id]: { ...old[id], draftRevision: img.captionDraftRevision ?? 0, persisted: draft.caption } }
              : old;
          draftRef.current = update(draftRef.current);
          setDrafts(update);
        }
        if (scope.current === owner) setSaveStatus('Bozze protette; premi Salva caption per applicarle');
      } catch (e: any) {
        if (scope.current !== owner) return;
        draftFailure.current = true;
        setError(e.response?.data?.error ?? 'Bozza non protetta sul server; resta aperta qui.');
      } finally {
        release();
      }
    }, 500);
    return () => clearInterval(timer);
  }, [drafts, busy]);
  async function saveCaptions() {
    await run('Salvataggio caption', async () => {
      setSaveStatus('Salvataggio…');
      const sent = structuredClone(draftRef.current);
      for (const [id, draft] of Object.entries(sent)) {
        const saved = await saveCaptionSnapshot(id, draft, request, draftRevision => {
          const update = (old: Record<string, Draft>) =>
            old[id] ? { ...old, [id]: { ...old[id], draftRevision, persisted: draft.caption } } : old;
          draftRef.current = update(draftRef.current);
          setDrafts(update);
        });
        const image = saved.images.find(x => x.id === id)!;
        const update = (old: Record<string, Draft>) => {
          const copy = { ...old };
          const remaining = captionDraftAfterSave(copy[id], draft, image);
          if (remaining) copy[id] = remaining;
          else delete copy[id];
          return copy;
        };
        draftRef.current = update(draftRef.current);
        setDrafts(update);
      }
      setSaveStatus(
        Object.entries(draftRef.current).some(([id, d]) => !sent[id] || d.caption !== sent[id].caption)
          ? 'Caption da salvare'
          : 'Caption salvate',
      );
    });
  }
  async function generate() {
    await run(
      'Preparazione caption',
      async () => {
        if (!generationId.current) generationId.current = crypto.randomUUID();
        const epoch = prefEpoch.current;
        await request('generateCaption', {
          requestId: generationId.current,
          preferences: prefCurrent.current,
          preferencesRevision: prefDirty.current
            ? prefBase.current
            : (current.current?.captionPreferencesRevision ?? 0),
        });
        generationId.current = null;
        prefDirty.current = prefEpoch.current !== epoch;
        prefBase.current = current.current?.captionPreferencesRevision ?? 0;
        setPrefStatus(prefDirty.current ? 'Modifiche da salvare' : 'Modello e istruzioni salvati');
      },
      'generate-caption',
    );
  }
  async function upload(files: File[]) {
    if (!files.length) return;
    await run('Caricamento immagini', async () => {
      const owner = scope.current;
      let count = 0;
      const failures: string[] = [];
      for (const file of files) {
        if (scope.current !== owner) return;
        setBusy(`Caricamento ${count + 1}/${files.length}: ${file.name}`);
        const form = new FormData();
        form.set('datasetName', dataset);
        form.append('files', file);
        try {
          await apiClient.post('/api/datasets/upload', form);
          if (scope.current !== owner) return;
          count++;
        } catch (e: any) {
          if (scope.current !== owner) return;
          failures.push(file.name + ': ' + (e.response?.data?.error ?? 'caricamento non confermato'));
        }
      }
      await refresh();
      if (scope.current !== owner) return;
      setSaveStatus(`${count} file caricati`);
      if (failures.length) setError(failures.join('\n'));
    });
  }
  const images = (state?.images ?? []).filter(x => !x.discarded),
    visible = images.filter(x =>
      (x.filename + ' ' + (drafts[x.id]?.caption ?? x.caption)).toLowerCase().includes(search.toLowerCase()) &&
      (categoryFilter === 'all' || x.category === categoryFilter) &&
      (membershipFilter === 'all' || (membershipFilter === 'included' ? !x.excluded : !!x.excluded)) &&
      (!tagFilter || x.tags.some(t=>t.toLowerCase().includes(tagFilter.toLowerCase()))) &&
      (!minQuality || (x.analysis && x.analysis.quality>=minQuality)),
    ),
    selected = images.filter(x => !x.excluded),
    automatic = state?.jobs.filter(x => x.kind === 'caption' && x.automatic) ?? [],
    unresolved = automatic.some(x =>
      ['prepared', 'enqueue-intent', 'active', 'blocked', 'unknown'].includes(x.automatic!.phase),
    );
  const phaseText: Record<string, string> = {
    prepared: 'Preparazione',
    'enqueue-intent': 'Accodamento da confermare',
    active: 'In elaborazione',
    blocked: 'In attesa',
    unknown: 'Esito da riconciliare',
    failed: 'Generazione non riuscita',
    conflict: 'Modifiche manuali preservate',
    applied: 'Caption generate e salvate',
    dismissed: 'Risultati ignorati',
  };
  function cancelSelection() {
    const old = gesture.current;
    gesture.current = null;
    setDrag(null);
    if (old && grid.current?.hasPointerCapture(old.pointerId)) grid.current.releasePointerCapture(old.pointerId);
  }
  useEffect(() => {
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        cancelSelection();
      }
    };
    window.addEventListener('keydown', escape);
    window.addEventListener('scroll', cancelSelection, true);
    return () => {
      window.removeEventListener('keydown', escape);
      window.removeEventListener('scroll', cancelSelection, true);
    };
  }, []);
  useEffect(() => {
    cancelSelection();
  }, [search, size, selectionMode]);
  function captionSources() {
    return captionTextSources(current.current?.images ?? [], draftRef.current, textScope, search);
  }

  const textPreviewFresh =
    textPreview &&
    textPreviewScope.current === scope.current &&
    captionPreviewCurrent(textPreview, captionSources(), findText, replacement);
  function applyCaptionText() {
    if (
      !textPreview ||
      textPreviewScope.current !== scope.current ||
      !captionPreviewCurrent(textPreview, captionSources(), findText, replacement)
    ) {
      setTextPreview(null);
      setError('Anteprima cambiata: ricalcola prima di applicare. Le caption sono preservate.');
      return;
    }
    for (const change of textPreview.changes) {
      const image = current.current?.images.find(x => x.id === change.id);
      if (image) changeCaption(image, change.next);
    }
    setTextPreview(null);
  }
  function gallery(items: Image[]) {
    const hits = new Set(drag ? selectionHits(drag) : []);
    const rectangle = drag ? selectionRectangle(drag.start, drag.end) : null;
    return (
      <div
        ref={grid}
        className="grid gap-6"
        style={{
          display: 'grid',
          gap: 24,
          gridTemplateColumns: `repeat(auto-fit,minmax(min(100%,${size}px),1fr))`,
          userSelect: drag ? 'none' : undefined,
        }}
        onPointerDown={event => {
          if (
            event.pointerType !== 'mouse' ||
            event.button !== 0 ||
            busy ||
            !state ||
            (event.target as Element).closest('input,textarea,button,a,label,select,summary,[contenteditable]')
          )
            return;
          const targets = Array.from(event.currentTarget.querySelectorAll<HTMLElement>('[data-selection-image]')).map(
            node => {
              const box = node.getBoundingClientRect();
              return {
                id: node.dataset.selectionImage!,
                rect: {
                  left: box.left + window.scrollX,
                  top: box.top + window.scrollY,
                  right: box.right + window.scrollX,
                  bottom: box.bottom + window.scrollY,
                },
              };
            },
          );
          const point = { x: event.pageX, y: event.pageY };
          gesture.current = {
            scope: scope.current,
            pointerId: event.pointerId,
            start: point,
            end: point,
            targets,
            excluded: selectionMode,
          };
          event.currentTarget.setPointerCapture(event.pointerId);
          setDrag(gesture.current);
          event.preventDefault();
        }}
        onPointerMove={event => {
          const old = gesture.current;
          if (!old || old.pointerId !== event.pointerId) return;
          if (old.scope !== scope.current) {
            cancelSelection();
            return;
          }
          gesture.current = { ...old, end: { x: event.pageX, y: event.pageY } };
          setDrag(gesture.current);
        }}
        onPointerUp={event => {
          const old = gesture.current;
          if (!old || old.pointerId !== event.pointerId) return;
          const ended = { ...old, end: { x: event.pageX, y: event.pageY } };
          const ids = selectionCommit(
            ended,
            scope.current,
            visible.map(x => x.id),
          );
          cancelSelection();
          if (ids.length)
            void run('Salvataggio selezione', () => request('edit', { ids, patch: { excluded: ended.excluded } }));
          else if (
            Math.hypot(ended.end.x - ended.start.x, ended.end.y - ended.start.y) < 5 &&
            ended.scope === scope.current
          ) {
            const target = old.targets.find(
              x =>
                old.start.x >= x.rect.left &&
                old.start.x <= x.rect.right &&
                old.start.y >= x.rect.top &&
                old.start.y <= x.rect.bottom,
            );
            const image = current.current?.images.find(x => x.id === target?.id);
            if (image)
              void run('Salvataggio selezione', () =>
                request('edit', { ids: [image.id], patch: { excluded: image.excluded ? 0 : 1 } }),
              );
          }
        }}
        onPointerCancel={() => {
          cancelSelection();
        }}
        onLostPointerCapture={() => {
          if (gesture.current) {
            cancelSelection();
          }
        }}
      >
        {rectangle && (
          <div
            aria-hidden="true"
            style={{
              position: 'fixed',
              pointerEvents: 'none',
              zIndex: 20,
              left: rectangle.left - window.scrollX,
              top: rectangle.top - window.scrollY,
              width: rectangle.right - rectangle.left,
              height: rectangle.bottom - rectangle.top,
              border: '2px solid #60a5fa',
              background: 'rgba(59,130,246,.15)',
            }}
          />
        )}
        {items.map(image => (
          <article
            key={image.id}
            style={{ minWidth: 0, padding: 12, background: '#111827', border: hits.has(image.id) ? '2px solid #60a5fa' : '2px solid transparent' }}
            className="min-w-0 rounded-xl space-y-3"
          >
            <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8, minWidth: 0 }}>
              <input
                type="checkbox"
                aria-label={'Seleziona ' + image.filename}
                className="h-5 w-5"
                checked={!image.excluded}
                disabled={!!busy}
                onChange={() =>
                  void run('Salvataggio selezione', () =>
                    request('edit', { ids: [image.id], patch: { excluded: image.excluded ? 0 : 1 } }),
                  )
                }
              />
              <span title={image.filename} style={{ flex: '1 1 120px', minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{image.filename}</span>
              {image.analysisStatus?.phase === 'complete' && (
                <label style={{ display: 'flex', alignItems: 'center', gap: 6, whiteSpace: 'nowrap', flexShrink: 0 }} className="text-sm">
                  <select
                    aria-label={'Categoria ' + image.filename}
                    className="rounded border border-gray-600 bg-gray-950 p-1"
                    value={image.category}
                    disabled={!!busy}
                    onChange={e =>
                      void run('Revisione categoria', () =>
                        request('edit', { ids: [image.id], patch: { category: e.target.value } }),
                      )
                    }
                  >
                    <option value="face">Volto</option>
                    <option value="body">Corpo</option>
                    <option value="variety">Varietà</option>
                    <option value="unclassified">Non classificata</option>
                  </select>
                  <span className="text-gray-400">{image.categorySource === 'manual' ? 'Rivista' : 'Automatica'}</span>
                </label>
              )}
            </div>
            {['queued','running','waiting'].includes(state?.analysisProgress.phase ?? '') && (
              <p className="text-xs text-amber-200" style={{ margin: '6px 0', minHeight: 16 }}>
                {image.analysisStatus?.phase !== 'complete' ? 'Analisi in attesa' :
                  state?.analysisProgress.tentative.includes(image.id) ? 'Proposta provvisoria: includi' : 'Proposta provvisoria: escludi'}
                {(image.reviewRevision ?? 0) > 0 ? ' · revisione protetta' : ''}
              </p>
            )}
            <div
              data-selection-image={image.id}
              style={{ cursor: 'crosshair' }}
              onClick={event => {
                if (
                  (event.nativeEvent as PointerEvent).pointerType === 'mouse' ||
                  (event.target as Element).closest('button')
                )
                  return;
                if (!busy)
                  void run('Salvataggio selezione', () =>
                    request('edit', { ids: [image.id], patch: { excluded: image.excluded ? 0 : 1 } }),
                  );
              }}
            >
              <PreviewImage dataset={dataset} image={image} height={Math.min(520, Math.max(220, size * .76))} />
            </div>
            <div className="my-2 flex flex-wrap items-center gap-2 text-xs" aria-label={'Qualità e tag '+image.filename}>
              <span className="rounded bg-gray-800 px-2 py-1">{image.analysis && Number.isFinite(image.analysis.quality) ? `Qualità ${Math.round(image.analysis.quality)}/100` : 'Qualità in attesa'}</span>
              {image.tags.map(tag=><span key={tag} className="rounded bg-gray-800 px-2 py-1">{tag}</span>)}
            </div>
            <label className="block text-sm">
              Caption
              <textarea
                aria-label={'Caption ' + image.filename}
                className={field + ' mt-2 text-base'}
                rows={5}
                value={drafts[image.id]?.caption ?? image.caption}
                onChange={e => changeCaption(image, e.target.value)}
              />
            </label>
            <label className="block text-xs mt-2">Tag
              <input aria-label={'Tag '+image.filename} className={field+' mt-1'} value={tagDrafts[image.id] ?? image.tags.join(', ')}
                onChange={e=>setTagDrafts(old=>({...old,[image.id]:e.target.value}))} placeholder="Tag separati da virgole" />
            </label>
            {tagDrafts[image.id] !== undefined && <button className="text-sm underline" disabled={!!busy}
              onClick={()=>void run('Salvataggio tag',async()=>{const value=tagDrafts[image.id];await request('edit',{ids:[image.id],patch:{tags:value.split(',').map(x=>x.trim()).filter(Boolean)}});
                setTagDrafts(old=>{if(old[image.id]!==value)return old;const copy={...old};delete copy[image.id];return copy;});})}>Salva tag</button>}
            {drafts[image.id] && (
              <div className="flex items-center gap-3 text-sm">
                <span className="text-amber-200">{drafts[image.id].persisted===drafts[image.id].caption?'Bozza salvata sul server · da applicare':'Bozza da salvare'}</span>
                <button
                  className="underline"
                  onClick={() =>
                    void run('Scarto bozza', async () => {
                      const d = draftRef.current[image.id];
                      if (current.current?.images.find(x => x.id === image.id)?.captionDraft)
                        await request('captionDraft', { id: image.id, draft: null, draftRevision: d.draftRevision });
                      setDrafts(old => {
                        const copy = { ...old };
                        delete copy[image.id];
                        return copy;
                      });
                    })
                  }
                >
                  Scarta bozza
                </button>
              </div>
            )}
          </article>
        ))}
      </div>
    );
  }
  return (
    <>
      <TopBar>
        <Link className="mr-4 shrink-0" href="/datasets">
          ← Dataset
        </Link>
        <h1 className="truncate flex-1 min-w-0">{dataset}</h1>
      </TopBar>
      <MainContent belowTopBar>
        <div style={{ maxWidth: 1500 }} className="mx-auto min-w-0 p-3 sm:p-6 space-y-6">
          <header className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h2 className="text-2xl font-semibold">Immagini e caption</h2>
              <p className="text-gray-400 text-sm mt-1">Carica, seleziona e descrivi le tue immagini.</p>
            </div>
            <label className={button + ' cursor-pointer'}>
              Carica immagini
              <input
                aria-label="Carica immagini"
                className="sr-only"
                type="file"
                multiple
                accept=".png,.jpg,.jpeg,.webp,.txt"
                disabled={!!busy}
                onChange={e => {
                  const files = Array.from(e.target.files ?? []);
                  e.target.value = '';
                  void upload(files);
                }}
              />
            </label>
          </header>
          <details
            className="rounded-xl border border-gray-700 p-4"
            open={advanced}
            onToggle={e => {
              (() => {
                setAdvanced(e.currentTarget.open);
                if (e.currentTarget.open) setAdvancedMounted(true);
              })();
              if (!e.currentTarget.open) void refresh();
            }}
          >
            <summary className="cursor-pointer text-lg">Avanzate</summary>
            {state && <p className="my-3 text-sm text-gray-400">Spazio: {(state.space.free/1e9).toFixed(2)} GB liberi · {(state.space.missingBytes/1e6).toFixed(1)} MB di copie mancanti · {(state.space.outputBytes/1e6).toFixed(1)} MB per risultati · riserva {(state.space.reserve/1e9).toFixed(0)} GB. Le copie derivate duplicate completate vengono consolidate quando necessario; gli originali restano separati.</p>}
            {advancedMounted && <AdvancedWorkspace dataset={dataset} onNativeView={onNativeView} />}
          </details>
          <section className="rounded-xl border border-gray-700 bg-gray-900 p-4 sm:p-5 space-y-4">
            {state && images.length > 0 && (
              <section aria-label="Analisi immagini" className="space-y-3">
                <div style={{ display:'flex', flexWrap:'wrap', alignItems:'end', gap:12 }}>
                  <label className="text-sm" style={{ whiteSpace:'nowrap' }}>Immagini per training
                    <input aria-label="Immagini per training" type="number" min={1} max={1000}
                      className="rounded bg-gray-950 p-2 ml-2" style={{ width:90 }} value={target} onChange={e=>setTarget(e.target.value)} />
                  </label>
                  <button className={button+' bg-blue-700'}
                    disabled={!!busy || ['queued','running','waiting'].includes(state.analysisProgress?.phase) || !Number.isInteger(Number(target)) || Number(target)<1 || Number(target)>1000}
                    onClick={()=>void run('Analisi',()=>request('automaticAnalysis',{ count:Number(target), retry:state.analysisProgress?.phase==='blocked' }), 'explicit-analysis')}>
                    {state.analysisProgress?.phase==='blocked' ? 'Riprova analisi' : 'Analisi'}
                  </button>
                  {Number(target)!==state.settings.count && <span className="text-sm text-gray-400">Obiettivo in bozza · applicato con Analisi</span>}
                </div>
                <div role="status" aria-live="polite" className="text-sm text-gray-300 space-y-1">
                  <strong>{({ ready:'Pronto per analisi', queued:'Analisi in coda', running:'Analisi in corso', waiting:'Conferma dei risultati in corso', complete:'Proposta pronta', blocked:'Analisi sospesa', unavailable:'Analisi non disponibile' })[state.analysisProgress?.phase ?? 'ready']}</strong>
                  <p>{state.analysisProgress?.done ?? 0}/{images.length} immagini analizzate · {selected.length} incluse · {images.length-selected.length} escluse</p>
                  {['queued','running','waiting'].includes(state.analysisProgress?.phase) && <>
                    <progress style={{ width:'100%' }} max={images.length || 1} value={state.analysisProgress.done} />
                    <p>{state.analysisProgress.currentFile ?? (state.analysisProgress.phase==='queued' ? 'In attesa del turno sulla GPU' : 'Caricamento modelli o verifica dei risultati')}
                    {state.analysisProgress.elapsedSeconds!==undefined && ` · trascorsi ${Math.floor(state.analysisProgress.elapsedSeconds/60)}m ${Math.floor(state.analysisProgress.elapsedSeconds%60)}s`}
                    {state.analysisProgress.etaSeconds!==undefined ? ` · restano circa ${Math.ceil(state.analysisProgress.etaSeconds/60)} min` : ' · Calcolo tempo…'}</p>
                  </>}
                  <p className="text-sm text-gray-400">
                    {state.space.free >= state.space.reserve+state.space.missingBytes+state.space.outputBytes ? 'Spazio disponibile' :
                      `Spazio insufficiente: ${(state.space.free/1e9).toFixed(2)} GB liberi, ${( (state.space.reserve+state.space.missingBytes+state.space.outputBytes)/1e9).toFixed(2)} GB necessari; mancano ${((state.space.reserve+state.space.missingBytes+state.space.outputBytes-state.space.free)/1e6).toFixed(1)} MB. Analisi verifica il recupero sicuro dello spazio; se non basta, resta sospesa e conserva gli originali.`}
                  </p>
                  {!!state.analysisProgress?.tentative.length && <p className="text-amber-200">Proposta provvisoria: {state.analysisProgress.tentative.length} immagini dai risultati verificati. La selezione finale e le tue revisioni restano protette.</p>}
                  {state.analysisProgress?.detail && <p className="text-amber-200">{state.analysisProgress.detail}</p>}
                  {state.analysisFlow?.phase==='complete' && Object.entries(state.analysisFlow.deficits ?? {}).some(([,n])=>n>0) && <p className="text-amber-200">Quote incomplete: {Object.entries(state.analysisFlow.deficits ?? {}).filter(([,n])=>n>0).map(([c,n])=>`${c}: mancano ${n}`).join(' · ')}</p>}
                </div>
              </section>
            )}
            <label className="block text-sm">
              Modello locale
              <select
                aria-label="Modello locale"
                className={field + ' mt-2'}
                value={pref.key}
                onChange={e => changePreferences({ ...pref, key: e.target.value })}
              >
                {[...new Set(captionModels.map(x => x.group))].map(group => (
                  <optgroup key={group} label={group}>
                    {captionModels
                      .filter(x => x.group === group)
                      .map(x => (
                        <option key={x.key} value={x.key}>
                          {x.label}
                        </option>
                      ))}
                  </optgroup>
                ))}
              </select>
            </label>
            <label className="block text-sm">
              Istruzioni per le caption
              <textarea
                aria-label="Istruzioni per le caption"
                rows={3}
                className={field + ' mt-2'}
                value={pref.instructions}
                onChange={e => changePreferences({ ...pref, instructions: e.target.value })}
                onBlur={() => void savePreferences()}
              />
            </label>
            <div className="flex flex-wrap items-center gap-3">
              <button
                className={button + ' bg-blue-700 border-blue-500 hover:bg-blue-600'}
                disabled={
                  !!busy ||
                  !state ||
                  !selected.length ||
                  unresolved ||
                  !!Object.keys(drafts).length ||
                  !state.captionHostSupported
                }
                onClick={() => void generate()}
              >
                Genera caption · {selected.length} immagini
              </button>
              <span aria-live="polite" className="text-sm text-gray-400">
                {prefStatus}
              </span>
            </div>
            {prefError.current && (
              <button
                className="underline text-sm"
                onClick={() => {
                  const s = current.current!;
                  setPref(initialPreferences(s));
                  setDrafts(
                    Object.fromEntries(
                      s.images
                        .filter(x => x.captionDraft)
                        .map(x => [
                          x.id,
                          {
                            caption: x.captionDraft!.caption,
                            revision: x.captionDraft!.baseRevision,
                            draftRevision: x.captionDraftRevision ?? 0,
                            persisted: x.captionDraft!.caption,
                          },
                        ]),
                    ),
                  );
                  prefDirty.current = false;
                  prefError.current = false;
                  prefBase.current = s.captionPreferencesRevision ?? 0;
                  setPrefStatus('Impostazioni ricaricate dal server');
                }}
              >
                Scarta la bozza delle impostazioni
              </button>
            )}
            {state && !state.captionHostSupported && (
              <p className="text-sm text-amber-200">
                Qui puoi preparare e modificare il dataset. Per generare caption serve AI Toolkit su un host Linux con
                GPU NVIDIA; nel preview e sul Mac i modelli non partono.
              </p>
            )}
            {Object.keys(drafts).length > 0 && (
              <p className="text-sm text-amber-200">
                Salva le caption manuali prima di generare. Durante una generazione le nuove bozze restano protette.
              </p>
            )}
          </section>
          <div aria-live="polite">
            {busy && <p className="text-blue-300">{busy}…</p>}
            {error && (
              <div role="alert" className="rounded-lg bg-red-950/60 text-red-200 p-4 whitespace-pre-wrap break-words">
                {error}
                <button className={button + ' mt-3 block'} onClick={() => void refresh()}>
                  Aggiorna dati confermati
                </button>
              </div>
            )}
            {!state && !error && <p>Caricamento dataset…</p>}
          </div>
          {automatic.length > 0 && (
            <section className="space-y-3" aria-label="Avanzamento caption">
              {automatic
                .slice()
                .reverse()
                .map(link => {
                  const row = state?.jobsLive.find(x => x.id === link.jobId),
                    auto = link.automatic!;
                  return (
                    <div key={auto.id} className="rounded-lg border border-gray-700 p-4 text-sm space-y-2">
                      <p>
                        {phaseText[auto.phase]} · {row?.step ?? 0}/{link.scope?.length ?? 0} immagini
                      </p>
                      {row?.status === 'queued' && auto.phase === 'active' && (
                        <p>In coda sul tuo host, in attesa del turno.</p>
                      )}
                      {auto.reason && <p className="text-amber-200 whitespace-pre-wrap">{auto.reason}</p>}
                      {['unknown', 'blocked', 'failed', 'conflict'].includes(auto.phase) && (
                        <div className="flex flex-wrap gap-3">
                          <button
                            className={button}
                            disabled={!!busy}
                            onClick={() =>
                              void run('Riconciliazione caption', () =>
                                request('reconcileCaption', {
                                  id: auto.id,
                                  explicit: true,
                                  blockedIds: Object.keys(draftRef.current),
                                }),
                              )
                            }
                          >
                            Riconcilia
                          </button>
                          {['failed', 'conflict'].includes(auto.phase) && (
                            <button
                              className={button}
                              disabled={!!busy}
                              onClick={() =>
                                void run('Conservazione caption manuali', () =>
                                  request('dismissCaption', { id: auto.id }),
                                )
                              }
                            >
                              Mantieni le caption attuali
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
            </section>
          )}
          <section className="space-y-4">
            <div className="flex flex-wrap justify-between gap-3 items-end">
              <div className="min-w-0">
                <h3 className="text-lg">{selected.length} {selected.length === 1 ? "immagine selezionata" : "immagini selezionate"}</h3>
                <p className="text-sm text-gray-400">
                  La selezione è salvata e indica quali immagini saranno descritte.
                </p>
              </div>
              <div className="flex gap-2 flex-wrap">
                <button
                  className={button}
                  disabled={!!busy || !visible.length}
                  onClick={() =>
                    void run('Selezione immagini', () =>
                      request('edit', { ids: visible.map(x => x.id), patch: { excluded: 0 } }),
                    )
                  }
                >
                  Seleziona tutte visibili
                </button>
                <button
                  className={button}
                  disabled={!!busy || !visible.length}
                  onClick={() =>
                    void run('Deselezione immagini', () =>
                      request('edit', { ids: visible.map(x => x.id), patch: { excluded: 1 } }),
                    )
                  }
                >
                  Deseleziona tutte visibili
                </button>
              </div>
            </div>
            <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(min(100%,220px),1fr))', gap:12 }}>
              <label className="text-sm">Categoria<select aria-label="Filtra categoria" className={field} value={categoryFilter} onChange={e=>setCategoryFilter(e.target.value)}><option value="all">Tutte le categorie</option><option value="face">Volto</option><option value="body">Corpo</option><option value="variety">Varietà</option><option value="unclassified">Da analizzare</option></select></label>
              <label className="text-sm">Selezione<select aria-label="Filtra selezione" className={field} value={membershipFilter} onChange={e=>setMembershipFilter(e.target.value)}><option value="all">Incluse ed escluse</option><option value="included">Solo incluse</option><option value="excluded">Solo escluse</option></select></label>
              <label className="text-sm">Tag<input aria-label="Filtra tag" className={field} value={tagFilter} onChange={e=>setTagFilter(e.target.value)} placeholder="Tutti i tag" /></label>
              <label className="text-sm">Qualità minima: {minQuality}<input aria-label="Qualità minima" type="range" className="w-full" min={0} max={100} value={minQuality} onChange={e=>setMinQuality(Number(e.target.value))} /></label>
              <input
                aria-label="Cerca immagini"
                placeholder="Cerca immagine o caption"
                className={field}
                value={search}
                onChange={e => setSearch(e.target.value)}
              />
              <label className="text-sm">
                Dimensione immagini
                <input
                  aria-label="Dimensione immagini"
                  className="mt-3 w-full"
                  type="range"
                  min={280}
                  max={680}
                  step={20}
                  value={size}
                  onChange={e => setSize(Number(e.target.value))}
                />
              </label>
            </div>
            <div className="flex items-center flex-wrap gap-3">
              <button
                className={button}
                disabled={!!busy || !Object.keys(drafts).length}
                onClick={() => void saveCaptions()}
              >
                Salva caption
              </button>
              <span className="text-sm text-gray-400" aria-live="polite">
                {Object.keys(drafts).length ? `${Object.keys(drafts).length} bozze da salvare` : saveStatus}
              </span>
              <button
                className={button}
                aria-expanded={textToolsOpen}
                onClick={() => {
                  setTextToolsOpen(!textToolsOpen);
                  setTextPreview(null);
                }}
              >
                Trova e sostituisci
              </button>
            </div>
            {textToolsOpen && (
              <section
                aria-label="Trova e sostituisci caption"
                style={{
                  border: '1px solid #374151',
                  borderRadius: 12,
                  padding: 16,
                  display: 'grid',
                  gap: 12,
                  minWidth: 0,
                }}
              >
                <label>
                  Trova
                  <input
                    aria-label="Trova testo caption"
                    className={field}
                    value={findText}
                    onChange={e => {
                      setFindText(e.target.value);
                      setTextPreview(null);
                    }}
                  />
                </label>
                <label>
                  Sostituisci con
                  <input
                    aria-label="Sostituisci testo caption"
                    className={field}
                    value={replacement}
                    onChange={e => {
                      setReplacement(e.target.value);
                      setTextPreview(null);
                    }}
                  />
                </label>
                <label>
                  Ambito
                  <select
                    aria-label="Ambito sostituzione"
                    className={field}
                    value={textScope}
                    onChange={e => {
                      setTextScope(e.target.value as 'selected' | 'visible');
                      setTextPreview(null);
                    }}
                  >
                    <option value="selected">Immagini selezionate (tutto il dataset)</option>
                    <option value="visible">Tutte le immagini visibili (anche non selezionate)</option>
                  </select>
                </label>
                <p className="text-sm text-gray-400">
                  Testo letterale, tutte le occorrenze. Applica modifica le bozze; poi usa Salva caption.
                </p>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
                  <button
                    className={button}
                    disabled={!findText || !!busy || !state}
                    onClick={() => {
                      textPreviewScope.current = scope.current;
                      setTextPreview(previewCaptionText(captionSources(), findText, replacement));
                    }}
                  >
                    Anteprima sostituzione
                  </button>
                  <button
                    className={button}
                    disabled={!textPreviewFresh || !textPreview?.changes.length || !!busy}
                    onClick={applyCaptionText}
                  >
                    Applica alle bozze
                  </button>
                  <button
                    className={button}
                    onClick={() => {
                      setTextPreview(null);
                      setTextToolsOpen(false);
                    }}
                  >
                    Annulla
                  </button>
                </div>
                {textPreview && (
                  <p aria-live="polite">
                    {textPreviewFresh
                      ? `${textPreview.changes.length} immagini · ${textPreview.occurrences} occorrenze`
                      : 'Anteprima obsoleta: ricalcola dopo le modifiche.'}
                  </p>
                )}
              </section>
            )}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
              <label>
                Rettangolo mouse
                <select
                  aria-label="Modalità rettangolo"
                  className={field}
                  value={selectionMode}
                  onChange={e => setSelectionMode(Number(e.target.value) as 0 | 1)}
                >
                  <option value={0}>Seleziona</option>
                  <option value={1}>Deseleziona</option>
                </select>
              </label>
              <span className="text-sm text-gray-400">
                Trascina sulle immagini. Esc annulla; checkbox anche da touch e tastiera.
              </span>
            </div>
            {gallery(visible)}
            {state && <p role="status" className="mb-3 text-sm text-gray-400">
              Ultimo salvataggio sul server · {state.managedSync.phase==='synced'?'Sincronizzato su HF':state.managedSync.phase==='local'?'HF non configurato':state.managedSync.phase==='conflict'?'Conflitto HF: entrambe le versioni conservate':state.managedSync.phase==='error'?'Errore HF, salvataggio locale conservato':state.managedSync.phase==='pending'?'Sincronizzazione HF in attesa':'Sincronizzazione HF in corso'}
              {state.managedSync.reason ? ' · '+state.managedSync.reason : ''}
            </p>}
            {state && !images.length && <p className="text-gray-400">Carica le prime immagini per cominciare.</p>}
          </section>

        </div>
      </MainContent>
    </>
  );
}
