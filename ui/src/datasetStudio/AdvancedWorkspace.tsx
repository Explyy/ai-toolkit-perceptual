'use client';
import React, { useEffect, useState, useRef } from 'react';
import { apiClient } from '@/utils/api';
import { captionerTypes } from '@/helpers/captionOptions';
import { defaultJobConfig } from '@/app/jobs/new/jobConfig';
import { modelArchs } from '@/app/jobs/new/options';
import { openCaptionDatasetModal } from '@/components/CaptionDatasetModal';
import AddImagesModal, { openImagesModal, useOpenImagesModalOnDrag } from '@/components/AddImagesModal';
import { MainContent, TopBar } from '@/components/layout';
import type { State, Image, Template } from './store';
import { select, CATEGORIES, stableJSON } from './domain';
import Link from 'next/link';
const imageCaptioners = captionerTypes.filter(x => x.group.includes('image'));
const inputClass = 'bg-gray-950 border border-gray-600 rounded p-2 w-full min-w-0';
const buttonClass =
  'rounded border border-gray-600 bg-gray-800 px-3 py-2 text-sm disabled:opacity-50 hover:bg-gray-700';
function draftConfig() {
  const cfg: any = structuredClone(defaultJobConfig),
    native = modelArchs.find(x => x.name === 'krea2')!;
  cfg.config.process[0].model.arch = native.name;
  for (const [key, v] of Object.entries(native.defaults ?? {})) {
    const parts = key.replace('config.process[0].', '').split('.');
    let p = cfg.config.process[0];
    for (const part of parts.slice(0, -1)) p = p[part] ??= {};
    p[parts.at(-1)!] = (v as any[])[0];
  }
  cfg.config.process[0].sample.samples = [];
  cfg.config.process[0].train.disable_sampling = true;
  return cfg;
}
function Photo({ dataset, image }: { dataset: string; image: Image }) {
  const [url, setUrl] = useState(''),
    [failed, setFailed] = useState(false),
    [attempt, setAttempt] = useState(0);
  const element = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let dead = false,
      created = '',
      loading = false;
    const load = () => {
      if (loading) return;
      loading = true;
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
    };
    const observer = new IntersectionObserver(
      entries => {
        if (entries.some(e => e.isIntersecting)) {
          load();
          observer.disconnect();
        }
      },
      { rootMargin: '400px' },
    );
    if (element.current) observer.observe(element.current);
    return () => {
      dead = true;
      observer.disconnect();
      if (created) URL.revokeObjectURL(created);
    };
  }, [dataset, image.id, attempt]);
  return (
    <div ref={element} style={{ height: 260 }}>
      {url ? (
        <img
          src={url}
          alt={image.filename}
          loading="lazy"
          style={{
            height: 260,
            width: '100%',
            objectFit: 'contain',
            opacity: image.excluded || image.discarded?.valueOf() ? 0.35 : 1,
          }}
        />
      ) : (
        <div className="bg-gray-900 flex items-center justify-center h-full">
          {failed ? (
            <button onClick={() => setAttempt(x => x + 1)}>Image unavailable · Retry</button>
          ) : (
            'Loading image…'
          )}
        </div>
      )}
    </div>
  );
}
export default function AdvancedWorkspace({ dataset, onNativeView }: { dataset: string; onNativeView: () => void }) {
  const [state, setState] = useState<(State & { jobsLive: any[]; hfConfigured: boolean; preview: boolean }) | null>(
      null,
    ),
    [error, setError] = useState(''),
    [busy, setBusy] = useState(''),
    [tab, setTab] = useState('Selection / Export');
  const [filter, setFilter] = useState(''),
    [category, setCategory] = useState('all'),
    [membership, setMembership] = useState('all'),
    [minScore, setMinScore] = useState(0),
    [tag, setTag] = useState(''),
    [cardSize, setCardSize] = useState(320);
  const [drafts, setDrafts] = useState<Record<string, { caption: string; tags: string; revision: number }>>({}),
    [target, setTarget] = useState(3),
    [tier, setTier] = useState(1024),
    [crop, setCrop] = useState('fit'),
    [repo, setRepo] = useState('daverave/Personal'),
    [testId, setTestId] = useState(''),
    [testPrompt, setTestPrompt] = useState('');
  const [captioner, setCaptioner] = useState(imageCaptioners[0].name),
    [captionModel, setCaptionModel] = useState(String(imageCaptioners[0].name_or_path_options![0].value)),
    [instructions, setInstructions] = useState(
      'Describe only visible features, clothing, framing, pose and background. Return one concise training caption.',
    ),
    [captionScope, setCaptionScope] = useState('included');
  const [templateName, setTemplateName] = useState(''),
    [subject, setSubject] = useState(''),
    [trigger, setTrigger] = useState(''),
    [training, setTraining] = useState(() => JSON.stringify(draftConfig(), null, 2)),
    [previous, setPrevious] = useState<string | null>(null),
    [approvedId, setApprovedId] = useState(''),
    [versionId, setVersionId] = useState(''),
    [gpu, setGpu] = useState('0');
  const restoredDraft = useRef(''),
    activeDataset = useRef(dataset);
  activeDataset.current = dataset;
  const current = useRef<any>(null),
    mutation = useRef(false);
  function accept(s: any) {
    if (s.dataset !== activeDataset.current) return;
    current.current = s;
    setState(s);
    if (restoredDraft.current !== dataset) {
      restoredDraft.current = dataset;
      const d = s.templateDraft;
      if (d) {
        setTemplateName(d.name);
        setSubject(d.subject);
        setTrigger(d.trigger);
        setTraining(d.training);
        setCaptioner(d.captioner);
        setCaptionModel(d.captionModel);
        setInstructions(d.instructions);
        setPrevious(d.previous);
      } else {
        setTemplateName('');
        setSubject('');
        setTrigger('');
        setTraining(JSON.stringify(draftConfig(), null, 2));
        setPrevious(null);
      }
    }
  }

  async function refresh() {
    try {
      const r = await apiClient.get('/api/dataset-studio', { params: { dataset } });
      accept(r.data);
    } catch (e: any) {
      setError(e.response?.data?.error ?? 'Unable to load dataset');
    }
  }
  useEffect(() => {
    setState(null);
    current.current = null;
    restoredDraft.current = '';
    setDrafts({});
    void refresh();
  }, [dataset]);
  useEffect(() => {
    const timer = setInterval(() => {
      if (!mutation.current) void refresh();
    }, 5000);
    return () => clearInterval(timer);
  }, [dataset]);
  useEffect(() => {
    if (!state) return;
    setTarget(state.settings.count);
    setTier(state.settings.tier);
    setCrop(state.settings.crop);
    setRepo(state.settings.repo);
    setTestId(state.settings.testId ?? '');
    setTestPrompt(state.settings.testPrompt);
  }, [
    state?.settings.count,
    state?.settings.tier,
    state?.settings.crop,
    state?.settings.repo,
    state?.settings.testId,
    state?.settings.testPrompt,
  ]);
  useEffect(() => {
    const handler = (e: BeforeUnloadEvent) => {
      if (Object.keys(drafts).length) {
        e.preventDefault();
        e.returnValue = '';
      }
    };
    window.addEventListener('beforeunload', handler);
    return () => window.removeEventListener('beforeunload', handler);
  }, [drafts]);
  useOpenImagesModalOnDrag(dataset, refresh);
  async function request(action: string, payload: any = {}) {
    const r = await apiClient.post('/api/dataset-studio', {
      dataset,
      revision: current.current.revision,
      action,
      ...payload,
    });
    accept(r.data);
    return r.data;
  }
  async function run(label: string, fn: () => Promise<any>) {
    if (mutation.current) return;
    mutation.current = true;
    setBusy(label);
    setError('');
    try {
      return await fn();
    } catch (e: any) {
      setError(e.response?.data?.error ?? e.message ?? 'Operation failed. Drafts remain; refresh and retry.');
    } finally {
      mutation.current = false;
      setBusy('');
    }
  }
  function changeDraft(x: Image, field: 'caption' | 'tags', value: string) {
    setDrafts(old => ({
      ...old,
      [x.id]: {
        caption: old[x.id]?.caption ?? x.caption,
        tags: old[x.id]?.tags ?? x.tags.join(', '),
        revision: old[x.id]?.revision ?? x.revision,
        [field]: value,
      },
    }));
  }
  async function settingsSave() {
    await request('settings', {
      settings: {
        ...current.current.settings,
        count: target,
        tier,
        crop,
        repo,
        testId: testId || null,
        testPrompt,
        instructions,
        model: captionModel,
      },
    });
  }
  async function exportVersion() {
    await settingsSave();
    let s = await request('export'),
      v = s.snapshots.find((x: any) => x.state === 'building');
    if (!v) {
      setVersionId(s.snapshots.at(-1)?.id ?? '');
      return;
    }
    for (let i = 0; i < v.source.length; i++) {
      setBusy('Export pair ' + (i + 1) + '/' + v.source.length);
      await request('exportFile', { id: v.id, index: i });
    }
    s = await request('exportFinish', { id: v.id });
    setVersionId(v.id);
  }
  async function syncVersion(id: string) {
    let s = await request('hfStart', { id, repo }),
      v = s.snapshots.find((x: any) => x.id === id);
    if (v.hf.phase === 'verified') return;
    const total = v.files.length + 1;
    if (['unknown', 'conflict', 'commit_started'].includes(v.hf.phase)) {
      await request('hfReconcile', { id });
    } else if (v.hf.phase === 'uploading') {
      for (let i = 0; i < total; i++) {
        setBusy('HF upload ' + (i + 1) + '/' + total);
        await request('hfUpload', { id, index: i });
      }
      await request('hfCommit', { id });
    }
    for (let i = 0; i < total; i++) {
      setBusy('HF readback ' + (i + 1) + '/' + total);
      await request('hfVerify', { id, index: i });
    }
  }
  async function approveTemplate() {
    return request('approve', {
      template: {
        approve: true,
        name: templateName,
        subject,
        trigger,
        previous,
        captioner,
        captionModel,
        training: JSON.parse(training),
        settings: {
          ...current.current.settings,
          count: target,
          tier,
          crop,
          repo,
          testId: testId || null,
          testPrompt,
          model: captionModel,
          instructions,
        },
      },
    });
  }
  function applyTemplate(t: Template, duplicate = false) {
    setTemplateName(t.name + (duplicate ? ' copy' : ''));
    setSubject(t.subject);
    setTrigger(t.trigger);
    setInstructions(t.settings.instructions);
    setCaptioner(t.captioner);
    setCaptionModel(t.captionModel);
    setTraining(JSON.stringify(t.training, null, 2));
    setTarget(t.settings.count);
    setTier(t.settings.tier);
    setCrop(t.settings.crop);
    setPrevious(duplicate ? null : t.id);
    setApprovedId(t.id);
    void run('Applying approved selection settings', () => request('apply', { id: t.id }));
  }
  const chosenCaptioner = imageCaptioners.find(x => x.name === captioner)!;
  const visible = (state?.images ?? []).filter(
    x =>
      (category === 'all' || x.category === category) &&
      (membership === 'all' ||
        (membership === 'included' && !x.excluded && !x.discarded) ||
        (membership === 'excluded' && !!x.excluded) ||
        (membership === 'removed' && !!x.discarded)) &&
      (!tag || x.tags.some(t => t.toLowerCase().includes(tag.toLowerCase()))) &&
      (!minScore || (!!x.analysis && x.analysis.quality >= minScore)) &&
      (!filter ||
        (x.filename + ' ' + (drafts[x.id]?.caption ?? x.caption) + ' ' + x.tags.join(' '))
          .toLowerCase()
          .includes(filter.toLowerCase())),
  );
  const selected = state ? select(state.images, state.settings.count, state.settings.testId) : null;
  function cards(items: Image[]) {
    return (
      <div
        className="grid gap-5"
        style={{ gridTemplateColumns: `repeat(auto-fit,minmax(min(100%,${cardSize}px),1fr))` }}
      >
        {items.map(x => (
          <article key={x.id} className="min-w-0 rounded-lg border border-gray-700 bg-gray-900 p-3 space-y-3">
            <label className="flex gap-2 items-center break-all">
              <span className="text-sm">{x.filename}</span>
            </label>
            <Photo dataset={dataset} image={x} />
            <p className="text-xs text-gray-400">
              {x.width}×{x.height} · {x.excluded ? 'Excluded' : x.discarded ? 'Removed' : 'Included'} · Score{' '}
              {x.analysis ? x.analysis.quality.toFixed(1) + ' /100' : 'unmeasured'}
            </p>
            <p className="text-xs text-gray-400">Measured pixels/resolution; no inferred aesthetic score.</p>
            <label className="block text-xs">
              Caption
              <p className="whitespace-pre-wrap mt-2 text-sm">{x.caption}</p>
            </label>
            <label className="block text-xs">
              Tags
              <input
                aria-label={'Tags ' + x.filename}
                className={inputClass + ' mt-1'}
                value={drafts[x.id]?.tags ?? x.tags.join(', ')}
                onChange={e => changeDraft(x, 'tags', e.target.value)}
              />
            </label>
            <div className="flex flex-wrap gap-2">
              <select
                aria-label={'Category ' + x.filename}
                className={inputClass + ' flex-1 w-auto'}
                value={x.category}
                disabled={!!busy}
                onChange={e =>
                  void run('Saving category', () =>
                    request('edit', { ids: [x.id], patch: { category: e.target.value } }),
                  )
                }
              >
                {['unclassified', ...CATEGORIES].map(c => (
                  <option key={c}>{c}</option>
                ))}
              </select>
              <button
                className={buttonClass}
                disabled={!!busy}
                onClick={() =>
                  void run('Saving pin', () => request('edit', { ids: [x.id], patch: { pinned: x.pinned ? 0 : 1 } }))
                }
              >
                {x.pinned ? 'Unpin' : 'Pin'}
              </button>
            </div>
            {drafts[x.id] && (
              <div className="flex flex-wrap gap-2">
                <button
                  className={buttonClass}
                  disabled={!!busy}
                  onClick={() =>
                    void run('Saving caption/tags', async () => {
                      await request('edit', {
                        ids: [x.id],
                        patch: {
                          baseRevision: drafts[x.id].revision,

                          tags: drafts[x.id].tags
                            .split(',')
                            .map(x => x.trim())
                            .filter(Boolean),
                        },
                      });
                      setDrafts(d => {
                        const copy = { ...d };
                        delete copy[x.id];
                        return copy;
                      });
                    })
                  }
                >
                  Salva tag
                </button>
                <button
                  className={buttonClass}
                  onClick={() =>
                    setDrafts(d => {
                      const copy = { ...d };
                      delete copy[x.id];
                      return copy;
                    })
                  }
                >
                  Scarta bozza tag
                </button>
                <span className="text-xs text-amber-300">Unsaved draft retained</span>
              </div>
            )}
          </article>
        ))}
      </div>
    );
  }
  return (
    <>
      <section>
        <div className="max-w-[1600px] mx-auto p-4 sm:p-6 space-y-6 min-w-0">
          <div className="flex flex-wrap justify-between gap-3">
            <div>
              <h2 className="text-xl">Operazioni avanzate</h2>
              <button className={buttonClass} onClick={onNativeView}>
                Vista media nativa
              </button>
              <p className="text-sm text-gray-400">
                Originals preserved. Checkboxes select temporary actions; Include/Exclude controls training membership.
              </p>
            </div>
            <button className={buttonClass} onClick={() => openImagesModal(dataset, refresh)}>
              Add images + paired TXT
            </button>
          </div>
          <nav className="flex flex-wrap gap-2">
            {['Metadati', 'Selection / Export', 'Templates', 'Training'].map(t => (
              <button
                key={t}
                className={buttonClass + (tab === t ? ' border-blue-400 text-blue-300' : '')}
                onClick={() => setTab(t)}
              >
                {t}
              </button>
            ))}
          </nav>
          <div aria-live="polite">
            {busy && <p className="text-blue-300">{busy}…</p>}
            {error && (
              <div role="alert" className="text-red-300 bg-red-950/50 p-3 rounded">
                {error}
                <button className={buttonClass + ' ml-3'} onClick={() => void refresh()}>
                  Refresh server state
                </button>
              </div>
            )}
            {!state && !error && <p>Loading dataset…</p>}
          </div>
          {state && tab === 'Metadati' && (
            <>
              <section className="border border-gray-700 rounded-lg p-4 space-y-3">
                <h3>View filters</h3>
                <div className="grid gap-3 grid-cols-1 sm:grid-cols-3">
                  <input
                    className={inputClass}
                    placeholder="Search filename / caption"
                    aria-label="Search"
                    value={filter}
                    onChange={e => setFilter(e.target.value)}
                  />
                  <select
                    className={inputClass}
                    aria-label="Category filter"
                    value={category}
                    onChange={e => setCategory(e.target.value)}
                  >
                    {['all', 'unclassified', ...CATEGORIES].map(c => (
                      <option key={c}>{c}</option>
                    ))}
                  </select>
                  <select
                    className={inputClass}
                    aria-label="Membership filter"
                    value={membership}
                    onChange={e => setMembership(e.target.value)}
                  >
                    {['all', 'included', 'excluded', 'removed'].map(c => (
                      <option key={c}>{c}</option>
                    ))}
                  </select>
                  <input
                    className={inputClass}
                    placeholder="Filter tag"
                    aria-label="Tag filter"
                    value={tag}
                    onChange={e => setTag(e.target.value)}
                  />
                  <label className="text-sm">
                    Minimum measured score
                    <input
                      className={inputClass}
                      type="number"
                      min={0}
                      max={100}
                      value={minScore}
                      onChange={e => setMinScore(Number(e.target.value))}
                    />
                  </label>
                  <label className="text-sm">
                    Grid card size: {cardSize}px
                    <input
                      className="w-full"
                      aria-label="Grid card size"
                      type="range"
                      min={240}
                      max={560}
                      step={20}
                      value={cardSize}
                      onChange={e => setCardSize(Number(e.target.value))}
                    />
                  </label>
                </div>
              </section>
              <button
                className={buttonClass}
                disabled={!!busy}
                onClick={() =>
                  void run('Misurazione pixel', async () => {
                    for (const x of current.current.images.filter((i: Image) => !i.analysis))
                      await request('analyze', { id: x.id });
                  })
                }
              >
                Misura qualità e similarità dai pixel
              </button>
              <h3>Included originals · {visible.filter(x => !x.excluded && !x.discarded).length}</h3>
              {cards(visible.filter(x => !x.excluded && !x.discarded))}
              {!!visible.filter(x => x.excluded || x.discarded).length && (
                <section className="space-y-4 border-t border-gray-600 pt-5">
                  <h3>Excluded / removed originals · not used for training</h3>
                  {cards(visible.filter(x => x.excluded || x.discarded))}
                </section>
              )}
              <section className="border border-gray-700 rounded-lg p-4 space-y-3">
                <h3>Local model captioning</h3>
                <p className="text-sm text-gray-400">
                  Uses AI Toolkit’s native image captioner catalog. Creates a stopped draft; native Start is explicit.
                  No API key. Generated captions stay scoped until you confirm application.
                </p>
                <div className="grid gap-3 sm:grid-cols-3">
                  <select
                    aria-label="Captioner"
                    className={inputClass}
                    value={captioner}
                    onChange={e => {
                      const c = imageCaptioners.find(x => x.name === e.target.value)!;
                      setCaptioner(c.name);
                      setCaptionModel(String(c.name_or_path_options![0].value));
                    }}
                  >
                    {imageCaptioners.map(x => (
                      <option key={x.name} value={x.name}>
                        {x.label}
                      </option>
                    ))}
                  </select>
                  <select
                    aria-label="Caption model"
                    className={inputClass}
                    value={captionModel}
                    onChange={e => setCaptionModel(e.target.value)}
                  >
                    {chosenCaptioner.name_or_path_options?.map(x => (
                      <option key={String(x.value)} value={String(x.value)}>
                        {x.label}
                      </option>
                    ))}
                  </select>
                  <select
                    aria-label="Caption scope"
                    className={inputClass}
                    value={captionScope}
                    onChange={e => setCaptionScope(e.target.value)}
                  >
                    <option value="included">Included in training only</option>
                    <option value="all">Tutte, anche escluse (solo bozza manuale)</option>
                  </select>
                </div>
                <textarea
                  aria-label="Caption instructions"
                  className={inputClass}
                  rows={4}
                  value={instructions}
                  onChange={e => setInstructions(e.target.value)}
                />
                <button
                  className={buttonClass}
                  disabled={!!busy}
                  onClick={() =>
                    void run('Preparing native caption draft', () =>
                      request('job', {
                        kind: 'caption',
                        scope: captionScope,
                        ids:
                          captionScope === 'all'
                            ? state.images.filter(x => !x.discarded).map(x => x.id)
                            : state.images.filter(x => !x.excluded && !x.discarded).map(x => x.id),
                        captioner,
                        model: captionModel,
                        instructions,
                        gpu,
                      }),
                    )
                  }
                >
                  Create scoped caption draft
                </button>
              </section>
            </>
          )}
          {state && tab === 'Selection / Export' && (
            <section className="space-y-4">
              <h3>Equal thirds: face / body / variety</h3>
              <p className="text-sm text-gray-400">
                Categories are manual; measured quality and similarity sort candidates. Pins are explicit overrides.
                Unsatisfied quotas block export.
              </p>
              <div className="grid gap-3 sm:grid-cols-4">
                <label>
                  Training images N
                  <input
                    className={inputClass}
                    type="number"
                    min={1}
                    max={1000}
                    value={target}
                    onChange={e => setTarget(Number(e.target.value))}
                  />
                </label>
                <label>
                  Bucket tier
                  <select className={inputClass} value={tier} onChange={e => setTier(Number(e.target.value))}>
                    {[512, 768, 1024].map(n => (
                      <option key={n}>{n}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Fit
                  <select className={inputClass} value={crop} onChange={e => setCrop(e.target.value)}>
                    <option value="fit">Native bucket fit</option>
                    <option value="center">Center crop</option>
                  </select>
                </label>
                <label>
                  Private HF repository
                  <input className={inputClass} value={repo} onChange={e => setRepo(e.target.value)} />
                </label>
              </div>
              <details className="border border-gray-700 p-4 rounded">
                <summary>Test samples (optional)</summary>
                <p className="text-sm my-3">
                  Reference/prompt for comparison; excluded from training; leave empty for normal training.
                </p>
                <select className={inputClass} value={testId} onChange={e => setTestId(e.target.value)}>
                  <option value="">None</option>
                  {state.images.map(x => (
                    <option key={x.id} value={x.id}>
                      {x.filename}
                    </option>
                  ))}
                </select>
                <textarea
                  aria-label="Test prompt"
                  className={inputClass + ' mt-3'}
                  value={testPrompt}
                  onChange={e => setTestPrompt(e.target.value)}
                />
              </details>
              <div className="flex flex-wrap gap-2">
                <button
                  className={buttonClass}
                  disabled={!!busy}
                  onClick={() => void run('Saving selection settings', settingsSave)}
                >
                  Save selection settings
                </button>
                <button
                  className={buttonClass}
                  disabled={!!busy}
                  onClick={() => void run('Preparing immutable export', exportVersion)}
                >
                  Export ordered JPEG + TXT snapshot
                </button>
              </div>
              <p>
                Saved selection: {selected?.selected.length}/{state.settings.count} ·{' '}
                {selected?.complete ? 'Ready' : 'Unsatisfied'}
              </p>
              <pre className="whitespace-pre-wrap break-words text-sm text-amber-200">
                {JSON.stringify(
                  { quotas: selected?.quotas, deficits: selected?.deficits, conflicts: selected?.conflicts },
                  null,
                  2,
                )}
              </pre>
              {state.snapshots.map(v => (
                <div key={v.id} className="p-4 border border-gray-700 rounded space-y-2 break-words">
                  <p>
                    {v.state} · {v.source.length} images · {v.digest ?? v.id}
                  </p>
                  <p className="text-sm">
                    HF:{' '}
                    {v.hf?.phase ??
                      (state.hfConfigured ? 'ready to configure sync' : 'server HF_TOKEN configuration required')}{' '}
                    {v.hf?.revision ?? ''}
                  </p>
                  {v.state === 'complete' && (
                    <div className="flex flex-wrap gap-2">
                      <button
                        className={buttonClass}
                        onClick={() =>
                          void run('Downloading ZIP', async () => {
                            const r = await apiClient.get('/api/dataset-studio', {
                              params: { dataset, download: v.id },
                              responseType: 'blob',
                            });
                            const url = URL.createObjectURL(r.data);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = 'dataset-studio-' + v.digest!.slice(0, 12) + '.zip';
                            a.click();
                            setTimeout(() => URL.revokeObjectURL(url), 1000);
                          })
                        }
                      >
                        Download ZIP
                      </button>
                      <button
                        className={buttonClass}
                        disabled={!!busy || !state.hfConfigured}
                        onClick={() => void run('Syncing / verifying HF', () => syncVersion(v.id))}
                      >
                        Sync / reconcile / read back HF
                      </button>
                      <button
                        className={buttonClass}
                        onClick={() => {
                          setVersionId(v.id);
                          setTab('Training');
                        }}
                      >
                        Use for training draft
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </section>
          )}
          {state && tab === 'Templates' && (
            <section className="space-y-4">
              <h3>Subject templates · approvals are immutable</h3>
              <p className="text-sm text-gray-400">
                Review model architecture, target and learning recipe. This editor starts with the native catalog’s
                Krea2 Raw draft; nothing is approved until you click Approve template. Later edits create a new
                approval.
              </p>
              <div className="grid gap-3 sm:grid-cols-3">
                <label>
                  Template name
                  <input className={inputClass} value={templateName} onChange={e => setTemplateName(e.target.value)} />
                </label>
                <label>
                  Subject
                  <input className={inputClass} value={subject} onChange={e => setSubject(e.target.value)} />
                </label>
                <label>
                  Trigger
                  <input className={inputClass} value={trigger} onChange={e => setTrigger(e.target.value)} />
                </label>
              </div>
              <p className="text-sm">
                Captioner: {chosenCaptioner.label} · {captionModel}. Selection: N{target}, tier{tier}. Caption
                instructions below are part of approval.
              </p>
              <textarea
                aria-label="Template caption instructions"
                className={inputClass}
                rows={4}
                value={instructions}
                onChange={e => setInstructions(e.target.value)}
              />
              <label>
                Reviewed native training configuration
                <textarea
                  aria-label="Reviewed native training configuration"
                  className={inputClass + ' font-mono text-xs mt-2'}
                  rows={18}
                  value={training}
                  onChange={e => setTraining(e.target.value)}
                />
              </label>
              <button
                className={buttonClass}
                disabled={!!busy}
                onClick={() =>
                  void run('Saving unapproved draft', () =>
                    request('draft', {
                      draft: {
                        name: templateName,
                        subject,
                        trigger,
                        training,
                        captioner,
                        captionModel,
                        instructions,
                        previous,
                      },
                    }),
                  )
                }
              >
                Save draft on server
              </button>
              <button
                className={buttonClass + ' ml-3'}
                disabled={!!busy}
                onClick={() => void run('Approving immutable template', approveTemplate)}
              >
                Approve template
              </button>
              {state.templates.map(t => (
                <div key={t.id} className="border border-gray-700 p-4 rounded space-y-2">
                  <p>
                    {t.name} · {t.subject} · approved {t.approvedAt}
                  </p>
                  <p className="text-xs break-all">
                    {t.id} · {t.training.config.process[0].model.arch} · target{' '}
                    {t.training.config.process[0].train.steps}
                  </p>
                  <div className="flex gap-3 flex-wrap">
                    <button className={buttonClass} onClick={() => applyTemplate(t)}>
                      Apply / edit draft for new approval
                    </button>
                    <button className={buttonClass} onClick={() => applyTemplate(t, true)}>
                      Duplicate draft
                    </button>
                  </div>
                </div>
              ))}
            </section>
          )}
          {state && tab === 'Training' && (
            <section className="space-y-4">
              <h3>Native jobs · actual database state</h3>
              <p className="text-sm text-gray-400">
                Create stopped drafts only. Inspect the native job page and explicitly Start on your configured GPU
                host. Existing campaign jobs are not inferred or controlled here.
              </p>
              {state.preview && (
                <p className="text-amber-200">
                  Local verification preview: Start/inference disabled; native job draft preparation is available.
                </p>
              )}
              <div className="grid gap-3 sm:grid-cols-3">
                <label>
                  Completed bucket snapshot
                  <select className={inputClass} value={versionId} onChange={e => setVersionId(e.target.value)}>
                    <option value="">Choose version</option>
                    {state.snapshots
                      .filter(v => v.state === 'complete')
                      .map(v => (
                        <option key={v.id} value={v.id}>
                          {v.digest?.slice(0, 16)} · {v.source.length} pairs
                        </option>
                      ))}
                  </select>
                </label>
                <label>
                  Approved template
                  <select className={inputClass} value={approvedId} onChange={e => setApprovedId(e.target.value)}>
                    <option value="">Choose approval</option>
                    {state.templates.map(t => (
                      <option key={t.id} value={t.id}>
                        {t.name} · {t.approvedAt}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Native GPU IDs
                  <input className={inputClass} value={gpu} onChange={e => setGpu(e.target.value)} />
                </label>
              </div>
              <button
                className={buttonClass}
                disabled={!!busy || !versionId || !approvedId}
                onClick={() =>
                  void run('Creating / reconciling native training draft', () =>
                    request('job', { kind: 'train', version: versionId, template: approvedId, gpu }),
                  )
                }
              >
                Create / reconcile training draft
              </button>
              {state.jobs
                .filter(x => x.state === 'unknown' || x.state === 'intent')
                .map(x => (
                  <p key={x.name} className="text-amber-200 break-all">
                    {x.name}: {x.state}; reselect exact version/template/scope to reconcile, never a replacement launch.
                  </p>
                ))}
              {state.jobsLive.map(job => {
                const link = state.jobs.find(x => x.jobId === job.id);
                const cfg = JSON.parse(job.job_config),
                  p = cfg.config.process[0];
                return (
                  <div key={job.id} className="p-4 border border-gray-700 rounded space-y-2">
                    <p className="break-all">
                      {job.name} · {job.job_type} · {job.status}
                    </p>
                    <p>
                      Step {job.step}/{job.total_steps ?? p.train?.steps ?? 'unknown'} · {job.speed_string} · {job.info}
                    </p>
                    <p className="text-xs break-all">
                      Dataset/version {link?.version ?? 'native caption dataset'} · approval {link?.template ?? '—'}
                    </p>
                    <div className="flex flex-wrap gap-2">
                      <Link className={buttonClass} href={'/jobs/' + job.id}>
                        Review native config / controls
                      </Link>
                      {link?.kind === 'caption' && !link.automatic && (
                        <>
                          <button
                            className={buttonClass}
                            onClick={() =>
                              openCaptionDatasetModal(link.folder!, refresh, { jobId: job.id, draftOnly: true })
                            }
                          >
                            Edit native local caption draft
                          </button>
                          <button
                            className={buttonClass}
                            disabled={!!busy}
                            onClick={() => {
                              if (
                                confirm(
                                  'Apply available generated captions to this exact scope? Existing manual edits since preparation will be preserved.',
                                )
                              )
                                void run('Applying scoped native captions', () =>
                                  request('applyCaption', { name: link.name, overwrite: true }),
                                );
                            }}
                          >
                            Apply generated captions (confirm overwrite)
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                );
              })}
            </section>
          )}
        </div>
      </section>
    </>
  );
}
