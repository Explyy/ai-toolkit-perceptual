'use client';
import { useEffect, useState } from 'react';
import Link from 'next/link';
import { apiClient } from '@/utils/api';
import type { CatalogEntry } from './catalog';
import { sha256, stableJSON } from './domain';
export default function Catalog() {
  const [data, setData] = useState<any>(),
    [entries, setEntries] = useState<Record<string, CatalogEntry[]>>({}),
    [busy, setBusy] = useState(''),
    [error, setError] = useState(''),
    [progress, setProgress] = useState<any>();
  const load = () =>
    apiClient
      .get('/api/dataset-studio/catalog')
      .then(r => setData(r.data))
      .catch(() => setError('Catalogo non disponibile. I dataset locali restano accessibili.'));
  useEffect(() => {
    void load();
  }, []);
  async function open(repo: string) {
    setBusy(repo);
    setError('');
    try {
      const r = await apiClient.get('/api/dataset-studio/catalog', { params: { repo } });
      setEntries(x => ({ ...x, [repo]: r.data.folders }));
    } catch (e: any) {
      setError(e.response?.data?.error ?? 'Cartelle HF non disponibili');
    } finally {
      setBusy('');
    }
  }
  async function importDataset(entry: CatalogEntry) {
    setBusy(entry.repo + '/' + entry.folder);
    setError('');
    setProgress({ done: 0, phase: 'Preparazione importazione' });
    const id = await sha256(
      new TextEncoder().encode(
        stableJSON({ repo: entry.repo, folder: entry.folder, revision: entry.revision, kind: entry.kind }),
      ),
    );
    const poll = setInterval(
      () =>
        void apiClient
          .get('/api/dataset-studio/catalog', { params: { import: id } })
          .then(r => setProgress(r.data))
          .catch(() => {}),
      1000,
    );
    try {
      const result = await apiClient.post('/api/dataset-studio/catalog', { entry });
      await load();
      window.location.assign('/datasets/' + encodeURIComponent(result.data.name));
    } catch (e: any) {
      setError(
        e.response?.data?.error ??
          'Importazione interrotta: sorgenti e lavoro parziale conservati. Riapri per riprendere.',
      );
    } finally {
      clearInterval(poll);
      setBusy('');
      setProgress(undefined);
    }
  }
  return (
    <section aria-label="Dataset Hugging Face" className="mt-6 text-gray-200">
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="text-lg">Hugging Face privato</h2>
        <button className="rounded bg-gray-800 px-3 py-2" onClick={() => void load()} disabled={!!busy}>
          Aggiorna catalogo
        </button>
      </div>
      <p className="my-2 text-sm text-gray-400">
        Le cartelle sono lette dal tuo account. Le immagini vengono scaricate solo quando apri un dataset, mantenendo
        gli originali e la riserva di spazio.
      </p>
      {!data && <p>Caricamento catalogo…</p>}
      {data && !data.configured && <p>Configura il token HF nelle impostazioni per vedere i dataset remoti.</p>}
      {(error || data?.remoteError) && (
        <p role="alert" className="text-amber-200">
          {error || data.remoteError}
        </p>
      )}
      {progress && (
        <p role="status">
          {progress.phase === 'downloading' ? `Importazione ${progress.done}/${progress.total}` : progress.phase}…{' '}
          {progress.missingCaptions ? `${progress.missingCaptions} caption mancanti verranno create vuote. ` : ''}Il
          trasferimento continua sul server.
        </p>
      )}
      {data?.repositories?.map((r: any) => (
        <div key={r.repo} className="my-4">
          <button disabled={!!busy} className="rounded bg-gray-800 px-3 py-2" onClick={() => void open(r.repo)}>
            {r.repo}
            {busy === r.repo ? ' · caricamento…' : ''}
          </button>
          {entries[r.repo]?.map(entry => {
            const cached = data.local?.find(
              (x: any) =>
                x.source?.repo === entry.repo &&
                x.source?.folder === entry.folder &&
                x.source?.revision === entry.revision,
            );
            return (
              <div key={entry.folder} className="my-2 flex flex-wrap items-center gap-3 pl-3">
                <span>
                  {entry.folder || 'Cartella principale'} ·{' '}
                  {entry.kind === 'managed'
                    ? 'dataset salvato'
                    : entry.kind === 'media'
                      ? 'media disponibili'
                      : 'immagini e caption'}
                </span>
                {cached ? (
                  <Link className="text-blue-300" href={'/datasets/' + encodeURIComponent(cached.name)}>
                    Apri copia locale
                  </Link>
                ) : entry.kind === 'media' ? (
                  <span className="text-sm text-amber-200">
                    Formato disponibile nel repository, non importabile nell’analisi immagini
                  </span>
                ) : (
                  <button
                    disabled={!!busy}
                    className="rounded bg-blue-700 px-3 py-1"
                    onClick={() => void importDataset(entry)}
                  >
                    {busy === entry.repo + '/' + entry.folder ? 'Importazione…' : 'Apri dataset'}
                  </button>
                )}
                {!!entry.unsupported && entry.kind !== 'media' && (
                  <span className="text-sm text-amber-200">
                    {entry.unsupported} media di altri formati: restano nel repository, non vengono importati.
                  </span>
                )}
              </div>
            );
          })}
        </div>
      ))}
    </section>
  );
}
