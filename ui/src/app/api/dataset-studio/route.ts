import { syncStatus } from '@/datasetStudio/managedSync';
import { freeBytes, reserveBytes, ANALYSIS_OUTPUT_BYTES } from '@/datasetStudio/space';
import { analysisProgress } from '@/datasetStudio/analysisProgress';
import { executeAnalysisCommand, reconcileAnalysis, analysisConfiguration } from '@/datasetStudio/analysisFlow';
import { NextResponse } from 'next/server';
import { getDataRoot, getDatasetsRoot, getHFToken, getTrainingFolder } from '@/server/settings';
import { StudioStore } from '@/datasetStudio/store';
import { Problem, ensure, settings, select } from '@/datasetStudio/domain';
import { body } from '@/datasetStudio/http';
import prisma from '@/server/prisma';
import fs from 'node:fs/promises';
import path from 'node:path';
import archiver from 'archiver';
import sharp from 'sharp';
import { PassThrough, Readable } from 'node:stream';
import { createJob, applyCaption, captionConfig } from '@/datasetStudio/jobs';
import { sync } from '@/datasetStudio/sync';
import { generateCaption, reconcileCaption, dismissCaption, captionHost } from '@/datasetStudio/captionFlow';
import os from 'node:os';
import { TOOLKIT_ROOT } from '@/paths';
export const runtime = 'nodejs';
async function store(name: string) {
  const st = await new StudioStore(await getDataRoot(), await getDatasetsRoot(), name).init();
  ensure(!(await fs.lstat(path.join(st.datasetRoot,'.studio-materializing.json')).catch((e: any)=>{if(e.code==='ENOENT')return null;throw e;})), 'Importazione in corso: attendi il completamento o riapri dal catalogo per riprendere',409);
  return st;
}
async function view(s: any, st: StudioStore) {
  const jobs = await prisma.job.findMany({
    where: {
      OR: [
        { id: { in: s.jobs.map((x: any) => x.jobId).filter(Boolean) } },
        { job_type: 'caption', job_ref: st.datasetRoot },
      ],
    },
    orderBy: { created_at: 'desc' },
  });
  const config = (await analysisConfiguration()).config;
  const missing = s.images.filter((x: any) => !x.discarded && x.analysis?.model?.config !== config);
  const space = { free: await freeBytes(st.folder), reserve: reserveBytes(), missingBytes: missing.reduce((n: number,x: any) => n+x.size,0), outputBytes: missing.length*ANALYSIS_OUTPUT_BYTES };
  return {
    ...s,
    managedSync: await syncStatus(st,s,!!(await getHFToken())),
    space,
    analysisProgress: await analysisProgress(st, s, jobs, (await analysisConfiguration()).config),
    jobsLive: jobs,
    selection: select(s.images, s.settings.count, s.settings.testId),
    hfConfigured: !!(await getHFToken()),
    preview: process.env.DATASET_STUDIO_PREVIEW === '1',
    analysisEnabled: process.env.DATASET_STUDIO_ANALYSIS_ENABLED === '1',
    captionHostSupported: process.env.DATASET_STUDIO_PREVIEW !== '1' && os.platform() === 'linux',
  };
}
function error(e: any) {
  return NextResponse.json(
    { error: e instanceof Problem ? e.message : 'Dataset operation failed; originals remain preserved' },
    { status: e instanceof Problem ? e.status : 500 },
  );
}
export async function GET(request: Request) {
  try {
    const q = new URL(request.url).searchParams,
      st = await store(q.get('dataset')!);
    const s = q.has('image') ? await st.raw() : await st.read();
    if (q.has('download')) {
      const v = s.snapshots.find(x => x.id === q.get('download'));
      ensure(v?.state === 'complete', 'Snapshot not complete', 409);
      const output = new PassThrough(),
        archive = archiver('zip', { zlib: { level: 6 } });
      archive.on('error', e => output.destroy(e));
      archive.pipe(output);
      for (const f of [...v.files, { path: 'manifest.json' }])
        archive.file(await st.file(v.id, f.path), { name: f.path });
      void archive.finalize();
      return new Response(Readable.toWeb(output) as ReadableStream, {
        headers: {
          'Content-Type': 'application/zip',
          'Content-Disposition': 'attachment; filename=dataset-studio-' + v.digest!.slice(0, 12) + '.zip',
        },
      });
    }
    if (q.has('image')) {
      const x = s.images.find(x => x.id === q.get('image'));
      ensure(x, 'Image not found', 404);
      return new Response(
        new Uint8Array(
          await sharp(await st.source(x))
            .rotate()
            .resize({ width: 640, height: 640, fit: 'inside', withoutEnlargement: true })
            .jpeg({ quality: 85 })
            .toBuffer(),
        ),
        { headers: { 'Content-Type': 'image/jpeg', 'Cache-Control': 'private, no-store' } },
      );
    }
    return NextResponse.json(await view(s, st));
  } catch (e) {
    return error(e);
  }
}
export async function POST(request: Request) {
  try {
    const x = await body(request),
      st = await store(x.dataset);
    let s;
    switch (x.action) {
      case 'title':
        s = await st.title(x.revision, x.title);
        break;
      case 'captionDraft':
        s = await st.captionDraft(x.revision, x.id, x.draft, x.draftRevision);
        break;
      case 'captionPreferences':
        s = await st.captionSettings(x.revision, x.preferences, x.preferencesRevision);
        break;
      case 'generateCaption':
        s = await generateCaption(
          st,
          x.revision,
          x,
          prisma,
          process.env.DATASET_STUDIO_DB_URL?.replace(/^file:/, '') ?? path.join(TOOLKIT_ROOT, 'aitk_db.db'),
          await captionHost(),
        );
        break;
      case 'reconcileCaption':
        s = await reconcileCaption(st, x.id, prisma, await captionHost(), x.blockedIds ?? [], x.explicit === true);
        break;
      case 'dismissCaption':
        s = await dismissCaption(st, x.revision, x.id);
        break;
      case 'edit':
        s = await st.edit(x.revision, x.ids, x.patch);
        break;
      case 'settings':
        s = await st.mutate(x.revision, s => {
          s.settings = settings(x.settings);
        });
        break;
      case 'automaticAnalysis':
        s = await executeAnalysisCommand(st, prisma,
          process.env.DATASET_STUDIO_DB_URL?.replace(/^file:/, '') ?? path.join(TOOLKIT_ROOT, 'aitk_db.db'),
          await captionHost(), { revision: x.revision, count: x.count, retry: x.retry === true });
        break;
      case 'reconcileAnalysis':
        s = await reconcileAnalysis(st, x.id, prisma, await captionHost(), x.retry === true);
        break;
      case 'analyze':
        s = await st.analyze(x.revision, x.id);
        break;
      case 'draft':
        s = await st.draft(x.revision, x.draft);
        break;
      case 'approve':
        captionConfig(
          x.template.captioner,
          x.template.captionModel,
          x.template.settings.instructions,
          'scope',
          'draft',
        );
        s = await st.approve(x.revision, x.template);
        break;
      case 'apply':
        s = await st.mutate(x.revision, s => {
          const t = s.templates.find(t => t.id === x.id);
          ensure(t, 'Approval not found');
          s.settings = structuredClone(t.settings);
          for (const saved of t.curation ?? []) {
            const image = s.images.find(x => x.id === saved.id && x.sha === saved.sha);
            if (image) {
              Object.assign(image, {
                category: saved.category,
                categorySource: 'manual',
                reviewRevision: (image.reviewRevision ?? 0) + 1,
                tags: [...saved.tags],
                pinned: saved.pinned,
                excluded: saved.excluded,
                discarded: saved.discarded,
              });
              image.revision++;
            }
          }
        });
        break;
      case 'export':
        s = await st.prepareExport(x.revision);
        break;
      case 'exportFile':
        s = await st.exportFile(x.revision, x.id, x.index);
        break;
      case 'exportFinish':
        s = await st.finishExport(x.revision, x.id);
        break;
      case 'job':
        s = await createJob(
          st,
          x.revision,
          x,
          prisma,
          await getTrainingFolder(),
          process.env.DATASET_STUDIO_DB_URL?.replace(/^file:/, '') ?? path.join(TOOLKIT_ROOT, 'aitk_db.db'),
        );
        break;
      case 'applyCaption':
        s = await applyCaption(st, x.revision, x.name, x.overwrite, prisma);
        break;
      case 'hfStart':
      case 'hfUpload':
      case 'hfCommit':
      case 'hfVerify':
      case 'hfReconcile':
        s = await sync(st, x.revision, x, await getHFToken());
        break;
      default:
        throw new Problem(400, 'Unsupported dataset action');
    }
    return NextResponse.json(await view(s, st));
  } catch (e) {
    return error(e);
  }
}
