import fs from 'node:fs/promises';
import path from 'node:path';
import { captionerTypes } from '@/helpers/captionOptions';
import { defaultCaptionJobConfig } from '@/helpers/captionJobConfig';
import { ensure, stableJSON, text } from './domain';
import { StudioStore, JobLink, atomic, contained, hash, validateTraining } from './store';
export function captionConfig(type: string, model: string, instructions: string, folder: string, name: string) {
  const option = captionerTypes.find(x => x.name === type && x.group.includes('image'));
  ensure(option, 'Choose a native image captioner');
  ensure(
    option.name_or_path_options?.some(x => x.value === model),
    'Model is not in native captioner catalog',
  );
  const cfg: any = structuredClone(defaultCaptionJobConfig);
  cfg.config.name = name;
  const p = cfg.config.process[0];
  p.type = option.name;
  for (const [key, value] of Object.entries(option.defaults ?? {})) {
    const prefix = 'config.process[0].caption.';
    if (key.startsWith(prefix)) p.caption[key.slice(prefix.length)] = (value as any[])[0];
  }
  p.device = 'cuda';
  p.caption.model_name_or_path = model;
  p.caption.path_to_caption = folder;
  p.caption.caption_prompt = text(instructions, 8000);
  p.caption.extensions = ['jpg', 'jpeg', 'png', 'webp'];
  p.caption.caption_extension = 'txt';
  p.caption.recaption = true;
  return cfg;
}
export async function reconcileJob(db: any, link: JobLink, gpu: string, dataset: string) {
  let existing = await db.job.findUnique({ where: { name: link.name } });
  if (!existing) {
    try {
      existing = await db.job.create({
        data: {
          name: link.name,
          gpu_ids: gpu,
          job_config: JSON.stringify(link.config),
          job_type: link.kind,
          job_ref: dataset,
          status: 'stopped',
          total_steps: link.kind === 'train' ? link.config.config.process[0].train.steps : null,
        },
      });
    } catch (error) {
      existing = await db.job.findUnique({ where: { name: link.name } });
      if (!existing) throw error;
    }
  }
  ensure(
    existing.job_type === link.kind &&
      existing.job_ref === dataset &&
      stableJSON(JSON.parse(existing.job_config)) === stableJSON(link.config),
    'Native job identity/config differs; reconcile without replacement',
    409,
  );
  return existing;
}
export async function createJob(
  st: StudioStore,
  revision: number,
  input: any,
  db: any,
  trainingFolder: string,
  sqlite: string,
) {
  const kind = input.kind;
  ensure(kind === 'train' || kind === 'caption', 'Invalid job kind');
  let link: JobLink;
  let s = await st.mutate(revision, async s => {
    let cfg: any, version: string | undefined, template: string | undefined, scope: any, folder: string | undefined;
    if (kind === 'train') {
      const v = s.snapshots.find(x => x.id === input.version),
        t = s.templates.find(x => x.id === input.template);
      ensure(v?.state === 'complete' && v.digest, 'Choose a completed immutable bucket version');
      ensure(t, 'Approve a reviewed template first');
      ensure(
        v.manifest.settings.tier === t.settings.tier &&
          v.manifest.settings.crop === t.settings.crop &&
          v.source.length === t.settings.count,
        'Snapshot N/tier/fit differs from approved template; apply/export or approve the new settings',
        409,
      );
      await st.verifySnapshot(v);
      cfg = validateTraining(t.training);
      version = v.id;
      template = t.id;
      const p = cfg.config.process[0];
      p.datasets = [
        {
          ...(p.datasets?.[0] ?? {}),
          mask_path: null,
          control_path: null,
          folder_path: path.join(st.folder, 'versions', v.id, 'training'),
          caption_ext: 'txt',
          resolution: [v.manifest.settings.tier],
          cache_latents_to_disk: false,
        },
      ];
      p.trigger_word = t.trigger;
      p.training_folder = trainingFolder;
      p.sqlite_db_path = sqlite;
      p.device = 'cuda';
      ensure(!p.train.start_step && !p.train.resume, 'New native job cannot silently resume an obsolete checkpoint');
    } else {
      const ids = input.ids;
      ensure(
        Array.isArray(ids) && ids.length > 0 && ids.length <= 1000 && new Set(ids).size === ids.length,
        'Select a caption scope',
      );
      scope = ids.map((id: string) => {
        const x = s.images.find(x => x.id === id);
        ensure(x && !x.discarded, 'Caption source not found');
        return structuredClone(x);
      });
      if (input.scope === 'included')
        ensure(
          scope.every((x: any) => !x.excluded),
          'Included caption scope contains excluded files',
        );
      const scopeId = hash(
        stableJSON({
          ids: scope.map((x: any) => ({ id: x.id, sha: x.sha, caption: x.caption })),
          type: input.captioner,
          model: input.model,
          instructions: input.instructions,
        }),
      );
      folder = path.join(st.folder, 'caption', scopeId);
      await contained(st.folder, folder, true);
      await fs.mkdir(folder, { recursive: true });
      for (const [i, img] of scope.entries()) {
        const stem = String(i + 1).padStart(6, '0');
        await atomic(path.join(folder, stem + path.extname(img.filename).toLowerCase()), await st.source(img));
      }
      cfg = captionConfig(input.captioner, input.model, input.instructions, folder!, 'draft');
      cfg.config.process[0].sqlite_db_path = sqlite;
    }
    const identity = hash(stableJSON({ kind, version, template, scope: scope?.map((x: any) => x.id), cfg }));
    const name = 'studio-' + kind + '-' + identity.slice(0, 24);
    cfg.config.name = name;
    link = { name, kind, state: 'intent', version, template, config: cfg, scope, folder };
    const old = s.jobs.find(x => x.name === name);
    if (old) {
      ensure(stableJSON(old.config) === stableJSON(cfg), 'Existing job config differs');
      link = old;
    } else s.jobs.push(link);
  });
  try {
    const job = await reconcileJob(db, link!, text(input.gpu ?? '0', 50), st.datasetRoot);
    s = await st.mutate(s.revision, s => {
      const x = s.jobs.find(x => x.name === link.name)!;
      x.state = 'linked';
      x.jobId = job.id;
    });
  } catch (error) {
    await st.mutate(s.revision, s => {
      s.jobs.find(x => x.name === link.name)!.state = 'unknown';
    });
    throw error;
  }
  return s;
}
export async function applyCaption(st: StudioStore, revision: number, name: string, overwrite: boolean, db: any) {
  ensure(overwrite === true, 'Confirm replacing manual captions explicitly');
  const s = await st.read();
  ensure(s.revision === revision, 'Stale caption application', 409);
  const link = s.jobs.find(x => x.name === name && x.kind === 'caption');
  ensure(link?.folder && link.scope, 'Caption scope missing');
  const job = await db.job.findUnique({ where: { name: link.name } });
  ensure(
    job && job.status === 'completed' && job.step === link.scope.length && job.total_steps === link.scope.length,
    'Native caption job must complete its full exact scope before application',
  );
  ensure(
    JSON.parse(job.job_config).config.process[0].caption.path_to_caption === link.folder,
    'Native caption job scope changed; application refused',
  );
  const generated: string[] = [];
  for (const [i, img] of link.scope.entries()) {
    await st.source(img);
    const file = await contained(st.folder, path.join(link.folder, String(i + 1).padStart(6, '0') + '.txt'));
    const bytes = await fs.readFile(file);
    ensure(bytes.length <= 64000, 'Generated caption too large');
    ensure(bytes.toString('utf8').trim().length > 0, 'Native caption output is empty; no partial scope application');
    generated.push(bytes.toString('utf8'));
  }
  return st.mutate(revision, s => {
    for (const [i, img] of link.scope!.entries()) {
      const current = s.images.find(x => x.id === img.id);
      ensure(
        current && current.caption === img.caption,
        'A manual caption changed after this job; refresh and preserve it',
        409,
      );
    }
    for (const [i, img] of link.scope!.entries()) {
      const current = s.images.find(x => x.id === img.id)!;
      current.caption = generated[i];
      current.captionOverride = true;
      current.caption_source = 'native caption job ' + job.id;
      current.revision++;
      current.captionRevision = (current.captionRevision ?? 0) + 1;
    }
  });
}
