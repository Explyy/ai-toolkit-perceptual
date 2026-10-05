import type { AnalysisSummary } from './analysisFlow';
import { ownedLock } from './ownerLock';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import sharp from 'sharp';
import { CaptionPreferences, preferences } from './captionModels';
import { captionerTypes } from '@/helpers/captionOptions';
import { modelArchs } from '@/app/jobs/new/options';
import {
  Analysis,
  ImageRecord,
  Settings,
  DEFAULT_SETTINGS,
  CATEGORIES,
  ensure,
  integer,
  text,
  settings,
  stableJSON,
  select,
  bucket,
  analyzePixels,
} from './domain';

export type Image = ImageRecord & {
  relative: string;
  tags: string[];
  captionOverride: boolean;
  captionRevision?: number;
  captionDraft?: { caption: string; baseRevision: number; revision: number };
  captionDraftRevision?: number;
};
export type Template = {
  id: string;
  name: string;
  subject: string;
  trigger: string;
  approvedAt: string;
  previous: string | null;
  settings: Settings;
  captioner: string;
  captionModel: string;
  training: any;
  curation?: {
    id: string;
    sha: string;
    category: Image['category'];
    tags: string[];
    pinned: number;
    excluded: number;
    discarded: number;
  }[];
};
export type ExportFile = { path: string; sha: string; size: number; width?: number; height?: number };
export type Snapshot = {
  id: string;
  state: 'building' | 'complete';
  manifest: any;
  source: Image[];
  files: ExportFile[];
  digest?: string;
  hf?: any;
};
export type JobLink = {
  name: string;
  kind: 'train' | 'caption' | 'analysis';
  state: 'intent' | 'linked' | 'unknown';
  jobId?: string;
  version?: string;
  template?: string;
  config: any;
  scope?: Image[];
  folder?: string;
  automatic?: {
    id: string;
    digest: string;
    gpu: string;
    phase:
      | 'prepared'
      | 'enqueue-intent'
      | 'active'
      | 'blocked'
      | 'unknown'
      | 'failed'
      | 'conflict'
      | 'applied'
      | 'dismissed';
    reason?: string;
    queued?: boolean;
    createdAt: string;
    appliedAt?: string;
  };
};
export type TemplateDraft = {
  name: string;
  subject: string;
  trigger: string;
  training: string;
  captioner: string;
  captionModel: string;
  instructions: string;
  previous: string | null;
};
export type State = {
  schema: 1;
  analysisFlow?: AnalysisSummary;
  templateDraft?: TemplateDraft;
  captionPreferences?: CaptionPreferences;
  captionPreferencesRevision?: number;
  revision: number;
  dataset: string;
  images: Image[];
  settings: Settings;
  templates: Template[];
  snapshots: Snapshot[];
  jobs: JobLink[];
};
export const hash = (value: string | Uint8Array) => createHash('sha256').update(value).digest('hex');
export function datasetName(value: unknown) {
  const name = text(value, 128);
  ensure(
    name.trim() === name && name.length > 0 && name !== '.' && name !== '..' && !/[\\/\x00-\x1f]/.test(name),
    'Invalid dataset name',
  );
  return name;
}
export async function contained(root: string, candidate: string, missing = false) {
  const base = await fs.realpath(root),
    requested = path.resolve(candidate);
  ensure(requested.startsWith(base + path.sep) || requested === base, 'Path outside owned storage', 403);
  let current = base;
  for (const part of path.relative(base, requested).split(path.sep).filter(Boolean)) {
    current = path.join(current, part);
    try {
      ensure(!(await fs.lstat(current)).isSymbolicLink(), 'Symlink data access refused', 403);
    } catch (e: any) {
      if (e.code === 'ENOENT' && missing) continue;
      throw e;
    }
  }
  if (!missing)
    ensure(
      (await fs.realpath(requested)).startsWith(base + path.sep) || requested === base,
      'Real path escapes storage',
      403,
    );
  return requested;
}
export function captionPath(image: string, extension: unknown) {
  ensure(typeof extension === 'string' && /^[a-zA-Z0-9]{1,16}$/.test(extension), 'Invalid caption extension');
  ensure(
    !/^(png|jpe?g|webp|gif|bmp|avif|tiff?|mp4|mov|webm|mkv|avi|m4v|wav|mp3|flac|ogg|m4a|aac)$/i.test(extension),
    'Caption extension cannot overwrite original media',
  );
  const target = image.replace(/\.[^/.]+$/, '') + '.' + extension;
  ensure(target !== image, 'Caption cannot overwrite original');
  return target;
}
export async function atomic(file: string, value: string | Uint8Array) {
  const tmp = file + '.' + randomUUID() + '.tmp';
  try {
    const h = await fs.open(tmp, 'wx', 0o600);
    try {
      await h.writeFile(value);
      await h.sync();
    } finally {
      await h.close();
    }
    await fs.rename(tmp, file);
  } finally {
    await fs.rm(tmp, { force: true });
  }
}
export function validateTraining(value: any) {
  ensure(
    value?.job === 'extension' && Array.isArray(value.config?.process) && value.config.process.length === 1,
    'One native extension training process required',
  );
  const p = value.config.process[0];
  ensure(
    p.type === 'diffusion_trainer' && typeof p.model?.arch === 'string' && typeof p.model?.name_or_path === 'string',
    'Review native architecture and model',
  );
  ensure(
    p.model.arch.length <= 80 && p.model.name_or_path.length <= 250 && !/[\x00-\x1f]/.test(p.model.name_or_path),
    'Invalid model',
  );
  const native = modelArchs.find(x => x.name === p.model.arch && x.group === 'image');
  ensure(native, 'Architecture is not in native image model catalog');
  const allowed = [
    native.defaults?.['config.process[0].model.name_or_path']?.[0],
    ...(native.customModelSelectOptions ?? []).flatMap(x => x.options.map(o => o.value)),
  ].filter(Boolean);
  ensure(allowed.includes(p.model.name_or_path), 'Model does not match native architecture catalog');
  if (p.model.arch === 'krea2')
    ensure(p.model.name_or_path === 'krea/Krea-2-Raw', 'Krea2 Raw architecture/model mismatch');
  integer(p.train?.steps, 1, 1000000);
  ensure(
    typeof p.train?.lr === 'number' && Number.isFinite(p.train.lr) && p.train.lr > 0 && p.train.lr <= 1,
    'Review learning rate',
  );
  ensure(['lora', 'lokr'].includes(p.network?.type), 'Review supported native network');
  ensure(
    !p.resume_from && !p.train.resume_from && !p.model.name_or_path.includes('optimizer.pt'),
    'Templates cannot carry obsolete resume paths',
  );
  return structuredClone(value);
}
export class StudioStore {
  root = '';
  folder = '';
  datasetRoot = '';
  constructor(
    private dataRoot: string,
    private datasetsRoot: string,
    public name: string,
  ) {
    datasetName(name);
  }
  async init() {
    await fs.mkdir(this.dataRoot, { recursive: true });
    const base = await fs.realpath(this.dataRoot);
    this.root = path.join(base, 'dataset-studio');
    await contained(base, this.root, true);
    await fs.mkdir(this.root, { recursive: true });
    this.folder = path.join(this.root, hash(this.name));
    await contained(this.root, this.folder, true);
    await fs.mkdir(this.folder, { recursive: true });
    const datasets = await fs.realpath(this.datasetsRoot);
    this.datasetRoot = await contained(datasets, path.join(datasets, this.name));
    await contained(this.folder, path.join(this.folder, 'state.json'), true);
    return this;
  }
  async locked<T>(action: () => Promise<T>) {
    return ownedLock(this.folder, action);
  }

  async raw(): Promise<State> {
    const file = await contained(this.folder, path.join(this.folder, 'state.json'), true);
    try {
      const s = JSON.parse(await fs.readFile(file, 'utf8'));
      ensure(s.schema === 1 && s.dataset === this.name, 'Unsupported state schema');
      return s;
    } catch (e: any) {
      if (e.code !== 'ENOENT') throw e;
      return {
        schema: 1,
        revision: 0,
        dataset: this.name,
        images: [],
        settings: { ...DEFAULT_SETTINGS, count: 3, model: '', repo: 'daverave/Personal' },
        templates: [],
        snapshots: [],
        jobs: [],
      };
    }
  }
  async save(s: State) {
    await atomic(path.join(this.folder, 'state.json'), stableJSON(s));
  }
  async scan(s: State) {
    const previous = new Map(s.images.map(x => [x.id, x]));
    const images: Image[] = [];
    const walk = async (folder: string) => {
      const entries = await fs.readdir(folder, { withFileTypes: true });
      for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
        if (entry.name.startsWith('.') || entry.name === '_controls') continue;
        const file = await contained(this.datasetRoot, path.join(folder, entry.name));
        if (entry.isDirectory()) {
          await walk(file);
          continue;
        }
        if (!entry.isFile() || !/\.(png|jpe?g|webp)$/i.test(entry.name)) continue;
        ensure(images.length < 2500, 'Dataset has more than2500 images; use direct smaller folders');
        const stat = await fs.stat(file);
        ensure(stat.size <= 24 * 1024 * 1024, 'Image exceeds24MB analysis limit');
        const bytes = await fs.readFile(file),
          sha = hash(bytes),
          relative = path.relative(this.datasetRoot, file);
        const id = hash(relative + '\0' + sha);
        const old = previous.get(id);
        const captionPath = await contained(this.datasetRoot, file.replace(/\.[^.]+$/, '.txt'), true);
        let originalCaption = '';
        try {
          ensure((await fs.stat(captionPath)).size < 64000, 'Caption exceeds64KB');
          originalCaption = await fs.readFile(captionPath, 'utf8');
        } catch (e: any) {
          if (e.code !== 'ENOENT') throw e;
        }
        let width = old?.width,
          height = old?.height;
        if (!width || !height) {
          const meta = await sharp(bytes).metadata();
          width = meta.width!;
          height = meta.height!;
          if ((meta.orientation ?? 1) >= 5) [width, height] = [height, width];
        }
        images.push({
          ...old,
          id,
          project: hash(this.name),
          sha,
          filename: entry.name,
          relative,
          mime: /png$/i.test(file) ? 'image/png' : /webp$/i.test(file) ? 'image/webp' : 'image/jpeg',
          size: stat.size,
          width,
          height,
          caption: old?.captionOverride ? old.caption : originalCaption,
          captionOverride: old?.captionOverride ?? false,
          caption_source: old?.captionOverride ? old.caption_source : 'original TXT',
          category: old?.category ?? 'unclassified',
          tags: old?.tags ?? [],
          pinned: old?.pinned ?? 0,
          excluded: old?.excluded ?? 0,
          discarded: old?.discarded ?? 0,
          analysis: old?.analysis ?? null,
          pose: old?.pose ?? null,
          categorySource:
            old?.categorySource ?? (old?.category && old.category !== 'unclassified' ? 'manual' : undefined),
          reviewRevision:
            old?.reviewRevision ??
            (old && (old.category !== 'unclassified' || old.excluded || old.pinned || old.discarded || old.revision > (old.captionRevision ?? 0)) ? 1 : 0),
          analysisStatus: old?.analysisStatus,
          revision: (old?.revision ?? 0) + (old && !old.captionOverride && old.caption !== originalCaption ? 1 : 0),
          captionRevision:
            (old?.captionRevision ?? 0) + (old && !old.captionOverride && old.caption !== originalCaption ? 1 : 0),
          created: old?.created ?? Date.now(),
        });
      }
    };
    await walk(this.datasetRoot);
    if (stableJSON(images) !== stableJSON(s.images)) {
      s.images = images;
      s.revision++;
      await this.save(s);
    }
    return s;
  }
  async read() {
    return this.locked(async () => this.scan(await this.raw()));
  }
  async mutate(rev: number, fn: (s: State) => Promise<void> | void) {
    return this.locked(async () => {
      const s = await this.scan(await this.raw());
      ensure(s.revision === rev, 'Stale dataset revision. Draft retained; refresh before retrying.', 409);
      await fn(s);
      s.revision++;
      await this.save(s);
      return s;
    });
  }
  async source(image: Image) {
    const file = await contained(this.datasetRoot, path.join(this.datasetRoot, image.relative));
    const bytes = await fs.readFile(file);
    ensure(hash(bytes) === image.sha, 'Original image changed; refresh before continuing', 409);
    return bytes;
  }
  async edit(rev: number, ids: string[], patch: any) {
    ensure(
      Array.isArray(ids) && ids.length > 0 && ids.length <= 2500 && new Set(ids).size === ids.length,
      'Select valid unique image identities',
    );
    return this.mutate(rev, s => {
      for (const id of ids) {
        const x = s.images.find(i => i.id === id);
        ensure(x, 'Image changed or disappeared', 409);
        if (patch.baseRevision !== undefined)
          ensure(x.revision === patch.baseRevision, 'Image changed since this caption draft; draft retained', 409);
        if (patch.caption !== undefined) {
          x.caption = text(patch.caption, 64000);
          x.captionOverride = true;
          x.caption_source = 'manual';
          x.captionRevision = (x.captionRevision ?? 0) + 1;
          if (x.captionDraft?.caption === x.caption && x.captionDraft.baseRevision === patch.baseRevision) {
            delete x.captionDraft;
            x.captionDraftRevision = (x.captionDraftRevision ?? 0) + 1;
          }
        }
        if (['category', 'pinned', 'excluded', 'discarded'].some(k => patch[k] !== undefined))
          x.reviewRevision = (x.reviewRevision ?? 0) + 1;
        if (patch.category !== undefined) {
          ensure([...CATEGORIES, 'unclassified'].includes(patch.category), 'Invalid category');
          x.category = patch.category;
          x.categorySource = 'manual';
        }
        if (patch.tags !== undefined) {
          ensure(Array.isArray(patch.tags) && patch.tags.length <= 30, 'Invalid tags');
          x.tags = [...new Set<string>(patch.tags.map((t: any) => text(t, 60).trim()).filter(Boolean))];
        }
        for (const k of ['excluded', 'discarded', 'pinned'] as const)
          if (patch[k] !== undefined) x[k] = integer(patch[k], 0, 1);
        x.revision++;
      }
    });
  }
  async analyze(rev: number, id: string) {
    const s = await this.read();
    ensure(s.revision === rev, 'Stale analysis request', 409);
    const x = s.images.find(x => x.id === id);
    ensure(x, 'Image missing');
    const b = await this.source(x);
    const pixels = await sharp(b).rotate().resize(64, 64, { fit: 'fill' }).ensureAlpha().raw().toBuffer();
    const analysis = analyzePixels(new Uint8ClampedArray(pixels), 64, 64, x.width, x.height);
    return this.mutate(rev, state => {
      const current = state.images.find(i => i.id === id)!;
      current.analysis = { ...analysis, model: current.analysis?.model };
    });
  }
  async captionDraft(rev: number, id: string, value: any, base: number) {
    return this.mutate(rev, s => {
      const image = s.images.find(x => x.id === id);
      ensure(image, 'Immagine non trovata', 404);
      ensure(
        Number.isInteger(base) && base === (image.captionDraftRevision ?? 0),
        'Bozza modificata altrove: nessun overwrite',
        409,
      );
      if (value === null) delete image.captionDraft;
      else {
        ensure(Number.isInteger(value.baseRevision), 'Revisione bozza non valida');
        image.captionDraft = {
          caption: text(value.caption, 64000),
          baseRevision: value.baseRevision,
          revision: base + 1,
        };
      }
      image.captionDraftRevision = base + 1;
    });
  }
  async captionSettings(rev: number, input: any, baseRevision?: number) {
    const value = preferences(input);
    const s = await this.read();
    ensure(s.revision === rev, 'Impostazioni cambiate: aggiorna e riprova', 409);
    if (stableJSON(s.captionPreferences) === stableJSON(value)) return s;
    return this.mutate(rev, s => {
      if (baseRevision !== undefined)
        ensure(
          (s.captionPreferencesRevision ?? 0) === baseRevision,
          'Impostazioni modificate altrove: bozza conservata',
          409,
        );
      s.captionPreferences = value;
      s.captionPreferencesRevision = (s.captionPreferencesRevision ?? 0) + 1;
    });
  }
  async draft(rev: number, input: any) {
    return this.mutate(rev, s => {
      const d: TemplateDraft = {
        name: text(input.name, 100),
        subject: text(input.subject, 100),
        trigger: text(input.trigger, 100),
        training: text(input.training, 200000),
        captioner: text(input.captioner, 100),
        captionModel: text(input.captionModel, 250),
        instructions: text(input.instructions, 8000),
        previous: input.previous ?? null,
      };
      const captioner = captionerTypes.find(x => x.name === d.captioner && x.group.includes('image'));
      ensure(
        captioner?.name_or_path_options?.some(x => x.value === d.captionModel),
        'Draft caption model is not in native catalog',
      );
      ensure(!d.previous || s.templates.some(x => x.id === d.previous), 'Previous approval not found');
      s.templateDraft = d;
    });
  }
  async approve(rev: number, input: any) {
    return this.mutate(rev, s => {
      ensure(input.approve === true, 'Explicit Approve template action required');
      const cfg = validateTraining(input.training);
      const captioner = captionerTypes.find(x => x.name === input.captioner && x.group.includes('image'));
      ensure(
        captioner?.name_or_path_options?.some(x => x.value === input.captionModel),
        'Model is not in native captioner catalog',
      );
      ensure(!input.previous || s.templates.some(t => t.id === input.previous), 'Previous approval not found');
      const approved: Template = {
        id: randomUUID(),
        name: text(input.name, 100),
        subject: text(input.subject, 100),
        trigger: text(input.trigger, 100),
        approvedAt: new Date().toISOString(),
        previous: input.previous ?? null,
        settings: settings(input.settings),
        captioner: text(input.captioner, 100),
        captionModel: text(input.captionModel, 250),
        training: cfg,
        curation: s.images.map(({ id, sha, category, tags, pinned, excluded, discarded }) => ({
          id,
          sha,
          category,
          tags: [...tags],
          pinned,
          excluded,
          discarded,
        })),
      };
      ensure(approved.name.trim() && approved.subject.trim(), 'Name and subject required');
      s.templates.push(approved);
    });
  }
  async prepareExport(rev: number) {
    return this.mutate(rev, s => {
      const result = select(s.images, s.settings.count, s.settings.testId);
      ensure(result.complete, 'Selection has unsatisfied quotas or pins; resolve before export');
      const manifest = {
        schema: 1,
        dataset: this.name,
        settings: s.settings,
        source: result.selected.map(x => ({
          id: x.id,
          sha: x.sha,
          filename: x.filename,
          caption: x.caption,
          category: x.category,
          analysis: x.analysis
            ? {
                ...x.analysis,
                model: x.analysis.model
                  ? {
                      version: x.analysis.model.version,
                      config: x.analysis.model.config,
                      runtime: x.analysis.model.runtime,
                      depthMapSha: x.analysis.model.depth.sha,
                      faceCount: x.analysis.model.faces.length,
                      personCount: x.analysis.model.persons.length,
                    }
                  : undefined,
              }
            : null,
        })),
        test: s.settings.testId ? { id: s.settings.testId, prompt: s.settings.testPrompt } : null,
        encoder: 'sharp-jpeg-92-v1',
      };
      const id = hash(stableJSON(manifest));
      if (!s.snapshots.some(x => x.id === id))
        s.snapshots.push({
          id,
          state: 'building',
          manifest,
          source: structuredClone(result.selected) as Image[],
          files: [],
        });
    });
  }
  async exportFile(rev: number, id: string, index: number) {
    return this.mutate(rev, async s => {
      const v = s.snapshots.find(x => x.id === id);
      ensure(v && v.state === 'building', 'Export is not building');
      integer(index, 0, v.source.length - 1);
      const img = v.source[index],
        out = path.join(this.folder, 'versions', id);
      await contained(this.folder, out, true);
      await fs.mkdir(path.join(out, 'training'), { recursive: true });
      const size = bucket(img.width, img.height, v.manifest.settings.tier),
        imageBytes = await sharp(await this.source(img))
          .rotate()
          .resize(size.width, size.height, { fit: v.manifest.settings.crop === 'center' ? 'cover' : 'fill' })
          .jpeg({ quality: 92 })
          .toBuffer();
      const stem = 'training/' + String(index + 1).padStart(6, '0'),
        caption = Buffer.from(img.caption);
      const files: ExportFile[] = [
        { path: stem + '.jpg', sha: hash(imageBytes), size: imageBytes.length, ...size },
        { path: stem + '.txt', sha: hash(caption), size: caption.length },
      ];
      for (const [i, f] of files.entries()) {
        const file = await contained(this.folder, path.join(out, f.path), true);
        await atomic(file, i === 0 ? imageBytes : caption);
      }
      v.files = v.files
        .filter(f => !f.path.startsWith(stem + '.'))
        .concat(files)
        .sort((a, b) => a.path.localeCompare(b.path));
    });
  }
  async finishExport(rev: number, id: string) {
    return this.mutate(rev, async s => {
      const v = s.snapshots.find(x => x.id === id);
      ensure(v && v.state === 'building' && v.files.length === v.source.length * 2, 'Finish all export pairs first');
      const out = path.join(this.folder, 'versions', id);
      if (v.manifest.test) {
        const img = s.images.find(x => x.id === v.manifest.test.id);
        ensure(img, 'Test source missing');
        const bytes = await this.source(img),
          dir = path.join(out, 'test');
        await contained(this.folder, dir, true);
        await fs.mkdir(dir, { recursive: true });
        const ext = path.extname(img.filename).toLowerCase();
        await atomic(path.join(dir, 'reference' + ext), bytes);
        await atomic(path.join(dir, 'prompt.txt'), v.manifest.test.prompt);
        v.files.push(
          { path: 'test/reference' + ext, sha: hash(bytes), size: bytes.length },
          {
            path: 'test/prompt.txt',
            sha: hash(Buffer.from(v.manifest.test.prompt)),
            size: Buffer.byteLength(v.manifest.test.prompt),
          },
        );
      }
      for (const f of v.files) {
        const b = await fs.readFile(await this.file(id, f.path));
        ensure(hash(b) === f.sha && b.length === f.size, 'Export readback differs');
      }
      const manifest = { ...v.manifest, files: v.files },
        raw = stableJSON(manifest),
        digest = hash(raw);
      await atomic(path.join(out, 'manifest.json'), raw);
      v.manifest = manifest;
      v.digest = digest;
      v.state = 'complete';
    });
  }
  async verifySnapshot(v: Snapshot) {
    ensure(v.state === 'complete' && v.digest, 'Completed immutable snapshot required');
    const raw = await fs.readFile(await this.file(v.id, 'manifest.json'));
    ensure(
      hash(raw) === v.digest && stableJSON(JSON.parse(raw.toString())) === stableJSON(v.manifest),
      'Snapshot manifest changed',
      409,
    );
    for (const f of v.files) {
      const bytes = await fs.readFile(await this.file(v.id, f.path));
      ensure(bytes.length === f.size && hash(bytes) === f.sha, 'Snapshot file changed', 409);
    }
  }
  async file(version: string, relative: string) {
    ensure(/^[0-9a-f]{64}$/.test(version), 'Invalid version');
    ensure(
      /^(training\/\d{6}\.(jpg|txt)|test\/(reference\.(png|jpg|jpeg|webp)|prompt.txt)|manifest.json)$/.test(relative),
      'Invalid snapshot file',
    );
    return contained(this.folder, path.join(this.folder, 'versions', version, relative));
  }
}
