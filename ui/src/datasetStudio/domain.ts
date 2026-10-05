import { modelDuplicate, diversity, type ModelSignals } from './analysisPolicy';
// Ported from approved Dataset Studio v1; native file storage is separate.
export const CATEGORIES = ['face', 'body', 'variety'] as const;
export type Category = (typeof CATEGORIES)[number] | 'unclassified';
export const ANALYZER = 'pixels-dhash-laplacian-v1';
export type Analysis = {
  version: string;
  dhash: string;
  sharpness: number;
  exposure: number;
  clipped: number;
  resolution: number;
  quality: number;
  model?: ModelSignals;
};
export type ImageRecord = {
  id: string;
  project: string;
  sha: string;
  filename: string;
  mime: string;
  size: number;
  width: number;
  height: number;
  caption: string;
  caption_source: string;
  category: Category;
  pinned: number;
  excluded: number;
  discarded: number;
  analysis: Analysis | null;
  pose: unknown;
  revision: number;
  created: number;
  categorySource?: 'manual' | 'automatic';
  reviewRevision?: number;
  analysisStatus?: { phase: 'pending' | 'complete' | 'unavailable' | 'failed'; reason?: string; config: string };
};
export type Settings = {
  count: number;
  tier: 512 | 768 | 1024;
  crop: 'fit' | 'center';
  testId: string | null;
  testPrompt: string;
  model: string;
  instructions: string;
  repo: string;
};
export const DEFAULT_SETTINGS: Settings = {
  count: 30,
  tier: 1024,
  crop: 'fit',
  testId: null,
  testPrompt: '',
  model: '',
  instructions:
    'Describe only visible features, clothing, framing, pose and background. Return one concise training caption.',
  repo: 'daverave/Personal',
};
export class Problem extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}
export function ensure(condition: unknown, message: string, status = 400): asserts condition {
  if (!condition) throw new Problem(status, message);
}
export function text(value: unknown, max = 1000): string {
  ensure(typeof value === 'string' && value.length <= max && !/[\u0000]/.test(value), 'Invalid text');
  return value;
}
export function id(value: unknown): string {
  ensure(typeof value === 'string' && /^[a-zA-Z0-9_-]{1,80}$/.test(value), 'Invalid identifier');
  return value;
}
export function repoId(value: unknown): string {
  const s = text(value, 160);
  ensure(/^[\w.-]+\/[\w.-]+$/.test(s) && !s.includes('..'), 'Use an owner/repository name');
  return s;
}
export function path(value: unknown): string {
  const s = text(value, 400);
  ensure(
    s.length > 0 &&
      !s.startsWith('/') &&
      !s.includes('\\') &&
      s.split('/').every(p => /^[\w.-]+$/.test(p) && p !== '.' && p !== '..'),
    'Invalid relative path',
  );
  return s;
}
export function revision(value: unknown): string {
  ensure(typeof value === 'string' && /^[0-9a-f]{40}$/.test(value), 'An immutable 40-character revision is required');
  return value;
}
export function filename(value: unknown): string {
  const s = text(value, 180);
  ensure(/^[^/\\\x00-\x1f]+\.(jpg|jpeg|png|webp)$/i.test(s) && !s.startsWith('.'), 'Use a PNG, JPEG or WebP filename');
  return s;
}
export function integer(v: unknown, min: number, max: number): number {
  ensure(Number.isInteger(v) && Number(v) >= min && Number(v) <= max, `Expected an integer from ${min} to ${max}`);
  return Number(v);
}
export function settings(v: unknown): Settings {
  ensure(!!v && typeof v === 'object', 'Settings required');
  const x = v as Settings;
  ensure([512, 768, 1024].includes(x.tier), 'Invalid resolution');
  ensure(x.crop === 'fit' || x.crop === 'center', 'Invalid crop');
  return {
    count: integer(x.count, 1, 1000),
    tier: x.tier,
    crop: x.crop,
    testId: x.testId === null ? null : id(x.testId),
    testPrompt: text(x.testPrompt, 4000),
    model: text(x.model, 180),
    instructions: text(x.instructions, 8000),
    repo: repoId(x.repo),
  };
}
export function stableJSON(v: unknown): string {
  if (Array.isArray(v)) return '[' + v.map(stableJSON).join(',') + ']';
  if (v && typeof v === 'object')
    return (
      '{' +
      Object.keys(v)
        .filter(k => (v as Record<string, unknown>)[k] !== undefined)
        .sort()
        .map(k => JSON.stringify(k) + ':' + stableJSON((v as Record<string, unknown>)[k]))
        .join(',') +
      '}'
    );
  return JSON.stringify(v) ?? 'null';
}
export async function sha256(bytes: Uint8Array<ArrayBufferLike> | ArrayBuffer | string): Promise<string> {
  const input =
    typeof bytes === 'string'
      ? new TextEncoder().encode(bytes)
      : bytes instanceof Uint8Array
        ? new Uint8Array(bytes)
        : bytes;
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256', input))]
    .map(x => x.toString(16).padStart(2, '0'))
    .join('');
}
export function bucket(width: number, height: number, tier = 1024) {
  integer(width, 8, 30000);
  integer(height, 8, 30000);
  ensure([512, 768, 1024].includes(tier), 'Invalid tier');
  const target = Math.min(width * height, tier * tier),
    scale = Math.sqrt(target / (width * height)),
    w = (width * scale) / 8,
    h = (height * scale) / 8;
  const candidates = [
    [Math.floor(w) * 8, Math.floor(h) * 8],
    [Math.floor(w) * 8, Math.ceil(h) * 8],
    [Math.ceil(w) * 8, Math.floor(h) * 8],
    [Math.ceil(w) * 8, Math.ceil(h) * 8],
  ].filter(([a, b]) => a > 0 && b > 0 && a * b <= tier * tier);
  ensure(candidates.length, 'Image is too narrow for an 8-pixel bucket');
  candidates.sort((a, b) => Math.abs(a[0] * a[1] - target) - Math.abs(b[0] * b[1] - target));
  return { width: candidates[0][0], height: candidates[0][1] };
}
export function hamming(a: string, b: string) {
  let n = BigInt('0x' + a) ^ BigInt('0x' + b),
    count = 0;
  while (n) {
    count++;
    n &= n - BigInt(1);
  }
  return count;
}
export function groups(images: ImageRecord[], threshold = 6): Map<string, string> {
  const sorted = [...images].sort((a, b) => a.sha.localeCompare(b.sha));
  const parent = new Map(sorted.map(x => [x.id, x.id]));
  const find = (x: string): string => {
    const p = parent.get(x)!;
    if (p === x) return x;
    const root = find(p);
    parent.set(x, root);
    return root;
  };
  for (let i = 0; i < sorted.length; i++)
    for (let j = i + 1; j < sorted.length; j++) {
      const a = sorted[i],
        b = sorted[j];
      if (
        a.sha === b.sha ||
        (a.analysis?.version === ANALYZER &&
          b.analysis?.version === ANALYZER &&
          a.analysis.sharpness > 1 &&
          b.analysis.sharpness > 1 &&
          Math.abs(a.width / a.height - b.width / b.height) < 0.15 &&
          hamming(a.analysis.dhash, b.analysis.dhash) <= threshold &&
          (!(a.analysis.model || b.analysis.model) || modelDuplicate(a, b)))
      )
        parent.set(find(b.id), find(a.id));
    }
  return new Map(sorted.map(x => [x.id, find(x.id)]));
}
export function select(images: ImageRecord[], count: number, testId: string | null = null) {
  integer(count, 1, 1000);
  const quotas = Object.fromEntries(
    CATEGORIES.map((c, i) => [c, Math.floor(count / 3) + (i < count % 3 ? 1 : 0)]),
  ) as Record<(typeof CATEGORIES)[number], number>;
  const eligible = images.filter(x => !x.discarded && !x.excluded && x.id !== testId),
    grouped = groups(eligible),
    chosen: ImageRecord[] = [],
    used = new Set<string>(),
    conflicts: string[] = [];
  const rank = (a: ImageRecord, b: ImageRecord) =>
    (b.analysis?.quality ?? -1) - (a.analysis?.quality ?? -1) || a.sha.localeCompare(b.sha) || a.id.localeCompare(b.id);
  for (const item of eligible.filter(x => x.pinned).sort(rank)) {
    if (item.category === 'unclassified') {
      conflicts.push(`Pinned image ${item.filename} needs a category`);
      continue;
    }
    chosen.push(item);
    used.add(grouped.get(item.id)!);
  }
  for (const c of CATEGORIES) {
    const pinned = chosen.filter(x => x.category === c).length;
    if (pinned > quotas[c]) conflicts.push(`${c}: ${pinned} pins exceed quota ${quotas[c]}`);
    const candidates = eligible.filter(x => x.category === c && !x.pinned);
    while (candidates.length) {
      candidates.sort(
        (a, b) =>
          (b.analysis?.quality ?? -1) +
            30 * diversity(b, chosen) -
            ((a.analysis?.quality ?? -1) + 30 * diversity(a, chosen)) || rank(a, b),
      );
      const item = candidates.shift()!;
      if (chosen.filter(x => x.category === c).length >= quotas[c]) break;
      const g = grouped.get(item.id)!;
      if (!used.has(g)) {
        chosen.push(item);
        used.add(g);
      }
    }
  }
  const deficits = Object.fromEntries(
    CATEGORIES.map(c => [c, Math.max(0, quotas[c] - chosen.filter(x => x.category === c).length)]),
  );
  return {
    selected: chosen.sort((a, b) => a.sha.localeCompare(b.sha)),
    quotas,
    deficits,
    conflicts,
    groups: Object.fromEntries(grouped),
    complete: chosen.length === count && conflicts.length === 0 && Object.values(deficits).every(x => x === 0),
  };
}
// Versioned 64x64 RGBA sampling; measurements describe pixels, never identity or aesthetics.
export function analyzePixels(
  rgba: Uint8ClampedArray,
  width = 64,
  height = 64,
  originalWidth = width,
  originalHeight = height,
): Analysis {
  ensure(rgba.length === width * height * 4 && width >= 9 && height >= 8, 'Invalid pixel buffer');
  const gray = new Float64Array(width * height);
  let sum = 0,
    clipped = 0;
  for (let i = 0; i < gray.length; i++) {
    const p = i * 4,
      g = ((rgba[p] * 0.2126 + rgba[p + 1] * 0.7152 + rgba[p + 2] * 0.0722) * rgba[p + 3]) / 255;
    gray[i] = g;
    sum += g;
    if (g < 8 || g > 247) clipped++;
  }
  let lap = 0,
    lap2 = 0,
    n = 0;
  for (let y = 1; y < height - 1; y++)
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x,
        v = gray[i - 1] + gray[i + 1] + gray[i - width] + gray[i + width] - 4 * gray[i];
      lap += v;
      lap2 += v * v;
      n++;
    }
  const sharpness = Math.max(0, lap2 / n - (lap / n) ** 2),
    exposure = sum / gray.length / 255;
  let hash = BigInt(0);
  for (let y = 0; y < 8; y++)
    for (let x = 0; x < 8; x++) {
      const yy = Math.min(height - 1, Math.floor(((y + 0.5) * height) / 8)),
        xx = Math.floor((x * width) / 9),
        next = Math.floor(((x + 1) * width) / 9);
      hash = (hash << BigInt(1)) | (gray[yy * width + xx] > gray[yy * width + next] ? BigInt(1) : BigInt(0));
    }
  const fraction = clipped / gray.length,
    resolution = Math.min(1, Math.sqrt(originalWidth * originalHeight) / 1024);
  return {
    version: ANALYZER,
    dhash: hash.toString(16).padStart(16, '0'),
    sharpness,
    exposure,
    clipped: fraction,
    resolution,
    quality:
      Math.round(
        1000 *
          (0.45 * Math.min(1, Math.log1p(sharpness) / 10) +
            0.2 * (1 - Math.abs(exposure - 0.5) * 2) +
            0.15 * (1 - fraction) +
            0.2 * resolution),
      ) / 10,
  };
}
export function validateAnalysis(v: unknown): Analysis {
  ensure(v !== null && typeof v === 'object', 'Analysis required');
  const a = v as Analysis;
  ensure(a.version === ANALYZER && /^[0-9a-f]{16}$/.test(a.dhash), 'Invalid analyzer');
  for (const k of ['sharpness', 'exposure', 'clipped', 'resolution', 'quality'] as const)
    ensure(Number.isFinite(a[k]) && a[k] >= 0, 'Invalid measurement');
  ensure(a.exposure <= 1 && a.clipped <= 1 && a.quality <= 100 && a.resolution <= 1, 'Invalid measurement');
  return a;
}
