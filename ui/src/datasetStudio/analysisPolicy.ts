import type { Category, ImageRecord } from './domain';
export const MODEL_ANALYZER = 'arcface-depth-pose-v1';
export type Point = [number, number, number];
export type Face = {
  landmarks?: [number, number][];
  box: [number, number, number, number];
  confidence: number;
  embedding: number[];
};
export type Person = { box: [number, number, number, number]; confidence: number; keypoints: Point[] };
export type ModelSignals = {
  version: typeof MODEL_ANALYZER;
  sha: string;
  config: string;
  faces: Face[];
  persons: Person[];
  depth: { grid: number[]; map: string; sha: string; relative: true };
  runtime: { device: string; torch: string; transformers: string; onnxruntime: string };
};
const area = (b: number[]) => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
const number = (n: unknown, lo = 0, hi = 1) => typeof n === 'number' && Number.isFinite(n) && n >= lo && n <= hi;
export function validateSignals(v: any, sha: string, config: string): ModelSignals {
  const bad = () => {
    throw new Error('Invalid or stale ArcFace/depth/pose result');
  };
  if (!v || v.version !== MODEL_ANALYZER || v.sha !== sha || v.config !== config) bad();
  if (!Array.isArray(v.faces) || v.faces.length > 32 || !Array.isArray(v.persons) || v.persons.length > 32) bad();
  for (const f of v.faces) {
    if (
      !Array.isArray(f.box) ||
      f.box.length !== 4 ||
      !f.box.every((n: any) => number(n)) ||
      area(f.box) <= 0 ||
      !number(f.confidence) ||
      !Array.isArray(f.embedding) ||
      f.embedding.length !== 512 ||
      !f.embedding.every((n: any) => number(n, -1, 1)) ||
      Math.abs(f.embedding.reduce((s: number, n: number) => s + n * n, 0) - 1) > 0.02
    )
      bad();
  }
  for (const f of v.faces)
    if (
      f.landmarks &&
      (f.landmarks.length !== 5 ||
        !f.landmarks.every((p: any) => Array.isArray(p) && p.length === 2 && p.every((n: any) => number(n))))
    )
      bad();
  for (const p of v.persons) {
    if (
      !Array.isArray(p.box) ||
      p.box.length !== 4 ||
      !p.box.every((n: any) => number(n)) ||
      area(p.box) <= 0 ||
      !number(p.confidence) ||
      !Array.isArray(p.keypoints) ||
      p.keypoints.length !== 17 ||
      !p.keypoints.every((k: any) => Array.isArray(k) && k.length === 3 && k.every((n: any) => number(n)))
    )
      bad();
  }
  if (
    !v.depth ||
    v.depth.relative !== true ||
    !Array.isArray(v.depth.grid) ||
    v.depth.grid.length !== 64 ||
    !v.depth.grid.every((n: any) => number(n)) ||
    !/^[0-9a-f]{64}$/.test(v.depth.sha) ||
    v.depth.map !== 'depth.png' ||
    !v.runtime ||
    Object.values(v.runtime).some(n => typeof n !== 'string')
  )
    bad();
  return v;
}
export function inferredCategory(m: ModelSignals): { category: Category; reason: string } {
  const f = [...m.faces].sort((a, b) => area(b.box) - area(a.box))[0];
  const p = [...m.persons].sort((a, b) => area(b.box) - area(a.box))[0];
  if (f && f.confidence >= 0.7 && area(f.box) >= 0.08 && f.box[3] - f.box[1] >= 0.28)
    return { category: 'face', reason: 'Volto dominante nell’inquadratura' };
  if (
    p &&
    p.confidence >= 0.5 &&
    area(p.box) >= 0.15 &&
    [11, 12, 13, 14].filter(i => p.keypoints[i][2] >= 0.4).length >= 3
  )
    return { category: 'body', reason: 'Corpo con bacino e gambe visibili' };
  if ((f && f.confidence >= 0.7) || (p && p.confidence >= 0.5))
    return { category: 'variety', reason: 'Inquadratura contestuale o posa parziale' };
  return { category: 'unclassified', reason: 'Nessun soggetto umano rilevato con sufficiente confidenza' };
}
const meanDistance = (a: number[], b: number[]) => a.reduce((s, n, i) => s + Math.abs(n - b[i]), 0) / a.length;
export function signalDistance(a: ModelSignals, b: ModelSignals) {
  const pa = a.persons[0],
    pb = b.persons[0];
  let pose = 1;
  if (pa && pb) {
    const visible = pa.keypoints.map((k, i) => (k[2] >= 0.4 && pb.keypoints[i][2] >= 0.4 ? i : -1)).filter(i => i >= 0);
    if (visible.length >= 4)
      pose =
        visible.reduce(
          (s, i) => s + Math.hypot(pa.keypoints[i][0] - pb.keypoints[i][0], pa.keypoints[i][1] - pb.keypoints[i][1]),
          0,
        ) / visible.length;
  }
  const fa = a.faces[0],
    fb = b.faces[0];
  if (pose === 1 && fa?.landmarks && fb?.landmarks)
    pose =
      fa.landmarks.reduce((s, p, i) => s + Math.hypot(p[0] - fb.landmarks![i][0], p[1] - fb.landmarks![i][1]), 0) / 5;
  const identity = fa && fb ? Math.max(0, 1 - fa.embedding.reduce((s, n, i) => s + n * fb.embedding[i], 0)) : 1;
  const framing = pa && pb ? meanDistance(pa.box, pb.box) : fa && fb ? meanDistance(fa.box, fb.box) : 1;
  return { pose, identity, framing, depth: meanDistance(a.depth.grid, b.depth.grid) };
}
export function modelDuplicate(a: ImageRecord, b: ImageRecord) {
  const ma = a.analysis?.model,
    mb = b.analysis?.model;
  if (!ma || !mb || ma.config !== mb.config) return false;
  const d = signalDistance(ma, mb);
  return d.identity < 0.015 && d.pose < 0.035 && d.framing < 0.04 && d.depth < 0.03;
}
export function diversity(a: ImageRecord, chosen: ImageRecord[]) {
  const m = a.analysis?.model;
  if (!m || !chosen.length) return 0;
  const distances = chosen
    .filter(x => x.analysis?.model?.config === m.config)
    .map(x => {
      const d = signalDistance(m, x.analysis!.model!);
      // Identity alone neither penalizes nor removes an image.
      return Math.min(1, 2 * d.pose + 2 * d.framing + d.depth);
    });
  return distances.length ? Math.min(...distances) : 1;
}
