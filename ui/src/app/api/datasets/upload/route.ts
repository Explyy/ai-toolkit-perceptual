import { NextResponse } from 'next/server';
import fs from 'node:fs/promises';
import path from 'node:path';
import { getDatasetsRoot } from '@/server/settings';
import { contained, datasetName } from '@/datasetStudio/store';
import { sameOrigin } from '@/datasetStudio/http';
import { ensure, Problem } from '@/datasetStudio/domain';
export async function POST(request: Request) {
  try {
    sameOrigin(request);
    const reader = request.body?.getReader();
    ensure(reader, 'Upload body missing');
    let total = 0;
    const parts: Uint8Array[] = [];
    try {
      for (;;) {
        const x = await reader.read();
        if (x.done) break;
        total += x.value.length;
        ensure(total <= 100 * 1024 * 1024, 'Upload batch exceeds100MB; upload smaller batches', 413);
        parts.push(x.value);
      }
    } catch (e) {
      await reader.cancel();
      throw e;
    }
    const bytes = new Uint8Array(total);
    let off = 0;
    for (const p of parts) {
      bytes.set(p, off);
      off += p.length;
    }
    const form = await new Request(request.url, {
      method: 'POST',
      headers: { 'Content-Type': request.headers.get('content-type') ?? '' },
      body: bytes,
    }).formData();
    const files = form.getAll('files');
    ensure(files.length > 0 && files.length <= 100, 'Upload between1 and100 files');
    const root = await fs.realpath(await getDatasetsRoot()),
      folder = await contained(root, path.join(root, datasetName(form.get('datasetName'))), true);
    await fs.mkdir(folder, { recursive: true });
    const saved: string[] = [],
      names = new Set<string>();
    for (const item of files) {
      ensure(item instanceof File, 'Invalid file');
      const name = item.name.replace(/[^a-zA-Z0-9.-]/g, '_');
      ensure(
        !name.startsWith('.') &&
          /\.(png|jpe?g|gif|bmp|webp|txt|json|caption|mp4|avi|mov|mkv|wmv|m4v|flv|mp3|wav|flac|ogg)$/i.test(name),
        'Unsupported upload filename',
      );
      ensure(!names.has(name), 'Duplicate upload filename', 409);
      names.add(name);
      const file = await contained(root, path.join(folder, name), true);
      try {
        await fs.access(file);
        throw new Problem(409, 'Original already exists: ' + name + '. Choose a new filename.');
      } catch (e: any) {
        if (e.code !== 'ENOENT') throw e;
      }
      ensure(item.size <= 24 * 1024 * 1024, 'File exceeds24MB limit', 413);
    }
    for (const item of files as File[]) {
      const name = item.name.replace(/[^a-zA-Z0-9.-]/g, '_');
      await fs.writeFile(await contained(root, path.join(folder, name), true), Buffer.from(await item.arrayBuffer()), {
        flag: 'wx',
        mode: 0o600,
      });
      saved.push(name);
    }
    return NextResponse.json({ message: 'Originals uploaded; existing files preserved', files: saved });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Problem ? e.message : 'Upload failed; originals preserved' },
      { status: e instanceof Problem ? e.status : 500 },
    );
  }
}
