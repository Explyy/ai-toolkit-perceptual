import { NextResponse } from 'next/server';
import fs from 'node:fs/promises';
import { getDatasetsRoot } from '@/server/settings';
import { contained, atomic, captionPath } from '@/datasetStudio/store';
import { body } from '@/datasetStudio/http';
import { text, ensure, Problem } from '@/datasetStudio/domain';
export async function POST(request: Request) {
  try {
    const x = await body(request);
    const root = await fs.realpath(await getDatasetsRoot());
    ensure(typeof x.imgPath === 'string', 'Image path required');
    const image = await contained(root, x.imgPath);
    ensure((await fs.stat(image)).isFile(), 'Image missing', 404);
    const ext = x.ext ?? 'txt';
    const caption = text(x.caption, 64000),
      file = await contained(root, captionPath(image, ext), true);
    await atomic(file, caption);
    return NextResponse.json({ success: true });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Problem ? e.message : 'Caption save failed; draft remains unsaved' },
      { status: e instanceof Problem ? e.status : 500 },
    );
  }
}
