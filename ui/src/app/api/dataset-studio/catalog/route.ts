import { NextResponse } from 'next/server';
import { getDataRoot, getDatasetsRoot, getHFToken } from '@/server/settings';
import { Hub } from '@/datasetStudio/hf';
import { nativeCatalog, folders, materialize, importStatus } from '@/datasetStudio/catalog';
import { body } from '@/datasetStudio/http';
import { Problem, ensure } from '@/datasetStudio/domain';
export const runtime = 'nodejs';
const error = (e: any) =>
  NextResponse.json(
    { error: e instanceof Problem ? e.message : 'Catalog operation failed; local originals preserved' },
    { status: e instanceof Problem ? e.status : 500 },
  );
export async function GET(request: Request) {
  try {
    const q = new URL(request.url).searchParams,
      data = await getDataRoot(),
      datasets = await getDatasetsRoot(),
      token = await getHFToken();
    if (q.has('import')) return NextResponse.json(await importStatus(data, q.get('import')!));
    if (q.has('repo')) {
      ensure(token, 'Hugging Face non configurato', 503);
      return NextResponse.json({
        folders: await folders(new Hub(token), q.get('repo')!, q.get('revision') ?? undefined),
      });
    }
    const local = await nativeCatalog(datasets, data);
    if (!token) return NextResponse.json({ local, repositories: [], configured: false });
    try {
      return NextResponse.json({ local, repositories: await new Hub(token).repositories(), configured: true });
    } catch (e: any) {
      return NextResponse.json({
        local,
        repositories: [],
        configured: true,
        remoteError: e instanceof Problem ? e.message : 'Catalogo HF temporaneamente non disponibile',
      });
    }
  } catch (e) {
    return error(e);
  }
}
export async function POST(request: Request) {
  try {
    const x = await body(request),
      token = await getHFToken();
    ensure(token, 'Hugging Face non configurato', 503);
    return NextResponse.json(await materialize(new Hub(token), await getDataRoot(), await getDatasetsRoot(), x.entry));
  } catch (e) {
    return error(e);
  }
}
