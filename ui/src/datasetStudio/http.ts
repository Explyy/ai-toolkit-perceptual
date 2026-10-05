import { ensure } from './domain';
export function sameOrigin(request: Request) {
  const origin = request.headers.get('origin'),
    url = new URL(request.url);
  const host = request.headers.get('host') ?? url.host;
  const scheme = request.headers.get('x-forwarded-proto') ?? url.protocol.slice(0, -1);
  // Next can normalize request.url to localhost. Host identifies the actual HTTP
  // target; never trust x-forwarded-host or an arbitrary origin fallback.
  ensure(
    !!origin && ['http', 'https'].includes(scheme) && !/[\s,/@\\]/.test(host),
    'Same-origin mutation required',
    403,
  );
  let expected: string;
  try {
    expected = new URL(scheme + '://' + host).origin;
  } catch {
    ensure(false, 'Same-origin mutation required', 403);
    return;
  }
  ensure(
    origin === expected && request.headers.get('sec-fetch-site') !== 'cross-site',
    'Same-origin mutation required',
    403,
  );
}
export async function body(request: Request) {
  sameOrigin(request);
  const reader = request.body?.getReader();
  ensure(reader, 'Request body missing');
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    for (;;) {
      const r = await reader.read();
      if (r.done) break;
      size += r.value.length;
      ensure(size <= 1500000, 'Request too large', 413);
      chunks.push(r.value);
    }
  } catch (e) {
    await reader.cancel();
    throw e;
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return JSON.parse(new TextDecoder().decode(bytes));
}
