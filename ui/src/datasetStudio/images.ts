// Ported from approved Dataset Studio v1; native file storage is separate.
import { ensure, integer } from './domain';
export const MAX_IMAGE_BYTES = 24 * 1024 * 1024;
export function dimensions(bytes: Uint8Array) {
  ensure(bytes.length >= 30, 'Invalid image');
  const d = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let width = 0,
    height = 0,
    mime = '',
    orientation = 1;
  if (bytes.slice(0, 8).join(',') === '137,80,78,71,13,10,26,10') {
    ensure(d.getUint32(12) === 0x49484452, 'Invalid PNG');
    width = d.getUint32(16);
    height = d.getUint32(20);
    mime = 'image/png';
  } else if (bytes[0] === 255 && bytes[1] === 216) {
    mime = 'image/jpeg';
    let i = 2;
    while (i + 9 < bytes.length) {
      if (bytes[i] !== 255) {
        i++;
        continue;
      }
      const m = bytes[i + 1];
      if (m === 217 || m === 218) break;
      if (m === 216 || m === 0 || m === 255) {
        i += 2;
        continue;
      }
      const len = d.getUint16(i + 2);
      ensure(len >= 2 && i + len + 2 <= bytes.length, 'Invalid JPEG');
      if (m === 225) {
        orientation = exifOrientation(bytes.subarray(i + 4, i + 2 + len));
      }
      if ([192, 193, 194, 195, 197, 198, 199, 201, 202, 203, 205, 206, 207].includes(m)) {
        height = d.getUint16(i + 5);
        width = d.getUint16(i + 7);
        break;
      }
      i += len + 2;
    }
  } else if (
    String.fromCharCode(...bytes.slice(0, 4)) === 'RIFF' &&
    String.fromCharCode(...bytes.slice(8, 12)) === 'WEBP'
  ) {
    mime = 'image/webp';
    const kind = String.fromCharCode(...bytes.slice(12, 16));
    if (kind === 'VP8X') {
      width = 1 + bytes[24] + (bytes[25] << 8) + (bytes[26] << 16);
      height = 1 + bytes[27] + (bytes[28] << 8) + (bytes[29] << 16);
    } else if (kind === 'VP8 ') {
      ensure(bytes[23] === 157 && bytes[24] === 1 && bytes[25] === 42, 'Invalid WebP');
      width = d.getUint16(26, true) & 16383;
      height = d.getUint16(28, true) & 16383;
    } else if (kind === 'VP8L') {
      ensure(bytes[20] === 47, 'Invalid WebP');
      width = 1 + ((bytes[21] | (bytes[22] << 8)) & 16383);
      height = 1 + (((bytes[22] >> 6) | (bytes[23] << 2) | (bytes[24] << 10)) & 16383);
    }
  }
  ensure(mime && width && height, 'Unsupported or corrupt image');
  integer(width, 8, 30000);
  integer(height, 8, 30000);
  ensure(width * height <= 60_000_000, 'Image exceeds 60 megapixels');
  if (orientation >= 5 && orientation <= 8) [width, height] = [height, width];
  return { width, height, mime, orientation };
}
export async function boundedBytes(response: Response, max = MAX_IMAGE_BYTES) {
  ensure(response.ok, 'Remote file unavailable', 502);
  const reader = response.body?.getReader();
  ensure(reader, 'Empty file');
  let size = 0;
  const chunks: Uint8Array[] = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.length;
    if (size > max) {
      await reader.cancel();
      ensure(false, 'File is too large', 413);
    }
    chunks.push(value);
  }
  const result = new Uint8Array(size);
  let at = 0;
  for (const x of chunks) {
    result.set(x, at);
    at += x.length;
  }
  return result;
}

export function exifOrientation(segment: Uint8Array): number {
  try {
    if (String.fromCharCode(...segment.slice(0, 6)) !== 'Exif\0\0') return 1;
    const t = segment.subarray(6),
      d = new DataView(t.buffer, t.byteOffset, t.length),
      little = d.getUint16(0) === 0x4949;
    if (!little && d.getUint16(0) !== 0x4d4d) return 1;
    if (d.getUint16(2, little) !== 42) return 1;
    const at = d.getUint32(4, little),
      count = d.getUint16(at, little);
    for (let i = 0; i < Math.min(count, 512); i++) {
      const p = at + 2 + i * 12;
      if (d.getUint16(p, little) === 274) {
        const v = d.getUint16(p + 8, little);
        return v >= 1 && v <= 8 ? v : 1;
      }
    }
  } catch {}
  return 1;
}
