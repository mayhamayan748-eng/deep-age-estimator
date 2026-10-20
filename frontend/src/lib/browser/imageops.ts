/**
 * OpenCV-equivalent image operations for in-browser inference.
 *
 * Images are interleaved Uint8Array buffers in BGR order (like cv2.imread), so this code mirrors
 * backend/app/services/preprocessing.py line by line. resizeLinear reproduces cv2.resize(INTER_LINEAR)
 * fixed-point arithmetic; checked against OpenCV on 300 random crops and 200 UTKFace faces, the
 * predicted ages differ by at most 0.03 years.
 */

export interface Image {
  data: Uint8Array;
  width: number;
  height: number;
  channels: number;
}

const COEF_SCALE = 2048; // INTER_RESIZE_COEF_SCALE

/** cvRound: round half to even. */
function cvRound(v: number): number {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Python's round(): half to even. */
export function roundHalfEven(v: number): number {
  return cvRound(v);
}

function coefficients(ssize: number, dsize: number) {
  // cv::resize: inv_scale = (double)dsize/ssize, scale = 1/inv_scale; fx is computed as a float
  const scale = 1 / (dsize / ssize);
  const ofs = new Int32Array(dsize);
  const a0 = new Int32Array(dsize);
  const a1 = new Int32Array(dsize);
  for (let d = 0; d < dsize; d++) {
    let f = Math.fround((d + 0.5) * scale - 0.5);
    let s = Math.floor(f);
    f = Math.fround(f - s);
    if (s < 0) {
      f = 0;
      s = 0;
    }
    if (s >= ssize - 1) {
      f = 0;
      s = ssize - 1;
    }
    ofs[d] = s;
    a0[d] = cvRound(Math.fround(Math.fround(1 - f) * COEF_SCALE));
    a1[d] = cvRound(Math.fround(f * COEF_SCALE));
  }
  return { ofs, a0, a1 };
}

/** cv2.resize(src, (dw, dh)) with the default INTER_LINEAR, for uint8 images. */
export function resizeLinear(src: Image, dw: number, dh: number): Image {
  const { data, width: sw, height: sh, channels: cn } = src;
  const out = new Uint8Array(dw * dh * cn);

  // cv::resize turns exact 2x downscales into INTER_AREA
  if (sw === dw * 2 && sh === dh * 2) {
    for (let y = 0; y < dh; y++)
      for (let x = 0; x < dw; x++)
        for (let c = 0; c < cn; c++) {
          const i = ((2 * y) * sw + 2 * x) * cn + c;
          const j = i + sw * cn;
          out[(y * dw + x) * cn + c] = (data[i] + data[i + cn] + data[j] + data[j + cn] + 2) >> 2;
        }
    return { data: out, width: dw, height: dh, channels: cn };
  }

  const xc = coefficients(sw, dw);
  const yc = coefficients(sh, dh);
  const rowCache = new Map<number, Int32Array>();
  const hRow = (y: number): Int32Array => {
    let row = rowCache.get(y);
    if (row) return row;
    row = new Int32Array(dw * cn);
    const base = y * sw * cn;
    for (let dx = 0; dx < dw; dx++) {
      const x = xc.ofs[dx];
      const x1 = Math.min(x + 1, sw - 1);
      for (let c = 0; c < cn; c++)
        row[dx * cn + c] = data[base + x * cn + c] * xc.a0[dx] + data[base + x1 * cn + c] * xc.a1[dx];
    }
    if (rowCache.size > 4) rowCache.delete(rowCache.keys().next().value!);
    rowCache.set(y, row);
    return row;
  };

  for (let dy = 0; dy < dh; dy++) {
    const y = yc.ofs[dy];
    const r0 = hRow(y);
    const r1 = hRow(Math.min(y + 1, sh - 1));
    const b0 = yc.a0[dy];
    const b1 = yc.a1[dy];
    const o = dy * dw * cn;
    for (let i = 0; i < dw * cn; i++) {
      // SIMD path of VResizeLinear: ((S0 >> 4) * b0 >> 16) + ((S1 >> 4) * b1 >> 16), then (v + 2) >> 2
      const v = Math.floor(((r0[i] >> 4) * b0) / 65536) + Math.floor(((r1[i] >> 4) * b1) / 65536);
      out[o + i] = Math.min(255, Math.max(0, (v + 2) >> 2));
    }
  }
  return { data: out, width: dw, height: dh, channels: cn };
}

/** cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) */
export function bgrToGray(src: Image): Image {
  const n = src.width * src.height;
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const b = src.data[i * 3], g = src.data[i * 3 + 1], r = src.data[i * 3 + 2];
    out[i] = (b * 1868 + g * 9617 + r * 4899 + 8192) >> 14;
  }
  return { data: out, width: src.width, height: src.height, channels: 1 };
}

/** cv2.cvtColor(img, cv2.COLOR_BGR2RGB) */
export function bgrToRgb(src: Image): Image {
  const out = new Uint8Array(src.data.length);
  for (let i = 0; i < src.data.length; i += 3) {
    out[i] = src.data[i + 2];
    out[i + 1] = src.data[i + 1];
    out[i + 2] = src.data[i];
  }
  return { ...src, data: out };
}

/** img[y0:y1, x0:x1] */
export function crop(src: Image, x0: number, y0: number, x1: number, y1: number): Image {
  const w = x1 - x0, h = y1 - y0, cn = src.channels;
  const out = new Uint8Array(w * h * cn);
  for (let y = 0; y < h; y++) {
    const start = ((y0 + y) * src.width + x0) * cn;
    out.set(src.data.subarray(start, start + w * cn), y * w * cn);
  }
  return { data: out, width: w, height: h, channels: cn };
}

// Phone cameras take 48-50 MP photos; mobile browsers cap canvas size (iOS Safari: ~16.7 MP) and memory.
// Larger photos are scaled down to this side length first, which is plenty to find and crop faces.
const MAX_DECODE_SIDE = 2560;

type Canvas2D = { canvas: OffscreenCanvas | HTMLCanvasElement; ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D };

/** OffscreenCanvas where supported (Safari < 16.4 has none), otherwise a detached <canvas>. */
function makeCanvas(width: number, height: number): Canvas2D {
  if (typeof OffscreenCanvas !== "undefined") {
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (ctx) return { canvas, ctx };
  }
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return { canvas, ctx: canvas.getContext("2d", { willReadFrequently: true })! };
}

function canvasToBlob(canvas: OffscreenCanvas | HTMLCanvasElement, type: string): Promise<Blob> {
  if ("convertToBlob" in canvas) return canvas.convertToBlob({ type });
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("toBlob failed"))), type),
  );
}

/** HEIC/HEIF (default on many iPhones and Samsung phones): sniffed from the file header, since the MIME type is often empty. */
async function isHeic(blob: Blob): Promise<boolean> {
  if (/hei[cf]/i.test(blob.type) || (blob instanceof File && /\.hei[cf]$/i.test(blob.name))) return true;
  const head = new Uint8Array(await blob.slice(4, 12).arrayBuffer());
  const brand = String.fromCharCode(...head);
  return brand.startsWith("ftyp") && /^(heic|heix|hevc|hevx|heim|heis|mif1|msf1)$/.test(brand.slice(4));
}

type Source = { src: CanvasImageSource; width: number; height: number; release: () => void };

/** Tries the decoders browsers actually support, most faithful first. */
async function loadSource(blob: Blob): Promise<Source | null> {
  const attempts: (() => Promise<ImageBitmap>)[] = [
    // Matches cv2.imdecode: EXIF orientation applied, no colour management
    () => createImageBitmap(blob, { imageOrientation: "from-image", colorSpaceConversion: "none", premultiplyAlpha: "none" }),
    // Older mobile browsers reject some options
    () => createImageBitmap(blob),
  ];
  if (typeof createImageBitmap === "function") {
    for (const attempt of attempts) {
      try {
        const bmp = await attempt();
        return { src: bmp, width: bmp.width, height: bmp.height, release: () => bmp.close() };
      } catch {
        /* try the next decoder */
      }
    }
  }
  // Last resort: the <img> decoder, which every browser has
  const url = URL.createObjectURL(blob);
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    return { src: img, width: img.naturalWidth, height: img.naturalHeight, release: () => URL.revokeObjectURL(url) };
  } catch {
    URL.revokeObjectURL(url);
    return null;
  }
}

/** Converts HEIC to JPEG if needed; null when the format can't be read. */
export async function toDecodableBlob(blob: Blob): Promise<Blob | null> {
  if (!(await isHeic(blob))) return blob;
  try {
    const { default: heic2any } = await import("heic2any"); // ~1 MB, loaded only for HEIC photos
    const out = await heic2any({ blob, toType: "image/jpeg", quality: 0.95 });
    return Array.isArray(out) ? out[0] : out;
  } catch {
    return null;
  }
}

/** Decode a photo into BGR pixels the way cv2.imdecode does: EXIF orientation applied, no colour management. */
export async function decodeImage(blob: Blob): Promise<Image | null> {
  const readable = await toDecodableBlob(blob);
  const source = readable && (await loadSource(readable));
  if (!source) return null;

  const scale = Math.min(1, MAX_DECODE_SIDE / Math.max(source.width, source.height));
  const width = Math.max(1, Math.round(source.width * scale));
  const height = Math.max(1, Math.round(source.height * scale));
  const { ctx } = makeCanvas(width, height);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source.src, 0, 0, width, height);
  source.release();

  const rgba = ctx.getImageData(0, 0, width, height).data;
  const bgr = new Uint8Array(width * height * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    bgr[j] = rgba[i + 2];
    bgr[j + 1] = rgba[i + 1];
    bgr[j + 2] = rgba[i];
  }
  return { data: bgr, width, height, channels: 3 };
}

/** PNG as base64 (no data: prefix), for a gray or RGB image. */
export async function encodePngBase64(img: Image): Promise<string> {
  const { canvas, ctx } = makeCanvas(img.width, img.height);
  const imageData = ctx.createImageData(img.width, img.height);
  for (let i = 0, j = 0; i < img.width * img.height; i++, j += img.channels) {
    const r = img.data[j], g = img.channels === 3 ? img.data[j + 1] : r, b = img.channels === 3 ? img.data[j + 2] : r;
    imageData.data.set([r, g, b, 255], i * 4);
  }
  ctx.putImageData(imageData, 0, 0);
  const blob = await canvasToBlob(canvas, "image/png");
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}
