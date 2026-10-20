/**
 * cv2.FaceDetectorYN re-implemented on onnxruntime-web: pad to a multiple of 32, run YuNet, decode the
 * three stride heads and apply cv2.dnn.NMSBoxes. Matches OpenCV's boxes and scores on the test photos.
 */
import type { InferenceSession, Tensor as OrtTensor } from "onnxruntime-web/wasm-core";
import type { Image } from "./imageops";

export type Box = [x: number, y: number, w: number, h: number];

const STRIDES = [8, 16, 32] as const;
const NMS_THRESHOLD = 0.3;
const TOP_K = 5000;

export async function detectFaces(
  session: InferenceSession,
  Tensor: typeof OrtTensor,
  img: Image,
  scoreThreshold: number,
): Promise<Box[]> {
  const { width: w, height: h } = img;
  const pw = (Math.floor((w - 1) / 32) + 1) * 32;
  const ph = (Math.floor((h - 1) / 32) + 1) * 32;

  // NCHW float32, BGR, 0-255, zero padding on the right/bottom
  const input = new Float32Array(3 * ph * pw);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const s = (y * w + x) * 3;
      for (let c = 0; c < 3; c++) input[c * ph * pw + y * pw + x] = img.data[s + c];
    }
  const out = await session.run({ input: new Tensor("float32", input, [1, 3, ph, pw]) });

  const boxes: Box[] = [];
  const scores: number[] = [];
  for (const s of STRIDES) {
    const cols = pw / s, rows = ph / s;
    const cls = out[`cls_${s}`].data as Float32Array;
    const obj = out[`obj_${s}`].data as Float32Array;
    const bbox = out[`bbox_${s}`].data as Float32Array;
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) {
        const i = r * cols + c;
        const score = Math.sqrt(Math.min(Math.max(cls[i], 0), 1) * Math.min(Math.max(obj[i], 0), 1));
        if (score <= scoreThreshold) continue;
        const cx = (c + bbox[i * 4]) * s;
        const cy = (r + bbox[i * 4 + 1]) * s;
        const bw = Math.exp(bbox[i * 4 + 2]) * s;
        const bh = Math.exp(bbox[i * 4 + 3]) * s;
        boxes.push([cx - bw / 2, cy - bh / 2, bw, bh]);
        scores.push(score);
      }
  }
  return nms(boxes, scores).map((i) => boxes[i]);
}

/** cv2.dnn.NMSBoxes on integer rects (FaceDetectorYN converts boxes to Rect2i before NMS). */
function nms(boxes: Box[], scores: number[]): number[] {
  const order = scores.map((s, i) => [s, i] as const).sort((a, b) => b[0] - a[0]).slice(0, TOP_K).map(([, i]) => i);
  const rects = boxes.map(([x, y, w, h]) => [Math.trunc(x), Math.trunc(y), Math.trunc(w), Math.trunc(h)]);
  const keep: number[] = [];
  for (const i of order) {
    const [ax, ay, aw, ah] = rects[i];
    const suppressed = keep.some((k) => {
      const [bx, by, bw, bh] = rects[k];
      const iw = Math.min(ax + aw, bx + bw) - Math.max(ax, bx);
      const ih = Math.min(ay + ah, by + bh) - Math.max(ay, by);
      const inter = iw > 0 && ih > 0 ? iw * ih : 0;
      return inter / (aw * ah + bw * bh - inter) > NMS_THRESHOLD;
    });
    if (!suppressed) keep.push(i);
  }
  return keep;
}
