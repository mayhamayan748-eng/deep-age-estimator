import { AnimatePresence, motion } from "motion/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { AgeCounter } from "../components/AgeCounter";
import { CornerMarks } from "../components/CornerMarks";
import { Dropzone } from "../components/Dropzone";
import { SectionHeading } from "../components/SectionHeading";
import { Webcam } from "../components/Webcam";
import {
  api, ApiError, IN_BROWSER, MAX_UPLOAD_MB, MODEL_LABELS, onModelDownload, pngSrc, type DownloadProgress,
  type Health, type ModelInfo, type ModelName, type PredictResponse,
} from "../lib/api";

type State =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "done"; result: PredictResponse }
  | { kind: "error"; message: string; noFace?: boolean };

const FALLBACK_MAE = 6;

export function Predict({ health, info, offline }: { health: Health | null; info: ModelInfo | null; offline: boolean }) {
  const [image, setImage] = useState<{ blob: Blob; url: string } | null>(null);
  const [webcam, setWebcam] = useState(false);
  const [model, setModel] = useState<ModelName | null>(null);
  const [state, setState] = useState<State>({ kind: "idle" });
  const requestId = useRef(0);

  const activeModel = model ?? health?.default_model ?? null;
  const loadedModels = (Object.keys(MODEL_LABELS) as ModelName[]).filter((m) => health?.models[m]);
  const mae = info?.models.find((m) => m.name === activeModel)?.test_mae ?? FALLBACK_MAE;

  const run = useCallback(async (blob: Blob, m: ModelName | null) => {
    const id = ++requestId.current;
    setState({ kind: "loading" });
    try {
      const result = await api.predict(blob, m ?? undefined);
      if (id === requestId.current) setState({ kind: "done", result });
    } catch (e) {
      if (id !== requestId.current) return;
      const err = e instanceof ApiError ? e : new ApiError(0, "Something went wrong");
      setState({ kind: "error", message: err.message, noFace: err.status === 422 });
    }
  }, []);

  const accept = async (original: Blob) => {
    // Some phones hand over photos with an empty MIME type, so only reject types that are clearly not images
    if (original.type && !original.type.startsWith("image/")) {
      setState({ kind: "error", message: "That file isn't an image. Try a JPG, PNG or WebP." });
      return;
    }
    if (original.size > MAX_UPLOAD_MB * 1024 * 1024) {
      setState({ kind: "error", message: `That image is larger than ${MAX_UPLOAD_MB} MB.` });
      return;
    }
    setWebcam(false);
    setState({ kind: "loading" });
    // HEIC photos (iPhone, Samsung) are converted to JPEG first so they can be previewed and decoded
    const { toDecodableBlob } = await import("../lib/browser/imageops");
    const blob = await toDecodableBlob(original);
    if (!blob) {
      setState({ kind: "error", message: "This photo's format can't be read in this browser. Try a JPG or PNG, or use the camera." });
      return;
    }
    setImage({ blob, url: URL.createObjectURL(blob) });
    void run(blob, model);
  };

  const reset = () => {
    requestId.current++;
    setImage(null);
    setState({ kind: "idle" });
  };

  // Re-run the current photo when the user switches model
  const switchModel = (m: ModelName) => {
    setModel(m);
    if (image) void run(image.blob, m);
  };

  // Free the previous object URL whenever the photo changes or the section unmounts
  useEffect(() => () => {
    if (image) URL.revokeObjectURL(image.url);
  }, [image]);

  const unavailable = offline || (health !== null && health.status !== "ok");

  return (
    <section id="predict" className="border-b border-line">
      <div className="mx-auto max-w-6xl px-4 py-20 sm:px-6 lg:py-28">
        <SectionHeading index="01" kicker="Predict" title={<>Show it a <em className="text-accent">face</em>.</>}>
          Every face in the photo is detected, cropped and shrunk to the 128×128 image the network was trained on —
          you'll see exactly what it saw.
        </SectionHeading>

        {loadedModels.length > 1 && (
          <div role="radiogroup" aria-label="Model" className="mb-6 flex flex-wrap items-center gap-2">
            <span className="label me-2">Model</span>
            {loadedModels.map((m) => (
              <button
                key={m}
                role="radio"
                aria-checked={activeModel === m}
                onClick={() => switchModel(m)}
                className={`rounded-full border px-4 py-1.5 font-mono text-xs transition-colors ${
                  activeModel === m ? "border-accent bg-accent-soft text-text" : "border-line text-muted hover:text-text"
                }`}
              >
                {MODEL_LABELS[m]}
                {m === health?.default_model && <span className="ms-1.5 text-accent">★</span>}
              </button>
            ))}
          </div>
        )}

        {unavailable && (
          <p role="status" className="mb-6 border-s-2 border-amber bg-surface px-4 py-3 text-sm text-muted">
            {offline
              ? IN_BROWSER
                ? "The model files couldn't be downloaded. Check your connection and reload the page."
                : "The prediction server isn't reachable right now. Start the backend and reload the page."
              : "The server is up but no trained model or face detector is loaded. Put the model files in ./models."}
          </p>
        )}

        <div className="grid gap-8 lg:grid-cols-[1.4fr_1fr]">
          <div>
            {webcam ? (
              <Webcam onCapture={accept} onCancel={() => setWebcam(false)} />
            ) : image ? (
              <PhotoWithBoxes
                url={image.url}
                state={state}
                onFile={accept}
                onWebcam={() => setWebcam(true)}
                onReset={reset}
              />
            ) : (
              <Dropzone onFile={accept} onWebcam={() => setWebcam(true)} />
            )}
          </div>

          <div aria-live="polite">
            <Results state={state} mae={mae} hasImage={!!image} onRetry={() => image && run(image.blob, model)} />
          </div>
        </div>
      </div>
    </section>
  );
}

function PhotoWithBoxes({ url, state, onFile, onWebcam, onReset }: {
  url: string;
  state: State;
  onFile: (f: File) => void;
  onWebcam: () => void;
  onReset: () => void;
}) {
  const result = state.kind === "done" ? state.result : null;
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  return (
    <div className="flex flex-col gap-4">
      {/* The wrapper shrinks to the image, so percentage boxes line up with image pixels.
          Dropping a new photo on it replaces the current one. */}
      <div
        className="relative mx-auto w-fit bg-surface"
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const f = e.dataTransfer.files[0];
          if (f) onFile(f);
        }}
      >
        <img src={url} alt="Uploaded photo" className="block max-h-[70vh] max-w-full" />

        {state.kind === "loading" && (
          <motion.div
            aria-hidden
            className="absolute inset-x-0 h-0.5 bg-accent shadow-[0_0_16px_var(--accent)]"
            animate={{ top: ["0%", "100%", "0%"] }}
            transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
          />
        )}

        {/* Boxes use physical left/top: they map to image pixels, which don't flip in RTL */}
        {result?.faces.map((f, i) => (
          <motion.div
            key={`${f.bbox.x}-${f.bbox.y}-${i}`}
            initial={{ opacity: 0, scale: 1.25 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: i * 0.15, ease: [0.22, 1, 0.36, 1] }}
            className="absolute"
            style={{
              left: `${(f.bbox.x / result.image_width) * 100}%`,
              top: `${(f.bbox.y / result.image_height) * 100}%`,
              width: `${(f.bbox.w / result.image_width) * 100}%`,
              height: `${(f.bbox.h / result.image_height) * 100}%`,
            }}
          >
            <CornerMarks size="size-[18%]" />
            <span className="absolute -top-7 left-0 whitespace-nowrap bg-accent px-2 py-0.5 font-mono text-xs font-medium text-bg">
              #{i + 1} · <AgeCounter value={f.age} delay={i * 0.15} />
            </span>
          </motion.div>
        ))}

        {dragging && (
          <div className="absolute inset-0 grid place-items-center border-2 border-dashed border-accent bg-bg/70">
            <p className="font-display text-2xl">Drop to replace</p>
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-center gap-3">
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={state.kind === "loading"}
          className="rounded-full bg-accent px-5 py-2.5 text-sm font-medium text-bg transition-transform hover:-translate-y-0.5 disabled:opacity-40"
        >
          Choose another photo
        </button>
        <button
          type="button"
          onClick={onWebcam}
          className="rounded-full border border-line px-5 py-2.5 text-sm font-medium transition-colors hover:border-accent"
        >
          Use webcam
        </button>
        <button type="button" onClick={onReset} className="label px-2 transition-colors hover:text-text">
          Clear
        </button>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/*,.heic,.heif"
        className="sr-only"
        aria-label="Choose another photo"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = "";
        }}
      />
    </div>
  );
}

function Results({ state, mae, hasImage, onRetry }: { state: State; mae: number; hasImage: boolean; onRetry: () => void }) {
  if (state.kind === "idle") {
    return (
      <div className="flex h-full min-h-48 flex-col justify-center border border-line p-6">
        <p className="label mb-2">Waiting for a photo</p>
        <p className="text-muted">Results appear here: the estimated age and the exact crop the model received for each face.</p>
      </div>
    );
  }
  if (state.kind === "loading") {
    return (
      <div className="flex flex-col gap-3" aria-busy="true">
        <p className="label">{IN_BROWSER ? <BrowserLoadingLabel /> : "Detecting faces…"}</p>
        {[0, 1].map((i) => (
          <div key={i} className="flex animate-pulse gap-4 border border-line p-4">
            <div className="size-24 bg-surface-2" />
            <div className="flex-1 space-y-3 py-2">
              <div className="h-3 w-1/3 bg-surface-2" />
              <div className="h-8 w-1/2 bg-surface-2" />
            </div>
          </div>
        ))}
      </div>
    );
  }
  if (state.kind === "error") {
    return (
      <div role="alert" className="border border-accent/40 bg-accent-soft p-6">
        <p className="label mb-2 text-accent">{state.noFace ? "No face found" : "Something went wrong"}</p>
        <p>{state.message}</p>
        {state.noFace && (
          <p className="mt-2 text-sm text-muted">Try a well-lit photo where the face is clearly visible and facing the camera.</p>
        )}
        {hasImage && !state.noFace && (
          <button type="button" onClick={onRetry} className="mt-4 rounded-full border border-line px-4 py-1.5 text-sm">
            Retry
          </button>
        )}
      </div>
    );
  }

  const { result } = state;
  return (
    <div className="flex flex-col gap-4">
      <p className="label">
        {result.faces.length} face{result.faces.length > 1 ? "s" : ""} · {MODEL_LABELS[result.model]}
      </p>
      <AnimatePresence>
        {result.faces.map((f, i) => (
          <motion.article
            key={i}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.15, duration: 0.5 }}
            className="flex gap-5 border border-line bg-surface p-4"
          >
            <figure className="shrink-0">
              <img
                src={pngSrc(f.crop)}
                alt={`Model input for face ${i + 1}`}
                width={112}
                height={112}
                className="size-28 [image-rendering:pixelated]"
              />
              <figcaption className="label mt-1.5 text-[10px]">model input · 128²</figcaption>
            </figure>
            <div className="flex min-w-0 flex-col justify-center">
              <p className="label">Face #{i + 1} · estimated age</p>
              <p className="font-display text-6xl leading-none">
                <AgeCounter value={f.age} delay={i * 0.15} />
                <span className="ms-2 text-xl text-muted">yrs</span>
              </p>
              <p className="mt-2 font-mono text-xs text-muted">
                ± {mae.toFixed(1)} years <span className="opacity-70">(mean error on unseen faces)</span>
              </p>
            </div>
          </motion.article>
        ))}
      </AnimatePresence>
    </div>
  );
}

/** "Downloading models… 12 / 25 MB" while model files arrive (first visit), then "Running the model…". */
function BrowserLoadingLabel() {
  const [p, setP] = useState<DownloadProgress>({ loaded: 0, total: 0, active: 0 });
  useEffect(() => onModelDownload(setP), []);
  if (p.active > 0 && p.total > 0) {
    const mb = (n: number) => (n / 1024 / 1024).toFixed(0);
    return <>Downloading models (first visit only)… {mb(p.loaded)} / {mb(p.total)} MB</>;
  }
  return <>Running the model in your browser…</>;
}
