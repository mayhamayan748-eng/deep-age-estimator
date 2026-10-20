import { useRef, useState } from "react";
import { MAX_UPLOAD_MB } from "../lib/api";

export function Dropzone({ onFile, onWebcam }: { onFile: (f: File) => void; onWebcam: () => void }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <div
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
      className={`relative flex aspect-[4/3] flex-col items-center justify-center gap-5 border border-dashed p-6 text-center transition-colors ${
        dragging ? "border-accent bg-accent-soft" : "border-line bg-surface"
      }`}
    >
      <svg viewBox="0 0 48 48" className="size-12 text-muted" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
        <path d="M6 16V8h8M34 8h8v8M42 32v8h-8M14 40H6v-8" />
        <circle cx="24" cy="21" r="7" />
        <path d="M13 37c1.5-6 5.5-9 11-9s9.5 3 11 9" />
      </svg>
      <div>
        <p className="font-display text-2xl">Drop a photo here</p>
        <p className="mt-1 text-sm text-muted">JPG, PNG or WebP · up to {MAX_UPLOAD_MB} MB · faces are detected automatically</p>
      </div>
      <div className="flex flex-wrap justify-center gap-3">
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="rounded-full bg-accent px-5 py-2.5 text-sm font-medium text-bg"
        >
          Choose a file
        </button>
        <button
          type="button"
          onClick={onWebcam}
          className="rounded-full border border-line px-5 py-2.5 text-sm font-medium transition-colors hover:border-accent"
        >
          Use webcam
        </button>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/*,.heic,.heif"
        className="sr-only"
        aria-label="Upload a photo"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = "";
        }}
      />
    </div>
  );
}
