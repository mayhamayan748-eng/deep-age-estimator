import { useEffect, useRef, useState } from "react";
import { CornerMarks } from "./CornerMarks";

export function Webcam({ onCapture, onCancel }: { onCapture: (b: Blob) => void; onCancel: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let stream: MediaStream | null = null;
    let cancelled = false;
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("This browser doesn't support camera access (it needs HTTPS or localhost).");
      return;
    }
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 } } })
      .then((s) => {
        if (cancelled) return s.getTracks().forEach((t) => t.stop());
        stream = s;
        if (videoRef.current) videoRef.current.srcObject = s;
      })
      .catch(() => setError("Camera access was denied or no camera was found."));
    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const capture = () => {
    const video = videoRef.current;
    if (!video) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")!.drawImage(video, 0, 0);
    canvas.toBlob((b) => b && onCapture(b), "image/jpeg", 0.92);
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="relative aspect-[4/3] overflow-hidden bg-surface">
        {error ? (
          <p className="absolute inset-0 grid place-items-center p-6 text-center text-muted">{error}</p>
        ) : (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            onLoadedData={() => setReady(true)}
            className="size-full object-cover"
          />
        )}
        <CornerMarks size="size-6" />
      </div>
      <div className="flex gap-3">
        <button
          type="button"
          onClick={capture}
          disabled={!ready}
          className="rounded-full bg-accent px-5 py-2.5 text-sm font-medium text-bg disabled:opacity-40"
        >
          Capture
        </button>
        <button type="button" onClick={onCancel} className="rounded-full border border-line px-5 py-2.5 text-sm font-medium">
          Cancel
        </button>
      </div>
    </div>
  );
}
