/** Viewfinder-style corner brackets that fill their (relative) parent. */
export function CornerMarks({ className = "border-accent", size = "size-4" }: { className?: string; size?: string }) {
  // Physical sides on purpose: these frame images, which are never mirrored in RTL
  const corners = [
    "left-0 top-0 border-l-2 border-t-2",
    "right-0 top-0 border-r-2 border-t-2",
    "bottom-0 left-0 border-b-2 border-l-2",
    "bottom-0 right-0 border-b-2 border-r-2",
  ];
  return (
    <>
      {corners.map((c) => (
        <span key={c} aria-hidden className={`pointer-events-none absolute ${size} ${c} ${className}`} />
      ))}
    </>
  );
}
