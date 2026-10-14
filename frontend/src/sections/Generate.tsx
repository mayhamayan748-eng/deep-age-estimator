import { motion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { SectionHeading } from "../components/SectionHeading";
import { api, ApiError, pngSrc, type GenerateResponse, type Health } from "../lib/api";

const randomSeed = () => Math.floor(Math.random() * 2 ** 31);

export function Generate({ health, offline }: { health: Health | null; offline: boolean }) {
  const [age, setAge] = useState(30);
  const [count, setCount] = useState(4);
  // A fixed seed keeps the same "people" while the age slider moves
  const [seed, setSeed] = useState(randomSeed);
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestId = useRef(0);

  const available = !offline && health?.generator === true;

  useEffect(() => {
    if (!available) return;
    const id = ++requestId.current;
    const timer = setTimeout(async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await api.generate(age, count, seed);
        if (id === requestId.current) setResult(res);
      } catch (e) {
        if (id === requestId.current) setError(e instanceof ApiError ? e.message : "Generation failed");
      } finally {
        if (id === requestId.current) setLoading(false);
      }
    }, 250); // debounce slider drags
    return () => clearTimeout(timer);
  }, [age, count, seed, available]);

  return (
    <section id="generate" className="border-b border-line bg-surface">
      <div className="mx-auto max-w-6xl px-4 py-20 sm:px-6 lg:py-28">
        <SectionHeading index="02" kicker="Generate" title={<>Now run it <em className="text-accent">backwards</em>.</>}>
          A conditional GAN learned to draw faces for a requested age. Drag the ruler: the random seed stays the same,
          so you watch the same imaginary people grow older.
        </SectionHeading>

        {!available ? (
          <p role="status" className="border-s-2 border-amber bg-bg px-4 py-3 text-sm text-muted">
            {offline || !health
              ? "Waiting for the server…"
              : "The GAN generator isn't loaded (gan_generator.keras is missing from ./models), so this feature is off."}
          </p>
        ) : (
          <>
            <div className="mb-8 grid gap-6 md:grid-cols-[1fr_auto] md:items-end">
              <div>
                <div className="mb-2 flex items-baseline justify-between">
                  <label htmlFor="age" className="label">Requested age</label>
                  <output htmlFor="age" className="font-display text-5xl leading-none tabular-nums">
                    {age}
                    <span className="ms-1 text-lg text-muted">yrs</span>
                  </output>
                </div>
                <input
                  id="age"
                  type="range"
                  min={1}
                  max={90}
                  value={age}
                  onChange={(e) => setAge(Number(e.target.value))}
                  className="ruler"
                />
                {/* Decade marks placed at the same fraction as the slider value, mirrored in RTL like the slider */}
                <div className="label relative h-7" aria-hidden>
                  {[1, 10, 20, 30, 40, 50, 60, 70, 80, 90].map((n) => (
                    <span
                      key={n}
                      className="absolute top-0 flex -translate-x-1/2 flex-col items-center rtl:translate-x-1/2"
                      style={{ insetInlineStart: `${((n - 1) / 89) * 100}%` }}
                    >
                      <span className="h-2 w-px bg-muted" />
                      {n}
                    </span>
                  ))}
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <div role="radiogroup" aria-label="Number of faces" className="flex rounded-full border border-line p-1">
                  {[4, 8].map((n) => (
                    <button
                      key={n}
                      role="radio"
                      aria-checked={count === n}
                      onClick={() => setCount(n)}
                      className={`rounded-full px-4 py-1.5 font-mono text-xs ${count === n ? "bg-text text-bg" : "text-muted"}`}
                    >
                      {n} faces
                    </button>
                  ))}
                </div>
                <button
                  type="button"
                  onClick={() => setSeed(randomSeed())}
                  className="rounded-full bg-accent px-5 py-2.5 text-sm font-medium text-bg transition-transform hover:-translate-y-0.5"
                >
                  New faces
                </button>
              </div>
            </div>

            {error && (
              <p role="alert" className="mb-4 border border-accent/40 bg-accent-soft px-4 py-3 text-sm">{error}</p>
            )}

            <FilmStrip result={result} count={count} loading={loading} />
            <p className="label mt-3">
              seed {seed} · 128×128 grayscale · generated from noise, not real people
            </p>
          </>
        )}
      </div>
    </section>
  );
}

function FilmStrip({ result, count, loading }: { result: GenerateResponse | null; count: number; loading: boolean }) {
  const images = result?.images ?? [];
  return (
    <div className="bg-[#0b0908] py-4" aria-busy={loading}>
      <div className="sprockets mx-3 mb-3 h-3 bg-[#2a2421]" aria-hidden />
      <div className={`grid gap-2 px-3 ${count > 4 ? "grid-cols-2 sm:grid-cols-4" : "grid-cols-2 md:grid-cols-4"}`}>
        {Array.from({ length: count }, (_, i) => (
          <div key={i} className="relative aspect-square overflow-hidden bg-[#1a1614]">
            {images[i] && (
              <motion.img
                // Re-mount on every new result so the stagger replays
                key={`${result!.seed}-${result!.age}-${i}`}
                src={pngSrc(images[i])}
                alt={`Generated face, age ${result!.age}`}
                initial={{ opacity: 0, scale: 1.08, filter: "blur(6px)" }}
                animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
                transition={{ duration: 0.45, delay: i * 0.06 }}
                className="size-full object-cover"
              />
            )}
            <span className="absolute bottom-1 left-1.5 font-mono text-[10px] text-[#e8b04a]">
              {String(i + 1).padStart(2, "0")}A
            </span>
            {loading && <div className="absolute inset-0 animate-pulse bg-[#ff5a36]/10" />}
          </div>
        ))}
      </div>
      <div className="sprockets mx-3 mt-3 h-3 bg-[#2a2421]" aria-hidden />
    </div>
  );
}
