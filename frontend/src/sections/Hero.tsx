import { motion } from "motion/react";
import { CornerMarks } from "../components/CornerMarks";
import { useT } from "../i18n";
import type { ModelInfo } from "../lib/api";

const ease = [0.22, 1, 0.36, 1] as const;

export function Hero({ info }: { info: ModelInfo | null }) {
  const t = useT();
  const best = info?.models.find((m) => m.name === info.default_model);
  const stats = [
    [t.hero.statData, t.hero.statDataValue],
    [t.hero.statMae, best?.test_mae != null ? t.hero.statMaeValue(best.test_mae.toFixed(2)) : "—"],
    [t.hero.statModels, t.hero.statModelsValue],
    [t.hero.statGenerator, t.hero.statGeneratorValue],
  ];
  const headline = t.hero.headline;

  return (
    <section id="top" className="relative overflow-hidden border-b border-line">
      <div className="mx-auto grid max-w-6xl gap-12 px-4 pb-16 pt-16 sm:px-6 lg:grid-cols-[1.25fr_1fr] lg:items-center lg:pb-24 lg:pt-24">
        <div>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6 }}
            className="label mb-6 flex items-center gap-3"
          >
            <span className="size-1.5 rounded-full bg-accent" /> {t.hero.kicker}
          </motion.p>

          {/* Words are animated one by one; the label gives screen readers the sentence with its spaces */}
          <h1 className="font-display text-5xl leading-[0.95] sm:text-7xl" aria-label={headline.join(" ")}>
            {headline.map((word, i) => (
              <motion.span
                aria-hidden
                key={`${word}-${i}`}
                initial={{ opacity: 0, y: "0.4em" }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.7, delay: 0.1 + i * 0.06, ease }}
                className={`me-[0.22em] inline-block ${i === t.hero.accentWord ? "italic text-accent" : ""}`}
              >
                {word}
              </motion.span>
            ))}
          </h1>

          <motion.p
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.7, ease }}
            className="mt-6 max-w-xl text-lg text-muted"
          >
            {t.hero.sub}
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.85, ease }}
            className="mt-8 flex flex-wrap gap-3"
          >
            <a href="#predict" className="rounded-full bg-accent px-6 py-3 font-medium text-bg transition-transform hover:-translate-y-0.5">
              {t.hero.ctaEstimate}
            </a>
            <a href="#generate" className="rounded-full border border-line px-6 py-3 font-medium transition-colors hover:border-accent">
              {t.hero.ctaGenerate}
            </a>
          </motion.div>
        </div>

        <HeroViewfinder />
      </div>

      <motion.dl
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.8, delay: 1.1 }}
        className="mx-auto grid max-w-6xl grid-cols-2 border-t border-line px-4 sm:px-6 lg:grid-cols-4"
      >
        {stats.map(([k, v]) => (
          <div key={k} className="border-line py-5 pe-4 [&:not(:last-child)]:lg:border-e">
            <dt className="label">{k}</dt>
            <dd className="mt-1 font-mono text-sm" dir="auto">{v}</dd>
          </div>
        ))}
      </motion.dl>
    </section>
  );
}

/** Decorative detector frame: an abstract face outline being scanned. */
function HeroViewfinder() {
  return (
    <motion.div
      aria-hidden
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 1, delay: 0.3, ease }}
      className="mx-auto w-full max-w-sm"
    >
      <div className="relative aspect-[4/5] bg-surface">
        <CornerMarks size="size-6" />
        <svg viewBox="0 0 200 250" className="absolute inset-8 h-[calc(100%-4rem)] w-[calc(100%-4rem)] text-muted">
          <motion.ellipse
            cx="100" cy="115" rx="58" ry="76" fill="none" stroke="currentColor" strokeWidth="1"
            initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.6, delay: 0.6, ease }}
          />
          {[[78, 100], [122, 100]].map(([cx, cy], i) => (
            <motion.circle key={i} cx={cx} cy={cy} r="3" className="fill-accent"
              initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 1.6 + i * 0.1 }} />
          ))}
          <motion.path d="M100 108 L96 132 L104 132" fill="none" stroke="currentColor" strokeWidth="1"
            initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 0.6, delay: 1.8 }} />
          <motion.path d="M84 150 Q100 160 116 150" fill="none" stroke="currentColor" strokeWidth="1"
            initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 0.6, delay: 2 }} />
          <line x1="100" y1="20" x2="100" y2="230" stroke="currentColor" strokeWidth="0.5" strokeDasharray="2 4" opacity="0.5" />
          <line x1="20" y1="115" x2="180" y2="115" stroke="currentColor" strokeWidth="0.5" strokeDasharray="2 4" opacity="0.5" />
        </svg>

        {/* scan line */}
        <motion.div
          className="absolute inset-x-6 h-px bg-accent shadow-[0_0_12px_var(--accent)]"
          initial={{ top: "8%" }}
          animate={{ top: ["8%", "92%", "8%"] }}
          transition={{ duration: 4, repeat: Infinity, ease: "easeInOut", delay: 1 }}
        />
      </div>

      {/* Readout sits below the frame so all four corner marks stay visible */}
      <div className="mt-3 flex justify-between px-1 font-mono text-[11px] text-muted">
        <span>FACE 01 · 128×128</span>
        <span className="text-accent">EST. AGE ▮▮</span>
      </div>
    </motion.div>
  );
}
