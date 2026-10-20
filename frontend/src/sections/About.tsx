import { motion } from "motion/react";
import { SectionHeading } from "../components/SectionHeading";
import { useT } from "../i18n";
import { IN_BROWSER, type ModelInfo } from "../lib/api";

const REPO = "https://github.com/mayhamayan748-eng/deep-age-estimator";
const REPORT = `${REPO}/blob/main/${encodeURIComponent("التنبؤ بعمر االنسان باستخدام التعلم العميق.pdf")}`;

export function About({ info }: { info: ModelInfo | null }) {
  const t = useT();
  const cards = t.about.cards.map((c, i) => ({ ...c, url: i === 0 ? "https://susanqq.github.io/UTKFace/" : undefined }));
  const preprocessing = IN_BROWSER ? [...t.about.preprocessing, t.about.browserStep] : t.about.preprocessing;

  return (
    <section id="about">
      <div className="mx-auto max-w-6xl px-4 py-20 sm:px-6 lg:py-28">
        <SectionHeading index="04" kicker={t.about.kicker} title={t.about.title} />

        <div className="grid gap-px border border-line bg-line md:grid-cols-3">
          {cards.map((c, i) => (
            <motion.div
              key={c.title}
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1, duration: 0.6 }}
              className="bg-bg p-6"
            >
              <h3 className="label mb-3 text-accent">{c.title}</h3>
              <p className="text-sm leading-relaxed">{c.body}</p>
              {c.link && c.url && (
                <a href={c.url} target="_blank" rel="noreferrer" className="mt-3 inline-block text-sm text-muted underline underline-offset-4 hover:text-text">
                  {c.link} ↗
                </a>
              )}
            </motion.div>
          ))}
        </div>

        {info && (
          <details className="mt-6 border border-line p-4">
            <summary className="label cursor-pointer">{t.about.preprocessingSummary}</summary>
            <ol className="mt-3 list-decimal space-y-1 ps-5 text-sm text-muted">
              {preprocessing.map((p) => <li key={p}>{p}</li>)}
            </ol>
          </details>
        )}

        <div className="mt-12 flex flex-col gap-6 border-t border-line pt-8 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="label">{t.about.author}</p>
            <p className="mt-1 font-display text-3xl">{t.about.authorName}</p>
          </div>
          <div className="flex flex-wrap gap-3">
            <a href={REPORT} target="_blank" rel="noreferrer" className="rounded-full bg-accent px-5 py-2.5 text-sm font-medium text-bg">
              {t.about.readReport}
            </a>
            <a href={REPO} target="_blank" rel="noreferrer" className="rounded-full border border-line px-5 py-2.5 text-sm font-medium hover:border-accent">
              {t.about.source}
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}
