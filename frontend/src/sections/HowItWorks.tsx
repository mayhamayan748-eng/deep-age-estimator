import { motion } from "motion/react";
import { SectionHeading } from "../components/SectionHeading";
import { useT } from "../i18n";
import { MODEL_LABELS, type FairFaceGroup, type ModelInfo } from "../lib/api";

const BAR_COLORS = ["bg-accent", "bg-amber", "bg-muted", "bg-text", "bg-line"];


export function HowItWorks({ info }: { info: ModelInfo | null }) {
  const t = useT();
  return (
    <section id="how" className="border-b border-line">
      <div className="mx-auto max-w-6xl px-4 py-20 sm:px-6 lg:py-28">
        <SectionHeading index="03" kicker={t.how.kicker} title={t.how.title} />

        <ol className="relative grid gap-4 md:grid-cols-5 md:gap-0">
          {/* connector line drawn as the section scrolls in */}
          <motion.span
            aria-hidden
            className="absolute start-0 top-[1.15rem] hidden h-px w-full origin-left bg-accent md:block rtl:origin-right"
            initial={{ scaleX: 0 }}
            whileInView={{ scaleX: 1 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 1.4, ease: [0.22, 1, 0.36, 1] }}
          />
          {t.how.steps.map(([title, text], i) => (
            <motion.li
              key={title}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-100px" }}
              transition={{ duration: 0.6, delay: 0.2 + i * 0.15 }}
              className="relative flex gap-4 md:flex-col md:pe-6"
            >
              <span className="relative z-10 grid size-9 shrink-0 place-items-center rounded-full border border-accent bg-bg font-mono text-xs text-accent">
                {String(i + 1).padStart(2, "0")}
              </span>
              <div>
                <h3 className="font-display text-2xl">{title}</h3>
                <p className="mt-1 text-sm text-muted">{text}</p>
              </div>
            </motion.li>
          ))}
        </ol>

        <Metrics info={info} />
      </div>
    </section>
  );
}

function Metrics({ info }: { info: ModelInfo | null }) {
  const t = useT();
  if (!info) {
    return <div className="mt-16 h-48 animate-pulse bg-surface" aria-label={t.how.loadingMetrics} />;
  }
  const models = info.models.filter((m) => m.test_mae != null);
  if (models.length === 0) {
    return <p className="mt-16 text-muted">{t.how.metricsMissing}</p>;
  }
  const groups = Object.keys(models[0].mae_by_age_group);
  const maxErr = Math.max(...models.flatMap((m) => Object.values(m.mae_by_age_group)));

  return (
    <>
    <div className="mt-20 grid gap-12 lg:grid-cols-2">
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.6 }}
      >
        <h3 className="label mb-4">
          {t.how.resultsOn(typeof info.dataset.test === "number" ? info.dataset.test.toLocaleString("en") : "")}
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-start font-mono text-sm">
            <thead>
              <tr className="border-b border-line text-muted">
                <th scope="col" className="py-2 text-start font-normal">{t.how.colModel}</th>
                <th scope="col" className="py-2 text-end font-normal">MAE</th>
                <th scope="col" className="py-2 text-end font-normal">RMSE</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.name} className="border-b border-line">
                  <th scope="row" className="py-3 text-start font-normal">
                    {MODEL_LABELS[m.name]}
                    {m.name === info.default_model && <span className="ms-2 text-accent">{t.how.inUse}</span>}
                  </th>
                  <td className="py-3 text-end">{m.test_mae!.toFixed(2)} {t.how.y}</td>
                  <td className="py-3 text-end">{m.test_rmse?.toFixed(2)} {t.how.y}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-muted">{t.how.tableNote}</p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 24 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.6, delay: 0.15 }}
      >
        <h3 className="label mb-4">{t.how.byAge}</h3>
        <div className="space-y-3">
          {groups.map((g) => (
            <div key={g} className="grid grid-cols-[3.5rem_1fr] items-center gap-3">
              <span dir="ltr" className="text-start font-mono text-xs text-muted">{g}</span>
              <div className="space-y-1">
                {models.map((m, mi) => {
                  const v = m.mae_by_age_group[g];
                  return (
                    <div key={m.name} className="flex items-center gap-2">
                      <motion.div
                        className={`h-2 origin-left rtl:origin-right ${BAR_COLORS[mi % BAR_COLORS.length]}`}
                        style={{ width: `${(v / maxErr) * 85}%` }}
                        initial={{ scaleX: 0 }}
                        whileInView={{ scaleX: 1 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.8, delay: mi * 0.1 }}
                      />
                      <span className="font-mono text-[10px] text-muted">{v.toFixed(1)}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 flex flex-wrap gap-4">
          {models.map((m, mi) => (
            <span key={m.name} className="label flex items-center gap-2">
              <span className={`size-2 ${BAR_COLORS[mi % BAR_COLORS.length]}`} />
              {MODEL_LABELS[m.name]}
            </span>
          ))}
        </div>
      </motion.div>
    </div>
    {info.fairface_test && <Diversity groups={info.fairface_test} />}
    </>
  );
}

/** Per-group FairFace results before and after adding diverse training data. */
function Diversity({ groups }: { groups: Record<string, FairFaceGroup> }) {
  const t = useT();
  const names = Object.keys(groups).filter((g) => g !== "All");
  const all = groups.All;
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.6 }}
      className="mt-16"
    >
      <h3 className="label mb-2">{t.how.diversityTitle(all?.faces.toLocaleString("en") ?? "")}</h3>
      <p className="mb-4 max-w-2xl text-sm text-muted">{t.how.diversityIntro}</p>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[34rem] font-mono text-sm">
          <thead>
            <tr className="border-b border-line text-muted">
              <th scope="col" className="py-2 text-start font-normal">{t.how.colGroup}</th>
              <th scope="col" className="py-2 text-end font-normal">{t.how.colBefore}</th>
              <th scope="col" className="py-2 text-end font-normal">{t.how.colAfter}</th>
              <th scope="col" className="py-2 text-end font-normal">{t.how.colInRange}</th>
            </tr>
          </thead>
          <tbody>
            {[...names, "All"].map((g) => {
              const r = groups[g];
              return (
                <tr key={g} className={`border-b border-line ${g === "All" ? "font-medium" : ""}`}>
                  <th scope="row" className="py-2.5 text-start font-normal">{t.how.groups[g] ?? g.replace("_", " / ")}</th>
                  <td className="py-2.5 text-end text-muted">{r.outside_before.toFixed(2)} {t.how.y}</td>
                  <td className="py-2.5 text-end text-accent">{r.outside_after.toFixed(2)} {t.how.y}</td>
                  <td className="py-2.5 text-end">
                    {/* Numbers and arrow read left-to-right in both languages */}
                    <span dir="ltr">{r.in_range_before.toFixed(0)}% → {r.in_range_after.toFixed(0)}%</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </motion.div>
  );
}
