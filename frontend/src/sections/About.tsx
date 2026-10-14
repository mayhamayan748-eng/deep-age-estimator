import { motion } from "motion/react";
import { SectionHeading } from "../components/SectionHeading";
import type { ModelInfo } from "../lib/api";

const REPO = "https://github.com/mayhamayan748-eng/deep-age-estimator";
const REPORT = `${REPO}/blob/main/${encodeURIComponent("التنبؤ بعمر االنسان باستخدام التعلم العميق.pdf")}`;

export function About({ info }: { info: ModelInfo | null }) {
  const cards = [
    {
      title: "Dataset",
      body: "UTKFace: 23,708 aligned and cropped face photos labelled with age, gender and ethnicity. Ages 0–90 are used, split 72% train, 8% validation and 20% test.",
      link: ["UTKFace website", "https://susanqq.github.io/UTKFace/"],
    },
    {
      title: "Architecture",
      body: "A 5-block CNN (Conv → BatchNorm → ReLU → MaxPool, 32→512 filters, global average pooling) trained with augmentation and Huber loss; the same CNN with GAN-generated faces added; and EfficientNetB0 fine-tuned from ImageNet.",
    },
    {
      title: "Generator",
      body: "A conditional DCGAN: the age goes into both networks and a projection discriminator checks that faces match it. Generated faces are filtered by the CNN before they are used for training.",
    },
  ];

  return (
    <section id="about">
      <div className="mx-auto max-w-6xl px-4 py-20 sm:px-6 lg:py-28">
        <SectionHeading index="04" kicker="About" title={<>The <em className="text-accent">project</em>.</>} />

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
              {c.link && (
                <a href={c.link[1]} target="_blank" rel="noreferrer" className="mt-3 inline-block text-sm text-muted underline underline-offset-4 hover:text-text">
                  {c.link[0]} ↗
                </a>
              )}
            </motion.div>
          ))}
        </div>

        {info && (
          <details className="mt-6 border border-line p-4">
            <summary className="label cursor-pointer">Preprocessing details</summary>
            <ol className="mt-3 list-decimal space-y-1 ps-5 text-sm text-muted">
              {info.preprocessing.map((p) => <li key={p}>{p}</li>)}
            </ol>
          </details>
        )}

        <div className="mt-12 flex flex-col gap-6 border-t border-line pt-8 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="label">Author</p>
            <p className="mt-1 font-display text-3xl">Ayham Ayan</p>
          </div>
          <div className="flex flex-wrap gap-3">
            <a href={REPORT} target="_blank" rel="noreferrer" className="rounded-full bg-accent px-5 py-2.5 text-sm font-medium text-bg">
              Read the report (PDF)
            </a>
            <a href={REPO} target="_blank" rel="noreferrer" className="rounded-full border border-line px-5 py-2.5 text-sm font-medium hover:border-accent">
              Source on GitHub
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}
