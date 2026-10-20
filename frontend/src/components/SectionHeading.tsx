import { motion } from "motion/react";
import type { ReactNode } from "react";

export function SectionHeading({ index, kicker, title, children }: {
  index: string;
  kicker: string;
  title: ReactNode | readonly [string, string, string];
  children?: ReactNode;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      className="mb-10 max-w-2xl"
    >
      <p className="label mb-4 flex items-center gap-3">
        <span className="text-accent">{index}</span>
        <span className="h-px w-8 bg-line" />
        {kicker}
      </p>
      <h2 className="font-display text-4xl leading-[1.05] sm:text-5xl">
        {Array.isArray(title) ? (
          <>
            {title[0]}
            <em className="text-accent">{title[1]}</em>
            {title[2]}
          </>
        ) : (
          title
        )}
      </h2>
      {children && <p className="mt-4 text-muted">{children}</p>}
    </motion.div>
  );
}
