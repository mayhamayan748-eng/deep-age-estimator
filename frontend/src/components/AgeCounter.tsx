import { animate, useReducedMotion } from "motion/react";
import { useEffect, useState } from "react";

/** Counts up from 0 to `value`; jumps straight to it when reduced motion is requested. */
export function AgeCounter({ value, delay = 0 }: { value: number; delay?: number }) {
  const reduce = useReducedMotion();
  const [shown, setShown] = useState(reduce ? value : 0);

  useEffect(() => {
    if (reduce) {
      setShown(value);
      return;
    }
    const controls = animate(0, value, {
      duration: 1.4,
      delay,
      ease: [0.16, 1, 0.3, 1],
      onUpdate: setShown,
    });
    return () => controls.stop();
  }, [value, delay, reduce]);

  return (
    <span aria-label={`${Math.round(value)} years`} className="tabular-nums">
      {Math.round(shown)}
    </span>
  );
}
