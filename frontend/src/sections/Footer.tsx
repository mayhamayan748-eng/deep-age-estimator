import { IN_BROWSER } from "../lib/api";

export function Footer() {
  return (
    <footer className="border-t border-line">
      <div className="label mx-auto flex max-w-6xl flex-col gap-2 px-4 py-6 sm:flex-row sm:justify-between sm:px-6">
        <span>Deep Age Estimator · graduation project</span>
        <span>{IN_BROWSER ? "Runs in your browser: photos never leave your device" : "Photos are processed in memory and never stored"}</span>
      </div>
    </footer>
  );
}
