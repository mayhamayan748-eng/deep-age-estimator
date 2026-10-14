import { useTheme } from "../hooks/useTheme";
import type { Health } from "../lib/api";

const NAV = [
  ["predict", "Predict"],
  ["generate", "Generate"],
  ["how", "How it works"],
  ["about", "About"],
] as const;

export function Header({ health, offline }: { health: Health | null; offline: boolean }) {
  const { theme, toggle } = useTheme();
  const status = offline ? "offline" : !health ? "connecting" : health.status === "ok" ? "online" : "degraded";
  const dot = { online: "bg-emerald-500", degraded: "bg-amber", offline: "bg-accent", connecting: "bg-muted animate-pulse" }[status];

  return (
    <header className="sticky top-0 z-50 border-b border-line bg-bg/80 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-6xl items-center gap-4 px-4 sm:px-6">
        <a href="#top" className="flex items-center gap-2 font-mono text-sm font-medium tracking-tight">
          <svg viewBox="0 0 32 32" className="size-6" aria-hidden>
            <path d="M7 11V7h4M21 7h4v4M25 21v4h-4M11 25H7v-4" stroke="currentColor" strokeWidth="2.5" fill="none" />
            <circle cx="16" cy="16" r="3" className="fill-accent" />
          </svg>
          age<span className="text-accent">·</span>lab
        </a>

        <nav aria-label="Sections" className="ms-auto hidden gap-6 md:flex">
          {NAV.map(([id, label]) => (
            <a key={id} href={`#${id}`} className="label transition-colors hover:text-text">
              {label}
            </a>
          ))}
        </nav>

        <div className="ms-auto flex items-center gap-3 md:ms-6">
          <span className="label hidden items-center gap-2 sm:flex" title={`API ${status}`}>
            <span className={`size-1.5 rounded-full ${dot}`} />
            API {status}
          </span>
          <button
            type="button"
            onClick={toggle}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            className="grid size-9 place-items-center rounded-full border border-line text-muted transition-colors hover:border-accent hover:text-text"
          >
            {theme === "dark" ? (
              <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
                <circle cx="12" cy="12" r="4" />
                <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
                <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
              </svg>
            )}
          </button>
        </div>
      </div>
    </header>
  );
}
