import { useT } from "../i18n";
import { IN_BROWSER } from "../lib/api";

export function Footer() {
  const t = useT();
  return (
    <footer className="border-t border-line">
      <div className="label mx-auto flex max-w-6xl flex-col gap-2 px-4 py-6 sm:flex-row sm:justify-between sm:px-6">
        <span>{t.footer.tagline}</span>
        <span>{IN_BROWSER ? t.footer.privacyBrowser : t.footer.privacyServer}</span>
      </div>
    </footer>
  );
}
