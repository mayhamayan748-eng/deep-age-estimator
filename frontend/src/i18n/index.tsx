import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { ar } from "./ar";
import { en, type Dict } from "./en";

export type Lang = "en" | "ar";
const DICTS: Record<Lang, Dict> = { en, ar };

/** The language index.html applied before first paint (saved choice, else the browser's language). */
function initialLang(): Lang {
  return document.documentElement.lang === "ar" ? "ar" : "en";
}

function apply(lang: Lang) {
  const root = document.documentElement;
  root.lang = lang;
  root.dir = lang === "ar" ? "rtl" : "ltr";
  document.title = DICTS[lang].meta.title;
  document.querySelector('meta[name="description"]')?.setAttribute("content", DICTS[lang].meta.description);
}

const LangContext = createContext<{ lang: Lang; t: Dict; toggle: () => void }>({ lang: "en", t: en, toggle: () => {} });

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>(initialLang);

  useEffect(() => apply(lang), [lang]);

  const toggle = useCallback(() => {
    setLang((prev) => {
      const next: Lang = prev === "en" ? "ar" : "en";
      try {
        localStorage.setItem("lang", next);
      } catch {
        /* storage unavailable: the choice still applies for this visit */
      }
      return next;
    });
  }, []);

  return <LangContext.Provider value={{ lang, t: DICTS[lang], toggle }}>{children}</LangContext.Provider>;
}

export const useLang = () => useContext(LangContext);
export const useT = () => useContext(LangContext).t;
