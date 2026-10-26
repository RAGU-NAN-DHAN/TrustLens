import { useEffect, useState } from "react";

export const THEMES = ["dark", "light", "graphite"] as const;
export const PALETTE = "heat";
export const UIS = ["terminal"] as const;
export type UiName = (typeof UIS)[number];

function load<T extends string>(key: string, allowed: readonly T[], fallback: T): T {
  try {
    const param = key === "trustlens.ui" ? "ui" : "theme";
    const fromUrl = new URLSearchParams(window.location.search).get(param);
    const saved = fromUrl ?? localStorage.getItem(key);
    if (saved && (allowed as readonly string[]).includes(saved)) return saved as T;
  } catch {
    // storage unavailable: use the default
  }
  return fallback;
}

function save(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // storage unavailable: the choice just won't persist
  }
}

export function useTheme(): [string, (t: string) => void] {
  const [theme, setTheme] = useState<string>(() => load("trustlens.theme", THEMES, "dark"));
  useEffect(() => {
    document.documentElement.dataset.palette = PALETTE;
    document.documentElement.dataset.theme = theme;
    save("trustlens.theme", theme);
  }, [theme]);
  return [theme, setTheme];
}

export function useUi(): [UiName, (u: UiName) => void] {
  const [ui, setUi] = useState<UiName>(() => load("trustlens.ui", UIS, "terminal"));
  useEffect(() => {
    document.body.dataset.ui = ui;
    save("trustlens.ui", ui);
  }, [ui]);
  return [ui, setUi];
}
