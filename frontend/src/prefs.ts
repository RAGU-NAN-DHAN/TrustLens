import { useEffect, useState } from "react";

export const PALETTES = ["mono", "signal", "aurora", "blueprint", "phosphor"] as const;
export const UIS = ["terminal", "console", "analytics"] as const;
export type UiName = (typeof UIS)[number];

function load<T extends string>(key: string, allowed: readonly T[], fallback: T): T {
  try {
    const param = key === "trustlens.ui" ? "ui" : "palette";
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

export function usePalette(): [string, (p: string) => void] {
  const [palette, setPalette] = useState<string>(() => load("trustlens.palette", PALETTES, "mono"));
  useEffect(() => {
    document.documentElement.dataset.palette = palette;
    save("trustlens.palette", palette);
  }, [palette]);
  return [palette, setPalette];
}

export function useUi(): [UiName, (u: UiName) => void] {
  const [ui, setUi] = useState<UiName>(() => load("trustlens.ui", UIS, "terminal"));
  useEffect(() => {
    document.body.dataset.ui = ui;
    save("trustlens.ui", ui);
  }, [ui]);
  return [ui, setUi];
}
