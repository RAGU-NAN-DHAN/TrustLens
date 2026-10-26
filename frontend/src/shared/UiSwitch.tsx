import { THEMES, UIS, type UiName } from "../prefs";

const THEME_LABEL: Record<string, string> = { dark: "Dark", light: "Light", graphite: "Graphite" };

interface Props {
  ui: UiName;
  onUi: (ui: UiName) => void;
  theme: string;
  onTheme: (t: string) => void;
  className?: string;
}

export default function UiSwitch({ ui, onUi, theme, onTheme, className = "" }: Props) {
  return (
    <div className={`ui-switch ${className}`}>
      {UIS.length > 1 && (
        <label>
          <span>Layout</span>
          <select value={ui} onChange={(e) => onUi(e.target.value as UiName)} aria-label="Layout">
            {UIS.map((u) => (
              <option key={u} value={u}>
                {u}
              </option>
            ))}
          </select>
        </label>
      )}
      {ui === "terminal" && (
        <label>
          <span>Theme</span>
          <select value={theme} onChange={(e) => onTheme(e.target.value)} aria-label="Background theme">
            {THEMES.map((t) => (
              <option key={t} value={t}>
                {THEME_LABEL[t]}
              </option>
            ))}
          </select>
        </label>
      )}
    </div>
  );
}
