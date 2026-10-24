import { PALETTES, UIS, type UiName } from "../prefs";

interface Props {
  ui: UiName;
  onUi: (ui: UiName) => void;
  palette: string;
  onPalette: (p: string) => void;
  className?: string;
}

export default function UiSwitch({ ui, onUi, palette, onPalette, className = "" }: Props) {
  return (
    <div className={`ui-switch ${className}`}>
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
      <label>
        <span>Colours</span>
        <select value={palette} onChange={(e) => onPalette(e.target.value)} aria-label="Colour theme">
          {PALETTES.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
