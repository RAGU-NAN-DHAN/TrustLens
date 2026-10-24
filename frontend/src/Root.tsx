import AnalyticsApp from "./analytics/AnalyticsApp";
import App from "./App";
import ConsoleApp from "./console/ConsoleApp";
import { usePalette, useUi } from "./prefs";

export default function Root() {
  const [ui, setUi] = useUi();
  const [palette, setPalette] = usePalette();
  const props = { ui, onUi: setUi, palette, onPalette: setPalette };
  if (ui === "console") return <ConsoleApp {...props} />;
  if (ui === "analytics") return <AnalyticsApp {...props} />;
  return <App {...props} />;
}
