import AnalyticsApp from "./analytics/AnalyticsApp";
import App from "./App";
import ConsoleApp from "./console/ConsoleApp";
import { useTheme, useUi } from "./prefs";

export default function Root() {
  const [ui, setUi] = useUi();
  const [theme, setTheme] = useTheme();
  const props = { ui, onUi: setUi, theme, onTheme: setTheme };
  if (ui === "console") return <ConsoleApp {...props} />;
  if (ui === "analytics") return <AnalyticsApp {...props} />;
  return <App {...props} />;
}
