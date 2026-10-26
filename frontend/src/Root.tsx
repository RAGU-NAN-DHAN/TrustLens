import App from "./App";
import { useTheme, useUi } from "./prefs";

export default function Root() {
  const [ui, setUi] = useUi();
  const [theme, setTheme] = useTheme();
  return <App ui={ui} onUi={setUi} theme={theme} onTheme={setTheme} />;
}
