// Dev-only animation test page. Run with `npm run anim` from frontend/; it is not part of `npm run build`.
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  root: "anim-viewer",
  plugins: [react()],
  server: { port: 5190, strictPort: true, fs: { allow: [".."] } },
});
