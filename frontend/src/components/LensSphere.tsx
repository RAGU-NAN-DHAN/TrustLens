import { useEffect, useRef } from "react";
import type { Verdict } from "../types/trustlens";

const RAMP = " .:-=+*%#@";
type RGB = [number, number, number];
const IDLE: RGB = [150, 160, 175];

function readPalette(): Record<Verdict | "idle", RGB> {
  const style = getComputedStyle(document.documentElement);
  const read = (name: string, fallback: RGB): RGB => {
    const parts = style.getPropertyValue(name).split(",").map((v) => Number(v.trim()));
    return parts.length === 3 && parts.every((n) => Number.isFinite(n)) ? (parts as RGB) : fallback;
  };
  return {
    supported: read("--sup-rgb", [0, 229, 255]),
    contradicted: read("--con-rgb", [255, 46, 136]),
    unverifiable: read("--unv-rgb", [176, 176, 176]),
    idle: IDLE,
  };
}
const CELL_W = 9;
const CELL_H = 14;

interface Props {
  verdicts: Verdict[];
  palette: string;
}

// Each claim owns a wedge of longitude, so the sphere is literally made of the answer's claims.
export default function LensSphere({ verdicts, palette }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const verdictsRef = useRef(verdicts);
  verdictsRef.current = verdicts;
  const colorsRef = useRef<Record<Verdict | "idle", RGB> | null>(null);

  // the parent applies data-palette in its own effect, which runs after ours, so re-read lazily on the next frame
  useEffect(() => {
    colorsRef.current = null;
  }, [palette]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    let frame = 0;
    let width = 0;
    let height = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    function resize() {
      const rect = canvas!.getBoundingClientRect();
      width = rect.width;
      height = rect.height;
      canvas!.width = Math.round(width * dpr);
      canvas!.height = Math.round(height * dpr);
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function draw(time: number) {
      const angle = time * (reduceMotion ? 0.00008 : 0.00025);
      const list = verdictsRef.current;
      const colors = colorsRef.current ?? (colorsRef.current = readPalette());
      ctx!.clearRect(0, 0, width, height);

      const cx = width / 2;
      const cy = height / 2;
      const radius = Math.min(width * 0.42, height * 0.4);

      const glow = ctx!.createRadialGradient(cx, cy, radius * 0.2, cx, cy, radius * 1.6);
      glow.addColorStop(0, "rgba(255,255,255,0.10)");
      glow.addColorStop(0.5, "rgba(255,255,255,0.03)");
      glow.addColorStop(1, "rgba(0,0,0,0)");
      ctx!.fillStyle = glow;
      ctx!.fillRect(0, 0, width, height);

      ctx!.font = `600 ${CELL_H - 2}px "JetBrains Mono", monospace`;
      ctx!.textBaseline = "middle";
      ctx!.textAlign = "center";

      const light = [-0.45, -0.55, 0.7];
      const ln = Math.hypot(light[0], light[1], light[2]);
      const lx = light[0] / ln, ly = light[1] / ln, lz = light[2] / ln;
      const cosA = Math.cos(angle), sinA = Math.sin(angle);
      const tilt = 0.35, cosT = Math.cos(tilt), sinT = Math.sin(tilt);

      for (let py = cy - radius; py <= cy + radius; py += CELL_H) {
        for (let px = cx - radius; px <= cx + radius; px += CELL_W) {
          const x = (px - cx) / radius;
          const y = (py - cy) / radius;
          const r2 = x * x + y * y;
          if (r2 > 1) continue;
          const z = Math.sqrt(1 - r2);

          const brightness = Math.max(0, x * lx + y * ly + z * lz);
          const ch = RAMP[Math.min(RAMP.length - 1, Math.floor(brightness * RAMP.length))];
          if (ch === " ") continue;

          // undo tilt, then spin around the vertical axis to find the surface point's longitude
          const ty = y * cosT - z * sinT;
          const tz = y * sinT + z * cosT;
          const rx = x * cosA - tz * sinA;
          const rz = x * sinA + tz * cosA;
          const lon = (Math.atan2(rx, rz) / (2 * Math.PI) + 1) % 1;
          const lat = Math.asin(Math.max(-1, Math.min(1, -ty)));

          const verdict: Verdict | "idle" = list.length ? list[Math.floor(lon * list.length) % list.length] : "idle";
          const [cr, cg, cb] = colors[verdict];
          const seam = list.length > 1 && Math.abs(((lon * list.length) % 1) - 0.5) > 0.46;
          const alpha = Math.min(1, 0.18 + brightness * 0.95) * (seam ? 0.35 : 1);
          const lift = brightness > 0.92 ? 0.55 : 0;
          const mix = (c: number) => Math.round(c + (255 - c) * lift);

          ctx!.fillStyle = `rgba(${mix(cr)},${mix(cg)},${mix(cb)},${alpha.toFixed(3)})`;
          ctx!.fillText(Math.abs(lat) > 1.25 ? "·" : ch, px, py);
        }
      }

      frame = requestAnimationFrame(draw);
    }

    resize();
    const observer = typeof ResizeObserver !== "undefined" ? new ResizeObserver(resize) : null;
    observer?.observe(canvas);
    frame = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(frame);
      observer?.disconnect();
    };
  }, []);

  return <canvas ref={canvasRef} className="lens-canvas" aria-hidden="true" />;
}
