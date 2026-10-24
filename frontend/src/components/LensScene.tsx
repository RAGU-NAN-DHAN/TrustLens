import { useEffect, useRef } from "react";
import type { Verdict } from "../types/trustlens";

export type SceneMode = "idle" | "loading" | Verdict;

type RGB = [number, number, number];
type Hit = [number, RGB] | null;

const RAMP = " .:-=+*%#@";
const CELL_W = 9;
const CELL_H = 14;
const FADE_MS = 600;
const IDLE_GREY: RGB = [150, 160, 175];

const HEAT_BANDS: [number, RGB][] = [
  [0.12, [255, 236, 140]],
  [0.26, [246, 196, 92]],
  [0.46, [236, 144, 60]],
  [0.84, [222, 98, 42]],
  [1.0, [214, 50, 30]],
];

function heatBand(t: number): RGB {
  for (const [edge, color] of HEAT_BANDS) if (t <= edge) return color;
  return HEAT_BANDS[HEAT_BANDS.length - 1][1];
}

function heatRamp(v: number): RGB {
  // 0 = dark ember, 1 = white-yellow core
  const stops: [number, RGB][] = [
    [0, [60, 10, 6]],
    [0.25, [160, 34, 20]],
    [0.5, [222, 96, 42]],
    [0.72, [244, 170, 72]],
    [0.9, [255, 232, 140]],
    [1, [255, 250, 220]],
  ];
  const x = Math.min(1, Math.max(0, v));
  for (let i = 1; i < stops.length; i++) {
    const [p1, c1] = stops[i];
    const [p0, c0] = stops[i - 1];
    if (x <= p1) {
      const f = (x - p0) / (p1 - p0);
      return [0, 1, 2].map((k) => Math.round(c0[k] + (c1[k] - c0[k]) * f)) as RGB;
    }
  }
  return stops[stops.length - 1][1];
}

// light page: keep the warm hue saturated (cut blue, then green) instead of just darkening to brown
function onLight([r, g, b]: RGB): RGB {
  if (r > 200 && g > 200 && b > 180) return [38, 28, 24];
  return [Math.round(r * 0.86), Math.round(g * 0.6), Math.round(b * 0.3)];
}

function hash(a: number, b: number): number {
  const s = Math.sin(a * 127.1 + b * 311.7) * 43758.5453;
  return s - Math.floor(s);
}

function noise(x: number, y: number): number {
  return (
    0.5 +
    0.25 * Math.sin(x * 1.7 + Math.sin(y * 1.3)) +
    0.15 * Math.sin(y * 2.9 - x * 1.1 + Math.sin(x * 2.3)) +
    0.1 * Math.sin((x + y) * 4.3)
  );
}

function readPalette(): Record<Verdict, RGB> {
  const style = getComputedStyle(document.documentElement);
  const read = (name: string, fallback: RGB): RGB => {
    const parts = style.getPropertyValue(name).split(",").map((v) => Number(v.trim()));
    return parts.length === 3 && parts.every((n) => Number.isFinite(n)) ? (parts as RGB) : fallback;
  };
  return {
    supported: read("--sup-rgb", [250, 226, 110]),
    contradicted: read("--con-rgb", [196, 46, 28]),
    unverifiable: read("--unv-rgb", [228, 116, 42]),
  };
}

const LIGHT = (() => {
  const l = [-0.45, -0.55, 0.7];
  const n = Math.hypot(l[0], l[1], l[2]);
  return [l[0] / n, l[1] / n, l[2] / n];
})();

// shaded, spinning sphere; returns brightness plus latitude/longitude of the surface point
function sphere(x: number, y: number, t: number, spin: number): { v: number; lat: number; lon: number } | null {
  const r2 = x * x + y * y;
  if (r2 > 1) return null;
  const z = Math.sqrt(1 - r2);
  const shade = Math.max(0, x * LIGHT[0] + y * LIGHT[1] + z * LIGHT[2]);
  const tilt = 0.35;
  const ty = y * Math.cos(tilt) - z * Math.sin(tilt);
  const tz = y * Math.sin(tilt) + z * Math.cos(tilt);
  const a = t * spin;
  const rx = x * Math.cos(a) - tz * Math.sin(a);
  const rz = x * Math.sin(a) + tz * Math.cos(a);
  const lon = (Math.atan2(rx, rz) / (2 * Math.PI) + 1) % 1;
  const lat = Math.asin(Math.max(-1, Math.min(1, -ty)));
  const meridian = 0.7 + 0.3 * Math.cos(lon * 2 * Math.PI * 9);
  return { v: Math.max(0.05, shade * meridian), lat, lon };
}

interface SceneCtx {
  t: number;
  row: number;
  colors: Record<Verdict, RGB>;
  heat: boolean;
}

function idleScene(x: number, y: number, c: SceneCtx): Hit {
  // breathe and squash slightly so the shape keeps changing within its bounds
  const breathe = 1 + 0.05 * Math.sin(c.t * 1.1);
  const squash = 1 + 0.14 * Math.sin(c.t * 0.55);
  const s = sphere(x / breathe, (y * squash) / breathe, c.t, 0.35);
  if (!s) return null;
  const color = c.heat ? heatBand(0.5 - s.lat / Math.PI) : IDLE_GREY;
  // keep the red lower band visible even where the globe is in shadow
  return [c.heat ? Math.max(s.v, 0.2) : s.v, color];
}

function supportedScene(x: number, y: number, c: SceneCtx, since: number): Hit {
  const d = Math.hypot(x, y);
  let v = 0;
  for (let k = 0; k < 3; k++) {
    const radius = ((c.t * 0.45 + k / 3) % 1) * 1.35;
    v = Math.max(v, Math.exp(-((d - radius) ** 2) / 0.003) * (1 - radius / 1.35) * 0.8);
  }
  // checkmark drawn in over the first second
  const pts: [number, number][] = [
    [-0.5, 0.02],
    [-0.14, 0.38],
    [0.52, -0.36],
  ];
  const lens = [Math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]), Math.hypot(pts[2][0] - pts[1][0], pts[2][1] - pts[1][1])];
  const total = lens[0] + lens[1];
  const drawn = Math.min(1, since / 900) * total;
  let acc = 0;
  for (let i = 0; i < 2; i++) {
    const [ax, ay] = pts[i];
    const [bx, by] = pts[i + 1];
    const dx = bx - ax;
    const dy = by - ay;
    const len2 = dx * dx + dy * dy;
    const u = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / len2));
    const along = acc + u * lens[i];
    const dist = Math.hypot(x - (ax + u * dx), y - (ay + u * dy));
    if (along <= drawn && dist < 0.085) v = Math.max(v, 1 - dist * 4);
    acc += lens[i];
  }
  if (v < 0.04) return null;
  return [v, c.colors.supported];
}

function crackedSphere(x: number, y: number, t: number): { v: number; crack: boolean } | null {
  const crack = 0.07 * Math.sin(y * 9) + 0.04 * Math.sin(y * 23 + 1.3);
  const gap = 0.1 + 0.05 * Math.sin(t * 2.2);
  if (Math.abs(x - crack) < gap * 0.5 && Math.abs(y) < 0.95) return { v: 0.35, crack: true };
  const right = x > crack;
  // plain shaded halves (no spinning meridians), so it never reads as the idle globe
  const hx = x + (right ? -gap : gap);
  const hy = y + (right ? 0.03 : -0.03);
  const r2 = hx * hx + hy * hy;
  if (r2 > 1) return null;
  const z = Math.sqrt(1 - r2);
  const shade = Math.max(0, hx * LIGHT[0] + hy * LIGHT[1] + z * LIGHT[2]);
  return { v: Math.max(0.06, shade * 0.85), crack: false };
}

function contradictedScene(x: number, y: number, c: SceneCtx): Hit {
  const burst = hash(Math.floor(c.t * 6), 7) > 0.72;
  const tear = burst && hash(c.row, Math.floor(c.t * 30)) > 0.55;
  const gx = x + (tear ? (hash(c.row, Math.floor(c.t * 30) + 1) - 0.5) * 0.34 : 0);
  const split = 0.035 + (burst ? 0.075 : 0);

  const main = crackedSphere(gx, y, c.t);
  if (main) {
    if (main.crack) return [0.25 + 0.6 * hash(c.row, Math.floor(c.t * 12)), [255, 246, 230]];
    // occasional bright scan rows during a burst
    if (tear && hash(c.row, 5) > 0.6) return [Math.min(1, main.v + 0.35), hash(c.row, 9) > 0.5 ? c.colors.supported : [255, 246, 230]];
    return [main.v, c.colors.contradicted];
  }
  // colour-split ghosts on either side of the shape
  const left = crackedSphere(gx + split, y, c.t);
  if (left) return [left.v * 0.7, c.colors.unverifiable];
  const right = crackedSphere(gx - split, y, c.t);
  if (right) return [right.v * 0.7, c.colors.supported];
  return null;
}

function unverifiableScene(x: number, y: number, c: SceneCtx): Hit {
  const d = Math.hypot(x, y);
  const n = noise(x * 2.2 + c.t * 0.35, y * 2.2 - c.t * 0.25);
  const edge = 1.05 + 0.25 * (n - 0.5);
  if (d > edge) return null;
  let v = (0.12 + 0.34 * n) * Math.min(1, (edge - d) * 3);
  // pulsing question mark: arc, stem, dot
  const pulse = 0.55 + 0.45 * Math.sin(c.t * 2.4);
  const ax = x;
  const ay = y + 0.28;
  const arc = Math.abs(Math.hypot(ax, ay) - 0.27) < 0.07 && !(ax < 0 && ay > 0);
  const stem = Math.abs(x) < 0.07 && y > 0.0 && y < 0.24;
  const dot = Math.hypot(x, y - 0.42) < 0.08;
  if (arc || stem || dot) {
    const [r, g, b] = c.colors.unverifiable;
    const lift = 0.35 * pulse;
    return [Math.max(v, 0.7 + 0.3 * pulse), [r + (255 - r) * lift, g + (255 - g) * lift, b + (255 - b) * lift].map(Math.round) as RGB];
  }
  return [Math.min(1, v), c.colors.unverifiable];
}

interface Props {
  mode: SceneMode;
  palette: string;
}

export default function LensScene({ mode, palette }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modeRef = useRef<{ current: SceneMode; previous: SceneMode | null; since: number }>({ current: mode, previous: null, since: 0 });
  const colorsRef = useRef<Record<Verdict, RGB> | null>(null);
  const heatRef = useRef(palette === "heat");
  heatRef.current = palette === "heat";

  useEffect(() => {
    colorsRef.current = null;
  }, [palette]);

  useEffect(() => {
    const m = modeRef.current;
    if (m.current !== mode) {
      modeRef.current = { current: mode, previous: m.current, since: performance.now() };
    }
  }, [mode]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let width = 0;
    let height = 0;
    let cols = 0;
    let rows = 0;
    let fire = new Float32Array(0);
    let frame = 0;
    let lastFire = 0;

    function resize() {
      const rect = canvas!.getBoundingClientRect();
      width = rect.width;
      height = rect.height;
      canvas!.width = Math.round(width * dpr);
      canvas!.height = Math.round(height * dpr);
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
      cols = Math.ceil(width / CELL_W);
      rows = Math.ceil(height / CELL_W);
      fire = new Float32Array(cols * rows);
    }

    function stepFire(feeding: boolean) {
      // flame burns in the middle of the panel: base just below centre, rising through it
      const radius = Math.min(width * 0.42, height * 0.4);
      const base = Math.min(rows - 1, Math.round((height / 2 + radius * 0.55) / CELL_W));
      const cx = cols / 2;
      const spread = (radius * 0.5) / CELL_W;
      for (let x = 0; x < cols; x++) {
        const k = Math.exp(-(((x - cx) / spread) ** 2) * 1.4);
        fire[base * cols + x] = feeding ? k * (0.78 + 0.22 * Math.random()) : 0;
        for (let y = base + 1; y < rows; y++) fire[y * cols + x] = 0;
      }
      if (!feeding) {
        // die out fast once loading ends so it doesn't overlap the result animation
        for (let i = 0; i < fire.length; i++) fire[i] *= 0.7;
      }
      for (let y = 0; y < base; y++) {
        for (let x = 0; x < cols; x++) {
          const src = (y + 1) * cols + x;
          const drift = Math.floor(Math.random() * 3) - 1;
          const dst = y * cols + Math.min(cols - 1, Math.max(0, x + drift));
          fire[dst] = Math.max(0, fire[src] - Math.random() * 0.06);
        }
      }
    }

    function drawFire(alpha: number) {
      const size = CELL_W - 1;
      const lightPage = document.documentElement.dataset.theme === "light";
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          const v = fire[y * cols + x];
          if (v < 0.05) continue;
          const hot = heatRamp(lightPage ? v * 0.8 : v);
          const [r, g, b] = lightPage ? onLight(hot) : hot;
          ctx!.fillStyle = `rgba(${r},${g},${b},${(alpha * Math.min(1, v * (lightPage ? 2.2 : 1.6))).toFixed(3)})`;
          ctx!.fillRect(x * CELL_W, y * CELL_W, size, size);
        }
      }
    }

    function fieldFor(m: SceneMode): ((x: number, y: number, c: SceneCtx, since: number) => Hit) | null {
      if (m === "idle") return idleScene;
      if (m === "supported") return supportedScene;
      if (m === "contradicted") return contradictedScene;
      if (m === "unverifiable") return unverifiableScene;
      return null;
    }

    function drawField(m: SceneMode, alpha: number, t: number, since: number) {
      const field = fieldFor(m);
      if (!field || alpha <= 0) return;
      const colors = colorsRef.current ?? (colorsRef.current = readPalette());
      const lightPage = document.documentElement.dataset.theme === "light";
      const cx = width / 2;
      const cy = height / 2;
      const radius = Math.min(width * 0.42, height * 0.4);
      ctx!.font = `${lightPage ? 700 : 600} ${CELL_H - 2}px "JetBrains Mono", monospace`;
      ctx!.textBaseline = "middle";
      ctx!.textAlign = "center";
      let row = 0;
      for (let py = CELL_H / 2; py < height; py += CELL_H, row++) {
        const y = (py - cy) / radius;
        if (Math.abs(y) > 1.45) continue;
        for (let px = CELL_W / 2; px < width; px += CELL_W) {
          const x = (px - cx) / radius;
          if (Math.abs(x) > 1.45) continue;
          const hit = field(x, y, { t, row, colors, heat: heatRef.current }, since);
          if (!hit) continue;
          const [v, rgb] = hit;
          // on a light page, darken and strengthen every colour so it keeps its contrast
          const [r, g, b] = lightPage ? onLight(rgb) : rgb;
          const ch = RAMP[Math.min(RAMP.length - 1, Math.max(1, Math.floor(v * RAMP.length)))];
          const strength = lightPage ? Math.min(1, 0.72 + v * 0.5) : Math.min(1, 0.3 + v * 0.9);
          ctx!.fillStyle = `rgba(${r},${g},${b},${(alpha * strength).toFixed(3)})`;
          ctx!.fillText(ch, px, py);
        }
      }
    }

    function draw(now: number) {
      const t = now / 1000;
      const m = modeRef.current;
      const fade = m.previous === null ? 1 : Math.min(1, (now - m.since) / FADE_MS);
      if (fade >= 1) m.previous = null;
      const since = now - m.since;

      ctx!.clearRect(0, 0, width, height);
      const glow = ctx!.createRadialGradient(width / 2, height / 2, 10, width / 2, height / 2, Math.min(width, height) * 0.6);
      glow.addColorStop(0, document.documentElement.dataset.theme === "light" ? "rgba(0,0,0,0.05)" : "rgba(255,255,255,0.07)");
      glow.addColorStop(1, "rgba(0,0,0,0)");
      ctx!.fillStyle = glow;
      ctx!.fillRect(0, 0, width, height);

      const fireOn = m.current === "loading" || m.previous === "loading";
      if (fireOn || fire.some((v) => v > 0.05)) {
        if (now - lastFire > 45) {
          stepFire(m.current === "loading");
          lastFire = now;
        }
        drawFire(m.current === "loading" ? fade : 1);
      }
      if (m.previous && m.previous !== "loading") drawField(m.previous, 1 - fade, t, since);
      if (m.current !== "loading") drawField(m.current, fade, t, since);

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
