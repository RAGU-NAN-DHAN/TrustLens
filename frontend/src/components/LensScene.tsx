import { useEffect, useRef } from "react";
import type { Verdict } from "../types/trustlens";

export type SceneMode = "idle" | "loading" | Verdict;

type RGB = [number, number, number];
type Hit = [number, RGB] | null;

const RAMP = " .:-=+*%#@";
const CELL_W = 9;
const CELL_H = 14;
const FADE_MS = 600;

const RADAR_AFTER_MS = 2500;
// before any new animation the eyes appear, react in that state's colour, then hand over
const EYES_MS = 1800;
// then the eyes shrink into the centre while the action's animation grows out of that point
const HANDOFF_MS = 600;

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
  // yellows become a deep gold, oranges and reds keep their hue but darken and saturate
  if (g > 170) return [184, 128, 0];
  return [Math.min(255, Math.round(r * 0.95)), Math.round(g * 0.62), Math.round(b * 0.15)];
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

interface SceneCtx {
  t: number;
  row: number;
  colors: Record<Verdict, RGB>;
  heat: boolean;
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

function contradictedScene(x: number, y: number, c: SceneCtx): Hit {
  // warning sign that beats, with rings spreading out
  const beat = Math.pow(Math.max(0, Math.sin(c.t * 3)), 6);
  const red = c.colors.contradicted;
  const d = Math.hypot(x, y);
  const r = (c.t * 0.6) % 1.3;
  const ring = Math.exp(-((d - r) ** 2) / 0.003) * (1 - r / 1.3);
  const inTri = y < 0.55 && y > -0.75 && Math.abs(x) < (y + 0.75) * 0.62;
  const bang = (Math.abs(x) < 0.07 && y > -0.4 && y < 0.18) || Math.hypot(x, y - 0.36) < 0.08;
  if (inTri) {
    if (bang) return [1, [255, 246, 230]];
    const o = c.colors.unverifiable;
    return [0.55 + beat * 0.45, [0, 1, 2].map((k) => Math.round(red[k] + (o[k] - red[k]) * beat)) as RGB];
  }
  return ring > 0.05 ? [ring, red] : null;
}

function radarScene(x: number, y: number, c: SceneCtx): Hit {
  const d = Math.hypot(x, y);
  if (d > 1.15) return null;
  const sweep = (c.t * 1.6) % (Math.PI * 2);
  let a = Math.atan2(y, x);
  if (a < 0) a += Math.PI * 2;
  const behind = (sweep - a + Math.PI * 2) % (Math.PI * 2);
  let v = behind < 1.2 ? (1 - behind / 1.2) * 0.8 : 0;
  if (Math.abs(d - 0.38) < 0.015 || Math.abs(d - 0.76) < 0.015 || Math.abs(d - 1.12) < 0.02) v = Math.max(v, 0.3);
  return v > 0.04 ? [v, behind < 0.08 ? [255, 246, 230] : c.colors.unverifiable] : null;
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
  // trust score 0..1 shown inside the verified seal
  score?: number;
  // words from the question, shown as radar blips while searching
  words?: string[];
}

export default function LensScene({ mode, palette, score, words = [] }: Props) {
  const wordsRef = useRef(words);
  wordsRef.current = words;
  const scoreRef = useRef(score);
  scoreRef.current = score;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modeRef = useRef<{ current: SceneMode; previous: SceneMode | null; since: number; previousRadar?: boolean }>({ current: mode, previous: null, since: performance.now() });
  const colorsRef = useRef<Record<Verdict, RGB> | null>(null);
  const heatRef = useRef(palette === "heat");
  heatRef.current = palette === "heat";

  useEffect(() => {
    colorsRef.current = null;
  }, [palette]);

  useEffect(() => {
    const m = modeRef.current;
    if (m.current !== mode) {
      modeRef.current = { current: mode, previous: m.current, since: performance.now(), previousRadar: m.current === "loading" && performance.now() - m.since > RADAR_AFTER_MS };
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
      // flame rises from the very bottom of the panel up to the upper middle
      const base = rows - 1;
      const cx = cols / 2;
      const spread = Math.min(cols * 0.22, (width * 0.2) / CELL_W);
      // cooling per row is set so the tips reach about 70% of the panel height
      const cool = 1.7 / (rows * 0.7);
      for (let x = 0; x < cols; x++) {
        const k = Math.exp(-(((x - cx) / spread) ** 2) * 1.4);
        fire[base * cols + x] = feeding ? k * (0.9 + 0.1 * Math.random()) : 0;
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
          fire[dst] = Math.max(0, fire[src] - Math.random() * cool);
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

    function fieldFor(m: SceneMode | "radar"): ((x: number, y: number, c: SceneCtx, since: number) => Hit) | null {
      if (m === "radar") return radarScene;
      if (m === "supported") return supportedScene;
      if (m === "contradicted") return contradictedScene;
      if (m === "unverifiable") return unverifiableScene;
      return null;
    }

    function drawField(m: SceneMode | "radar", alpha: number, t: number, since: number) {
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

    // idle: a pair of glowing eyes that follow the cursor, blink, and glance around when left alone
    const eyes = { gx: 0, gy: 0, nextBlink: performance.now() + 2500, blinkAt: -1, double: false, glance: { x: 0, y: 0, until: 0 } };
    const pointer = { x: 0, y: 0, at: -1e9 };
    function onPointer(e: PointerEvent) {
      const rect = canvas!.getBoundingClientRect();
      pointer.x = e.clientX - rect.left;
      pointer.y = e.clientY - rect.top;
      pointer.at = performance.now();
    }
    window.addEventListener("pointermove", onPointer);

    // expression weights blend smoothly so the face never jumps between moods
    const mood = { focus: 0, happy: 0, angry: 0, doubt: 0 };
    const sheet = document.createElement("canvas");
    const sg = sheet.getContext("2d")!;

    function drawEyes(state: SceneMode, alpha: number, shrink: number, now: number, excite: number) {
      if (alpha <= 0) return;
      const lightPage = document.documentElement.dataset.theme === "light";
      const tint = tintFor(state);
      const [r, g, b] = lightPage ? onLight(tint) : tint;

      // where: big in the middle on idle and during the lead-in, then small at the top so the scene plays below
      const unit = (Math.min(width, height) / 9) * (1 - shrink * 0.75);
      const cx = width / 2;
      const cy = height / 2;

      const target = {
        focus: state === "loading" ? 1 : 0,
        happy: state === "supported" ? 1 : 0,
        angry: state === "contradicted" ? 1 : 0,
        doubt: state === "unverifiable" ? 1 : 0,
      };
      (Object.keys(mood) as (keyof typeof mood)[]).forEach((k) => (mood[k] += (target[k] - mood[k]) * 0.1));

      // gaze: scan left-right while searching, look up and away when unsure, otherwise follow the cursor
      let tx: number;
      let ty: number;
      if (state === "loading") {
        tx = Math.sin(now / 260) * unit * 0.5;
        ty = Math.sin(now / 1300) * unit * 0.1;
      } else if (state === "unverifiable") {
        tx = -unit * 0.4;
        ty = -unit * 0.3;
      } else if (now - pointer.at < 4000) {
        const dx = pointer.x - cx;
        const dy = pointer.y - cy;
        const d = Math.hypot(dx, dy) || 1;
        const reach = Math.min(1, d / (unit * 4));
        tx = (dx / d) * unit * 0.45 * reach;
        ty = (dy / d) * unit * 0.35 * reach;
      } else {
        if (now > eyes.glance.until) eyes.glance = { x: (Math.random() - 0.5) * 0.7, y: (Math.random() - 0.5) * 0.3, until: now + 1000 + Math.random() * 2000 };
        tx = eyes.glance.x * unit;
        ty = eyes.glance.y * unit;
      }
      eyes.gx += (tx - eyes.gx) * 0.15;
      eyes.gy += (ty - eyes.gy) * 0.15;

      let blink = 0;
      if (eyes.blinkAt < 0 && now > eyes.nextBlink && mood.happy < 0.5 && mood.angry < 0.5) {
        eyes.blinkAt = now;
        eyes.double = Math.random() < 0.25;
      }
      if (eyes.blinkAt >= 0 && now >= eyes.blinkAt) {
        const p = (now - eyes.blinkAt) / 160;
        if (p < 1) blink = Math.sin(p * Math.PI);
        else if (eyes.double) {
          eyes.double = false;
          eyes.blinkAt = now + 70;
        } else {
          eyes.blinkAt = -1;
          eyes.nextBlink = now + 2500 + Math.random() * 3500;
        }
      }

      // the eyes are drawn on their own sheet so the lids can be cut out without erasing the scene behind
      if (sheet.width !== canvas!.width || sheet.height !== canvas!.height) {
        sheet.width = canvas!.width;
        sheet.height = canvas!.height;
      }
      sg.setTransform(dpr, 0, 0, dpr, 0, 0);
      sg.clearRect(0, 0, width, height);
      const ew = unit * 0.75 * (1 + excite * 0.12);
      const eh = unit * 1.1 * (1 + excite * 0.18) * (1 - blink) + 1.5;
      const bounce = mood.happy * Math.abs(Math.sin(now / 220)) * unit * 0.12;
      const shake = mood.angry * Math.sin(now / 35) * unit * 0.05 * Math.max(0, Math.sin(now / 600));
      const breathe = Math.sin(now / 900) * unit * 0.04;

      for (const side of [-1, 1]) {
        const ex = cx + side * unit * 1.55 + eyes.gx * 0.6 + shake;
        const ey = cy + eyes.gy * 0.5 + breathe - bounce - (side > 0 ? mood.doubt * unit * 0.25 : 0);
        sg.globalCompositeOperation = "source-over";
        sg.fillStyle = `rgb(${r},${g},${b})`;
        sg.beginPath();
        sg.ellipse(ex, ey, ew, eh, 0, 0, Math.PI * 2);
        sg.fill();
        sg.fillStyle = lightPage ? "rgba(255,255,255,0.7)" : "rgba(255,248,230,0.85)";
        sg.beginPath();
        sg.ellipse(ex - ew * 0.3 + eyes.gx * 0.3, ey - eh * 0.35 + eyes.gy * 0.25, ew * 0.22, Math.max(1, eh * 0.16), 0, 0, Math.PI * 2);
        sg.fill();

        sg.globalCompositeOperation = "destination-out";
        sg.fillStyle = "#000";
        // focused (loading) and doubtful (left eye): a flat upper lid comes down
        const lid = Math.max(mood.focus * 0.4, side < 0 ? mood.doubt * 0.55 : 0);
        if (lid > 0.01) sg.fillRect(ex - ew * 1.5, ey - eh * 1.5, ew * 3, eh * 1.5 * lid + eh * 0.5 * lid);
        // angry: lids slant down toward the nose
        if (mood.angry > 0.01) {
          const inner = ex - side * ew * 1.2;
          const outer = ex + side * ew * 1.2;
          sg.beginPath();
          sg.moveTo(outer, ey - eh * 1.6);
          sg.lineTo(inner, ey - eh * 1.6);
          sg.lineTo(inner, ey - eh * 1.6 + eh * 1.75 * mood.angry);
          sg.lineTo(outer, ey - eh * 1.6 + eh * 0.55 * mood.angry);
          sg.closePath();
          sg.fill();
        }
        // happy: the lower lid pushes up into a ^ ^ smile
        if (false) {
          sg.beginPath();
          sg.ellipse(ex, ey + eh * (1.9 - mood.happy * 1.05), ew * 1.6, eh * 1.15, 0, 0, Math.PI * 2);
          sg.fill();
        }
      }
      sg.globalCompositeOperation = "source-over";
      ctx!.save();
      ctx!.setTransform(1, 0, 0, 1, 0, 0);
      ctx!.globalAlpha = alpha;
      ctx!.shadowColor = `rgba(${r},${g},${b},0.9)`;
      ctx!.shadowBlur = (lightPage ? 6 : unit * 0.5) * dpr;
      ctx!.drawImage(sheet, 0, 0);
      ctx!.restore();
    }

    const sparks: { x: number; y: number; vx: number; vy: number; life: number }[] = [];
    function drawSparkles(now: number, since: number) {
      // a burst when the verdict lands, then a gentle trickle
      const rate = since < 1500 ? 0.9 : 0.15;
      if (Math.random() < rate) {
        const a = Math.random() * Math.PI * 2;
        const r0 = Math.min(width, height) * (0.25 + Math.random() * 0.2);
        sparks.push({ x: width / 2 + Math.cos(a) * r0, y: height / 2 + Math.sin(a) * r0 * 0.7, vx: Math.cos(a) * 0.4, vy: -0.5 - Math.random() * 0.6, life: 1 });
      }
      const lightPage = document.documentElement.dataset.theme === "light";
      const [r, g, b] = lightPage ? onLight(tintFor("supported")) : tintFor("supported");
      for (let i = sparks.length - 1; i >= 0; i--) {
        const p = sparks[i];
        p.x += p.vx;
        p.y += p.vy;
        p.life -= 0.012;
        if (p.life <= 0) {
          sparks.splice(i, 1);
          continue;
        }
        const size = 3 + 3 * Math.sin(p.life * Math.PI) + Math.sin(now / 90 + i) * 0.8;
        ctx!.save();
        ctx!.globalAlpha = p.life;
        ctx!.fillStyle = `rgb(${r},${g},${b})`;
        ctx!.translate(p.x, p.y);
        ctx!.rotate(Math.PI / 4);
        ctx!.fillRect(-size / 2, -size / 2, size, size);
        ctx!.restore();
      }
    }

    // verified: a seal stamps down after the eyes and keeps turning slowly, trust score in the centre
    function drawSeal(alpha: number, now: number, after: number) {
      if (alpha <= 0) return;
      const lightPage = document.documentElement.dataset.theme === "light";
      const [r, g, b] = lightPage ? onLight(tintFor("supported")) : tintFor("supported");
      const col = (a: number) => `rgba(${r},${g},${b},${a})`;
      const stamp = Math.min(1, after / 380);
      // lands from 1.6x down to 1x with a small overshoot, like a rubber stamp
      const scale = stamp < 1 ? 1.6 - 0.68 * (1 - Math.pow(1 - stamp, 3)) : 1 - 0.08 * Math.exp(-(after - 380) / 120) * Math.cos((after - 380) / 40);
      const R = Math.min(width, height) * 0.34;
      ctx!.save();
      ctx!.globalAlpha = alpha * Math.min(1, after / 200);
      ctx!.translate(width / 2, height / 2);
      ctx!.scale(scale, scale);

      // impact ring right when it lands
      if (after > 380 && after < 1100) {
        const p = (after - 380) / 720;
        ctx!.strokeStyle = col(0.5 * (1 - p));
        ctx!.lineWidth = 2;
        ctx!.beginPath();
        ctx!.arc(0, 0, R * (1.05 + p * 0.5), 0, Math.PI * 2);
        ctx!.stroke();
      }

      ctx!.strokeStyle = col(1);
      ctx!.lineWidth = 2.5;
      ctx!.beginPath();
      ctx!.arc(0, 0, R, 0, Math.PI * 2);
      ctx!.stroke();
      ctx!.lineWidth = 2;
      ctx!.beginPath();
      ctx!.arc(0, 0, R * 0.7, 0, Math.PI * 2);
      ctx!.stroke();

      for (let i = 0; i < 48; i++) {
        const a = (i / 48) * Math.PI * 2;
        ctx!.fillStyle = col(0.35 + 0.65 * Math.max(0, Math.sin(now / 500 + i)));
        ctx!.fillRect(Math.cos(a) * R * 1.12 - 1.5, Math.sin(a) * R * 1.12 - 1.5, 3, 3);
      }

      const text = " VERIFIED · EVIDENCE BACKED · VERIFIED · EVIDENCE BACKED ·";
      const fontPx = Math.max(10, Math.round(R * 0.1));
      ctx!.save();
      ctx!.rotate(now / 2500);
      ctx!.font = `700 ${fontPx}px "JetBrains Mono", monospace`;
      ctx!.textAlign = "center";
      ctx!.textBaseline = "middle";
      ctx!.fillStyle = col(1);
      for (let i = 0; i < text.length; i++) {
        ctx!.save();
        ctx!.rotate((i / text.length) * Math.PI * 2);
        ctx!.fillText(text[i], 0, -R * 0.85);
        ctx!.restore();
      }
      ctx!.restore();

      const sc = scoreRef.current;
      ctx!.fillStyle = col(1);
      ctx!.textAlign = "center";
      ctx!.textBaseline = "middle";
      ctx!.font = `700 ${Math.round(R * 0.3)}px "JetBrains Mono", monospace`;
      ctx!.fillText(sc === undefined ? "OK" : `${Math.round(sc * 100)}%`, 0, -R * 0.06);
      ctx!.font = `600 ${Math.round(R * 0.09)}px "JetBrains Mono", monospace`;
      ctx!.fillText("TRUST", 0, R * 0.2);
      ctx!.restore();
    }

    // the question's words sit around the radar and light up as the sweep passes them
    function drawBlips(alpha: number, t: number) {
      const list = wordsRef.current.slice(0, 8);
      if (!list.length || alpha <= 0) return;
      const lightPage = document.documentElement.dataset.theme === "light";
      const radius = Math.min(width * 0.42, height * 0.4);
      const sweep = (t * 1.6) % (Math.PI * 2);
      ctx!.save();
      ctx!.textAlign = "center";
      ctx!.textBaseline = "middle";
      list.forEach((word, i) => {
        // spread evenly round the circle, at varying distances, same spot every frame
        const a = (i / list.length) * Math.PI * 2 + 0.4;
        const d = radius * (0.35 + 0.5 * ((i * 0.37) % 1));
        const x = width / 2 + Math.cos(a) * d;
        const y = height / 2 + Math.sin(a) * d;
        const angle = (a + Math.PI * 2) % (Math.PI * 2);
        const behind = (sweep - angle + Math.PI * 2) % (Math.PI * 2);
        const glow = behind < 2.2 ? 1 - behind / 2.2 : 0;
        const [r, g, b] = lightPage ? onLight([246, 180, 80]) : glow > 0.85 ? [255, 246, 230] : [246, 180, 80];
        ctx!.globalAlpha = alpha * (0.18 + 0.82 * glow);
        ctx!.fillStyle = `rgb(${r},${g},${b})`;
        ctx!.beginPath();
        ctx!.arc(x, y - 12, 2.5 + 2 * glow, 0, Math.PI * 2);
        ctx!.fill();
        ctx!.font = `${glow > 0.5 ? 700 : 500} ${Math.max(11, Math.round(radius * 0.07))}px "JetBrains Mono", monospace`;
        ctx!.fillText(word.toUpperCase(), x, y + 4);
      });
      ctx!.restore();
    }

    function tintFor(m: SceneMode): RGB {
      const colors = colorsRef.current ?? (colorsRef.current = readPalette());
      if (m === "supported") return colors.supported;
      if (m === "contradicted") return colors.contradicted;
      if (m === "unverifiable") return colors.unverifiable;
      return [246, 180, 80];
    }

    function draw(now: number) {
      const t = now / 1000;
      const m = modeRef.current;
      const since = now - m.since;
      const clamp = (v: number) => Math.min(1, Math.max(0, v));
      const lead = m.current === "idle" ? 0 : EYES_MS;
      const out = m.previous === null ? 1 : clamp(since / FADE_MS); // old scene fades out
      const fade = clamp((since - lead) / HANDOFF_MS); // new scene appears as the eyes hand over
      const after = Math.max(0, since - lead); // time since the new scene itself started
      if (out >= 1 && fade >= 1) m.previous = null;

      ctx!.clearRect(0, 0, width, height);
      const glow = ctx!.createRadialGradient(width / 2, height / 2, 10, width / 2, height / 2, Math.min(width, height) * 0.6);
      glow.addColorStop(0, document.documentElement.dataset.theme === "light" ? "rgba(0,0,0,0.05)" : "rgba(255,255,255,0.07)");
      glow.addColorStop(1, "rgba(0,0,0,0)");
      ctx!.fillStyle = glow;
      ctx!.fillRect(0, 0, width, height);

      const grow = 1;
      ctx!.save();
      ctx!.translate(width / 2, height / 2);
      ctx!.scale(grow, grow);
      ctx!.translate(-width / 2, -height / 2);

      // loading: the flame burns while the answer is written, then flattens into a radar sweep
      const loading = m.current === "loading";
      const morph = loading ? clamp((after - RADAR_AFTER_MS) / 900) : 0;
      if (loading || fire.some((v) => v > 0.05)) {
        if (now - lastFire > 45) {
          stepFire(loading && since >= lead);
          lastFire = now;
        }
        if (morph < 1) {
          ctx!.save();
          ctx!.translate(width / 2, height / 2);
          ctx!.scale(1 + morph * 0.8, 1 - morph * 0.9);
          ctx!.translate(-width / 2, -height / 2);
          drawFire((loading ? fade : 1 - out) * (1 - morph));
          ctx!.restore();
        }
      }
      if (morph > 0) {
        drawField("radar", morph * fade, t, after);
        drawBlips(morph * fade, t);
      }
      else if (m.previous === "loading" && m.previousRadar) drawField("radar", 1 - out, t, since);

      if (m.previous && m.previous !== "loading" && m.previous !== "idle" && m.previous !== "supported") drawField(m.previous, 1 - out, t, since);
      if (!loading && m.current !== "idle" && m.current !== "supported") drawField(m.current, fade, t, after);

      ctx!.restore();

      if (m.current === "supported") {
        drawSeal(fade, now, after);
        if (after > 380) drawSparkles(now, after - 380);
      } else if (m.previous === "supported") drawSeal(1 - out, now, 1e6);

      // every action starts with the eyes reacting in that state's colour and expression,
      // then they shrink into the centre and the action's own animation takes over
      if (m.current === "idle") drawEyes("idle", m.previous === null ? 1 : out, 0, now, 0);
      else if (since < lead + HANDOFF_MS) {
        const hand = clamp((since - lead) / HANDOFF_MS);
        // fade in quickly, hold briefly, fade out into the action's animation
        const appear = clamp(since / 180);
        drawEyes(m.current, Math.min(appear, 1 - hand), 0, now, since < lead ? Math.sin(clamp(since / lead) * Math.PI) * 0.6 : 0);
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
      window.removeEventListener("pointermove", onPointer);
    };
  }, []);

  return <canvas ref={canvasRef} className="lens-canvas" aria-hidden="true" />;
}
