import { drawTerritory, familyColor, familyDash } from "./families";
import type { Snapshot } from "./protocol";

export type ColorMode = "family" | "colony" | "carrying" | "food" | "error" | "aggression";

export interface Theme {
  background: string;
  food: string;
  muted: string;
  accent: string;
  text: string;
  dark: boolean;
}

/** A stable colour per colony slot. The golden angle keeps neighbouring slots far apart in hue. */
export function colonyColor(slot: number): string {
  const hue = (slot * 137.508) % 360;
  return `hsl(${hue.toFixed(1)}, 70%, 55%)`;
}

/** World coordinates (unit square) to canvas pixels. Canvas y grows downward; so does ours. */
export function worldToCanvas(x: number, y: number, size: number): [number, number] {
  return [x * size, y * size];
}

/**
 * Lightness (percent) for a prediction error on a one-hue scale. Error is a
 * magnitude, so it gets a single hue running from near the background (low
 * error) to far from it (high): darker on a light page, lighter on a dark one.
 */
export function errorLightness(value: number, max: number, dark: boolean): number {
  const v = Number.isFinite(value) ? value : 0;
  const t = max > 0 ? Math.min(1, Math.max(0, v / max)) : 0;
  return dark ? 38 + 44 * t : 70 - 42 * t;
}

export function errorColor(value: number, max: number, dark: boolean): string {
  return `hsl(22, 85%, ${errorLightness(value, max, dark).toFixed(1)}%)`;
}

/** Aggression is a magnitude in [0, 1]: one hue, the same lightness ramp as error. */
export function aggressionColor(value: number, dark: boolean): string {
  return `hsl(350, 75%, ${errorLightness(value, 1, dark).toFixed(1)}%)`;
}

/** The value below which `fraction` of the entries fall. Used so one outlier doesn't wash out a colour scale. */
export function percentile(values: Float32Array, fraction: number): number {
  if (values.length === 0) return 0;
  const sorted = Float32Array.from(values).sort();
  return sorted[Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))]!;
}

/**
 * The slot of the living member nearest to a canvas point, if one is within
 * `radius` pixels. Distance is measured across the wrap-around edges.
 */
export function pickMember(snap: Snapshot, x: number, y: number, size: number, radius = 12): number | null {
  const pos = snap.arrays.member_pos;
  const slots = snap.arrays.member_slot;
  if (!pos || !slots || size <= 0) return null;
  const wrap = (d: number) => d - Math.round(d);
  let best: number | null = null;
  let bestDistance = radius;
  for (let i = 0; i < slots.length; i++) {
    const dx = wrap(pos[2 * i]! - x / size) * size;
    const dy = wrap(pos[2 * i + 1]! - y / size) * size;
    const distance = Math.hypot(dx, dy);
    if (distance <= bestDistance) {
      best = slots[i]!;
      bestDistance = distance;
    }
  }
  return best;
}

const TWO_PI = Math.PI * 2;

export function drawWorld(
  ctx: CanvasRenderingContext2D,
  snap: Snapshot,
  size: number,
  mode: ColorMode,
  theme: Theme,
  shadeTerritory = false,
  trails: CanvasImageSource | null = null,
): void {
  const a = snap.arrays;
  ctx.fillStyle = theme.background;
  ctx.fillRect(0, 0, size, size);
  if (shadeTerritory) {
    const grid = snap.shapes.territory_owner?.[0] ?? 0;
    drawTerritory(ctx, a.territory_owner, a.territory_strength, grid, size, theme.dark);
  }

  // Food: small dots.
  const food = a.food_pos ?? new Float32Array();
  ctx.fillStyle = theme.food;
  ctx.beginPath();
  for (let i = 0; i < food.length; i += 2) {
    const [x, y] = worldToCanvas(food[i]!, food[i + 1]!, size);
    ctx.moveTo(x + 1.5, y);
    ctx.arc(x, y, 1.5, 0, TWO_PI);
  }
  ctx.fill();

  // "food" mode: the fading paths carriers have taken, under everything that moves
  if (mode === "food" && trails) ctx.drawImage(trails, 0, 0, size, size);

  // Nests: a ring per colony, growing with its member count.
  const nests = a.colony_nest ?? new Float32Array();
  const colonySlots = a.colony_slot ?? new Float32Array();
  const colonyMembers = a.colony_members ?? new Float32Array();
  const colonyFamily = a.colony_family;
  ctx.lineWidth = 2;
  for (let k = 0; k < colonySlots.length; k++) {
    const [x, y] = worldToCanvas(nests[2 * k]!, nests[2 * k + 1]!, size);
    const family = colonyFamily?.[k];
    // a nest's family shows in its colour AND its ring pattern
    ctx.strokeStyle = family === undefined || mode === "colony" ? colonyColor(colonySlots[k]!) : familyColor(family, theme.dark);
    ctx.setLineDash(family === undefined ? [] : familyDash(family));
    ctx.beginPath();
    ctx.arc(x, y, 5 + Math.sqrt(colonyMembers[k]!), 0, TWO_PI);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // Members: a short mark pointing along the heading, with a dot at the head when carrying.
  const pos = a.member_pos ?? new Float32Array();
  const heading = a.member_heading ?? new Float32Array();
  const colony = a.member_colony ?? new Float32Array();
  const carrying = a.member_carrying ?? new Float32Array();
  const errors = a.member_error ?? new Float32Array();
  const memberFamily = a.member_family;
  const aggression = a.member_aggression;
  const slots = a.member_slot ?? new Float32Array();
  const errorMax = mode === "error" ? percentile(errors, 0.95) : 0;
  const selected = snap.inspect?.alive ? snap.inspect.slot : null;
  const length = 6;
  ctx.lineWidth = 1.5;
  for (let i = 0; i < heading.length; i++) {
    const [x, y] = worldToCanvas(pos[2 * i]!, pos[2 * i + 1]!, size);
    const dx = Math.cos(heading[i]!) * length;
    const dy = Math.sin(heading[i]!) * length;
    const isCarrying = carrying[i]! > 0.5;
    if (mode === "food" && !isCarrying && slots[i] !== selected) {
      // everyone not moving food recedes to a faint dot
      ctx.fillStyle = theme.muted;
      ctx.globalAlpha = 0.35;
      ctx.fillRect(x - 1, y - 1, 2, 2);
      ctx.globalAlpha = 1;
      continue;
    }
    const color =
      mode === "colony" || (mode === "family" && !memberFamily)
        ? colonyColor(colony[i]!)
        : mode === "family"
          ? familyColor(memberFamily![i]!, theme.dark)
          : mode === "aggression"
            ? aggressionColor(aggression?.[i] ?? 0, theme.dark)
            : mode === "error"
          ? errorColor(errors[i] ?? 0, errorMax, theme.dark)
          : isCarrying || mode === "food"
            ? theme.accent
            : theme.muted;
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.moveTo(x - dx / 2, y - dy / 2);
    ctx.lineTo(x + dx / 2, y + dy / 2);
    ctx.stroke();
    if (slots[i] === selected) {
      // the member being inspected
      ctx.save();
      ctx.strokeStyle = theme.text;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 10, 0, TWO_PI);
      ctx.stroke();
      ctx.restore();
    }
    if (isCarrying) {
      ctx.fillStyle = mode === "carrying" ? theme.accent : theme.food;      ctx.beginPath();
      ctx.arc(x + dx / 2, y + dy / 2, 2, 0, TWO_PI);
      ctx.fill();
    }
  }

  // strikes this tick: a small cross where each striker stands
  const strikes = a.strike_pos ?? new Float32Array();
  ctx.strokeStyle = theme.text;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < strikes.length; i += 2) {
    const [x, y] = worldToCanvas(strikes[i]!, strikes[i + 1]!, size);
    ctx.moveTo(x - 4, y - 4);
    ctx.lineTo(x + 4, y + 4);
    ctx.moveTo(x + 4, y - 4);
    ctx.lineTo(x - 4, y + 4);
  }
  ctx.stroke();
}
