import type { Snapshot } from "./protocol";

export type ColorMode = "colony" | "carrying";

export interface Theme {
  background: string;
  food: string;
  muted: string;
  accent: string;
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

const TWO_PI = Math.PI * 2;

export function drawWorld(
  ctx: CanvasRenderingContext2D,
  snap: Snapshot,
  size: number,
  mode: ColorMode,
  theme: Theme,
): void {
  const a = snap.arrays;
  ctx.fillStyle = theme.background;
  ctx.fillRect(0, 0, size, size);

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

  // Nests: a ring per colony, growing with its member count.
  const nests = a.colony_nest ?? new Float32Array();
  const colonySlots = a.colony_slot ?? new Float32Array();
  const colonyMembers = a.colony_members ?? new Float32Array();
  ctx.lineWidth = 2;
  for (let k = 0; k < colonySlots.length; k++) {
    const [x, y] = worldToCanvas(nests[2 * k]!, nests[2 * k + 1]!, size);
    ctx.strokeStyle = colonyColor(colonySlots[k]!);
    ctx.beginPath();
    ctx.arc(x, y, 5 + Math.sqrt(colonyMembers[k]!), 0, TWO_PI);
    ctx.stroke();
  }

  // Members: a short mark pointing along the heading, with a dot at the head when carrying.
  const pos = a.member_pos ?? new Float32Array();
  const heading = a.member_heading ?? new Float32Array();
  const colony = a.member_colony ?? new Float32Array();
  const carrying = a.member_carrying ?? new Float32Array();
  const length = 6;
  ctx.lineWidth = 1.5;
  for (let i = 0; i < heading.length; i++) {
    const [x, y] = worldToCanvas(pos[2 * i]!, pos[2 * i + 1]!, size);
    const dx = Math.cos(heading[i]!) * length;
    const dy = Math.sin(heading[i]!) * length;
    const isCarrying = carrying[i]! > 0.5;
    const color = mode === "colony" ? colonyColor(colony[i]!) : isCarrying ? theme.accent : theme.muted;
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.moveTo(x - dx / 2, y - dy / 2);
    ctx.lineTo(x + dx / 2, y + dy / 2);
    ctx.stroke();
    if (isCarrying) {
      ctx.fillStyle = mode === "colony" ? theme.food : theme.accent;
      ctx.beginPath();
      ctx.arc(x + dx / 2, y + dy / 2, 2, 0, TWO_PI);
      ctx.fill();
    }
  }
}
