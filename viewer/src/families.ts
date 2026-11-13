// Family colours and territory shading.
//
// Six hues cannot all be told apart under colour-blindness, so colour is not
// the only cue: each family's nests also get their own ring pattern, and the
// inspector and charts name families by number.

const LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"];
const DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#2f9e2f"];

// Ring patterns for nests, one per family (canvas setLineDash arrays).
const DASHES: number[][] = [[], [6, 3], [2, 3], [8, 3, 2, 3], [1, 2], [4, 4]];

function index(k: number): number {
  const n = LIGHT.length;
  return ((Math.trunc(k) % n) + n) % n;
}

export function familyColor(k: number, dark: boolean): string {
  return (dark ? DARK : LIGHT)[index(k)]!;
}

export function familyDash(k: number): number[] {
  return DASHES[index(k)]!;
}

/**
 * Shade each owned cell in its family's colour, more strongly where the family
 * holds more of the scent. Unowned cells (owner -1) are left blank.
 */
export function drawTerritory(
  ctx: CanvasRenderingContext2D,
  owner: Float32Array | undefined,
  strength: Float32Array | undefined,
  grid: number,
  size: number,
  dark: boolean,
): void {
  if (!owner || !strength || grid <= 0 || owner.length !== grid * grid || strength.length !== owner.length) return;
  // whole-pixel edges, each cell ending where the next begins: overlapping
  // translucent cells would draw a visible lattice
  const edge = (i: number) => Math.round((i * size) / grid);
  ctx.save();
  for (let i = 0; i < owner.length; i++) {
    const k = owner[i]!;
    if (k < 0) continue;
    ctx.globalAlpha = 0.08 + 0.22 * Math.min(1, Math.max(0, strength[i]!));
    ctx.fillStyle = familyColor(k, dark);
    const x = i % grid;
    const y = Math.floor(i / grid);
    ctx.fillRect(edge(x), edge(y), edge(x + 1) - edge(x), edge(y + 1) - edge(y));
  }
  ctx.restore();
}
