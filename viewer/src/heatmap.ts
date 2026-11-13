// A diverging colour scale for signed values (weights, activations) and a cell-grid drawer.

export interface ColorParts {
  hue: number;
  saturation: number; // percent
  lightness: number; // percent
}

const NEGATIVE_HUE = 215; // blue
const POSITIVE_HUE = 22; // orange

/**
 * Two hues with a neutral grey midpoint at zero: negative values go blue,
 * positive go orange, and the further from zero the stronger the colour.
 * Values beyond ±scale are clamped.
 */
export function divergingParts(value: number, scale: number, dark: boolean): ColorParts {
  const v = Number.isFinite(value) ? value : 0;
  const t = scale > 0 ? Math.min(1, Math.abs(v) / scale) : 0;
  const neutral = dark ? 24 : 92;
  const strong = dark ? 62 : 48;
  return {
    hue: v < 0 ? NEGATIVE_HUE : POSITIVE_HUE,
    saturation: Math.round(85 * t),
    lightness: Number((neutral + (strong - neutral) * t).toFixed(1)),
  };
}

export function divergingColor(value: number, scale: number, dark: boolean): string {
  const c = divergingParts(value, scale, dark);
  return `hsl(${c.hue}, ${c.saturation}%, ${c.lightness}%)`;
}

/** Draw rows of cells with a 1px gap between them. */
export function drawHeatmap(
  ctx: CanvasRenderingContext2D,
  rows: number[][],
  x: number,
  y: number,
  cell: number,
  scale: number,
  dark: boolean,
): void {
  rows.forEach((row, r) => {
    row.forEach((value, c) => {
      ctx.fillStyle = divergingColor(value, scale, dark);
      ctx.fillRect(x + c * cell, y + r * cell, cell - 1, cell - 1);
    });
  });
}
