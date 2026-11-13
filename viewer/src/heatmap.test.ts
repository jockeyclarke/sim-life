import { expect, test } from "vitest";
import { divergingColor, divergingParts } from "./heatmap";

test("divergingColor is neutral at zero, differs by sign, and clamps", () => {
  for (const dark of [false, true]) {
    const zero = divergingParts(0, 1, dark);
    const pos = divergingParts(0.8, 1, dark);
    const neg = divergingParts(-0.8, 1, dark);
    expect(zero.saturation).toBe(0); // a neutral midpoint, never a hue
    expect(pos.hue).not.toBe(neg.hue);
    expect(pos.saturation).toBeGreaterThan(0);
    expect(pos.lightness).toBe(neg.lightness); // equal size, equal weight
    expect(divergingColor(7, 1, dark)).toBe(divergingColor(1, 1, dark));
    expect(divergingColor(-7, 1, dark)).toBe(divergingColor(-1, 1, dark));
    const half = divergingParts(0.5, 1, dark);
    const full = divergingParts(1, 1, dark);
    expect(Math.abs(full.lightness - zero.lightness)).toBeGreaterThan(Math.abs(half.lightness - zero.lightness));
  }
});

test("divergingColor handles a zero scale and non-finite values", () => {
  expect(divergingColor(0.3, 0, false)).toBe(divergingColor(0, 1, false));
  expect(divergingColor(NaN, 1, false)).toBe(divergingColor(0, 1, false));
});
