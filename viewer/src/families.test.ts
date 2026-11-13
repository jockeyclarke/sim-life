import { expect, test } from "vitest";
import { drawTerritory, familyColor, familyDash } from "./families";

function recorder() {
  const fills: { style: string; alpha: number; rect: number[] }[] = [];
  const ctx = {
    fillStyle: "",
    globalAlpha: 1,
    fillRect(x: number, y: number, w: number, h: number) {
      fills.push({ style: String(this.fillStyle), alpha: this.globalAlpha, rect: [x, y, w, h] });
    },
    save() {},
    restore() {},
  };
  return { ctx: ctx as unknown as CanvasRenderingContext2D, fills };
}

test("family colours are stable, distinct and differ by scheme", () => {
  const light = [0, 1, 2, 3, 4, 5].map((k) => familyColor(k, false));
  const dark = [0, 1, 2, 3, 4, 5].map((k) => familyColor(k, true));
  expect(new Set(light).size).toBe(6);
  expect(new Set(dark).size).toBe(6);
  expect(familyColor(2, false)).toBe(light[2]);
  expect(light).not.toEqual(dark);
  expect(familyColor(-1, false)).toBe(familyColor(5, false)); // never throws on an odd index
  expect(familyColor(9, false)).toBe(familyColor(3, false));
});

test("each family's nests get a distinct ring pattern", () => {
  const patterns = [0, 1, 2, 3, 4, 5].map((k) => JSON.stringify(familyDash(k)));
  expect(new Set(patterns).size).toBe(6);
});

test("unowned cells are not filled and owned cells are shaded by strength", () => {
  const { ctx, fills } = recorder();
  const owner = new Float32Array([-1, 0, 1, -1]);
  const strength = new Float32Array([0, 1, 0.5, 0]);
  drawTerritory(ctx, owner, strength, 2, 100, false);
  expect(fills.length).toBe(2);
  expect(fills[0]!.rect).toEqual([50, 0, 50, 50]);
  expect(fills[0]!.style).toBe(familyColor(0, false));
  expect(fills[0]!.alpha).toBeGreaterThan(fills[1]!.alpha);
});

test("drawTerritory tolerates missing arrays and an empty grid", () => {
  const { ctx, fills } = recorder();
  drawTerritory(ctx, undefined, undefined, 0, 100, false);
  drawTerritory(ctx, new Float32Array(), new Float32Array(), 0, 100, true);
  drawTerritory(ctx, new Float32Array([0, 1]), new Float32Array([1]), 2, 100, false); // mismatched sizes
  expect(fills.length).toBe(0);
});

test("territory cells tile exactly, without overlapping", () => {
  const { ctx, fills } = recorder();
  const n = 3;
  drawTerritory(ctx, new Float32Array(n * n).fill(0), new Float32Array(n * n).fill(1), n, 100, false);
  const covered = fills.reduce((sum, f) => sum + f.rect[2]! * f.rect[3]!, 0);
  expect(covered).toBe(100 * 100);
});
