import { expect, test } from "vitest";
import { colonyColor, worldToCanvas } from "./worldView";

test("colony colours are stable and distinct for neighbouring slots", () => {
  expect(colonyColor(3)).toBe(colonyColor(3));
  expect(colonyColor(3)).not.toBe(colonyColor(4));
  const all = new Set(Array.from({ length: 64 }, (_, i) => colonyColor(i)));
  expect(all.size).toBe(64);
});

test("worldToCanvas maps the unit square onto the canvas", () => {
  expect(worldToCanvas(0, 0, 800)).toEqual([0, 0]);
  expect(worldToCanvas(0.5, 0.25, 800)).toEqual([400, 200]);
});
