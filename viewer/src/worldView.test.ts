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

import { errorLightness, pickMember } from "./worldView";
import type { Snapshot } from "./protocol";

function world(members: [number, number, number][]): Snapshot {
  return {
    tick: 0,
    metrics: {} as Snapshot["metrics"],
    status: {} as Snapshot["status"],
    inspect: null,
    arrays: {
      member_pos: new Float32Array(members.flatMap(([x, y]) => [x, y])),
      member_slot: new Float32Array(members.map(([, , slot]) => slot)),
    },
    shapes: {},
  };
}

test("pickMember returns the nearest member within reach", () => {
  const snap = world([[0.5, 0.5, 7], [0.51, 0.5, 9], [0.9, 0.9, 11]]);
  expect(pickMember(snap, 400, 400, 800)).toBe(7);
  expect(pickMember(snap, 407, 400, 800)).toBe(9);
  expect(pickMember(snap, 722, 715, 800)).toBe(11); // 2px + 5px away
});

test("pickMember returns null on empty space and on an empty world", () => {
  const snap = world([[0.5, 0.5, 7]]);
  expect(pickMember(snap, 100, 100, 800)).toBeNull();
  expect(pickMember(snap, 413, 400, 800)).toBeNull(); // 13px: just out of reach
  expect(pickMember(world([]), 400, 400, 800)).toBeNull();
  expect(pickMember({ ...snap, arrays: {} }, 400, 400, 800)).toBeNull();
});

test("pickMember finds a member across the wrap-around edge", () => {
  const snap = world([[0.999, 0.5, 4]]);
  expect(pickMember(snap, 2, 400, 800)).toBe(4);
});

test("errorLightness is monotonic and clamps above max", () => {
  for (const dark of [false, true]) {
    const steps = [0, 0.25, 0.5, 0.75, 1].map((v) => errorLightness(v, 1, dark));
    const rising = steps.every((v, i) => i === 0 || v > steps[i - 1]!);
    const falling = steps.every((v, i) => i === 0 || v < steps[i - 1]!);
    expect(dark ? rising : falling).toBe(true); // more error = further from the background
    expect(errorLightness(5, 1, dark)).toBe(errorLightness(1, 1, dark));
    expect(errorLightness(-1, 1, dark)).toBe(errorLightness(0, 1, dark));
  }
});

test("errorLightness handles max of zero and non-finite values", () => {
  expect(Number.isFinite(errorLightness(0, 0, false))).toBe(true);
  expect(errorLightness(NaN, 1, false)).toBe(errorLightness(0, 1, false));
});

import { homeNest } from "./worldView";

test("homeNest finds the selected member's colony among the nests, or null", () => {
  const base = world([[0.5, 0.5, 7]]);
  const snap: Snapshot = {
    ...base,
    arrays: { ...base.arrays, colony_slot: new Float32Array([4, 9, 2]) },
    inspect: { slot: 7, alive: true, colony: 9 } as Snapshot["inspect"],
  };
  expect(homeNest(snap)).toBe(1);
  expect(homeNest({ ...snap, inspect: { slot: 7, alive: true, colony: 5 } as Snapshot["inspect"] })).toBeNull();
  expect(homeNest({ ...snap, inspect: null })).toBeNull();
  expect(homeNest({ ...snap, inspect: { slot: 7, alive: false } as Snapshot["inspect"] })).toBeNull();
});
