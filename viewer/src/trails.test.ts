import { expect, test } from "vitest";
import type { Snapshot } from "./protocol";
import { TrailTracker } from "./trails";

// [x, y, slot, carrying]
function snap(tick: number, members: [number, number, number, number][]): Snapshot {
  return {
    tick,
    metrics: {} as Snapshot["metrics"],
    status: {} as Snapshot["status"],
    inspect: null,
    arrays: {
      member_pos: new Float32Array(members.flatMap(([x, y]) => [x, y])),
      member_slot: new Float32Array(members.map((m) => m[2])),
      member_carrying: new Float32Array(members.map((m) => m[3])),
    },
    shapes: {},
  };
}

test("a carrier leaves a segment from where it was to where it is", () => {
  const t = new TrailTracker();
  expect(t.update(snap(1, [[0.1, 0.1, 3, 1]])).segments).toEqual([]);
  const { segments, reset } = t.update(snap(2, [[0.12, 0.1, 3, 1]]));
  expect(reset).toBe(false);
  expect(segments).toHaveLength(1);
  const [x0, y0, x1, y1] = segments[0]!;
  expect([x0, y0, x1, y1].map((v) => +v.toFixed(4))).toEqual([0.1, 0.1, 0.12, 0.1]);
});

test("members not carrying leave no trail, and a trail starts afresh after a drop", () => {
  const t = new TrailTracker();
  t.update(snap(1, [[0.1, 0.1, 3, 1], [0.5, 0.5, 4, 0]]));
  expect(t.update(snap(2, [[0.12, 0.1, 3, 0], [0.52, 0.5, 4, 0]])).segments).toEqual([]);
  expect(t.update(snap(3, [[0.14, 0.1, 3, 1]])).segments).toEqual([]); // picked up again: no line back to the drop
  expect(t.update(snap(4, [[0.16, 0.1, 3, 1]])).segments).toHaveLength(1);
});

test("a step across the wrap-around edge draws no line across the world", () => {
  const t = new TrailTracker();
  t.update(snap(1, [[0.995, 0.5, 3, 1]]));
  expect(t.update(snap(2, [[0.005, 0.5, 3, 1]])).segments).toEqual([]);
});

test("time going backwards (a new run, a reseed, a resume) clears the trails", () => {
  const t = new TrailTracker();
  t.update(snap(100, [[0.1, 0.1, 3, 1]]));
  const later = t.update(snap(5, [[0.11, 0.1, 3, 1]]));
  expect(later.reset).toBe(true);
  expect(later.segments).toEqual([]);
});

test("the same snapshot twice adds nothing", () => {
  const t = new TrailTracker();
  const s = snap(1, [[0.1, 0.1, 3, 1]]);
  t.update(s);
  t.update(snap(2, [[0.12, 0.1, 3, 1]]));
  const again = snap(2, [[0.12, 0.1, 3, 1]]);
  expect(t.update(again).segments).toEqual([]);
});
