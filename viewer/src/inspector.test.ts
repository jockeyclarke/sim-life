import { expect, test } from "vitest";
import { ACTION_NAMES, ErrorTrace, INPUT_NAMES, PREDICTED_NAMES, logScale, weightChange } from "./inspector";
import { decodeSnapshot } from "./protocol";
import { readFixture } from "./testFixture";

test("ErrorTrace keeps order, caps length, and resets on a new slot", () => {
  const trace = new ErrorTrace(5);
  for (let t = 1; t <= 8; t++) trace.push(3, t, t / 10);
  expect(trace.points()).toEqual([[4, 0.4], [5, 0.5], [6, 0.6], [7, 0.7], [8, 0.8]]);
  trace.push(9, 20, 0.1);
  expect(trace.points()).toEqual([[20, 0.1]]);
});

test("ErrorTrace ignores a repeated or earlier tick", () => {
  const trace = new ErrorTrace();
  trace.push(3, 10, 0.5);
  trace.push(3, 10, 0.7); // a paused engine re-sends the same snapshot
  trace.push(3, 9, 0.9);
  expect(trace.points()).toEqual([[10, 0.5]]);
});

test("ErrorTrace skips values that cannot be charted", () => {
  const trace = new ErrorTrace();
  trace.push(3, 1, 0); // a member's first tick has no error yet
  trace.push(3, 2, NaN);
  trace.push(3, 3, 0.2);
  expect(trace.points()).toEqual([[3, 0.2]]);
});

test("weightChange subtracts elementwise and survives a shape mismatch", () => {
  expect(weightChange([[1, 2], [3, 4]], [[0.5, 2], [1, 1]])).toEqual([[0.5, 0], [2, 3]]);
  expect(weightChange([[1, 2], [3, 4]], [[1, 2, 3]])).toEqual([[0, 0], [0, 0]]);
  expect(weightChange([[1, 2]], [])).toEqual([[0, 0]]);
});

test("logScale gives power-of-ten bounds around the data", () => {
  const s = logScale([0.014, 1.4, 0.3]);
  expect(s.lo).toBe(0.01);
  expect(s.hi).toBe(10);
  expect(s.ticks).toEqual([0.01, 0.1, 1, 10]);
  expect(s.y(0.01)).toBe(0);
  expect(s.y(10)).toBe(1);
  expect(s.y(0.1)).toBeCloseTo(1 / 3);
});

test("logScale handles no data and a single value", () => {
  expect(logScale([]).hi).toBeGreaterThan(logScale([]).lo);
  const one = logScale([0.5]);
  expect(one.hi).toBeGreaterThan(one.lo);
});

test("name lists match the protocol's lengths", () => {
  const snap = decodeSnapshot(readFixture("snapshot.bin"));
  expect(snap.inspect?.alive).toBe(true);
  if (!snap.inspect?.alive) return;
  expect(INPUT_NAMES.length).toBe(snap.inspect.inputs.length);
  expect(ACTION_NAMES.length).toBe(snap.inspect.actions.length);
  expect(PREDICTED_NAMES.length).toBe(snap.inspect.predicted.length);
  expect(PREDICTED_NAMES).toEqual(INPUT_NAMES.slice(0, 6));
});

import { selectionChanged } from "./inspector";

test("ErrorTrace.clear empties it", () => {
  const trace = new ErrorTrace();
  trace.push(1, 10, 0.5);
  trace.clear();
  expect(trace.points()).toEqual([]);
  trace.push(1, 5, 0.3); // an earlier tick is accepted after a clear, as after an engine restart
  expect(trace.points()).toEqual([[5, 0.3]]);
});

test("selectionChanged follows the server's selection, not the slot", () => {
  const alive = (slot: number, selection: number) => ({ slot, selection, alive: true }) as never;
  const dead = (slot: number, selection: number) => ({ slot, selection, alive: false }) as never;
  expect(selectionChanged(null, alive(7, 1))).toBe(true);
  expect(selectionChanged(1, alive(7, 1))).toBe(false);
  expect(selectionChanged(1, dead(7, 1))).toBe(false); // the same member, now dead: keep its trace
  expect(selectionChanged(1, alive(7, 2))).toBe(true); // same slot, a different member
  expect(selectionChanged(1, null)).toBe(true); // deselected
  expect(selectionChanged(null, null)).toBe(false);
});

test("the fixture's inspection carries a selection identity", () => {
  const snap = decodeSnapshot(readFixture("snapshot.bin"));
  expect(snap.inspect?.selection).toBe(1);
});
