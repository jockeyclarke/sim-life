import { expect, test } from "vitest";
import { nearestMeasured, niceScale, reseedTicks, seriesOf } from "./populationView";
import type { HistoryRecord } from "./protocol";

test("seriesOf sorts by tick and skips non-finite values", () => {
  const recs = [{ tick: 200, members: 5 }, { tick: 100, members: 3 }, { tick: 300, members: NaN }] as HistoryRecord[];
  expect(seriesOf(recs, "members")).toEqual([[100, 3], [200, 5]]);
});

test("seriesOf skips records where the field is missing", () => {
  const recs = [{ tick: 100 }, { tick: 200, delivery_rate: 0.5 }] as HistoryRecord[];
  expect(seriesOf(recs, "delivery_rate")).toEqual([[200, 0.5]]);
});

test("niceScale gives round bounds that contain the data", () => {
  const s = niceScale(3, 97, 5);
  expect(s.lo).toBeLessThanOrEqual(3);
  expect(s.hi).toBeGreaterThanOrEqual(97);
  expect(s.ticks.every((t) => Number.isInteger(t / 10))).toBe(true);
  expect(s.ticks.length).toBeGreaterThanOrEqual(3);
  expect(s.ticks.length).toBeLessThanOrEqual(8);
});

test("niceScale handles small fractions without float noise", () => {
  const s = niceScale(0, 0.0031, 4);
  expect(s.hi).toBeGreaterThanOrEqual(0.0031);
  for (const t of s.ticks) expect(String(t).length).toBeLessThan(8);
});

test("niceScale handles a flat series", () => {
  const s = niceScale(4, 4, 5);
  expect(s.hi).toBeGreaterThan(s.lo);
});

test("niceScale handles an empty range of zeros", () => {
  const s = niceScale(0, 0, 5);
  expect(s.hi).toBeGreaterThan(s.lo);
});

test("reseedTicks collects reseed events only", () => {
  const recs = [
    { tick: 100, events: [{ event: "reseed", tick: 70 }] },
    { tick: 200 },
    { tick: 300, events: [{ event: "extinct", tick: 300 }, { event: "reseed", tick: 250 }] },
  ] as HistoryRecord[];
  expect(reseedTicks(recs)).toEqual([70, 250]);
});

test("event-only history lines mark reseeds but add no data points", () => {
  const recs = [
    { tick: 100, members: 5 },
    { tick: 142, events: [{ event: "reseed", tick: 142 }] },
    { tick: 200, members: 7 },
  ] as HistoryRecord[];
  expect(seriesOf(recs, "members")).toEqual([[100, 5], [200, 7]]);
  expect(reseedTicks(recs)).toEqual([142]);
  expect(nearestMeasured(recs, 150)?.tick).toBe(100);
  expect(nearestMeasured([recs[1]!], 150)).toBeNull();
});
