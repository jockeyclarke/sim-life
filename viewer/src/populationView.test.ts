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

import { eventTicks } from "./populationView";

test("eventTicks collects events by name", () => {
  const recs = [
    { tick: 30, events: [{ event: "switch", tick: 30, name: "learning", value: false }] },
    { tick: 100, members: 4 },
    { tick: 142, events: [{ event: "reseed", tick: 142 }] },
    { tick: 150, events: [{ event: "switch", tick: 150 }, { event: "extinct", tick: 150 }] },
  ] as HistoryRecord[];
  expect(eventTicks(recs, "switch")).toEqual([30, 150]);
  expect(eventTicks(recs, "reseed")).toEqual([142]);
  expect(reseedTicks(recs)).toEqual([142]);
  expect(eventTicks(recs, "nope")).toEqual([]);
});

test("a stage 1 history draws no points on the new charts", () => {
  const old = [{ tick: 100, members: 5, colonies: 1, energy: 3, delivered: 0, max_generation: 0, delivery_rate: 0 }] as HistoryRecord[];
  expect(seriesOf(old, "pred_error")).toEqual([]);
  expect(seriesOf(old, "learning_rate")).toEqual([]);
});

import { familiesPresent, familySeries, strikeSeries } from "./populationView";

const fam = (tick: number, shares: number[] | undefined, extra = {}) =>
  ({ tick, family_territory: shares, family_members: shares?.map((s) => s * 100), ...extra }) as unknown as HistoryRecord;

test("familySeries reads one family's share and skips records without family fields", () => {
  const recs = [fam(100, [0.5, 0.2]), fam(200, undefined), { tick: 250, events: [] } as unknown as HistoryRecord, fam(300, [0.1, 0.4])];
  expect(familySeries(recs, "family_territory", 1)).toEqual([[100, 0.2], [300, 0.4]]);
  expect(familySeries(recs, "family_members", 0)).toEqual([[100, 50], [300, 10]]);
  expect(familySeries(recs, "family_territory", 7)).toEqual([]);
});

test("a family at zero stays on the chart until the end", () => {
  const recs = [fam(100, [0.5, 0.2, 0]), fam(200, [0.7, 0, 0]), fam(300, [0.9, 0, 0])];
  expect(familySeries(recs, "family_territory", 1)).toEqual([[100, 0.2], [200, 0], [300, 0]]);
  expect(familiesPresent(recs)).toEqual([0, 1]); // family 2 never held ground
});

test("strikeSeries sums the three kinds and skips older records", () => {
  const recs = [
    { tick: 100, strikes_own: 2, strikes_foreign: 1, strikes_border: 3 },
    { tick: 200, members: 4 },
  ] as unknown as HistoryRecord[];
  expect(strikeSeries(recs)).toEqual([[100, 6]]);
});

test("eventTicks finds takeover events", () => {
  const recs = [{ tick: 900, events: [{ event: "takeover", tick: 900, family: 2 }] }] as unknown as HistoryRecord[];
  expect(eventTicks(recs, "takeover")).toEqual([900]);
});
