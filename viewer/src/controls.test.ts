import { expect, test } from "vitest";
import { describeStatus, sliderToTicks, ticksToSlider } from "./controls";
import type { Metrics, Status } from "./protocol";

const metrics: Metrics = { tick: 12345, colonies: 7, members: 310, energy: 99.5, delivered: 40, max_generation: 12 };
const status: Status = {
  run_id: "r", extinct: false, extinctions: 2, reseed_on_extinction: true, paused: false, ticks_per_frame: 1,
  learning: true, inheritance: "learned",
};

test("the speed slider covers 1 to 1000 ticks per frame on a log scale", () => {
  expect(sliderToTicks(0)).toBe(1);
  expect(sliderToTicks(3)).toBe(1000);
  expect(sliderToTicks(1)).toBe(10);
  for (const n of [1, 2, 10, 37, 500, 1000]) expect(sliderToTicks(ticksToSlider(n))).toBe(n);
});

test("status line reports the counts", () => {
  const text = describeStatus(status, metrics);
  expect(text).toContain("tick 12,345");
  expect(text).toContain("7 colonies");
  expect(text).toContain("310 members");
  expect(text).toContain("2 extinctions");
});

test("status line uses singular forms", () => {
  const text = describeStatus({ ...status, extinctions: 1 }, { ...metrics, colonies: 1, members: 1 });
  expect(text).toContain("1 colony");
  expect(text).toContain("1 member ");
  expect(text).toContain("1 extinction");
  expect(text).not.toContain("extinctions");
});

test("status line says when the run is extinct with reseeding off", () => {
  expect(describeStatus({ ...status, extinct: true, reseed_on_extinction: false }, metrics)).toContain(
    "extinct — reseeding is off",
  );
  expect(describeStatus(status, metrics)).not.toContain("reseeding is off");
});

test("status line still reads cleanly with the stage 2 fields present", () => {
  const s = { ...status, learning: false, inheritance: "birth" as const };
  expect(describeStatus(s, metrics)).toContain("tick 12,345");
});
