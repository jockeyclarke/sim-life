import { expect, test } from "vitest";
import { decodeSnapshot, parseServerMessage } from "./protocol";
import { readFixture } from "./testFixture";

test("decodes the engine's fixture", () => {
  const snap = decodeSnapshot(readFixture("snapshot.bin"));
  expect(snap.tick).toBe(0);
  expect(snap.arrays.member_pos!.length).toBe(192 * 2);
  expect(snap.shapes.colony_nest).toEqual([16, 2]);
  expect(snap.status.run_id).toBe("fixture");
  expect(snap.metrics.members).toBe(192);
  for (const v of snap.arrays.member_pos!) {
    expect(v).toBeGreaterThanOrEqual(0);
    expect(v).toBeLessThan(1);
  }
});

test("decoded arrays do not alias the frame's buffer", () => {
  const buf = readFixture("snapshot.bin");
  const snap = decodeSnapshot(buf);
  const before = snap.arrays.member_pos![0];
  new Uint8Array(buf).fill(0);
  expect(snap.arrays.member_pos![0]).toBe(before);
});

test("rejects a truncated frame", () => {
  expect(() => decodeSnapshot(readFixture("snapshot.bin").slice(0, 40))).toThrow();
});

test("rejects a frame whose arrays overrun the buffer", () => {
  const buf = readFixture("snapshot.bin");
  expect(() => decodeSnapshot(buf.slice(0, buf.byteLength - 8))).toThrow(/overrun/);
});

test("rejects an empty frame and a frame that is not a snapshot", () => {
  expect(() => decodeSnapshot(new ArrayBuffer(2))).toThrow();
  const head = new TextEncoder().encode('{"type":"other"}');
  const frame = new Uint8Array(4 + head.length);
  new DataView(frame.buffer).setUint32(0, head.length, true);
  frame.set(head, 4);
  expect(() => decodeSnapshot(frame.buffer)).toThrow(/snapshot/);
});

test("parses known server messages and rejects the rest", () => {
  expect(parseServerMessage('{"type":"runs","runs":["a"],"current":"a"}')).toEqual({
    type: "runs", runs: ["a"], current: "a",
  });
  expect(parseServerMessage('{"type":"history","run_id":"a","records":[]}')?.type).toBe("history");
  expect(parseServerMessage('{"type":"record","run_id":"a","record":{"tick":1}}')?.type).toBe("record");
  for (const bad of ["{", "[]", "null", '{"type":"nope"}', '{"type":"runs"}', '{"type":"history","run_id":"a"}']) {
    expect(parseServerMessage(bad)).toBeNull();
  }
});

test("fixture carries an inspection and member errors", () => {
  const snap = decodeSnapshot(readFixture("snapshot.bin"));
  expect(snap.inspect?.alive).toBe(true);
  if (snap.inspect?.alive) {
    expect(snap.inspect.slot).toBe(0);
    expect(snap.inspect.inputs.length).toBe(10);
    expect(snap.inspect.weights.w3.length).toBe(6);
  }
  expect(snap.arrays.member_error!.length).toBe(192);
  expect(snap.status.learning).toBe(true);
  expect(snap.status.inheritance).toBe("learned");
});

test("a snapshot from an older engine has no inspection", () => {
  const head = new TextEncoder().encode('{"type":"snapshot","tick":1,"metrics":{},"status":{},"arrays":[]}');
  const frame = new Uint8Array(4 + head.length);
  new DataView(frame.buffer).setUint32(0, head.length, true);
  frame.set(head, 4);
  expect(decodeSnapshot(frame.buffer).inspect).toBeNull();
});
