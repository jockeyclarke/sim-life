// The wire format shared with the engine. See engine/simlife/protocol.py.

export interface Metrics {
  tick: number;
  colonies: number;
  members: number;
  energy: number;
  delivered: number;
  max_generation: number;
  /** Mean prediction error and learning rate of living members. Absent in runs recorded before stage 2. */
  pred_error?: number;
  learning_rate?: number;
}

export interface Status {
  run_id: string;
  extinct: boolean;
  extinctions: number;
  reseed_on_extinction: boolean;
  paused: boolean;
  ticks_per_frame: number;
  learning: boolean;
  inheritance: "learned" | "birth";
}

/** The selected member's internals. See engine/simlife/inspect.py. */
export interface Inspection {
  slot: number;
  alive: true;
  colony: number;
  age: number;
  lifespan: number;
  generation: number;
  carrying: boolean;
  learning_rate: number;
  inputs: number[]; // 10
  hidden: number[]; // 8
  actions: number[]; // turn, speed, signal, memory 0, memory 1
  predicted: number[]; // 6: last tick's guess
  actual: number[]; // 6: what was then sensed
  next_prediction: number[]; // 6
  error: number;
  error_avg: number;
  weights: { w1: number[][]; b1: number[]; w2: number[][]; b2: number[]; w3: number[][]; b3: number[] };
}

/** The selected member has died. Its slot may since have been taken by a newborn. */
export interface DeadInspection {
  slot: number;
  alive: false;
}

export interface RunEvent {
  event: string;
  tick: number;
}

/**
 * One line of a run's history. Most lines are measures taken every 100 ticks.
 * A few are event-only lines (tick and events, no measures), written the
 * moment something happens; code that charts measures must skip those.
 */
export interface HistoryRecord extends Metrics {
  delivery_rate: number;
  events?: RunEvent[];
}

export type ServerMessage =
  | { type: "runs"; runs: string[]; current: string }
  | { type: "history"; run_id: string; records: HistoryRecord[] }
  | { type: "record"; run_id: string; record: HistoryRecord };

export type ControlMessage =
  | { type: "pause"; value: boolean }
  | { type: "speed"; ticks_per_frame: number }
  | { type: "reseed_on_extinction"; value: boolean }
  | { type: "learning"; value: boolean }
  | { type: "inheritance"; value: "learned" | "birth" }
  | { type: "select"; slot: number | null }
  | { type: "list_runs" }
  | { type: "get_history"; run_id: string };

export interface Snapshot {
  tick: number;
  metrics: Metrics;
  status: Status;
  inspect: Inspection | DeadInspection | null;
  /** Flat float32 data per array name; use `shapes` for the dimensions. */
  arrays: Record<string, Float32Array>;
  shapes: Record<string, number[]>;
}

interface ArraySpec {
  name: string;
  shape: number[];
}

/**
 * Decode one binary snapshot frame:
 *   4 bytes   little-endian header length L
 *   L bytes   UTF-8 JSON header
 *   rest      float32 arrays (little-endian), back to back, in header order
 * Throws on anything truncated or malformed.
 */
export function decodeSnapshot(buf: ArrayBuffer): Snapshot {
  if (buf.byteLength < 4) throw new Error("snapshot frame is too short");
  const headerLength = new DataView(buf).getUint32(0, true);
  const bodyStart = 4 + headerLength;
  if (bodyStart > buf.byteLength) throw new Error("snapshot header is truncated");

  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headerLength)));
  if (header?.type !== "snapshot" || !Array.isArray(header.arrays)) {
    throw new Error("frame is not a snapshot");
  }

  const arrays: Record<string, Float32Array> = {};
  const shapes: Record<string, number[]> = {};
  let offset = bodyStart;
  for (const spec of header.arrays as ArraySpec[]) {
    const count = spec.shape.reduce((a, b) => a * b, 1);
    const end = offset + 4 * count;
    if (end > buf.byteLength) throw new Error(`snapshot array ${spec.name} would overrun the frame`);
    // Copy: the header length is arbitrary, so `offset` is rarely 4-byte
    // aligned and a Float32Array view straight onto `buf` would throw.
    arrays[spec.name] = new Float32Array(buf.slice(offset, end));
    shapes[spec.name] = spec.shape;
    offset = end;
  }
  return {
    tick: header.tick,
    metrics: header.metrics,
    status: header.status,
    inspect: header.inspect ?? null,
    arrays,
    shapes,
  };
}

export function parseServerMessage(text: string): ServerMessage | null {
  let message: unknown;
  try {
    message = JSON.parse(text);
  } catch {
    return null;
  }
  if (typeof message !== "object" || message === null || Array.isArray(message)) return null;
  const m = message as Record<string, unknown>;
  if (m.type === "runs" && Array.isArray(m.runs) && typeof m.current === "string") {
    return m as unknown as ServerMessage;
  }
  if (m.type === "history" && typeof m.run_id === "string" && Array.isArray(m.records)) {
    return m as unknown as ServerMessage;
  }
  if (m.type === "record" && typeof m.run_id === "string" && typeof m.record === "object" && m.record !== null) {
    return m as unknown as ServerMessage;
  }
  return null;
}
