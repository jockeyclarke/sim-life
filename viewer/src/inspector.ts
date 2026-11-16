// The inspector: one member's senses, activations, predictions, error and weights, live.

import { divergingColor, drawHeatmap } from "./heatmap";
import type { DeadInspection, Inspection, Snapshot } from "./protocol";
import { cssVar, fitCanvas } from "./theme";
import { familyColor } from "./families";

// In protocol order. See engine/simlife/senses.py (observe) and brain.py (act).
export const INPUT_NAMES = [
  "food nearby", "food direction", "nest nearby", "nest direction", "crowd", "heard",
  "territory left", "territory right", "foreigners",
  "carrying", "energy", "memory 0", "memory 1",
];
/**
 * The two lines under the error trace. Always two, each short enough not to
 * wrap, so the panel below keeps its place as the numbers change.
 */
/**
 * Half-width of the prediction bars' axis, which always runs -extent..+extent so
 * zero stays in the middle. It is 1 (inputs are mostly in -1..1) and grows in
 * steps of 0.5 only when a value goes beyond, so the scale rarely moves.
 */
export function barExtent(values: number[]): number {
  const largest = Math.max(0, ...values.filter(Number.isFinite).map(Math.abs));
  return Math.max(1, Math.ceil(largest * 2) / 2);
}

export function traceCaption(points: [number, number][]): [string, string] {
  if (points.length === 0) return ["waiting for this member's", "first scored prediction"];
  const first = points[0]![1];
  const latest = points[points.length - 1]![1];
  return ["running average, log scale", `first seen ${first.toPrecision(2)} · now ${latest.toPrecision(2)}`];
}

export const ACTION_NAMES =["turn", "speed", "signal", "memory 0", "memory 1", "aggression"];
export const PREDICTED_NAMES = INPUT_NAMES.slice(0, 9);

const INPUT_CODES = ["fN", "fD", "nN", "nD", "cr", "sh", "tL", "tR", "fo", "ca", "en", "m0", "m1"];
const ACTION_CODES = ["turn", "spd", "sig", "m0", "m1", "agg"];
const HIDDEN_CODES = Array.from({ length: 8 }, (_, i) => `h${i}`);

/** The selected member's prediction error over time, as seen by this page since selection. */
export class ErrorTrace {
  private slot: number | null = null;
  private data: [number, number][] = [];

  constructor(private readonly capacity = 600) {}

  push(slot: number, tick: number, error: number): void {
    if (slot !== this.slot) {
      this.slot = slot;
      this.data = [];
    }
    if (!(error > 0) || !Number.isFinite(error)) return; // no error yet, or not chartable on a log scale
    const last = this.data[this.data.length - 1];
    if (last && tick <= last[0]) return; // a paused engine re-sends the same snapshot
    this.data.push([tick, error]);
    if (this.data.length > this.capacity) this.data.shift();
  }

  points(): [number, number][] {
    return this.data;
  }

  clear(): void {
    this.slot = null;
    this.data = [];
  }
}

/** The family of the selected member, from the snapshot's member arrays; null if unknown. */
export function familyOf(snap: Snapshot): number | null {
  const slot = snap.inspect?.slot;
  const slots = snap.arrays.member_slot;
  const families = snap.arrays.member_family;
  if (slot === undefined || !slots || !families) return null;
  const i = slots.indexOf(slot);
  return i >= 0 ? families[i]! : null;
}

/**
 * Whether the inspector is now following a different member (or none).
 * Members are told apart by the server's selection identity, never by slot:
 * slots are reused after a death.
 */
export function selectionChanged(previous: number | null, inspect: Inspection | DeadInspection | null): boolean {
  return (inspect?.selection ?? null) !== previous;
}

/** Elementwise now − then. A shape mismatch gives zeros in `now`'s shape. */
export function weightChange(now: number[][], then: number[][]): number[][] {
  const same = now.length === then.length && now.every((row, r) => row.length === then[r]!.length);
  return now.map((row, r) => row.map((value, c) => (same ? value - then[r]![c]! : 0)));
}

export interface LogScale {
  lo: number;
  hi: number;
  ticks: number[];
  /** 0 at lo, 1 at hi. */
  y(value: number): number;
}

/** A log axis with power-of-ten bounds. Prediction error falls by orders of magnitude, which a linear axis hides. */
export function logScale(values: number[]): LogScale {
  const positive = values.filter((v) => v > 0 && Number.isFinite(v));
  let loExp = positive.length ? Math.floor(Math.log10(Math.min(...positive))) : -3;
  let hiExp = positive.length ? Math.ceil(Math.log10(Math.max(...positive))) : 0;
  if (hiExp <= loExp) hiExp = loExp + 1;
  const ticks: number[] = [];
  for (let e = loExp; e <= hiExp; e++) ticks.push(10 ** e);
  return {
    lo: 10 ** loExp,
    hi: 10 ** hiExp,
    ticks,
    y: (value) => (Math.log10(value) - loExp) / (hiExp - loExp),
  };
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: Partial<HTMLElementTagNameMap[K]> = {},
  children: (Node | string)[] = [],
): HTMLElementTagNameMap[K] {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}

const fmt = (v: number, digits = 2) => (Object.is(v, -0) ? 0 : v).toFixed(digits);
const signed = (v: number, digits = 3) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`;
const isDark = () => matchMedia("(prefers-color-scheme: dark)").matches;

/** A named list of values, each with a colour swatch: one column of the senses → hidden → actions row. */
class ValueList {
  readonly root = el("div", { className: "values" });
  private readonly rows: { swatch: HTMLElement; number: HTMLElement }[];

  constructor(title: string, names: string[]) {
    this.root.append(el("h4", { textContent: title }));
    this.rows = names.map((name) => {
      const swatch = el("span", { className: "swatch" });
      const number = el("span", { className: "num" });
      this.root.append(el("div", { className: "value", title: name }, [el("span", { className: "name", textContent: name }), swatch, number]));
      return { swatch, number };
    });
  }

  set(values: number[], dark: boolean): void {
    this.rows.forEach((row, i) => {
      const v = values[i] ?? 0;
      row.swatch.style.background = divergingColor(v, 1, dark);
      row.number.textContent = fmt(v);
    });
  }
}

interface Matrix {
  key: "w1" | "w3" | "w2";
  caption: string;
  rowCodes: string[];
  colCodes: string[];
  describe(row: number, col: number): string;
  canvas: HTMLCanvasElement;
}

const CELL = 18;
const LABEL_W = 34;
const HEADER_H = 14;

export class Inspector {
  private readonly trace = new ErrorTrace();
  private baseline: Inspection["weights"] | null = null;
  private selection: number | null = null;
  private showChange = false;
  private last: Inspection | null = null;

  private readonly message = el("p", { className: "hint" });
  private readonly body = el("div");
  private readonly who = el("p", { className: "who" });
  private readonly inputs = new ValueList("Senses", INPUT_NAMES);
  private readonly hidden = new ValueList("Hidden", HIDDEN_CODES);
  private readonly actions = new ValueList("Actions", ACTION_NAMES);
  private readonly bars = el("canvas");
  private readonly traceCanvas = el("canvas");
  private readonly traceNote = el("p", { className: "hint trace-note" });
  private readonly readout = el("p", { className: "hint", textContent: "hover a weight to read it" });
  private readonly matrices: Matrix[];

  constructor(private readonly root: HTMLElement) {
    const matrix = (key: Matrix["key"], caption: string, rowCodes: string[], colCodes: string[], describe: Matrix["describe"]): Matrix => {
      const canvas = el("canvas");
      const m = { key, caption, rowCodes, colCodes, describe, canvas };
      canvas.addEventListener("mousemove", (event) => this.hoverWeight(m, event));
      canvas.addEventListener("mouseleave", () => (this.readout.textContent = "hover a weight to read it"));
      return m;
    };
    this.matrices = [
      matrix("w1", "Senses → hidden · trained during life", HIDDEN_CODES, INPUT_CODES,
        (r, c) => `${INPUT_NAMES[c]} → hidden ${r}`),
      matrix("w3", "Hidden → predictions · trained during life", INPUT_CODES.slice(0, 9), HIDDEN_CODES,
        (r, c) => `hidden ${c} → predicted ${PREDICTED_NAMES[r]}`),
      matrix("w2", "Hidden → actions · changes only at division", ACTION_CODES, HIDDEN_CODES,
        (r, c) => `hidden ${c} → ${ACTION_NAMES[r]}`),
    ];

    const toggle = el("input", { type: "checkbox" });
    toggle.onchange = () => {
      this.showChange = toggle.checked;
      if (this.last) this.drawWeights(this.last);
    };

    this.body.append(
      this.who,
      el("h3", { textContent: "This tick" }),
      el("div", { className: "flow" }, [this.inputs.root, this.hidden.root, this.actions.root]),
      el("h3", { textContent: "Last prediction against what happened" }),
      el("div", { className: "legend" }, [
        el("span", { className: "solid", textContent: "predicted" }),
        el("span", { className: "outline", textContent: "actual" }),
      ]),
      this.bars,
      el("h3", { textContent: "Prediction error since selected" }),
      this.traceNote,
      this.traceCanvas,
      el("h3", { textContent: "Weights" }),
      el("label", { className: "check" }, [toggle, " Show change since selected"]),
      ...this.matrices.flatMap((m) => [el("p", { className: "caption", textContent: m.caption }), m.canvas]),
      this.readout,
    );
    root.append(this.message, this.body);
    this.show(null, "click a member in the world to inspect it");
  }

  /** The selected member's family, looked up from the snapshot's arrays. */
  private family: number | null = null;

  update(snap: Snapshot): void {
    const inspect = snap.inspect;
    // a dead member is no longer in the arrays: keep the family last seen
    const family = familyOf(snap);
    if (selectionChanged(this.selection, inspect)) {
      // a different member, or none: nothing from the previous one carries over
      this.selection = inspect?.selection ?? null;
      this.family = null;
      this.trace.clear();
      this.baseline = null;
      this.last = null;
    }
    if (family !== null) this.family = family;
    if (inspect === null) {
      this.show(null, "click a member in the world to inspect it");
      return;
    }
    if (!inspect.alive) {
      // the same member, now dead: keep its last picture and error trace on screen
      this.show(this.last, this.last ? "this member has died" : "that member has died");
      return;
    }
    if (this.baseline === null) this.baseline = structuredClone(inspect.weights);
    this.last = inspect;
    this.trace.push(inspect.selection, snap.tick, inspect.error_avg);
    this.show(inspect, "");
  }

  private show(inspect: Inspection | null, message: string): void {
    this.message.textContent = message;
    this.message.hidden = message === "";
    this.body.hidden = inspect === null;
    if (inspect === null) return;
    const dark = isDark();

    const swatch = el("span", { className: "colony" });
    const family = this.family;
    swatch.style.background = family === null ? "transparent" : familyColor(family, dark);
    this.who.replaceChildren(
      `member ${inspect.slot} · `, swatch, ` family ${family ?? "?"} · colony ${inspect.colony} · age ${inspect.age.toLocaleString("en-US")} of ${inspect.lifespan.toLocaleString("en-US")}`,
      el("br"),
      `generation ${inspect.generation} · learning rate ${inspect.learning_rate.toFixed(4)} · ${inspect.carrying ? "carrying food" : "not carrying"}`,
    );
    this.inputs.set(inspect.inputs, dark);
    this.hidden.set(inspect.hidden, dark);
    this.actions.set(inspect.actions, dark);
    this.drawBars(inspect);
    this.drawTrace();
    this.drawWeights(inspect);
  }

  /** Six pairs of bars on one shared axis: what it predicted, and what then happened. */
  private drawBars(inspect: Inspection): void {
    const width = this.root.clientWidth;
    if (width <= 0) return;
    const rowH = 24;
    const labelW = 84;
    const errorW = 52;
    const height = PREDICTED_NAMES.length * rowH + 16;
    const ctx = fitCanvas(this.bars, width, height);
    this.bars.style.height = `${height}px`;
    ctx.clearRect(0, 0, width, height);
    ctx.font = "11px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    const text = cssVar("--text");
    const muted = cssVar("--muted");
    const predictedColor = cssVar("--series");
    const actualColor = cssVar("--overlay");

    const hi = barExtent([...inspect.predicted, ...inspect.actual]);
    const lo = -hi;
    const x0 = labelW;
    const plotW = Math.max(10, width - labelW - errorW);
    const xOf = (v: number) => x0 + ((v - lo) / (hi - lo)) * plotW;

    // axis: zero line and end labels
    ctx.strokeStyle = cssVar("--line");
    ctx.lineWidth = 1;
    for (const v of [lo, 0, hi]) {
      const x = Math.round(xOf(v)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height - 16);
      ctx.stroke();
      ctx.fillStyle = muted;
      ctx.textAlign = "center";
      ctx.fillText(fmt(v, 1), x, height - 7);
    }
    ctx.textAlign = "right";
    ctx.fillText("error²", width, height - 7);

    PREDICTED_NAMES.forEach((name, i) => {
      const y = i * rowH;
      const predicted = inspect.predicted[i] ?? 0;
      const actual = inspect.actual[i] ?? 0;
      ctx.fillStyle = text;
      ctx.textAlign = "left";
      ctx.fillText(name, 0, y + rowH / 2);

      const bar = (value: number, top: number, color: string, filled: boolean) => {
        const a = xOf(Math.min(0, value));
        const w = Math.max(1, Math.abs(xOf(value) - xOf(0)));
        if (filled) {
          ctx.fillStyle = color;
          ctx.fillRect(a, top, w, 7);
        } else {
          // outlined, not filled: the two series differ by more than colour
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(a + 0.75, top + 0.75, Math.max(0.5, w - 1.5), 5.5);
        }
      };
      bar(predicted, y + 3, predictedColor, true);
      bar(actual, y + 13, actualColor, false);

      ctx.fillStyle = text;
      ctx.textAlign = "right";
      ctx.fillText(((predicted - actual) ** 2).toFixed(4), width, y + rowH / 2);
    });
  }

  private drawTrace(): void {
    const width = this.root.clientWidth;
    if (width <= 0) return;
    const height = 120;
    const pad = { left: 44, right: 8, top: 6, bottom: 18 };
    const ctx = fitCanvas(this.traceCanvas, width, height);
    this.traceCanvas.style.height = `${height}px`;
    ctx.clearRect(0, 0, width, height);
    ctx.font = "11px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    const muted = cssVar("--muted");

    const points = this.trace.points();
    const scale = logScale(points.map((p) => p[1]));
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const t0 = points[0]?.[0] ?? 0;
    const t1 = Math.max(points[points.length - 1]?.[0] ?? 1, t0 + 1);
    const xOf = (tick: number) => pad.left + ((tick - t0) / (t1 - t0)) * plotW;
    const yOf = (value: number) => pad.top + plotH - scale.y(value) * plotH;

    ctx.strokeStyle = cssVar("--line");
    ctx.lineWidth = 1;
    ctx.fillStyle = muted;
    ctx.textAlign = "right";
    for (const tick of scale.ticks) {
      const y = Math.round(yOf(tick)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
      ctx.fillText(tick >= 1 ? String(tick) : tick.toFixed(Math.max(0, -Math.floor(Math.log10(tick)))), pad.left - 6, y);
    }
    ctx.textAlign = "left";
    ctx.fillText(`tick ${t0.toLocaleString("en-US")}`, pad.left, height - 8);
    ctx.textAlign = "right";
    ctx.fillText(t1.toLocaleString("en-US"), pad.left + plotW, height - 8);

    this.traceNote.textContent = traceCaption(points).join("\n");
    if (points.length === 0) return;

    ctx.strokeStyle = cssVar("--series");
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    points.forEach(([tick, value], i) => {
      if (i === 0) ctx.moveTo(xOf(tick), yOf(value));
      else ctx.lineTo(xOf(tick), yOf(value));
    });
    ctx.stroke();
    if (points.length === 1) {
      ctx.fillStyle = cssVar("--series");
      ctx.beginPath();
      ctx.arc(xOf(points[0]![0]), yOf(points[0]![1]), 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  private shown(inspect: Inspection, key: Matrix["key"]): number[][] {
    return this.showChange && this.baseline ? weightChange(inspect.weights[key], this.baseline[key]) : inspect.weights[key];
  }

  private drawWeights(inspect: Inspection): void {
    const dark = isDark();
    const muted = cssVar("--muted");
    // one scale across all three, so colours are comparable between them
    const largest = Math.max(...this.matrices.flatMap((m) => this.shown(inspect, m.key).flat().map(Math.abs)));
    const scale = this.showChange ? Math.max(largest, 1e-3) : Math.max(largest, 0.5);

    for (const m of this.matrices) {
      const rows = this.shown(inspect, m.key);
      const width = LABEL_W + m.colCodes.length * CELL;
      const height = HEADER_H + m.rowCodes.length * CELL;
      const ctx = fitCanvas(m.canvas, width, height);
      m.canvas.style.width = `${width}px`;
      m.canvas.style.height = `${height}px`;
      ctx.clearRect(0, 0, width, height);
      ctx.font = "10px system-ui, sans-serif";
      ctx.textBaseline = "middle";
      ctx.fillStyle = muted;
      ctx.textAlign = "center";
      m.colCodes.forEach((code, c) => ctx.fillText(code, LABEL_W + c * CELL + CELL / 2, HEADER_H / 2));
      ctx.textAlign = "right";
      m.rowCodes.forEach((code, r) => ctx.fillText(code, LABEL_W - 5, HEADER_H + r * CELL + CELL / 2));
      drawHeatmap(ctx, rows, LABEL_W, HEADER_H, CELL, scale, dark);
    }
  }

  private hoverWeight(m: Matrix, event: MouseEvent): void {
    if (!this.last) return;
    const box = m.canvas.getBoundingClientRect();
    const col = Math.floor((event.clientX - box.left - LABEL_W) / CELL);
    const row = Math.floor((event.clientY - box.top - HEADER_H) / CELL);
    const value = this.last.weights[m.key][row]?.[col];
    if (value === undefined || col < 0 || row < 0) {
      this.readout.textContent = "hover a weight to read it";
      return;
    }
    const then = this.baseline?.[m.key][row]?.[col];
    const change = then === undefined ? "" : ` · ${signed(value - then)} since selected`;
    this.readout.textContent = `${m.describe(row, col)}: ${signed(value)}${change}`;
  }
}
