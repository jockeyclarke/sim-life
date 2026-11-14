import type { HistoryRecord } from "./protocol";
import { familyColor, familyDash } from "./families";
import { cssVar, fitCanvas } from "./theme";

type Field = "colonies" | "members" | "energy" | "delivery_rate" | "pred_error" | "learning_rate";
type Point = [tick: number, value: number];

/** (tick, value) pairs sorted by tick, skipping records where the field is missing or not finite. */
export function seriesOf(records: HistoryRecord[], field: keyof HistoryRecord): Point[] {
  const points: Point[] = [];
  for (const record of records) {
    const value = record[field];
    if (typeof value === "number" && Number.isFinite(value)) points.push([record.tick, value]);
  }
  return points.sort((a, b) => a[0] - b[0]);
}

/** Ticks at which events of one kind happened ("reseed", "switch", "extinct"). */
export function eventTicks(records: HistoryRecord[], name: string): number[] {
  return records.flatMap((r) => (r.events ?? []).filter((e) => e.event === name).map((e) => e.tick));
}

/** Ticks at which the run started fresh colonies after an extinction. */
export function reseedTicks(records: HistoryRecord[]): number[] {
  return eventTicks(records, "reseed");
}

/**
 * The record closest to a tick, ignoring event-only lines (which carry a tick
 * and an event but no measures).
 */
export function nearestMeasured(records: HistoryRecord[], tick: number): HistoryRecord | null {
  let best: HistoryRecord | null = null;
  for (const record of records) {
    if (typeof record.members !== "number") continue;
    if (best === null || Math.abs(record.tick - tick) < Math.abs(best.tick - tick)) best = record;
  }
  return best;
}

/** One family's value over time ("family_territory" share or "family_members"). */
export function familySeries(records: HistoryRecord[], field: "family_territory" | "family_members", k: number): Point[] {
  const points: Point[] = [];
  for (const record of records) {
    const value = record[field]?.[k];
    if (typeof value === "number" && Number.isFinite(value)) points.push([record.tick, value]);
  }
  return points.sort((a, b) => a[0] - b[0]);
}

/** Families that held any territory at some point in these records. */
export function familiesPresent(records: HistoryRecord[]): number[] {
  const seen = new Set<number>();
  for (const record of records) record.family_territory?.forEach((share, k) => share > 0 && seen.add(k));
  return [...seen].sort((a, b) => a - b);
}

/** All strikes per record interval: own + foreign + border ground. */
export function strikeSeries(records: HistoryRecord[]): Point[] {
  const points: Point[] = [];
  for (const r of records) {
    if (typeof r.strikes_own !== "number") continue;
    points.push([r.tick, r.strikes_own + (r.strikes_foreign ?? 0) + (r.strikes_border ?? 0)]);
  }
  return points.sort((a, b) => a[0] - b[0]);
}

export interface Scale {
  lo: number;
  hi: number;
  ticks: number[];
}

/** Round axis bounds and tick values (steps of 1, 2 or 5 times a power of ten) containing [min, max]. */
export function niceScale(min: number, max: number, target: number): Scale {
  if (min === max) {
    // a flat series still needs a range to be drawn in
    const pad = min === 0 ? 1 : Math.abs(min) * 0.5;
    min = min === 0 ? 0 : min - pad;
    max = max + pad;
  }
  const raw = (max - min) / Math.max(1, target);
  const power = 10 ** Math.floor(Math.log10(raw));
  const mantissa = raw / power;
  const step = (mantissa <= 1 ? 1 : mantissa <= 2 ? 2 : mantissa <= 5 ? 5 : 10) * power;
  const decimals = Math.max(0, -Math.floor(Math.log10(step)));
  const round = (v: number) => Number(v.toFixed(decimals));

  const lo = round(Math.floor(min / step) * step);
  const hi = round(Math.ceil(max / step) * step);
  const ticks: number[] = [];
  for (let i = 0; lo + i * step <= hi + step / 2; i++) ticks.push(round(lo + i * step));
  return { lo, hi, ticks };
}

function formatValue(v: number): string {
  const abs = Math.abs(v);
  if (abs === 0) return "0";
  if (abs >= 1e6) return `${Number((v / 1e6).toFixed(1))}M`;
  if (abs >= 1e3) return `${Number((v / 1e3).toFixed(1))}k`;
  if (abs >= 10) return String(Math.round(v));
  if (abs >= 1) return String(Number(v.toFixed(1)));
  return String(Number(v.toPrecision(2)));
}

type ChartField = Field | "strikes" | "family_territory";

/** The (tick, value) points one chart plots for one run. */
function pointsFor(records: HistoryRecord[], field: ChartField): Point[] {
  if (field === "strikes") return strikeSeries(records);
  if (field === "family_territory") return []; // drawn per family, below
  return seriesOf(records, field);
}

const CHARTS: { field: ChartField; title: string }[] = [
  { field: "colonies", title: "Colonies" },
  { field: "members", title: "Members" },
  { field: "energy", title: "Total energy" },
  { field: "delivery_rate", title: "Food per member per tick" },
  { field: "pred_error", title: "Prediction error" },
  { field: "learning_rate", title: "Learning rate" },
  { field: "family_territory", title: "Territory by family" },
  { field: "strikes", title: "Strikes per 100 ticks" },
];

const HEIGHT = 92;
const PAD = { left: 48, right: 8, top: 20, bottom: 6 };
const AXIS_HEIGHT = 16; // extra room under the last chart for tick labels

/**
 * Four small charts sharing one tick axis. Each measure gets its own chart
 * and its own y scale, so nothing is ever drawn against two axes.
 */
export class PopulationView {
  private current: HistoryRecord[] = [];
  private currentId = "";
  private overlay: HistoryRecord[] = [];
  private overlayId: string | null = null;
  private hoverTick: number | null = null;
  private queued = false;

  private readonly legend = document.createElement("div");
  private readonly readout = document.createElement("div");
  private readonly canvases: HTMLCanvasElement[];

  constructor(private readonly root: HTMLElement) {
    root.style.position = "relative";
    this.legend.className = "legend";
    this.readout.className = "readout";
    this.canvases = CHARTS.map((_, i) => {
      const canvas = document.createElement("canvas");
      const height = HEIGHT + (i === CHARTS.length - 1 ? AXIS_HEIGHT : 0);
      canvas.style.height = `${height}px`;
      canvas.addEventListener("mousemove", (event) => this.onHover(event, canvas));
      canvas.addEventListener("mouseleave", () => this.setHover(null));
      return canvas;
    });
    root.append(this.legend, this.readout, ...this.canvases);

    new ResizeObserver(() => this.queue()).observe(root);
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.queue());
    this.queue();
  }

  setCurrent(runId: string, records: HistoryRecord[]): void {
    this.currentId = runId;
    this.current = [...records];
    this.queue();
  }

  append(record: HistoryRecord): void {
    this.current.push(record);
    this.queue();
  }

  setOverlay(runId: string | null, records: HistoryRecord[]): void {
    this.overlayId = runId;
    this.overlay = runId === null ? [] : [...records];
    this.queue();
  }

  /** Coalesce redraws: many records can arrive between two screen refreshes. */
  private queue(): void {
    if (this.queued) return;
    this.queued = true;
    requestAnimationFrame(() => {
      this.queued = false;
      this.draw();
    });
  }

  private maxTick(): number {
    const last = (records: HistoryRecord[]) => records.reduce((m, r) => Math.max(m, r.tick), 0);
    return Math.max(last(this.current), last(this.overlay), 1);
  }

  private plotWidth(): number {
    return Math.max(10, this.root.clientWidth - PAD.left - PAD.right);
  }

  private onHover(event: MouseEvent, canvas: HTMLCanvasElement): void {
    const x = event.clientX - canvas.getBoundingClientRect().left - PAD.left;
    const fraction = Math.min(1, Math.max(0, x / this.plotWidth()));
    this.setHover(fraction * this.maxTick());
  }

  private setHover(tick: number | null): void {
    this.hoverTick = tick;
    this.queue();
  }

  private drawLegend(): void {
    this.legend.replaceChildren();
    if (this.overlayId === null) return; // one series: the chart titles name it
    const entry = (label: string, color: string, dashed: boolean) => {
      const span = document.createElement("span");
      span.textContent = label;
      span.style.setProperty("--swatch", color);
      if (dashed) span.className = "dashed";
      return span;
    };
    this.legend.append(
      entry(`this run (${this.currentId})`, cssVar("--series"), false),
      entry(`compared run (${this.overlayId})`, cssVar("--overlay"), true),
    );
  }

  private draw(): void {
    this.drawLegend();
    const width = this.root.clientWidth;
    if (width <= 0) return;
    const text = cssVar("--text");
    const muted = cssVar("--muted");
    const line = cssVar("--line");
    const colors = { current: cssVar("--series"), overlay: cssVar("--overlay"), event: cssVar("--event") };

    const maxTick = this.maxTick();
    const plotW = this.plotWidth();
    const plotH = HEIGHT - PAD.top - PAD.bottom;
    const xOf = (tick: number) => PAD.left + (tick / maxTick) * plotW;
    // vertical rules: a fresh start after extinction, or a switch being flipped
    const reseeds = [
      ...reseedTicks(this.current),
      ...eventTicks(this.current, "switch"),
      ...eventTicks(this.current, "takeover"),
    ];
    const dark = matchMedia("(prefers-color-scheme: dark)").matches;
    const empty = this.current.length === 0 && this.overlay.length === 0;

    CHARTS.forEach((chart, index) => {
      const canvas = this.canvases[index]!;
      const isLast = index === CHARTS.length - 1;
      const ctx = fitCanvas(canvas, width, HEIGHT + (isLast ? AXIS_HEIGHT : 0));
      ctx.clearRect(0, 0, width, HEIGHT + AXIS_HEIGHT);
      ctx.font = "11px system-ui, sans-serif";
      ctx.textBaseline = "middle";

      const now = pointsFor(this.current, chart.field);
      const then = pointsFor(this.overlay, chart.field);
      // Territory: one line per family in its own colour, on a 0-1 share axis.
      // The overlay run is left off this chart; six more lines would be unreadable.
      const families = chart.field === "family_territory" ? familiesPresent(this.current) : [];
      const familyLines = families.map((k) => familySeries(this.current, "family_territory", k));
      const values = [...now, ...then, ...familyLines.flat()].map((p) => p[1]);
      const scale =
        chart.field === "family_territory"
          ? niceScale(0, 1, 2)
          : niceScale(Math.min(0, ...values), Math.max(0, ...values), 3);
      const yOf = (v: number) => PAD.top + plotH - ((v - scale.lo) / (scale.hi - scale.lo)) * plotH;

      // title, in text ink (never the series colour)
      ctx.fillStyle = text;
      ctx.textAlign = "left";
      ctx.fillText(chart.title, PAD.left, 8);

      // hover readout: this chart's value at the crosshair, right-aligned on the title row.
      // Values are in text ink; the short line beside each carries the series identity.
      if (families.length > 0) {
        // legend: a short line in each family's colour, then its number
        let right = PAD.left + plotW;
        for (const k of [...families].reverse()) {
          const label = `${k}`;
          ctx.textAlign = "right";
          ctx.fillStyle = text;
          ctx.fillText(label, right, 8);
          right -= ctx.measureText(label).width + 3;
          ctx.strokeStyle = familyColor(k, dark);
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(right - 10, 8);
          ctx.lineTo(right, 8);
          ctx.stroke();
          right -= 16;
        }
        ctx.textAlign = "left";
      } else if (this.hoverTick !== null) {
        let right = PAD.left + plotW;
        const entries: [HistoryRecord | null, string, number[]][] = [
          [this.overlayId === null ? null : nearestMeasured(this.overlay, this.hoverTick), colors.overlay, [3, 2]],
          [nearestMeasured(this.current, this.hoverTick), colors.current, []],
        ];
        for (const [record, color, dash] of entries) {
          const v = chart.field === "strikes"
            ? (record ? strikeSeries([record])[0]?.[1] : undefined)
            : record?.[chart.field as Field];
          if (typeof v !== "number" || !Number.isFinite(v)) continue;
          const label = formatValue(v);
          ctx.textAlign = "right";
          ctx.fillStyle = text;
          ctx.fillText(label, right, 8);
          right -= ctx.measureText(label).width + 4;
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.setLineDash(dash);
          ctx.beginPath();
          ctx.moveTo(right - 12, 8);
          ctx.lineTo(right, 8);
          ctx.stroke();
          ctx.setLineDash([]);
          right -= 22;
        }
        ctx.textAlign = "left";
      }

      // recessive grid and y labels
      ctx.strokeStyle = line;
      ctx.lineWidth = 1;
      ctx.fillStyle = muted;
      ctx.textAlign = "right";
      for (const tick of scale.ticks) {
        const y = Math.round(yOf(tick)) + 0.5;
        ctx.beginPath();
        ctx.moveTo(PAD.left, y);
        ctx.lineTo(PAD.left + plotW, y);
        ctx.stroke();
        ctx.fillText(formatValue(tick), PAD.left - 6, y);
      }

      if (empty) {
        ctx.textAlign = "center";
        ctx.fillText("no history yet", PAD.left + plotW / 2, PAD.top + plotH / 2);
      }

      // reseed events on this run
      ctx.strokeStyle = colors.event;
      ctx.setLineDash([2, 3]);
      for (const tick of reseeds) {
        const x = Math.round(xOf(tick)) + 0.5;
        ctx.beginPath();
        ctx.moveTo(x, PAD.top);
        ctx.lineTo(x, PAD.top + plotH);
        ctx.stroke();
      }

      const path = (points: Point[], color: string, dash: number[]) => {
        if (points.length === 0) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.lineJoin = "round";
        ctx.setLineDash(dash);
        ctx.beginPath();
        points.forEach(([tick, value], i) => {
          if (i === 0) ctx.moveTo(xOf(tick), yOf(value));
          else ctx.lineTo(xOf(tick), yOf(value));
        });
        ctx.stroke();
        if (points.length === 1) {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(xOf(points[0]![0]), yOf(points[0]![1]), 3, 0, Math.PI * 2);
          ctx.fill();
        }
      };
      path(then, colors.overlay, [5, 4]); // dashed as well as coloured, so identity is never colour alone
      path(now, colors.current, []);
      families.forEach((k, i) => path(familyLines[i]!, familyColor(k, dark), familyDash(k)));
      ctx.setLineDash([]);

      if (this.hoverTick !== null && !empty) {
        const x = Math.round(xOf(this.hoverTick)) + 0.5;
        ctx.strokeStyle = muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, PAD.top);
        ctx.lineTo(x, PAD.top + plotH);
        ctx.stroke();
      }

      if (isLast) {
        const axis = niceScale(0, maxTick, 6);
        ctx.fillStyle = muted;
        ctx.textAlign = "center";
        for (const tick of axis.ticks) {
          if (tick > maxTick) continue;
          ctx.fillText(formatValue(tick), xOf(tick), HEIGHT + AXIS_HEIGHT / 2);
        }
        ctx.textAlign = "right";
        ctx.fillText("tick", PAD.left - 6, HEIGHT + AXIS_HEIGHT / 2);
      }
    });

    // One line of text says which tick the crosshair is on; the values sit on each chart's title row.
    const at = this.hoverTick === null ? null : (nearestMeasured(this.current, this.hoverTick) ?? nearestMeasured(this.overlay, this.hoverTick));
    this.readout.textContent = at ? `at tick ${at.tick.toLocaleString("en-US")}` : "hover a chart to read values";
  }
}
