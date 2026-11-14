import type { Snapshot } from "./protocol";

/** One piece of a carrier's path, in world coordinates: [x0, y0, x1, y1]. */
export type Segment = [number, number, number, number];

/**
 * Follows food carriers from snapshot to snapshot. For each member carrying
 * food now that was also carrying in the previous snapshot, it yields the
 * segment between the two positions. Steps across the wrap-around edge are
 * skipped so no line is drawn across the whole world.
 */
export class TrailTracker {
  private last = new Map<number, [number, number]>();
  private lastTick = -1;

  update(snap: Snapshot): { segments: Segment[]; reset: boolean } {
    const reset = snap.tick < this.lastTick;
    if (reset) this.last.clear();
    if (snap.tick === this.lastTick) return { segments: [], reset };
    this.lastTick = snap.tick;

    const pos = snap.arrays.member_pos ?? new Float32Array();
    const slots = snap.arrays.member_slot ?? new Float32Array();
    const carrying = snap.arrays.member_carrying ?? new Float32Array();
    const next = new Map<number, [number, number]>();
    const segments: Segment[] = [];
    for (let i = 0; i < slots.length; i++) {
      if (!(carrying[i]! > 0.5)) continue;
      const here: [number, number] = [pos[2 * i]!, pos[2 * i + 1]!];
      const before = this.last.get(slots[i]!);
      if (before && Math.abs(here[0] - before[0]) < 0.5 && Math.abs(here[1] - before[1]) < 0.5) {
        segments.push([before[0], before[1], here[0], here[1]]);
      }
      next.set(slots[i]!, here);
    }
    this.last = next;
    return { segments, reset };
  }
}

/**
 * The trail layer: a transparent canvas that fades a little with every new
 * snapshot, so a path dims over roughly `1 / fade` snapshots.
 */
export class TrailLayer {
  readonly canvas = document.createElement("canvas");
  private tracker = new TrailTracker();

  constructor(private fade = 0.04) {}

  clear(): void {
    this.canvas.getContext("2d")!.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  /** Add one snapshot's movement. `width`/`height` are the world canvas's backing-store pixels. */
  add(snap: Snapshot, width: number, height: number, color: string): void {
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width; // resizing also clears
      this.canvas.height = height;
    }
    const { segments, reset } = this.tracker.update(snap);
    const ctx = this.canvas.getContext("2d")!;
    if (reset) this.clear();
    ctx.globalCompositeOperation = "destination-out";
    ctx.fillStyle = `rgba(0, 0, 0, ${this.fade})`;
    ctx.fillRect(0, 0, width, height);
    ctx.globalCompositeOperation = "source-over";
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.6;
    ctx.lineWidth = 1.5 * (window.devicePixelRatio || 1);
    ctx.lineCap = "round";
    ctx.beginPath();
    for (const [x0, y0, x1, y1] of segments) {
      ctx.moveTo(x0 * width, y0 * height);
      ctx.lineTo(x1 * width, y1 * height);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
}
