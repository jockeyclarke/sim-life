import { Connection } from "./connection";
import { mountControls } from "./controls";
import type { Snapshot } from "./protocol";
import { cssVar, fitCanvas } from "./theme";
import { type ColorMode, drawWorld } from "./worldView";

const canvas = document.querySelector<HTMLCanvasElement>("#world")!;
const port = new URLSearchParams(location.search).get("port") ?? "8765";
const url = `ws://${location.hostname || "127.0.0.1"}:${port}`;

let latest: Snapshot | null = null;
let drawn: Snapshot | null = null;
let mode: ColorMode = "colony";

const controls = mountControls(document.querySelector<HTMLElement>("#controls")!, (m) => connection.send(m), {
  onColorMode(next) {
    mode = next;
    drawn = null; // force a redraw even while paused
  },
  onOverlay() {},
});
controls.setConnection(false);

const connection = new Connection(url, {
  onSnapshot(snapshot) {
    latest = snapshot;
    controls.update(snapshot.status, snapshot.metrics);
  },
  onMessage(message) {
    if (message.type === "runs") controls.setRuns(message.runs, message.current);
  },
  onState(state) {
    controls.setConnection(state === "connected");
  },
});

// Snapshots can arrive faster than the screen refreshes; draw only the newest, once per frame.
let lastSize = 0;
function frame(): void {
  const size = canvas.clientWidth;
  if (latest && (latest !== drawn || size !== lastSize)) {
    const ctx = fitCanvas(canvas, size, size);
    drawWorld(ctx, latest, size, mode, {
      background: cssVar("--world"),
      food: cssVar("--food"),
      muted: cssVar("--muted"),
      accent: cssVar("--accent"),
    });
    drawn = latest;
    lastSize = size;
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
