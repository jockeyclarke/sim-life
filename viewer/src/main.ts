import { Connection } from "./connection";
import { mountControls } from "./controls";
import { PopulationView } from "./populationView";
import type { Snapshot } from "./protocol";
import { cssVar, fitCanvas } from "./theme";
import { type ColorMode, drawWorld } from "./worldView";

const canvas = document.querySelector<HTMLCanvasElement>("#world")!;
// Page options:  ?port=8765  the engine's port;  ?compare=<run id>|previous  preselect a comparison run
const query = new URLSearchParams(location.search);
const port = query.get("port") ?? "8765";
let compare = query.get("compare");
const url = `ws://${location.hostname || "127.0.0.1"}:${port}`;

let latest: Snapshot | null = null;
let drawn: Snapshot | null = null;
let mode: ColorMode = "colony";
let currentRun = "";
let overlayRun: string | null = null;

const population = new PopulationView(document.querySelector<HTMLElement>("#population")!);

const controls = mountControls(document.querySelector<HTMLElement>("#controls")!, (m) => connection.send(m), {
  onColorMode(next) {
    mode = next;
    drawn = null; // force a redraw even while paused
  },
  onOverlay(runId) {
    overlayRun = runId;
    if (runId === null) population.setOverlay(null, []);
    else connection.send({ type: "get_history", run_id: runId });
  },
});
controls.setConnection(false);

const connection = new Connection(url, {
  onSnapshot(snapshot) {
    latest = snapshot;
    controls.update(snapshot.status, snapshot.metrics);
  },
  onMessage(message) {
    if (message.type === "runs") {
      currentRun = message.current;
      controls.setRuns(message.runs, message.current);
      if (compare !== null) {
        controls.chooseOverlay(compare);
        compare = null; // only on first load; after that the menu is the user's
      }
    } else if (message.type === "history") {
      // the engine sends the current run's history on connect; anything else is a comparison we asked for
      if (message.run_id === currentRun) population.setCurrent(message.run_id, message.records);
      else if (message.run_id === overlayRun) population.setOverlay(message.run_id, message.records);
    } else if (message.type === "record" && message.run_id === currentRun) {
      population.append(message.record);
    }
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
