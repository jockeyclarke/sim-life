import type { ControlMessage, Metrics, Status } from "./protocol";
import type { ColorMode } from "./worldView";

/** Slider position 0..3 maps to 1..1000 ticks per frame (powers of ten). */
export function sliderToTicks(position: number): number {
  return Math.max(1, Math.min(1000, Math.round(10 ** position)));
}

export function ticksToSlider(ticks: number): number {
  return Math.log10(Math.max(1, ticks));
}

function plural(n: number, one: string, many: string): string {
  return `${n.toLocaleString("en-US")} ${n === 1 ? one : many}`;
}

export function describeStatus(status: Status, metrics: Metrics): string {
  let text = [
    `tick ${metrics.tick.toLocaleString("en-US")}`,
    plural(metrics.colonies, "colony", "colonies"),
    plural(metrics.members, "member", "members"),
    plural(status.extinctions, "extinction", "extinctions"),
  ].join(" · ");
  if (status.extinct && !status.reseed_on_extinction) text += " · extinct — reseeding is off";
  return text;
}

export interface ControlsOptions {
  onColorMode(mode: ColorMode): void;
  onShadeTerritory(on: boolean): void;
  onOverlay(runId: string | null): void;
}

export interface Controls {
  update(status: Status, metrics: Metrics): void;
  setConnection(connected: boolean): void;
  setRuns(runs: string[], current: string): void;
  /** Pick a comparison run as if the user had chosen it. "previous" means the newest other run. */
  chooseOverlay(runId: string): void;
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

export function mountControls(
  root: HTMLElement,
  send: (message: ControlMessage) => void,
  options: ControlsOptions,
): Controls {
  let paused = false;

  const connection = el("p", { className: "connection" });
  const statusLine = el("p", { className: "status" }, ["waiting for the engine…"]);
  const runLine = el("p", { className: "run" });

  const pauseButton = el("button", { type: "button", textContent: "Pause" });
  pauseButton.onclick = () => send({ type: "pause", value: !paused });

  const speedValue = el("output", { textContent: "1" });
  const speed = el("input", { type: "range", min: "0", max: "3", step: "0.01", value: "0" });
  speed.oninput = () => {
    const ticks = sliderToTicks(Number(speed.value));
    speedValue.textContent = String(ticks);
    send({ type: "speed", ticks_per_frame: ticks });
  };

  const colorMode = el("select", {}, [
    el("option", { value: "family", textContent: "by family" }),
    el("option", { value: "colony", textContent: "by colony" }),
    el("option", { value: "carrying", textContent: "by carrying food" }),
    el("option", { value: "error", textContent: "by prediction error" }),
    el("option", { value: "aggression", textContent: "by aggression" }),
  ]);
  colorMode.onchange = () => options.onColorMode(colorMode.value as ColorMode);

  const shade = el("input", { type: "checkbox", checked: true });
  shade.onchange = () => options.onShadeTerritory(shade.checked);

  const reseed = el("input", { type: "checkbox", checked: true });
  reseed.onchange = () => send({ type: "reseed_on_extinction", value: reseed.checked });

  const learning = el("input", { type: "checkbox", checked: true });
  learning.onchange = () => send({ type: "learning", value: learning.checked });

  const inheritance = el("select", {}, [
    el("option", { value: "learned", textContent: "what the parent learned" }),
    el("option", { value: "birth", textContent: "what the parent was born with" }),
  ]);
  inheritance.onchange = () => send({ type: "inheritance", value: inheritance.value as "learned" | "birth" });

  const overlay = el("select", {}, [el("option", { value: "", textContent: "none" })]);
  overlay.onchange = () => options.onOverlay(overlay.value || null);

  root.append(
    connection,
    statusLine,
    runLine,
    el("div", { className: "row" }, [pauseButton]),
    el("label", {}, ["Ticks per frame ", speedValue, speed]),
    el("label", {}, ["Colour members ", colorMode]),
    el("label", { className: "check" }, [shade, " Shade territory"]),
    el("label", { className: "check" }, [reseed, " Start fresh colonies when all have died"]),
    el("label", { className: "check" }, [learning, " Members learn during their lives"]),
    el("label", {}, ["Offspring inherit ", inheritance]),
    el("label", {}, ["Compare with run ", overlay]),
  );

  return {
    update(status, metrics) {
      paused = status.paused;
      pauseButton.textContent = paused ? "Resume" : "Pause";
      statusLine.textContent = describeStatus(status, metrics);
      runLine.textContent = `run ${status.run_id}`;
      // Don't fight the user's hand: leave a control alone while it has focus.
      if (document.activeElement !== speed) {
        speed.value = String(ticksToSlider(status.ticks_per_frame));
        speedValue.textContent = String(status.ticks_per_frame);
      }
      if (document.activeElement !== reseed) reseed.checked = status.reseed_on_extinction;
      // an engine from before stage 2 sends neither field; leave the controls as they are
      if (document.activeElement !== learning && typeof status.learning === "boolean") {
        learning.checked = status.learning;
      }
      if (document.activeElement !== inheritance && status.inheritance) inheritance.value = status.inheritance;
    },
    setConnection(connected) {
      connection.textContent = connected ? "connected to the engine" : "engine not reachable — retrying";
      connection.dataset.state = connected ? "connected" : "disconnected";
    },
    setRuns(runs, current) {
      const chosen = overlay.value;
      const others = runs.filter((id) => id !== current);
      overlay.replaceChildren(
        el("option", { value: "", textContent: "none" }),
        ...others.map((id) => el("option", { value: id, textContent: id })),
      );
      overlay.value = others.includes(chosen) ? chosen : "";
    },
    chooseOverlay(runId) {
      const ids = Array.from(overlay.options, (o) => o.value).filter((id) => id !== "");
      const wanted = runId === "previous" ? ids[0] : runId;
      if (wanted === undefined || !ids.includes(wanted)) return;
      overlay.value = wanted;
      options.onOverlay(wanted);
    },
  };
}
