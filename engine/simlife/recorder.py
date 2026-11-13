"""A run's directory on disk: settings, history, checkpoints.

    runs/<run_id>/settings.json      seed and Config, written once
    runs/<run_id>/history.jsonl      one JSON record per line, appended
    runs/<run_id>/checkpoints/state-<tick>.npz
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import Config
from .state import WorldState, init_world

_PARAM_PREFIX = "params."
_EXTINCTIONS = "_extinctions"


def save_state(path: Path, state: WorldState, extinctions: int = 0) -> None:
    """Write a checkpoint. The file appears under its final name only when complete."""
    arrays = {}
    for name, value in state._asdict().items():
        if name == "key":
            arrays[name] = np.asarray(jax.random.key_data(value))
        elif name == "params":
            for pname, pvalue in value.items():
                arrays[_PARAM_PREFIX + pname] = np.asarray(pvalue)
        else:
            arrays[name] = np.asarray(value)
    arrays[_EXTINCTIONS] = np.asarray(extinctions)

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _load(path: Path, cfg: Config) -> tuple[WorldState, int]:
    template = init_world(jax.random.key(0), cfg)
    with np.load(path) as data:
        fields = {}
        for name, expected in template._asdict().items():
            if name == "key":
                fields[name] = jax.random.wrap_key_data(jnp.asarray(data[name]))
                continue
            if name == "params":
                loaded = {p: data[_PARAM_PREFIX + p] for p in expected}
                pairs = [(f"params.{p}", loaded[p], expected[p]) for p in expected]
                fields[name] = {p: jnp.asarray(v) for p, v in loaded.items()}
            else:
                pairs = [(name, data[name], expected)]
                fields[name] = jnp.asarray(data[name], dtype=expected.dtype)
            for label, got, want in pairs:
                if got.shape != want.shape:
                    raise ValueError(
                        f"checkpoint {path} does not match this run's settings: "
                        f"'{label}' has shape {got.shape} but the capacity settings "
                        f"give {want.shape}"
                    )
        extinctions = int(data[_EXTINCTIONS]) if _EXTINCTIONS in data else 0
    return WorldState(**fields), extinctions


def load_state(path: Path, cfg: Config) -> WorldState:
    return _load(Path(path), cfg)[0]


def read_history(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "history.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a line cut short by an interrupted write
    return records


def list_runs(runs_dir: Path) -> list[str]:
    """Run ids, newest first."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    ids = [p.name for p in runs_dir.iterdir() if (p / "settings.json").is_file()]
    return sorted(ids, reverse=True)


class Recorder:
    def __init__(self, runs_dir: Path, cfg: Config, seed: int, _existing: Path | None = None):
        self.cfg = cfg
        self.seed = seed
        self._previous: tuple[int, int] | None = None  # (tick, delivered) of last record
        if _existing is not None:
            self.run_dir = Path(_existing)
            self.run_id = self.run_dir.name
            history = read_history(self.run_dir)
            if history:
                self._previous = (history[-1]["tick"], history[-1]["delivered"])
            return

        base = f"{time.strftime('%Y%m%d-%H%M%S')}-s{seed}"
        run_id, n = base, 1
        while True:
            try:
                (Path(runs_dir) / run_id / "checkpoints").mkdir(parents=True)
                break
            except FileExistsError:
                n += 1
                run_id = f"{base}-{n}"
        self.run_id = run_id
        self.run_dir = Path(runs_dir) / run_id
        settings = {"run_id": run_id, "seed": seed, "config": cfg.to_dict()}
        (self.run_dir / "settings.json").write_text(json.dumps(settings, indent=2))

    def record(self, metrics: dict, events: list[dict] | None = None) -> dict:
        """Append one history record, adding delivery_rate. Returns the record.

        delivery_rate is pellets delivered per member per tick since the last
        record: the measure of how well the population forages.
        """
        rate = 0.0
        if self._previous is not None:
            ticks = metrics["tick"] - self._previous[0]
            member_ticks = metrics["members"] * ticks
            if member_ticks > 0:
                rate = (metrics["delivered"] - self._previous[1]) / member_ticks
        self._previous = (metrics["tick"], metrics["delivered"])

        record = {**metrics, "delivery_rate": rate}
        if events:
            record["events"] = list(events)
        with open(self.run_dir / "history.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def checkpoint(self, state: WorldState, extinctions: int = 0) -> Path:
        path = self.run_dir / "checkpoints" / f"state-{int(state.tick):012d}.npz"
        save_state(path, state, extinctions)
        return path


class Resumed(NamedTuple):
    recorder: Recorder
    state: WorldState
    cfg: Config
    seed: int
    extinctions: int


def resume_run(run_dir: Path) -> Resumed:
    """Pick a run back up from its latest complete checkpoint."""
    run_dir = Path(run_dir)
    settings = json.loads((run_dir / "settings.json").read_text())
    cfg = Config.from_dict(settings["config"])

    # Only finished checkpoints end in .npz; one being written ends in .tmp.
    checkpoints = sorted((run_dir / "checkpoints").glob("state-*.npz"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint to resume from in {run_dir}")
    state, extinctions = _load(checkpoints[-1], cfg)

    # Drop history written after that checkpoint; those ticks will be re-run.
    tick = int(state.tick)
    kept = [r for r in read_history(run_dir) if r["tick"] <= tick]
    tmp = run_dir / "history.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r) + "\n" for r in kept))
    os.replace(tmp, run_dir / "history.jsonl")

    recorder = Recorder(run_dir.parent, cfg, settings["seed"], _existing=run_dir)
    return Resumed(recorder, state, cfg, settings["seed"], extinctions)
