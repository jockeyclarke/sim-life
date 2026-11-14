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
from .brain import init_params
from .state import WorldState, init_world

_EXTRA_PREFIX = "_runner."
STRIKE_KINDS = ("own", "foreign", "border")


def save_state(path: Path, state: WorldState, extra: dict | None = None) -> None:
    """Write a checkpoint. The file appears under its final name only when complete.

    `extra` holds the runner's own small numbers and switches (see Runner).
    """
    arrays = {}
    for name, value in state._asdict().items():
        if name == "key":
            arrays[name] = np.asarray(jax.random.key_data(value))
        elif isinstance(value, dict):  # params and birth_params
            for pname, pvalue in value.items():
                arrays[f"{name}.{pname}"] = np.asarray(pvalue)
        else:
            arrays[name] = np.asarray(value)
    for name, value in (extra or {}).items():
        arrays[_EXTRA_PREFIX + name] = np.asarray(value)

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _load(path: Path, cfg: Config) -> tuple[WorldState, dict]:
    template = init_world(jax.random.key(0), cfg)
    with np.load(path) as data:

        def read(label: str, want):
            if label not in data.files:
                raise ValueError(
                    f"checkpoint {path} was recorded by an older version of simlife "
                    f"(it has no '{label}') and cannot be resumed; its history can "
                    "still be viewed and compared"
                )
            got = data[label]
            if got.shape != want.shape:
                raise ValueError(
                    f"checkpoint {path} does not match this run's settings: "
                    f"'{label}' has shape {got.shape} but the capacity settings "
                    f"give {want.shape}"
                )
            return jnp.asarray(got, dtype=want.dtype)

        fields = {}
        for name, expected in template._asdict().items():
            if name == "key":
                fields[name] = jax.random.wrap_key_data(jnp.asarray(data[name]))
            elif isinstance(expected, dict):
                fields[name] = {p: read(f"{name}.{p}", w) for p, w in expected.items()}
            else:
                fields[name] = read(name, expected)
        extra = {
            name[len(_EXTRA_PREFIX):]: data[name].item()
            for name in data.files
            if name.startswith(_EXTRA_PREFIX)
        }
    return WorldState(**fields), extra


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


def _read_founders(run_dir: Path) -> dict | None:
    path = run_dir / FOUNDERS_FILE
    if not path.exists():
        return None
    with np.load(path) as data:
        params = {k[len("params."):]: data[k] for k in data.files if k.startswith("params.")}
        return {"params": params, "lr": data["lr"]}


def list_runs(runs_dir: Path) -> list[str]:
    """Run ids, newest first."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    ids = [p.name for p in runs_dir.iterdir() if (p / "settings.json").is_file()]
    return sorted(ids, reverse=True)


FOUNDERS_FILE = "founders.npz"


def load_survivors(run_dir: Path, cfg: Config) -> tuple[dict, np.ndarray]:
    """(params, lr) of the living members in a run's latest checkpoint.

    Used to seed a new run. Raises ValueError if the source's networks have a
    different shape from this run's, or if nothing in it is alive.
    """
    run_dir = Path(run_dir)
    checkpoints = sorted((run_dir / "checkpoints").glob("state-*.npz"))
    if not checkpoints:
        raise ValueError(f"no checkpoint to take survivors from in {run_dir}")
    expected = init_params(jax.random.key(0), cfg)
    with np.load(checkpoints[-1]) as data:
        alive = data["alive"].astype(bool)
        if not alive.any():
            raise ValueError(f"{run_dir} has no living members to seed from")
        params = {}
        for name, want in expected.items():
            label = f"params.{name}"
            if label not in data.files or data[label].shape[1:] != want.shape:
                got = data[label].shape[1:] if label in data.files else "missing"
                raise ValueError(
                    f"survivors in {run_dir} have a different network shape: "
                    f"'{name}' is {got}, this run needs {tuple(want.shape)}"
                )
            params[name] = data[label][alive]
        lr = data["lr"][alive] if "lr" in data.files else np.full(alive.sum(), cfg.initial_lr)
    return params, lr


class Recorder:
    def __init__(
        self,
        runs_dir: Path,
        cfg: Config,
        seed: int,
        _existing: Path | None = None,
        founders: dict | None = None,
        seeded_from: str | None = None,
    ):
        self.cfg = cfg
        self.seed = seed
        self.founders = founders  # pool of survivors this run's worlds are seeded from
        self._previous: tuple[int, int] | None = None  # (tick, delivered) of last record
        self._previous_strikes = {kind: 0 for kind in STRIKE_KINDS}  # cumulative, at last record
        if _existing is not None:
            self.run_dir = Path(_existing)
            self.run_id = self.run_dir.name
            self.founders = _read_founders(self.run_dir)
            measured = [r for r in read_history(self.run_dir) if "delivered" in r]
            if measured:
                self._previous = (measured[-1]["tick"], measured[-1]["delivered"])
                for kind in STRIKE_KINDS:
                    self._previous_strikes[kind] = measured[-1].get(f"strikes_{kind}_total", 0)
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
        if seeded_from is not None:
            settings["seeded_from"] = seeded_from
        if founders is not None:  # kept with the run, for its fresh starts and resumes
            arrays = {f"params.{k}": np.asarray(v) for k, v in founders["params"].items()}
            np.savez(self.run_dir / FOUNDERS_FILE, lr=np.asarray(founders["lr"]), **arrays)
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
        # strikes since the previous record, by the ground they happened on
        for kind in STRIKE_KINDS:
            total = metrics.get(f"strikes_{kind}_total")
            if total is not None:
                record[f"strikes_{kind}"] = total - self._previous_strikes[kind]
                self._previous_strikes[kind] = total
        if events:
            record["events"] = list(events)
        with open(self.run_dir / "history.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def event(self, event: dict) -> dict:
        """Append an event-only line straight away, so it survives an exit.

        Event lines carry a tick and the event but no measures; readers that
        chart measures skip them.
        """
        line = {"tick": event["tick"], "events": [event]}
        with open(self.run_dir / "history.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        return line

    def checkpoint(self, state: WorldState, extra: dict | None = None) -> Path:
        path = self.run_dir / "checkpoints" / f"state-{int(state.tick):012d}.npz"
        save_state(path, state, extra)
        return path


class Resumed(NamedTuple):
    recorder: Recorder
    state: WorldState
    cfg: Config
    seed: int
    extinctions: int
    reseed_on_extinction: bool
    extinct: bool


def resume_run(run_dir: Path) -> Resumed:
    """Pick a run back up from its latest complete checkpoint."""
    run_dir = Path(run_dir)
    settings = json.loads((run_dir / "settings.json").read_text())
    cfg = Config.from_dict(settings["config"])

    # Only finished checkpoints end in .npz; one being written ends in .tmp.
    checkpoints = sorted((run_dir / "checkpoints").glob("state-*.npz"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint to resume from in {run_dir}")
    state, extra = _load(checkpoints[-1], cfg)

    # Drop history written after that checkpoint; those ticks will be re-run.
    tick = int(state.tick)
    kept = [r for r in read_history(run_dir) if r["tick"] <= tick]
    tmp = run_dir / "history.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r) + "\n" for r in kept))
    os.replace(tmp, run_dir / "history.jsonl")

    recorder = Recorder(run_dir.parent, cfg, settings["seed"], _existing=run_dir)
    return Resumed(
        recorder,
        state,
        cfg,
        settings["seed"],
        extinctions=int(extra.get("extinctions", 0)),
        reseed_on_extinction=bool(extra.get("reseed_on_extinction", True)),
        extinct=bool(extra.get("extinct", False)),
    )
