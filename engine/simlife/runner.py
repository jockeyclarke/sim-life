"""Drives the pure step function: recording cadence, checkpoints, extinction."""

from __future__ import annotations

import jax

from .config import Config
from .metrics import compute_metrics
from .recorder import Recorder
from .state import WorldState, init_world
from .step import run_ticks


class Runner:
    record_interval = 100
    checkpoint_interval = 50_000

    def __init__(
        self,
        cfg: Config,
        seed: int,
        recorder: Recorder,
        state: WorldState | None = None,
        extinctions: int = 0,
    ):
        self.cfg = cfg
        self.seed = seed
        self.recorder = recorder
        self.state = state if state is not None else init_world(jax.random.key(seed), cfg)
        self.reseed_on_extinction = True
        self.extinct = False
        self.extinctions = extinctions
        self.last_record: dict | None = None
        self._pending_events: list[dict] = []
        self._last_checkpoint_tick = int(self.state.tick)

    def status(self) -> dict:
        return {
            "run_id": self.recorder.run_id,
            "extinct": self.extinct,
            "extinctions": self.extinctions,
            "reseed_on_extinction": self.reseed_on_extinction,
        }

    def checkpoint(self) -> None:
        self.recorder.checkpoint(self.state, self.extinctions)
        self._last_checkpoint_tick = int(self.state.tick)

    def advance(self, n: int) -> list[dict]:
        """Run up to n ticks. Returns the history records written meanwhile.

        Ticks run in pieces that end on record boundaries, so records always
        fall on multiples of record_interval whatever n is.
        """
        written: list[dict] = []
        remaining = n
        while self._ensure_living(written) and remaining > 0:
            tick = int(self.state.tick)
            piece = min(remaining, self.record_interval - tick % self.record_interval)
            self.state = run_ticks(self.state, self.cfg, piece)
            remaining -= piece
            tick += piece

            if not self._ensure_living(written):
                break
            if tick % self.record_interval == 0:
                written.append(self._record())
            interval = self.checkpoint_interval
            if tick // interval > self._last_checkpoint_tick // interval:
                self.checkpoint()
        return written

    def _record(self, extra_events: list[dict] | None = None) -> dict:
        events = self._pending_events + (extra_events or [])
        self._pending_events = []
        self.last_record = self.recorder.record(compute_metrics(self.state), events)
        return self.last_record

    def _ensure_living(self, written: list[dict]) -> bool:
        """Handle extinction. Returns False when the run is idle (extinct, no reseeding).

        Fresh colonies are created only when none is left alive, which is the
        same as restarting by hand. Colonies are never added to a living world.
        """
        if bool(self.state.col_alive.any()):
            self.extinct = False
            return True

        tick = int(self.state.tick)
        if self.reseed_on_extinction:
            self.extinctions += 1
            key = jax.random.fold_in(jax.random.key(self.seed), self.extinctions)
            self.state = init_world(key, self.cfg)._replace(
                tick=self.state.tick, delivered_total=self.state.delivered_total
            )
            self.extinct = False
            # reported with the next interval record
            self._pending_events.append({"event": "reseed", "tick": tick})
            return True

        if not self.extinct:
            self.extinct = True
            # no later record will come, so write this one now
            written.append(self._record([{"event": "extinct", "tick": tick}]))
        return False
