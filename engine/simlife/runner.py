"""Drives the pure step function: recording cadence, checkpoints, extinction."""

from __future__ import annotations

import threading
from pathlib import Path

import jax
import jax.numpy as jnp

from .config import Config
from .metrics import compute_metrics
from .recorder import Recorder, resume_run
from .state import WorldState, init_world
from .step import run_ticks


# switch name -> the WorldState field that holds it
SWITCHES = {"learning": "learning_on", "inherit_learned": "inherit_learned"}


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
        self._fresh_world = jax.jit(lambda key: init_world(key, cfg))
        self.state = state if state is not None else init_world(jax.random.key(seed), cfg)
        self.reseed_on_extinction = True
        self.extinct = False
        self.extinctions = extinctions
        self._last_checkpoint_tick = int(self.state.tick)
        # advance() may run in a worker thread while a viewer flips a switch;
        # the lock keeps a switch from being overwritten by a piece in flight.
        self._lock = threading.Lock()
        self._families_alive = self._count_families()

    @classmethod
    def resume(cls, run_dir: Path) -> "Runner":
        """Continue a recorded run, with its switches as they were left."""
        r = resume_run(run_dir)
        runner = cls(r.cfg, r.seed, r.recorder, state=r.state, extinctions=r.extinctions)
        runner.reseed_on_extinction = r.reseed_on_extinction
        runner.extinct = r.extinct
        return runner

    def _count_families(self) -> int:
        s = self.state
        return len(set(s.col_family[s.col_alive].tolist()))

    def _watch_takeover(self, written: list[dict]) -> None:
        """Record, once, the moment a single family is left holding the field."""
        now = self._count_families()
        if now == 1 and self._families_alive > 1:
            family = int(self.state.col_family[self.state.col_alive][0])
            written.append(self.recorder.event(
                {"event": "takeover", "family": family, "tick": int(self.state.tick)}
            ))
        self._families_alive = now

    def status(self) -> dict:
        return {
            "run_id": self.recorder.run_id,
            "extinct": self.extinct,
            "extinctions": self.extinctions,
            "reseed_on_extinction": self.reseed_on_extinction,
            "learning": bool(self.state.learning_on),
            "inheritance": "learned" if bool(self.state.inherit_learned) else "birth",
        }

    def set_switch(self, name: str, value: bool) -> dict | None:
        """Flip the learning or inheritance switch, from the next tick on.

        Returns the history line recording the change, or None if the switch
        already had that value.
        """
        if name not in SWITCHES:
            raise ValueError(f"unknown switch {name!r}; expected one of {sorted(SWITCHES)}")
        field = SWITCHES[name]
        with self._lock:
            if bool(getattr(self.state, field)) == bool(value):
                return None
            self.state = self.state._replace(**{field: jnp.bool_(value)})
            event = {
                "event": "switch", "name": name, "value": bool(value),
                "tick": int(self.state.tick),
            }
            return self.recorder.event(event)

    def checkpoint(self) -> None:
        extra = {
            "extinctions": self.extinctions,
            "reseed_on_extinction": self.reseed_on_extinction,
            "extinct": self.extinct,
        }
        self.recorder.checkpoint(self.state, extra)
        self._last_checkpoint_tick = int(self.state.tick)

    def advance(self, n: int) -> list[dict]:
        """Run up to n ticks. Returns the history lines written meanwhile.

        Ticks run in pieces that end on a record boundary or on the tick
        every colony died, whichever comes first. So records fall on multiples
        of record_interval, and a reseed happens on the exact tick of
        extinction, whatever n is: the run is the same however it is chunked.
        """
        written: list[dict] = []
        remaining = n
        while remaining > 0:
            with self._lock:
                if not self._ensure_living(written):
                    break
                before = int(self.state.tick)
                piece = min(remaining, self.record_interval - before % self.record_interval)
                self.state = run_ticks(self.state, self.cfg, piece)
                tick = int(self.state.tick)
                remaining -= tick - before

                self._watch_takeover(written)
                formed = int(self.state.founded) + self.cfg.grace_ticks
                if before < formed <= tick:
                    written.append(self.recorder.event({"event": "families_formed", "tick": formed}))
                if tick > before and tick % self.record_interval == 0:
                    written.append(self.recorder.record(compute_metrics(self.state)))
                interval = self.checkpoint_interval
                if tick // interval > self._last_checkpoint_tick // interval:
                    self.checkpoint()
        with self._lock:
            self._ensure_living(written)
        return written

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
            self.state = self._fresh_world(key)._replace(
                tick=self.state.tick,
                delivered_total=self.state.delivered_total,
                # a fresh start is a new population, not a new experiment
                learning_on=self.state.learning_on,
                inherit_learned=self.state.inherit_learned,
                # the run's cumulative counts go on
                strikes=self.state.strikes,
                colonies_died=self.state.colonies_died,
                colony_life_total=self.state.colony_life_total,
            )
            fresh = self.state
            self.state = fresh._replace(
                col_born=jnp.where(fresh.col_alive, fresh.tick, 0),
                founded=fresh.tick,  # a new world: its own grace period
            )
            self._families_alive = self._count_families()
            self.extinct = False
            written.append(self.recorder.event({"event": "reseed", "tick": tick}))
            return True

        if not self.extinct:
            self.extinct = True
            written.append(self.recorder.event({"event": "extinct", "tick": tick}))
        return False
