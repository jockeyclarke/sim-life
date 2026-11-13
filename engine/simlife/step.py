"""The whole simulation as one pure function: step(state, cfg) -> state."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .brain import act_all
from .config import Config
from .lifecycle import age_and_die, grow, kill_dead_colonies, regrow_food, split
from .physics import deliver, metabolise, move, pickup
from .senses import observe
from .state import WorldState


def step(state: WorldState, cfg: Config) -> WorldState:
    """Advance the world by one tick. The order of the phases is fixed."""
    key, k_grow, k_split, k_food = jax.random.split(state.key, 4)

    obs = observe(state, cfg)  # (N, 10)
    turn, speed, signal, memory = act_all(state.params, obs)  # every network at once

    alive = state.alive
    state = state._replace(
        key=key,
        signal=jnp.where(alive, signal, 0.0),
        memory=jnp.where(alive[:, None], memory, 0.0),
    )
    state = move(state, turn, speed, cfg)
    state = pickup(state, cfg)
    state = deliver(state, cfg)
    state = metabolise(state, speed, cfg)
    state = age_and_die(state, cfg)
    state = kill_dead_colonies(state)
    state = grow(state, cfg, k_grow)
    state = split(state, cfg, k_split)
    state = regrow_food(state, cfg, k_food)
    return state._replace(tick=state.tick + 1)


@partial(jax.jit, static_argnames=("cfg",))
def run_ticks(state: WorldState, cfg: Config, n: int) -> WorldState:
    """Up to n ticks in one compiled call, stopping early if every colony dies.

    cfg is a compile-time constant (changing it recompiles). n is an ordinary
    runtime value, so any chunk size reuses the same compiled loop.

    Stopping on the exact tick of extinction is what lets the runner start
    fresh colonies at the same tick however the run is chunked. The caller
    reads how far it got from the returned state's tick.
    """

    def keep_going(carry):
        i, s = carry
        return (i < n) & s.col_alive.any()

    def one_tick(carry):
        i, s = carry
        return i + 1, step(s, cfg)

    _, final = jax.lax.while_loop(keep_going, one_tick, (jnp.int32(0), state))
    return final
