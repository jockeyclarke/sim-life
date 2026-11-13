"""Population measures read off a state, as plain Python numbers."""

from __future__ import annotations

import jax.numpy as jnp

from .state import WorldState


def compute_metrics(state: WorldState) -> dict:
    generation = jnp.where(state.alive, state.generation, 0).max()
    return {
        "tick": int(state.tick),
        "colonies": int(state.col_alive.sum()),
        "members": int(state.alive.sum()),
        "energy": float(jnp.where(state.col_alive, state.energy, 0.0).sum()),
        "delivered": int(state.delivered_total),
        "max_generation": int(generation),
    }
