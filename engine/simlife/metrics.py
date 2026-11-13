"""Population measures read off a state, as plain Python numbers."""

from __future__ import annotations

import jax.numpy as jnp

from .state import WorldState


def compute_metrics(state: WorldState) -> dict:
    generation = jnp.where(state.alive, state.generation, 0).max()
    alive = state.alive
    scored = alive & state.has_prev
    n_alive = jnp.maximum(alive.sum(), 1)
    n_scored = jnp.maximum(scored.sum(), 1)
    pred_error = jnp.where(scored, state.pred_error_avg, 0.0).sum() / n_scored
    learning_rate = jnp.where(alive, state.lr, 0.0).sum() / n_alive
    return {
        "tick": int(state.tick),
        "colonies": int(state.col_alive.sum()),
        "members": int(state.alive.sum()),
        "energy": float(jnp.where(state.col_alive, state.energy, 0.0).sum()),
        "delivered": int(state.delivered_total),
        "max_generation": int(generation),
        # averages over living members; 0.0 when there are none
        "pred_error": float(pred_error),
        "learning_rate": float(learning_rate),
    }
