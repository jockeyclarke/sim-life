"""Population measures read off a state, as plain Python numbers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .state import WorldState
from .territory import owner_grid


def family_metrics(state: WorldState) -> dict:
    """Per-family measures (lists indexed by family; a dead family reads 0)."""
    K = state.territory.shape[0]
    member_family = state.col_family[state.colony]
    alive = state.alive.astype(jnp.float32)
    members = jax.ops.segment_sum(alive, member_family, K)
    colonies = jax.ops.segment_sum(state.col_alive.astype(jnp.int32), state.col_family, K)
    n = jnp.maximum(members, 1.0)
    mean = jax.ops.segment_sum(alive * state.aggression, member_family, K) / n
    square = jax.ops.segment_sum(alive * state.aggression**2, member_family, K) / n
    sd = jnp.sqrt(jnp.maximum(square - mean**2, 0.0))
    owner, _ = owner_grid(state, _GridSize(state.territory.shape[1]))
    cells = owner.size
    territory = jnp.zeros(K).at[jnp.where(owner >= 0, owner, K)].add(1.0, mode="drop") / cells

    def numbers(a, kind=float) -> list:
        return [kind(v) for v in a.tolist()]

    died = state.colonies_died
    life = state.colony_life_total
    return {
        "families": int((colonies > 0).sum()),
        "family_members": numbers(members, int),
        "family_colonies": numbers(colonies, int),
        "family_territory": numbers(territory),
        "family_aggression_mean": numbers(jnp.where(members > 0, mean, 0.0)),
        "family_aggression_sd": numbers(jnp.where(members > 0, sd, 0.0)),
        "strikes_own_total": int(state.strikes[0]),
        "strikes_foreign_total": int(state.strikes[1]),
        "strikes_border_total": int(state.strikes[2]),
        "colonies_died_home": int(died[0]),
        "colonies_died_foreign": int(died[1]),
        # mean lifespan, in ticks, of colonies that have died, by founding ground
        "colony_life_home": float(life[0] / died[0]) if int(died[0]) else 0.0,
        "colony_life_foreign": float(life[1] / died[1]) if int(died[1]) else 0.0,
    }


class _GridSize:
    """owner_grid only needs the grid size from a Config."""

    def __init__(self, g: int):
        self.territory_grid = g


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
        **family_metrics(state),
    }
