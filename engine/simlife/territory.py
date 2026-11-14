"""Territory: each family's scent on a coarse grid, laid by its members and nests.

Nobody draws borders. A family's region is simply where its scent is
strongest, so territory follows wherever its nests and members are.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .config import Config
from .state import WorldState


def cell_of(points: Array, cfg: Config) -> Array:
    """Flat index y * G + x of the grid cell containing each point."""
    G = cfg.territory_grid
    xy = jnp.clip((points * G).astype(jnp.int32), 0, G - 1)
    return xy[..., 1] * G + xy[..., 0]


def update_territory(state: WorldState, cfg: Config) -> WorldState:
    """One tick of scent: fade, spread to neighbouring cells, then fresh marks."""
    G, K = cfg.territory_grid, cfg.n_families
    t = state.territory * cfg.scent_fade

    neighbours = (
        jnp.roll(t, 1, axis=1) + jnp.roll(t, -1, axis=1)
        + jnp.roll(t, 1, axis=2) + jnp.roll(t, -1, axis=2)
    ) / 4  # roll wraps, as the world does
    t = (1 - cfg.scent_spread) * t + cfg.scent_spread * neighbours

    member_family = state.col_family[state.colony]
    member_marks = jax.ops.segment_sum(
        state.alive * cfg.mark_rate,
        member_family * G * G + cell_of(state.pos, cfg),
        num_segments=K * G * G,
    )
    nest_marks = jax.ops.segment_sum(
        state.col_alive * cfg.nest_mark,
        state.col_family * G * G + cell_of(state.nest, cfg),
        num_segments=K * G * G,
    )
    t = t + (member_marks + nest_marks).reshape(K, G, G)
    return state._replace(territory=t)


def scent_at(state: WorldState, points: Array, family: Array, cfg: Config) -> tuple[Array, Array]:
    """(own family's scent, all other families' scent combined) at each point."""
    flat = state.territory.reshape(cfg.n_families, -1)[:, cell_of(points, cfg)]  # (K, M)
    own = jnp.take_along_axis(flat, family[None, :], axis=0)[0]
    return own, flat.sum(axis=0) - own


def home_share(own: Array, foreign: Array) -> Array:
    """Own family's share of the scent: 1 at home, 0 abroad, 0.5 on unmarked ground."""
    total = own + foreign
    return jnp.where(total < 1e-6, 0.5, own / jnp.maximum(total, 1e-6))


def owner_grid(state: WorldState, cfg: Config) -> tuple[Array, Array]:
    """(owner, strength) per cell: the family with most scent (or -1 if the
    ground is unmarked), and that family's share of the scent there."""
    total = state.territory.sum(axis=0)
    marked = total >= 1e-3
    owner = jnp.where(marked, jnp.argmax(state.territory, axis=0), -1).astype(jnp.int32)
    strength = jnp.where(marked, state.territory.max(axis=0) / jnp.maximum(total, 1e-6), 0.0)
    return owner, strength
