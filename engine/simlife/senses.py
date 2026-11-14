"""What an entity can sense. Everything is local and relative to its heading."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .config import Config
from .state import WorldState
from .territory import scent_at


def wrap_delta(a: Array, b: Array) -> Array:
    """a - b on the wrap-around world: each component in [-0.5, 0.5)."""
    return (a - b + 0.5) % 1.0 - 0.5


def wrap01(p: Array) -> Array:
    """Wrap coordinates into [0, 1).

    In float32, `x % 1.0` returns exactly 1.0 for a tiny negative x, which
    would put an entity outside the world, so that case is folded back to 0.
    """
    p = p % 1.0
    return jnp.where(p >= 1.0, 0.0, p)


def torus_distance(a: Array, b: Array) -> Array:
    return jnp.sqrt((wrap_delta(a, b) ** 2).sum(-1))


def sensor_points(pos: Array, heading: Array, cfg: Config) -> tuple[Array, Array]:
    """Two sample points ahead of each entity, to the left and right of its heading."""

    def point(angle: Array) -> Array:
        direction = jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=-1)
        return wrap01(pos + cfg.sensor_distance * direction)

    return point(heading + cfg.sensor_angle), point(heading - cfg.sensor_angle)


def food_scent(points: Array, food_pos: Array, food_alive: Array, cfg: Config) -> Array:
    """Sum over live pellets of exp(-distance / food_scent_length), per point."""
    dist = torus_distance(points[:, None, :], food_pos[None, :, :])  # (M, F)
    return jnp.where(food_alive, jnp.exp(-dist / cfg.food_scent_length), 0.0).sum(1)


def nest_scent(points: Array, nest: Array, cfg: Config) -> Array:
    """exp(-distance / nest_scent_length) from each point to its paired nest."""
    return jnp.exp(-torus_distance(points, nest) / cfg.nest_scent_length)


def neighbour_stats(state: WorldState, cfg: Config) -> tuple[Array, Array]:
    """(crowd, heard) from colony-mates sharing the entity's grid cell.

    Comparing every pair of entities would cost N squared. Instead each entity
    is dropped into a bucket keyed by (colony, grid cell); per-bucket totals
    come from one segment_sum, and each entity reads its bucket's total back
    and subtracts its own contribution.
    """
    G = cfg.neighbour_grid
    cell = jnp.clip((state.pos * G).astype(jnp.int32), 0, G - 1)
    bucket = state.colony * (G * G) + cell[:, 1] * G + cell[:, 0]
    n_buckets = cfg.capacity_colonies * G * G

    present = state.alive.astype(jnp.float32)
    count = jax.ops.segment_sum(present, bucket, n_buckets)
    signal = jax.ops.segment_sum(present * state.signal, bucket, n_buckets)

    others = count[bucket] - present
    others_signal = signal[bucket] - present * state.signal
    has_others = state.alive & (others > 0.5)
    crowd = jnp.where(state.alive, jnp.tanh(others / 8.0), 0.0)
    heard = jnp.where(has_others, others_signal / jnp.maximum(others, 1.0), 0.0)
    return crowd, heard


def foreign_nearby(state: WorldState, cfg: Config) -> Array:
    """tanh(count / 4) of living members of OTHER families in the entity's grid cell.

    Same bucket trick as neighbour_stats: count everyone per cell, count each
    family per cell, and subtract the entity's own family.
    """
    G, K = cfg.neighbour_grid, cfg.n_families
    xy = jnp.clip((state.pos * G).astype(jnp.int32), 0, G - 1)
    cell = xy[:, 1] * G + xy[:, 0]
    family = state.col_family[state.colony]
    present = state.alive.astype(jnp.float32)
    everyone = jax.ops.segment_sum(present, cell, G * G)
    by_family = jax.ops.segment_sum(present, family * G * G + cell, K * G * G)
    foreign = everyone[cell] - by_family[family * G * G + cell]
    return jnp.where(state.alive, jnp.tanh(foreign / 4.0), 0.0)


def territory_sense(state: WorldState, points: Array, cfg: Config) -> Array:
    """tanh(own family's scent - everyone else's) at each entity's sample point."""
    own, foreign = scent_at(state, points, state.col_family[state.colony], cfg)
    return jnp.tanh(own - foreign)


def observe(state: WorldState, cfg: Config) -> Array:
    """The (N, 13) input matrix. Column order is fixed, and the predicted
    senses come first so that obs[:, :n_predictions] selects them:

    0 food scent left     1 food scent right
    2 nest scent left     3 nest scent right
    4 crowd               5 signal heard
    6 territory left      7 territory right     8 foreign members nearby
    9 carrying            10 colony energy
    11 memory 0           12 memory 1
    """
    left, right = sensor_points(state.pos, state.heading, cfg)
    own_nest = state.nest[state.colony]
    crowd, heard = neighbour_stats(state, cfg)
    columns = [
        jnp.tanh(food_scent(left, state.food_pos, state.food_alive, cfg)),
        jnp.tanh(food_scent(right, state.food_pos, state.food_alive, cfg)),
        nest_scent(left, own_nest, cfg),
        nest_scent(right, own_nest, cfg),
        crowd,
        heard,
        territory_sense(state, left, cfg),
        territory_sense(state, right, cfg),
        foreign_nearby(state, cfg),
        state.carrying.astype(jnp.float32),
        jnp.tanh(state.energy[state.colony] / 100.0),
        state.memory[:, 0],
        state.memory[:, 1],
    ]
    return jnp.where(state.alive[:, None], jnp.stack(columns, axis=1), 0.0)
