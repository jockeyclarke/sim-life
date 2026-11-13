"""What an entity can sense. Everything is local and relative to its heading."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .config import Config
from .state import WorldState


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
    zeros = jnp.zeros_like(crowd)
    columns = [
        jnp.tanh(food_scent(left, state.food_pos, state.food_alive, cfg)),
        jnp.tanh(food_scent(right, state.food_pos, state.food_alive, cfg)),
        nest_scent(left, own_nest, cfg),
        nest_scent(right, own_nest, cfg),
        crowd,
        heard,
        zeros,  # territory left (filled in by the territory task)
        zeros,  # territory right
        zeros,  # foreign members nearby
        state.carrying.astype(jnp.float32),
        jnp.tanh(state.energy[state.colony] / 100.0),
        state.memory[:, 0],
        state.memory[:, 1],
    ]
    return jnp.where(state.alive[:, None], jnp.stack(columns, axis=1), 0.0)
