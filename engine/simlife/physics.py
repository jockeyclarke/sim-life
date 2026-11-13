"""Movement, food pickup, delivery and metabolism. Each returns a new state."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .config import Config
from .senses import torus_distance, wrap01
from .state import WorldState


def move(state: WorldState, turn: Array, speed: Array, cfg: Config) -> WorldState:
    """turn in [-1, 1] and speed in [0, 1], as the network emits them."""
    heading = (state.heading + turn * cfg.max_turn) % (2 * jnp.pi)
    direction = jnp.stack([jnp.cos(heading), jnp.sin(heading)], axis=-1)
    pos = wrap01(state.pos + (speed * cfg.max_speed)[:, None] * direction)
    return state._replace(
        heading=jnp.where(state.alive, heading, state.heading),
        pos=jnp.where(state.alive[:, None], pos, state.pos),
    )


def pickup(state: WorldState, cfg: Config) -> WorldState:
    """Members touching food pick it up; a contested pellet goes to one member.

    Each seeking member targets its nearest live pellet. A pellet targeted by
    several members goes to the lowest member index, so food is conserved.
    """
    N, F = cfg.capacity_members, cfg.capacity_food
    dist = torus_distance(state.pos[:, None, :], state.food_pos[None, :, :])  # (N, F)
    dist = jnp.where(state.food_alive[None, :], dist, jnp.inf)
    target = jnp.argmin(dist, axis=1)
    reach = jnp.min(dist, axis=1) <= cfg.pickup_radius
    claims = state.alive & ~state.carrying & reach

    me = jnp.arange(N)
    claimant = jnp.where(claims, me, N)  # N means "no claim"
    winner = jax.ops.segment_min(claimant, target, num_segments=F)  # (F,)
    got = claims & (winner[target] == me)
    taken = winner < N
    return state._replace(
        carrying=state.carrying | got,
        food_alive=state.food_alive & ~taken,
    )


def deliver(state: WorldState, cfg: Config) -> WorldState:
    """A carrying member at its own nest drops its food into the colony pool."""
    own_nest = state.nest[state.colony]
    at_nest = torus_distance(state.pos, own_nest) <= cfg.nest_radius
    drops = state.alive & state.carrying & at_nest
    gained = jax.ops.segment_sum(
        drops.astype(jnp.float32) * cfg.food_energy,
        state.colony,
        num_segments=cfg.capacity_colonies,
    )
    return state._replace(
        carrying=state.carrying & ~drops,
        energy=state.energy + gained,
        delivered_total=state.delivered_total + drops.sum().astype(jnp.int32),
    )


def metabolise(state: WorldState, speed: Array, cfg: Config) -> WorldState:
    """Every living member costs its colony energy; moving faster costs more."""
    cost = state.alive * (cfg.base_cost + cfg.speed_cost * speed**2)
    spent = jax.ops.segment_sum(
        cost, state.colony, num_segments=cfg.capacity_colonies
    )
    return state._replace(energy=state.energy - spent)
