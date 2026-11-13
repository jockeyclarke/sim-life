"""Births and deaths. There is no fitness function here: only energy and age."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .config import Config
from .senses import wrap01
from .state import WorldState, patch_positions, random_lifespan


def member_counts(state: WorldState, cfg: Config) -> Array:
    """Living members per colony, shape (C,)."""
    return jax.ops.segment_sum(
        state.alive.astype(jnp.int32), state.colony, num_segments=cfg.capacity_colonies
    )


def age_and_die(state: WorldState, cfg: Config) -> WorldState:
    age = state.age + state.alive.astype(jnp.int32)
    alive = state.alive & (age < state.lifespan)
    return state._replace(age=age, alive=alive, carrying=state.carrying & alive)


def kill_dead_colonies(state: WorldState) -> WorldState:
    """A colony out of energy, or out of members, dies with everyone in it."""
    C = state.col_alive.shape[0]
    counts = jax.ops.segment_sum(
        state.alive.astype(jnp.int32), state.colony, num_segments=C
    )
    col_alive = state.col_alive & (state.energy > 0) & (counts > 0)
    alive = state.alive & col_alive[state.colony]
    return state._replace(
        col_alive=col_alive,
        energy=jnp.where(col_alive, state.energy, 0.0),
        alive=alive,
        carrying=state.carrying & alive,
    )


def grow(state: WorldState, cfg: Config, key: Array) -> WorldState:
    """At most one birth per colony: a random member divides, with mutation.

    The child is a copy of the parent's network plus small noise. This is the
    only place weights change in stage 1, and nothing here judges the parent:
    it is chosen uniformly at random. Selection happens only because colonies
    that gather more energy divide more and starve less.
    """
    N, C = cfg.capacity_members, cfg.capacity_colonies
    k_parent, k_mutate, k_heading, k_life = jax.random.split(key, 4)

    counts = member_counts(state, cfg)
    surplus = state.energy > cfg.birth_cost + cfg.reserve_per_member * counts
    want = state.col_alive & (counts > 0) & surplus  # (C,)

    # Hand out free member slots to the colonies that want one. Shapes are
    # fixed, so "allocate" means: sort slot indices with free ones first, give
    # the k-th wanting colony the k-th free slot, and send the rest to index N,
    # which the scatter below drops.
    free = jnp.argsort(state.alive, stable=True)
    n_free = (~state.alive).sum()
    rank = jnp.cumsum(want) - 1
    ok = want & (rank < n_free)
    slot = jnp.where(ok, free[jnp.clip(rank, 0, N - 1)], N)

    # One uniformly random living member of each colony.
    score = jax.random.uniform(k_parent, (N,))
    mine = state.alive[None, :] & (state.colony[None, :] == jnp.arange(C)[:, None])
    parent = jnp.argmax(jnp.where(mine, score[None, :], -1.0), axis=1)  # (C,)

    def put(field: Array, value: Array) -> Array:
        return field.at[slot].set(value, mode="drop")

    names = sorted(state.params)
    noise_keys = dict(zip(names, jax.random.split(k_mutate, len(names))))
    params = {
        name: put(
            state.params[name],
            state.params[name][parent]
            + cfg.mutation_sigma
            * jax.random.normal(noise_keys[name], state.params[name][parent].shape),
        )
        for name in names
    }

    return state._replace(
        alive=put(state.alive, jnp.ones(C, bool)),
        pos=put(state.pos, state.pos[parent]),
        heading=put(
            state.heading, jax.random.uniform(k_heading, (C,), maxval=2 * jnp.pi)
        ),
        colony=put(state.colony, jnp.arange(C, dtype=jnp.int32)),
        carrying=put(state.carrying, jnp.zeros(C, bool)),
        age=put(state.age, jnp.zeros(C, jnp.int32)),
        lifespan=put(state.lifespan, random_lifespan(k_life, (C,), cfg)),
        memory=put(state.memory, jnp.zeros((C, 2))),
        signal=put(state.signal, jnp.zeros(C)),
        generation=put(state.generation, state.generation[parent] + 1),
        params=params,
        energy=state.energy - jnp.where(ok, cfg.birth_cost, 0.0),
    )
