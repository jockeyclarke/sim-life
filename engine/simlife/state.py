"""The whole world as one record of fixed-shape arrays."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from .brain import init_params
from .config import Config


class WorldState(NamedTuple):
    key: Array  # PRNG key; split every tick
    tick: Array  # () i32

    # Members, capacity N. Slots with alive == False hold stale values.
    pos: Array  # (N, 2) f32 in [0, 1)
    heading: Array  # (N,) f32 radians
    alive: Array  # (N,) bool
    colony: Array  # (N,) i32 colony slot
    carrying: Array  # (N,) bool
    age: Array  # (N,) i32 ticks
    lifespan: Array  # (N,) i32 ticks
    memory: Array  # (N, 2) f32, written by the network last tick
    signal: Array  # (N,) f32, emitted by the network last tick
    generation: Array  # (N,) i32 divisions since the founding network
    params: dict  # w1 (N,H,I)  b1 (N,H)  w2 (N,A,H)  b2 (N,A)

    # Colonies, capacity C.
    nest: Array  # (C, 2) f32
    energy: Array  # (C,) f32 shared pool
    col_alive: Array  # (C,) bool
    col_generation: Array  # (C,) i32 splits since a founding colony

    # Food, capacity F. Pellet i belongs to patch i % P.
    food_pos: Array  # (F, 2) f32
    food_alive: Array  # (F,) bool
    patch_pos: Array  # (P, 2) f32

    delivered_total: Array  # () i32 pellets delivered to any nest, ever


def random_lifespan(key: Array, shape: tuple, cfg: Config) -> Array:
    jitter = jax.random.randint(
        key, shape, -cfg.lifespan_jitter, cfg.lifespan_jitter + 1
    )
    return (cfg.lifespan_mean + jitter).astype(jnp.int32)


def patch_positions(key: Array, patch_pos: Array, cfg: Config) -> Array:
    """A uniform random point inside each pellet's patch, for all F pellets."""
    F = cfg.capacity_food
    ka, kr = jax.random.split(key)
    angle = jax.random.uniform(ka, (F,), maxval=2 * jnp.pi)
    # sqrt makes the points uniform over the disc's area, not bunched at the centre
    radius = cfg.patch_radius * jnp.sqrt(jax.random.uniform(kr, (F,)))
    centre = patch_pos[jnp.arange(F) % cfg.num_patches]
    offset = radius[:, None] * jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=1)
    p = (centre + offset) % 1.0
    return jnp.where(p >= 1.0, 0.0, p)  # see senses.wrap01


def init_world(key: Array, cfg: Config) -> WorldState:
    N, C, F, P = (
        cfg.capacity_members,
        cfg.capacity_colonies,
        cfg.capacity_food,
        cfg.num_patches,
    )
    n_start = cfg.initial_colonies * cfg.initial_members
    if cfg.initial_colonies > C or n_start > N:
        raise ValueError("initial colonies and members exceed the capacities")

    (k_state, k_nest, k_patch, k_food, k_net, k_head, k_life, k_age) = (
        jax.random.split(key, 8)
    )

    slot = jnp.arange(N)
    alive = slot < n_start
    # Colony c's members sit in contiguous slots. Unused slots point at colony 0.
    colony = jnp.where(alive, slot // cfg.initial_members, 0).astype(jnp.int32)

    nest = jax.random.uniform(k_nest, (C, 2))
    col_alive = jnp.arange(C) < cfg.initial_colonies

    # One random network per colony, copied to every member: a colony starts clonal.
    founders = jax.vmap(lambda k: init_params(k, cfg))(jax.random.split(k_net, C))
    params = jax.tree.map(lambda a: a[colony], founders)

    lifespan = random_lifespan(k_life, (N,), cfg)
    # Stagger starting ages so a founding cohort doesn't all die on one tick.
    age = (jax.random.uniform(k_age, (N,)) * (lifespan // 2)).astype(jnp.int32)

    patch_pos = jax.random.uniform(k_patch, (P, 2))

    return WorldState(
        key=k_state,
        tick=jnp.int32(0),
        pos=nest[colony],
        heading=jax.random.uniform(k_head, (N,), maxval=2 * jnp.pi),
        alive=alive,
        colony=colony,
        carrying=jnp.zeros(N, bool),
        age=age,
        lifespan=lifespan,
        memory=jnp.zeros((N, 2)),
        signal=jnp.zeros(N),
        generation=jnp.zeros(N, jnp.int32),
        params=params,
        nest=nest,
        energy=jnp.where(col_alive, cfg.initial_energy, 0.0),
        col_alive=col_alive,
        col_generation=jnp.zeros(C, jnp.int32),
        food_pos=patch_positions(k_food, patch_pos, cfg),
        food_alive=jnp.ones(F, bool),
        patch_pos=patch_pos,
        delivered_total=jnp.int32(0),
    )
