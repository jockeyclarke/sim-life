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
    params: dict  # w1 (N,H,I) b1 (N,H) w2 (N,A,H) b2 (N,A) w3 (N,P,H) b3 (N,P)

    # Lifetime learning, per member.
    birth_params: dict  # each member's weights as they were at birth
    lr: Array  # (N,) f32 learning rate, inherited and mutated
    prev_obs: Array  # (N, I) f32 the observation acted on last tick
    has_prev: Array  # (N,) bool False until a member has lived one full tick
    last_pred: Array  # (N, P) f32 the prediction that was just compared with reality
    pred_error: Array  # (N,) f32 this tick's mean squared prediction error
    pred_error_avg: Array  # (N,) f32 running average of pred_error

    # Colonies, capacity C.
    nest: Array  # (C, 2) f32
    energy: Array  # (C,) f32 shared pool
    col_alive: Array  # (C,) bool
    col_generation: Array  # (C,) i32 splits since a founding colony
    col_family: Array  # (C,) i32 set by region when the grace period ends; daughters inherit it
    col_born: Array  # (C,) i32 tick the colony was founded
    col_foreign: Array  # (C,) bool founded where another family's scent dominated

    # Food, capacity F. Pellet i belongs to patch i % P.
    food_pos: Array  # (F, 2) f32
    food_alive: Array  # (F,) bool
    patch_pos: Array  # (P, 2) f32

    delivered_total: Array  # () i32 pellets delivered to any nest, ever

    # Families, territory and aggression (stage 3).
    family_centre: Array  # (K, 2) f32 each family's regional centre
    founded: Array  # () i32 tick this world began; its grace period runs from here
    territory: Array  # (K, G, G) f32 each family's scent
    aggression: Array  # (N,) f32 aggression output last tick
    striking: Array  # (N,) bool struck someone this tick
    strikes: Array  # (3,) i32 cumulative strikes on own, foreign, border ground
    colonies_died: Array  # (2,) i32 colonies that died, founded at home / on foreign ground
    colony_life_total: Array  # (2,) i32 their summed lifespans, in ticks

    # Switches. They live in the state, as traced values, so flipping one
    # never recompiles and every checkpoint remembers how it was set.
    learning_on: Array  # () bool
    inherit_learned: Array  # () bool  True: children copy learned weights; False: birth weights


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


def nearest_centre(points: Array, centres: Array) -> Array:
    """Index of the nearest centre to each point, on the wrap-around world."""
    delta = (points[:, None, :] - centres[None, :, :] + 0.5) % 1.0 - 0.5
    return jnp.argmin((delta**2).sum(-1), axis=1).astype(jnp.int32)


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
    K = cfg.n_families

    slot = jnp.arange(N)
    alive = slot < n_start
    # Colony c's members sit in contiguous slots. Unused slots point at colony 0.
    colony = jnp.where(alive, slot // cfg.initial_members, 0).astype(jnp.int32)

    # Independent colonies, as in stages 1 and 2: scattered nests, one random
    # network each. Families are regions: each colony belongs to the nearest
    # of K centres. (They are re-formed when the grace period ends.)
    col = jnp.arange(C)
    col_alive = col < cfg.initial_colonies
    k_centre, k_where = jax.random.split(k_nest)
    nest = jax.random.uniform(k_where, (C, 2))
    family_centre = jax.random.uniform(k_centre, (K, 2))
    col_family = nearest_centre(nest, family_centre)

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
        birth_params=params,
        lr=jnp.full(N, cfg.initial_lr),
        prev_obs=jnp.zeros((N, cfg.n_inputs)),
        has_prev=jnp.zeros(N, bool),
        last_pred=jnp.zeros((N, cfg.n_predictions)),
        pred_error=jnp.zeros(N),
        pred_error_avg=jnp.zeros(N),
        nest=nest,
        energy=jnp.where(col_alive, cfg.initial_energy, 0.0),
        col_alive=col_alive,
        col_generation=jnp.zeros(C, jnp.int32),
        food_pos=patch_positions(k_food, patch_pos, cfg),
        food_alive=jnp.ones(F, bool),
        patch_pos=patch_pos,
        delivered_total=jnp.int32(0),
        col_family=col_family,
        family_centre=family_centre,
        founded=jnp.int32(0),
        col_born=jnp.zeros(C, jnp.int32),
        col_foreign=jnp.zeros(C, bool),
        territory=jnp.zeros((K, cfg.territory_grid, cfg.territory_grid)),
        aggression=jnp.zeros(N),
        striking=jnp.zeros(N, bool),
        strikes=jnp.zeros(3, jnp.int32),
        colonies_died=jnp.zeros(2, jnp.int32),
        colony_life_total=jnp.zeros(2, jnp.int32),
        learning_on=jnp.bool_(True),
        inherit_learned=jnp.bool_(True),
    )
