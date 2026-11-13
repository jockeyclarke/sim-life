"""Builders for small hand-placed worlds, and state comparison."""

import jax
import jax.numpy as jnp

from simlife.config import Config
from simlife.state import WorldState, init_world

SMALL = dict(
    capacity_members=8,
    capacity_colonies=2,
    capacity_food=4,
    num_patches=1,
    initial_colonies=2,
    initial_members=2,
)


def small_cfg(**overrides) -> Config:
    return Config(**{**SMALL, **overrides})


def blank(cfg: Config, seed: int = 0) -> WorldState:
    """A world with no members and no food; two live colonies with 10 energy."""
    s = init_world(jax.random.key(seed), cfg)
    N, C, F = cfg.capacity_members, cfg.capacity_colonies, cfg.capacity_food
    nest = s.nest.at[0].set(jnp.array([0.2, 0.2])).at[1].set(jnp.array([0.8, 0.8]))
    return s._replace(
        alive=jnp.zeros(N, bool),
        colony=jnp.zeros(N, jnp.int32),
        age=jnp.zeros(N, jnp.int32),
        lifespan=jnp.full(N, 10_000, jnp.int32),
        food_alive=jnp.zeros(F, bool),
        nest=nest,
        energy=jnp.full(C, 10.0),
        col_alive=jnp.ones(C, bool),
        patch_pos=jnp.full_like(s.patch_pos, 0.5),
    )


def member(s: WorldState, i: int, pos, heading=0.0, colony=0, **fields) -> WorldState:
    """Make member slot i alive at pos; extra keyword fields set s.<field>[i]."""
    updates = dict(
        alive=s.alive.at[i].set(True),
        pos=s.pos.at[i].set(jnp.array(pos, jnp.float32)),
        heading=s.heading.at[i].set(heading),
        colony=s.colony.at[i].set(colony),
    )
    for name, value in fields.items():
        updates[name] = getattr(s, name).at[i].set(jnp.asarray(value))
    return s._replace(**updates)


def food(s: WorldState, i: int, pos) -> WorldState:
    return s._replace(
        food_pos=s.food_pos.at[i].set(jnp.array(pos, jnp.float32)),
        food_alive=s.food_alive.at[i].set(True),
    )


def leaves(s: WorldState) -> list:
    """Every array in the state, with the PRNG key as raw bits."""
    return jax.tree.leaves(s._replace(key=jax.random.key_data(s.key)))


def states_equal(a: WorldState, b: WorldState) -> bool:
    return all(bool((x == y).all()) for x, y in zip(leaves(a), leaves(b), strict=True))
