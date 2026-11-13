"""Births and deaths. There is no fitness function here: only energy and age."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .config import Config
from .senses import wrap01
from .state import WorldState, nearest_centre, patch_positions, random_lifespan
from .territory import home_share, scent_at


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

    # Record each death by where the colony was founded: at home or on foreign ground.
    died = state.col_alive & ~col_alive
    ground = state.col_foreign.astype(jnp.int32)
    life = jnp.where(died, state.tick - state.col_born, 0)
    return state._replace(
        col_alive=col_alive,
        energy=jnp.where(col_alive, state.energy, 0.0),
        alive=alive,
        colonies_died=state.colonies_died + jax.ops.segment_sum(died.astype(jnp.int32), ground, 2),
        colony_life_total=state.colony_life_total + jax.ops.segment_sum(life, ground, 2),
        carrying=state.carrying & alive,
    )


def grow(state: WorldState, cfg: Config, key: Array) -> WorldState:
    """At most one birth per colony, from two random parents in that colony.

    Each hidden unit, with all its connections in and out, comes whole from
    one parent or the other; then every weight gets small noise. This is the
    only place the action weights ever change, and nothing here judges the
    parents: they are chosen uniformly at random. Selection happens only
    because colonies that gather more energy divide more and starve less.
    """
    N, C = cfg.capacity_members, cfg.capacity_colonies
    k_parent, k_other, k_mutate, k_heading, k_life, k_lr, k_cross = jax.random.split(key, 7)

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

    # Two uniformly random living members of each colony, drawn independently
    # (a colony of one gets the same member twice).
    mine = state.alive[None, :] & (state.colony[None, :] == jnp.arange(C)[:, None])

    def pick(k: Array) -> Array:
        score = jax.random.uniform(k, (N,))
        return jnp.argmax(jnp.where(mine, score[None, :], -1.0), axis=1)  # (C,)

    parent, other = pick(k_parent), pick(k_other)

    def put(field: Array, value: Array) -> Array:
        return field.at[slot].set(value, mode="drop")

    # What a child copies is the inheritance switch: the parent's weights as
    # they are now (including what it learned in life), or as they were at its
    # own birth (learning discarded). Either way the child gets small noise on
    # top, and those become the child's own birth weights.
    def source(name: str, who: Array) -> Array:
        return jnp.where(
            state.inherit_learned, state.params[name][who], state.birth_params[name][who]
        )

    # Crossover by whole hidden unit: for unit h the child takes row h of w1,
    # entry h of b1, and column h of w2 and w3 from the same parent. Output
    # biases are picked one by one. Nothing is ever averaged.
    k_units, k_b2, k_b3 = jax.random.split(k_cross, 3)
    from_a = jax.random.bernoulli(k_units, 0.5, (C, cfg.n_hidden))
    choose = {
        "w1": from_a[:, :, None],
        "b1": from_a,
        "w2": from_a[:, None, :],
        "w3": from_a[:, None, :],
        "b2": jax.random.bernoulli(k_b2, 0.5, (C, cfg.n_actions)),
        "b3": jax.random.bernoulli(k_b3, 0.5, (C, cfg.n_predictions)),
    }
    names = sorted(state.params)
    noise_keys = dict(zip(names, jax.random.split(k_mutate, len(names))))
    child = {}
    for name in names:
        mixed = jnp.where(choose[name], source(name, parent), source(name, other))
        noise = jax.random.normal(noise_keys[name], mixed.shape)
        child[name] = mixed + cfg.mutation_sigma * noise

    # The learning rate is inherited too, nudged up or down by a few percent,
    # so evolution can turn learning up or down, or off.
    nudge = jnp.exp(cfg.lr_mutation_sigma * jax.random.normal(k_lr, (C,)))
    child_lr = jnp.clip(state.lr[parent] * nudge, 0.0, cfg.max_lr)

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
        generation=put(
            state.generation, jnp.maximum(state.generation[parent], state.generation[other]) + 1
        ),
        params={name: put(state.params[name], child[name]) for name in names},
        birth_params={name: put(state.birth_params[name], child[name]) for name in names},
        lr=put(state.lr, child_lr),
        # a newborn has no previous observation, so it does not learn on its first tick
        has_prev=put(state.has_prev, jnp.zeros(C, bool)),
        prev_obs=put(state.prev_obs, jnp.zeros((C, cfg.n_inputs))),
        last_pred=put(state.last_pred, jnp.zeros((C, cfg.n_predictions))),
        pred_error=put(state.pred_error, jnp.zeros(C)),
        pred_error_avg=put(state.pred_error_avg, jnp.zeros(C)),
        energy=state.energy - jnp.where(ok, cfg.birth_cost, 0.0),
    )


def split(state: WorldState, cfg: Config, key: Array) -> WorldState:
    """At most one colony splits per tick: half its members found a daughter.

    Splitting moves members between colonies and never touches their weights.
    """
    N = cfg.capacity_members
    k_pick, k_dir = jax.random.split(key)

    counts = member_counts(state, cfg)
    eligible = state.col_alive & (counts >= cfg.split_size)
    do = eligible.any() & (~state.col_alive).any()
    parent = jnp.argmax(eligible)  # lowest-index eligible colony
    daughter = jnp.argmax(~state.col_alive)  # lowest free colony slot

    # Rank the parent's members in random order and move the first half, so
    # neither side can end up empty.
    mine = state.alive & (state.colony == parent)
    score = jnp.where(mine, jax.random.uniform(k_pick, (N,)), 2.0)
    rank = jnp.zeros(N, jnp.int32).at[jnp.argsort(score)].set(jnp.arange(N))
    moving = do & mine & (rank < counts[parent] // 2)

    angle = jax.random.uniform(k_dir, (), maxval=2 * jnp.pi)
    offset = cfg.split_distance * jnp.stack([jnp.cos(angle), jnp.sin(angle)])
    half = state.energy[parent] / 2

    def on_daughter(field: Array, value: Array) -> Array:
        return jnp.where(do, field.at[daughter].set(value), field)

    energy = on_daughter(state.energy.at[parent].set(half), half)
    new_nest = wrap01(state.nest[parent] + offset)
    family = state.col_family[parent]
    own, foreign = scent_at(state, new_nest[None, :], family[None], cfg)
    founded_foreign = home_share(own, foreign)[0] < 0.5
    return state._replace(
        colony=jnp.where(moving, daughter, state.colony).astype(jnp.int32),
        nest=on_daughter(state.nest, new_nest),
        col_family=on_daughter(state.col_family, family),
        col_born=on_daughter(state.col_born, state.tick),
        col_foreign=on_daughter(state.col_foreign, founded_foreign),
        energy=jnp.where(do, energy, state.energy),
        col_alive=on_daughter(state.col_alive, True),
        col_generation=on_daughter(
            state.col_generation, state.col_generation[parent] + 1
        ),
    )


def regrow_food(state: WorldState, cfg: Config, key: Array) -> WorldState:
    """Each eaten pellet may reappear somewhere in its patch. This rate is the scarcity."""
    k_which, k_where = jax.random.split(key)
    chance = jax.random.uniform(k_which, (cfg.capacity_food,))
    revive = ~state.food_alive & (chance < cfg.food_regrow_prob)
    fresh = patch_positions(k_where, state.patch_pos, cfg)
    return state._replace(
        food_pos=jnp.where(revive[:, None], fresh, state.food_pos),
        food_alive=state.food_alive | revive,
    )


def form_families(state: WorldState, cfg: Config) -> WorldState:
    """On the tick the grace period ends, divide the world into families by region.

    Every colony joins the family of its nest's nearest regional centre, and
    the territory scent is cleared so the contest starts fresh. On any other
    tick nothing changes.
    """
    now = state.tick == state.founded + cfg.grace_ticks
    return state._replace(
        col_family=jnp.where(now, nearest_centre(state.nest, state.family_centre), state.col_family),
        territory=jnp.where(now, 0.0, state.territory),
    )
