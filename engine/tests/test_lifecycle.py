import jax
import jax.numpy as jnp

from helpers import blank, member, small_cfg, states_equal
from simlife.lifecycle import age_and_die, grow, kill_dead_colonies

cfg = small_cfg()
N = cfg.capacity_members
MEMBER_FIELDS = [
    "pos", "heading", "alive", "colony", "carrying", "age", "lifespan",
    "memory", "signal", "generation",
]


def rich_and_poor():
    """Colony 0: two clonal members, 100 energy. Colony 1: one member, 1 energy."""
    s = member(member(blank(cfg), 0, (0.3, 0.3)), 1, (0.4, 0.4))
    s = member(s, 2, (0.7, 0.7), colony=1)
    return s._replace(energy=jnp.array([100.0, 1.0]))


def test_member_dies_at_lifespan_and_drops_food():
    s = member(blank(cfg), 0, (0.5, 0.5), age=99, lifespan=100, carrying=True)
    s = member(s, 1, (0.5, 0.5), age=5, lifespan=100)
    s2 = age_and_die(s, cfg)
    assert not bool(s2.alive[0]) and not bool(s2.carrying[0])
    assert bool(s2.alive[1]) and int(s2.age[1]) == 6


def test_colony_with_no_energy_dies_with_all_members():
    s = rich_and_poor()._replace(energy=jnp.array([0.0, 5.0]))
    s2 = kill_dead_colonies(s)
    assert not bool(s2.col_alive[0]) and bool(s2.col_alive[1])
    assert not bool(s2.alive[0]) and not bool(s2.alive[1]) and bool(s2.alive[2])


def test_colony_with_no_members_dies():
    s = member(blank(cfg), 0, (0.5, 0.5), colony=0)  # colony 1 has energy, no members
    s2 = kill_dead_colonies(s)
    assert bool(s2.col_alive[0]) and not bool(s2.col_alive[1])
    assert s2.energy[1] == 0


def test_grow_adds_one_mutated_child_and_charges_energy():
    s = rich_and_poor()
    s2 = grow(s, cfg, jax.random.key(0))
    assert int(s2.alive.sum()) == int(s.alive.sum()) + 1
    child = int(jnp.argmax(s2.alive & ~s.alive))
    assert int(s2.colony[child]) == 0 and int(s2.age[child]) == 0
    assert int(s2.generation[child]) == 1 and not bool(s2.carrying[child])
    assert jnp.isclose(s2.energy[0], s.energy[0] - cfg.birth_cost)
    assert s2.energy[1] == s.energy[1]
    d = s2.params["w1"][child] - s.params["w1"][0]  # colony 0 is clonal here
    assert 0 < float(jnp.abs(d).max()) < 10 * cfg.mutation_sigma
    parent_positions = s.pos[:2]
    assert bool((s2.pos[child] == parent_positions).all(axis=1).any())


def test_grow_never_overwrites_a_live_member():
    s = rich_and_poor()
    s2 = grow(s, cfg, jax.random.key(0))
    for name in MEMBER_FIELDS:
        assert bool((getattr(s2, name)[s.alive] == getattr(s, name)[s.alive]).all()), name
    for name in s.params:
        assert bool((s2.params[name][s.alive] == s.params[name][s.alive]).all()), name


def test_grow_when_members_full_does_nothing():
    s = blank(cfg)
    s = s._replace(alive=jnp.ones(N, bool), energy=jnp.array([100.0, 100.0]))
    assert states_equal(grow(s, cfg, jax.random.key(0)), s)


def test_grow_two_colonies_one_free_slot():
    s = blank(cfg)
    alive = jnp.ones(N, bool).at[7].set(False)
    colony = jnp.array([0, 0, 0, 0, 1, 1, 1, 0], jnp.int32)
    s = s._replace(alive=alive, colony=colony, energy=jnp.array([100.0, 100.0]))
    s2 = grow(s, cfg, jax.random.key(0))
    assert int(s2.alive.sum()) == N
    assert jnp.isclose(s2.energy.sum(), 200.0 - cfg.birth_cost)


def test_poor_colony_does_not_grow():
    s = rich_and_poor()._replace(energy=jnp.array([1.0, 1.0]))
    assert states_equal(grow(s, cfg, jax.random.key(0)), s)
