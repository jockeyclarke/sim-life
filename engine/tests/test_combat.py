import dataclasses

import jax
import jax.numpy as jnp

from helpers import blank, leaves, member, small_cfg
from simlife.combat import strike
from simlife.config import Config
from simlife.state import init_world
from simlife.step import run_ticks

# three colonies: 0 and 2 are family 0, 1 is family 1; 8x8 territory grid
cfg = small_cfg(capacity_colonies=3, initial_colonies=3, n_families=2, territory_grid=8, grace_ticks=0)


def world(territory_family=None):
    s = blank(cfg)
    s = s._replace(energy=jnp.full(3, 10.0), col_alive=jnp.ones(3, bool))
    if territory_family is not None:
        s = s._replace(territory=jnp.zeros((2, 8, 8)).at[territory_family].set(5.0))
    return s


def fighters(s, aggression=(1.0, 0.0), victim_colony=1, gap=0.005, **victim):
    s = member(s, 0, (0.5, 0.5), colony=0, aggression=aggression[0])
    s = member(s, 1, (0.5 + gap, 0.5), colony=victim_colony, aggression=aggression[1], **victim)
    return s


def test_an_aggressive_member_strikes_the_nearest_foreigner():
    s = fighters(world(0))
    s = member(s, 2, (0.512, 0.5), colony=1)  # also foreign, but further away
    s2 = strike(s, cfg)
    assert bool(s2.striking[0]) and not bool(s2.striking[1])
    # the nearer one was struck: colony 1 paid once
    assert jnp.isclose(s2.energy[1], 10.0 - cfg.strike_damage)


def test_out_of_reach_or_calm_means_no_strike():
    assert not bool(strike(fighters(world(0), gap=0.05), cfg).striking.any())
    assert not bool(strike(fighters(world(0), aggression=(0.4, 0.0)), cfg).striking.any())


def test_the_same_family_never_strikes():
    s = fighters(world(0), victim_colony=2)  # colony 2 is family 0 too
    s2 = strike(s, cfg)
    assert not bool(s2.striking.any()) and bool((s2.energy == s.energy).all())


def test_strike_effects_and_costs():
    s = fighters(world(0), carrying=True)
    s2 = strike(s, cfg)
    assert not bool(s2.carrying[1])  # the food is knocked loose and lost
    assert jnp.isclose(s2.energy[1], 10.0 - cfg.strike_damage * 1.0)  # on the striker's own ground
    assert jnp.isclose(s2.energy[0], 10.0 - cfg.strike_cost)
    assert bool((s2.alive == s.alive).all())


def test_strikes_are_stronger_at_home_than_abroad():
    at_home = 10.0 - strike(fighters(world(0)), cfg).energy[1]
    abroad = 10.0 - strike(fighters(world(1)), cfg).energy[1]
    neutral = 10.0 - strike(fighters(world(None)), cfg).energy[1]
    assert jnp.isclose(at_home, cfg.strike_damage) and jnp.isclose(abroad, 0.0, atol=1e-6)
    assert jnp.isclose(neutral, cfg.strike_damage / 2)


def test_strikes_are_classified_by_ground():
    own = strike(fighters(world(0)), cfg).strikes
    foreign = strike(fighters(world(1)), cfg).strikes
    border = strike(fighters(world(None)), cfg).strikes
    assert own.tolist() == [1, 0, 0] and foreign.tolist() == [0, 1, 0] and border.tolist() == [0, 0, 1]


def test_mutual_strikes_both_count():
    s2 = strike(fighters(world(None), aggression=(1.0, 1.0)), cfg)
    assert bool(s2.striking[0]) and bool(s2.striking[1]) and int(s2.strikes.sum()) == 2


def test_nobody_dies_from_a_strike():
    s = fighters(world(0))._replace(energy=jnp.array([0.01, 0.01, 0.01]))
    s2 = strike(s, cfg)
    assert bool((s2.alive == s.alive).all()) and bool((s2.col_alive == s.col_alive).all())


def test_everyone_maximally_aggressive_stays_finite():
    c = Config(grace_ticks=0)  # six families of three nests: foreigners do meet
    s = init_world(jax.random.key(0), c)
    # aggression output = sigmoid(w2[5] . h + b2[5]); a huge bias pins it at 1
    params = dict(s.params)
    params["b2"] = params["b2"].at[:, 5].set(50.0)
    params["w2"] = params["w2"].at[:, 5, :].set(0.0)
    s = run_ticks(s._replace(params=params, birth_params=params), c, 3000)
    for leaf in leaves(s):
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            assert bool(jnp.isfinite(leaf).all())
    assert int(s.strikes.sum()) > 0


def test_no_strikes_during_the_grace_period():
    c = dataclasses.replace(cfg, grace_ticks=100)
    s = fighters(world(0), carrying=True)._replace(tick=jnp.int32(99))
    s2 = strike(s, c)
    assert not bool(s2.striking.any()) and bool(s2.carrying[1]) and bool((s2.energy == s.energy).all())
    assert bool(strike(s._replace(tick=jnp.int32(100)), c).striking[0])
