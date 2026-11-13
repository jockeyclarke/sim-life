import dataclasses

import jax
import jax.numpy as jnp

from helpers import leaves, states_equal
from simlife.config import Config
from simlife.state import init_world
from simlife.step import run_ticks, step

base = Config(
    capacity_members=256, capacity_colonies=16, capacity_food=64, initial_colonies=4
)
# no births and no deaths, so any change in a member's weights is learning
cfg = dataclasses.replace(
    base, birth_cost=1e9, initial_energy=1e6, lifespan_mean=1_000_000, lifespan_jitter=0
)
N = cfg.capacity_members


def world(seed=7, **flags):
    s = init_world(jax.random.key(seed), cfg)
    return s._replace(**{k: jnp.bool_(v) for k, v in flags.items()})


def unchanged(a, b, name):
    return bool((a.params[name] == b.params[name]).all())


def test_learning_changes_hidden_and_prediction_weights_only():
    s0 = world()
    s1 = run_ticks(s0, cfg, 20)
    assert int(s1.generation.max()) == 0 and bool((s1.alive == s0.alive).all())
    assert unchanged(s0, s1, "w2") and unchanged(s0, s1, "b2")
    live = s0.alive
    assert bool((s1.params["w1"][live] != s0.params["w1"][live]).any())
    assert bool((s1.params["w3"][live] != s0.params["w3"][live]).any())
    assert bool((s1.params["w1"][~live] == s0.params["w1"][~live]).all())  # dead slots untouched
    assert all(bool((s1.birth_params[k] == s0.birth_params[k]).all()) for k in s0.params)


def test_learning_off_leaves_all_weights_alone_but_still_measures_error():
    s0 = world(learning_on=False)
    s1 = run_ticks(s0, cfg, 20)
    assert all(unchanged(s0, s1, k) for k in s0.params)
    assert bool((s1.pred_error[s1.alive] > 0).any())
    assert bool((s1.pred_error_avg[s1.alive] > 0).any())


def test_no_learning_on_a_members_first_tick():
    s0 = world()
    s1 = jax.jit(lambda s: step(s, cfg))(s0)
    assert all(unchanged(s0, s1, k) for k in s0.params)  # has_prev was False
    assert bool(s1.has_prev[s1.alive].all()) and not bool(s1.has_prev[~s1.alive].any())
    assert bool((s1.pred_error == 0).all())


def error_ratio(learning: bool) -> float:
    s = world(seed=3, learning_on=learning)
    early = run_ticks(s, cfg, 60)
    late = run_ticks(early, cfg, 340)
    a = late.alive
    return float(late.pred_error_avg[a].mean() / early.pred_error_avg[a].mean())


def test_prediction_error_falls_over_a_life():
    assert error_ratio(True) < 0.6
    assert error_ratio(False) > 0.8


def test_last_pred_is_the_prediction_that_was_scored():
    s = run_ticks(world(), cfg, 5)
    s1 = jax.jit(lambda s: step(s, cfg))(s)
    a = s1.alive
    actual = s1.prev_obs[:, :6]  # what was sensed this tick
    scored = jnp.mean((s1.last_pred - actual) ** 2, axis=1)
    assert jnp.allclose(s1.pred_error[a], scored[a], rtol=1e-5, atol=1e-7)


def test_maximum_learning_rate_long_run_stays_finite():
    s = init_world(jax.random.key(0), base)  # births and deaths on
    s = run_ticks(s._replace(lr=jnp.full(N, base.max_lr)), base, 5000)
    for leaf in leaves(s):
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            assert bool(jnp.isfinite(leaf).all())
    garden = run_ticks(world()._replace(lr=jnp.full(N, cfg.max_lr)), cfg, 5000)
    for leaf in leaves(garden):
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            assert bool(jnp.isfinite(leaf).all())


def test_learning_run_is_deterministic_and_chunk_independent():
    s = world()
    assert states_equal(run_ticks(s, cfg, 200), run_ticks(run_ticks(s, cfg, 100), cfg, 100))
    assert not states_equal(run_ticks(s, cfg, 50), run_ticks(world(learning_on=False), cfg, 50))
