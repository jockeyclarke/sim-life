import jax
import jax.numpy as jnp

from helpers import leaves, states_equal
from simlife.config import Config
from simlife.state import init_world
from simlife.step import run_ticks, step

cfg = Config(
    capacity_members=256, capacity_colonies=16, capacity_food=64, initial_colonies=4
)


def world(seed=7):
    return init_world(jax.random.key(seed), cfg)


def test_same_seed_same_run():
    assert states_equal(run_ticks(world(), cfg, 200), run_ticks(world(), cfg, 200))


def test_chunking_does_not_change_the_run():
    s = world()
    whole = run_ticks(s, cfg, 200)
    halves = run_ticks(run_ticks(s, cfg, 100), cfg, 100)
    assert states_equal(whole, halves)


def test_different_seed_different_run():
    assert not states_equal(run_ticks(world(7), cfg, 50), run_ticks(world(8), cfg, 50))


def test_tick_counter_and_shapes_preserved():
    s = world()
    s2 = run_ticks(s, cfg, 37)
    assert int(s2.tick) == 37
    assert [a.shape for a in leaves(s2)] == [a.shape for a in leaves(s)]
    assert [a.dtype for a in leaves(s2)] == [a.dtype for a in leaves(s)]


def test_members_actually_move_and_think():
    s = world()
    s2 = run_ticks(s, cfg, 10)
    both = s.alive & s2.alive
    assert bool((s2.pos[both] != s.pos[both]).any())
    assert bool((s2.memory[both] != 0).any())


def test_long_run_stays_finite_and_in_bounds():
    s = run_ticks(world(0), cfg, 5000)
    for leaf in leaves(s):
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            assert bool(jnp.isfinite(leaf).all())
    assert bool(((s.pos >= 0) & (s.pos < 1)).all())
    assert bool(((s.food_pos >= 0) & (s.food_pos < 1)).all())
    assert bool(((s.nest >= 0) & (s.nest < 1)).all())


def test_energy_ledger_balances_over_one_tick():
    one = jax.jit(lambda s: step(s, cfg))
    s = run_ticks(world(3), cfg, 300)
    checked = 0
    for _ in range(200):
        s2 = one(s)
        if bool((s2.col_alive | ~s.col_alive).all()):  # no colony died this tick
            members = int(s.alive.sum())
            births = int((s2.alive & (s2.age == 0)).sum())
            delivered = int(s2.delivered_total - s.delivered_total)
            income = cfg.food_energy * delivered - cfg.birth_cost * births
            change = float(s2.energy.sum() - s.energy.sum())
            spent = income - change
            assert members * cfg.base_cost - 1e-3 <= spent
            assert spent <= members * (cfg.base_cost + cfg.speed_cost) + 1e-3
            checked += 1
        s = s2
    assert checked > 100


def test_step_compiles_once_for_any_state():
    traces = []

    def counted(s):
        traces.append(1)
        return step(s, cfg)

    jitted = jax.jit(counted)
    jitted(world(1))
    jitted(world(2))
    assert len(traces) == 1
