import jax.numpy as jnp

from helpers import blank, food, member, small_cfg
from simlife.senses import neighbour_stats, observe, wrap_delta

cfg = small_cfg()


def test_wrap_delta_takes_short_way_round():
    d = wrap_delta(jnp.array([0.05, 0.5]), jnp.array([0.95, 0.5]))
    assert jnp.allclose(d, jnp.array([0.10, 0.0]), atol=1e-6)


def test_food_on_the_left_smells_stronger_on_the_left():
    s = food(member(blank(cfg), 0, (0.5, 0.5), heading=0.0), 0, (0.5, 0.56))
    obs = observe(s, cfg)
    assert obs[0, 0] > obs[0, 1] > 0


def test_dead_food_has_no_scent():
    s = food(member(blank(cfg), 0, (0.5, 0.5)), 0, (0.5, 0.56))
    s = s._replace(food_alive=jnp.zeros_like(s.food_alive))
    obs = observe(s, cfg)
    assert obs[0, 0] == 0 and obs[0, 1] == 0


def test_nest_scent_comes_from_own_nest_only():
    # colony 0's nest is at (0.2, 0.2), colony 1's at (0.8, 0.8)
    s = member(blank(cfg), 0, (0.8, 0.8), colony=0)
    s = member(s, 1, (0.2, 0.2), colony=0)
    obs = observe(s, cfg)
    assert obs[0, 2] < 0.1 and obs[0, 3] < 0.1  # on the other colony's nest
    assert obs[1, 2] > 0.8 and obs[1, 3] > 0.8  # on its own


def test_neighbours_same_colony_same_cell_only():
    s = member(blank(cfg), 0, (0.51, 0.51), colony=0)
    s = member(s, 1, (0.52, 0.52), colony=0, signal=0.8)
    s = member(s, 2, (0.515, 0.515), colony=1, signal=0.3)
    s = member(s, 3, (0.9, 0.1), colony=0, signal=-1.0)  # same colony, far away
    crowd, heard = neighbour_stats(s, cfg)
    assert jnp.isclose(crowd[0], jnp.tanh(1 / 8)) and jnp.isclose(heard[0], 0.8)
    assert crowd[2] == 0 and heard[2] == 0


def test_observation_passes_through_carrying_and_memory_and_zeroes_dead():
    s = member(blank(cfg), 0, (0.5, 0.5), carrying=True, memory=[0.3, -0.7])
    s = s._replace(memory=s.memory.at[5].set(jnp.array([0.9, 0.9])))  # slot 5 is dead
    obs = observe(s, cfg)
    assert obs.shape == (8, 10)
    assert obs[0, 6] == 1 and jnp.allclose(obs[0, 8:], jnp.array([0.3, -0.7]))
    assert jnp.isclose(obs[0, 7], jnp.tanh(10.0 / 100))
    assert bool((obs[5] == 0).all())
