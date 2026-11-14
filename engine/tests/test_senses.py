import jax.numpy as jnp

from helpers import blank, food, member, small_cfg
from simlife.senses import neighbour_stats, observe, wrap_delta

cfg = small_cfg()


def test_wrap_delta_takes_short_way_round():
    d = wrap_delta(jnp.array([0.05, 0.5]), jnp.array([0.95, 0.5]))
    assert jnp.allclose(d, jnp.array([0.10, 0.0]), atol=1e-6)


def test_food_direction_is_positive_left_negative_right_and_zero_ahead():
    # heading 0 (along +x): the left sensor is at larger y
    me = member(blank(cfg), 0, (0.5, 0.5), heading=0.0)
    left = observe(food(me, 0, (0.5, 0.56)), cfg)
    right = observe(food(me, 0, (0.5, 0.44)), cfg)
    ahead = observe(food(me, 0, (0.56, 0.5)), cfg)
    assert left[0, 1] > 0.2 and right[0, 1] < -0.2
    assert jnp.isclose(left[0, 1], -right[0, 1], atol=1e-5)
    assert jnp.abs(ahead[0, 1]) < 1e-5
    assert left[0, 0] > 0 and jnp.isclose(left[0, 0], right[0, 0], atol=1e-5)  # amount ignores side


def test_food_direction_does_not_depend_on_how_much_food_there_is():
    me = member(blank(cfg), 0, (0.5, 0.5), heading=0.0)
    one = observe(food(me, 0, (0.5, 0.56)), cfg)
    two = observe(food(food(me, 0, (0.5, 0.56)), 1, (0.5, 0.56)), cfg)
    assert jnp.isclose(one[0, 1], two[0, 1], atol=1e-5)
    assert two[0, 0] > one[0, 0]  # but more food nearby reads as more


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
    assert obs.shape == (8, 13)
    assert obs[0, 9] == 1 and jnp.allclose(obs[0, 11:], jnp.array([0.3, -0.7]))
    assert jnp.isclose(obs[0, 10], jnp.tanh(10.0 / 100))
    assert bool((obs[5] == 0).all())


# --- stage 3: territory and foreign members --------------------------------------

import dataclasses  # noqa: E402

from simlife.senses import foreign_nearby  # noqa: E402

tcfg = dataclasses.replace(cfg, territory_grid=8, n_families=2)


def scented(s, family, value=5.0):
    return s._replace(territory=jnp.zeros((2, 8, 8)).at[family].set(value))


def test_territory_sense_is_positive_on_own_ground_and_negative_on_foreign():
    s = member(blank(tcfg), 0, (0.5, 0.5), colony=0)  # colony 0 is family 0
    home = observe(scented(s, 0), tcfg)
    away = observe(scented(s, 1), tcfg)
    assert home[0, 6] > 0.9 and home[0, 7] > 0.9
    assert away[0, 6] < -0.9 and away[0, 7] < -0.9
    assert observe(s._replace(territory=jnp.zeros((2, 8, 8))), tcfg)[0, 6] == 0


def test_territory_sense_compares_left_and_right():
    # heading 0 (along +x): the left sensor is at larger y. Mark only the cell above.
    s = member(blank(tcfg), 0, (0.56, 0.495), heading=0.0, colony=0)
    left_cell_y = int((0.495 + 0.02 * jnp.sin(0.6)) * 8)
    t = jnp.zeros((2, 8, 8)).at[0, left_cell_y, 4].set(5.0)
    obs = observe(s._replace(territory=t), tcfg)
    assert obs[0, 6] > obs[0, 7]


def test_foreign_nearby_counts_other_families_only():
    c = dataclasses.replace(cfg, capacity_colonies=3, initial_colonies=3, n_families=2)
    s = blank(c)  # colonies 0 and 2 are family 0; colony 1 is family 1
    s = member(s, 0, (0.51, 0.51), colony=0)
    s = member(s, 1, (0.52, 0.52), colony=0)  # colony-mate
    s = member(s, 2, (0.515, 0.51), colony=2)  # same family, other colony
    s = member(s, 3, (0.512, 0.52), colony=1)  # foreign
    s = member(s, 4, (0.518, 0.515), colony=1)  # foreign
    s = member(s, 5, (0.9, 0.9), colony=1)  # foreign but far away
    near = foreign_nearby(s, c)
    assert jnp.isclose(near[0], jnp.tanh(2 / 4))
    assert jnp.isclose(near[3], jnp.tanh(3 / 4))  # members 0, 1, 2 are foreign to it
    assert near[5] == 0 and near[6] == 0  # far away; dead
    assert jnp.isclose(observe(s, c)[0, 8], near[0])


def test_observation_column_order():
    s = member(blank(tcfg), 0, (0.5, 0.5), carrying=True, memory=[0.3, -0.7])
    obs = observe(s, tcfg)
    assert obs.shape == (8, 13)
    assert obs[0, 9] == 1 and jnp.isclose(obs[0, 10], jnp.tanh(10.0 / 100))
    assert jnp.allclose(obs[0, 11:], jnp.array([0.3, -0.7]))
    assert bool((obs[1:] == 0).all())
