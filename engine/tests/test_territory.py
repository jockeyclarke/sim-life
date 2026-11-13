import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from helpers import blank, member, small_cfg
from simlife.config import Config
from simlife.physics import metabolise
from simlife.state import init_world
from simlife.step import run_ticks
from simlife.territory import cell_of, home_share, owner_grid, scent_at, update_territory

# a small grid so cells are easy to reason about; nothing fades or spreads unless asked
cfg = small_cfg(territory_grid=8, scent_fade=1.0, scent_spread=0.0, n_families=2)
G = 8


def cell(x, y):
    return int(y * G + x)


def test_cell_of_maps_points_to_flat_grid_indices():
    pts = jnp.array([[0.0, 0.0], [0.99, 0.0], [0.0, 0.99], [0.5, 0.26]])
    assert cell_of(pts, cfg).tolist() == [0, 7, 56, cell(4, 2)]


def test_marks_accumulate_where_members_are_and_nests_mark_too():
    s = member(blank(cfg), 0, (0.51, 0.51), colony=0)  # colony 0 is family 0
    s = update_territory(s, cfg)
    flat = s.territory.reshape(2, -1)
    assert jnp.isclose(flat[0, cell(4, 4)], cfg.mark_rate)
    nest_cell = int(cell_of(s.nest[:1], cfg)[0])
    assert jnp.isclose(flat[0, nest_cell], cfg.nest_mark)
    assert jnp.isclose(flat[1, int(cell_of(s.nest[1:2], cfg)[0])], cfg.nest_mark)  # colony 1, family 1
    assert jnp.isclose(flat.sum(), cfg.mark_rate + 2 * cfg.nest_mark)


def test_scent_fades():
    c = dataclasses.replace(cfg, scent_fade=0.5, mark_rate=0.0, nest_mark=0.0)
    s = blank(c)._replace(territory=jnp.ones((2, G, G)))
    s = update_territory(s, c)
    assert bool(jnp.allclose(s.territory, 0.5))


def test_spread_moves_scent_to_neighbours_and_wraps_at_edges():
    c = dataclasses.replace(cfg, scent_spread=0.4, mark_rate=0.0, nest_mark=0.0)
    t = jnp.zeros((2, G, G)).at[0, 0, 0].set(1.0)
    s = update_territory(blank(c)._replace(territory=t), c)
    assert jnp.isclose(s.territory[0, 0, 0], 0.6)
    for y, x in [(0, 1), (1, 0), (0, G - 1), (G - 1, 0)]:  # wraps both ways
        assert jnp.isclose(s.territory[0, y, x], 0.1), (y, x)
    assert jnp.isclose(s.territory.sum(), 1.0)  # spreading moves scent, it doesn't create it


def test_unmarked_ground_is_neutral_and_unowned():
    s = blank(cfg)._replace(territory=jnp.zeros((2, G, G)))
    own, foreign = scent_at(s, jnp.array([[0.5, 0.5]]), jnp.array([0]), cfg)
    assert float(home_share(own, foreign)[0]) == 0.5
    owner, strength = owner_grid(s, cfg)
    assert bool((owner == -1).all()) and bool((strength == 0).all())


def test_owner_is_the_family_with_most_scent():
    t = jnp.zeros((2, G, G)).at[0, 2, 3].set(3.0).at[1, 2, 3].set(1.0).at[1, 5, 5].set(2.0)
    s = blank(cfg)._replace(territory=t)
    owner, strength = owner_grid(s, cfg)
    assert int(owner[2, 3]) == 0 and jnp.isclose(strength[2, 3], 0.75)
    assert int(owner[5, 5]) == 1 and jnp.isclose(strength[5, 5], 1.0)
    own, foreign = scent_at(s, jnp.array([[3.5 / G, 2.5 / G]] * 2), jnp.array([0, 1]), cfg)
    assert jnp.allclose(own, jnp.array([3.0, 1.0])) and jnp.allclose(foreign, jnp.array([1.0, 3.0]))
    assert jnp.allclose(home_share(own, foreign), jnp.array([0.75, 0.25]))


def test_nest_on_foreign_ground_pays_upkeep_and_at_home_does_not():
    s = member(blank(cfg), 0, (0.6, 0.6), colony=0)
    nest_cell = int(cell_of(s.nest[:1], cfg)[0])
    y, x = divmod(nest_cell, G)
    no_speed = jnp.zeros(cfg.capacity_members)
    home = s._replace(territory=jnp.zeros((2, G, G)).at[0, y, x].set(5.0))
    at_home = metabolise(home, no_speed, cfg).energy[0]
    assert jnp.isclose(at_home, 10.0 - cfg.base_cost)  # no upkeep on its own ground

    foreign = s._replace(territory=jnp.zeros((2, G, G)).at[1, y, x].set(5.0))
    assert jnp.isclose(metabolise(foreign, no_speed, cfg).energy[0], at_home - cfg.upkeep_rate)
    neutral = s._replace(territory=jnp.zeros((2, G, G)))
    assert jnp.isclose(metabolise(neutral, no_speed, cfg).energy[0], at_home - cfg.upkeep_rate / 2)


def test_territory_stays_finite_and_bounded_over_a_long_run():
    c = Config(capacity_members=256, capacity_colonies=16, capacity_food=64, initial_colonies=6)
    s = run_ticks(init_world(jax.random.key(0), c), c, 20_000)
    t = np.asarray(s.territory)
    assert np.isfinite(t).all() and (t >= 0).all()
    ceiling = (c.capacity_members * c.mark_rate + c.capacity_colonies * c.nest_mark) / (1 - c.scent_fade)
    assert t.max() <= ceiling
