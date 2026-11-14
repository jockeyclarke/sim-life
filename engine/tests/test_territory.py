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
cfg = small_cfg(territory_grid=8, scent_fade=1.0, scent_spread=0.0, n_families=2, grace_ticks=0)
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


def test_no_upkeep_during_the_grace_period():
    c = dataclasses.replace(cfg, grace_ticks=100)
    s = member(blank(c), 0, (0.6, 0.6), colony=0)
    y, x = divmod(int(cell_of(s.nest[:1], c)[0]), G)
    foreign = s._replace(territory=jnp.zeros((2, G, G)).at[1, y, x].set(5.0), tick=jnp.int32(50))
    no_speed = jnp.zeros(c.capacity_members)
    assert jnp.isclose(metabolise(foreign, no_speed, c).energy[0], 10.0 - c.base_cost)
    later = foreign._replace(tick=jnp.int32(100))
    assert jnp.isclose(metabolise(later, no_speed, c).energy[0], 10.0 - c.base_cost - c.upkeep_rate)


# --- families form when the grace period ends -----------------------------------

from simlife.lifecycle import form_families  # noqa: E402
from simlife.senses import torus_distance  # noqa: E402


def test_families_form_exactly_when_the_grace_period_ends():
    c = Config(capacity_members=256, capacity_colonies=16, capacity_food=64, initial_colonies=8,
               grace_ticks=50)
    s = init_world(jax.random.key(2), c)
    scrambled = s._replace(col_family=(s.col_family + 1) % c.n_families,
                           territory=jnp.ones_like(s.territory))
    before = form_families(scrambled._replace(tick=jnp.int32(49)), c)
    assert bool((before.col_family == scrambled.col_family).all()) and bool(before.territory.all())
    at = form_families(scrambled._replace(tick=jnp.int32(50)), c)
    d = torus_distance(at.nest[:, None, :], at.family_centre[None, :, :])
    assert bool((at.col_family == jnp.argmin(d, axis=1)).all())
    assert not bool(at.territory.any())
    later = form_families(scrambled._replace(tick=jnp.int32(51)), c)
    assert bool((later.col_family == scrambled.col_family).all())


def test_grace_is_counted_from_when_the_world_was_founded():
    c = dataclasses.replace(cfg, grace_ticks=100)
    s = member(blank(c), 0, (0.6, 0.6), colony=0)
    y, x = divmod(int(cell_of(s.nest[:1], c)[0]), G)
    foreign = s._replace(territory=jnp.zeros((2, G, G)).at[1, y, x].set(5.0),
                         founded=jnp.int32(1000), tick=jnp.int32(1050))
    no_speed = jnp.zeros(c.capacity_members)
    assert jnp.isclose(metabolise(foreign, no_speed, c).energy[0], 10.0 - c.base_cost)
    assert jnp.isclose(metabolise(foreign._replace(tick=jnp.int32(1100)), no_speed, c).energy[0],
                       10.0 - c.base_cost - c.upkeep_rate)


def test_families_are_formed_in_the_record_of_the_grace_tick():
    from simlife.step import step
    c = Config(capacity_members=256, capacity_colonies=16, capacity_food=64, initial_colonies=8,
               grace_ticks=50)
    s = init_world(jax.random.key(2), c)
    s = s._replace(col_family=(s.col_family + 1) % c.n_families, tick=jnp.int32(49))
    s2 = jax.jit(lambda s: step(s, c))(s)  # the step that produces tick 50
    d = torus_distance(s2.nest[:, None, :], s2.family_centre[None, :, :])
    assert int(s2.tick) == 50 and bool((s2.col_family == jnp.argmin(d, axis=1)).all())


def test_formation_clears_founded_on_foreign_ground():
    c = Config(capacity_members=256, capacity_colonies=16, capacity_food=64, initial_colonies=8,
               grace_ticks=50)
    s = init_world(jax.random.key(2), c)
    s = s._replace(col_foreign=jnp.ones_like(s.col_foreign), tick=jnp.int32(50))
    assert not bool(form_families(s, c).col_foreign.any())
