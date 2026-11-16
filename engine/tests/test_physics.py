import dataclasses

import jax.numpy as jnp

from helpers import blank, food, member, small_cfg
from simlife.physics import deliver, metabolise, move, pickup

cfg = small_cfg()
N = cfg.capacity_members


def test_move_wraps_and_ignores_dead():
    s = member(blank(cfg), 0, (0.999, 0.5), heading=0.0)
    before = s
    s = move(s, jnp.zeros(N), jnp.ones(N), cfg)
    assert jnp.allclose(s.pos[0], jnp.array([0.003, 0.5]), atol=1e-5)
    assert bool((s.pos[1:] == before.pos[1:]).all())
    assert bool((s.heading[1:] == before.heading[1:]).all())


def test_move_turns_by_max_turn_and_speed_scales_distance():
    s = member(blank(cfg), 0, (0.5, 0.5), heading=0.0)
    s = move(s, jnp.ones(N), jnp.full(N, 0.5), cfg)
    assert jnp.isclose(s.heading[0], cfg.max_turn)
    step = jnp.linalg.norm(s.pos[0] - jnp.array([0.5, 0.5]))
    assert jnp.isclose(step, 0.5 * cfg.max_speed, rtol=1e-3)


def test_move_never_leaves_the_unit_square():
    # a tiny negative coordinate must not wrap to exactly 1.0
    s = member(blank(cfg), 0, (0.0, 0.0), heading=jnp.pi)
    s = move(s, jnp.zeros(N), jnp.full(N, 1e-6), cfg)
    assert bool(((s.pos >= 0) & (s.pos < 1)).all())


def test_pickup_conserves_food_under_contention():
    s = member(blank(cfg), 0, (0.5, 0.5))
    s = member(s, 1, (0.505, 0.5))
    s = food(s, 0, (0.502, 0.5))
    s2 = pickup(s, cfg)
    assert bool(s2.carrying[0]) and not bool(s2.carrying[1])
    assert int(s2.food_alive.sum()) == int(s.food_alive.sum()) - 1


def test_pickup_out_of_reach_does_nothing():
    s = food(member(blank(cfg), 0, (0.5, 0.5)), 0, (0.5, 0.5 + 2 * cfg.pickup_radius))
    s2 = pickup(s, cfg)
    assert not bool(s2.carrying[0]) and bool(s2.food_alive[0])


def test_carrying_member_does_not_pick_up():
    s = food(member(blank(cfg), 0, (0.5, 0.5), carrying=True), 0, (0.5, 0.5))
    s2 = pickup(s, cfg)
    assert bool(s2.food_alive[0]) and bool(s2.carrying[0])


def test_dead_member_does_not_pick_up():
    s = food(blank(cfg), 0, (0.5, 0.5))
    s = s._replace(pos=s.pos.at[3].set(jnp.array([0.5, 0.5])))
    assert bool(pickup(s, cfg).food_alive[0])


def test_deliver_credits_own_colony_only():
    # colony 0's nest is at (0.2, 0.2), colony 1's at (0.8, 0.8)
    wrong = deliver(member(blank(cfg), 0, (0.8, 0.8), colony=0, carrying=True), cfg)
    assert bool(wrong.carrying[0]) and int(wrong.delivered_total) == 0
    assert bool((wrong.energy == 10.0).all())

    right = deliver(member(blank(cfg), 0, (0.2, 0.2), colony=0, carrying=True), cfg)
    assert not bool(right.carrying[0]) and int(right.delivered_total) == 1
    assert jnp.isclose(right.energy[0], 10.0 + cfg.food_energy) and right.energy[1] == 10.0


def test_metabolise_charges_colony_pool():
    s = member(member(blank(cfg), 0, (0.5, 0.5)), 1, (0.6, 0.6))
    speed = jnp.zeros(N).at[1].set(1.0).at[4].set(1.0)  # slot 4 is dead
    s2 = metabolise(s, speed, dataclasses.replace(cfg, upkeep_rate=0.0))  # upkeep is tested in test_territory
    assert jnp.isclose(s2.energy[0], 10.0 - (2 * cfg.base_cost + cfg.speed_cost))
    assert s2.energy[1] == 10.0


def test_carried_food_spoils_with_half_life():
    # colony 0's nest is at (0.2, 0.2); a pellet carried one half-life is worth half
    fresh = deliver(member(blank(cfg), 0, (0.2, 0.2), colony=0, carrying=True), cfg)
    assert jnp.isclose(fresh.energy[0], 10.0 + cfg.food_energy)
    old = member(blank(cfg), 0, (0.2, 0.2), colony=0, carrying=True, carry_ticks=cfg.food_half_life)
    assert jnp.isclose(deliver(old, cfg).energy[0], 10.0 + cfg.food_energy / 2, atol=1e-4)


def test_carry_ticks_count_while_carrying_and_reset_otherwise():
    s = member(blank(cfg), 0, (0.5, 0.5), colony=0, carrying=True, carry_ticks=7)
    s = member(s, 1, (0.6, 0.6), colony=0, carry_ticks=9)  # not carrying: stale count
    s = deliver(s, cfg)  # away from the nest, so nothing is dropped
    assert int(s.carry_ticks[0]) == 8 and int(s.carry_ticks[1]) == 0
    home = deliver(member(blank(cfg), 0, (0.2, 0.2), colony=0, carrying=True, carry_ticks=5), cfg)
    assert int(home.carry_ticks[0]) == 0  # dropped off: the count starts over
