import jax
import jax.numpy as jnp

from helpers import blank, member, small_cfg, states_equal
from simlife.lifecycle import age_and_die, grow, kill_dead_colonies

cfg = small_cfg()
N = cfg.capacity_members
MEMBER_FIELDS = [
    "pos", "heading", "alive", "colony", "carrying", "age", "lifespan",
    "memory", "signal", "generation",
    "lr", "has_prev", "prev_obs", "last_pred", "pred_error", "pred_error_avg",
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
        assert bool((s2.birth_params[name][s.alive] == s.birth_params[name][s.alive]).all()), name


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


# --- splitting and food regrowth -------------------------------------------

from helpers import food  # noqa: E402
from simlife.lifecycle import regrow_food, split  # noqa: E402
from simlife.senses import torus_distance, wrap_delta  # noqa: E402

split_cfg = small_cfg(split_size=4)


def ready_to_split():
    """Colony 0 has split_size members and 40 energy; colony slot 1 is free."""
    s = blank(split_cfg)
    for i in range(4):
        s = member(s, i, (0.3 + 0.01 * i, 0.3), colony=0, carrying=(i == 0))
    return s._replace(
        energy=jnp.array([40.0, 0.0]), col_alive=jnp.array([True, False])
    )


def test_split_partitions_members_and_energy():
    s = ready_to_split()
    s2 = split(s, split_cfg, jax.random.key(0))
    d = int(jnp.argmax(s2.col_alive & ~s.col_alive))
    n_parent = int((s2.alive & (s2.colony == 0)).sum())
    n_d = int((s2.alive & (s2.colony == d)).sum())
    assert n_parent + n_d == split_cfg.split_size and n_d > 0 and n_parent > 0
    assert jnp.isclose(s2.energy[0], 20.0) and jnp.isclose(s2.energy[d], 20.0)
    assert int(s2.col_generation[d]) == int(s.col_generation[0]) + 1
    gap = jnp.linalg.norm(wrap_delta(s2.nest[d], s2.nest[0]))
    assert jnp.isclose(gap, split_cfg.split_distance, atol=1e-5)
    assert bool((s2.params["w1"] == s.params["w1"]).all())
    assert bool((s2.pos == s.pos).all()) and bool((s2.carrying == s.carrying).all())


def test_split_never_leaves_either_side_empty():
    s = ready_to_split()
    for seed in range(30):
        s2 = split(s, split_cfg, jax.random.key(seed))
        assert int((s2.alive & (s2.colony == 0)).sum()) == 2
        assert int((s2.alive & (s2.colony == 1)).sum()) == 2


def test_split_below_size_does_nothing():
    s = ready_to_split()
    s = s._replace(alive=s.alive.at[3].set(False))
    assert states_equal(split(s, split_cfg, jax.random.key(0)), s)


def test_split_when_colonies_full_does_nothing():
    s = ready_to_split()._replace(col_alive=jnp.array([True, True]))
    assert states_equal(split(s, split_cfg, jax.random.key(0)), s)


def test_regrow_only_revives_dead_pellets_inside_their_patch():
    c = small_cfg(food_regrow_prob=1.0)
    s = food(blank(c), 0, (0.1, 0.9))  # one live pellet, far from the patch at (0.5, 0.5)
    s2 = regrow_food(s, c, jax.random.key(0))
    assert bool(s2.food_alive.all())
    assert bool((s2.food_pos[0] == s.food_pos[0]).all())
    dist = torus_distance(s2.food_pos[1:], s.patch_pos[0])
    assert bool((dist <= c.patch_radius + 1e-6).all())


def test_regrow_prob_zero_revives_nothing():
    c = small_cfg(food_regrow_prob=0.0)
    s = food(blank(c), 0, (0.1, 0.9))
    assert states_equal(regrow_food(s, c, jax.random.key(0)), s)


# --- stage 2: inheritance of learned weights and learning rate -------------------

quiet = small_cfg(mutation_sigma=1e-6)


def learned_parent(inherit_learned: bool):
    """Colony 0: one rich member whose current weights are its birth weights + 1 (as if learned)."""
    s = member(blank(quiet), 0, (0.3, 0.3))
    params = {k: v.at[0].add(1.0) for k, v in s.params.items()}
    return s._replace(
        params=params, energy=jnp.array([100.0, 0.0]),
        col_alive=jnp.array([True, False]), inherit_learned=jnp.bool_(inherit_learned),
    )


def the_child(before, after):
    return int(jnp.argmax(after.alive & ~before.alive))


def test_child_inherits_learned_weights_by_default():
    s = learned_parent(True)
    s2 = grow(s, quiet, jax.random.key(0))
    c = the_child(s, s2)
    for k in s.params:
        assert jnp.allclose(s2.params[k][c], s.params[k][0], atol=1e-4), k
        assert bool((s2.birth_params[k][c] == s2.params[k][c]).all()), k
        assert bool((s2.birth_params[k][0] == s.birth_params[k][0]).all()), k  # parent untouched


def test_child_inherits_birth_weights_when_switched():
    s = learned_parent(False)
    s2 = grow(s, quiet, jax.random.key(0))
    c = the_child(s, s2)
    for k in s.params:
        assert jnp.allclose(s2.params[k][c], s.birth_params[k][0], atol=1e-4), k
        assert not jnp.allclose(s2.params[k][c], s.params[k][0], atol=0.5), k
        assert bool((s2.birth_params[k][c] == s2.params[k][c]).all()), k


def test_child_learning_rate_is_inherited_mutated_and_clipped():
    s = learned_parent(True)
    ratios = []
    for seed in range(40):
        s2 = grow(s, quiet, jax.random.key(seed))
        lr = float(s2.lr[the_child(s, s2)])
        assert 0 <= lr <= quiet.max_lr
        ratios.append(lr / quiet.initial_lr)
    assert len(set(ratios)) > 30  # mutated
    assert 0.9 < sum(ratios) / len(ratios) < 1.1  # around the parent's
    assert all(0.5 < r < 2.0 for r in ratios)

    top = s._replace(lr=s.lr.at[0].set(quiet.max_lr))
    for seed in range(20):
        s2 = grow(top, quiet, jax.random.key(seed))
        assert float(s2.lr[the_child(top, s2)]) <= quiet.max_lr


def test_child_starts_with_no_previous_observation():
    s = learned_parent(True)
    n = quiet.capacity_members
    s = s._replace(  # stale values in every slot, as a dead member would leave behind
        has_prev=jnp.ones(n, bool), prev_obs=jnp.ones((n, 10)), last_pred=jnp.ones((n, 6)),
        pred_error=jnp.ones(n), pred_error_avg=jnp.ones(n),
    )
    s2 = grow(s, quiet, jax.random.key(0))
    c = the_child(s, s2)
    assert not bool(s2.has_prev[c])
    assert not bool(s2.prev_obs[c].any()) and not bool(s2.last_pred[c].any())
    assert s2.pred_error[c] == 0 and s2.pred_error_avg[c] == 0
    assert bool(s2.has_prev[0])  # the parent keeps its own
