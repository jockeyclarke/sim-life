import jax

from simlife.config import Config
from simlife.state import init_world


def test_config_roundtrip_and_hashable():
    cfg = Config(capacity_members=128)
    assert Config.from_dict(cfg.to_dict()) == cfg
    hash(cfg)


def test_init_world_shapes_and_counts():
    cfg = Config()
    s = init_world(jax.random.key(0), cfg)
    assert s.pos.shape == (2048, 2) and s.params["w1"].shape == (2048, 8, cfg.n_inputs)
    n = cfg.initial_colonies
    assert int(s.alive.sum()) == n * cfg.initial_members and int(s.col_alive.sum()) == n
    assert int(s.tick) == 0 and int(s.delivered_total) == 0
    assert bool(((s.pos >= 0) & (s.pos < 1)).all())


def test_initial_colony_is_clonal_and_colonies_differ():
    s = init_world(jax.random.key(0), Config())
    w = s.params["w1"]
    m0 = s.alive & (s.colony == 0)
    m1 = s.alive & (s.colony == 1)
    assert bool((w[m0] == w[m0][0]).all())
    assert not bool((w[m0][0] == w[m1][0]).all())


def test_initial_members_start_at_nest_with_energy():
    cfg = Config()
    s = init_world(jax.random.key(0), cfg)
    assert bool((s.energy[s.col_alive] == cfg.initial_energy).all())
    assert bool((s.lifespan[s.alive] >= cfg.lifespan_mean - cfg.lifespan_jitter).all())
    assert not bool(s.carrying.any())
    assert bool((s.pos[s.alive] == s.nest[s.colony[s.alive]]).all())


# --- stage 2: learning state ---------------------------------------------------


def test_world_has_learning_state():
    cfg = Config()
    s = init_world(jax.random.key(0), cfg)
    P = cfg.n_predictions
    assert s.params["w3"].shape == (2048, P, 8) and s.params["b3"].shape == (2048, P)
    assert s.birth_params.keys() == s.params.keys()
    assert all(bool((s.birth_params[k] == s.params[k]).all()) for k in s.params)
    assert bool((s.lr[s.alive] == cfg.initial_lr).all()) and not bool(s.has_prev.any())
    assert bool(s.learning_on) and bool(s.inherit_learned)
    assert s.prev_obs.shape == (2048, cfg.n_inputs) and s.last_pred.shape == (2048, P)
    assert not bool(s.pred_error.any()) and not bool(s.pred_error_avg.any())


def test_prediction_weights_are_clonal_within_a_colony():
    s = init_world(jax.random.key(0), Config())
    w = s.params["w3"]
    m0 = s.alive & (s.colony == 0)
    m1 = s.alive & (s.colony == 1)
    assert bool((w[m0] == w[m0][0]).all())
    assert not bool((w[m0][0] == w[m1][0]).all())


# --- stage 3: families ----------------------------------------------------------

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from simlife.senses import torus_distance  # noqa: E402


def nearest_centre(s):
    d = torus_distance(s.nest[:, None, :], s.family_centre[None, :, :])
    return np.asarray(jnp.argmin(d, axis=1))


def test_founding_world_has_independent_colonies_and_regional_families():
    cfg = Config()
    s = init_world(jax.random.key(0), cfg)
    assert int(s.col_alive.sum()) == 18 and s.territory.shape == (6, 64, 64)
    assert s.family_centre.shape == (6, 2) and int(s.founded) == 0
    w = s.params["w1"]
    firsts = [w[s.alive & (s.colony == c)] for c in range(18)]
    assert all(bool((f == f[0]).all()) for f in firsts)  # a colony is clonal
    assert len({bytes(np.asarray(f[0])) for f in firsts}) == 18  # colonies are independent
    live = np.asarray(s.col_alive)
    assert (np.asarray(s.col_family)[live] == nearest_centre(s)[live]).all()
    assert s.params["w2"].shape == (cfg.capacity_members, 6, 8)
    assert s.params["w3"].shape == (cfg.capacity_members, 9, 8)
    assert not bool(s.territory.any()) and not bool(s.striking.any())
    assert s.strikes.shape == (3,) and s.colonies_died.shape == (2,)
    assert bool((s.col_born == 0).all()) and not bool(s.col_foreign.any())


def test_founding_world_ignores_switches():
    cfg = Config()
    a = init_world(jax.random.key(4), cfg)
    b = init_world(jax.random.key(4), cfg)._replace(
        learning_on=jnp.bool_(False), inherit_learned=jnp.bool_(False)
    )
    assert bool((a.params["w1"] == b.params["w1"]).all())
    assert bool((a.nest == b.nest).all()) and bool((a.col_family == b.col_family).all())
