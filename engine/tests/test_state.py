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
    assert s.pos.shape == (2048, 2) and s.params["w1"].shape == (2048, 8, 10)
    assert int(s.alive.sum()) == 16 * 12 and int(s.col_alive.sum()) == 16
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
