import jax
import jax.numpy as jnp

from simlife.brain import act, act_all, init_params
from simlife.config import Config


def test_act_ranges_and_shapes():
    p = init_params(jax.random.key(1), Config())
    turn, speed, signal, mem = act(p, jnp.ones(10) * 3.0)
    assert turn.shape == () and mem.shape == (2,)
    assert -1 <= turn <= 1 and 0 <= speed <= 1 and -1 <= signal <= 1
    assert bool((jnp.abs(mem) <= 1).all())


def test_act_all_matches_single():
    cfg = Config()
    keys = jax.random.split(jax.random.key(2), 5)
    ps = jax.vmap(lambda k: init_params(k, cfg))(keys)
    xs = jax.random.normal(jax.random.key(3), (5, 10))
    turn_all, *_ = act_all(ps, xs)
    turn_3, *_ = act(jax.tree.map(lambda a: a[3], ps), xs[3])
    assert jnp.allclose(turn_all[3], turn_3)


def test_zero_weights_give_neutral_output():
    p = jax.tree.map(jnp.zeros_like, init_params(jax.random.key(0), Config()))
    turn, speed, signal, mem = act(p, jnp.ones(10))
    assert turn == 0 and speed == 0.5 and signal == 0


def test_world_founders_come_from_init_params():
    from simlife.state import init_world

    cfg = Config()
    s = init_world(jax.random.key(0), cfg)
    assert s.params.keys() == init_params(jax.random.key(0), cfg).keys()
