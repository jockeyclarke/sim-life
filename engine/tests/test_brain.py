import jax
import jax.numpy as jnp

from simlife.brain import act, act_all, init_params
from simlife.config import Config


def test_act_ranges_and_shapes():
    p = init_params(jax.random.key(1), Config())
    turn, speed, signal, mem = act(p, jnp.ones(Config().n_inputs) * 3.0)[:4]
    assert turn.shape == () and mem.shape == (2,)
    assert -1 <= turn <= 1 and 0 <= speed <= 1 and -1 <= signal <= 1
    assert bool((jnp.abs(mem) <= 1).all())


def test_act_all_matches_single():
    cfg = Config()
    keys = jax.random.split(jax.random.key(2), 5)
    ps = jax.vmap(lambda k: init_params(k, cfg))(keys)
    xs = jax.random.normal(jax.random.key(3), (5, cfg.n_inputs))
    turn_all, *_ = act_all(ps, xs)
    turn_3, *_ = act(jax.tree.map(lambda a: a[3], ps), xs[3])
    assert jnp.allclose(turn_all[3], turn_3)


def test_act_returns_aggression_in_unit_range():
    p = init_params(jax.random.key(1), Config())
    out = act(p, jnp.ones(Config().n_inputs) * 3.0)
    assert len(out) == 5 and out[4].shape == () and 0 <= out[4] <= 1
    zero = jax.tree.map(jnp.zeros_like, p)
    assert act(zero, jnp.ones(Config().n_inputs))[4] == 0.5


def test_zero_weights_give_neutral_output():
    p = jax.tree.map(jnp.zeros_like, init_params(jax.random.key(0), Config()))
    turn, speed, signal, mem = act(p, jnp.ones(Config().n_inputs))[:4]
    assert turn == 0 and speed == 0.5 and signal == 0


def test_world_founders_come_from_init_params():
    from simlife.state import init_world

    cfg = Config()
    s = init_world(jax.random.key(0), cfg)
    assert s.params.keys() == init_params(jax.random.key(0), cfg).keys()


# --- stage 2: prediction and learning ------------------------------------------

from simlife.brain import hidden, learn, learn_all, predict, prediction_loss  # noqa: E402

CFG = Config()
P0 = init_params(jax.random.key(11), CFG)
X = jax.random.uniform(jax.random.key(12), (CFG.n_inputs,))
TARGET = jax.random.uniform(jax.random.key(13), (CFG.n_predictions,))


def same(a, b, keys=None):
    return all(bool((a[k] == b[k]).all()) for k in (keys or a))


def test_predict_is_a_linear_readout_of_the_hidden_layer():
    h = hidden(P0, X)
    assert h.shape == (CFG.n_hidden,) and bool((jnp.abs(h) <= 1).all())
    assert jnp.allclose(predict(P0, X), P0["w3"] @ h + P0["b3"])


def test_loss_gradient_never_touches_action_weights():
    g = jax.grad(prediction_loss)(P0, X, TARGET)
    assert bool((g["w2"] == 0).all()) and bool((g["b2"] == 0).all())
    assert bool((g["w1"] != 0).any()) and bool((g["w3"] != 0).any())


def test_learning_reduces_error_on_a_repeated_example():
    q = P0
    first = prediction_loss(P0, X, TARGET)
    for _ in range(200):
        q, _, _ = learn(q, X, TARGET, 0.05, 1.0)
    assert prediction_loss(q, X, TARGET) < 0.05 * first


def test_learn_reports_loss_and_prediction_from_before_the_update():
    q, loss, pred = learn(P0, X, TARGET, 0.05, 1.0)
    assert jnp.allclose(pred, predict(P0, X))
    assert jnp.isclose(loss, prediction_loss(P0, X, TARGET))
    assert same(q, P0, ["w2", "b2"]) and not same(q, P0, ["w1"])


def test_zero_learning_rate_changes_nothing():
    q, _, _ = learn(P0, X, TARGET, 0.0, 1.0)
    assert same(q, P0)


def test_gradient_clip_bounds_the_step():
    far = jnp.full(CFG.n_predictions, 1e4)
    q, _, _ = learn(P0, X, far, 1.0, 1.0)
    moved = jnp.sqrt(sum(((q[k] - P0[k]) ** 2).sum() for k in P0))
    assert 0.5 < float(moved) <= 1.0 + 1e-4
    gentle, _, _ = learn(P0, X, predict(P0, X) + 1e-3, 1.0, 1.0)  # tiny gradient: not rescaled
    assert float(jnp.sqrt(sum(((gentle[k] - P0[k]) ** 2).sum() for k in P0))) < 0.01


def test_learn_all_matches_single():
    keys = jax.random.split(jax.random.key(2), 5)
    ps = jax.vmap(lambda k: init_params(k, CFG))(keys)
    xs = jax.random.uniform(jax.random.key(3), (5, CFG.n_inputs))
    ts = jax.random.uniform(jax.random.key(4), (5, CFG.n_predictions))
    lrs = jnp.array([0.0, 0.01, 0.02, 0.05, 0.1])
    qs, losses, preds = learn_all(ps, xs, ts, lrs, 1.0)
    q3, loss3, pred3 = learn(jax.tree.map(lambda a: a[3], ps), xs[3], ts[3], lrs[3], 1.0)
    assert all(jnp.allclose(qs[k][3], q3[k], atol=1e-6) for k in q3)
    assert jnp.isclose(losses[3], loss3) and jnp.allclose(preds[3], pred3)
    assert same(jax.tree.map(lambda a: a[0], qs), jax.tree.map(lambda a: a[0], ps))  # lr 0
