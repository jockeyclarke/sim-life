"""The network, written for ONE entity.

Nothing here knows about the population. `act_all` is the same function
lifted over a leading population axis by `jax.vmap`: every array in `params`
and the observation gain one extra dimension, and JAX runs all entities as
single batched matrix operations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .config import Config


def init_params(key: Array, cfg: Config) -> dict:
    """Random weights for one entity (no leading population axis)."""
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    s = cfg.init_sigma
    return {
        # inputs -> hidden: trained during life, and mutated at division
        "w1": jax.random.normal(k1, (cfg.n_hidden, cfg.n_inputs)) * s,
        "b1": jax.random.normal(k2, (cfg.n_hidden,)) * s,
        # hidden -> actions: changed only by mutation at division
        "w2": jax.random.normal(k3, (cfg.n_actions, cfg.n_hidden)) * s,
        "b2": jax.random.normal(k4, (cfg.n_actions,)) * s,
        # hidden -> predictions: trained during life, and mutated at division
        "w3": jax.random.normal(k5, (cfg.n_predictions, cfg.n_hidden)) * s,
        "b3": jax.random.normal(k6, (cfg.n_predictions,)) * s,
    }


def hidden(params: dict, x: Array) -> Array:
    """The hidden layer's activations. Both sets of outputs read these."""
    return jnp.tanh(params["w1"] @ x + params["b1"])


def act(params: dict, x: Array) -> tuple[Array, Array, Array, Array]:
    """One entity's decision for one tick.

    Returns (turn in [-1, 1], speed in [0, 1], signal in [-1, 1],
    memory (2,) in [-1, 1]). Scaling to world units happens in physics.
    """
    o = params["w2"] @ hidden(params, x) + params["b2"]
    return jnp.tanh(o[0]), jax.nn.sigmoid(o[1]), jnp.tanh(o[2]), jnp.tanh(o[3:5])


def predict(params: dict, x: Array) -> Array:
    """The entity's guess at its predicted senses next tick, given what it senses now."""
    return params["w3"] @ hidden(params, x) + params["b3"]


def _loss_and_prediction(params: dict, x_prev: Array, target: Array) -> tuple[Array, Array]:
    prediction = predict(params, x_prev)
    return jnp.mean((prediction - target) ** 2), prediction


def prediction_loss(params: dict, x_prev: Array, target: Array) -> Array:
    """How wrong last tick's guess was: mean squared error against what was then sensed.

    The loss does not depend on w2 or b2, so its gradient for them is exactly
    zero: learning cannot touch the action weights.
    """
    return _loss_and_prediction(params, x_prev, target)[0]


def learn(
    params: dict, x_prev: Array, target: Array, lr: Array, clip: float
) -> tuple[dict, Array, Array]:
    """One gradient step for one entity. Returns (new params, loss, prediction).

    The loss and prediction are those from before the update. JAX derives the
    backward pass from the forward code above: there is no hand-written
    backpropagation anywhere in this project.
    """
    (loss, prediction), grads = jax.value_and_grad(_loss_and_prediction, has_aux=True)(
        params, x_prev, target
    )
    # If the gradient as a whole is longer than `clip`, shrink it to that length.
    norm = jnp.sqrt(sum((g**2).sum() for g in grads.values()))
    scale = jnp.minimum(1.0, clip / jnp.maximum(norm, 1e-12))
    new_params = {k: params[k] - lr * scale * grads[k] for k in params}
    return new_params, loss, prediction


act_all = jax.vmap(act)
# Every array gains a leading population axis except `clip`, which is shared.
# This one line is "per-entity gradients for the whole population".
learn_all = jax.vmap(learn, in_axes=(0, 0, 0, 0, None))
