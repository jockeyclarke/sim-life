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


def act(params: dict, x: Array) -> tuple[Array, Array, Array, Array]:
    """One entity's decision for one tick.

    Returns (turn in [-1, 1], speed in [0, 1], signal in [-1, 1],
    memory (2,) in [-1, 1]). Scaling to world units happens in physics.
    """
    h = jnp.tanh(params["w1"] @ x + params["b1"])
    o = params["w2"] @ h + params["b2"]
    return jnp.tanh(o[0]), jax.nn.sigmoid(o[1]), jnp.tanh(o[2]), jnp.tanh(o[3:5])


act_all = jax.vmap(act)
