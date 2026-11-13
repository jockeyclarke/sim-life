"""The learning phase of a tick: every member scores and corrects last tick's guess."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from .brain import learn_all
from .config import Config
from .state import WorldState


def learn_phase(state: WorldState, obs: Array, cfg: Config) -> WorldState:
    """Compare each member's previous prediction with what it senses now, and learn.

    Last tick a member saw `prev_obs` and (implicitly) predicted its next
    senses. Now it sees `obs`. The error between the two is backpropagated
    through the prediction outputs and the hidden layer, with the member's own
    learning rate. The only signal is the member's own senses: nothing about
    food delivered, energy or survival enters the gradient.
    """
    target = obs[:, : cfg.n_predictions]
    new_params, loss, prediction = learn_all(
        state.params, state.prev_obs, target, state.lr, cfg.grad_clip
    )

    scored = state.alive & state.has_prev  # has a previous guess to score
    update = scored & state.learning_on

    def pick(new: Array, old: Array) -> Array:
        return jnp.where(update.reshape((-1,) + (1,) * (new.ndim - 1)), new, old)

    s = cfg.error_smoothing
    first = state.pred_error_avg == 0  # start the average at the first error, not at zero
    average = jnp.where(first, loss, s * state.pred_error_avg + (1 - s) * loss)

    return state._replace(
        params={k: pick(new_params[k], state.params[k]) for k in state.params},
        last_pred=jnp.where(scored[:, None], prediction, 0.0),
        pred_error=jnp.where(scored, loss, 0.0),
        pred_error_avg=jnp.where(scored, average, state.pred_error_avg),
        prev_obs=obs,
        has_prev=state.alive,
    )
