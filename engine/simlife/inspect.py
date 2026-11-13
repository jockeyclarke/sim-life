"""One entity's internals as plain numbers, for the viewer's inspector."""

from __future__ import annotations

import numpy as np

from .brain import act, hidden, predict
from .config import Config
from .state import WorldState


def inspect_entity(state: WorldState, cfg: Config, slot: int) -> dict:
    """Everything about member `slot` that the inspector shows. JSON-ready.

    `inputs` is the observation the member last acted on. `hidden` and
    `actions` are recomputed from it with the member's current weights, which
    is the same arithmetic the tick ran. This reads one index out of the
    population arrays: the batched engine keeps nothing per entity that has to
    be dug out.
    """
    params = {k: v[slot] for k, v in state.params.items()}
    x = state.prev_obs[slot]
    turn, speed, signal, memory = act(params, x)
    actions = [float(turn), float(speed), float(signal), float(memory[0]), float(memory[1])]

    def numbers(a) -> list:
        return np.asarray(a).astype(float).tolist()

    return {
        "slot": int(slot),
        "alive": bool(state.alive[slot]),
        "colony": int(state.colony[slot]),
        "age": int(state.age[slot]),
        "lifespan": int(state.lifespan[slot]),
        "generation": int(state.generation[slot]),
        "carrying": bool(state.carrying[slot]),
        "learning_rate": float(state.lr[slot]),
        "inputs": numbers(x),
        "hidden": numbers(hidden(params, x)),
        "actions": actions,  # turn, speed, signal, memory 0, memory 1
        "predicted": numbers(state.last_pred[slot]),  # last tick's guess ...
        "actual": numbers(x[: cfg.n_predictions]),  # ... and what was then sensed
        "next_prediction": numbers(predict(params, x)),  # its guess for the coming tick
        "error": float(state.pred_error[slot]),
        "error_avg": float(state.pred_error_avg[slot]),
        "weights": {k: numbers(v) for k, v in params.items()},
    }
