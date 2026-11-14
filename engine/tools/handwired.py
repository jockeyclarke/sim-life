"""Does directed foraging pay in this world? Plant hand-wired colonies among evolved ones.

A world is seeded from a nursery's survivors as usual, then every third
founding colony is given a hand-wired network instead (learning rate 0, so
only division noise changes it). The run stays inside the grace period, so
this measures foraging alone: no upkeep, no fighting. Members are told apart
by how close their input weights still are to the hand-wired ones.

    .venv/bin/python engine/tools/handwired.py runs/nursery --variant full --seed 1 [--ticks 15000]

Variants (what the network does):
    full       not carrying: turn toward food;  carrying: turn toward the nest
    food-only  not carrying: turn toward food;  carrying: go straight
    nest-only  not carrying: go straight;       carrying: turn toward the nest

How the wiring works. Inputs: 1 food direction and 3 nest direction (+ means
on the left), 9 carrying (0 or 1). A positive turn swings toward
the left sensor. Each behaviour is a pair of hidden units, h = tanh(±g·signal
+ gate), read out as their difference. A gate of -6 saturates both units at
-1, so the difference, and the behaviour, vanishes; a gate of 0 lets the
signal through. Carrying moves the gate, which switches between the pairs.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from simlife.config import Config
from simlife.recorder import load_survivors
from simlife.senses import observe, wrap_delta
from simlife.state import init_world
from simlife.step import run_ticks

FOOD_DIR, NEST_DIR, CARRYING = 1, 3, 9
GATE = 6.0  # large enough to saturate tanh
FOOD_GAIN, NEST_GAIN, READOUT = 2.0, 2.0, 1.5


def handwired(cfg: Config, food: bool, nest: bool, nest_gain: float = NEST_GAIN) -> dict:
    H, I, A, P = cfg.n_hidden, cfg.n_inputs, cfg.n_actions, cfg.n_predictions
    w1, b1 = np.zeros((H, I)), np.zeros(H)
    w2, b2 = np.zeros((A, H)), np.zeros(A)
    if food:  # units 0, 1: on when not carrying
        w1[0, FOOD_DIR], w1[1, FOOD_DIR] = FOOD_GAIN, -FOOD_GAIN
        w1[0, CARRYING] = w1[1, CARRYING] = -GATE
        w2[0, 0], w2[0, 1] = READOUT, -READOUT
    if nest:  # units 2, 3: on when carrying
        for unit, sign in [(2, 1), (3, -1)]:
            w1[unit, NEST_DIR] = sign * nest_gain
            w1[unit, CARRYING], b1[unit] = GATE, -GATE
        w2[0, 2], w2[0, 3] = READOUT, -READOUT
    b2[1] = 3.0  # speed: sigmoid(3) = 0.95
    b2[5] = -3.0  # aggression low (it does nothing during the grace period anyway)
    f = lambda a: jnp.asarray(a, jnp.float32)
    return dict(w1=f(w1), b1=f(b1), w2=f(w2), b2=f(b2), w3=f(np.zeros((P, H))), b3=f(np.zeros(P)))


def plant(state, cfg, net):
    """Give every third founding colony the hand-wired network."""
    chosen = (state.colony % 3 == 0) & state.alive
    def put(field, value):
        mask = chosen.reshape((-1,) + (1,) * (field.ndim - 1))
        return jnp.where(mask, value[None], field)
    params = {k: put(state.params[k], net[k]) for k in net}
    birth = {k: put(state.birth_params[k], net[k]) for k in net}
    return state._replace(params=params, birth_params=birth, lr=jnp.where(chosen, 0.0, state.lr))


def is_handwired(state, net):
    w = np.asarray(state.params["w1"]) - np.asarray(net["w1"])[None]
    return np.linalg.norm(w.reshape(len(w), -1), axis=1) / np.linalg.norm(np.asarray(net["w1"])) < 0.5


def behaviour(state, cfg, kinds):
    """Fraction carrying; carriers' mean facing toward their nest; foragers'
    food steering (correlation of turn with food direction), per kind."""
    from simlife.brain import act
    obs = observe(state, cfg)
    turn = np.asarray(jax.vmap(act)(state.params, obs)[0])
    obs = np.asarray(obs)
    to_nest = np.asarray(wrap_delta(state.nest[state.colony], state.pos))
    h = np.asarray(state.heading)
    facing = (np.cos(h) * to_nest[:, 0] + np.sin(h) * to_nest[:, 1]) / (np.linalg.norm(to_nest, axis=1) + 1e-9)
    carrying = np.asarray(state.carrying)
    return {k: (m, carrying[m], facing[m], turn[m], obs[m, FOOD_DIR]) for k, m in kinds.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nursery", type=Path)
    parser.add_argument("--variant", choices=["full", "food-only", "nest-only"], default="full")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ticks", type=int, default=15000)
    parser.add_argument("--nest-gain", type=float, default=NEST_GAIN)
    args = parser.parse_args()

    cfg = Config.from_dict(json.loads((args.nursery / "settings.json").read_text())["config"])
    assert args.ticks < cfg.grace_ticks, "stay inside the grace period"
    params, lr = load_survivors(args.nursery, cfg)
    founders = {"params": params, "lr": lr}
    net = handwired(cfg, food=args.variant != "nest-only", nest=args.variant != "food-only", nest_gain=args.nest_gain)
    state = plant(init_world(jax.random.PRNGKey(args.seed), cfg, founders), cfg, net)

    print(f"{args.variant}, seed {args.seed}: 6 hand-wired founding colonies, 12 evolved (initial members each: {cfg.initial_members})")
    print("   tick   hand-wired members (colonies)   evolved members (colonies)   delivered")
    samples = {"hand-wired": [], "evolved": []}
    for t in range(0, args.ticks, 1000):
        state = run_ticks(state, cfg, 1000)
        alive = np.asarray(state.alive)
        hw = is_handwired(state, net) & alive
        ev = ~hw & alive
        col = np.asarray(state.colony)
        print(f"  {int(state.tick):6d}   {hw.sum():6d} ({len(np.unique(col[hw])):3d})"
              f"                  {ev.sum():6d} ({len(np.unique(col[ev])):3d})"
              f"                   {int(state.delivered_total):6d}")
        if t >= args.ticks - 5000:  # behaviour over the last 5,000 ticks
            for k, (m, c, f, tu, fd) in behaviour(state, cfg, {"hand-wired": hw, "evolved": ev}).items():
                samples[k].append((c, f, tu, fd))
        if state.alive.sum() == 0:
            break

    print("\nbehaviour, last 5,000 ticks      carrying   carriers facing nest   foragers steer by food")
    for k, rows in samples.items():
        if not rows:
            continue
        c, f, tu, fd = (np.concatenate(x) for x in zip(*rows))
        n = ~c
        steer = np.corrcoef(tu[n], fd[n])[0, 1] if n.sum() > 20 and tu[n].std() > 0 else float("nan")
        face = f[c].mean() if c.sum() else float("nan")
        print(f"  {k:12s} ({len(c):6d} samples)  {c.mean():6.0%}   {face:+19.2f}   {steer:+20.2f}")


if __name__ == "__main__":
    main()
