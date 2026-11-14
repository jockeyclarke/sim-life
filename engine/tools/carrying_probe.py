"""Do members behave differently when carrying food?

Loads a run's latest checkpoint, runs the world on for a while, and samples
every living member's network outputs. Two measures, per family:

1. Behaviour as it happens: mean speed and turn, how squarely members face
   their own nest, and how strongly they steer toward nest scent and food
   scent, split by whether they are carrying.
2. A flip test: the same member, the same moment, with only the "carrying"
   input changed. This asks the network directly whether that input matters,
   with nothing else about the situation changing.

    .venv/bin/python engine/tools/carrying_probe.py runs/<run id> [--ticks 5000] [--every 10]

Steering is the correlation between the turn output and how much stronger a
scent is on one sensor than the other, signed so that +1 means always turning
toward the stronger side, -1 always away, and 0 that the scent doesn't steer.
(A positive turn increases the heading, which swings it toward the sensor
placed at heading + sensor_angle, the one observe() calls "left".)
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from simlife.brain import act
from simlife.config import Config
from simlife.recorder import load_state
from simlife.senses import observe, wrap_delta
from simlife.step import step

CARRYING = 9  # observation column


def outputs(params, obs):
    turn, speed, *_ = jax.vmap(act)(params, obs)
    return turn, speed


@jax.jit
def sample(state, obs, flipped):
    turn, speed = outputs(state.params, obs)
    turn_f, speed_f = outputs(state.params, flipped)
    to_nest = wrap_delta(state.nest[state.colony], state.pos)
    facing = jnp.stack([jnp.cos(state.heading), jnp.sin(state.heading)], 1)
    norm = jnp.linalg.norm(to_nest, axis=1) + 1e-9
    nest_facing = (facing * to_nest).sum(1) / norm
    return dict(
        turn=turn, speed=speed, turn_f=turn_f, speed_f=speed_f,
        nest_facing=nest_facing,
        nest_lr=obs[:, 3],  # nest direction: + means nest on the left
        food_lr=obs[:, 1],  # food direction: + means food on the left
    )


def corr(a, b):
    if len(a) < 20 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--ticks", type=int, default=5000)
    parser.add_argument("--every", type=int, default=10)
    args = parser.parse_args()

    # read-only: load the latest checkpoint without touching the run's files
    cfg = Config.from_dict(json.loads((args.run / "settings.json").read_text())["config"])
    state = load_state(sorted((args.run / "checkpoints").glob("state-*.npz"))[-1], cfg)
    start = int(state.tick)
    tick = jax.jit(step, static_argnames="cfg")

    rows = {k: [] for k in ["family", "carrying", "turn", "speed", "turn_f", "speed_f",
                            "nest_facing", "nest_lr", "food_lr"]}
    for t in range(args.ticks):
        state = tick(state, cfg)
        if t % args.every:
            continue
        obs = observe(state, cfg)
        flipped = obs.at[:, CARRYING].set(1.0 - obs[:, CARRYING])
        s = jax.device_get(sample(state, obs, flipped))
        alive = np.asarray(state.alive)
        rows["family"].append(np.asarray(state.col_family)[np.asarray(state.colony)][alive])
        rows["carrying"].append(np.asarray(state.carrying)[alive])
        for k, v in s.items():
            rows[k].append(np.asarray(v)[alive])
    d = {k: np.concatenate(v) for k, v in rows.items()}

    print(f"{args.run.name}: from tick {start:,}, {args.ticks:,} ticks, "
          f"{len(d['family']):,} member-samples\n")
    print("behaviour as it happens          not carrying | carrying")
    for fam in ["all", *np.unique(d["family"])]:
        m = np.ones_like(d["carrying"]) if fam == "all" else d["family"] == fam
        c, n = m & d["carrying"], m & ~d["carrying"]
        if c.sum() < 20 or n.sum() < 20:
            continue
        print(f"  family {fam}:  {m.sum():,} samples, {c.sum() / m.sum():.0%} carrying")
        def line(label, f):
            print(f"    {label:28s} {f(n):+7.3f}      | {f(c):+7.3f}")
        line("speed (0..1)", lambda x: d["speed"][x].mean())
        line("|turn| (0..1)", lambda x: np.abs(d["turn"][x]).mean())
        line("facing own nest (-1..1)", lambda x: d["nest_facing"][x].mean())
        line("steers by nest scent (corr)", lambda x: corr(d["turn"][x], d["nest_lr"][x]))
        line("steers by food scent (corr)", lambda x: corr(d["turn"][x], d["food_lr"][x]))

    print("\nflip test: same member, same moment, only 'carrying' changed")
    print("                                 as observed | flipped")
    for fam in ["all", *np.unique(d["family"])]:
        m = np.ones_like(d["carrying"]) if fam == "all" else d["family"] == fam
        if m.sum() < 20:
            continue
        print(f"  family {fam}:")
        print(f"    mean |change in speed|       {np.abs(d['speed_f'][m] - d['speed'][m]).mean():.3f}")
        print(f"    mean |change in turn|        {np.abs(d['turn_f'][m] - d['turn'][m]).mean():.3f}")
        # nest steering for foragers if they were carrying, vs real carriers
        n = m & ~d["carrying"]
        if n.sum() >= 20:
            print(f"    foragers' nest steering      {corr(d['turn'][n], d['nest_lr'][n]):+.3f}      | "
                  f"{corr(d['turn_f'][n], d['nest_lr'][n]):+.3f}  (if they were carrying)")


if __name__ == "__main__":
    main()
