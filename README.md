# sim-life

A small artificial-life simulation. Colonies of entities, each driven by a tiny
neural network, forage for food in a shared world. There is no fitness
function: a colony lives while its shared energy pool lasts, grows when it has
a surplus, splits when it gets large, and dies when it runs out.

All three stages are built. Networks change in two ways: by mutation and
two-parent crossover when a colony gains a member, and during life, as each
member learns to predict its own next senses. A world starts with independent colonies that have 20,000 ticks
to learn to forage. Then they are divided by region into six **families**
that mark territory, pay upkeep on foreign ground, and can evolve to strike
members of other families. See
[the design](docs/superpowers/specs/2026-11-12-colony-life-sim-design.md).

## Setup

Requires Python 3.12 or newer and Node 22 or newer.

```sh
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cd viewer && npm install
```

## Running

Start the engine. It runs flat out when nobody is watching and slows to 30
frames a second when a viewer connects.

```sh
.venv/bin/python -m simlife run --seed 0
```

In a second terminal, start the viewer and open the address it prints
(normally http://localhost:5173):

```sh
cd viewer && npm run dev
```

Stop the engine with Ctrl-C. It saves a checkpoint on the way out, and a
resumed run keeps the switches it was left with.

### Without a viewer

```sh
.venv/bin/python -m simlife run --no-serve --ticks 300000 --seed 1
```

### Stage 3: evolve, then seed

Worlds started from random networks rarely produce more than one family that
can forage, so the contest between families seldom begins. The reliable way
is two steps:

1. **Evolve a source population** (a "nursery"): a headless run long enough
   to cover the grace period, during which nothing competes.

   ```sh
   .venv/bin/python -m simlife run --no-serve --ticks 20000 --seed 11 --runs-dir runs
   ```

   A nursery is already provided at `runs/nursery` (seed 11, 449 members).

2. **Seed contest worlds from it.** Each founding colony starts from one of
   the nursery's survivors; families form by region when the new world's
   grace period ends.

   ```sh
   .venv/bin/python -m simlife run --seed 1 --seed-from runs/nursery
   ```

### Resuming a run

```sh
.venv/bin/python -m simlife run --resume runs/<run id>
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `--seed N` | 0 | Random seed. The same seed and settings give the same run. |
| `--ticks N` | none | Stop after N ticks. |
| `--no-serve` | off | Don't start the viewer server. Requires `--ticks`. |
| `--resume RUN_DIR` | none | Continue a recorded run from its latest checkpoint. |
| `--no-reseed` | off | When every colony has died, stop instead of starting fresh ones. |
| `--no-learning` | off | Switch lifetime learning off, for a pure-evolution run. |
| `--inherit learned\|birth` | learned | What offspring copy: what the parent learned, or what the parent was born with. |
| `--seed-from RUN_DIR` | none | Found this run's colonies from the living members of an earlier run's latest checkpoint. The survivors are copied into the new run, so its fresh starts and resumes use them too. |
| `--runs-dir DIR` | `runs` | Where runs are recorded. |
| `--host`, `--port` | 127.0.0.1, 8765 | Where the viewer server listens. |

## The viewer

- **World.** Rings are nests, green dots are food, short marks are members
  pointing along their heading. A dot at a member's head means it is carrying
  food. Each family has its own colour and its own nest-ring pattern (six
  colours are not all distinguishable for everyone, so the pattern is a
  second cue). Crosses mark this tick's strikes.
- **Shade territory.** Tints the ground by which family's scent dominates
  it, more strongly where that family's hold is stronger.
- **Ticks per frame.** How many simulation ticks run between redraws, from 1
  to 1000.
- **Start fresh colonies when all have died.** If every colony dies, the run
  begins again with new random colonies and marks the moment on the charts.
  Colonies are never added while any colony is alive.
- **Members learn during their lives.** Switches lifetime learning on or off.
  Prediction error is still measured when it is off, so the two can be
  compared.
- **Offspring inherit.** Whether a new member copies what its parent learned
  or what its parent was born with. Both switches can be flipped mid-run;
  each flip is marked on the charts and recorded in the run's history.
- **Colour members** by family (the default), colony, carrying food,
  prediction error, or aggression. **Food in transit** shows only the
  carriers, each leaving a fading trail, with everyone else as a faint dot,
  so you can watch food flow from the field into the nests.
- **Inspector.** Click a member to select it; click empty space to clear.
  Its home nest gets a bold ring, with a faint dashed line from the member.
  The panel shows what it senses, its hidden activations and actions this
  tick, its last prediction against what happened, its prediction error
  since you selected it, and its weights. "Show change since selected"
  switches the weights to how far each has moved: the action weights never
  move during a life, the others do.
- **Population.** Colonies, members, total energy, food delivered per member
  per tick, mean prediction error, and mean learning rate. The learning rate
  is inherited and mutates, so its drift shows whether evolution is turning
  learning up or down. Stage 3 adds **territory by family** (each family's
  share of the ground; a takeover shows as one line rising while the others
  fall to zero, and is marked with a rule) and **strikes per 100 ticks**.
- **Compare with run.** Overlays a recorded run on the charts as a dashed line.

Page options: `?port=8765` sets the engine's port, and `?compare=previous`
(or `?compare=<run id>`) preselects a comparison run.

## Recorded runs

Each run gets a directory under `runs/`:

| File | Contents |
|---|---|
| `settings.json` | The seed and every setting. |
| `history.jsonl` | One record of measures every 100 ticks, plus a line for each extinction or fresh start at the tick it happened. |
| `checkpoints/state-<tick>.npz` | The full world state, every 50,000 ticks and on exit. |

Runs recorded by an earlier stage cannot be resumed, because the network and
world have changed shape. Their histories still load for comparison.

## Settings

Every tunable number is a field of `Config` in
[engine/simlife/config.py](engine/simlife/config.py). There are no
command-line overrides yet: edit the defaults there. A resumed run uses the
settings it was started with.

## Reading the code

The engine is written so the whole simulation is one pure function,
`step(state, cfg) -> state`, over a record of fixed-shape arrays.

| File | What to read it for |
|---|---|
| [state.py](engine/simlife/state.py) | The world as one record of arrays; how it is initialised. |
| [brain.py](engine/simlife/brain.py) | The network and its learning step, written for one entity and lifted over the population with `vmap`. The gradient comes from `jax.value_and_grad`. |
| [learning.py](engine/simlife/learning.py) | The learning phase of a tick: score last tick's prediction, take a gradient step. |
| [senses.py](engine/simlife/senses.py) | The 10 inputs; neighbour statistics without comparing every pair. |
| [physics.py](engine/simlife/physics.py) | Movement, food pickup with contention, delivery, energy costs. |
| [lifecycle.py](engine/simlife/lifecycle.py) | Births into fixed-capacity arrays, mutation, colony splitting. |
| [step.py](engine/simlife/step.py) | The order of one tick, and the compiled loop. |
| [runner.py](engine/simlife/runner.py) | Recording, checkpoints, and what happens at extinction. |
| [territory.py](engine/simlife/territory.py) | Each family's scent field: laid, spread, faded; who owns which ground. |
| [combat.py](engine/simlife/combat.py) | Strikes: home advantage, costs, and how they are counted. |
| [inspect.py](engine/simlife/inspect.py) | One member's internals read out of the population arrays. |
| [protocol.py](engine/simlife/protocol.py), [server.py](engine/simlife/server.py) | The snapshot format and the viewer server. |

## Tests

```sh
.venv/bin/pytest                 # engine, about a minute
.venv/bin/pytest -m slow         # do seeded worlds reach the contest? about 20 minutes
cd viewer && npm test            # viewer
```

After changing the snapshot format or the default starting population,
regenerate the viewer's test fixture:

```sh
.venv/bin/python engine/tools/write_fixture.py
```
