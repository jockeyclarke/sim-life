# sim-life

A small artificial-life simulation. Colonies of entities, each driven by a tiny
neural network, forage for food in a shared world. There is no fitness
function: a colony lives while its shared energy pool lasts, grows when it has
a surplus, splits when it gets large, and dies when it runs out.

This is stage 1 of three: evolution only. Networks change only by mutation
when a member divides. Lifetime learning (stage 2) and poison (stage 3) are
described in [the design](docs/superpowers/specs/2026-11-12-colony-life-sim-design.md).

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

Stop the engine with Ctrl-C. It saves a checkpoint on the way out.

### Without a viewer

```sh
.venv/bin/python -m simlife run --no-serve --ticks 300000 --seed 1
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
| `--runs-dir DIR` | `runs` | Where runs are recorded. |
| `--host`, `--port` | 127.0.0.1, 8765 | Where the viewer server listens. |

## The viewer

- **World.** Rings are nests, green dots are food, short marks are members
  pointing along their heading. A dot at a member's head means it is carrying
  food.
- **Ticks per frame.** How many simulation ticks run between redraws, from 1
  to 1000.
- **Start fresh colonies when all have died.** If every colony dies, the run
  begins again with new random colonies and marks the moment on the charts.
  Colonies are never added while any colony is alive.
- **Population.** Colonies, members, total energy, and food delivered per
  member per tick. The last is the measure of how well the population forages.
- **Compare with run.** Overlays a recorded run on the charts as a dashed line.

Page options: `?port=8765` sets the engine's port, and `?compare=previous`
(or `?compare=<run id>`) preselects a comparison run.

## Recorded runs

Each run gets a directory under `runs/`:

| File | Contents |
|---|---|
| `settings.json` | The seed and every setting. |
| `history.jsonl` | One record every 100 ticks. |
| `checkpoints/state-<tick>.npz` | The full world state, every 50,000 ticks and on exit. |

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
| [brain.py](engine/simlife/brain.py) | The network, written for one entity and lifted over the population with `vmap`. |
| [senses.py](engine/simlife/senses.py) | The 10 inputs; neighbour statistics without comparing every pair. |
| [physics.py](engine/simlife/physics.py) | Movement, food pickup with contention, delivery, energy costs. |
| [lifecycle.py](engine/simlife/lifecycle.py) | Births into fixed-capacity arrays, mutation, colony splitting. |
| [step.py](engine/simlife/step.py) | The order of one tick, and the compiled loop. |
| [runner.py](engine/simlife/runner.py) | Recording, checkpoints, and what happens at extinction. |
| [protocol.py](engine/simlife/protocol.py), [server.py](engine/simlife/server.py) | The snapshot format and the viewer server. |

## Tests

```sh
.venv/bin/pytest                 # engine, about 30 seconds
.venv/bin/pytest -m slow         # does a default world take hold? about 3 minutes
cd viewer && npm test            # viewer
```

After changing the snapshot format or the default starting population,
regenerate the viewer's test fixture:

```sh
.venv/bin/python engine/tools/write_fixture.py
```
