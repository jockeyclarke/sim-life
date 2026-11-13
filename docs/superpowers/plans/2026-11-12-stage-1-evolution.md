# Stage 1: Evolution Only — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A headless-capable simulation in which colonies of tiny-network entities forage, grow, split and die under energy alone, with a browser view of the world and of population history.

**Architecture:** A Python engine whose whole simulation is one pure, jitted JAX function `step(state, cfg) -> state` over a fixed-capacity record of arrays. A `Runner` advances it in chunks, records history and checkpoints to disk, and handles extinction. A WebSocket server publishes binary snapshots to a separate TypeScript canvas viewer, which sends back JSON control messages.

**Tech Stack:** Python 3.14, JAX (CPU), NumPy, `websockets`, pytest. TypeScript, Vite, Vitest, Canvas 2D. No UI framework.

**Spec:** `docs/superpowers/specs/2026-11-12-colony-life-sim-design.md`. This plan implements its "Stages → 1. Evolution only". Stages 2 and 3 get their own plans.

## Global Constraints

- No fitness function, score or reward anywhere. Survival and reproduction follow only from energy.
- No Python loop over entities, colonies or food. Per-entity code is written for one entity and lifted with `jax.vmap`; everything else is whole-array.
- Fixed-capacity arrays with `alive` masks for members, colonies and food. Shapes never change after `init_world`.
- All randomness comes from explicit JAX keys carried in the state. Same seed and same `Config` give bit-identical runs, however the ticks are chunked.
- New colonies are never created while any colony is alive, except by a colony splitting.
- Weights are never averaged across members.
- The signal and memory values are never given meaning by engine code beyond being passed in and out of the network.
- JAX runs on CPU. Do not install or enable any Apple GPU backend.
- Python: use the installed `python3` (3.14.7) in a project `.venv`. `jax>=0.12`. Node 26.
- The world is the unit square `[0, 1)²` with wrap-around edges. All distances use the wrapped (torus) metric.
- Stage 1 network: 10 inputs, 8 hidden (tanh), 5 action outputs. No prediction outputs, no lifetime learning, no stored birth weights, no poison. Do not build these early.
- Work on branch `stage-1-evolution`. End each commit message with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.

## Review Focus

1. **Capacity full.** With every member slot (or colony slot) in use, a colony that qualifies to grow (or split) simply doesn't; no energy is charged, nothing crashes, no live slot is overwritten. → Tasks 5, 6.
2. **Numeric blow-up over a long run.** After thousands of ticks of mutation and movement, no state array contains NaN or Inf and every position stays in `[0, 1)`. → Task 7.
3. **Extinction.** When the last colony dies the engine neither spins nor crashes: with reseeding on it starts fresh colonies and logs the event; with it off it idles and reports `extinct`. → Task 8.
4. **Interrupted or mismatched checkpoint.** A half-written checkpoint file is never picked up on resume, and resuming a run directory whose saved shapes don't match its settings gives a clear error. → Task 8.
5. **Viewer misbehaviour.** A viewer that disconnects mid-run, connects late, sends malformed JSON or an unknown message type never stops the engine; two viewers at once both receive snapshots. → Task 9.

---

## File Structure

```
pyproject.toml
README.md
engine/simlife/
  config.py      Config: every tunable number, hashable, serialisable
  state.py       WorldState record and init_world
  brain.py       the network for one entity
  senses.py      scents, neighbour statistics, the 10-value observation
  physics.py     movement, food pickup, delivery, metabolism
  lifecycle.py   ageing, colony death, growth, splitting, food regrowth
  step.py        step and run_ticks
  metrics.py     population measures from a state
  recorder.py    run directory: settings, history, checkpoints
  runner.py      advances the sim; recording cadence; extinction
  protocol.py    snapshot encoding, control-message parsing
  server.py      WebSocket server and pacing loop
  __main__.py    command line
engine/tools/write_fixture.py   writes a snapshot fixture for viewer tests
engine/tests/test_*.py          one per module
viewer/
  package.json, tsconfig.json, vite.config.ts, index.html
  src/protocol.ts        snapshot decoding and message types
  src/connection.ts      WebSocket with reconnect
  src/worldView.ts       canvas drawing of the world
  src/populationView.ts  history charts with overlay
  src/controls.ts        pause, speed, switches, run picker
  src/main.ts            wiring
  src/__fixtures__/snapshot.bin
  src/*.test.ts
runs/                    created at run time, git-ignored
```

---

### Task 1: Project scaffold, Config, WorldState, init_world

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `engine/simlife/__init__.py`, `engine/simlife/config.py`, `engine/simlife/state.py`
- Test: `engine/tests/test_state.py`

**Interfaces:**
- Produces: `Config` (frozen dataclass; `to_dict() -> dict`, `Config.from_dict(d) -> Config`), `WorldState` (NamedTuple), `init_world(key: Array, cfg: Config) -> WorldState`.

`Config` fields and defaults (all are starting values; Task 12 tunes them):

| Field | Default | Field | Default |
|---|---|---|---|
| `capacity_members` | 2048 | `max_speed` | 0.004 |
| `capacity_colonies` | 64 | `max_turn` | 0.5 (radians/tick) |
| `capacity_food` | 512 | `sensor_distance` | 0.02 |
| `num_patches` | 8 | `sensor_angle` | 0.6 (radians) |
| `patch_radius` | 0.05 | `food_scent_length` | 0.08 |
| `initial_colonies` | 16 | `nest_scent_length` | 0.15 |
| `initial_members` | 12 | `pickup_radius` | 0.01 |
| `initial_energy` | 60.0 | `nest_radius` | 0.02 |
| `n_inputs` | 10 | `food_energy` | 5.0 |
| `n_hidden` | 8 | `base_cost` | 0.002 |
| `n_actions` | 5 | `speed_cost` | 0.004 |
| `init_sigma` | 0.5 | `birth_cost` | 4.0 |
| `mutation_sigma` | 0.03 | `reserve_per_member` | 2.0 |
| `lifespan_mean` | 4000 | `split_size` | 32 |
| `lifespan_jitter` | 1000 | `split_distance` | 0.15 |
| `neighbour_grid` | 32 | `food_regrow_prob` | 0.002 |

`WorldState` fields (N, C, F, P = the four capacities):

| Field | Shape, dtype | Field | Shape, dtype |
|---|---|---|---|
| `key` | PRNG key | `nest` | (C,2) f32 |
| `tick` | () i32 | `energy` | (C,) f32 |
| `pos` | (N,2) f32 | `col_alive` | (C,) bool |
| `heading` | (N,) f32 | `col_generation` | (C,) i32 |
| `alive` | (N,) bool | `food_pos` | (F,2) f32 |
| `colony` | (N,) i32 | `food_alive` | (F,) bool |
| `carrying` | (N,) bool | `patch_pos` | (P,2) f32 |
| `age` | (N,) i32 | `delivered_total` | () i32 |
| `lifespan` | (N,) i32 | `params` | dict: `w1` (N,8,10), `b1` (N,8), `w2` (N,5,8), `b2` (N,5), all f32 |
| `memory` | (N,2) f32 | | |
| `signal` | (N,) f32 | | |
| `generation` | (N,) i32 | | |

- [ ] **Step 1: Scaffold.** `git switch -c stage-1-evolution`. Create `pyproject.toml` (package `simlife` under `engine/`, deps `jax>=0.12`, `numpy`, `websockets`; dev dep `pytest`; pytest `testpaths = ["engine/tests"]`). `.gitignore`: `.venv/`, `runs/`, `node_modules/`, `dist/`, `__pycache__/`. Run `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`. Verify: `.venv/bin/python -c "import jax; print(jax.__version__, jax.devices())"` prints a version ≥ 0.12 and a CPU device.

- [ ] **Step 2: Write the failing tests**

```python
def test_config_roundtrip_and_hashable():
    cfg = Config(capacity_members=128)
    assert Config.from_dict(cfg.to_dict()) == cfg
    hash(cfg)

def test_init_world_shapes_and_counts():
    cfg = Config()
    s = init_world(jax.random.key(0), cfg)
    assert s.pos.shape == (2048, 2) and s.params["w1"].shape == (2048, 8, 10)
    assert int(s.alive.sum()) == 16 * 12 and int(s.col_alive.sum()) == 16
    assert int(s.tick) == 0 and int(s.delivered_total) == 0
    assert bool(((s.pos >= 0) & (s.pos < 1)).all())

def test_initial_colony_is_clonal_and_colonies_differ():
    s = init_world(jax.random.key(0), Config())
    w = s.params["w1"]; m0 = s.alive & (s.colony == 0); m1 = s.alive & (s.colony == 1)
    assert bool((w[m0] == w[m0][0]).all())           # members of a colony share weights
    assert not bool((w[m0][0] == w[m1][0]).all())    # colonies differ

def test_initial_members_start_at_nest_with_energy():
    cfg = Config(); s = init_world(jax.random.key(0), cfg)
    assert bool((s.energy[s.col_alive] == cfg.initial_energy).all())
    assert bool((s.lifespan[s.alive] >= cfg.lifespan_mean - cfg.lifespan_jitter).all())
    assert not bool(s.carrying.any())
```

- [ ] **Step 3: Run** `.venv/bin/pytest engine/tests/test_state.py -v`. Expected: FAIL (imports missing).

- [ ] **Step 4: Implement** `Config`, `WorldState`, `init_world`. Colonies occupy colony slots `0..initial_colonies-1`; colony `c`'s members occupy contiguous member slots. Each colony draws one network with `normal * init_sigma` and copies it to its members. Members start at their nest with random headings, random ages in `[0, lifespan/2)` so they don't all die on the same tick, `generation` 0. Nests and patch centres are uniform random; all `capacity_food` pellets start alive, each uniformly within `patch_radius` of a patch (pellet `i` belongs to patch `i % num_patches`).

- [ ] **Step 5: Run** the tests. Expected: PASS.

- [ ] **Step 6: Commit** (include the spec and this plan): `git add -A && git commit -m "feat: scaffold, Config, WorldState, init_world"`

---

### Task 2: Brain

**Files:**
- Create: `engine/simlife/brain.py`
- Test: `engine/tests/test_brain.py`

**Interfaces:**
- Consumes: `Config`.
- Produces: `init_params(key, cfg) -> dict` (one entity, no leading axis); `act(params: dict, x: Array[10]) -> tuple[turn (), speed (), signal (), memory (2,)]`; `act_all = jax.vmap(act)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_act_ranges_and_shapes():
    p = init_params(jax.random.key(1), Config())
    turn, speed, signal, mem = act(p, jnp.ones(10) * 3.0)
    assert turn.shape == () and mem.shape == (2,)
    assert -1 <= turn <= 1 and 0 <= speed <= 1 and -1 <= signal <= 1
    assert bool((jnp.abs(mem) <= 1).all())

def test_act_all_matches_single():
    cfg = Config(); keys = jax.random.split(jax.random.key(2), 5)
    ps = jax.vmap(lambda k: init_params(k, cfg))(keys)
    xs = jax.random.normal(jax.random.key(3), (5, 10))
    turn_all, *_ = act_all(ps, xs)
    turn_3, *_ = act(jax.tree.map(lambda a: a[3], ps), xs[3])
    assert jnp.allclose(turn_all[3], turn_3)

def test_zero_weights_give_neutral_output():
    p = jax.tree.map(jnp.zeros_like, init_params(jax.random.key(0), Config()))
    turn, speed, signal, mem = act(p, jnp.ones(10))
    assert turn == 0 and speed == 0.5 and signal == 0
```

- [ ] **Step 2: Run** `.venv/bin/pytest engine/tests/test_brain.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** `h = tanh(w1 @ x + b1)`, `o = w2 @ h + b2`; `turn = tanh(o[0])`, `speed = sigmoid(o[1])`, `signal = tanh(o[2])`, `memory = tanh(o[3:5])`. Scaling to world units happens in physics, not here.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: single-entity network with vmapped population form`.

---

### Task 3: Senses

**Files:**
- Create: `engine/simlife/senses.py`
- Test: `engine/tests/test_senses.py`

**Interfaces:**
- Consumes: `WorldState`, `Config`.
- Produces:
  - `wrap_delta(a, b) -> Array` — `a - b` on the torus, each component in `[-0.5, 0.5)`.
  - `sensor_points(pos, heading, cfg) -> tuple[left (N,2), right (N,2)]` — points `sensor_distance` away at `heading ± sensor_angle`, wrapped.
  - `food_scent(points (M,2), food_pos, food_alive, cfg) -> (M,)` — `sum over live pellets of exp(-d / food_scent_length)`.
  - `nest_scent(points (M,2), nest (M,2), cfg) -> (M,)` — `exp(-d / nest_scent_length)`.
  - `neighbour_stats(state, cfg) -> tuple[crowd (N,), heard (N,)]`.
  - `observe(state, cfg) -> (N,10)` in this fixed order: food L, food R, nest L, nest R, crowd, heard, carrying, colony energy, memory 0, memory 1.

Definitions the tests rely on: "left" is `heading + sensor_angle`. Neighbours are alive members of the **same colony** in the same cell of a `neighbour_grid × neighbour_grid` grid, excluding self. `crowd = tanh(count / 8)`. `heard` = mean `signal` of those neighbours, 0 when there are none. Colony energy input = `tanh(energy[colony] / 100)`. Food scent input = `tanh` of the raw sum. Dead members' rows are all zero.

- [ ] **Step 1: Write the failing tests** (build small states with `Config(capacity_members=8, capacity_colonies=2, capacity_food=4, num_patches=1, initial_colonies=2, initial_members=2)` and `state._replace(...)` to place things by hand)

```python
def test_wrap_delta_takes_short_way_round():
    d = wrap_delta(jnp.array([0.05, 0.5]), jnp.array([0.95, 0.5]))
    assert jnp.allclose(d, jnp.array([0.10, 0.0]), atol=1e-6)

def test_food_on_the_left_smells_stronger_on_the_left():
    # member at (0.5,0.5) heading 0 (along +x); one live pellet at (0.5,0.56)
    obs = observe(s, cfg)
    assert obs[0, 0] > obs[0, 1]

def test_dead_food_has_no_scent():
    # same, with food_alive all False
    assert obs[0, 0] == 0 and obs[0, 1] == 0

def test_nest_scent_peaks_at_own_nest_only():
    # member sitting on colony 1's nest but belonging to colony 0, far from nest 0
    assert obs[0, 2] < 0.1 and obs[0, 3] < 0.1

def test_neighbours_same_colony_same_cell_only():
    # members 0,1 same colony same cell, signal[1]=0.8; member 2 other colony same cell, signal 0.3
    crowd, heard = neighbour_stats(s, cfg)
    assert jnp.isclose(crowd[0], jnp.tanh(1 / 8)) and jnp.isclose(heard[0], 0.8)
    assert crowd[2] == 0 and heard[2] == 0

def test_observation_passes_through_carrying_and_memory_and_zeroes_dead():
    # member 0 carrying, memory (0.3,-0.7); member 5 dead
    assert obs[0, 6] == 1 and jnp.allclose(obs[0, 8:], jnp.array([0.3, -0.7]))
    assert bool((obs[5] == 0).all())
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** For `neighbour_stats`, compute a flat index `colony * G*G + cell`, use `jax.ops.segment_sum` for per-bucket count and signal sum over alive members, gather back per member, then subtract the member's own contribution.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: senses and observation vector`.

---

### Task 4: Physics — movement, pickup, delivery, metabolism

**Files:**
- Create: `engine/simlife/physics.py`
- Test: `engine/tests/test_physics.py`

**Interfaces:**
- Consumes: `wrap_delta`, `WorldState`, `Config`.
- Produces (each returns a new `WorldState`):
  - `move(state, turn (N,), speed (N,), cfg)` — `heading += turn * max_turn`; `pos += speed * max_speed * (cos, sin)`; wrap into `[0,1)`. Dead members don't move.
  - `pickup(state, cfg)` — each alive, non-carrying member targets its nearest live pellet if within `pickup_radius`; a pellet targeted by several members goes to the lowest member index; the winner's `carrying` becomes True and the pellet's `food_alive` False.
  - `deliver(state, cfg)` — a carrying alive member within `nest_radius` of its own nest stops carrying; its colony gains `food_energy`; `delivered_total` rises by one.
  - `metabolise(state, speed (N,), cfg)` — each alive member costs its colony `base_cost + speed_cost * speed²` (speed in `[0,1]`).

- [ ] **Step 1: Write the failing tests** (hand-built small states as in Task 3)

```python
def test_move_wraps_and_ignores_dead():
    # member 0 at (0.999,0.5) heading 0, speed 1 → x wraps to ~0.003; dead member unchanged
def test_pickup_conserves_food_under_contention():
    # members 0 and 1 both within pickup_radius of the single live pellet
    s2 = pickup(s, cfg)
    assert bool(s2.carrying[0]) and not bool(s2.carrying[1])
    assert int(s2.food_alive.sum()) == int(s.food_alive.sum()) - 1
def test_carrying_member_does_not_pick_up():
    # member already carrying, sitting on a pellet → pellet stays alive
def test_deliver_credits_own_colony_only():
    # carrying member of colony 0 on nest 1 → nothing; on nest 0 → energy[0] += 5.0, delivered_total == 1
def test_metabolise_charges_colony_pool():
    # 2 alive members of colony 0 at speed 0 and 1 → energy[0] drops by 0.002*2 + 0.004
    # colony 1 with no alive members → unchanged
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Pickup uses one (N,F) wrapped-distance matrix, `argmin` per member, and `jax.ops.segment_min` of member index over targeted pellets to pick winners.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: movement, food pickup, delivery, metabolism`.

---

### Task 5: Lifecycle I — ageing, colony death, growth

**Files:**
- Create: `engine/simlife/lifecycle.py`
- Test: `engine/tests/test_lifecycle.py`

**Interfaces:**
- Consumes: `WorldState`, `Config`.
- Produces:
  - `age_and_die(state, cfg)` — alive members' `age += 1`; members with `age >= lifespan` die (`alive`, `carrying` False).
  - `kill_dead_colonies(state)` — a colony with `energy <= 0` or no alive members becomes dead; all its members die; its `energy` is set to 0.
  - `grow(state, cfg, key)` — at most one birth per colony per tick.

Growth rule: colony `c` wants a birth when alive and `energy[c] > birth_cost + reserve_per_member * member_count[c]`. The parent is a uniformly random alive member of that colony. The child takes a free member slot and gets: parent's `params` plus `normal * mutation_sigma` on every array, parent's `pos` and `colony`, a random heading, `age` 0, a fresh `lifespan` (`lifespan_mean ± uniform lifespan_jitter`), `generation = parent + 1`, `carrying` False, `memory` and `signal` 0. The colony pays `birth_cost`.

Slot allocation, since the tests don't determine it:

```python
free = jnp.argsort(state.alive, stable=True)      # free slots first, in index order
n_free = (~state.alive).sum()
rank = jnp.cumsum(want) - 1                        # want: (C,) bool
ok = want & (rank < n_free)
slot = jnp.where(ok, free[rank], N)                # N is out of range
# write every child field with  arr.at[slot].set(child_value, mode="drop")
```

- [ ] **Step 1: Write the failing tests**

```python
def test_member_dies_at_lifespan_and_drops_food():
def test_colony_with_no_energy_dies_with_all_members():
def test_colony_with_no_members_dies():
def test_grow_adds_one_mutated_child_and_charges_energy():
    # colony 0 rich, colony 1 poor
    s2 = grow(s, cfg, jax.random.key(0))
    assert int(s2.alive.sum()) == int(s.alive.sum()) + 1
    child = int(jnp.argmax(s2.alive & ~s.alive))
    assert int(s2.colony[child]) == 0 and int(s2.age[child]) == 0 and int(s2.generation[child]) == 1
    assert jnp.isclose(s2.energy[0], s.energy[0] - cfg.birth_cost) and s2.energy[1] == s.energy[1]
    d = s2.params["w1"][child] - s.params["w1"][0]      # colony 0 is clonal in the fixture
    assert 0 < float(jnp.abs(d).max()) < 10 * cfg.mutation_sigma
def test_grow_never_overwrites_a_live_member():
    # every pre-existing alive slot is bit-identical in s2 across all member fields
def test_grow_when_members_full_does_nothing():      # Review Focus 1
    # alive all True, colony 0 rich
    s2 = grow(s, cfg, key)
    assert bool(jax.tree.all(jax.tree.map(lambda a, b: (a == b).all(), s2._replace(key=s.key), s)))
def test_grow_two_colonies_one_free_slot():          # Review Focus 1
    # two rich colonies, exactly one free slot → exactly one birth, exactly one birth_cost charged
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Pick parents with a (C,N) mask and `argmax` over uniform random scores.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: ageing, colony death, member division with mutation`.

---

### Task 6: Lifecycle II — colony splitting, food regrowth

**Files:**
- Modify: `engine/simlife/lifecycle.py`
- Test: `engine/tests/test_lifecycle.py`

**Interfaces:**
- Produces:
  - `split(state, cfg, key)` — at most one split per tick: the lowest-index alive colony with `member_count >= split_size`, if a free colony slot exists. Each of its members moves to the daughter with probability ½ (independently). The daughter takes the lowest free colony slot, a nest at `parent nest + split_distance` in a random direction (wrapped), half the parent's energy (the parent keeps the other half), and `col_generation = parent + 1`. Moved members keep everything else, including `pos` and `carrying`.
  - `regrow_food(state, cfg, key)` — each dead pellet comes back with probability `food_regrow_prob` at a new uniform position within `patch_radius` of its patch (`i % num_patches`).

- [ ] **Step 1: Write the failing tests**

```python
def test_split_partitions_members_and_energy():
    # colony 0 has split_size members, energy 40
    s2 = split(s, cfg, jax.random.key(0))
    d = int(jnp.argmax(s2.col_alive & ~s.col_alive))
    n_parent = int((s2.alive & (s2.colony == 0)).sum()); n_d = int((s2.alive & (s2.colony == d)).sum())
    assert n_parent + n_d == cfg.split_size and n_d > 0 and n_parent > 0
    assert jnp.isclose(s2.energy[0], 20.0) and jnp.isclose(s2.energy[d], 20.0)
    assert int(s2.col_generation[d]) == int(s.col_generation[0]) + 1
    assert jnp.isclose(jnp.linalg.norm(wrap_delta(s2.nest[d], s2.nest[0])), cfg.split_distance, atol=1e-5)
    assert bool((s2.params["w1"] == s.params["w1"]).all())     # splitting never touches weights
def test_split_below_size_does_nothing():
def test_split_when_colonies_full_does_nothing():      # Review Focus 1
def test_regrow_only_revives_dead_pellets_inside_their_patch():
    # food_regrow_prob=1.0: all pellets alive afterwards; previously-alive positions unchanged;
    # every revived pellet within patch_radius of patch_pos[i % num_patches]
def test_regrow_prob_zero_revives_nothing():
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: colony splitting and food regrowth`.

---

### Task 7: step, run_ticks, determinism, speed

**Files:**
- Create: `engine/simlife/step.py`
- Test: `engine/tests/test_step.py`

**Interfaces:**
- Consumes: everything from Tasks 2–6.
- Produces: `step(state, cfg) -> WorldState`; `run_ticks(state, cfg, n: int) -> WorldState` (jitted, `cfg` and `n` static, `jax.lax.scan` inside).

Order within one tick, fixed: split the state key into this tick's sub-keys → `observe` → `act_all` → store `signal` and `memory` (zero for dead members) → `move` → `pickup` → `deliver` → `metabolise` → `age_and_die` → `kill_dead_colonies` → `grow` → `split` → `regrow_food` → `tick += 1`.

- [ ] **Step 1: Write the failing tests** (use `cfg = Config(capacity_members=256, capacity_colonies=16, capacity_food=64, initial_colonies=4)` to keep them fast)

```python
def test_same_seed_same_run():
    a = run_ticks(init_world(jax.random.key(7), cfg), cfg, 200)
    b = run_ticks(init_world(jax.random.key(7), cfg), cfg, 200)
    assert states_equal(a, b)          # helper: every leaf bit-identical, keys compared via key_data
def test_chunking_does_not_change_the_run():
    s = init_world(jax.random.key(7), cfg)
    assert states_equal(run_ticks(s, cfg, 200), run_ticks(run_ticks(s, cfg, 100), cfg, 100))
def test_different_seed_different_run():
def test_tick_counter_and_shapes_preserved():
def test_long_run_stays_finite_and_in_bounds():        # Review Focus 2
    s = run_ticks(init_world(jax.random.key(0), cfg), cfg, 5000)
    for leaf in jax.tree.leaves(s._replace(key=None)):
        if jnp.issubdtype(leaf.dtype, jnp.floating): assert bool(jnp.isfinite(leaf).all())
    assert bool(((s.pos >= 0) & (s.pos < 1)).all())
def test_energy_ledger_balances_over_one_tick():
    # total energy after == before + food_energy*deliveries − metabolism − birth_cost*births
    # (compute each term from the before/after states; tolerance 1e-3; skip colonies that died this tick)
def test_no_python_loop_over_entities():
    # step traces once: jit it, call twice with different states, assert the trace counter is 1
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** `step` and `run_ticks`.
- [ ] **Step 4: Run.** Expected: PASS. If the finite test fails, find and fix the source (do not mask with `nan_to_num`).
- [ ] **Step 5: Measure speed.** Run `.venv/bin/python -c "..."` timing `run_ticks` for 2000 ticks at default `Config` after one warm-up call. Record ticks/second in the commit message. If it is below 100 ticks/s, replace the pairwise food-scent sum in `senses.food_scent` with a `neighbour_grid`-style binned density field sampled at the sensor points, keep the Task 3 tests passing, and re-measure.
- [ ] **Step 6: Commit** `feat: pure step function and jitted run loop`.

---

### Task 8: Metrics, recorder, runner, headless command line

**Files:**
- Create: `engine/simlife/metrics.py`, `engine/simlife/recorder.py`, `engine/simlife/runner.py`, `engine/simlife/__main__.py`
- Test: `engine/tests/test_recorder.py`, `engine/tests/test_runner.py`

**Interfaces:**
- Produces:
  - `compute_metrics(state) -> dict` with keys `tick, colonies, members, energy, delivered, max_generation` (plain Python ints/floats).
  - `save_state(path: Path, state)`, `load_state(path: Path, cfg: Config) -> WorldState`.
  - `class Recorder`: `Recorder(runs_dir: Path, cfg: Config, seed: int)` creates `runs_dir/<run_id>/` with `settings.json` = `{"run_id", "seed", "config"}`; `run_id` is `YYYYMMDD-HHMMSS-s<seed>`. Methods `record(metrics: dict, events: list[dict] = [])` (appends one JSON line to `history.jsonl`, adding `delivery_rate`), `checkpoint(state)` (writes `checkpoints/state-<tick>.npz`), attribute `run_id`, `run_dir`.
  - `resume_run(run_dir: Path) -> tuple[Recorder, WorldState, Config, int]` (last is the seed) — loads the highest-tick checkpoint and appends to the existing history.
  - `read_history(run_dir: Path) -> list[dict]`; `list_runs(runs_dir: Path) -> list[str]` (newest first).
  - `class Runner`: `Runner(cfg, seed, recorder, state=None)`; attributes `state`, `cfg`, `reseed_on_extinction: bool = True`, `extinct: bool`, `extinctions: int`; `advance(n: int) -> list[dict]` (events that occurred); `status() -> dict` (`run_id, extinct, extinctions, reseed_on_extinction`); `record_interval = 100`, `checkpoint_interval = 50_000`.
  - CLI: `python -m simlife run [--seed 0] [--runs-dir runs] [--resume RUN_DIR] [--ticks N] [--no-serve] [--host 127.0.0.1] [--port 8765]`. This task implements `--no-serve` (requires `--ticks`); Task 9 adds serving.

Rules:
- `delivery_rate = (delivered − previous delivered) / (members × (tick − previous tick))`, 0 when members is 0 or on the first record. This is the stage's measure of foraging skill.
- `advance(n)` records whenever `tick // record_interval` has increased and checkpoints whenever `tick // checkpoint_interval` has increased. It also checkpoints when the CLI exits.
- Extinction: if no colony is alive at the start of `advance`: with reseeding on, replace the state with `init_world(jax.random.fold_in(jax.random.key(seed), extinctions + 1), cfg)` keeping the old `tick` and `delivered_total`, increment `extinctions`, record the event `{"event": "reseed", "tick": t}` immediately, then advance. With reseeding off, set `extinct = True`, record `{"event": "extinct", "tick": t}` once, and return without advancing.
- Checkpoints are written to a `.tmp` name and renamed, so a partial file never has the final name.

- [ ] **Step 1: Write the failing tests**

```python
def test_state_roundtrips_through_disk_and_run_continues_identically(tmp_path):
    s = run_ticks(init_world(jax.random.key(3), cfg), cfg, 100)
    save_state(tmp_path / "s.npz", s)
    assert states_equal(run_ticks(load_state(tmp_path / "s.npz", cfg), cfg, 100), run_ticks(s, cfg, 100))
def test_recorder_writes_settings_and_history(tmp_path):
    # settings.json holds seed and Config.to_dict(); two record() calls → two lines; second has delivery_rate
def test_runner_records_on_interval_and_resume_matches_uninterrupted(tmp_path):
    # run A: advance 300. run B: advance 150, checkpoint, resume_run, advance 150 → states_equal
    # history of B has no duplicated ticks
def test_reseed_on_extinction_logs_and_continues(tmp_path):      # Review Focus 3
    # cfg with initial_energy=0.001 so all colonies die within a few ticks
    r.advance(50); r.advance(50)
    assert r.extinctions >= 1 and compute_metrics(r.state)["colonies"] > 0
    assert any(e.get("event") == "reseed" for rec in read_history(r.recorder.run_dir) for e in rec.get("events", []))
    assert int(r.state.tick) == 100
def test_extinction_with_reseed_off_idles(tmp_path):             # Review Focus 3
    r.reseed_on_extinction = False; r.advance(50); t = int(r.state.tick); r.advance(50); r.advance(50)
    assert r.extinct and int(r.state.tick) == t
    # exactly one "extinct" event in history
def test_resume_ignores_partial_checkpoint(tmp_path):            # Review Focus 4
    # write garbage to checkpoints/state-999999.npz.tmp → resume_run loads the real latest one
def test_resume_with_mismatched_shapes_raises_clear_error(tmp_path):   # Review Focus 4
    # edit settings.json capacity_members after checkpointing
    with pytest.raises(ValueError, match="capacity"): resume_run(run_dir)
def test_cli_headless_run(tmp_path):
    # subprocess: python -m simlife run --no-serve --ticks 200 --seed 1 --runs-dir tmp_path → exit 0,
    # one run dir containing settings.json, history.jsonl (2 lines), one checkpoint
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Store the PRNG key in the `.npz` via `jax.random.key_data` and restore with `jax.random.wrap_key_data`. `load_state` checks every array's shape against `cfg` and raises `ValueError` naming the mismatched capacity.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: metrics, run recording, checkpoints, extinction handling, headless CLI`.

---

### Task 9: Snapshot protocol and WebSocket server

**Files:**
- Create: `engine/simlife/protocol.py`, `engine/simlife/server.py`, `engine/tools/write_fixture.py`
- Modify: `engine/simlife/__main__.py`
- Test: `engine/tests/test_protocol.py`, `engine/tests/test_server.py`

**Interfaces:**
- Consumes: `Runner`, `compute_metrics`, `read_history`, `list_runs`.
- Produces:
  - `encode_snapshot(state, metrics: dict, status: dict) -> bytes`; `decode_snapshot(data: bytes) -> tuple[dict, dict[str, np.ndarray]]`.
  - `parse_control(text: str) -> dict | None` — returns the validated message, or `None` for malformed JSON, a non-object, an unknown `type`, or a wrong-typed field.
  - `async serve(runner: Runner, runs_dir: Path, host: str, port: int) -> None`.

**Snapshot frame (binary, engine → viewer):** 4-byte little-endian unsigned header length, then that many bytes of UTF-8 JSON header, then the arrays back to back as little-endian float32 in header order.

Header: `{"type": "snapshot", "tick", "metrics": {...}, "status": {...}, "arrays": [{"name", "shape"}, ...]}`.

Arrays, in order, holding only alive members / alive colonies / alive pellets:

| Name | Shape | Content |
|---|---|---|
| `member_pos` | (M,2) | positions |
| `member_heading` | (M,) | radians |
| `member_colony` | (M,) | colony slot |
| `member_carrying` | (M,) | 0 or 1 |
| `member_slot` | (M,) | member slot index (stable identity; stage 2 uses it for selection) |
| `colony_slot` | (K,) | colony slot |
| `colony_nest` | (K,2) | nest positions |
| `colony_energy` | (K,) | energy |
| `colony_members` | (K,) | member count |
| `food_pos` | (L,2) | live pellets |

**Text messages (JSON).** Viewer → engine: `{"type":"pause","value":bool}`, `{"type":"speed","ticks_per_frame":int}` (clamped to 1–1000), `{"type":"reseed_on_extinction","value":bool}`, `{"type":"list_runs"}`, `{"type":"get_history","run_id":str}`. Engine → viewer: `{"type":"runs","runs":[str],"current":str}`, `{"type":"history","run_id":str,"records":[...]}`, and on each record interval `{"type":"record","run_id":str,"record":{...}}`.

**Pacing.** With at least one viewer and not paused: advance `ticks_per_frame` (default 1), send a snapshot to every viewer, and hold the loop to 30 frames/s. Paused with viewers: send nothing new, sleep. No viewers: advance in chunks of 500 as fast as possible, yielding to the event loop between chunks so new connections are accepted. A newly connected viewer immediately gets `runs`, the current run's `history`, and one snapshot (even when paused). `get_history` must reject a `run_id` that isn't in `list_runs` (no path traversal).

- [ ] **Step 1: Write the failing tests**

```python
def test_snapshot_roundtrip_contains_only_the_living():
    h, a = decode_snapshot(encode_snapshot(s, compute_metrics(s), {"run_id": "x"}))
    assert h["tick"] == 0 and a["member_pos"].shape == (192, 2) and a["colony_nest"].shape == (16, 2)
    assert a["member_pos"].dtype == np.float32 and set(a["member_carrying"]) <= {0.0, 1.0}
def test_snapshot_of_extinct_world_has_empty_arrays():
def test_parse_control_accepts_valid_and_rejects_junk():          # Review Focus 5
    assert parse_control('{"type":"pause","value":true}') == {"type": "pause", "value": True}
    for bad in ['{', '[]', '{"type":"nope"}', '{"type":"pause","value":"yes"}', '{"type":"speed","ticks_per_frame":"x"}']:
        assert parse_control(bad) is None
    assert parse_control('{"type":"speed","ticks_per_frame":99999}')["ticks_per_frame"] == 1000
# test_server.py: start serve() on an ephemeral port in the test's event loop, use websockets' client
async def test_new_viewer_gets_runs_history_and_a_snapshot():
async def test_two_viewers_both_receive_snapshots():              # Review Focus 5
async def test_engine_survives_disconnect_and_junk():             # Review Focus 5
    # viewer A sends "{{{" and {"type":"nope"}, then closes abruptly; viewer B connects afterwards
    # and receives a snapshot whose tick is greater than the first one A saw
async def test_headless_engine_advances_with_no_viewer():
    # start, wait 0.5 s with nobody connected, then connect → first snapshot tick > 0
async def test_pause_stops_ticks_and_resume_restarts():
async def test_get_history_rejects_unknown_run_id():
    # {"type":"get_history","run_id":"../../etc"} → no history message, connection stays open
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Run `runner.advance` via `asyncio.to_thread` so the event loop stays responsive. Wrap each send in its own exception handler and drop viewers whose send fails. Wire `python -m simlife run` (without `--no-serve`) to `serve`, checkpointing on Ctrl-C.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Write the viewer fixture.** `engine/tools/write_fixture.py` writes `encode_snapshot` of `init_world(jax.random.key(0), Config())` with status `{"run_id": "fixture", "extinct": false, "extinctions": 0, "reseed_on_extinction": true}` to `viewer/src/__fixtures__/snapshot.bin`. Run it.
- [ ] **Step 6: Commit** `feat: snapshot protocol and WebSocket server`.

---

### Task 10: Viewer — connection, world view, controls

**Files:**
- Create: `viewer/package.json`, `viewer/tsconfig.json`, `viewer/vite.config.ts`, `viewer/index.html`, `viewer/src/protocol.ts`, `viewer/src/connection.ts`, `viewer/src/worldView.ts`, `viewer/src/controls.ts`, `viewer/src/main.ts`
- Test: `viewer/src/protocol.test.ts`, `viewer/src/worldView.test.ts`

**Interfaces:**
- Consumes: the frame and message formats from Task 9; `viewer/src/__fixtures__/snapshot.bin`.
- Produces:
  - `protocol.ts`: `interface Snapshot { tick: number; metrics: Metrics; status: Status; arrays: Record<string, Float32Array>; shapes: Record<string, number[]> }`; `decodeSnapshot(buf: ArrayBuffer): Snapshot` (throws `Error` on a truncated or malformed frame); types `Metrics`, `Status`, `HistoryRecord`, `ServerMessage`, `ControlMessage`.
  - `connection.ts`: `class Connection { constructor(url: string, handlers: { onSnapshot, onMessage, onState }); send(msg: ControlMessage): void }` — reconnects every 2 s after a close; `onState("connected" | "disconnected")`.
  - `worldView.ts`: `colonyColor(slot: number): string` (stable, well-separated hues: `hsl((slot * 137.508) % 360, 70%, 55%)`); `worldToCanvas(x: number, y: number, size: number): [number, number]`; `drawWorld(ctx, snap: Snapshot, size: number, mode: "colony" | "carrying"): void`.
  - `controls.ts`: `mountControls(root: HTMLElement, send: (m: ControlMessage) => void): { update(status: Status, metrics: Metrics): void }`.

- [ ] **Step 1: Scaffold.** In `viewer/`: `npm init -y`, `npm install -D typescript vite vitest`. Scripts: `dev` = `vite`, `build` = `tsc --noEmit && vite build`, `test` = `vitest run`. `tsconfig` strict.

- [ ] **Step 2: Write the failing tests**

```ts
test("decodes the engine's fixture", () => {
  const snap = decodeSnapshot(readFixture("snapshot.bin"));
  expect(snap.tick).toBe(0);
  expect(snap.arrays.member_pos.length).toBe(192 * 2);
  expect(snap.shapes.colony_nest).toEqual([16, 2]);
  expect(snap.status.run_id).toBe("fixture");
  for (const v of snap.arrays.member_pos) { expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThan(1); }
});
test("rejects a truncated frame", () => {
  expect(() => decodeSnapshot(readFixture("snapshot.bin").slice(0, 40))).toThrow();
});
test("rejects a frame whose arrays overrun the buffer", () => { /* cut 8 bytes off the end */ });
test("colony colours are stable and distinct for neighbouring slots", () => {
  expect(colonyColor(3)).toBe(colonyColor(3)); expect(colonyColor(3)).not.toBe(colonyColor(4));
});
test("worldToCanvas maps the unit square onto the canvas", () => {
  expect(worldToCanvas(0, 0, 800)).toEqual([0, 0]); expect(worldToCanvas(0.5, 0.25, 800)).toEqual([400, 200]);
});
```

- [ ] **Step 3: Run** `npm test`. Expected: FAIL.
- [ ] **Step 4: Implement.** `decodeSnapshot` must copy array bytes into fresh `Float32Array`s (the payload offset is not guaranteed to be 4-byte aligned). `drawWorld` draws, in order: food as small dots, nests as rings in the colony's colour sized by `colony_members`, members as short oriented marks in the colony's colour (mode `colony`) or highlighted when carrying (mode `carrying`). `main.ts` connects to `ws://<page host>:8765`, draws the most recent snapshot once per animation frame, and catches decode errors (log, keep the last good frame). Controls: pause/resume, a speed slider (1–1000 ticks per frame), colour-mode toggle, reseed-on-extinction checkbox, and a status line showing tick, colonies, members, extinctions, connection state, and "extinct — reseeding is off" when applicable. Page layout: world canvas on the left, a side column with controls and an empty container `#population` for Task 11.
- [ ] **Step 5: Run** `npm test && npm run build`. Expected: PASS, build succeeds.
- [ ] **Step 6: Look at it.** Start `python -m simlife run --seed 0` and `npm run dev`, open the page. Confirm: entities move; nests and food are visible; pause freezes the tick counter; the speed slider changes the rate; stopping and restarting the engine reconnects without a page reload.
- [ ] **Step 7: Commit** `feat: viewer with world view and controls`.

---

### Task 11: Viewer — population view with run overlay

**Files:**
- Create: `viewer/src/populationView.ts`
- Modify: `viewer/src/main.ts`, `viewer/src/controls.ts`
- Test: `viewer/src/populationView.test.ts`

**Interfaces:**
- Consumes: `HistoryRecord` (`tick, colonies, members, energy, delivered, delivery_rate, max_generation, events?`), server messages `runs`, `history`, `record`.
- Produces:
  - `seriesOf(records: HistoryRecord[], field: keyof HistoryRecord): [number, number][]` — `(tick, value)` pairs sorted by tick, skipping records where the field is missing or not finite.
  - `niceScale(min: number, max: number, target: number): { lo: number; hi: number; ticks: number[] }` — round axis bounds and tick values; when `min === max` returns a non-zero-width range.
  - `class PopulationView { constructor(root: HTMLElement); setCurrent(runId: string, records: HistoryRecord[]): void; append(record: HistoryRecord): void; setOverlay(runId: string | null, records: HistoryRecord[]): void }`.

Four small charts stacked, sharing a tick axis: colonies, members, total energy, delivery rate. The current run is a solid line; the overlay run, if any, is a dashed line of a second colour, with a two-entry legend naming both run ids. Reseed events on the current run are marked with a vertical rule. The run picker (in controls) lists `runs` other than the current one plus "none"; choosing one sends `get_history` and passes the reply to `setOverlay`.

- [ ] **Step 1: Write the failing tests**

```ts
test("seriesOf sorts by tick and skips non-finite values", () => {
  const recs = [{ tick: 200, members: 5 }, { tick: 100, members: 3 }, { tick: 300, members: NaN }] as HistoryRecord[];
  expect(seriesOf(recs, "members")).toEqual([[100, 3], [200, 5]]);
});
test("niceScale gives round bounds that contain the data", () => {
  const s = niceScale(3, 97, 5);
  expect(s.lo).toBeLessThanOrEqual(3); expect(s.hi).toBeGreaterThanOrEqual(97);
  expect(s.ticks.every(t => Number.isInteger(t / 10))).toBe(true);
});
test("niceScale handles a flat series", () => { const s = niceScale(4, 4, 5); expect(s.hi).toBeGreaterThan(s.lo); });
test("niceScale handles an empty range of zeros", () => { const s = niceScale(0, 0, 5); expect(s.hi).toBeGreaterThan(s.lo); });
```

- [ ] **Step 2: Run** `npm test`. Expected: FAIL.
- [ ] **Step 3: Implement.** Draw on canvas. An empty history draws axes and the text "no history yet". Charts must be readable in both light and dark colour schemes.
- [ ] **Step 4: Run** `npm test && npm run build`. Expected: PASS.
- [ ] **Step 5: Look at it.** Run seed 0 for a few thousand ticks, stop, start seed 1, open the viewer, pick the seed-0 run as overlay. Confirm both lines appear with a legend, and that new records extend the current line live.
- [ ] **Step 6: Commit** `feat: population view with recorded-run overlay`.

---

### Task 12: Bootstrapping, tuning, and the stage's success check

**Files:**
- Modify: `engine/simlife/config.py` (defaults only)
- Create: `README.md`, `engine/tests/test_bootstrap.py`

**Interfaces:**
- Consumes: the whole engine through `python -m simlife run`.
- Produces: tuned `Config` defaults; a README covering setup, running headless, running with the viewer, resuming, and where run data lives.

The spec expects that with random starting weights every colony may starve before any forages well. This task tunes starting conditions so that random wanderers sometimes deliver food, then checks the stage's success criterion: colonies visibly get better at foraging over generations.

- [ ] **Step 1: Write the bootstrap test**

```python
@pytest.mark.slow
def test_default_world_bootstraps():
    # 5 seeds, default Config, 20_000 ticks each via Runner with reseeding off
    # at least 3 of 5: not extinct, delivered > 0, max_generation >= 5, and at least one colony split
    # (a split is visible as a colony with col_generation > 0)
```

- [ ] **Step 2: Run** `.venv/bin/pytest engine/tests/test_bootstrap.py -v -m slow`. Expected: likely FAIL on untuned defaults.
- [ ] **Step 3: Tune.** Change only `Config` defaults. Levers, in the order to try them: `initial_energy` and `base_cost` (how long a naive colony lasts), `pickup_radius`, `nest_radius` and `patch_radius` (how easily a wanderer succeeds by chance), `capacity_food` and `food_regrow_prob` (abundance), `food_energy` against `birth_cost` (how much one delivery buys). Do not add mechanisms, reward terms, or special-case help for early colonies. Keep food scarce enough that the member count stays well under `capacity_members`; if the population pins at capacity, scarcity is not the limit and tuning is not finished.
- [ ] **Step 4: Run** the bootstrap test and the full suite. Expected: PASS.
- [ ] **Step 5: Success check.** Run 3 seeds headless for 300,000 ticks each. For each, compare mean `delivery_rate` over the first 10% of records after the last reseed with the mean over the last 10%. Stage 1 succeeds if the later figure is clearly higher in at least 2 of 3 seeds and the population view shows it. Report the six numbers. If it does not succeed, report that plainly with the numbers and the charts; do not adjust the measure.
- [ ] **Step 6: Write the README** and **commit** `feat: tuned defaults, bootstrap test, README`.
