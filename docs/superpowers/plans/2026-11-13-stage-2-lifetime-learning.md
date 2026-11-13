# Stage 2: Lifetime Learning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Each entity learns during its life by predicting its own next senses, with switches for learning and inheritance, and an inspector that shows one entity's inputs, activations, predictions, error and weights live.

**Architecture:** The network gains a second set of outputs (predictions) reading the same hidden layer. Each tick, every entity's prediction error from the previous tick is backpropagated with `vmap(grad(loss))` over the population and applied with that entity's own learning rate. The two switches live in the world state as traced booleans, so flipping them never recompiles and is saved with every checkpoint. The server attaches one selected entity's internals to each snapshot.

**Tech Stack:** As stage 1: Python 3.14, JAX (CPU), `websockets`, pytest; TypeScript, Vite, Vitest, Canvas 2D.

**Spec:** `docs/superpowers/specs/2026-11-12-colony-life-sim-design.md` — "The entity → Network / Two kinds of change", "Inheritance", "Viewer → Inspector", and "Stages → 2. Lifetime learning". Stage 1's plan is `docs/superpowers/plans/2026-11-12-stage-1-evolution.md`.

## Global Constraints

- Everything in stage 1's Global Constraints still holds: no fitness function or reward, no Python loop over entities, fixed shapes with alive masks, explicit keys, same seed and settings give the same run however it is chunked, never inject colonies while any is alive, weights never averaged.
- The only learning signal is an entity's own prediction error. Nothing derived from energy, food delivered, or survival may enter a gradient.
- Lifetime learning changes `w1`, `b1`, `w3`, `b3` only. The action weights `w2`, `b2` change only by mutation at division. This must hold by construction (the loss does not depend on them) and be tested.
- Per-entity gradients are computed by transforming a single-entity loss (`jax.vmap(jax.grad(...))`), never by a hand-written backward pass.
- Predicted senses are observation columns 0–5, in order: food left, food right, nest left, nest right, crowd, signal heard. Carrying, colony energy and memory are inputs but are not predicted.
- Default inheritance copies the parent's **current, learned** weights (Paul's choice). The alternative copies the parent's **birth** weights.
- With learning off, a run must behave as stage 1 does: weights change only at division.
- Work on branch `stage-2-learning`. End each commit message with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.

## Review Focus

1. **Old checkpoints.** Resuming a stage 1 run (whose checkpoints lack the new fields) gives a one-line error saying the run was recorded by an older version; no traceback, and the old run's history still loads in the viewer for comparison. → Task 1.
2. **Learning blow-up.** With every learning rate at the maximum and a long run, no weight, prediction or error becomes NaN or Inf. → Task 3.
3. **The selected member dies, or its slot is reused.** The inspector says the member has died; it never silently starts showing the newborn that took the slot. → Task 6.
4. **A switch flipped mid-run.** It takes effect on the next tick, is written to history as an event at that tick, survives checkpoint and resume, survives a fresh start after extinction, and reaches every connected viewer. → Tasks 5, 6.
5. **Clicks that select nothing.** Clicking empty space, clicking while disconnected, selecting a slot number outside the population, or selecting in an extinct world never crashes engine or viewer; it clears the selection. → Tasks 6, 7.

---

## File Structure

Modified engine files: `config.py`, `state.py`, `brain.py`, `step.py`, `lifecycle.py`, `metrics.py`, `recorder.py`, `runner.py`, `protocol.py`, `server.py`, `__main__.py`. New: `engine/simlife/learning.py` (the learning phase of a tick), `engine/simlife/inspect.py` (one entity's internals as plain numbers).

Modified viewer files: `protocol.ts`, `controls.ts`, `worldView.ts`, `populationView.ts`, `main.ts`, `index.html`. New: `viewer/src/inspector.ts` (the panel), `viewer/src/heatmap.ts` (colour scale and cell drawing).

---

### Task 1: State and settings for learning

**Files:**
- Modify: `engine/simlife/config.py`, `engine/simlife/state.py`, `engine/simlife/brain.py` (`init_params` only), `engine/simlife/recorder.py`, `engine/tests/helpers.py`
- Test: `engine/tests/test_state.py`, `engine/tests/test_recorder.py`

**Interfaces:**
- Produces new `Config` fields: `n_predictions: int = 6`, `initial_lr: float = 0.02`, `max_lr: float = 0.5`, `lr_mutation_sigma: float = 0.1`, `grad_clip: float = 1.0`, `error_smoothing: float = 0.98`.
- Produces new `WorldState` fields (N = member capacity, H = 8, P = 6):

| Field | Shape, dtype | Meaning |
|---|---|---|
| `params["w3"]`, `params["b3"]` | (N,P,H), (N,P) f32 | prediction outputs |
| `birth_params` | dict, same keys and shapes as `params` | each member's weights as they were at birth |
| `lr` | (N,) f32 | each member's learning rate |
| `prev_obs` | (N,10) f32 | the observation acted on last tick |
| `has_prev` | (N,) bool | False until a member has lived one full tick |
| `last_pred` | (N,P) f32 | the prediction that was just compared with reality |
| `pred_error` | (N,) f32 | this tick's mean squared prediction error |
| `pred_error_avg` | (N,) f32 | exponential moving average of `pred_error` |
| `learning_on` | () bool | the learning switch |
| `inherit_learned` | () bool | True: children copy learned weights; False: birth weights |

- `init_world` sets: `w3`, `b3` random like the other weights and clonal per colony; `birth_params` equal to `params`; `lr = initial_lr`; `prev_obs`, `last_pred`, `pred_error`, `pred_error_avg` zero; `has_prev` False; `learning_on` True; `inherit_learned` True.
- `load_state` raises `ValueError` containing "older version" when a checkpoint lacks any field the current state has.

- [ ] **Step 1: Branch.** `git switch -c stage-2-learning`.
- [ ] **Step 2: Write the failing tests**

```python
def test_world_has_learning_state():
    cfg = Config(); s = init_world(jax.random.key(0), cfg)
    assert s.params["w3"].shape == (2048, 6, 8) and s.params["b3"].shape == (2048, 6)
    assert all(bool((s.birth_params[k] == s.params[k]).all()) for k in s.params)
    assert bool((s.lr[s.alive] == cfg.initial_lr).all()) and not bool(s.has_prev.any())
    assert bool(s.learning_on) and bool(s.inherit_learned)
    assert s.prev_obs.shape == (2048, 10) and s.last_pred.shape == (2048, 6)

def test_prediction_weights_are_clonal_within_a_colony():
    # as stage 1's clonal test, for "w3"

def test_resuming_a_checkpoint_from_an_older_version_says_so(tmp_path):
    # save a state, rewrite the .npz without "lr" and "params.w3", then:
    with pytest.raises(ValueError, match="older version"): load_state(path, cfg)

def test_cli_resume_of_older_run_is_a_one_line_error(tmp_path):
    # same doctored checkpoint inside a run dir; python -m simlife run --resume → exit 1,
    # "older version" in stderr, "Traceback" not in stderr
```

- [ ] **Step 3: Run** `.venv/bin/pytest -q`. Expected: the new tests FAIL; note any stage 1 tests that break on the new fields.
- [ ] **Step 4: Implement.** Update `engine/tests/helpers.py` (`blank`, `leaves`, `states_equal`) so hand-built worlds and comparisons cover the new fields. `save_state`/`load_state` must handle `birth_params` the way they handle `params`.
- [ ] **Step 5: Run** the full suite. Expected: PASS, including all stage 1 tests.
- [ ] **Step 6: Regenerate** the viewer fixture (`.venv/bin/python engine/tools/write_fixture.py`) and run `npm test` in `viewer/`. Expected: PASS.
- [ ] **Step 7: Commit** `feat: world state and settings for lifetime learning`.

---

### Task 2: Prediction outputs and the single-entity learning rule

**Files:**
- Modify: `engine/simlife/brain.py`
- Test: `engine/tests/test_brain.py`

**Interfaces:**
- Consumes: `params` with keys `w1 b1 w2 b2 w3 b3`.
- Produces, all for ONE entity:
  - `hidden(params, x: Array[10]) -> Array[8]` — `tanh(w1 @ x + b1)`.
  - `act(params, x)` — unchanged signature and results.
  - `predict(params, x) -> Array[6]` — `w3 @ hidden(params, x) + b3` (linear output).
  - `prediction_loss(params, x_prev: Array[10], target: Array[6]) -> ()` — mean squared error of `predict(params, x_prev)` against `target`.
  - `learn(params, x_prev, target, lr: (), clip: float) -> tuple[new_params, loss (), prediction Array[6]]` — one gradient step. The gradient across all arrays is scaled down if its overall norm exceeds `clip`. `prediction` and `loss` are those from **before** the update.
  - `learn_all = jax.vmap(learn, in_axes=(0, 0, 0, 0, None))`.

- [ ] **Step 1: Write the failing tests**

```python
def test_loss_gradient_never_touches_action_weights():
    g = jax.grad(prediction_loss)(p, x, target)
    assert bool((g["w2"] == 0).all()) and bool((g["b2"] == 0).all())
    assert bool((g["w1"] != 0).any()) and bool((g["w3"] != 0).any())

def test_learning_reduces_error_on_a_repeated_example():
    q = p
    first = prediction_loss(p, x, target)
    for _ in range(200): q, _, _ = learn(q, x, target, 0.05, 1.0)
    assert prediction_loss(q, x, target) < 0.05 * first

def test_learn_reports_loss_and_prediction_from_before_the_update():
    q, loss, pred = learn(p, x, target, 0.05, 1.0)
    assert jnp.allclose(pred, predict(p, x)) and jnp.isclose(loss, prediction_loss(p, x, target))
    assert bool((q["w2"] == p["w2"]).all()) and bool((q["b2"] == p["b2"]).all())

def test_zero_learning_rate_changes_nothing():
    q, _, _ = learn(p, x, target, 0.0, 1.0)
    assert all(bool((q[k] == p[k]).all()) for k in p)

def test_gradient_clip_bounds_the_step():
    # target far away (1e4) → huge gradient; with lr=1.0, clip=1.0 the total change in weights has norm <= 1.0 (+1e-5)

def test_learn_all_matches_single():
    # 5 entities with different params, lrs; learn_all row 3 == learn on entity 3 (allclose)

def test_learning_does_not_change_what_act_returns_for_fixed_hidden_weights():
    # act still passes all three stage 1 tests unchanged
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Use `jax.value_and_grad(prediction_loss, has_aux=True)` so loss, prediction and gradient come from one pass. `init_params` already produces `w3`, `b3` from Task 1.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: prediction outputs and single-entity learning step`.

---

### Task 3: Learning inside the tick

**Files:**
- Create: `engine/simlife/learning.py`
- Modify: `engine/simlife/step.py`
- Test: `engine/tests/test_learning.py`, `engine/tests/test_step.py`

**Interfaces:**
- Consumes: `learn_all`, `observe`, `WorldState`.
- Produces: `learn_phase(state, obs: Array[N,10], cfg) -> WorldState`, called in `step` right after `observe` and before `act_all`.

What `learn_phase` does, for members where `alive & has_prev` and only when `learning_on`:
- updates `params` by one `learn` step with `x_prev = prev_obs`, `target = obs[:, :6]`, the member's own `lr`;
- sets `last_pred`, `pred_error`, and `pred_error_avg = s * avg + (1 - s) * error` with `s = error_smoothing`.

For every other member, `params` are untouched. For members that are alive with `has_prev` while learning is off, `last_pred`, `pred_error` and `pred_error_avg` are still computed, so error can be watched in a pure-evolution run. Then, for all members: `prev_obs = obs`, `has_prev = alive`.

`step` then acts on `obs` with the updated `params`. Nothing else in the tick order changes.

- [ ] **Step 1: Write the failing tests** (medium config as in `test_step.py`)

```python
def test_learning_changes_hidden_and_prediction_weights_only():
    s0 = world(); s1 = run_ticks(s0, cfg, 20)          # 20 ticks: too few for any birth with initial settings? assert it
    assert int(s1.generation.max()) == 0                # no births happened, so any change is learning
    same = s0.alive & s1.alive
    assert bool((s1.params["w2"][same] == s0.params["w2"][same]).all())
    assert bool((s1.params["w1"][same] != s0.params["w1"][same]).any())
    assert bool((s1.params["w3"][same] != s0.params["w3"][same]).any())
    assert bool((s1.birth_params["w1"] == s0.birth_params["w1"]).all())

def test_learning_off_leaves_all_weights_alone_but_still_measures_error():
    s0 = world()._replace(learning_on=jnp.bool_(False)); s1 = run_ticks(s0, cfg, 20)
    assert all(bool((s1.params[k] == s0.params[k]).all()) for k in s0.params)
    assert bool((s1.pred_error[s1.alive] > 0).any())

def test_no_learning_on_a_members_first_tick():
    s1 = jax.jit(lambda s: step(s, cfg))(world())
    assert bool((s1.params["w1"] == world().params["w1"]).all())    # has_prev was False
    assert bool(s1.has_prev[s1.alive].all()) and bool((s1.pred_error == 0).all())

def test_prediction_error_falls_over_a_life():
    # learning on vs off, same seed, 400 ticks, members alive throughout and generation 0
    # on: mean pred_error_avg at tick 400 < 0.6 * mean pred_error_avg at tick 60
    # off: the same ratio is > 0.8

def test_last_pred_is_the_prediction_that_was_scored():
    # one jitted step from a state with has_prev True: pred_error == mean((last_pred - prev_obs_after[:, :6])**2) per alive member

def test_maximum_learning_rate_long_run_stays_finite():          # Review Focus 2
    s = world()._replace(lr=jnp.full(N, cfg.max_lr)); s = run_ticks(s, cfg, 5000)
    # every floating leaf finite

def test_learning_run_is_deterministic_and_chunk_independent():
    # states_equal(run 200) == run 100 + 100, with learning on
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full suite. Expected: PASS. Stage 1 tests that assumed weights never change in life must be read again: fix the test only where its premise was "no learning exists", by switching learning off in that test.
- [ ] **Step 5: Measure speed** as in stage 1 (2000 ticks at default `Config` after warm-up), with learning on and off. Record both in the commit message. If learning on is below 150 ticks/s, report it and continue.
- [ ] **Step 6: Commit** `feat: per-entity prediction learning inside the tick`.

---

### Task 4: Inheritance of learned weights and learning rate

**Files:**
- Modify: `engine/simlife/lifecycle.py` (`grow`), `engine/simlife/runner.py` (fresh worlds keep the switches)
- Test: `engine/tests/test_lifecycle.py`, `engine/tests/test_runner.py`

**Interfaces:**
- `grow` — same signature. For a child of parent `p`:
  - source weights = `params[p]` when `inherit_learned`, else `birth_params[p]`;
  - child `params` = source + `mutation_sigma` noise (all six arrays); child `birth_params` = child `params`;
  - child `lr = clip(lr[p] * exp(lr_mutation_sigma * normal), 0, max_lr)`;
  - child `has_prev` False; `prev_obs`, `last_pred`, `pred_error`, `pred_error_avg` zero.
- `Runner`: a fresh world after extinction carries over `learning_on` and `inherit_learned` from the dead one.

- [ ] **Step 1: Write the failing tests**

```python
def test_child_inherits_learned_weights_by_default():
    # parent colony's params["w1"] set to birth + 1.0 (as if learned); mutation_sigma tiny (1e-6)
    # child's w1 ≈ parent's CURRENT w1; child's birth_params == child's params
def test_child_inherits_birth_weights_when_switched():
    # same, inherit_learned False → child's w1 ≈ parent's BIRTH w1
def test_child_learning_rate_is_inherited_mutated_and_clipped():
    # over many seeds: child lr != parent lr, mean ratio near 1, never above max_lr, never below 0
    # parent lr == max_lr → child lr <= max_lr
def test_child_starts_with_no_previous_observation():
def test_fresh_world_after_extinction_keeps_the_switches(tmp_path):    # Review Focus 4
    # doomed config, learning off and inherit birth set on the runner's state, advance through a reseed → both still off
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat: inheritance mode and evolving learning rate`.

---

### Task 5: Switches in the runner, history, metrics and command line

**Files:**
- Modify: `engine/simlife/runner.py`, `engine/simlife/metrics.py`, `engine/simlife/__main__.py`
- Test: `engine/tests/test_runner.py`, `engine/tests/test_recorder.py`

**Interfaces:**
- `Runner.set_switch(name: str, value: bool) -> dict | None` — `name` is `"learning"` or `"inherit_learned"`. Changes the flag in `self.state`; if the value actually changed, writes and returns the history event `{"event": "switch", "name": name, "value": value, "tick": tick}`; otherwise returns `None`. Unknown names raise `ValueError`.
- `Runner.status()` gains `"learning": bool` and `"inheritance": "learned" | "birth"`.
- `compute_metrics` gains `"pred_error"` (mean `pred_error_avg` over alive members with `has_prev`; 0.0 if none) and `"learning_rate"` (mean `lr` over alive members; 0.0 if none).
- CLI: `--no-learning` and `--inherit {learned,birth}` (default `learned`), applied through `set_switch` at start. On `--resume` they are applied only if given; otherwise the run keeps its saved switches.

- [ ] **Step 1: Write the failing tests**

```python
def test_set_switch_records_an_event_once(tmp_path):                 # Review Focus 4
    r.advance(30); e = r.set_switch("learning", False)
    assert e == {"event": "switch", "name": "learning", "value": False, "tick": 30}
    assert r.set_switch("learning", False) is None
    assert r.status()["learning"] is False and r.status()["inheritance"] == "learned"
    # exactly one switch event in history
def test_switches_survive_checkpoint_and_resume(tmp_path):           # Review Focus 4
def test_switch_takes_effect_on_the_next_tick(tmp_path):
    # learning off → advance 20 → no weight change among survivors with generation 0
def test_set_switch_rejects_unknown_names(tmp_path):
def test_metrics_include_prediction_error_and_learning_rate():
    # fresh world: pred_error 0.0, learning_rate == initial_lr; after 50 ticks pred_error > 0
def test_metrics_of_an_extinct_world_do_not_divide_by_zero():
def test_cli_switch_flags(tmp_path):
    # build_parser + make_runner: --no-learning, --inherit birth → status reflects both; settings on resume kept unless a flag is given
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat: learning and inheritance switches, recorded and resumable`.

---

### Task 6: Protocol and server — switches, selection, inspection

**Files:**
- Create: `engine/simlife/inspect.py`
- Modify: `engine/simlife/protocol.py`, `engine/simlife/server.py`, `engine/tools/write_fixture.py`
- Test: `engine/tests/test_inspect.py`, `engine/tests/test_protocol.py`, `engine/tests/test_server.py`

**Interfaces:**
- `inspect_entity(state, cfg, slot: int) -> dict` — plain Python numbers and lists:

```
{"slot", "alive", "colony", "age", "lifespan", "generation", "carrying", "learning_rate",
 "inputs": [10], "hidden": [8], "actions": [5],        # turn, speed, signal, memory 0, memory 1, as act returns them
 "predicted": [6], "actual": [6],                       # last_pred against prev_obs[:6]
 "next_prediction": [6],                                # predict(params, prev_obs): its guess for the coming tick
 "error", "error_avg",
 "weights": {"w1": [[..]], "b1": [..], "w2": [[..]], "b2": [..], "w3": [[..]], "b3": [..]}}
```

  `inputs` is `prev_obs` (the observation it last acted on); `hidden` and `actions` are recomputed from it with the current weights.
- New control messages accepted by `parse_control`: `{"type":"learning","value":bool}`, `{"type":"inheritance","value":"learned"|"birth"}`, `{"type":"select","slot":int|null}` (`slot` must be an int ≥ 0, not a bool, or null).
- Snapshot header gains `"inspect"`: `null` when nothing is selected; `{"slot": n, "alive": false}` when the selected member has died; otherwise the `inspect_entity` dict.
- Snapshot gains the array `member_error` (M,), `pred_error_avg` for each living member, placed after `member_slot`.
- Snapshot `status` already carries the runner's status, so it now includes `learning` and `inheritance`.

Selection rules (server): one selection shared by all viewers. A `select` with a slot ≥ member capacity, or whose member is not alive, clears the selection. The server remembers the selected member's last seen age; if the slot is no longer alive, **or its age is lower than last seen** (the slot was reused by a newborn), the member is reported as died and stays reported that way until a new `select` arrives. `learning` and `inheritance` controls go through `Runner.set_switch`, and any event it returns is broadcast as a `record` message so charts mark it.

- [ ] **Step 1: Write the failing tests**

```python
def test_inspect_entity_is_json_ready_and_consistent():
    s = run_ticks(world(), cfg, 30); slot = int(jnp.argmax(s.alive)); d = inspect_entity(s, cfg, slot)
    json.dumps(d)
    assert len(d["inputs"]) == 10 and len(d["hidden"]) == 8 and len(d["actions"]) == 5
    assert len(d["weights"]["w1"]) == 8 and len(d["weights"]["w1"][0]) == 10 and len(d["weights"]["w3"]) == 6
    assert d["error"] == pytest.approx(np.mean((np.array(d["predicted"]) - np.array(d["actual"])) ** 2), rel=1e-4)
    assert d["actual"] == pytest.approx(d["inputs"][:6])
def test_parse_control_new_messages():
    # accepts the three; rejects {"type":"select","slot":-1}, slot true, slot 1.5, slot "3";
    # {"type":"inheritance","value":"other"}; {"type":"learning","value":1}
def test_snapshot_carries_member_error_and_inspect():
    # encode with inspect=None → header["inspect"] is None; with a dict → round-trips; member_error shape (M,)
async def test_select_then_inspect_arrives(server):
async def test_selected_member_death_is_reported_not_replaced(server):     # Review Focus 3
    # select a slot; kill it by editing runner.state (alive False) → inspect == {"slot": n, "alive": False}
    # then make the slot alive again with age 0 (a newborn) → still {"slot": n, "alive": False}
async def test_select_out_of_range_or_dead_slot_clears_selection(server):  # Review Focus 5
    # slot 10**9, and a dead slot → inspect is None; engine keeps ticking
async def test_select_in_an_extinct_world_clears_selection(server):        # Review Focus 5
async def test_switch_controls_reach_runner_history_and_every_viewer(server):   # Review Focus 4
    # two viewers; A sends learning false → both see status.learning False; B receives a "record" whose events include the switch
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** `encode_snapshot(state, metrics, status, inspect=None)`. Keep the fixture world at tick 0 and have `write_fixture.py` pass `inspect_entity(state, cfg, 0)`, so viewer tests get a real inspection payload with stable counts. Add `learning` and `inheritance` to the fixture's status. Regenerate the fixture.
- [ ] **Step 4: Run** the full suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat: selection, inspection payload and switch controls on the wire`.

---

### Task 7: Viewer — switches, colour by error, click to select

**Files:**
- Modify: `viewer/src/protocol.ts`, `viewer/src/controls.ts`, `viewer/src/worldView.ts`, `viewer/src/main.ts`, `viewer/index.html`
- Test: `viewer/src/protocol.test.ts`, `viewer/src/worldView.test.ts`, `viewer/src/controls.test.ts`

**Interfaces:**
- `protocol.ts`: `Status` gains `learning: boolean`, `inheritance: "learned" | "birth"`. `Snapshot` gains `inspect: Inspection | DeadInspection | null`, with `interface DeadInspection { slot: number; alive: false }` and `Inspection` matching Task 6's dict with `alive: true`. `ControlMessage` gains the three new messages. `Metrics` and `HistoryRecord` gain optional `pred_error?: number`, `learning_rate?: number` (absent in stage 1 runs).
- `worldView.ts`: `ColorMode` gains `"error"`. `errorColor(value: number, max: number): string` — a single-hue light-to-dark scale (sequential, since error is a magnitude). `pickMember(snap: Snapshot, x: number, y: number, size: number, radius = 12): number | null` — the `member_slot` of the living member nearest to canvas point (x, y) if within `radius` pixels, measuring across the wrap-around edges; `null` otherwise. `drawWorld` draws a ring around the selected member when `snap.inspect?.alive`.
- `controls.ts`: a "Learning during life" checkbox and an "Offspring inherit" select (`learned weights` / `birth weights`), sending the new controls and following `status` like the existing reseed checkbox. Colour menu gains "by prediction error".

- [ ] **Step 1: Write the failing tests**

```ts
test("fixture carries an inspection and member errors", () => {
  const snap = decodeSnapshot(readFixture("snapshot.bin"));
  expect(snap.inspect?.alive).toBe(true);
  expect(snap.arrays.member_error!.length).toBe(192);
  expect(snap.status.learning).toBe(true);
});
test("pickMember returns the nearest member within reach", () => { /* build a 3-member snapshot by hand */ });
test("pickMember returns null on empty space and on an empty world", () => {});       // Review Focus 5
test("pickMember finds a member across the wrap-around edge", () => { /* member at x=0.999, click at x=2px */ });
test("errorColor is monotonic in lightness and clamps above max", () => {});
test("errorColor handles max of zero", () => {});
```

- [ ] **Step 2: Run** `npm test`. Expected: FAIL.
- [ ] **Step 3: Implement.** In `main.ts`, a click on the world canvas sends `{type: "select", slot: pickMember(...)}` (null clears). The error colour scale's maximum is the 95th percentile of `member_error` in the current snapshot, so one outlier doesn't wash out the rest. Validate the error scale's endpoints against both page backgrounds with the dataviz skill's validator, as for the chart lines.
- [ ] **Step 4: Run** `npm test && npm run build`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: viewer switches, error colouring and click-to-select`.

---

### Task 8: Viewer — the inspector

**Files:**
- Create: `viewer/src/inspector.ts`, `viewer/src/heatmap.ts`
- Modify: `viewer/src/main.ts`, `viewer/index.html`
- Test: `viewer/src/inspector.test.ts`, `viewer/src/heatmap.test.ts`

**Interfaces:**
- `heatmap.ts`: `divergingColor(value: number, scale: number, theme): string` — two hues with a neutral midpoint at zero, clamped at ±`scale`. `drawHeatmap(ctx, rows: number[][], x, y, cell, scale, theme): void`.
- `inspector.ts`:
  - `const INPUT_NAMES: string[]` (10), `ACTION_NAMES` (5), `PREDICTED_NAMES` (6), in protocol order.
  - `class ErrorTrace { push(slot: number, tick: number, error: number): void; points(): [number, number][] }` — keeps at most 600 points; clears itself when `slot` changes; ignores a tick it already holds.
  - `weightChange(now: number[][], then: number[][]): number[][]` — elementwise difference; a shape mismatch returns zeros of `now`'s shape.
  - `class Inspector { constructor(root: HTMLElement); update(snap: Snapshot): void }`.

What the panel shows for a living selection, top to bottom:
1. **Who:** slot, colony (with its colour swatch), age of lifespan, generation, learning rate, carrying or not.
2. **This tick:** three labelled strips of cells — the 10 inputs, the 8 hidden activations, the 5 actions — each cell coloured by value and showing its number on hover.
3. **Prediction against reality:** for each of the 6 predicted senses, a pair of bars (predicted, actual) on one shared scale, with the squared error beside it.
4. **Prediction error over this member's life:** a line chart from `ErrorTrace`, since selection.
5. **Weights:** three heatmaps — `w1` (inputs → hidden), `w3` (hidden → predictions), `w2` (hidden → actions) — row and column labels from the name lists. `w1` and `w3` are captioned "trained during life", `w2` "changes only at division". A toggle switches all three between "value" and "change since selected" (`weightChange` against the weights at selection).

Other states: nothing selected → "click a member in the world to inspect it"; `alive: false` → "this member has died", with the last error trace kept on screen.

- [ ] **Step 1: Write the failing tests**

```ts
test("ErrorTrace keeps order, caps length, and resets on a new slot", () => {});
test("ErrorTrace ignores a repeated tick", () => {});            // paused engine re-sends the same snapshot
test("weightChange subtracts elementwise and survives a shape mismatch", () => {});
test("divergingColor is neutral at zero, differs by sign, and clamps", () => {});
test("divergingColor handles a zero scale", () => {});
test("name lists match the protocol's lengths", () => { /* 10, 5, 6, and the fixture's inspect lengths */ });
```

- [ ] **Step 2: Run** `npm test`. Expected: FAIL.
- [ ] **Step 3: Implement.** Charts follow the dataviz skill: bars for the predicted/actual pairs with a two-entry legend and distinct fills plus labels, one y-axis per chart, text in text ink.
- [ ] **Step 4: Run** `npm test && npm run build`. Expected: PASS.
- [ ] **Step 5: Look at it** by driving headless Chrome with real waits, as in stage 1: select a member, confirm each of the five sections renders with plausible values, switch the weight view to "change since selected" and confirm `w2` shows no change while `w1` and `w3` do, wait for the selected member to die and confirm the died message, click empty space and confirm the panel clears. Check light and dark schemes and a 360px-wide panel for overflow.
- [ ] **Step 6: Commit** `feat: entity inspector`.

---

### Task 9: Population charts, README, and the stage's success check

**Files:**
- Modify: `viewer/src/populationView.ts`, `README.md`, `engine/tests/test_bootstrap.py`
- Test: `viewer/src/populationView.test.ts`

**Interfaces:**
- `populationView.ts`: two more charts, "Prediction error" (`pred_error`) and "Learning rate" (`learning_rate`). Switch events are marked with a vertical rule like reseeds, via `eventTicks(records, name: string): number[]` replacing `reseedTicks` (keep `reseedTicks` as a thin wrapper if anything else uses it). A run recorded before stage 2 simply has no line on the two new charts.

- [ ] **Step 1: Write the failing tests**

```ts
test("eventTicks collects events by name", () => {});
test("a stage 1 history draws no points on the new charts", () => { /* seriesOf(old, "pred_error") is [] */ });
```

- [ ] **Step 2: Run, implement, run** `npm test && npm run build`. Expected: PASS.
- [ ] **Step 3: Bootstrap.** Run `.venv/bin/pytest -m slow`. Learning changes behaviour, so the default world may no longer take hold. If it fails, tune only `initial_lr` downward first; report what changed.
- [ ] **Step 4: Success check.** The stage succeeds if one entity's prediction error can be watched falling during its life.
  - Automated part: Task 3's `test_prediction_error_falls_over_a_life`.
  - Watched part: in the driven browser, select a newborn (generation > 0, age < 50) in a living world and record its error trace for 500 ticks; report the first and last 50-tick means.
- [ ] **Step 5: Comparison, reported but not pass/fail.** Three seeds × 100,000 ticks headless, each run three ways: learning on with learned inheritance, learning on with birth inheritance, learning off. Report for each: whether it took hold, final members, mean `pred_error` and mean `learning_rate` over the last 10% of records. Say plainly what the numbers do and do not show; the prior work suggests a modest effect at most, and the mean learning rate drifting toward zero would itself be a finding.
- [ ] **Step 6: README.** Document the switches, the new CLI flags, the inspector, the new charts, and that stage 1 runs cannot be resumed.
- [ ] **Step 7: Commit** `feat: learning charts, README, stage 2 success check`.
