# Stage 3: Families, Territory and Aggression — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Colonies belong to a few fixed families that mark territory, pay upkeep on foreign ground, and can evolve to strike members of other families; children have two parents.

**Architecture:** A per-family scent grid joins the world state and is updated inside `step` (deposit, spread, fade). New senses read it; a sixth action output drives strikes, resolved as one vectorised phase. Crossover replaces single-parent copying in `grow`. Colony founding and death statistics are accumulated in the state so the success measures are exact; the runner, recorder, protocol and viewer gain family-level measures and views.

**Tech Stack:** As stages 1–2: Python 3.14, JAX (CPU), `websockets`, pytest; TypeScript, Vite, Vitest, Canvas 2D.

**Spec:** `docs/superpowers/specs/2026-11-12-colony-life-sim-design.md` — "Families (stage 3)", "Inheritance", "Inputs", "Outputs", "Network", "Viewer", "Stages → 3". Earlier plans: `docs/superpowers/plans/2026-11-12-stage-1-evolution.md`, `docs/superpowers/plans/2026-11-13-stage-2-lifetime-learning.md`.

## Global Constraints

- Everything in stages 1 and 2's Global Constraints still holds (no fitness function or reward; no Python loop over entities; fixed shapes and alive masks; explicit keys; same seed and settings give the same run however chunked; never inject colonies while any is alive; weights never averaged; only prediction error enters a gradient; `w2`/`b2` change only at division).
- A family never changes. Daughters belong to their parent's family. No new family ever appears during a run. A fresh start after total extinction creates a full set of founding families.
- Each colony keeps its own energy. Families never share energy.
- No member is killed by a strike. Members die only of age or with their colony.
- Members of the same family never strike each other.
- The founding world (families' networks and nest layout) depends on the seed alone, never on the switches.
- Crossover takes each hidden unit whole (its row of `w1`, its `b1` entry, its columns of `w2` and `w3`) from one parent. Never average.
- Runs recorded before this stage need not resume (Paul agreed). They fail with the existing "older version" error.
- Work on branch `stage-3-territory`. End each commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A family dies out, or one takes over.** Measures for a dead family read zero; the "takeover" event is written once; charts and colours don't break with families at zero. → Task 6, Task 9.
2. **A colony with one member reproduces.** Its child has that member as both parents; nothing crashes or indexes another colony. → Task 5.
3. **Everyone maximally aggressive.** With every aggression output forced to 1, a long run stays finite, no member dies except by age or colony death, and colonies may starve. → Task 4.
4. **Territory field over a long run.** Values stay finite and bounded (fading guarantees it), and ownership is well defined where no family has marked. → Task 2.
5. **Viewer meets an older engine or history.** Snapshots or records without family fields render without throwing; family charts are simply empty. → Tasks 8, 9.

---

## File Structure

New engine files: `engine/simlife/territory.py` (the scent field: update, read, ownership), `engine/simlife/combat.py` (strikes). Modified: `config.py`, `state.py`, `brain.py`, `senses.py`, `physics.py` (upkeep), `lifecycle.py`, `step.py`, `metrics.py`, `recorder.py`, `runner.py`, `protocol.py`, `inspect.py`, `server.py`, `engine/tools/write_fixture.py`.

New viewer file: `viewer/src/families.ts` (family colours, territory shading). Modified: `protocol.ts`, `worldView.ts`, `controls.ts`, `inspector.ts`, `populationView.ts`, `main.ts`.

---

### Task 1: Shapes, families at founding, new state

**Files:**
- Modify: `engine/simlife/config.py`, `engine/simlife/state.py`, `engine/tests/helpers.py`
- Test: `engine/tests/test_state.py`, plus stage 1–2 tests whose premise is the old shapes

**Interfaces:**
- New `Config` fields and changed defaults:

| Field | Value | Field | Value |
|---|---|---|---|
| `n_inputs` | 13 (was 10) | `territory_grid` | 64 |
| `n_actions` | 6 (was 5) | `mark_rate` | 0.02 |
| `n_predictions` | 9 (was 6) | `nest_mark` | 0.2 |
| `initial_colonies` | 18 (was 16) | `scent_fade` | 0.995 |
| `n_families` | 6 | `scent_spread` | 0.1 |
| `family_spread` | 0.08 | `upkeep_rate` | 0.01 |
| `strike_threshold` | 0.5 | `strike_reach` | 0.015 |
| `strike_damage` | 0.5 | `strike_cost` | 0.1 |
| `border_band` | 0.2 | | |

- Colony `c` founded at start belongs to family `c % n_families` (so small test configs still work). Each family has one founding network, copied to every member of every founding nest of that family. Each family has a centre drawn uniformly; its nests lie uniformly within `family_spread` of it (wrapped).
- New `WorldState` fields (C colonies, K = `n_families`, G = `territory_grid`):

| Field | Shape, dtype | Meaning |
|---|---|---|
| `col_family` | (C,) i32 | family of each colony |
| `col_born` | (C,) i32 | tick the colony was founded |
| `col_foreign` | (C,) bool | founded where another family's scent dominated |
| `territory` | (K,G,G) f32 | each family's scent |
| `aggression` | (N,) f32 | aggression output last tick |
| `striking` | (N,) bool | struck someone this tick |
| `strikes` | (3,) i32 | cumulative strikes on own, foreign, border ground |
| `colonies_died` | (2,) i32 | colonies that died, founded at home / on foreign ground |
| `colony_life_total` | (2,) i32 | summed lifespans of those colonies, in ticks |

- `init_world`: `col_family` as above (dead slots 0), `col_born` 0, `col_foreign` False, `territory` zeros, `aggression` zeros, `striking` False, counters zero.

- [ ] **Step 1: Branch.** `git switch -c stage-3-territory`.
- [ ] **Step 2: Write the failing tests**

```python
def test_founding_families():
    cfg = Config(); s = init_world(jax.random.key(0), cfg)
    assert int(s.col_alive.sum()) == 18 and s.territory.shape == (6, 64, 64)
    fam = s.col_family[:18]
    assert sorted(np.bincount(np.asarray(fam)).tolist()) == [3] * 6
    # one network per family: members of two nests of the same family are identical
    a = s.alive & (s.colony == 0); b = s.alive & (s.colony == 6)          # both family 0
    assert bool((s.params["w1"][a][0] == s.params["w1"][b][0]).all())
    c = s.alive & (s.colony == 1)                                           # family 1
    assert not bool((s.params["w1"][a][0] == s.params["w1"][c][0]).all())
    assert s.params["w2"].shape == (cfg.capacity_members, 6, 8) and s.params["w3"].shape[1] == 9

def test_family_nests_start_near_one_another():
    # every pair of nests in one family is within 2 * family_spread (torus distance)

def test_founding_world_ignores_switches():
    # init_world twice with the same key; flip learning_on/inherit_learned on one afterwards;
    # params, nests, col_family identical (switches are set after init and never read by it)
```

- [ ] **Step 3: Run** `.venv/bin/pytest -q`. Expected: new tests FAIL, and stage 1–2 tests that hard-code 10 inputs, 5 actions, 6 predictions, 16 colonies or 192 members fail too.
- [ ] **Step 4: Implement.** Update every such stage 1–2 test to read the sizes from `Config` instead of hard-coding them (do not weaken what they assert). `helpers.blank` gives both colonies family 0 and 1. Checkpoint save/load needs no change beyond the new fields (it walks the state).
- [ ] **Step 5: Run** the full suite. Expected: PASS.
- [ ] **Step 6: Commit** `feat: families at founding and stage 3 state`.

---

### Task 2: The territory field and upkeep

**Files:**
- Create: `engine/simlife/territory.py`
- Modify: `engine/simlife/physics.py` (`metabolise` adds upkeep), `engine/simlife/step.py`
- Test: `engine/tests/test_territory.py`

**Interfaces:**
- `cell_of(points: (M,2), cfg) -> (M,) i32` — flat index `y*G + x` into the territory grid.
- `update_territory(state, cfg) -> WorldState` — in this order: fade (`* scent_fade`); spread (each cell becomes `(1 - scent_spread) * itself + scent_spread * mean of its 4 neighbours`, wrapping); deposit (`mark_rate` per living member at its cell for its family; `nest_mark` per living colony at its nest cell for its family).
- `scent_at(state, points: (M,2), family: (M,) i32, cfg) -> tuple[own (M,), foreign (M,)]` — own family's scent and the sum of all other families' at each point's cell.
- `home_share(own, foreign) -> (M,)` — `own / (own + foreign)`, and 0.5 where both are below 1e-6 (unmarked ground is neutral).
- `owner_grid(state, cfg) -> tuple[owner (G,G) i32, strength (G,G) f32]` — family with the most scent, or −1 where the total is below 1e-3; strength is that family's share of the total.
- `metabolise` additionally charges each living colony `upkeep_rate * (1 - home_share)` at its nest cell.
- `step`: `update_territory` runs after `metabolise`.

- [ ] **Step 1: Write the failing tests**

```python
def test_marks_accumulate_where_members_are_and_fade_elsewhere():
def test_spread_moves_scent_to_neighbours_and_wraps_at_edges():
def test_unmarked_ground_is_neutral_and_unowned():
    # home_share 0.5; owner -1
def test_owner_is_the_family_with_most_scent():
def test_nest_on_foreign_ground_pays_upkeep_and_at_home_does_not():
    # family 1 scent saturates the cell of colony 0's nest → colony 0 pays ≈ upkeep_rate per tick extra;
    # colony 0's own scent there instead → ≈ 0 extra
def test_territory_stays_finite_and_bounded_over_a_long_run():       # Review Focus 4
    # 20,000 ticks, default config: finite, >= 0, and max <= (members * mark_rate + nests * nest_mark) / (1 - scent_fade)
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** with `jnp.roll` for the spread and `jax.ops.segment_sum` for deposits (flat index `family * G*G + cell`).
- [ ] **Step 4: Run** the full suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat: territory scent and upkeep on foreign ground`.

---

### Task 3: New senses

**Files:**
- Modify: `engine/simlife/senses.py`, `engine/simlife/learning.py` (only if it hard-codes 6)
- Test: `engine/tests/test_senses.py`

**Interfaces:**
- `observe(state, cfg) -> (N, 13)` with this fixed column order (the predicted senses are the first 9, so `obs[:, :n_predictions]` stays correct):

```
0 food left      1 food right     2 nest left      3 nest right
4 crowd          5 heard          6 territory left 7 territory right
8 foreign nearby
9 carrying       10 colony energy 11 memory 0      12 memory 1
```

- Territory left/right = `tanh(own - foreign)` at the two sensor points, for the member's own family.
- `foreign_nearby(state, cfg) -> (N,)` = `tanh(count / 4)`, where `count` is the number of living members of other families in the member's `neighbour_grid` cell.

- [ ] **Step 1: Write the failing tests**

```python
def test_territory_sense_is_positive_on_own_ground_and_negative_on_foreign():
def test_territory_sense_compares_left_and_right():
def test_foreign_nearby_counts_other_families_only():
    # same cell: one colony-mate, one member of another colony of the SAME family, two of another family → tanh(2/4)
def test_observation_column_order():
    # carrying at column 9, colony energy 10, memory 11:13; dead rows all zero
```

- [ ] **Step 2: Run.** Expected: FAIL. Update the stage 1–2 senses tests that index columns 6–9.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat: territory and foreign-member senses`.

---

### Task 4: Aggression and strikes

**Files:**
- Create: `engine/simlife/combat.py`
- Modify: `engine/simlife/brain.py` (`act` returns aggression), `engine/simlife/step.py`, `engine/simlife/inspect.py`
- Test: `engine/tests/test_combat.py`, `engine/tests/test_brain.py`

**Interfaces:**
- `act(params, x) -> (turn, speed, signal, memory (2,), aggression)` — `aggression = sigmoid(o[5])`. `inspect_entity`'s `"actions"` gains it as a sixth value.
- `step` stores `aggression` (0 for dead members), and calls `strike(state, cfg)` right after `pickup`.
- `strike(state, cfg) -> WorldState`:
  - A **striker** is a living member with `aggression > strike_threshold` and at least one living member of another family within `strike_reach` (torus distance). Its **victim** is the nearest such member.
  - For each strike: the victim stops carrying (the food is lost); the victim's colony loses `strike_damage * home` where `home = home_share` at the striker's position for the striker's family; the striker's colony loses `strike_cost`.
  - `striking` is True for strikers this tick, False otherwise.
  - `strikes[0]` (own ground, `home > 0.5 + border_band/2`), `strikes[1]` (foreign, `home < 0.5 - border_band/2`), `strikes[2]` (border, otherwise) increase by the strikes of each kind.
  - Distances use one (N,N) torus distance matrix; members of the striker's own family, dead members and itself are masked out.

- [ ] **Step 1: Write the failing tests**

```python
def test_an_aggressive_member_strikes_the_nearest_foreigner():
def test_the_same_family_never_strikes():
    # two colonies of one family, touching, aggression 1 → no strike, no energy change
def test_strike_effects_and_costs():
    # victim carrying → not carrying; victim colony −strike_damage*home; striker colony −strike_cost
def test_strikes_are_stronger_at_home_than_abroad():
    # same strike on the striker's own saturated ground vs the victim's: damage ratio ≈ (1.0) / (≈0)
def test_strikes_are_classified_by_ground():
def test_nobody_dies_from_a_strike():
    # alive mask unchanged by strike() whatever the energies
def test_everyone_maximally_aggressive_stays_finite():            # Review Focus 3
    # force w2/b2 so aggression ≈ 1 for all; 5,000 ticks; all floats finite; members only leave
    # through age_and_die or kill_dead_colonies (run strike() alone on a world: alive unchanged)
def test_act_returns_six_actions_and_aggression_in_unit_range():
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full suite. Expected: PASS.
- [ ] **Step 5: Measure speed** at default `Config` (2000 ticks after warm-up). Record it in the commit message. If below 150 ticks/s, report and continue; do not change the mechanics to gain speed.
- [ ] **Step 6: Commit** `feat: aggression and strikes`.

---

### Task 5: Two-parent crossover, family inheritance, colony records

**Files:**
- Modify: `engine/simlife/lifecycle.py`
- Test: `engine/tests/test_lifecycle.py`

**Interfaces:**
- `grow` — same signature. Parents A and B are drawn independently and uniformly from the colony's living members (a one-member colony gives A = B). The source weights for each parent are `params` or `birth_params` per `inherit_learned`, as in stage 2. For each hidden unit `h`, a fair coin picks A or B; the child takes from that parent `w1[h, :]`, `b1[h]`, `w2[:, h]`, `w3[:, h]`. `b2` and `b3` are taken element by element from A or B by fair coins. Then mutation as before. `lr` comes from A, mutated as before. `generation = max(generation[A], generation[B]) + 1`.
- `split` — the daughter's `col_family` is the parent's; `col_born = tick`; `col_foreign = home_share < 0.5` at the daughter's nest for that family.
- `kill_dead_colonies(state)` — for each colony that dies this tick, adds 1 to `colonies_died[col_foreign]` and `tick - col_born` to `colony_life_total[col_foreign]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_child_takes_each_hidden_unit_whole_from_one_parent():
    # parents with distinct constant weights (A all +1, B all −1), mutation 1e-6:
    # for every h: (w1[h,:], b1[h], w2[:,h], w3[:,h]) all ≈ +1 or all ≈ −1; over 40 seeds both occur
def test_crossover_never_averages():
    # no child weight is near 0 in the setup above
def test_one_member_colony_has_one_parent():                      # Review Focus 2
def test_parents_come_from_the_childs_own_colony():
def test_generation_is_one_more_than_the_older_parent():
def test_daughter_belongs_to_its_parents_family():
def test_daughter_founded_on_foreign_ground_is_marked():
def test_colony_deaths_are_counted_by_founding_ground():
```

- [ ] **Step 2: Run.** Expected: FAIL. Stage 2 inheritance tests (learned vs birth, learning-rate mutation) must still pass; adjust only their setup if it relied on a single parent.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat: two-parent crossover and family inheritance`.

---

### Task 6: Family measures, strikes in history, takeover

**Files:**
- Modify: `engine/simlife/metrics.py`, `engine/simlife/recorder.py`, `engine/simlife/runner.py`
- Test: `engine/tests/test_recorder.py`, `engine/tests/test_runner.py`

**Interfaces:**
- `compute_metrics` gains (lists have length `n_families`; a dead family reads 0):
  - `"families"` — number of families with a living colony;
  - `"family_members"`, `"family_colonies"`;
  - `"family_territory"` — share of all grid cells owned by each family (from `owner_grid`);
  - `"family_aggression_mean"`, `"family_aggression_sd"` — over that family's living members;
  - `"strikes_own_total"`, `"strikes_foreign_total"`, `"strikes_border_total"` — cumulative;
  - `"colonies_died_home"`, `"colonies_died_foreign"`, `"colony_life_home"`, `"colony_life_foreign"` — counts and mean lifespans (0.0 when none).
- `Recorder.record` adds `"strikes_own"`, `"strikes_foreign"`, `"strikes_border"`: counts since the previous record, the same way it computes `delivery_rate` (restored from the last measured record on resume).
- `Runner`: after every piece, if the number of living families has dropped to 1 from more than 1, write event `{"event": "takeover", "family": k, "tick": t}` once. On a fresh start after extinction, carry over `strikes`, `colonies_died` and `colony_life_total`, and set the new founders' `col_born` to the current tick.

- [ ] **Step 1: Write the failing tests**

```python
def test_family_metrics_are_json_ready_and_sized_by_family():
def test_dead_family_reads_zero():                                 # Review Focus 1
def test_strikes_are_recorded_per_interval_and_survive_resume():
def test_takeover_event_written_once(tmp_path):                    # Review Focus 1
    # kill every colony but family 2's → one takeover event naming family 2; more advances add none
def test_fresh_start_keeps_cumulative_counters_and_dates_founders(tmp_path):
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat: family measures, strike history, takeover event`.

---

### Task 7: Protocol, server and fixture

**Files:**
- Modify: `engine/simlife/protocol.py`, `engine/tools/write_fixture.py`
- Test: `engine/tests/test_protocol.py`

**Interfaces:**
- Snapshot arrays gain, in this order after `member_error`: `member_family` (M,), `member_aggression` (M,); after `colony_members`: `colony_family` (K,); after `food_pos`: `strike_pos` (S,2) positions of this tick's strikers, `territory_owner` (G,G) family or −1, `territory_strength` (G,G).
- Header gains `"n_families"`.
- The fixture is regenerated (now 18 nests, 216 members, 6 families).

- [ ] **Step 1: Write the failing tests**

```python
def test_snapshot_carries_families_territory_and_strikes():
def test_snapshot_of_a_world_with_no_strikes_has_an_empty_strike_array():
```

- [ ] **Step 2: Run, implement, run** the full engine suite. Expected: PASS. Update the stage 1–2 protocol test that lists array names.
- [ ] **Step 3: Regenerate** the fixture; run `npm test` in `viewer/`. Expected: only tests that hard-code 192 members or 16 nests fail; update them to read sizes from the fixture.
- [ ] **Step 4: Commit** `feat: families, territory and strikes on the wire`.

---

### Task 8: Viewer — families, territory, aggression, strikes

**Files:**
- Create: `viewer/src/families.ts`
- Modify: `viewer/src/protocol.ts`, `viewer/src/worldView.ts`, `viewer/src/controls.ts`, `viewer/src/inspector.ts`, `viewer/src/main.ts`, `viewer/index.html`
- Test: `viewer/src/families.test.ts`, `viewer/src/inspector.test.ts`, `viewer/src/protocol.test.ts`

**Interfaces:**
- `families.ts`: `familyColor(k: number, dark: boolean): string` — six fixed colours, light and dark sets, run through the dataviz skill's validator (`--pairs all`); if a pair fails, add secondary encoding (nest ring dash patterns by family) rather than more hues. `drawTerritory(ctx, owner: Float32Array, strength: Float32Array, grid: number, size: number, dark: boolean): void` — each owned cell filled in its family colour at alpha `0.08 + 0.22 * strength`; unowned cells left blank.
- `worldView.ts`: `ColorMode` gains `"family"` (the new default) and `"aggression"` (a one-hue scale like the error scale). Nests are drawn in their family colour. Strikes are drawn as small crosses at `strike_pos`. Territory shading is drawn under everything when enabled.
- `controls.ts`: a "Shade territory" checkbox (view-only, never sent to the engine); colour menu adds "by family" and "by aggression".
- `inspector.ts`: `INPUT_NAMES` (13), `ACTION_NAMES` (6, adding "aggression"), `PREDICTED_NAMES` = first 9, in Task 3's order; the "who" line shows the family with its colour swatch.
- `protocol.ts`: the new arrays are optional (`?`) in practice: every reader tolerates their absence.

- [ ] **Step 1: Write the failing tests**

```ts
test("family colours are stable, distinct and differ by scheme", () => {});
test("drawTerritory tolerates missing arrays and an empty grid", () => {});   // via a stub context recording fillRect calls
test("unowned cells are not filled", () => {});
test("name lists match the fixture's inspection lengths", () => {});           // 13, 6, 9
test("a snapshot without family arrays still decodes and pickMember works", () => {});   // Review Focus 5
```

- [ ] **Step 2: Run** `npm test`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm test && npm run build`. Expected: PASS.
- [ ] **Step 5: Look at it** in driven headless Chrome against a live engine: families coloured, territory shading on and off, strike crosses appear, aggression colouring, inspector shows 13 senses and 6 actions; light and dark; no horizontal overflow at 1600, 1200 and 400 px.
- [ ] **Step 6: Commit** `feat: viewer families, territory shading, aggression and strikes`.

---

### Task 9: Population charts and README

**Files:**
- Modify: `viewer/src/populationView.ts`, `README.md`
- Test: `viewer/src/populationView.test.ts`

**Interfaces:**
- `familySeries(records, field: "family_territory" | "family_members", k: number): [number, number][]`.
- Two new charts: **Territory by family** (one line per family in its family colour, shares on a single 0–1 axis, legend with the family number; families never present are omitted), and **Strikes per 100 ticks** (one line: own + foreign + border). Takeover events get a vertical rule like reseeds.

- [ ] **Step 1: Write the failing tests**

```ts
test("familySeries reads one family's share and skips records without family fields", () => {});   // Review Focus 5
test("a family at zero stays on the chart until the end, then shows its takeover", () => {});      // Review Focus 1: series continues at 0
test("eventTicks finds takeover events", () => {});
```

- [ ] **Step 2: Run, implement, run** `npm test && npm run build`. Expected: PASS.
- [ ] **Step 3: README.** Families, territory, aggression, crossover, the new colour modes, territory shading, the two charts, and that earlier runs cannot be resumed.
- [ ] **Step 4: Commit** `feat: territory and strike charts, README`.

---

### Task 10: Tuning and the success check

**Files:**
- Modify: `engine/simlife/config.py` (defaults only), `engine/tests/test_bootstrap.py`

- [ ] **Step 1: Bootstrap test.** Update it for families: of 5 seeds at 20,000 ticks with reseeding off, at least 3 not extinct, with at least one split, and at least 2 families still alive. Run `.venv/bin/pytest -m slow`. Expected: likely FAIL before tuning.
- [ ] **Step 2: Tune** only these defaults: `mark_rate`, `nest_mark`, `scent_fade`, `scent_spread`, `upkeep_rate`, `strike_damage`, `strike_cost`, `strike_reach`, `family_spread`. Do not add mechanisms. Record every value tried and its effect in the ledger.
- [ ] **Step 3: Success check.** 3 seeds × 200,000 ticks headless, default settings. Report, per seed:
  - **Territories form and move:** share of records with 2–4 families alive; mean absolute change in each family's territory share per 10,000 ticks (frozen borders read near zero).
  - **Intruders fail:** `colony_life_foreign` against `colony_life_home`, with counts.
  - **Fighting pattern (hoped for):** strikes split own / border / foreign; mean of `family_aggression_sd`.
  - Takeovers and when they happened.
  State plainly which parts were met. Do not change a measure after seeing results.
- [ ] **Step 4: Commit** `feat: stage 3 tuning and success check`.

---

## Added 2026-11-13, after tuning (Paul's choices)

### Task 11: Families form when the grace period ends

- `Config`: remove `family_spread` and `founder_variation`; keep `grace_ticks` (20,000).
- `WorldState` gains `family_centre` (K,2) f32 and `founded` () i32 (the tick this world began).
- `init_world`: nests uniform; one random network per colony; `family_centre` uniform; `col_family` = nearest centre (torus) of each nest.
- Competition (upkeep, strikes) applies when `tick >= founded + grace_ticks`.
- `form_families(state, cfg)` in `step`: on the tick `tick == founded + grace_ticks`, every colony's family becomes its nest's nearest centre and `territory` is zeroed.
- `Runner`: a fresh start sets `founded` to the current tick; writes event `{"event": "families_formed", "tick"}` once per world when the grace period ends.
- Tests: colonies independent at founding; family = nearest centre; families re-formed and scent cleared exactly at the grace tick and not before; fresh start restarts the grace period; event written once.

### Task 12: Seed a new run from an earlier run's survivors

- `init_world(key, cfg, founders: dict | None = None)`: when given, `founders` is a pool of networks (leading axis = survivors); each founding colony gets one drawn uniformly (by the world key).
- `load_survivors(run_dir) -> tuple[dict, dict]` (pool params, pool lr): from the latest checkpoint, living members only. Raises `ValueError` naming the mismatch if the source's network shape differs from this `Config`.
- `Recorder` writes the pool to `founders.npz` in the new run and `"seeded_from"` into `settings.json`; `resume_run` and fresh starts reuse it.
- CLI: `--seed-from RUN_DIR`.
- Tests: founders come from the pool; shape mismatch is a clear error; resume and fresh start reuse the pool; same seed and same pool give the same world.
