# Colony life simulation: design

Date: 2026-11-12

Items marked **(call)** were decided by Claude to fill a gap and were not
explicitly chosen by Paul. Everything else was agreed in conversation.

## Intent

A playground for watching two things:

1. **Emergent roles.** Entities in a group settling into different jobs
   without being told to, "like a cell with all of its various proteins
   performing different functions".
2. **Learning at a micro scale.** Tiny neural networks whose training can be
   watched weight by weight, as a way to build intuition for how training
   behaves.

It is also a vehicle for learning how a population of small networks is
implemented efficiently with a modern ML library.

Emergent roles are the hoped-for result, not a guaranteed one. The design
makes them possible and advantageous. Whether they appear depends on tuning
done once the simulation runs. The rules are expected to change as we
experiment.

## Goals

- No fitness function. Survival and reproduction follow only from energy.
- Colonies are the unit that lives, dies and reproduces. Members of a colony
  share one energy pool, so they share a fate.
- Many colonies share one world and compete by depleting the same food.
- Each entity runs a tiny network and learns during its life by predicting
  its own next sensory input.
- Any single entity can be inspected live: inputs, activations, outputs,
  prediction error and weights.
- The engine can run without a display, and a viewer can attach and detach
  at any time.
- Learning and inheritance mode are switches, so the same world can be run
  with and without them and compared.

## Non-goals

- Hand-written forward pass or backpropagation. The library does this.
- GPU acceleration. The networks are small enough for CPU.
- A computed measure of specialisation in the first version. Roles are
  judged by eye from the viewer. **(call)**
- Entities seeing individual neighbours, sensing other colonies, or knowing
  their own age.
- Direct conflict between colonies. They compete only through shared food.
- Learning actions directly from a reward signal. See "Later ideas".

## Anti-goals

These would make the project a failure even if it runs:

- Roles that are scripted, assigned, or rewarded directly. Any score that
  says "this entity found a functional place" reintroduces the arbitrary
  fitness function this project exists to avoid.
- Meaning assigned by us to the signal channel or the memory values.
- An engine where one entity's state cannot be pulled out and read.

## The world

- A continuous 2D space in which entities move freely. Edges wrap around.
  **(call)**
- **Food** appears in patches away from nests and is depleted when taken.
  It regrows or reappears over time at a limited rate, which is the source
  of scarcity.
- **Nests** are fixed points, one per colony.
- **Poison** (stage 3) drifts through the world in slow clouds. Passing
  near it is harmless. A cloud overlapping a nest drains that colony's
  energy for as long as it stays.
- Each of food, nest and poison gives off a **scent** that weakens with
  distance. An entity smells only its own colony's nest. **(call)**

## Colonies and members

- A colony has a nest, an energy pool, and a set of members.
- **Energy in:** a member that touches food while not carrying picks it up
  automatically. A carrying member that reaches its nest drops it
  automatically, adding energy to the pool.
- **Energy out:** each member costs energy every tick, with faster movement
  costing more. Poison at the nest drains the pool. Neutralising poison
  costs energy.
- **Neutralising poison:** a member that moves into poison while not
  carrying food removes some of it, at a cost to its colony's pool.
- **Growth:** when the pool holds a surplus, one member chosen at random
  divides. **(call: random choice)**
- **Splitting:** when a colony reaches a size limit, its members are
  partitioned between the parent and a new daughter colony with its own
  nest, and the energy pool is divided. **(call: trigger and partition
  details are the builder's to tune)**
- **Death:** a colony whose pool reaches zero dies with all its members. A
  colony with no members is also dead.
- **Member lifespan:** members die of age, so colonies turn over
  continually. **(call)**

### Inheritance

Inheritance happens member by member, like cell division.

- A new member is a copy of one existing member's network weights plus a
  small random mutation.
- Paul's choice: the copy is taken from the parent's **current, learned**
  weights, so what a member learned in life is passed on.
- A switch selects the alternative, in which the copy is taken from the
  parent's **birth** weights and lifetime learning is discarded. This needs
  each member's birth weights to be kept alongside its current weights.
  **(call: the switch)**
- Each member also carries its own learning rate, inherited and mutated the
  same way. If learning is harmful, evolution can drive it toward zero.
  **(call)**
- Weights are never averaged across members.

Because division is random, there is no selection between lineages inside
a colony. Selection happens only through colonies surviving and splitting.

## The entity

Everything an entity senses is local and relative to its own heading. It
never knows its map coordinates.

### Inputs

| Sense | Values | Predicted? |
|---|---|---|
| Food scent, left and right | 2 | yes |
| Nest scent, left and right | 2 | yes |
| Poison scent, left and right (stage 3) | 2 | yes |
| Nearby colony-mates (local crowding) | 1 | yes |
| Signal heard from nearby colony-mates | 1 | yes |
| Carrying food | 1 | no |
| Colony energy level | 1 | no |
| Own memory, as written last tick | 2 | no |

10 inputs in stages 1 and 2, 12 in stage 3.

"Left and right" means two sample points offset to either side of the
entity's heading, so it can steer by comparing them.

### Outputs

| Output | Values | Notes |
|---|---|---|
| Turn | 1 | Relative to current heading |
| Speed | 1 | Faster costs more energy |
| Signal emitted | 1 | Heard by nearby colony-mates next tick |
| Memory written | 2 | Read back by this entity next tick |

The signal and memory have no meaning assigned by us.

### Network

One small network with a shared hidden layer and two sets of outputs:

- **Action outputs:** the 5 values above.
- **Prediction outputs:** the entity's guess at next tick's value for each
  sense marked "predicted" (6 values, 8 in stage 3).

Size is roughly 12 inputs, 8 hidden units, 5 + 8 outputs, a little over
200 weights. **(call: hidden size)**

### Two kinds of change

- **During life:** the prediction error (guess versus what was actually
  sensed next tick) is backpropagated. This updates the hidden layer and
  the prediction outputs. It does not update the weights that feed the
  action outputs.
- **At division:** every weight, including the action weights, receives a
  small mutation.

So behaviour changes during life only indirectly, because the action
outputs read a hidden layer that learning is reshaping. Nothing during a
life rewards a good action directly. Only colony survival does that.

A global switch turns lifetime learning off, giving a pure-evolution
control run.

### Prior work

This arrangement follows Nolfi, Elman and Parisi, "Learning and Evolution
in Neural Networks", *Adaptive Behavior* 3(1), 1994, as reported in
secondary sources. They found prediction learning raised average fitness
but not peak fitness. Their explanation assumed learned weights were
discarded each generation, so it applies to the "birth weights" switch
setting and should not be expected to carry over to the default.

## How it gets made

### Engine

- Python with JAX, running on CPU.
- The simulation is one pure function from world state to world state. The
  state is a record of arrays: positions, headings, energies, weights,
  birth weights, ages, colony membership, and so on.
- The network is written for a single entity. Running it over the whole
  population, and computing every entity's gradient, is done by
  transforming that function (`vmap`, `grad`, `jit`) and never by looping
  over entities.
- Births and deaths use a fixed-capacity population with an "alive" mask,
  because compiled array code needs fixed shapes. The same applies to
  colonies.
- Randomness uses explicit keys, so a run is reproducible from its seed and
  settings. **(call: reproducibility as a requirement)**
- The engine runs with or without a viewer attached, and can run faster
  than real time when nothing is watching.
- State can be saved to disk and resumed. **(call)**
- Each run records its seed, settings and a history of population measures
  (colony count, member count, energy, food delivered) to disk, whether or
  not a viewer is attached. Comparing switch settings means running the
  same seed once per setting and overlaying the recorded histories.
  Stepping several worlds side by side is not required. **(call)**
- **Extinction:** if every colony dies, the run starts a fresh set of
  random colonies and records that it did so. This is the same as
  restarting by hand. A switch disables it. New colonies are never injected
  while any colony is alive, because that would be an outside intervention
  in who survives. **(call)**
- Early extinction is the likely first result with random starting weights.
  Stage 1 includes tuning starting energy and food density so that random
  wanderers sometimes deliver food.

### Viewer

- A separate browser page in TypeScript, drawing on a canvas.
- The engine publishes snapshots over a WebSocket. The viewer sends back
  control messages: pause, speed, switches, and which entity is selected.
- **World view:** entities, nests, food and poison. Members can be coloured
  by colony or by a visible internal quantity such as memory values or
  distance from the nest, which is how roles are spotted.
- **Inspector:** for the selected entity, its inputs, hidden activations,
  outputs, predictions against what was actually sensed, prediction error
  over time, and its weights.
- **Population view:** colony count, member count, energy and food
  delivered over time, with the option to overlay a previously recorded
  run. This is part of stage 1, since it is how improvement over
  generations is seen.

### Environment notes

- The machine is an Apple M5 Pro MacBook with Python 3.14.7 and Node 26
  installed and no Python environment manager.
- Checked on 2026-11-12: JAX 0.12.0.1 supports Python 3.12 to 3.15 and
  ships Apple Silicon builds for each, so the installed Python works.
- The project uses its own isolated Python environment.
- Apple GPU support for JAX is experimental and is not to be used.

## Stages

Each stage is usable on its own and is finished before the next begins.

1. **Evolution only.** World, food, nests, colonies with shared energy,
   member division with mutation, colony splitting and death, headless
   running, run recording, the world view and the population view.
   Lifetime learning is absent or off.
   *Success:* colonies visibly get better at foraging over generations.
2. **Lifetime learning.** Prediction outputs, per-entity training,
   per-member learning rate, the learning switch, the inheritance switch,
   and the inspector. *Success:* one entity's prediction error can be
   watched falling during its life.
3. **Poison.** Poison clouds, poison scent, neutralising, and the
   colouring modes that help spot roles. *Success:* at least some colonies
   show members that consistently stay near the nest while others range
   out. This outcome is hoped for and not guaranteed.

## Left to the builder

- All numeric tuning: energy costs, food regrowth, scent falloff, mutation
  size, lifespan, colony size limit, population capacity, tick rate.
- How scents are computed (for example, coarse grid fields versus direct
  distance sums).
- The snapshot format and how often snapshots are sent.
- Project layout, tooling and test approach.
- Initial conditions: how many colonies, where nests are placed, and how
  the first weights are randomised.

A reasonable starting scale is a few thousand members across a few dozen
colonies, running at interactive speed. **(call)**

## Later ideas

Not part of this design. Recorded so they aren't lost.

- Letting division favour some members, which allows a lineage to spread
  inside a colony at the colony's expense (the simulation's version of
  cancer).
- An evolved "is this situation good or bad" network that supplies a
  lifetime reward for actions, after Ackley and Littman.
- Evolution supplying the training signal itself, after Nolfi and Parisi's
  auto-teaching networks.
- Stealing or fighting between colonies.
- A computed measure of specialisation.
