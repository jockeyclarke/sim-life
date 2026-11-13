# Colony life simulation: design

Date: 2026-11-12

Items marked **(call)** were decided by Claude to fill a gap and were not
explicitly chosen by Paul. Everything else was agreed in conversation.

**Revised 2026-11-13:** stage 3 is now families, territory and aggression,
replacing poison (Paul's choice). Two-parent crossover replaces
single-parent copying. Stages 1 and 2 are built.

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
- Entities seeing individual neighbours or knowing their own age.
- Killing in combat. A strike imposes costs; members still die only of age
  or with their colony.
- New families arising during a run. The founding families are all there
  will be (Paul's choice; see "Later ideas").
- Families sharing energy. Each nest keeps its own pool (Paul's choice).
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
- Each of food and nest gives off a **scent** that weakens with distance.
  An entity smells only its own colony's nest. **(call)**
- **Territory scent** (stage 3). Every member marks the ground where it
  goes, automatically, more strongly near its own nest. Each family has its
  own scent, which spreads a little and fades over time. A spot "belongs" to
  the family whose scent is strongest there, so territory emerges from where
  a family's nests and members are; nobody draws borders. Marking as a
  chosen action is a later idea. **(call: automatic marking)**

## Colonies and members

- A colony has a nest, an energy pool, and a set of members.
- **Energy in:** a member that touches food while not carrying picks it up
  automatically. A carrying member that reaches its nest drops it
  automatically, adding energy to the pool.
- **Energy out:** each member costs energy every tick, with faster movement
  costing more. Stage 3 adds foreign-ground upkeep and the cost of strikes,
  below.
- **Growth:** when the pool holds a surplus, the colony gains a member. Its
  parents are chosen at random from the colony. **(call: random choice)**
- **Splitting:** when a colony reaches a size limit, its members are
  partitioned between the parent and a new daughter colony with its own
  nest, and the energy pool is divided. **(call: trigger and partition
  details are the builder's to tune)**
- **Death:** a colony whose pool reaches zero dies with all its members. A
  colony with no members is also dead.
- **Member lifespan:** members die of age, so colonies turn over
  continually. **(call)**

### Families (stage 3)

- Every colony belongs to a **family**, which never changes. A daughter
  colony belongs to its parent's family. Members of the same family are
  friends; members of different families are foes.
- A run starts with a few founding families, each with several nests and
  one founding network shared by all of them. Six families of three nests
  each, with each family's nests near one another. **(call: the numbers
  and the clustering, both settings)**
- **Where daughters go.** As in stages 1 and 2, a daughter colony's nest is
  placed a fixed distance from its parent's, in a random direction. So a
  family grows outward from its existing nests, and intruder nests arise
  where a family's frontier nests split across a border.
- Families can die out. If one family takes over the whole map, that is the
  end of the story for that run (Paul's choice): the run simply carries on
  with that one family, nothing is restarted, and a "takeover" event is
  recorded. **(call: carry on and record)** A fresh start after total
  extinction begins again with a full set of founding families.
- **Upkeep on foreign ground:** a nest standing where another family's
  scent dominates pays an extra cost each tick, growing with how strongly
  foreign the ground is. A nest founded deep inside another family's
  territory therefore starves unless its own family can claim the ground.
- **Aggression:** each member has an aggression output. A member whose
  aggression is high, with a member of another family within reach, strikes
  the nearest one:
  - the struck member drops any food it carries, and that food is lost;
    **(call: lost rather than stolen)**
  - the struck member's colony loses some energy, scaled by how strongly the
    ground smells of the striker's family: strong at home, weak abroad;
  - the striker's colony pays a fixed cost for every strike, wherever it
    happens.

  Nobody is killed by a strike. Strikes do drain energy, so repeated
  strikes can wear a colony down until it starves: that is how a family
  drives out an intruder actively, alongside upkeep. Damage per strike is
  small next to what a colony earns from food, so this is attrition, not
  slaughter. **(call: strikes may starve a colony; tuning keeps it slow)** The home advantage plus the fixed cost make
  defending cheap and raiding expensive, which is what should let borders
  hold rather than turning the field into a killing ground. How aggressive
  members are is left entirely to evolution and learning.

### Inheritance

Inheritance happens member by member.

- A new member has **two parents**, chosen at random from its colony. Each
  hidden unit, together with all of its connections in and out, is taken
  whole from one parent or the other, at random; then every weight gets a
  small random mutation. A colony of one member has one parent. (Paul's
  choice: two-parent crossover.) Swapping whole units rather than averaging
  numbers keeps each unit working: two networks' hidden units can do
  different jobs, so averaging would break both.
- Paul's choice: the copy is taken from the parent's **current, learned**
  weights, so what a member learned in life is passed on.
- A switch selects the alternative, in which the copy is taken from the
  parent's **birth** weights and lifetime learning is discarded. This needs
  each member's birth weights to be kept alongside its current weights.
  **(call: the switch)**
- Each member also carries its own learning rate, taken from one of its
  parents and mutated. If learning is harmful, evolution can drive it
  toward zero. **(call)**
- Weights are never averaged across members.
- Crossover makes a colony's members more alike than single-parent copying
  would, which works somewhat against members diverging into roles. Paul
  accepted that trade.

Because parents are chosen at random, there is no selection between
lineages inside a colony. Selection happens only through colonies surviving
and splitting.

## The entity

Everything an entity senses is local and relative to its own heading. It
never knows its map coordinates.

### Inputs

| Sense | Values | Predicted? |
|---|---|---|
| Food scent, left and right | 2 | yes |
| Nest scent, left and right | 2 | yes |
| Territory, left and right: own family's scent minus all others' (stage 3) | 2 | yes |
| Foreign members nearby (stage 3) | 1 | yes |
| Nearby colony-mates (local crowding) | 1 | yes |
| Signal heard from nearby colony-mates | 1 | yes |
| Carrying food | 1 | no |
| Colony energy level | 1 | no |
| Own memory, as written last tick | 2 | no |

10 inputs in stages 1 and 2, 13 in stage 3.

"Left and right" means two sample points offset to either side of the
entity's heading, so it can steer by comparing them.

### Outputs

| Output | Values | Notes |
|---|---|---|
| Turn | 1 | Relative to current heading |
| Speed | 1 | Faster costs more energy |
| Signal emitted | 1 | Heard by nearby colony-mates next tick |
| Memory written | 2 | Read back by this entity next tick |
| Aggression (stage 3) | 1 | Strikes a foreign member within reach when high |

The signal and memory have no meaning assigned by us.

### Network

One small network with a shared hidden layer and two sets of outputs:

- **Action outputs:** the values above (5, or 6 in stage 3).
- **Prediction outputs:** the entity's guess at next tick's value for each
  sense marked "predicted" (6 values, 9 in stage 3).

In stage 3: 13 inputs, 8 hidden units, 6 + 9 outputs, about 250 weights.
**(call: hidden size)** Runs recorded before stage 3 cannot be resumed,
because the network's shape changes; their histories can still be viewed.

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
  settings. **(call: reproducibility as a requirement)** The founding
  world, including the founding families' networks and nest layout, depends
  on the seed alone, never on the switches, so runs that differ only in a
  switch start from the same world and can be overlaid.
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
- **World view:** entities, nests and food. Members can be coloured by
  colony or by a visible internal quantity such as memory values or
  distance from the nest, which is how roles are spotted. Stage 3 adds:
  colouring members and nests by family; colouring members by aggression;
  a toggle that shades the ground by which family owns it; and a brief mark
  where a strike happens. **(call: these four)**
- **Inspector:** for the selected entity, its inputs, hidden activations,
  outputs, predictions against what was actually sensed, prediction error
  over time, and its weights.
- **Population view:** colony count, member count, energy and food
  delivered over time, with the option to overlay a previously recorded
  run. This is part of stage 1, since it is how improvement over
  generations is seen. Stage 3 adds each family's share of the territory
  over time, which shows borders pushing and pulling and a takeover as one
  family's share shrinking to nothing, and the number of strikes over time.

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
3. **Families, territory and aggression.** Two-parent crossover, families,
   territory scent, upkeep on foreign ground, aggression and strikes, their
   senses, and the viewer additions. *Success*, in Paul's words: "the field
   dominated by two to four colonies vying for territory with constant
   push/pull of the boundaries and the occasional complete takeover." In
   measurable terms:
   - **Territories form and move.** For long stretches two to four families
     each hold a clear region, and their shares of the territory keep
     changing rather than freezing.
   - **Intruders fail.** Nests founded where another family's scent
     dominates die sooner, on average, than nests founded on their own
     family's ground.
   - **Fighting has a pattern** (hoped for, not guaranteed). Strikes happen
     mostly near borders, and members of one family differ in how
     aggressive they are.

   These are computed, not judged by eye. **(call)** The run's history
   records each family's territory share and member count, the number of
   strikes split by whether they happened on the striker's own ground, foreign
   ground or near a border, and each family's mean and spread of aggression.
   Each colony's founding records its family and whether it was founded on
   foreign ground, and its death is recorded too, so nest lifespans can be
   compared afterwards. The earlier non-goal about not computing a measure of
   specialisation still stands for roles in general; these are the specific
   measures this stage's success needs.

## Left to the builder

- All numeric tuning: energy costs, food regrowth, scent falloff, mutation
  size, lifespan, colony size limit, population capacity, tick rate.
- How scents are computed (for example, coarse grid fields versus direct
  distance sums).
- The snapshot format and how often snapshots are sent.
- Project layout, tooling and test approach.
- Initial conditions: how many colonies, where nests are placed, and how
  the first weights are randomised.
- Stage 3 mechanics: the territory field's resolution, how fast scent is
  laid, spreads and fades; how upkeep grows with foreign scent; strike
  reach, threshold, damage, cost and how damage scales with home scent; how
  "foreign members nearby" is counted.

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
- A schism: a very large family splitting into two, so competition can
  begin again after a takeover.
- Marking territory as a chosen action, which would allow a marker role.
- Stealing: a striker taking the food its victim drops.
- Poison clouds, the original stage 3 plan.
- A computed measure of specialisation.
