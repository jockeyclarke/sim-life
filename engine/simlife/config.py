"""Every tunable number in the simulation.

Config is frozen and hashable so it can be a static argument to jitted
functions: changing a value recompiles, and nothing inside the step function
reads mutable settings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class Config:
    # Capacities. Array shapes are fixed by these for the whole run.
    capacity_members: int = 2048
    capacity_colonies: int = 128
    capacity_food: int = 512
    num_patches: int = 8
    patch_radius: float = 0.05

    # Starting conditions.
    initial_colonies: int = 18
    initial_members: int = 12
    initial_energy: float = 60.0

    # Network.
    n_inputs: int = 13
    n_hidden: int = 8
    n_actions: int = 6
    init_sigma: float = 0.5
    mutation_sigma: float = 0.03

    # Lifetime learning. Each member predicts its next senses (the first
    # n_predictions observation columns) and learns from its own error.
    n_predictions: int = 9
    initial_lr: float = 0.02
    max_lr: float = 0.5
    lr_mutation_sigma: float = 0.1  # learning rate mutates by about this fraction per division
    grad_clip: float = 1.0  # cap on the size of one gradient, to stop blow-ups
    error_smoothing: float = 0.98  # for the running average of prediction error

    # Lifespan, in ticks.
    lifespan_mean: int = 4000
    lifespan_jitter: int = 1000

    # Movement and sensing. The world is the unit square with wrap-around.
    max_speed: float = 0.004
    max_turn: float = 0.5
    sensor_distance: float = 0.02
    sensor_angle: float = 0.6
    food_scent_length: float = 0.08
    # food direction input = tanh(gain * (left - right) / (left + right))
    food_direction_gain: float = 4.0
    nest_scent_length: float = 0.15
    neighbour_grid: int = 32
    pickup_radius: float = 0.02
    nest_radius: float = 0.04

    # Energy.
    food_energy: float = 10.0
    base_cost: float = 0.001
    speed_cost: float = 0.004
    birth_cost: float = 4.0
    reserve_per_member: float = 2.0

    # Families and territory (stage 3).
    n_families: int = 6
    # Families form when the grace period ends: each living colony joins the
    # family of the nearest of n_families regional centres.
    grace_ticks: int = 20_000  # ticks after a world is founded with no upkeep or strikes
    territory_grid: int = 64
    mark_rate: float = 0.02  # scent each member lays per tick where it stands
    nest_mark: float = 0.2  # scent each nest lays per tick
    scent_fade: float = 0.995
    scent_spread: float = 0.1
    upkeep_rate: float = 0.01  # extra cost per tick for a nest on wholly foreign ground

    # Aggression (stage 3).
    strike_threshold: float = 0.5
    strike_reach: float = 0.015
    strike_damage: float = 0.5  # at full home advantage; scaled by home share
    strike_cost: float = 0.1  # paid by the striker's colony, wherever it strikes
    border_band: float = 0.2  # home share within 0.5 ± band/2 counts as border

    # Colony splitting and food regrowth.
    split_size: int = 32
    split_distance: float = 0.15
    food_regrow_prob: float = 0.001

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown Config fields: {sorted(unknown)}")
        return cls(**d)
