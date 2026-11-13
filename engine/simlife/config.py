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
    initial_colonies: int = 16
    initial_members: int = 12
    initial_energy: float = 60.0

    # Network.
    n_inputs: int = 10
    n_hidden: int = 8
    n_actions: int = 5
    init_sigma: float = 0.5
    mutation_sigma: float = 0.03

    # Lifetime learning. Each member predicts its next senses (the first
    # n_predictions observation columns) and learns from its own error.
    n_predictions: int = 6
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
