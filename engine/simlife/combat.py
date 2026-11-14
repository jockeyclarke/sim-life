"""Strikes between members of different families.

Nothing here tells anyone to fight. A member strikes when its own aggression
output is high and a foreigner is within reach. A strike costs the striker's
colony a fixed amount and costs the victim's colony an amount that grows with
how much the ground smells of the striker's family: defending at home is
cheap and hurts, raiding abroad is dear and barely scratches. Nobody is
killed by a strike.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Config
from .senses import torus_distance
from .state import WorldState
from .territory import home_share, scent_at


def strike(state: WorldState, cfg: Config) -> WorldState:
    C = cfg.capacity_colonies
    family = state.col_family[state.colony]

    # One (N, N) distance matrix; only living members of another family count.
    distance = torus_distance(state.pos[:, None, :], state.pos[None, :, :])
    foe = state.alive[None, :] & (family[None, :] != family[:, None])
    distance = jnp.where(foe, distance, jnp.inf)
    victim = jnp.argmin(distance, axis=1)  # each member's nearest foreigner
    in_reach = jnp.min(distance, axis=1) <= cfg.strike_reach
    competing = state.tick >= state.founded + cfg.grace_ticks  # nobody strikes during the grace period
    striking = state.alive & (state.aggression > cfg.strike_threshold) & in_reach & competing

    own, foreign = scent_at(state, state.pos, family, cfg)
    home = home_share(own, foreign)  # at the striker's position, for its family

    damage = jnp.where(striking, cfg.strike_damage * home, 0.0)
    cost = jnp.where(striking, cfg.strike_cost, 0.0)
    energy = (
        state.energy
        - jax.ops.segment_sum(damage, state.colony[victim], num_segments=C)
        - jax.ops.segment_sum(cost, state.colony, num_segments=C)
    )
    struck = jnp.zeros_like(state.alive).at[victim].max(striking)

    upper, lower = 0.5 + cfg.border_band / 2, 0.5 - cfg.border_band / 2
    on_own, on_foreign = home > upper, home < lower
    kinds = jnp.stack([on_own, on_foreign, ~on_own & ~on_foreign]) & striking
    return state._replace(
        energy=energy,
        carrying=state.carrying & ~struck,  # knocked loose; the food is lost
        striking=striking,
        strikes=state.strikes + kinds.sum(axis=1).astype(jnp.int32),
    )
