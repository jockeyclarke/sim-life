"""Does a default world get going by itself? Slow: run with  pytest -m slow"""

import jax.numpy as jnp
import pytest

from simlife.config import Config
from simlife.metrics import compute_metrics
from simlife.recorder import Recorder
from simlife.runner import Runner


@pytest.mark.slow
def test_default_world_bootstraps(tmp_path):
    cfg = Config()
    took_hold = []
    for seed in range(5):
        runner = Runner(cfg, seed, Recorder(tmp_path, cfg, seed))
        runner.reseed_on_extinction = False  # no second chances
        runner.advance(20_000)
        m = compute_metrics(runner.state)
        s = runner.state
        has_split = int(jnp.where(s.col_alive, s.col_generation, 0).max()) > 0
        took_hold.append(
            not runner.extinct and m["delivered"] > 0
            and m["max_generation"] >= 5 and has_split
        )
        # scarcity, not array capacity, must be what limits the population
        assert m["members"] < 0.8 * cfg.capacity_members, (seed, m)
    assert sum(took_hold) >= 3, took_hold
