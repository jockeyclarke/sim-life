"""Does the evolve-then-seed workflow produce a contest? Slow: run with  pytest -m slow

Random-start worlds rarely reach the contest between families (see the stage 3
ledger), so stage 3 worlds are seeded from an evolved source population.
"""

import pytest

from simlife.config import Config
from simlife.metrics import compute_metrics
from simlife.recorder import Recorder, load_survivors
from simlife.runner import Runner


@pytest.mark.slow
def test_seeded_worlds_reach_the_contest(tmp_path):
    cfg = Config()
    nursery = Runner(cfg, 11, Recorder(tmp_path / "nursery", cfg, 11))
    nursery.advance(cfg.grace_ticks)  # a grace period's worth of evolution
    nursery.checkpoint()
    params, lr = load_survivors(nursery.recorder.run_dir, cfg)
    founders = {"params": params, "lr": lr}

    alive, contested = 0, 0
    for seed in range(5):
        recorder = Recorder(tmp_path / "worlds", cfg, seed, founders=founders)
        runner = Runner(cfg, seed, recorder)
        runner.reseed_on_extinction = False  # no second chances
        runner.advance(cfg.grace_ticks + 10_000)  # 10,000 ticks of competition
        m = compute_metrics(runner.state)
        alive += not runner.extinct
        contested += m["families"] >= 2
        # scarcity, not array capacity, must be what limits the population
        assert m["members"] < 0.8 * cfg.capacity_members, (seed, m)
    assert alive >= 4 and contested >= 2, (alive, contested)
