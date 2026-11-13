import json
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from helpers import states_equal
from simlife.config import Config
from simlife.recorder import Recorder, load_survivors, resume_run
from simlife.runner import Runner
from simlife.state import init_world

cfg = Config(capacity_members=256, capacity_colonies=16, capacity_food=64, initial_colonies=4)


def pool_of(values):
    """A founder pool whose members have every weight equal to one of `values`."""
    template = init_world(jax.random.key(0), cfg).params
    n = len(values)
    params = {k: jnp.stack([jnp.full(v.shape[1:], x) for x in values]) for k, v in template.items()}
    return {"params": params, "lr": jnp.linspace(0.01, 0.03, n)}


def test_founders_come_from_the_pool():
    pool = pool_of([1.0, 2.0, 3.0])
    s = init_world(jax.random.key(5), cfg, founders=pool)
    for c in range(cfg.initial_colonies):
        w = s.params["w1"][s.alive & (s.colony == c)]
        assert bool((w == w[0, 0, 0]).all()) and float(w[0, 0, 0]) in (1.0, 2.0, 3.0)
    lrs = {round(float(v), 5) for v in np.asarray(s.lr[s.alive])}
    assert lrs <= {0.01, 0.02, 0.03}
    assert bool((s.birth_params["w1"] == s.params["w1"]).all())


def test_same_seed_and_pool_give_the_same_world():
    pool = pool_of([1.0, 2.0, 3.0])
    assert states_equal(init_world(jax.random.key(5), cfg, founders=pool),
                        init_world(jax.random.key(5), cfg, founders=pool))


def survivors_run(tmp_path, kill_first=0):
    rec = Recorder(tmp_path / "source", cfg, 1)
    s = init_world(jax.random.key(1), cfg)
    s = s._replace(alive=s.alive.at[:kill_first].set(False))
    rec.checkpoint(s)
    return rec.run_dir, s


def test_load_survivors_takes_living_members_only(tmp_path):
    run_dir, s = survivors_run(tmp_path, kill_first=12)  # colony 0 dead
    params, lr = load_survivors(run_dir, cfg)
    n = int(s.alive.sum())
    assert params["w1"].shape == (n,) + s.params["w1"].shape[1:] and lr.shape == (n,)
    assert np.array_equal(params["w1"], np.asarray(s.params["w1"])[np.asarray(s.alive)])


def test_load_survivors_refuses_a_different_network_shape(tmp_path):
    run_dir, _ = survivors_run(tmp_path)
    other = Config(capacity_members=256, capacity_colonies=16, capacity_food=64,
                   initial_colonies=4, n_hidden=6)
    with pytest.raises(ValueError, match="shape"):
        load_survivors(run_dir, other)


def test_load_survivors_with_none_alive_is_a_clear_error(tmp_path):
    run_dir, _ = survivors_run(tmp_path, kill_first=256)
    with pytest.raises(ValueError, match="no living members"):
        load_survivors(run_dir, cfg)


def test_seeded_run_keeps_its_founders_for_resume_and_fresh_starts(tmp_path):
    pool = pool_of([4.0, 5.0])
    rec = Recorder(tmp_path, cfg, 3, founders=pool, seeded_from="somewhere/run")
    settings = json.loads((rec.run_dir / "settings.json").read_text())
    assert settings["seeded_from"] == "somewhere/run"
    r = Runner(cfg, 3, rec)
    assert float(r.state.params["w1"][0, 0, 0]) in (4.0, 5.0)
    # every colony dies: the fresh start draws from the same founders
    r.state = r.state._replace(col_alive=jnp.zeros_like(r.state.col_alive),
                               alive=jnp.zeros_like(r.state.alive))
    r.advance(1)
    assert r.extinctions == 1 and float(r.state.params["w1"][r.state.alive][0, 0, 0]) in (4.0, 5.0)
    r.checkpoint()
    resumed = resume_run(rec.run_dir)
    assert set(np.unique(np.asarray(resumed.recorder.founders["params"]["w1"])).tolist()) == {4.0, 5.0}


def test_cli_seed_from(tmp_path):
    def run(*args):
        return subprocess.run([sys.executable, "-m", "simlife", "run", "--no-serve", *args],
                              capture_output=True, text=True, timeout=180)
    first = run("--ticks", "200", "--seed", "1", "--runs-dir", str(tmp_path / "a"))
    assert first.returncode == 0, first.stderr
    (source,) = list((tmp_path / "a").iterdir())
    second = run("--ticks", "100", "--seed", "2", "--runs-dir", str(tmp_path / "b"),
                 "--seed-from", str(source))
    assert second.returncode == 0, second.stderr
    (new,) = list((tmp_path / "b").iterdir())
    assert json.loads((new / "settings.json").read_text())["seeded_from"] == str(source)
    assert (new / "founders.npz").exists()
    missing = run("--ticks", "10", "--runs-dir", str(tmp_path / "c"), "--seed-from", str(tmp_path / "nope"))
    assert missing.returncode == 1 and "Traceback" not in missing.stderr
