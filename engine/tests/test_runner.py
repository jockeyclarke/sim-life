import subprocess
import sys

import jax

from helpers import states_equal
from simlife.config import Config
from simlife.metrics import compute_metrics
from simlife.recorder import Recorder, read_history, resume_run
from simlife.runner import Runner

cfg = Config(
    capacity_members=256, capacity_colonies=16, capacity_food=64, initial_colonies=4
)
doomed = Config(
    capacity_members=256, capacity_colonies=16, capacity_food=64, initial_colonies=4,
    initial_energy=0.001,
)


def events_of(run_dir):
    return [e for rec in read_history(run_dir) for e in rec.get("events", [])]


def test_runner_records_on_interval_and_resume_matches_uninterrupted(tmp_path):
    a = Runner(cfg, 4, Recorder(tmp_path / "a", cfg, 4))
    a.advance(300)

    b = Runner(cfg, 4, Recorder(tmp_path / "b", cfg, 4))
    b.advance(150)
    b.checkpoint()
    r = resume_run(b.recorder.run_dir)
    b2 = Runner(r.cfg, r.seed, r.recorder, state=r.state, extinctions=r.extinctions)
    b2.advance(150)

    assert states_equal(a.state, b2.state)
    assert [x["tick"] for x in read_history(a.recorder.run_dir)] == [100, 200, 300]
    assert [x["tick"] for x in read_history(b.recorder.run_dir)] == [100, 200, 300]


def test_odd_sized_advances_still_record_on_the_interval(tmp_path):
    r = Runner(cfg, 1, Recorder(tmp_path, cfg, 1))
    for _ in range(7):
        r.advance(37)
    assert int(r.state.tick) == 259
    assert [x["tick"] for x in read_history(r.recorder.run_dir)] == [100, 200]


def test_reseed_on_extinction_logs_and_continues(tmp_path):
    r = Runner(doomed, 0, Recorder(tmp_path, doomed, 0))
    r.advance(50)
    r.advance(50)
    assert r.extinctions >= 1 and not r.extinct
    assert compute_metrics(r.state)["colonies"] > 0
    assert any(e.get("event") == "reseed" for e in events_of(r.recorder.run_dir))
    assert int(r.state.tick) == 100


def test_reseeded_worlds_differ_from_each_other(tmp_path):
    r = Runner(doomed, 0, Recorder(tmp_path, doomed, 0))
    r.advance(50)
    first = r.state.nest
    r.advance(50)
    assert r.extinctions == 2 and not bool((r.state.nest == first).all())


def test_extinction_with_reseed_off_idles(tmp_path):
    r = Runner(doomed, 0, Recorder(tmp_path, doomed, 0))
    r.reseed_on_extinction = False
    r.advance(50)
    t = int(r.state.tick)
    r.advance(50)
    r.advance(50)
    assert r.extinct and int(r.state.tick) == t and r.status()["extinct"]
    assert [e["event"] for e in events_of(r.recorder.run_dir)] == ["extinct"]


def test_turning_reseed_back_on_revives_an_extinct_run(tmp_path):
    r = Runner(doomed, 0, Recorder(tmp_path, doomed, 0))
    r.reseed_on_extinction = False
    r.advance(50)
    r.reseed_on_extinction = True
    r.advance(10)
    assert not r.extinct and r.extinctions >= 1 and int(r.state.tick) == 60


def test_cli_headless_run(tmp_path):
    out = subprocess.run(
        [sys.executable, "-m", "simlife", "run", "--no-serve", "--ticks", "200",
         "--seed", "1", "--runs-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    (run_dir,) = list(tmp_path.iterdir())
    assert (run_dir / "settings.json").exists()
    assert len(read_history(run_dir)) == 2
    assert len(list((run_dir / "checkpoints").iterdir())) == 1


def test_cli_no_serve_without_ticks_is_a_usage_error(tmp_path):
    out = subprocess.run(
        [sys.executable, "-m", "simlife", "run", "--no-serve", "--runs-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 2 and "--ticks" in out.stderr
