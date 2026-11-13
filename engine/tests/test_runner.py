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
    initial_energy=0.001, food_energy=0.0,  # dies on tick 1, and food cannot save it
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
    assert r.reseed_on_extinction is True and r.extinct is False
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
    assert r.extinctions >= 2 and not bool((r.state.nest == first).all())


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
    t = int(r.state.tick)  # stopped on the tick of extinction
    r.reseed_on_extinction = True
    r.advance(10)
    assert not r.extinct and r.extinctions >= 1 and int(r.state.tick) == t + 10


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


# --- serving from the command line -------------------------------------------

import signal  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402


def test_cli_port_in_use_is_a_clear_error_and_leaves_no_run(tmp_path):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        out = subprocess.run(
            [sys.executable, "-m", "simlife", "run", "--port", str(port),
             "--runs-dir", str(tmp_path)],
            capture_output=True, text=True, timeout=60,
        )
    assert out.returncode == 1
    assert f"port {port}" in out.stderr and "Traceback" not in out.stderr
    assert list(tmp_path.iterdir()) == []


def test_cli_terminate_signal_saves_the_run(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    proc = subprocess.Popen(
        [sys.executable, "-m", "simlife", "run", "--port", str(port),
         "--runs-dir", str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert "serving" in proc.stdout.readline()
        time.sleep(3.0)  # let it compile and run some ticks
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=30)
    finally:
        proc.kill()
    assert proc.returncode == 0, err
    assert "run saved" in out and "Traceback" not in err
    (run_dir,) = list(tmp_path.iterdir())
    assert len(list((run_dir / "checkpoints").glob("state-*.npz"))) >= 1


# --- review fixes: extinction, resume, serving -------------------------------

import json as _json  # noqa: E402

from simlife.__main__ import build_parser, make_runner  # noqa: E402

fragile = Config(
    capacity_members=256, capacity_colonies=16, capacity_food=64, initial_colonies=4,
    initial_energy=0.5,
)


def reseed_ticks(run_dir):
    return [e["tick"] for e in events_of(run_dir) if e["event"] == "reseed"]


def test_reseed_ticks_do_not_depend_on_chunk_size(tmp_path):
    a = Runner(fragile, 0, Recorder(tmp_path / "a", fragile, 0))
    a.advance(500)
    a.advance(500)
    b = Runner(fragile, 0, Recorder(tmp_path / "b", fragile, 0))
    for _ in range(142):
        b.advance(7)
    b.advance(6)
    assert a.extinctions == b.extinctions >= 1  # at least one reseed crossed
    assert reseed_ticks(a.recorder.run_dir) == reseed_ticks(b.recorder.run_dir)
    assert states_equal(a.state, b.state)


def test_reseed_event_survives_an_exit_before_the_next_record(tmp_path):
    r = Runner(doomed, 0, Recorder(tmp_path, doomed, 0))
    while r.extinctions == 0:
        r.advance(7)
    assert int(r.state.tick) < 100
    r.checkpoint()  # e.g. Ctrl-C right after a reseed
    assert len(reseed_ticks(r.recorder.run_dir)) == r.extinctions >= 1

    r2 = Runner.resume(r.recorder.run_dir)
    r2.advance(100)
    assert len(reseed_ticks(r2.recorder.run_dir)) == r2.extinctions


def test_resume_keeps_the_reseed_switch_and_the_extinct_state(tmp_path):
    r = Runner(doomed, 0, Recorder(tmp_path, doomed, 0))
    r.reseed_on_extinction = False
    r.advance(50)
    assert r.extinct
    t = int(r.state.tick)
    r.checkpoint()

    r2 = Runner.resume(r.recorder.run_dir)
    assert r2.reseed_on_extinction is False and r2.extinct
    r2.advance(50)
    assert int(r2.state.tick) == t
    assert [e["event"] for e in events_of(r2.recorder.run_dir)] == ["extinct"]


def test_cli_no_reseed_flag_turns_the_switch_off(tmp_path):
    args = build_parser().parse_args(["run", "--no-reseed", "--runs-dir", str(tmp_path)])
    assert make_runner(args).reseed_on_extinction is False
    args = build_parser().parse_args(["run", "--runs-dir", str(tmp_path)])
    assert make_runner(args).reseed_on_extinction is True


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_cli_resume_from_another_directory_serves_the_runs_history(tmp_path):
    from websockets.sync.client import connect

    data = tmp_path / "data"
    subprocess.run(
        [sys.executable, "-m", "simlife", "run", "--no-serve", "--ticks", "300",
         "--seed", "1", "--runs-dir", str(data)],
        check=True, capture_output=True, timeout=120,
    )
    (run_dir,) = list(data.iterdir())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "simlife", "run", "--resume", str(run_dir), "--port", str(port)],
        cwd=elsewhere, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert "serving" in proc.stdout.readline()
        with connect(f"ws://127.0.0.1:{port}") as ws:
            runs = _json.loads(ws.recv(timeout=30))
            history = _json.loads(ws.recv(timeout=30))
        assert run_dir.name in runs["runs"]
        assert [r["tick"] for r in history["records"]][:3] == [100, 200, 300]
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.communicate(timeout=30)


def test_cli_headless_run_saves_on_terminate(tmp_path):
    proc = subprocess.Popen(
        [sys.executable, "-m", "simlife", "run", "--no-serve", "--ticks", "100000000",
         "--runs-dir", str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert "headless" in proc.stdout.readline()
        time.sleep(4.0)
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=60)
    finally:
        proc.kill()
    assert proc.returncode == 0, err
    assert "run saved" in out and "Traceback" not in err
    (run_dir,) = list(tmp_path.iterdir())
    (checkpoint,) = list((run_dir / "checkpoints").glob("state-*.npz"))
    assert int(checkpoint.stem.split("-")[1]) > 0


# --- stage 2 -------------------------------------------------------------------

import jax.numpy as jnp  # noqa: E402


def test_fresh_world_after_extinction_keeps_the_switches(tmp_path):
    r = Runner(doomed, 0, Recorder(tmp_path, doomed, 0))
    r.state = r.state._replace(learning_on=jnp.bool_(False), inherit_learned=jnp.bool_(False))
    r.advance(20)
    assert r.extinctions >= 1
    assert not bool(r.state.learning_on) and not bool(r.state.inherit_learned)

import dataclasses  # noqa: E402

import pytest  # noqa: E402

still = dataclasses.replace(  # no births or deaths: weight changes can only be learning
    cfg, birth_cost=1e9, initial_energy=1e6, lifespan_mean=1_000_000, lifespan_jitter=0
)


def switch_events(run_dir):
    return [e for e in events_of(run_dir) if e["event"] == "switch"]


def test_set_switch_records_an_event_once(tmp_path):
    r = Runner(cfg, 0, Recorder(tmp_path, cfg, 0))
    assert r.status()["learning"] is True and r.status()["inheritance"] == "learned"
    r.advance(30)
    line = r.set_switch("learning", False)
    event = {"event": "switch", "name": "learning", "value": False, "tick": 30}
    assert line == {"tick": 30, "events": [event]}
    assert r.set_switch("learning", False) is None
    assert r.status()["learning"] is False and r.status()["inheritance"] == "learned"
    r.set_switch("inherit_learned", False)
    assert r.status()["inheritance"] == "birth"
    assert [e["name"] for e in switch_events(r.recorder.run_dir)] == ["learning", "inherit_learned"]


def test_switches_survive_checkpoint_and_resume(tmp_path):
    r = Runner(cfg, 0, Recorder(tmp_path, cfg, 0))
    r.set_switch("learning", False)
    r.set_switch("inherit_learned", False)
    r.advance(40)
    r.checkpoint()
    r2 = Runner.resume(r.recorder.run_dir)
    assert r2.status()["learning"] is False and r2.status()["inheritance"] == "birth"
    assert len(switch_events(r2.recorder.run_dir)) == 2


def test_switch_takes_effect_on_the_next_tick(tmp_path):
    r = Runner(still, 0, Recorder(tmp_path, still, 0))
    r.advance(10)
    learned = r.state.params["w1"]
    r.set_switch("learning", False)
    r.advance(20)
    assert bool((r.state.params["w1"] == learned).all())
    r.set_switch("learning", True)
    r.advance(1)
    assert bool((r.state.params["w1"] != learned).any())


def test_set_switch_rejects_unknown_names(tmp_path):
    r = Runner(cfg, 0, Recorder(tmp_path, cfg, 0))
    with pytest.raises(ValueError, match="unknown switch"):
        r.set_switch("tick", False)
    with pytest.raises(ValueError, match="unknown switch"):
        r.set_switch("alive", False)


def test_cli_switch_flags(tmp_path):
    parse = build_parser().parse_args
    r = make_runner(parse(["run", "--no-learning", "--inherit", "birth", "--runs-dir", str(tmp_path)]))
    assert r.status()["learning"] is False and r.status()["inheritance"] == "birth"
    assert len(switch_events(r.recorder.run_dir)) == 2
    r.advance(10)
    r.checkpoint()

    kept = make_runner(parse(["run", "--resume", str(r.recorder.run_dir)]))
    assert kept.status()["learning"] is False and kept.status()["inheritance"] == "birth"
    changed = make_runner(parse(["run", "--resume", str(r.recorder.run_dir), "--inherit", "learned"]))
    assert changed.status()["learning"] is False and changed.status()["inheritance"] == "learned"

    plain = make_runner(parse(["run", "--runs-dir", str(tmp_path)]))
    assert plain.status()["learning"] is True and plain.status()["inheritance"] == "learned"
    assert switch_events(plain.recorder.run_dir) == []


# --- stage 3 -------------------------------------------------------------------

from simlife.config import Config as _Config  # noqa: E402

families_cfg = _Config(capacity_members=256, capacity_colonies=24, capacity_food=64)


def takeovers(run_dir):
    return [e for e in events_of(run_dir) if e["event"] == "takeover"]


def test_takeover_event_written_once(tmp_path):
    r = Runner(families_cfg, 0, Recorder(tmp_path, families_cfg, 0))
    s = r.state
    others = s.col_alive & (s.col_family != 2)
    r.state = s._replace(col_alive=s.col_alive & ~others, alive=s.alive & ~others[s.colony])
    r.advance(10)
    r.advance(10)
    events = takeovers(r.recorder.run_dir)
    assert len(events) == 1 and events[0]["family"] == 2


def test_fresh_start_keeps_cumulative_counters_and_dates_founders(tmp_path):
    r = Runner(doomed, 0, Recorder(tmp_path, doomed, 0))
    r.state = r.state._replace(strikes=jnp.array([5, 6, 7], jnp.int32),
                               colonies_died=jnp.array([1, 2], jnp.int32),
                               colony_life_total=jnp.array([10, 20], jnp.int32))
    r.advance(5)
    assert r.extinctions >= 1
    s = r.state
    assert s.strikes.tolist()[:3] >= [5, 6, 7]
    assert int(s.colonies_died.sum()) >= 3 and int(s.colony_life_total.sum()) >= 30
    assert bool((s.col_born[s.col_alive] > 0).all())


def families_formed(run_dir):
    return [e for e in events_of(run_dir) if e["event"] == "families_formed"]


def test_families_formed_event_once_per_world(tmp_path):
    c = _Config(capacity_members=256, capacity_colonies=24, capacity_food=64, grace_ticks=150)
    r = Runner(c, 0, Recorder(tmp_path, c, 0))
    r.advance(120)
    assert families_formed(r.recorder.run_dir) == []
    r.advance(100)
    assert [e["tick"] for e in families_formed(r.recorder.run_dir)] == [150]


def test_fresh_start_restarts_the_grace_period(tmp_path):
    r = Runner(doomed, 0, Recorder(tmp_path, doomed, 0))
    r.advance(5)
    assert r.extinctions >= 1 and int(r.state.founded) > 0
    assert int(r.state.founded) <= int(r.state.tick)
