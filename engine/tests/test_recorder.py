import json

import jax
import pytest

from helpers import states_equal
from simlife.config import Config
from simlife.metrics import compute_metrics
from simlife.recorder import (
    Recorder, list_runs, load_state, read_history, resume_run, save_state,
)
from simlife.state import init_world
from simlife.step import run_ticks

cfg = Config(
    capacity_members=256, capacity_colonies=16, capacity_food=64, initial_colonies=4
)


def test_metrics_are_plain_numbers():
    m = compute_metrics(init_world(jax.random.key(0), cfg))
    stage_1_and_2 = {k: v for k, v in m.items() if not k.startswith(("famil", "strikes", "colon"))
                     or k == "colonies"}
    assert stage_1_and_2 == {
        "tick": 0, "colonies": 4, "members": 48,
        "energy": pytest.approx(4 * cfg.initial_energy),
        "delivered": 0, "max_generation": 0,
        "pred_error": 0.0, "learning_rate": pytest.approx(cfg.initial_lr),
    }
    json.dumps(m)


def test_state_roundtrips_through_disk_and_run_continues_identically(tmp_path):
    s = run_ticks(init_world(jax.random.key(3), cfg), cfg, 100)
    save_state(tmp_path / "s.npz", s)
    loaded = load_state(tmp_path / "s.npz", cfg)
    assert states_equal(loaded, s)
    assert states_equal(run_ticks(loaded, cfg, 100), run_ticks(s, cfg, 100))


def test_save_leaves_no_temporary_file(tmp_path):
    save_state(tmp_path / "s.npz", init_world(jax.random.key(0), cfg))
    assert [p.name for p in tmp_path.iterdir()] == ["s.npz"]


def test_recorder_writes_settings_and_history(tmp_path):
    rec = Recorder(tmp_path, cfg, seed=5)
    settings = json.loads((rec.run_dir / "settings.json").read_text())
    assert settings["seed"] == 5 and settings["run_id"] == rec.run_id
    assert Config.from_dict(settings["config"]) == cfg

    base = {"colonies": 4, "energy": 1.0, "max_generation": 0}
    rec.record({**base, "tick": 100, "members": 10, "delivered": 0})
    rec.record({**base, "tick": 200, "members": 10, "delivered": 5}, [{"event": "x"}])
    history = read_history(rec.run_dir)
    assert [r["tick"] for r in history] == [100, 200]
    assert history[0]["delivery_rate"] == 0
    assert history[1]["delivery_rate"] == pytest.approx(5 / (10 * 100))
    assert history[1]["events"] == [{"event": "x"}] and "events" not in history[0]


def test_delivery_rate_is_zero_with_no_members(tmp_path):
    rec = Recorder(tmp_path, cfg, seed=0)
    base = {"colonies": 0, "energy": 0.0, "max_generation": 0, "members": 0}
    rec.record({**base, "tick": 100, "delivered": 0})
    assert rec.record({**base, "tick": 200, "delivered": 3})["delivery_rate"] == 0


def test_two_recorders_in_the_same_second_get_distinct_runs(tmp_path):
    a, b = Recorder(tmp_path, cfg, 1), Recorder(tmp_path, cfg, 1)
    assert a.run_dir != b.run_dir
    assert set(list_runs(tmp_path)) == {a.run_id, b.run_id}


def test_read_history_skips_a_half_written_last_line(tmp_path):
    rec = Recorder(tmp_path, cfg, seed=0)
    rec.record({"tick": 100, "colonies": 1, "members": 1, "energy": 1.0,
                "delivered": 0, "max_generation": 0})
    with open(rec.run_dir / "history.jsonl", "a") as f:
        f.write('{"tick": 200, "colo')
    assert [r["tick"] for r in read_history(rec.run_dir)] == [100]


def test_resume_ignores_partial_checkpoint(tmp_path):
    rec = Recorder(tmp_path, cfg, seed=2)
    s = run_ticks(init_world(jax.random.key(2), cfg), cfg, 40)
    rec.checkpoint(s)
    (rec.run_dir / "checkpoints" / "state-000000999999.npz.tmp").write_bytes(b"junk")
    resumed = resume_run(rec.run_dir)
    assert states_equal(resumed.state, s) and resumed.seed == 2 and resumed.cfg == cfg


def test_resume_with_mismatched_shapes_raises_clear_error(tmp_path):
    rec = Recorder(tmp_path, cfg, seed=2)
    rec.checkpoint(init_world(jax.random.key(2), cfg))
    path = rec.run_dir / "settings.json"
    settings = json.loads(path.read_text())
    settings["config"]["capacity_members"] = 512
    path.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="capacity"):
        resume_run(rec.run_dir)


def test_resume_without_any_checkpoint_raises_clear_error(tmp_path):
    rec = Recorder(tmp_path, cfg, seed=2)
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        resume_run(rec.run_dir)


# --- stage 2 -------------------------------------------------------------------

import subprocess  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402


def strip_fields(path, *names):
    """Rewrite a checkpoint without some arrays, as an older version would have saved it."""
    with np.load(path) as data:
        kept = {k: data[k] for k in data.files if k not in names}
    with open(path, "wb") as f:
        np.savez(f, **kept)


def test_new_fields_roundtrip_through_disk(tmp_path):
    s = run_ticks(init_world(jax.random.key(3), cfg), cfg, 30)
    save_state(tmp_path / "s.npz", s)
    loaded = load_state(tmp_path / "s.npz", cfg)
    assert states_equal(loaded, s)
    assert loaded.learning_on.dtype == s.learning_on.dtype
    assert loaded.birth_params.keys() == s.birth_params.keys()


def test_resuming_a_checkpoint_from_an_older_version_says_so(tmp_path):
    path = tmp_path / "s.npz"
    save_state(path, init_world(jax.random.key(0), cfg))
    strip_fields(path, "lr", "params.w3", "params.b3")
    with pytest.raises(ValueError, match="older version"):
        load_state(path, cfg)


def test_cli_resume_of_older_run_is_a_one_line_error(tmp_path):
    default = Config()
    rec = Recorder(tmp_path, default, seed=0)
    path = rec.checkpoint(init_world(jax.random.key(0), default))
    strip_fields(path, "lr", "birth_params.w1")
    out = subprocess.run(
        [sys.executable, "-m", "simlife", "run", "--no-serve", "--ticks", "10",
         "--resume", str(rec.run_dir)],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 1
    assert "older version" in out.stderr and "Traceback" not in out.stderr
    assert len(out.stderr.strip().splitlines()) == 1


def test_metrics_include_prediction_error_and_learning_rate():
    s = init_world(jax.random.key(0), cfg)
    m = compute_metrics(s)
    assert m["pred_error"] == 0.0 and m["learning_rate"] == pytest.approx(cfg.initial_lr)
    m = compute_metrics(run_ticks(s, cfg, 50))
    assert m["pred_error"] > 0 and 0 < m["learning_rate"] <= cfg.max_lr
    import json as _json
    _json.dumps(m)


def test_metrics_of_an_extinct_world_do_not_divide_by_zero():
    import jax.numpy as jnp
    s = init_world(jax.random.key(0), cfg)
    s = s._replace(alive=jnp.zeros_like(s.alive), col_alive=jnp.zeros_like(s.col_alive))
    m = compute_metrics(s)
    assert m["pred_error"] == 0.0 and m["learning_rate"] == 0.0 and m["members"] == 0


# --- stage 3 -------------------------------------------------------------------


def test_family_metrics_are_json_ready_and_sized_by_family():
    c = Config()
    s = run_ticks(init_world(jax.random.key(0), c), c, 50)
    m = compute_metrics(s)
    import json as _json
    _json.dumps(m)
    assert m["families"] == 6
    for key in ("family_members", "family_colonies", "family_territory",
                "family_aggression_mean", "family_aggression_sd"):
        assert len(m[key]) == 6, key
    assert sum(m["family_members"]) == m["members"] and sum(m["family_colonies"]) == m["colonies"]
    assert 0 < sum(m["family_territory"]) <= 1.0 + 1e-6
    assert all(0 <= a <= 1 for a in m["family_aggression_mean"])
    assert all(sd >= 0 for sd in m["family_aggression_sd"])
    for key in ("strikes_own_total", "strikes_foreign_total", "strikes_border_total",
                "colonies_died_home", "colonies_died_foreign"):
        assert isinstance(m[key], int), key
    assert m["colony_life_home"] >= 0 and m["colony_life_foreign"] >= 0


def test_dead_family_reads_zero():
    import jax.numpy as jnp
    c = Config()
    s = init_world(jax.random.key(0), c)
    gone = s.col_alive & (s.col_family == 2)
    s = s._replace(col_alive=s.col_alive & ~gone, alive=s.alive & ~gone[s.colony])
    m = compute_metrics(s)
    assert m["families"] == 5
    assert m["family_members"][2] == 0 and m["family_colonies"][2] == 0
    assert m["family_aggression_mean"][2] == 0.0 and m["family_aggression_sd"][2] == 0.0


def test_colony_lifespans_are_means_by_founding_ground():
    import jax.numpy as jnp
    s = init_world(jax.random.key(0), cfg)._replace(
        colonies_died=jnp.array([2, 1], jnp.int32), colony_life_total=jnp.array([600, 50], jnp.int32)
    )
    m = compute_metrics(s)
    assert m["colonies_died_home"] == 2 and m["colony_life_home"] == 300.0
    assert m["colonies_died_foreign"] == 1 and m["colony_life_foreign"] == 50.0


def test_strikes_are_recorded_per_interval_and_survive_resume(tmp_path):
    rec = Recorder(tmp_path, cfg, seed=0)
    base = {"colonies": 1, "members": 1, "energy": 1.0, "delivered": 0, "max_generation": 0}
    first = rec.record({**base, "tick": 100, "strikes_own_total": 3, "strikes_foreign_total": 1, "strikes_border_total": 0})
    assert (first["strikes_own"], first["strikes_foreign"], first["strikes_border"]) == (3, 1, 0)
    again = Recorder(tmp_path, cfg, 0, _existing=rec.run_dir)  # as on resume
    second = again.record({**base, "tick": 200, "strikes_own_total": 10, "strikes_foreign_total": 1, "strikes_border_total": 4})
    assert (second["strikes_own"], second["strikes_foreign"], second["strikes_border"]) == (7, 0, 4)
