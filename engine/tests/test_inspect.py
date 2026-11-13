import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simlife.config import Config
from simlife.inspect import inspect_entity
from simlife.recorder import Recorder
from simlife.runner import Runner
from simlife.server import SimServer
from simlife.state import init_world
from simlife.step import run_ticks

cfg = Config(
    capacity_members=256, capacity_colonies=16, capacity_food=64, initial_colonies=4
)


def test_inspect_entity_is_json_ready_and_consistent():
    s = run_ticks(init_world(jax.random.key(7), cfg), cfg, 30)
    slot = int(jnp.argmax(s.alive & s.has_prev))
    d = inspect_entity(s, cfg, slot)
    json.dumps(d)
    assert d["slot"] == slot and d["alive"] is True
    assert len(d["inputs"]) == 10 and len(d["hidden"]) == 8 and len(d["actions"]) == 5
    assert len(d["predicted"]) == len(d["actual"]) == len(d["next_prediction"]) == 6
    w = d["weights"]
    assert (len(w["w1"]), len(w["w1"][0])) == (8, 10)
    assert (len(w["w2"]), len(w["w2"][0])) == (5, 8)
    assert (len(w["w3"]), len(w["w3"][0])) == (6, 8)
    assert len(w["b1"]) == 8 and len(w["b2"]) == 5 and len(w["b3"]) == 6
    scored = np.mean((np.array(d["predicted"]) - np.array(d["actual"])) ** 2)
    assert d["error"] == pytest.approx(scored, rel=1e-4)
    assert d["actual"] == pytest.approx(d["inputs"][:6])
    assert d["learning_rate"] == pytest.approx(float(s.lr[slot]))
    assert d["age"] == int(s.age[slot]) and d["colony"] == int(s.colony[slot])
    assert all(-1 <= v <= 1 for v in d["hidden"]) and 0 <= d["actions"][1] <= 1


# --- the server's selection rules, without running its loop -------------------


@pytest.fixture
def srv(tmp_path):
    runner = Runner(cfg, 0, Recorder(tmp_path, cfg, 0))
    runner.advance(5)
    return SimServer(runner, tmp_path)


def test_selecting_a_living_member_gives_its_inspection(srv):
    srv.select(3)
    assert srv.inspection()["slot"] == 3 and srv.inspection()["alive"] is True
    srv.select(None)
    assert srv.inspection() is None


def test_selected_member_death_is_reported_not_replaced(srv):
    r = srv.runner
    srv.select(3)
    r.state = r.state._replace(alive=r.state.alive.at[3].set(False))
    assert srv.inspection() == {"slot": 3, "alive": False}
    # a newborn takes the slot: still reported as died, even once it is older
    # than the member we were watching was
    r.state = r.state._replace(
        alive=r.state.alive.at[3].set(True), age=r.state.age.at[3].set(0)
    )
    assert srv.inspection() == {"slot": 3, "alive": False}
    r.state = r.state._replace(age=r.state.age.at[3].set(999_999))
    assert srv.inspection() == {"slot": 3, "alive": False}


def test_slot_reused_between_two_looks_is_caught(srv):
    r = srv.runner
    srv.select(3)
    srv.inspection()
    # between snapshots the member died and a newborn took its slot and aged
    r.state = r.state._replace(
        tick=r.state.tick + 500, age=r.state.age.at[3].set(400)
    )
    assert srv.inspection() == {"slot": 3, "alive": False}


def test_selected_member_keeps_being_inspected_as_it_ages(srv):
    srv.select(3)
    srv.runner.advance(50)
    if bool(srv.runner.state.alive[3]):
        assert srv.inspection()["alive"] is True


def test_select_out_of_range_or_dead_slot_clears_selection(srv):
    srv.select(3)
    srv.select(10**9)
    assert srv.inspection() is None
    dead = int(jnp.argmin(srv.runner.state.alive))
    srv.select(3)
    srv.select(dead)
    assert srv.inspection() is None


def test_select_in_an_extinct_world_clears_selection(srv):
    r = srv.runner
    r.state = r.state._replace(
        alive=jnp.zeros_like(r.state.alive), col_alive=jnp.zeros_like(r.state.col_alive)
    )
    srv.select(3)
    assert srv.inspection() is None
