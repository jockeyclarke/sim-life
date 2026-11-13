import jax
import jax.numpy as jnp
import numpy as np

from simlife.config import Config
from simlife.metrics import compute_metrics
from simlife.protocol import decode_snapshot, encode_snapshot, parse_control
from simlife.state import init_world


def test_snapshot_roundtrip_contains_only_the_living():
    s = init_world(jax.random.key(0), Config())
    h, a = decode_snapshot(encode_snapshot(s, compute_metrics(s), {"run_id": "x"}))
    assert h["type"] == "snapshot" and h["tick"] == 0 and h["status"] == {"run_id": "x"}
    assert h["metrics"]["members"] == 192
    assert a["member_pos"].shape == (192, 2) and a["colony_nest"].shape == (16, 2)
    assert a["food_pos"].shape == (512, 2) and a["member_slot"].shape == (192,)
    assert a["member_pos"].dtype == np.float32
    assert set(a["member_carrying"]) <= {0.0, 1.0}
    assert list(a["colony_members"]) == [12.0] * 16
    assert list(a) == [
        "member_pos", "member_heading", "member_colony", "member_carrying",
        "member_slot", "colony_slot", "colony_nest", "colony_energy",
        "colony_members", "food_pos",
    ]


def test_snapshot_member_rows_line_up_with_state_slots():
    s = init_world(jax.random.key(0), Config())
    s = s._replace(alive=s.alive.at[0].set(False))
    _, a = decode_snapshot(encode_snapshot(s, compute_metrics(s), {}))
    slots = a["member_slot"].astype(int)
    assert slots[0] == 1 and len(slots) == 191
    assert np.array_equal(a["member_pos"], np.asarray(s.pos)[slots])


def test_snapshot_of_extinct_world_has_empty_arrays():
    s = init_world(jax.random.key(0), Config())
    s = s._replace(
        alive=jnp.zeros_like(s.alive), col_alive=jnp.zeros_like(s.col_alive),
        food_alive=jnp.zeros_like(s.food_alive),
    )
    _, a = decode_snapshot(encode_snapshot(s, compute_metrics(s), {}))
    assert a["member_pos"].shape == (0, 2) and a["colony_energy"].shape == (0,)
    assert a["food_pos"].shape == (0, 2)


def test_parse_control_accepts_valid_and_rejects_junk():
    assert parse_control('{"type":"pause","value":true}') == {"type": "pause", "value": True}
    assert parse_control('{"type":"list_runs"}') == {"type": "list_runs"}
    assert parse_control('{"type":"get_history","run_id":"abc"}') == {
        "type": "get_history", "run_id": "abc"}
    for bad in [
        "{", "[]", "null", "7", '{"type":"nope"}', '{"value":true}',
        '{"type":"pause","value":"yes"}', '{"type":"pause"}',
        '{"type":"speed","ticks_per_frame":"x"}', '{"type":"speed","ticks_per_frame":true}',
        '{"type":"speed","ticks_per_frame":2.5}', '{"type":"get_history","run_id":5}',
        '{"type":["pause"],"value":true}',
    ]:
        assert parse_control(bad) is None, bad


def test_parse_control_clamps_speed_and_drops_extra_fields():
    assert parse_control('{"type":"speed","ticks_per_frame":99999}')["ticks_per_frame"] == 1000
    assert parse_control('{"type":"speed","ticks_per_frame":-4}')["ticks_per_frame"] == 1
    assert parse_control('{"type":"pause","value":false,"x":1}') == {
        "type": "pause", "value": False}
