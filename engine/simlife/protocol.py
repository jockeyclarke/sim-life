"""The wire format between engine and viewer.

Snapshot frame (binary, engine -> viewer):

    4 bytes   little-endian unsigned length L of the header
    L bytes   UTF-8 JSON header
    rest      the arrays named in header["arrays"], back to back, in order,
              each as little-endian float32

Everything else is a JSON text message; see parse_control and server.py.
"""

from __future__ import annotations

import json
import struct

import numpy as np

from .state import WorldState
from .territory import owner_grid

MAX_TICKS_PER_FRAME = 1000


def encode_snapshot(
    state: WorldState, metrics: dict, status: dict, inspect: dict | None = None
) -> bytes:
    """`inspect` is the selected member's internals (see inspect.py), or None."""
    alive = np.asarray(state.alive)
    col_alive = np.asarray(state.col_alive)
    food_alive = np.asarray(state.food_alive)
    colony = np.asarray(state.colony)
    members = np.bincount(colony[alive], minlength=col_alive.shape[0])
    owner, strength = owner_grid(state, _Grid(state.territory.shape[1]))

    arrays = [
        ("member_pos", np.asarray(state.pos)[alive]),
        ("member_heading", np.asarray(state.heading)[alive]),
        ("member_colony", colony[alive]),
        ("member_carrying", np.asarray(state.carrying)[alive]),
        ("member_slot", np.flatnonzero(alive)),
        ("member_error", np.asarray(state.pred_error_avg)[alive]),
        ("member_family", np.asarray(state.col_family)[colony][alive]),
        ("member_aggression", np.asarray(state.aggression)[alive]),
        ("colony_slot", np.flatnonzero(col_alive)),
        ("colony_nest", np.asarray(state.nest)[col_alive]),
        ("colony_energy", np.asarray(state.energy)[col_alive]),
        ("colony_members", members[col_alive]),
        ("colony_family", np.asarray(state.col_family)[col_alive]),
        ("food_pos", np.asarray(state.food_pos)[food_alive]),
        ("strike_pos", np.asarray(state.pos)[np.asarray(state.striking) & alive]),
        ("territory_owner", np.asarray(owner)),  # family per cell, or -1 where unmarked
        ("territory_strength", np.asarray(strength)),  # that family's share of the scent
    ]
    header = {
        "type": "snapshot",
        "tick": int(state.tick),
        "metrics": metrics,
        "status": status,
        "inspect": inspect,
        "n_families": int(state.territory.shape[0]),
        "arrays": [{"name": name, "shape": list(a.shape)} for name, a in arrays],
    }
    head = json.dumps(header).encode()
    body = b"".join(a.astype("<f4").tobytes() for _, a in arrays)
    return struct.pack("<I", len(head)) + head + body


class _Grid:
    def __init__(self, g: int):
        self.territory_grid = g


def decode_snapshot(data: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    (length,) = struct.unpack_from("<I", data, 0)
    header = json.loads(data[4 : 4 + length])
    offset = 4 + length
    arrays = {}
    for spec in header["arrays"]:
        count = int(np.prod(spec["shape"]))
        flat = np.frombuffer(data, dtype="<f4", count=count, offset=offset)
        arrays[spec["name"]] = flat.reshape(spec["shape"])
        offset += 4 * count
    return header, arrays


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_control(text: str) -> dict | None:
    """Validate a message from a viewer. Returns None for anything not understood.

    The result contains only the known fields, so nothing else a viewer sends
    can reach the engine.
    """
    try:
        message = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(message, dict):
        return None
    kind = message.get("type")
    if not isinstance(kind, str):
        return None

    if kind == "inheritance":
        value = message.get("value")
        return {"type": kind, "value": value} if value in ("learned", "birth") else None
    if kind == "select":
        if "slot" not in message:
            return None
        slot = message["slot"]
        if slot is None or (_is_int(slot) and slot >= 0):
            return {"type": kind, "slot": slot}
        return None
    if kind in ("pause", "reseed_on_extinction", "learning"):
        value = message.get("value")
        return {"type": kind, "value": value} if isinstance(value, bool) else None
    if kind == "speed":
        n = message.get("ticks_per_frame")
        if not _is_int(n):
            return None
        return {"type": kind, "ticks_per_frame": max(1, min(MAX_TICKS_PER_FRAME, n))}
    if kind == "list_runs":
        return {"type": kind}
    if kind == "get_history":
        run_id = message.get("run_id")
        return {"type": kind, "run_id": run_id} if isinstance(run_id, str) else None
    return None
