"""Write a real snapshot frame for the viewer's decoder tests.

Run from the repository root after changing the snapshot format or the
default starting population:

    .venv/bin/python engine/tools/write_fixture.py
"""

from pathlib import Path

import jax

from simlife.config import Config
from simlife.inspect import inspect_entity
from simlife.metrics import compute_metrics
from simlife.protocol import encode_snapshot
from simlife.state import init_world

OUT = Path(__file__).resolve().parents[2] / "viewer" / "src" / "__fixtures__" / "snapshot.bin"

state = init_world(jax.random.key(0), Config())
status = {
    "run_id": "fixture",
    "extinct": False,
    "extinctions": 0,
    "reseed_on_extinction": True,
    "paused": False,
    "ticks_per_frame": 1,
    "learning": True,
    "inheritance": "learned",
}
OUT.parent.mkdir(parents=True, exist_ok=True)
cfg = Config()
inspect = {**inspect_entity(state, cfg, 0), "selection": 1}  # a real inspection payload for the viewer's tests
OUT.write_bytes(encode_snapshot(state, compute_metrics(state), status, inspect))
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
