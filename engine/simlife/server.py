"""WebSocket server and pacing loop.

Engine -> viewer:  binary snapshot frames (see protocol.py), and JSON text
    {"type": "runs", "runs": [...], "current": id}
    {"type": "history", "run_id": id, "records": [...]}
    {"type": "record", "run_id": id, "record": {...}}
A snapshot's header also carries "inspect": the selected member's internals.
Viewer -> engine:  JSON text control messages (see protocol.parse_control).
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import time
from pathlib import Path

from websockets.asyncio.server import ServerConnection, serve as ws_serve
from websockets.exceptions import ConnectionClosed

from .inspect import inspect_entity
from .metrics import compute_metrics
from .protocol import encode_snapshot, parse_control
from .recorder import list_runs, read_history
from .runner import Runner

log = logging.getLogger("simlife.server")

FRAME_SECONDS = 1 / 30
HEADLESS_CHUNK = 500
SEND_TIMEOUT = 2.0


class SimServer:
    def __init__(self, runner: Runner, runs_dir: Path, ticks: int | None = None):
        self.runner = runner
        self.runs_dir = Path(runs_dir)
        self.paused = False
        self.ticks_per_frame = 1
        self.port: int | None = None
        self._ticks_left = ticks  # None means run until closed
        self._viewers: set[ServerConnection] = set()
        self._closing = False
        self._server = None
        # One selection, shared by every viewer. A member is identified by its
        # slot AND the tick it was born, because slots are reused after death.
        self._selected: int | None = None
        self._selected_born = 0
        self._selected_died = False

    async def start(self, host: str, port: int) -> None:
        self._server = await ws_serve(self._handle, host, port)
        self.port = self._server.sockets[0].getsockname()[1]

    def request_stop(self) -> None:
        """Ask run() to finish after the work in flight; it then checkpoints."""
        self._closing = True

    async def close(self) -> None:
        self._closing = True
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    # --- the simulation loop -------------------------------------------------

    async def run(self) -> None:
        """Advance the simulation until closed, then checkpoint.

        With a viewer watching, run ticks_per_frame ticks per frame at 30
        frames a second. With nobody watching, run flat out in large chunks.
        """
        try:
            while not self._closing and self._ticks_left != 0:
                started = time.monotonic()
                if not self._viewers:
                    await self._advance(HEADLESS_CHUNK)
                    # a viewer that connected mid-chunk has a stale picture
                    await self._broadcast(self._snapshot())
                    # yield so new connections are accepted; don't spin on a dead world
                    await asyncio.sleep(0.1 if self.runner.extinct else 0)
                    continue
                if not self.paused:
                    records = await self._advance(self.ticks_per_frame)
                    await self._broadcast(self._snapshot())
                    for record in records:
                        await self._broadcast(self._record_message(record))
                elapsed = time.monotonic() - started
                await asyncio.sleep(max(0.0, FRAME_SECONDS - elapsed))
        finally:
            self.runner.checkpoint()

    async def _advance(self, n: int) -> list[dict]:
        if self._ticks_left is not None:
            n = min(n, self._ticks_left)
            self._ticks_left -= n
        # advance blocks while JAX computes, so it runs off the event loop
        return await asyncio.to_thread(self.runner.advance, n)

    # --- selection -----------------------------------------------------------

    @staticmethod
    def _born(state, slot: int) -> int:
        return int(state.tick) - int(state.age[slot])

    def select(self, slot: int | None) -> None:
        """Choose the member to inspect. Anything that isn't a living member clears it."""
        state = self.runner.state
        self._selected_died = False
        if slot is None or slot >= state.alive.shape[0] or not bool(state.alive[slot]):
            self._selected = None
            return
        self._selected = slot
        self._selected_born = self._born(state, slot)

    def inspection(self) -> dict | None:
        """The selected member's internals; a death notice once it has died; or None."""
        if self._selected is None:
            return None
        slot, state = self._selected, self.runner.state
        same_member = bool(state.alive[slot]) and self._born(state, slot) == self._selected_born
        if self._selected_died or not same_member:
            # Stays "died" until a new selection, even if a newborn now holds the slot.
            self._selected_died = True
            return {"slot": slot, "alive": False}
        return inspect_entity(state, self.runner.cfg, slot)

    # --- messages ------------------------------------------------------------

    def _snapshot(self) -> bytes:
        status = {
            **self.runner.status(),
            "paused": self.paused,
            "ticks_per_frame": self.ticks_per_frame,
        }
        state = self.runner.state
        return encode_snapshot(state, compute_metrics(state), status, self.inspection())

    def _runs_message(self) -> str:
        return json.dumps({
            "type": "runs",
            "runs": list_runs(self.runs_dir),
            "current": self.runner.recorder.run_id,
        })

    def _history_message(self, run_id: str) -> str:
        records = read_history(self.runs_dir / run_id)
        return json.dumps({"type": "history", "run_id": run_id, "records": records})

    def _record_message(self, record: dict) -> str:
        run_id = self.runner.recorder.run_id
        return json.dumps({"type": "record", "run_id": run_id, "record": record})

    async def _broadcast(self, message: str | bytes) -> None:
        """Send to every viewer. A viewer that fails or stalls is dropped, not waited for."""
        for viewer in list(self._viewers):
            try:
                await asyncio.wait_for(viewer.send(message), SEND_TIMEOUT)
            except Exception:  # noqa: BLE001 - any failing viewer must not stop the engine
                self._viewers.discard(viewer)
                # Cut the connection too, so the page notices and reconnects
                # instead of showing "connected" over a frozen picture.
                viewer.transport.abort()

    # --- one viewer ----------------------------------------------------------

    async def _handle(self, viewer: ServerConnection) -> None:
        try:
            await viewer.send(self._runs_message())
            await viewer.send(self._history_message(self.runner.recorder.run_id))
            await viewer.send(self._snapshot())
            self._viewers.add(viewer)
            async for message in viewer:
                await self._on_message(viewer, message)
        except ConnectionClosed:
            pass
        finally:
            self._viewers.discard(viewer)

    async def _on_message(self, viewer: ServerConnection, message: str | bytes) -> None:
        control = parse_control(message) if isinstance(message, str) else None
        if control is None:
            log.warning("ignoring a message that is not a known control: %.80r", message)
            return

        kind = control["type"]
        if kind == "list_runs":
            await viewer.send(self._runs_message())
            return
        if kind == "get_history":
            # only ids of real run directories are accepted, never a path
            if control["run_id"] in list_runs(self.runs_dir):
                await viewer.send(self._history_message(control["run_id"]))
            else:
                log.warning("ignoring a history request for unknown run %.80r",
                            control["run_id"])
            return

        if kind == "pause":
            self.paused = control["value"]
        elif kind == "speed":
            self.ticks_per_frame = control["ticks_per_frame"]
        elif kind == "reseed_on_extinction":
            self.runner.reseed_on_extinction = control["value"]
        elif kind == "select":
            self.select(control["slot"])
        elif kind in ("learning", "inheritance"):
            name = "learning" if kind == "learning" else "inherit_learned"
            value = control["value"] if kind == "learning" else control["value"] == "learned"
            # set_switch waits for the piece in flight, so keep it off the event loop
            line = await asyncio.to_thread(self.runner.set_switch, name, value)
            if line is not None:
                await self._broadcast(self._record_message(line))  # charts mark the change
        # confirm the change to everyone, even while paused
        await self._broadcast(self._snapshot())


class ListenError(OSError):
    """The server could not open its port."""


async def serve(
    runner: Runner, runs_dir: Path, host: str, port: int, ticks: int | None = None
) -> None:
    server = SimServer(runner, runs_dir, ticks)
    try:
        await server.start(host, port)
    except OSError as error:
        raise ListenError(error.errno, error.strerror) from error

    # Stop cleanly, with a checkpoint, on Ctrl-C and on an ordinary terminate.
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, server.request_stop)

    print(f"simlife: run {runner.recorder.run_id} serving on ws://{host}:{server.port}",
          flush=True)
    try:
        await server.run()
    finally:
        await server.close()
        print(f"run saved in {runner.recorder.run_dir}", flush=True)
