import asyncio
import json

import pytest
from websockets.asyncio.client import connect

from simlife.config import Config
from simlife.protocol import decode_snapshot
from simlife.recorder import Recorder
from simlife.runner import Runner
from simlife.server import SimServer

cfg = Config(
    capacity_members=256, capacity_colonies=16, capacity_food=64, initial_colonies=4
)


@pytest.fixture
async def server(tmp_path):
    runner = Runner(cfg, 0, Recorder(tmp_path, cfg, 0))
    srv = SimServer(runner, tmp_path)
    await srv.start("127.0.0.1", 0)
    task = asyncio.create_task(srv.run())
    yield srv
    await srv.close()
    await task


def url(srv):
    return f"ws://127.0.0.1:{srv.port}"


async def next_snapshot(ws, timeout=10):
    async with asyncio.timeout(timeout):
        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                return decode_snapshot(message)[0]


async def greeting(ws):
    """The three messages every new viewer gets, in order."""
    runs = json.loads(await ws.recv())
    history = json.loads(await ws.recv())
    snapshot = decode_snapshot(await ws.recv())[0]
    return runs, history, snapshot


async def test_new_viewer_gets_runs_history_and_a_snapshot(server):
    async with connect(url(server)) as ws:
        runs, history, snapshot = await asyncio.wait_for(greeting(ws), 10)
    run_id = server.runner.recorder.run_id
    assert runs == {"type": "runs", "runs": [run_id], "current": run_id}
    assert history["type"] == "history" and history["run_id"] == run_id
    assert snapshot["status"]["run_id"] == run_id and snapshot["status"]["paused"] is False


async def test_two_viewers_both_receive_snapshots(server):
    async with connect(url(server)) as a, connect(url(server)) as b:
        await greeting(a)
        await greeting(b)
        ta, tb = (await next_snapshot(a))["tick"], (await next_snapshot(b))["tick"]
        assert (await next_snapshot(a))["tick"] > ta
        assert (await next_snapshot(b))["tick"] > tb


async def test_engine_survives_disconnect_and_junk(server):
    a = await connect(url(server))
    _, _, first = await greeting(a)
    await a.send("{{{")
    await a.send('{"type":"nope"}')
    await a.send(b"\x00\x01binary")
    assert (await next_snapshot(a))["tick"] >= first["tick"]
    a.transport.abort()  # drop the connection without a closing handshake

    async with connect(url(server)) as b:
        await greeting(b)
        assert (await next_snapshot(b))["tick"] > first["tick"]


async def test_headless_engine_advances_with_no_viewer(server):
    await asyncio.sleep(3.0)  # nobody connected; includes first-call compile time
    async with connect(url(server)) as ws:
        _, _, snapshot = await greeting(ws)
    assert snapshot["tick"] > 0


async def test_pause_stops_ticks_and_resume_restarts(server):
    async with connect(url(server)) as ws:
        await greeting(ws)
        await ws.send('{"type":"pause","value":true}')
        async with asyncio.timeout(10):
            while not (snap := await next_snapshot(ws))["status"]["paused"]:
                pass
        # work already in flight when the pause arrived may still land once
        await asyncio.sleep(1.0)
        await ws.send('{"type":"speed","ticks_per_frame":2}')  # any control echoes a snapshot
        async with asyncio.timeout(10):
            while (snap := await next_snapshot(ws))["status"]["ticks_per_frame"] != 2:
                pass
        paused_at = snap["tick"]

        await asyncio.sleep(0.5)
        await ws.send('{"type":"speed","ticks_per_frame":3}')
        async with asyncio.timeout(10):
            while (snap := await next_snapshot(ws))["status"]["ticks_per_frame"] != 3:
                pass
        assert snap["tick"] == paused_at

        await ws.send('{"type":"pause","value":false}')
        async with asyncio.timeout(10):
            while (await next_snapshot(ws))["tick"] <= paused_at:
                pass


async def test_reseed_switch_reaches_the_runner(server):
    async with connect(url(server)) as ws:
        await greeting(ws)
        await ws.send('{"type":"reseed_on_extinction","value":false}')
        async with asyncio.timeout(10):
            while (await next_snapshot(ws))["status"]["reseed_on_extinction"]:
                pass
    assert server.runner.reseed_on_extinction is False


async def test_get_history_rejects_unknown_run_id(server):
    async with connect(url(server)) as ws:
        await greeting(ws)
        await ws.send('{"type":"get_history","run_id":"../../etc"}')
        await ws.send('{"type":"list_runs"}')
        async with asyncio.timeout(10):
            while True:
                message = await ws.recv()
                if isinstance(message, str):
                    reply = json.loads(message)
                    assert reply["type"] != "history"
                    if reply["type"] == "runs":
                        break


async def test_get_history_returns_a_known_run(server):
    run_id = server.runner.recorder.run_id
    async with connect(url(server)) as ws:
        await greeting(ws)
        await ws.send(json.dumps({"type": "get_history", "run_id": run_id}))
        async with asyncio.timeout(10):
            while True:
                message = await ws.recv()
                if isinstance(message, str) and json.loads(message)["type"] == "history":
                    assert json.loads(message)["run_id"] == run_id
                    break


async def test_a_viewer_that_fails_a_send_is_disconnected(server):
    class FailingViewer:
        def __init__(self):
            self.aborted = False
            self.transport = self

        async def send(self, message):
            raise ConnectionError("gone")

        def abort(self):
            self.aborted = True

    viewer = FailingViewer()
    server._viewers.add(viewer)
    await server._broadcast(b"frame")
    assert viewer not in server._viewers
    assert viewer.aborted  # otherwise the page shows "connected" with a frozen picture


# --- stage 2 -------------------------------------------------------------------


async def test_select_then_inspect_arrives(server):
    async with connect(url(server)) as ws:
        _, _, first = await greeting(ws)
        assert first["inspect"] is None and first["status"]["learning"] is True
        await ws.send('{"type":"select","slot":2}')
        async with asyncio.timeout(10):
            while (snap := await next_snapshot(ws))["inspect"] is None:
                pass
        assert snap["inspect"]["slot"] == 2 and len(snap["inspect"]["inputs"]) == cfg.n_inputs
        await ws.send('{"type":"select","slot":1000000000}')
        async with asyncio.timeout(10):
            while (await next_snapshot(ws))["inspect"] is not None:
                pass


async def test_switch_controls_reach_runner_history_and_every_viewer(server):
    async with connect(url(server)) as a, connect(url(server)) as b:
        await greeting(a)
        await greeting(b)
        await a.send('{"type":"learning","value":false}')
        await a.send('{"type":"inheritance","value":"birth"}')
        seen_record = False
        async with asyncio.timeout(15):
            while True:
                message = await b.recv()
                if isinstance(message, str):
                    m = json.loads(message)
                    events = m.get("record", {}).get("events", [])
                    seen_record = seen_record or any(e.get("event") == "switch" for e in events)
                else:
                    status = decode_snapshot(message)[0]["status"]
                    if status["learning"] is False and status["inheritance"] == "birth" and seen_record:
                        break
    assert server.runner.status()["learning"] is False
    assert server.runner.status()["inheritance"] == "birth"
