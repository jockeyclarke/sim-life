"""Command line:  python -m simlife run [options]"""

from __future__ import annotations

import argparse
import shutil
import signal
import sys
from pathlib import Path

from .config import Config
from .recorder import Recorder
from .runner import Runner

HEADLESS_CHUNK = 500


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="simlife")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the simulation")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--runs-dir", type=Path, default=Path("runs"))
    run.add_argument("--resume", type=Path, metavar="RUN_DIR",
                     help="continue a recorded run from its latest checkpoint")
    run.add_argument("--ticks", type=int, help="stop after this many ticks")
    run.add_argument("--no-serve", action="store_true",
                     help="run without the viewer server (requires --ticks)")
    run.add_argument("--no-reseed", action="store_true",
                     help="when every colony has died, stop instead of starting fresh ones")
    run.add_argument("--no-learning", action="store_true",
                     help="switch lifetime learning off (a pure-evolution run)")
    run.add_argument("--inherit", choices=["learned", "birth"],
                     help="what offspring copy: the parent's learned weights (the default "
                          "for a new run) or the weights the parent was born with")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8765)
    return parser


def make_runner(args: argparse.Namespace) -> Runner:
    if args.resume:
        runner = Runner.resume(args.resume)  # keeps the switches the run was left with
    else:
        cfg = Config()
        runner = Runner(cfg, args.seed, Recorder(args.runs_dir, cfg, args.seed))
    if args.no_reseed:
        runner.reseed_on_extinction = False
    # A resumed run keeps its saved switches unless a flag says otherwise.
    if args.no_learning:
        runner.set_switch("learning", False)
    if args.inherit:
        runner.set_switch("inherit_learned", args.inherit == "learned")
    return runner


def run_headless(runner: Runner, ticks: int) -> None:
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    # Finish the chunk in hand, then save: on Ctrl-C and on an ordinary terminate.
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    print(f"simlife: run {runner.recorder.run_id} headless for {ticks:,} ticks", flush=True)
    start = int(runner.state.tick)
    try:
        while not stop and int(runner.state.tick) - start < ticks:
            n = min(HEADLESS_CHUNK, ticks - (int(runner.state.tick) - start))
            for line in runner.advance(n):
                for event in line.get("events", []):
                    print(f"tick {event['tick']:>9}  {event['event']}", flush=True)
                if "members" in line and line["tick"] % 5000 == 0:
                    print(
                        f"tick {line['tick']:>9}  colonies {line['colonies']:>3}  "
                        f"members {line['members']:>5}  "
                        f"delivery rate {line['delivery_rate']:.5f}",
                        flush=True,
                    )
            if runner.extinct:
                print("every colony has died and reseeding is off; stopping", flush=True)
                break
        if stop:
            print("interrupted", flush=True)
    finally:
        runner.checkpoint()
        print(f"run saved in {runner.recorder.run_dir}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.no_serve and args.ticks is None:
        parser.error("--no-serve requires --ticks")

    try:
        runner = make_runner(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.no_serve:
        run_headless(runner, args.ticks)
        return 0

    import asyncio

    from .server import ListenError, serve

    try:
        # a resumed run may live outside --runs-dir; serve the directory it is actually in
        runs_dir = runner.recorder.run_dir.parent
        asyncio.run(serve(runner, runs_dir, args.host, args.port, ticks=args.ticks))
    except ListenError as error:
        print(
            f"error: cannot listen on {args.host} port {args.port} ({error.strerror}). "
            "Is another engine already running? Use --port to pick another.",
            file=sys.stderr,
        )
        if not args.resume:
            # nothing ran, so don't leave an empty run behind
            shutil.rmtree(runner.recorder.run_dir, ignore_errors=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
