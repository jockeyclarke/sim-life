"""Command line:  python -m simlife run [options]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config
from .recorder import Recorder, resume_run
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
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8765)
    return parser


def make_runner(args: argparse.Namespace) -> Runner:
    if args.resume:
        r = resume_run(args.resume)
        return Runner(r.cfg, r.seed, r.recorder, state=r.state, extinctions=r.extinctions)
    cfg = Config()
    return Runner(cfg, args.seed, Recorder(args.runs_dir, cfg, args.seed))


def run_headless(runner: Runner, ticks: int) -> None:
    done = 0
    try:
        while done < ticks:
            n = min(HEADLESS_CHUNK, ticks - done)
            for record in runner.advance(n):
                if record["tick"] % 5000 == 0 or record.get("events"):
                    print(
                        f"tick {record['tick']:>9}  colonies {record['colonies']:>3}  "
                        f"members {record['members']:>5}  "
                        f"delivery rate {record['delivery_rate']:.5f}",
                        flush=True,
                    )
            if runner.extinct:
                print("every colony has died and reseeding is off; stopping", flush=True)
                break
            done += n
    except KeyboardInterrupt:
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

    from .server import serve

    try:
        asyncio.run(serve(runner, args.runs_dir, args.host, args.port, ticks=args.ticks))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
