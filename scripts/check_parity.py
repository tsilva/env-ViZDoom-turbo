#!/usr/bin/env python3
"""Thin wrapper around TurboBench's ViZDoom parity profile."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turbobench", default="turbobench")
    parser.add_argument("--python", default="3.14", dest="python_minor")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    candidate = (
        f"env-vizdoom-turbo@artifact:{args.wheel.resolve()}"
        if args.wheel
        else f"env-vizdoom-turbo@checkout:{root}"
    )
    command = [
        args.turbobench,
        "parity",
        "vizdoom/basic-v2",
        "--candidate",
        candidate,
        "--output",
        str(args.output),
        "--python",
        args.python_minor,
    ]
    if not args.wheel:
        command.extend(("--allow-dirty", "--quick"))
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
