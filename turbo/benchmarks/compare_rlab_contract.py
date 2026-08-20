#!/usr/bin/env python3
"""Compare deterministic traces for the optimized 32-lane RLab profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from itertools import zip_longest
from pathlib import Path

import numpy as np


def _update_array(digest: hashlib._Hash, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    if array.dtype.hasobject:
        digest.update(json.dumps(array.tolist(), sort_keys=True).encode("utf-8"))
    else:
        digest.update(array.tobytes())


def _update_infos(digest: hashlib._Hash, infos: dict[str, np.ndarray]) -> None:
    for key in sorted(infos):
        digest.update(key.encode("utf-8"))
        _update_array(digest, infos[key])


def _event_hash(*arrays: np.ndarray, infos: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        _update_array(digest, array)
    _update_infos(digest, infos)
    return digest.hexdigest()


def _trace() -> dict[str, object]:
    import env_vizdoom_turbo
    from env_vizdoom_turbo import EnvViZDoomTurboVecEnv

    env = EnvViZDoomTurboVecEnv(
        game="VizdoomBasic-v1",
        num_envs=32,
        num_threads=32,
        use_restricted_actions="discrete",
        obs_copy="safe_view",
        obs_resize=(84, 84),
        obs_grayscale=True,
        obs_layout="chw",
        frame_stack=4,
        frame_skip=4,
        maxpool_last_two=False,
        sticky_action_prob=0.0,
        obs_resize_algorithm="area",
        info_filter={"mode": "all", "keys": ["killcount"]},
        game_variables=["KILLCOUNT"],
    )
    trace = hashlib.sha256()
    event_hashes: list[str] = []
    event_names: list[str] = []
    reset_count = 0
    try:
        observations, infos = env.reset(seed=123)
        images = np.stack(env.get_images())
        action_rng = np.random.default_rng(918)
        event_hashes.append(_event_hash(observations, images, infos=infos))
        event_names.append("initial-reset")
        _update_array(trace, observations)
        _update_array(trace, images)
        _update_infos(trace, infos)
        for step in range(480):
            actions = action_rng.integers(
                env.single_action_space.n,
                size=32,
                dtype=np.int64,
            )
            observations, rewards, terminated, truncated, infos = env.step(actions)
            images = np.stack(env.get_images())
            values = (observations, images, rewards, terminated, truncated)
            event_hashes.append(_event_hash(*values, infos=infos))
            event_names.append(f"step-{step}")
            for value in values:
                _update_array(trace, value)
            _update_infos(trace, infos)
            reset_mask = terminated | truncated
            if np.any(reset_mask):
                state_indices = np.full(32, -1, dtype=np.int32)
                state_indices[reset_mask] = 0
                observations, infos = env.reset(
                    options={
                        "reset_mask": reset_mask,
                        "state_indices": state_indices,
                    }
                )
                images = np.stack(env.get_images())
                event_hashes.append(_event_hash(reset_mask, observations, images, infos=infos))
                event_names.append(f"reset-after-step-{step}")
                _update_array(trace, reset_mask)
                _update_array(trace, observations)
                _update_array(trace, images)
                _update_infos(trace, infos)
                reset_count += 1
        return {
            "sha256": trace.hexdigest(),
            "reset_count": reset_count,
            "event_hashes": event_hashes,
            "event_names": event_names,
            "package_path": str(Path(env_vizdoom_turbo.__file__).resolve()),
        }
    finally:
        env.close()


def _assignment(value: str) -> tuple[str, str]:
    name, separator, setting = value.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("environment values must use NAME=VALUE")
    return name, setting


def _run(python: Path, environment: dict[str, str]) -> dict[str, object]:
    completed = subprocess.run(
        [str(python.absolute()), str(Path(__file__).resolve()), "--trace"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **environment},
        cwd=Path(__file__).resolve().parent,
    )
    return json.loads(completed.stdout)


def _summary(trace: dict[str, object]) -> dict[str, object]:
    return {
        "sha256": trace["sha256"],
        "reset_count": trace["reset_count"],
        "event_count": len(trace["event_hashes"]),
        "package_path": trace["package_path"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--baseline-python", type=Path)
    parser.add_argument("--candidate-python", type=Path)
    parser.add_argument("--baseline-env", action="append", default=[], type=_assignment)
    parser.add_argument("--candidate-env", action="append", default=[], type=_assignment)
    args = parser.parse_args()
    if args.trace:
        print(json.dumps(_trace(), sort_keys=True))
        return 0
    if args.baseline_python is None or args.candidate_python is None:
        parser.error("--baseline-python and --candidate-python are required")

    baseline_env = dict(args.baseline_env)
    candidate_env = dict(args.candidate_env)
    baseline = _run(args.baseline_python, baseline_env)
    candidate = _run(args.candidate_python, candidate_env)
    first_mismatch = next(
        (
            index
            for index, (baseline_hash, candidate_hash) in enumerate(
                zip_longest(baseline["event_hashes"], candidate["event_hashes"])
            )
            if baseline_hash != candidate_hash
        ),
        None,
    )
    passed = (
        baseline["sha256"] == candidate["sha256"]
        and baseline["reset_count"] == candidate["reset_count"]
        and first_mismatch is None
    )
    mismatch_event = None
    if first_mismatch is not None:
        baseline_names = baseline["event_names"]
        candidate_names = candidate["event_names"]
        mismatch_event = {
            "index": first_mismatch,
            "baseline": (
                baseline_names[first_mismatch] if first_mismatch < len(baseline_names) else None
            ),
            "candidate": (
                candidate_names[first_mismatch] if first_mismatch < len(candidate_names) else None
            ),
        }
    print(
        json.dumps(
            {
                "schema_version": 1,
                "passed": passed,
                "baseline": _summary(baseline),
                "candidate": _summary(candidate),
                "baseline_env": baseline_env,
                "candidate_env": candidate_env,
                "first_mismatch": mismatch_event,
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
