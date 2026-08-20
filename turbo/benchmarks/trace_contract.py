#!/usr/bin/env python3
"""Emit deterministic hashes for every supported environment and image mode."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import env_vizdoom_turbo
import numpy as np
from env_vizdoom_turbo import EnvViZDoomTurboVecEnv
from env_vizdoom_turbo._env_vizdoom_turbo import preprocess_into

SCENARIOS = (
    "basic",
    "basic_audio",
    "basic_notifications",
    "deadly_corridor",
    "deathmatch",
    "defend_the_center",
    "defend_the_line",
    "health_gathering",
    "health_gathering_supreme",
    "my_way_home",
    "predict_position",
    "take_cover",
)
PROFILES = {
    "area_gray_chw": {
        "obs_resize": (37, 43),
        "obs_crop": (1, 2, 3, 4),
        "obs_crop_mode": "remove",
        "obs_grayscale": True,
        "obs_resize_algorithm": "area",
        "obs_layout": "chw",
        "frame_stack": 4,
        "maxpool_last_two": True,
    },
    "bilinear_rgb_hwc": {
        "obs_resize": (31, 35),
        "obs_crop": (2, 1, 4, 3),
        "obs_crop_mode": "mask",
        "obs_crop_fill": 19,
        "obs_grayscale": False,
        "obs_resize_algorithm": "bilinear",
        "obs_layout": "hwc",
        "frame_stack": 2,
        "maxpool_last_two": False,
    },
    "nearest_rgb_chw": {
        "obs_resize": (29, 41),
        "obs_crop": None,
        "obs_grayscale": False,
        "obs_resize_algorithm": "nearest",
        "obs_layout": "chw",
        "frame_stack": 3,
        "maxpool_last_two": True,
    },
}


def _array_fingerprint(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype.hasobject:
        payload = json.dumps(array.tolist(), sort_keys=True).encode()
    else:
        payload = np.ascontiguousarray(array).tobytes()
    return {
        "shape": array.shape,
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _info_fingerprint(infos: dict[str, np.ndarray]) -> dict[str, Any]:
    return {key: _array_fingerprint(infos[key]) for key in sorted(infos)}


def _trace(scenario: str, profile: dict[str, Any]) -> dict[str, Any]:
    env = EnvViZDoomTurboVecEnv(
        game=scenario,
        num_envs=2,
        num_threads=2,
        use_restricted_actions="minimal",
        obs_copy="copy",
        frame_skip=3,
        noop_reset_max=3,
        sticky_action_prob=0.25,
        reward_clip=True,
        info_filter="all",
        **profile,
    )
    try:
        observations, infos = env.reset(seed=[123, 456])
        transitions = [
            {
                "observation": _array_fingerprint(observations),
                "infos": _info_fingerprint(infos),
                "render": _array_fingerprint(np.stack(env.get_images())),
            }
        ]
        for step in range(6):
            actions = (np.arange(env.num_envs, dtype=np.int64) + step) % env.single_action_space.n
            observations, rewards, terminated, truncated, infos = env.step(actions)
            transitions.append(
                {
                    "observation": _array_fingerprint(observations),
                    "reward": _array_fingerprint(rewards),
                    "terminated": _array_fingerprint(terminated),
                    "truncated": _array_fingerprint(truncated),
                    "infos": _info_fingerprint(infos),
                    "render": _array_fingerprint(np.stack(env.get_images())),
                }
            )
            reset_mask = terminated | truncated
            if np.any(reset_mask):
                state_indices = np.full(env.num_envs, -1, dtype=np.int32)
                state_indices[reset_mask] = 0
                observations, infos = env.reset(
                    options={
                        "reset_mask": reset_mask,
                        "state_indices": state_indices,
                    }
                )
                transitions.append(
                    {
                        "reset_mask": reset_mask.tolist(),
                        "observation": _array_fingerprint(observations),
                        "infos": _info_fingerprint(infos),
                    }
                )
        return {
            "action_table_hash": env.action_table_hash,
            "buttons": env.buttons,
            "observation_space": repr(env.observation_space),
            "transitions": transitions,
        }
    finally:
        env.close()


def _preprocessing_traces() -> dict[str, Any]:
    rng = np.random.default_rng(7429)
    traces = {}
    for case in range(240):
        raw_height = int(rng.integers(2, 20))
        raw_width = int(rng.integers(2, 24))
        top = int(rng.integers(0, raw_height))
        bottom = int(rng.integers(0, raw_height - top))
        left = int(rng.integers(0, raw_width))
        right = int(rng.integers(0, raw_width - left))
        crop = [top, bottom, left, right]
        out_height = int(rng.integers(1, 22))
        out_width = int(rng.integers(1, 26))
        out_channels = 1 if case % 2 else 3
        algorithm = ("nearest", "bilinear", "area")[case % 3]
        mask_crop = case % 5 in {1, 4}
        current = rng.integers(
            0,
            256,
            size=(3, raw_height, raw_width, 3),
            dtype=np.uint8,
        )
        previous = (
            rng.integers(
                0,
                256,
                size=(3, raw_height, raw_width, 3),
                dtype=np.uint8,
            )
            if case % 4
            else None
        )
        output = np.empty(
            (3, out_height, out_width, out_channels),
            dtype=np.uint8,
        )
        preprocess_into(
            current,
            output,
            crop,
            mask_crop,
            case % 256,
            algorithm,
            previous,
        )
        traces[str(case)] = {
            "algorithm": algorithm,
            "crop": crop,
            "mask_crop": mask_crop,
            "previous": previous is not None,
            "output": _array_fingerprint(output),
        }
    return traces


def main() -> int:
    traces = {
        f"{scenario}/{profile_name}": _trace(scenario, profile)
        for scenario in SCENARIOS
        for profile_name, profile in PROFILES.items()
    }
    print(
        json.dumps(
            {
                "schema_version": 1,
                "python": sys.version,
                "package_path": str(Path(env_vizdoom_turbo.__file__).resolve()),
                "preprocessing": _preprocessing_traces(),
                "traces": traces,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
