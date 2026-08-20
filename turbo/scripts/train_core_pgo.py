#!/usr/bin/env python3
"""Train a ViZDoom core profile on the release throughput workload."""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import tempfile
from pathlib import Path

import numpy as np
import vizdoom as vzd


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _new_game(config: Path, config_directory: Path, lane: int):
    game = vzd.DoomGame()
    game.load_config(str(config))
    game.set_doom_config_path(str(config_directory / f"engine-{lane}.ini"))
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_audio_buffer_enabled(False)
    game.set_screen_format(vzd.ScreenFormat.DOOM_256_COLORS8)
    game.set_mode(vzd.Mode.PLAYER)
    game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
    game.add_game_args("+viz_turbo_profile 1")
    game.set_seed(lane)
    return game


def train(*, steps: int, lanes: int) -> None:
    config = Path(vzd.scenarios_path) / "basic.cfg"
    lane_indices = tuple(range(lanes))
    with tempfile.TemporaryDirectory(prefix="env-vizdoom-turbo-pgo-") as directory:
        games = [_new_game(config, Path(directory), lane) for lane in lane_indices]
        with concurrent.futures.ThreadPoolExecutor(max_workers=lanes) as pool:
            list(pool.map(lambda game: game.init(), games))
            action_width = len(games[0].get_available_buttons())
            variable_width = len(games[0].get_available_game_variables())
            height = games[0].get_screen_height()
            width = games[0].get_screen_width()
            actions = np.zeros((lanes, action_width), dtype=np.float64)
            for lane in lane_indices:
                action = lane % (action_width + 1)
                if action:
                    actions[lane, action - 1] = 1.0
            frames = np.empty((lanes, height, width), dtype=np.uint8)
            palettes = np.empty((lanes, 256, 3), dtype=np.uint8)
            rewards = np.empty(lanes, dtype=np.float32)
            terminated = np.empty(lanes, dtype=np.bool_)
            truncated = np.empty(lanes, dtype=np.bool_)
            game_variables = np.empty((lanes, variable_width), dtype=np.float64)
            stepper = vzd._TurboBatchStepper(
                games,
                4,
                True,
                actions,
                frames,
                palettes,
                rewards,
                terminated,
                truncated,
                game_variables,
            )
            native_api = stepper.native_api()
            context = ctypes.c_void_p(native_api[0])
            start_all = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p)(native_api[1])
            finish_lane = ctypes.CFUNCTYPE(
                ctypes.c_uint64,
                ctypes.c_void_p,
                ctypes.c_size_t,
            )(native_api[2])
            reset_lane = ctypes.CFUNCTYPE(
                ctypes.c_uint,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_uint,
            )(native_api[5])
            try:
                for step in range(steps):
                    if start_all(context) & 4:
                        raise RuntimeError("native PGO training start failed")
                    statuses = list(
                        pool.map(
                            lambda lane: finish_lane(context, lane),
                            lane_indices,
                        )
                    )
                    if any(status & 4 for status in statuses):
                        raise RuntimeError("native PGO training step failed")
                    reset_lanes = [lane for lane, status in enumerate(statuses) if status & 3]
                    if reset_lanes:
                        seeds = [lane + (step + 1) * lanes for lane in reset_lanes]
                        reset_statuses = list(
                            pool.map(
                                lambda lane, seed: reset_lane(context, lane, seed),
                                reset_lanes,
                                seeds,
                            )
                        )
                        if any(status & 4 for status in reset_statuses):
                            raise RuntimeError("native PGO training reset failed")
            finally:
                del stepper
                list(pool.map(lambda game: game.close(), games))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=_positive_int, default=2200)
    parser.add_argument("--lanes", type=_positive_int, default=32)
    args = parser.parse_args()
    train(steps=args.steps, lanes=args.lanes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
