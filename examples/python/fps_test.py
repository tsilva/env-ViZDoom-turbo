#!/usr/bin/env python3

"""Report local ViZDoom step throughput for manual diagnosis.

This single-build helper has no matched baseline or validity gates. Do not use
its output for provider comparisons or public performance claims; use
TurboBench's ``vizdoom/basic-v1`` profile for those purposes.

Change the iteration count, resolution, window visibility, or state copying to
inspect local behavior. The default includes state copying to resemble a normal
consumer workload.
"""

import os
from argparse import ArgumentParser
from random import choice
from time import time

import tqdm

import vizdoom as vzd


DEFAULT_CONFIG = os.path.join(vzd.scenarios_path, "basic.cfg")
DEFAULT_ITERATIONS = 10000

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        dest="config",
        default=DEFAULT_CONFIG,
        nargs="?",
        help="Path to the configuration file of the scenario."
        " Please see "
        "../../scenarios/*cfg for more scenarios.",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        default=DEFAULT_ITERATIONS,
        type=int,
        help="Number of iterations (actions/steps) to run",
    )
    args = parser.parse_args()

    print(
        "Diagnostic only: use TurboBench vizdoom/basic-v1 for comparisons "
        "and public performance claims."
    )

    game = vzd.DoomGame()

    # Use other config file if you wish.
    game.load_config(args.config)

    # Override some options for the test if you wish.
    # game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    # game.set_screen_format(vzd.ScreenFormat.CRCGCB)

    # game.set_depth_buffer_enabled(False)
    # game.set_labels_buffer_enabled(False)
    # game.set_automap_buffer_enabled(False)
    # game.set_audio_buffer_enabled(False)
    # game.set_objects_info_enabled(False)
    # game.set_sectors_info_enabled(False)
    # game.set_notifications_buffer_enabled(False)

    game.set_window_visible(False)

    actions_num = game.get_available_buttons_size()
    actions = [
        [True if i == j else False for i in range(actions_num)]
        for j in range(actions_num)
    ]
    actions.append([False for _ in range(actions_num)])  # Idle action

    start = time()
    game.init()

    print(
        f"Checking FPS with {args.config} and selected features. It may take some time. Be patient."
    )
    print("Config:")
    print("Iterations/steps:", args.iterations)
    print("Resolution:", game.get_screen_width(), "x", game.get_screen_height())
    print("Buffer format:", game.get_screen_format())
    print("Depth buffer:", game.is_depth_buffer_enabled())
    print("Labels buffer:", game.is_labels_buffer_enabled())
    print("Automap buffer:", game.is_automap_buffer_enabled())
    print("Audio buffer:", game.is_audio_buffer_enabled())
    print("Objects info:", game.is_objects_info_enabled())
    print("Sectors info:", game.is_sectors_info_enabled())
    print("Notifications buffer:", game.is_notifications_buffer_enabled())
    print("=====================")

    for i in tqdm.trange(args.iterations, leave=False):
        if game.is_episode_finished():
            game.new_episode()

        # Copying happens here
        s = game.get_state()
        game.make_action(choice(actions))

    end = time()
    t = end - start

    print("Results:")
    print("Time:", round(t, 3), "s")
    print("FPS:", round(args.iterations / t, 2))

    game.close()
