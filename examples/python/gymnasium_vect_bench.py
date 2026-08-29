#!/usr/bin/env python3

"""Report local Gymnasium vectorization throughput for manual diagnosis.

This single-build helper has no matched baseline or validity gates. Do not use
its output for provider comparisons or public performance claims; use
TurboBench's ``vizdoom/basic-v1`` profile for those purposes.
"""

import argparse
import time
import warnings

import gymnasium

# Importing the wrapper registers the ViZDoom environments in Gymnasium, so it should be imported before creating the environment
from vizdoom import gymnasium_wrapper  # noqa


warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--n_envs", type=int, default=1, help="Number of envs")
parser.add_argument(
    "--mode", type=str, default="async", help="Gymnasium vectorization mode"
)
args = parser.parse_args()
seed = 42
n_steps = 1000


if __name__ == "__main__":
    print(
        "Diagnostic only: use TurboBench vizdoom/basic-v1 for comparisons "
        "and public performance claims."
    )

    # Pick an environment VizdoomCorridor-v1
    envs = gymnasium.make_vec(
        "VizdoomCorridor-v1", num_envs=args.n_envs, vectorization_mode=args.mode
    )

    # Time one local configuration.
    start = time.time()

    observation, info = envs.reset()
    for _ in range(n_steps):
        # No learning here; this is a local diagnostic.
        actions = envs.action_space.sample()
        observations, rewards, terminations, truncations, infos = envs.step(actions)
        # no need for env.reset() here since the default is AutoReset(https://farama.org/Vector-Autoreset-Mode)
        # if terminated or truncated:
        #    observation, info = env.reset()
    print(f"{args.n_envs}  {n_steps * args.n_envs / round(time.time() - start, 1)}")

    envs.close()
