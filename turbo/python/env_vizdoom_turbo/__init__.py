from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

import gymnasium as gym

from .action_tables import ActionTable
from .env import EnvViZDoomTurboVecEnv, scenario_buttons

GYMNASIUM_ENV_ID = "EnvViZDoomTurbo-v0"
_GYMNASIUM_VECTOR_ENTRY_POINT = "env_vizdoom_turbo:_make_gymnasium_vec_env"
_COMPATIBILITY_ENV_SPECS = {
    "EnvViZDoomBasicTurbo-v0": "VizdoomBasic-v1",
    "EnvViZDoomBasicPlus-v1": "VizdoomBasic-Plus-v1",
    "EnvViZDoomDeadlyCorridorTurbo-v0": "VizdoomDeadlyCorridor-v1",
    "EnvViZDoomDefendCenterTurbo-v0": "VizdoomDefendCenter-v1",
    "EnvViZDoomDefendLineTurbo-v0": "VizdoomDefendLine-v1",
    "EnvViZDoomDefendLinePlus-v1": "VizdoomDefendLine-Plus-v1",
    "EnvViZDoomHealthGatheringTurbo-v0": "VizdoomHealthGathering-v1",
    "EnvViZDoomHealthGatheringSupremeTurbo-v0": "VizdoomHealthGatheringSupreme-v1",
    "EnvViZDoomMyWayHomeTurbo-v0": "VizdoomMyWayHome-v1",
    "EnvViZDoomPredictPositionTurbo-v0": "VizdoomPredictPosition-v1",
    "EnvViZDoomTakeCoverTurbo-v0": "VizdoomTakeCover-v1",
}

try:
    __version__ = version("env-vizdoom-turbo")
except PackageNotFoundError:
    __version__ = "0.0.0"


def _make_gymnasium_vec_env(*, game: str, num_envs: int = 1, **kwargs: Any) -> EnvViZDoomTurboVecEnv:
    return EnvViZDoomTurboVecEnv(game=game, num_envs=num_envs, **kwargs)


def _register_gymnasium_envs() -> None:
    existing = gym.registry.get(GYMNASIUM_ENV_ID)
    if existing is None:
        gym.register(
            id=GYMNASIUM_ENV_ID,
            entry_point=None,
            vector_entry_point=_GYMNASIUM_VECTOR_ENTRY_POINT,
        )
    elif not (
        existing.entry_point is None
        and existing.vector_entry_point == _GYMNASIUM_VECTOR_ENTRY_POINT
        and existing.kwargs == {}
        and existing.max_episode_steps is None
        and existing.additional_wrappers == ()
    ):
        raise gym.error.Error(
            f"Gymnasium environment ID {GYMNASIUM_ENV_ID!r} is already "
            "registered with a conflicting specification"
        )

    for env_id, game in _COMPATIBILITY_ENV_SPECS.items():
        if env_id not in gym.registry:
            gym.register(
                id=env_id,
                entry_point=None,
                vector_entry_point="env_vizdoom_turbo:EnvViZDoomTurboVecEnv",
                kwargs={"game": game},
            )


_register_gymnasium_envs()

__all__ = [
    "ActionTable",
    "GYMNASIUM_ENV_ID",
    "EnvViZDoomTurboVecEnv",
    "__version__",
    "scenario_buttons",
]
