from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

import gymnasium as gym

from .action_tables import ActionTable
from .env import VizDoomTurboVecEnv, VizdoomTurboVecEnv, scenario_buttons

GYMNASIUM_ENV_ID = "Vizdoom-Turbo-v0"
_GYMNASIUM_VECTOR_ENTRY_POINT = "vizdoom_turbo:_make_gymnasium_vec_env"
_COMPATIBILITY_ENV_SPECS = {
    "VizdoomBasic-Turbo-v0": "VizdoomBasic-v1",
    "VizdoomBasic-Plus-v1": "VizdoomBasic-Plus-v1",
    "VizdoomDeadlyCorridor-Turbo-v0": "VizdoomDeadlyCorridor-v1",
    "VizdoomDefendCenter-Turbo-v0": "VizdoomDefendCenter-v1",
    "VizdoomDefendLine-Turbo-v0": "VizdoomDefendLine-v1",
    "VizdoomDefendLine-Plus-v1": "VizdoomDefendLine-Plus-v1",
    "VizdoomHealthGathering-Turbo-v0": "VizdoomHealthGathering-v1",
    "VizdoomHealthGatheringSupreme-Turbo-v0": "VizdoomHealthGatheringSupreme-v1",
    "VizdoomMyWayHome-Turbo-v0": "VizdoomMyWayHome-v1",
    "VizdoomPredictPosition-Turbo-v0": "VizdoomPredictPosition-v1",
    "VizdoomTakeCover-Turbo-v0": "VizdoomTakeCover-v1",
}

try:
    __version__ = version("vizdoom-turbo")
except PackageNotFoundError:
    __version__ = "0.0.0"


def _make_gymnasium_vec_env(*, game: str, num_envs: int = 1, **kwargs: Any) -> VizdoomTurboVecEnv:
    return VizdoomTurboVecEnv(game=game, num_envs=num_envs, **kwargs)


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
                vector_entry_point="vizdoom_turbo:VizdoomTurboVecEnv",
                kwargs={"game": game},
            )


_register_gymnasium_envs()

__all__ = [
    "ActionTable",
    "GYMNASIUM_ENV_ID",
    "VizDoomTurboVecEnv",
    "VizdoomTurboVecEnv",
    "__version__",
    "scenario_buttons",
]
