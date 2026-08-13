"""High-throughput Gymnasium vector environments backed by ViZDoom."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import operator
import os
import secrets
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Thread
from types import MappingProxyType
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import vizdoom as vzd
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from ._vizdoom_turbo import ActionHistory, ImageProcessor
from .action_tables import ActionTable, resolve_custom_action
from .enemy_variants import (
    plus_scenario,
    plus_scenario_alias,
    resolve_enemy_variants,
)
from .surface_variants import (
    load_surface_themes,
    resolve_surface_variants,
)

_DEFAULT_STATE = "default"
_BUILTIN_SCENARIOS = {
    "basic": "basic.cfg",
    "basic_audio": "basic_audio.cfg",
    "basic_notifications": "basic_notifications.cfg",
    "deadly_corridor": "deadly_corridor.cfg",
    "deathmatch": "deathmatch.cfg",
    "defend_the_center": "defend_the_center.cfg",
    "defend_the_line": "defend_the_line.cfg",
    "health_gathering": "health_gathering.cfg",
    "health_gathering_supreme": "health_gathering_supreme.cfg",
    "my_way_home": "my_way_home.cfg",
    "predict_position": "predict_position.cfg",
    "take_cover": "take_cover.cfg",
}


class _LanePool:
    """Persistent fixed-lane workers without per-step Future allocations."""

    def __init__(self, num_threads: int):
        self._num_threads = num_threads
        self._condition = Condition()
        self._generation = 0
        self._completed_workers = 0
        self._closed = False
        self._jobs: Sequence[tuple[Any, tuple[Any, ...]]] = ()
        self._results: list[Any] = []
        self._errors: list[BaseException | None] = []
        self._threads = [
            Thread(
                target=self._worker,
                args=(worker,),
                name=f"vizdoom-turbo-{worker}",
                daemon=True,
            )
            for worker in range(num_threads)
        ]
        for thread in self._threads:
            thread.start()

    def _worker(self, worker: int) -> None:
        observed_generation = 0
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed or self._generation != observed_generation
                )
                if self._closed:
                    return
                observed_generation = self._generation
                jobs = self._jobs
            for index in range(worker, len(jobs), self._num_threads):
                callback, arguments = jobs[index]
                try:
                    self._results[index] = callback(*arguments)
                except BaseException as exc:
                    self._errors[index] = exc
            with self._condition:
                self._completed_workers += 1
                if self._completed_workers == self._num_threads:
                    self._condition.notify()

    def run(self, jobs: Sequence[tuple[Any, tuple[Any, ...]]]) -> list[Any]:
        with self._condition:
            if self._closed:
                raise RuntimeError("lane pool is closed")
            self._jobs = jobs
            self._results = [None] * len(jobs)
            self._errors = [None] * len(jobs)
            self._completed_workers = 0
            self._generation += 1
            self._condition.notify_all()
            self._condition.wait_for(lambda: self._completed_workers == self._num_threads)
            results = self._results
            errors = self._errors
            self._jobs = ()
            self._results = []
            self._errors = []
        for error in errors:
            if error is not None:
                raise error
        return results

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        for thread in self._threads:
            thread.join()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(result)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(result)


def _probability(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be between 0.0 and 1.0") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return result


def _normalize_pair(value: Any, name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a (height, width) pair")
    try:
        height, width = value
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a (height, width) pair") from exc
    return _positive_int(height, f"{name} height"), _positive_int(width, f"{name} width")


def _normalize_crop(value: Any) -> tuple[int, int, int, int]:
    if value is None:
        return 0, 0, 0, 0
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError("obs_crop must contain top, bottom, left, right")
    try:
        top, bottom, left, right = value
    except (TypeError, ValueError) as exc:
        raise ValueError("obs_crop must contain top, bottom, left, right") from exc
    return tuple(
        _nonnegative_int(item, name)
        for item, name in zip(
            (top, bottom, left, right),
            ("obs_crop top", "obs_crop bottom", "obs_crop left", "obs_crop right"),
            strict=True,
        )
    )


def _normalize_reward_clip(value: Any) -> tuple[float, float] | None:
    if value is False:
        return None
    if value is True:
        return -1.0, 1.0
    try:
        low, high = value
        low, high = float(low), float(high)
    except (TypeError, ValueError) as exc:
        raise ValueError("reward_clip must be a bool or a (low, high) pair") from exc
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError("reward_clip bounds must be finite with low <= high")
    return low, high


def _normalize_seed(
    seed: int | Sequence[int | None] | None,
    num_envs: int,
) -> list[int | None]:
    if seed is None:
        return [None] * num_envs
    if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes, bytearray)):
        result = [None if value is None else int(value) for value in seed]
        if len(result) != num_envs:
            raise ValueError("seed sequence length must match num_envs")
        return result
    base = int(seed)
    return [base + lane for lane in range(num_envs)]


def _enemy_variant_rng(seed: int, role: str) -> np.random.Generator:
    """Return an RNG lane isolated from gameplay/no-op/sticky-action draws."""
    value = int(seed) & np.iinfo(np.uint64).max
    domain = int.from_bytes(hashlib.sha256(role.encode("utf-8")).digest()[:4], "little")
    sequence = np.random.SeedSequence(
        [value & np.iinfo(np.uint32).max, value >> 32, 0x56445A50, domain]
    )
    return np.random.default_rng(sequence)


def _surface_variant_rng(seed: int, role: str) -> np.random.Generator:
    """Return a surface-role RNG isolated from every other reset draw."""
    value = int(seed) & np.iinfo(np.uint64).max
    domain = int.from_bytes(hashlib.sha256(role.encode("utf-8")).digest()[:4], "little")
    sequence = np.random.SeedSequence(
        [value & np.iinfo(np.uint32).max, value >> 32, 0x53524643, domain]
    )
    return np.random.default_rng(sequence)


def _enum_name(value: Any) -> str:
    return str(getattr(value, "name", value)).split(".")[-1]


def _is_stable_integration(value: Any) -> bool:
    name = getattr(value, "name", None)
    if name is not None and str(name).strip().casefold() == "stable":
        return True
    if isinstance(value, str):
        return value.strip().casefold() == "stable"
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        return operator.index(value) == 1
    except TypeError:
        return False


@dataclass(frozen=True)
class _Scenario:
    config_path: Path
    doom_map: str | None
    doom_skill: int | None


def _resolve_scenario(game: str | Path | None, scenario: str | Path | None) -> _Scenario:
    requested = scenario if scenario not in (None, "scenario") else game
    if requested is None:
        requested = "VizdoomBasic-v1"
    candidate = Path(str(requested)).expanduser()
    if candidate.is_file():
        return _Scenario(candidate.resolve(), None, None)
    plus_alias = plus_scenario_alias(requested)
    if plus_alias is not None:
        config_path, _wad_hash = plus_scenario(plus_alias)
        return _Scenario(config_path, None, None)
    alias = str(requested).strip().casefold().removesuffix(".cfg")
    if alias in _BUILTIN_SCENARIOS:
        return _Scenario(Path(vzd.scenarios_path) / _BUILTIN_SCENARIOS[alias], None, None)
    try:
        import vizdoom.gymnasium_wrapper  # noqa: F401

        spec = gym.spec(str(requested))
    except (gym.error.Error, ImportError) as exc:
        choices = ", ".join(sorted(_BUILTIN_SCENARIOS))
        raise ValueError(
            f"unknown ViZDoom game/scenario {requested!r}; use a registered Vizdoom id, "
            f"a .cfg path, or one of: {choices}"
        ) from exc
    config_name = spec.kwargs.get("scenario_config_file")
    if not config_name:
        raise ValueError(f"Gymnasium environment {requested!r} is not a ViZDoom scenario")
    return _Scenario(
        Path(vzd.scenarios_path) / str(config_name),
        str(spec.kwargs["doom_map"]) if spec.kwargs.get("doom_map") else None,
        int(spec.kwargs["doom_skill"]) if spec.kwargs.get("doom_skill") else None,
    )


def scenario_buttons(
    game: str | Path | None = "VizdoomBasic-v1",
    *,
    scenario: str | Path | None = None,
) -> tuple[str, ...]:
    """Return ordered native button labels without starting a game instance."""
    resolved = _resolve_scenario(game, scenario)
    template = vzd.DoomGame()
    try:
        template.load_config(str(resolved.config_path))
        return tuple(_enum_name(button) for button in template.get_available_buttons())
    finally:
        template.close()


@dataclass(frozen=True)
class _StateAsset:
    label: str
    payload: bytes | None


def _state_asset(value: Any) -> _StateAsset:
    if value is None or str(value).strip().casefold() in {"", "default", "none"}:
        return _StateAsset(_DEFAULT_STATE, None)
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        return _StateAsset(f"sha256:{hashlib.sha256(payload).hexdigest()}", payload)
    path = Path(value).expanduser()
    if path.is_file():
        payload = path.read_bytes()
        return _StateAsset(str(value), payload)
    raise FileNotFoundError(f"ViZDoom state {value!r} is not a readable save file")


def _resolve_state_catalog(
    state: Any,
    state_catalog: Sequence[Any] | None,
) -> tuple[tuple[_StateAsset, ...], int]:
    if state is not None and state_catalog is not None:
        raise ValueError("state and state_catalog are mutually exclusive")
    values = (
        ((_DEFAULT_STATE if state is None else state),)
        if state_catalog is None
        else tuple(state_catalog)
    )
    if not values:
        raise ValueError("state_catalog must not be empty")
    assets = tuple(_state_asset(value) for value in values)
    labels = tuple(asset.label for asset in assets)
    if len(set(labels)) != len(labels):
        raise ValueError("state_catalog must contain unique state labels")
    return assets, 0


@dataclass(frozen=True)
class _EpisodeOrigin:
    game_seed: int
    state_index: int
    noop_count: int
    enemy_variant_indices: tuple[int, ...]
    surface_variant_indices: tuple[int, ...]


@dataclass(frozen=True)
class _LiveSnapshot:
    owner: str
    config_hash: str
    origin: _EpisodeOrigin
    action_history: np.ndarray
    stack: np.ndarray
    stack_head: int
    raw_frame: np.ndarray
    rng_state: dict[str, Any]
    enemy_variant_rng_states: tuple[dict[str, Any], ...]
    surface_variant_rng_states: tuple[dict[str, Any], ...]
    last_action: np.ndarray
    state_index: int
    episode_return: float
    info_frame_stacks: tuple[np.ndarray, ...]

    @property
    def nbytes(self) -> int:
        return (
            self.action_history.nbytes
            + self.stack.nbytes
            + self.raw_frame.nbytes
            + sum(stack.nbytes for stack in self.info_frame_stacks)
        )

    def __reduce__(self):
        raise TypeError("live ViZDoom snapshots are session-local and cannot be pickled")


class _SignalFrameStacks:
    """Typed policy-transition histories stored independently from image frames."""

    def __init__(
        self,
        num_envs: int,
        frame_stack: int,
        schema: Mapping[str, Mapping[str, Any]],
    ):
        self.keys = tuple(schema)
        self.frame_stack = frame_stack
        self._shapes = tuple(tuple(schema[key]["shape"]) for key in self.keys)
        self._dtypes = tuple(np.dtype(schema[key]["dtype"]) for key in self.keys)
        self._buffers = tuple(
            np.zeros((num_envs, frame_stack, *shape), dtype=dtype)
            for shape, dtype in zip(self._shapes, self._dtypes, strict=True)
        )
        self._num_envs = num_envs

    def _validate_values(
        self,
        values: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, ...]:
        if len(values) != len(self.keys):
            raise RuntimeError("selected info history values are incomplete")
        validated = []
        for key, value, shape, dtype in zip(
            self.keys,
            values,
            self._shapes,
            self._dtypes,
            strict=True,
        ):
            array = np.asarray(value)
            expected_shape = (self._num_envs, *shape)
            if array.shape != expected_shape:
                raise RuntimeError(
                    f"info signal {key!r} has shape {array.shape}; expected {expected_shape}"
                )
            if array.dtype != dtype:
                raise RuntimeError(f"info signal {key!r} has dtype {array.dtype}; expected {dtype}")
            validated.append(array)
        return tuple(validated)

    def reset(self, values: Sequence[np.ndarray], mask: np.ndarray) -> None:
        validated = self._validate_values(values)
        for lane in np.flatnonzero(mask):
            for buffer, value in zip(self._buffers, validated, strict=True):
                buffer[int(lane)] = value[int(lane)]

    def append(self, values: Sequence[np.ndarray]) -> None:
        validated = self._validate_values(values)
        for buffer, value in zip(self._buffers, validated, strict=True):
            for slot in range(self.frame_stack - 1):
                buffer[:, slot] = buffer[:, slot + 1]
            buffer[:, -1] = value

    def capture_lane(self, lane: int) -> tuple[np.ndarray, ...]:
        return tuple(buffer[lane].copy() for buffer in self._buffers)

    def restore_lane(self, lane: int, stacks: Sequence[np.ndarray]) -> None:
        if len(stacks) != len(self.keys):
            raise RuntimeError("snapshot info histories are incomplete")
        for key, buffer, stack, shape, dtype in zip(
            self.keys,
            self._buffers,
            stacks,
            self._shapes,
            self._dtypes,
            strict=True,
        ):
            array = np.asarray(stack)
            expected_shape = (self.frame_stack, *shape)
            if array.shape != expected_shape or array.dtype != dtype:
                raise RuntimeError(
                    f"snapshot info history {key!r} does not match the environment schema"
                )
            buffer[lane] = array

    def add_infos(self, result: dict[str, np.ndarray], present: np.ndarray) -> None:
        for key, buffer in zip(self.keys, self._buffers, strict=True):
            history_key = f"{key}_frame_stack"
            result[history_key] = buffer.copy()
            result[f"_{history_key}"] = present.copy()


class VizdoomTurboVecEnv(VectorEnv):
    """Direct Gymnasium vector environment for independent ViZDoom instances.

    ViZDoom's native frame-advance operation releases the GIL. This environment
    uses a provider-owned thread pool to advance lanes concurrently, then applies
    crop, max-pooling, resize, grayscale conversion, and frame stacking in
    preallocated batched buffers.

    Autoreset is permanently disabled. Terminal lanes retain their final policy
    observation and must be selected by a masked reset before any further step.

    ``info_frame_stack_keys`` optionally adds oldest-to-newest, lane-local
    policy-transition histories for selected reset-and-step info signals. Their
    depth is always ``frame_stack``; existing current-transition fields remain
    unchanged.
    """

    metadata = {
        "autoreset_mode": AutoresetMode.DISABLED,
        "render_modes": ["rgb_array"],
        "render_fps": int(vzd.DEFAULT_TICRATE),
        "turbo_api_version": 2,
        "transition_transport": "numpy",
    }
    supports_live_snapshots = True

    def __init__(
        self,
        game: str | Path,
        state: Any = None,
        scenario: str | Path | None = None,
        info: Any = None,
        use_restricted_actions: Any | str | ActionTable = "default",
        record: bool = False,
        players: int = 1,
        inttype: Any = "stable",
        obs_type: Any = "image",
        render_mode: Literal["rgb_array"] | None = None,
        *,
        num_envs: int = 1,
        num_threads: int | None = None,
        rom_path: str | None = None,
        transport: str = "default",
        obs_copy: Literal["copy", "safe_view", "unsafe_view"] = "safe_view",
        obs_resize: tuple[int, int] | None = (84, 84),
        obs_crop: tuple[int, int, int, int] | None = None,
        obs_crop_mode: Literal["remove", "mask"] = "remove",
        obs_crop_fill: int = 0,
        obs_grayscale: bool = True,
        obs_resize_algorithm: Literal["nearest", "bilinear", "area"] = "area",
        obs_layout: Literal["hwc", "chw"] = "chw",
        frame_skip: int = 4,
        frame_stack: int = 4,
        maxpool_last_two: bool = False,
        noop_reset_max: int = 0,
        use_fire_reset: bool = False,
        sticky_action_prob: float = 0.0,
        reward_clip: bool | tuple[float, float] = False,
        info_filter: str | Mapping[str, Any] = "all",
        info_frame_stack_keys: Sequence[str] | None = None,
        state_catalog: Sequence[Any] | None = None,
        doom_map: str | None = None,
        doom_skill: int | None = None,
        game_args: str | None = None,
        game_variables: Sequence[str] | None = None,
        enemy_variants: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
        surface_variants: Mapping[str, Sequence[str]] | None = None,
        treat_episode_timeout_as_truncation: bool = True,
        vizdoom_config: Mapping[str, Any] | None = None,
    ):
        if transport == "default":
            transport = "numpy"
        if transport != "numpy":
            raise ValueError("transport must be 'default' or 'numpy'")
        if isinstance(use_restricted_actions, str) and use_restricted_actions == "default":
            use_restricted_actions = "filtered"
        if info not in (None, "data"):
            raise ValueError("info must be None/'data'; use game_variables for ViZDoom signals")
        if record:
            raise ValueError("record=True is unsupported on the native vector path")
        if players != 1:
            raise ValueError("VizdoomTurboVecEnv currently supports players=1")
        if _enum_name(obs_type).casefold() not in {"image", "observations.image"}:
            raise ValueError("VizdoomTurboVecEnv supports image observations only")
        if not _is_stable_integration(inttype):
            raise ValueError("inttype must select the Stable integration")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        if use_fire_reset:
            raise ValueError("use_fire_reset is not applicable to ViZDoom")

        self.num_envs = _positive_int(num_envs, "num_envs")
        self.num_threads = (
            self.num_envs
            if num_threads is None
            else min(_positive_int(num_threads, "num_threads"), self.num_envs)
        )
        self.frame_skip = _positive_int(frame_skip, "frame_skip")
        self.frame_stack = _positive_int(frame_stack, "frame_stack")
        self.noop_reset_max = _nonnegative_int(noop_reset_max, "noop_reset_max")
        self.sticky_action_prob = _probability(sticky_action_prob, "sticky_action_prob")
        self.maxpool_last_two = bool(maxpool_last_two)
        self.reward_clip = _normalize_reward_clip(reward_clip)
        self.obs_grayscale = bool(obs_grayscale)
        self.obs_layout = str(obs_layout).casefold()
        if self.obs_layout not in {"hwc", "chw"}:
            raise ValueError("obs_layout must be 'hwc' or 'chw'")
        self.obs_copy = str(obs_copy).casefold()
        if self.obs_copy not in {"copy", "safe_view", "unsafe_view"}:
            raise ValueError("obs_copy must be 'copy', 'safe_view', or 'unsafe_view'")
        self.observation_ownership = (
            "owned"
            if self.obs_copy == "copy"
            else "unsafe_view"
            if self.obs_copy == "unsafe_view"
            else "safe_view"
        )
        self.observation_buffer_depth = (
            None if self.obs_copy == "copy" else 1 if self.obs_copy == "unsafe_view" else 2
        )
        self.obs_crop = _normalize_crop(obs_crop)
        resolved_resize = _normalize_pair(obs_resize, "obs_resize")
        self.obs_crop_mode = str(obs_crop_mode).casefold()
        if self.obs_crop_mode not in {"remove", "mask"}:
            raise ValueError("obs_crop_mode must be 'remove' or 'mask'")
        self.obs_crop_fill = _nonnegative_int(obs_crop_fill, "obs_crop_fill")
        if self.obs_crop_fill > 255:
            raise ValueError("obs_crop_fill must be in [0, 255]")
        self.obs_resize_algorithm = str(obs_resize_algorithm).casefold()
        if self.obs_resize_algorithm not in {"nearest", "bilinear", "area"}:
            raise ValueError("obs_resize_algorithm must be 'nearest', 'bilinear', or 'area'")
        self.treat_episode_timeout_as_truncation = bool(treat_episode_timeout_as_truncation)
        self.game = str(game or "VizdoomBasic-v1")
        self.transport = transport
        self.render_mode = render_mode
        self.autoreset_mode = AutoresetMode.DISABLED
        self.closed = False
        self._owner = secrets.token_hex(16)
        requested_scenario = scenario if scenario not in (None, "scenario") else game
        self._plus_scenario = plus_scenario_alias(requested_scenario)
        if self._plus_scenario is not None:
            (
                self._enemy_variant_specs,
                self.enemy_variant_catalog_sha256,
            ) = resolve_enemy_variants(self._plus_scenario, enemy_variants)
            _plus_config, self.enemy_variant_wad_sha256 = plus_scenario(self._plus_scenario)
            (
                self._surface_variant_specs,
                self.surface_variant_catalog_sha256,
            ) = resolve_surface_variants(self._plus_scenario, surface_variants)
            self.surface_variant_themes = load_surface_themes(self._plus_scenario)
            self.surface_variant_wad_sha256 = self.enemy_variant_wad_sha256
        else:
            if enemy_variants is not None:
                raise ValueError("enemy_variants is only supported by Plus environments")
            if surface_variants is not None:
                raise ValueError("surface_variants is only supported by Plus environments")
            self._enemy_variant_specs = MappingProxyType({})
            self._surface_variant_specs = MappingProxyType({})
            self.surface_variant_themes = MappingProxyType({})
            self.enemy_variant_catalog_sha256 = None
            self.enemy_variant_wad_sha256 = None
            self.surface_variant_catalog_sha256 = None
            self.surface_variant_wad_sha256 = None
        self.enemy_variant_roles = tuple(self._enemy_variant_specs)
        self.enemy_variants = MappingProxyType(
            {
                role: tuple(variant.variant_id for variant in variants)
                for role, variants in self._enemy_variant_specs.items()
            }
        )
        self.surface_variant_roles = tuple(self._surface_variant_specs)
        self.surface_variants = MappingProxyType(
            {
                role: tuple(variant.variant_id for variant in variants)
                for role, variants in self._surface_variant_specs.items()
            }
        )
        self._scenario = _resolve_scenario(game, scenario)
        self._doom_map = doom_map or self._scenario.doom_map
        self._doom_skill = int(doom_skill) if doom_skill is not None else self._scenario.doom_skill
        self._game_args = game_args
        self._rom_path = rom_path
        self._vizdoom_config = dict(vizdoom_config or {})
        self._requested_game_variables = tuple(game_variables or ())
        self._assets, self._default_state_index = _resolve_state_catalog(state, state_catalog)
        self.state_catalog = tuple(asset.label for asset in self._assets)
        exact_info_filter = (
            isinstance(info_filter, Mapping)
            and str(info_filter.get("mode", "")).casefold() == "all"
            and tuple(str(key).casefold() for key in info_filter.get("keys", ())) == ("killcount",)
        )
        self._optimized_profile = (
            self.num_envs == 32
            and self.num_threads == 32
            and self._scenario.config_path.resolve()
            == (Path(vzd.scenarios_path) / "basic.cfg").resolve()
            and self._doom_map is None
            and self._doom_skill is None
            and self._game_args is None
            and self._rom_path is None
            and not self._vizdoom_config
            and len(self._assets) == 1
            and self._assets[0].label == _DEFAULT_STATE
            and self._assets[0].payload is None
            and str(use_restricted_actions).casefold() == "discrete"
            and self.obs_copy == "safe_view"
            and resolved_resize == (84, 84)
            and self.obs_grayscale
            and self.obs_layout == "chw"
            and self.frame_stack == 4
            and self.frame_skip == 4
            and not self.maxpool_last_two
            and self.sticky_action_prob == 0.0
            and self.obs_resize_algorithm == "area"
            and self.obs_crop == (0, 0, 0, 0)
            and self.obs_crop_mode == "remove"
            and self.noop_reset_max == 0
            and self.reward_clip is None
            and exact_info_filter
            and tuple(
                str(variable).strip().casefold() for variable in self._requested_game_variables
            )
            == ("killcount",)
            and self.treat_episode_timeout_as_truncation
        )
        self._native_stepper_type = getattr(vzd, "_TurboBatchStepper", None)
        native_stepper_available = self._native_stepper_type is not None and all(
            hasattr(self._native_stepper_type, name)
            for name in (
                "indexed_frame_view",
                "native_api",
                "palette_view",
                "step_lane_into",
            )
        )
        native_processor_available = all(
            hasattr(ImageProcessor, name)
            for name in (
                "reset_native_batch_into",
                "step_native_batch_into",
            )
        )
        self._use_indexed_native = (
            native_stepper_available
            and native_processor_available
            and os.environ.get("VIZDOOM_TURBO_DISABLE_NATIVE_PIPELINE") != "1"
            and not self.maxpool_last_two
            and self.obs_grayscale
            and self.obs_resize_algorithm == "area"
            and obs_resize == (84, 84)
        )

        self._tempdir = tempfile.TemporaryDirectory(prefix="vizdoom-turbo-")
        template = self._new_game()
        try:
            self.raw_width = int(template.get_screen_width())
            self.raw_height = int(template.get_screen_height())
            self._use_indexed_native = (
                self._use_indexed_native and self.raw_width == 320 and self.raw_height == 240
            )
            self._button_enums = tuple(template.get_available_buttons())
            self.buttons = tuple(_enum_name(button) for button in self._button_enums)
            self._binary = np.asarray(
                [int(button.value) < int(vzd.BINARY_BUTTON_COUNT) for button in self._button_enums],
                dtype=np.bool_,
            )
            self._game_variables = tuple(template.get_available_game_variables())
            self.game_variable_names = tuple(
                _enum_name(variable).casefold() for variable in self._game_variables
            )
            self._configure_action_space(use_restricted_actions, template)
        finally:
            template.close()

        if (
            self.obs_crop[0] + self.obs_crop[1] >= self.raw_height
            or self.obs_crop[2] + self.obs_crop[3] >= self.raw_width
        ):
            raise ValueError("obs_crop removes the entire source image")
        source_h = (
            self.raw_height
            if self.obs_crop_mode == "mask"
            else self.raw_height - self.obs_crop[0] - self.obs_crop[1]
        )
        source_w = (
            self.raw_width
            if self.obs_crop_mode == "mask"
            else self.raw_width - self.obs_crop[2] - self.obs_crop[3]
        )
        self.obs_height, self.obs_width = resolved_resize or (source_h, source_w)
        channels = 1 if self.obs_grayscale else 3
        stacked_channels = channels * self.frame_stack
        single_shape = (
            (stacked_channels, self.obs_height, self.obs_width)
            if self.obs_layout == "chw"
            else (self.obs_height, self.obs_width, stacked_channels)
        )
        self.single_observation_space = gym.spaces.Box(0, 255, shape=single_shape, dtype=np.uint8)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self._stack = np.zeros(
            (
                self.num_envs,
                self.frame_stack,
                self.obs_height,
                self.obs_width,
                channels,
            ),
            dtype=np.uint8,
        )
        self._stack_heads = np.zeros(self.num_envs, dtype=np.int64)
        self._image_processor = ImageProcessor(
            self.num_envs,
            self.raw_height,
            self.raw_width,
            self.obs_height,
            self.obs_width,
            channels,
            list(self.obs_crop),
            self.obs_crop_mode == "mask",
            self.obs_crop_fill,
            self.obs_resize_algorithm,
            self.frame_stack,
            self.obs_layout,
            self.num_threads,
            self._optimized_profile,
        )
        raw_shape = (self.raw_height, self.raw_width, 3)
        self._raw_frame_batch = np.zeros((self.num_envs, *raw_shape), dtype=np.uint8)
        self._raw_frames = [self._raw_frame_batch[lane] for lane in range(self.num_envs)]
        self._previous_raw_batch = np.zeros((self.num_envs, *raw_shape), dtype=np.uint8)
        self._previous_raw = [self._previous_raw_batch[lane] for lane in range(self.num_envs)]
        buffer_count = 1 if self.obs_copy == "unsafe_view" else 2
        self._obs_buffers = [
            np.empty((self.num_envs, *single_shape), dtype=np.uint8) for _ in range(buffer_count)
        ]
        self._reward_buffers = [
            np.empty(self.num_envs, dtype=np.float32) for _ in range(buffer_count)
        ]
        self._terminated_buffers = [
            np.empty(self.num_envs, dtype=np.bool_) for _ in range(buffer_count)
        ]
        self._truncated_buffers = [
            np.empty(self.num_envs, dtype=np.bool_) for _ in range(buffer_count)
        ]
        self._buffer_index = 0
        self._initialized = np.zeros(self.num_envs, dtype=np.bool_)
        self._pending_reset = np.zeros(self.num_envs, dtype=np.bool_)
        self._all_initialized = False
        self._has_pending_reset = False
        self._active_state_indices = np.full(
            self.num_envs, self._default_state_index, dtype=np.int32
        )
        self._active_state_indices.setflags(write=False)
        self._active_enemy_variant_indices = np.full(
            (self.num_envs, len(self.enemy_variant_roles)), -1, dtype=np.int32
        )
        self._active_enemy_variant_indices.setflags(write=False)
        self._active_surface_variant_indices = np.full(
            (self.num_envs, len(self.surface_variant_roles)), -1, dtype=np.int32
        )
        self._active_surface_variant_indices.setflags(write=False)
        self._enemy_variant_by_scenario_index = tuple(
            {variant.scenario_index: variant for variant in self._enemy_variant_specs[role]}
            for role in self.enemy_variant_roles
        )
        self._surface_variant_by_scenario_index = tuple(
            {variant.scenario_index: variant for variant in self._surface_variant_specs[role]}
            for role in self.surface_variant_roles
        )
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_origins = [
            _EpisodeOrigin(
                lane,
                self._default_state_index,
                0,
                (-1,) * len(self.enemy_variant_roles),
                (-1,) * len(self.surface_variant_roles),
            )
            for lane in range(self.num_envs)
        ]
        self._action_history = ActionHistory(self.num_envs, len(self._button_enums))
        self._last_actions = np.zeros((self.num_envs, len(self._button_enums)), dtype=np.float64)
        self._rngs = [np.random.default_rng(lane) for lane in range(self.num_envs)]
        self._enemy_variant_rngs = [
            [_enemy_variant_rng(lane, role) for lane in range(self.num_envs)]
            for role in self.enemy_variant_roles
        ]
        self._surface_variant_rngs = [
            [_surface_variant_rng(lane, role) for lane in range(self.num_envs)]
            for role in self.surface_variant_roles
        ]
        self._seed_values = [None] * self.num_envs
        self._signal_names = (
            *self.game_variable_names,
            "episode_time",
            "episode_return",
            "player_dead",
            "pending_reset",
        )
        self._signals = np.zeros((self.num_envs, len(self._signal_names)), dtype=np.float64)
        self._all_info_present = np.ones(self.num_envs, dtype=np.bool_)
        self._configure_info_filter(info_filter)
        signal_schema = (
            {
                name: MappingProxyType(
                    {
                        "dtype": "float64",
                        "shape": (),
                        "available_on_reset": self._info_mode == "all",
                        "available_on_step": self._info_mode != "none",
                    }
                )
                for name in self._info_keys
            }
            if self._info_mode != "none"
            else {}
        )
        if self._plus_scenario is not None:
            for role in (*self.enemy_variant_roles, *self.surface_variant_roles):
                signal_schema[f"{role}_variant_index"] = MappingProxyType(
                    {
                        "dtype": "int32",
                        "shape": (),
                        "available_on_reset": True,
                        "available_on_step": False,
                    }
                )
        self._configure_info_frame_stacks(info_frame_stack_keys, signal_schema)
        for key in self.info_frame_stack_keys:
            original = signal_schema[key]
            signal_schema[f"{key}_frame_stack"] = MappingProxyType(
                {
                    "dtype": original["dtype"],
                    "shape": (self.frame_stack, *original["shape"]),
                    "available_on_reset": True,
                    "available_on_step": True,
                }
            )
        self.signal_schema = MappingProxyType(signal_schema)
        self.live_snapshots_deterministic = True
        self.capabilities = MappingProxyType(
            {
                "supported_action_modes": (
                    "all",
                    "filtered",
                    "multi_discrete",
                    "custom_discrete",
                ),
                "supported_observation_layouts": ("chw", "hwc"),
                "supported_observation_color_modes": ("grayscale", "rgb"),
                "supported_resize_algorithms": ("nearest", "bilinear", "area"),
                "supported_crop_modes": ("remove", "mask"),
                "supported_observation_copy_modes": (
                    "copy",
                    "safe_view",
                    "unsafe_view",
                ),
                "supported_transition_transports": ("numpy",),
                "supports_async_step": False,
                "supports_branching": False,
                "supports_device_api": False,
                "supports_emulator_ram": False,
                "supports_enemy_variants": self._plus_scenario is not None,
                "supports_fire_reset": False,
                "supports_info_frame_stack": True,
                "supports_live_snapshots": True,
                "supports_maxpool_last_two": True,
                "supports_noop_reset": True,
                "supports_per_lane_rgb": render_mode == "rgb_array",
                "supports_reward_clipping": True,
                "supports_snapshot_codec": False,
                "supports_state_catalog": True,
                "supports_sticky_action_prob": True,
                "supports_surface_variants": self._plus_scenario is not None,
            }
        )
        config_payload = {
            "scenario": str(self._scenario.config_path.resolve()),
            "doom_map": self._doom_map,
            "doom_skill": self._doom_skill,
            "buttons": self.buttons,
            "variables": self.game_variable_names,
            "frame_skip": self.frame_skip,
            "frame_stack": self.frame_stack,
            "info_frame_stack_keys": self.info_frame_stack_keys,
            "crop": self.obs_crop,
            "crop_mode": self.obs_crop_mode,
            "resize": (self.obs_height, self.obs_width),
            "grayscale": self.obs_grayscale,
            "layout": self.obs_layout,
            "enemy_variants": dict(self.enemy_variants),
            "enemy_variant_catalog_sha256": self.enemy_variant_catalog_sha256,
            "enemy_variant_wad_sha256": self.enemy_variant_wad_sha256,
            "surface_variants": dict(self.surface_variants),
            "surface_variant_catalog_sha256": self.surface_variant_catalog_sha256,
            "surface_variant_wad_sha256": self.surface_variant_wad_sha256,
        }
        self._config_hash = hashlib.sha256(
            json.dumps(config_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self._pool = _LanePool(self.num_threads)
        self._games = [self._new_game() for _ in range(self.num_envs)]
        for lane, lane_game in enumerate(self._games):
            lane_game.set_seed(lane)
        try:
            self._pool.run([(lane_game.init, ()) for lane_game in self._games])
        except BaseException:
            self.close()
            raise
        self._native_stepper = None
        if (
            self._use_indexed_native
            and hasattr(self._image_processor, "step_native_batch_into")
            and hasattr(self._image_processor, "reset_native_batch_into")
        ):
            self._native_actions_buffer = np.empty(
                (self.num_envs, len(self._button_enums)), dtype=np.float64
            )
            self._native_rewards = np.empty(self.num_envs, dtype=np.float32)
            self._native_terminated = np.empty(self.num_envs, dtype=np.bool_)
            self._native_truncated = np.empty(self.num_envs, dtype=np.bool_)
            self._native_game_variables = np.empty(
                (self.num_envs, len(self._game_variables)), dtype=np.float64
            )
            self._native_reset_seeds = np.zeros(self.num_envs, dtype=np.uint32)
            self._native_indexed_frames = np.empty(
                (self.num_envs, self.raw_height, self.raw_width), dtype=np.uint8
            )
            self._native_palettes = np.empty((self.num_envs, 256, 3), dtype=np.uint8)
            self._native_terminal_indexed_frames = np.empty_like(self._native_indexed_frames)
            self._native_terminal_palettes = np.empty_like(self._native_palettes)
            self._native_stepper = self._native_stepper_type(
                self._games,
                self.frame_skip,
                self.treat_episode_timeout_as_truncation,
                self._native_actions_buffer,
                self._native_indexed_frames,
                self._native_palettes,
                self._native_rewards,
                self._native_terminated,
                self._native_truncated,
                self._native_game_variables,
            )
            self._native_indexed_storage = self._native_indexed_frames
            self._native_palette_storage = self._native_palettes
            self._native_indexed_frames = tuple(
                self._native_stepper.indexed_frame_view(lane) for lane in range(self.num_envs)
            )
            self._native_palettes = tuple(
                self._native_stepper.palette_view(lane) for lane in range(self.num_envs)
            )
            native_api = self._native_stepper.native_api()
            if len(native_api) != 10 or any(int(address) <= 0 for address in native_api):
                raise RuntimeError("ViZDoom native batch API must contain 10 valid addresses")
            self._native_api = native_api[:5]
            self._native_reset_api = native_api[5]
            self._native_background_api = (
                native_api[6]
                if os.environ.get("VIZDOOM_TURBO_DISABLE_BACKGROUND_PROVENANCE") != "1"
                and (self.obs_crop_mode == "mask" or self.obs_crop == (0, 0, 0, 0))
                else None
            )
            self._native_reset_start_api = (
                native_api[7]
                if self._optimized_profile
                and os.environ.get("VIZDOOM_TURBO_DISABLE_ASYNC_RESET") != "1"
                else None
            )
            self._native_error_api = native_api[8:10]

    def _new_game(self):
        game = vzd.DoomGame()
        game.load_config(str(self._scenario.config_path))
        game.set_doom_config_path(
            str(Path(self._tempdir.name) / f"engine-{secrets.token_hex(8)}.ini")
        )
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_audio_buffer_enabled(False)
        game.set_screen_format(
            vzd.ScreenFormat.DOOM_256_COLORS8
            if self._use_indexed_native
            else vzd.ScreenFormat.RGB24
        )
        game.set_mode(vzd.Mode.PLAYER)
        if self._rom_path:
            game.set_doom_game_path(str(Path(self._rom_path).expanduser()))
        if self._doom_map:
            game.set_doom_map(self._doom_map)
        if self._doom_skill is not None:
            game.set_doom_skill(self._doom_skill)
        if self._optimized_profile:
            game.add_game_args("+viz_turbo_profile 1")
        if self._game_args:
            game.add_game_args(self._game_args)
        if self._vizdoom_config:
            game.set_config(self._vizdoom_config)
        for name in self._requested_game_variables:
            normalized = str(name).strip().upper()
            try:
                variable = getattr(vzd.GameVariable, normalized)
            except AttributeError as exc:
                raise ValueError(f"unknown ViZDoom game variable {name!r}") from exc
            if variable not in game.get_available_game_variables():
                game.add_available_game_variable(variable)
        return game

    def _configure_action_space(self, value: Any, template: Any) -> None:
        mode_name = _enum_name(value).casefold() if value is not None else "filtered"
        self.use_restricted_actions = value
        self._custom_actions: np.ndarray | None = None
        if mode_name in {"all", "filtered", "multi_discrete"}:
            self.action_mode = mode_name
            self.action_preset = None
            self.action_table = None
            self.action_meanings = self.buttons
            self.action_table_hash = None
            if np.all(self._binary):
                if mode_name == "multi_discrete":
                    self.single_action_space = gym.spaces.MultiDiscrete(
                        np.full(len(self.buttons), 2, dtype=np.int64)
                    )
                else:
                    self.single_action_space = gym.spaces.MultiBinary(len(self.buttons))
            else:
                low = np.zeros(len(self.buttons), dtype=np.float32)
                high = np.ones(len(self.buttons), dtype=np.float32)
                for index, (button, binary) in enumerate(
                    zip(self._button_enums, self._binary, strict=True)
                ):
                    if not binary:
                        maximum = abs(float(template.get_button_max_value(button)))
                        maximum = maximum if maximum > 0 else np.finfo(np.float32).max
                        low[index], high[index] = -maximum, maximum
                self.single_action_space = gym.spaces.Box(low, high, dtype=np.float32)
            self.action_space = batch_space(self.single_action_space, self.num_envs)
            return
        if np.any(~self._binary):
            binary_names = tuple(
                name for name, binary in zip(self.buttons, self._binary, strict=True) if binary
            )
        else:
            binary_names = self.buttons
        resolved = resolve_custom_action(
            "minimal" if mode_name == "discrete" else value,
            buttons=binary_names,
        )
        button_index = {name: index for index, name in enumerate(self.buttons)}
        actions = np.zeros((len(resolved.table), len(self.buttons)), dtype=np.float64)
        for action_index, labels in enumerate(resolved.table):
            for label in labels:
                actions[action_index, button_index[label]] = 1.0
        self._custom_actions = actions
        self.action_mode = "custom_discrete"
        self.action_preset = resolved.preset
        self.action_table = resolved.table
        self.action_meanings = resolved.meanings
        self.action_table_hash = resolved.table_hash
        self.single_action_space = gym.spaces.Discrete(len(resolved.table))
        self.action_space = gym.spaces.MultiDiscrete(
            np.full(self.num_envs, len(resolved.table), dtype=np.int64)
        )

    def _configure_info_filter(self, value: str | Mapping[str, Any]) -> None:
        if isinstance(value, Mapping):
            unknown = set(value) - {"mode", "keys"}
            if unknown:
                raise ValueError(f"unknown info_filter keys: {sorted(unknown)}")
            mode = str(value.get("mode", "all"))
            keys = value.get("keys")
            selected = self._signal_names if keys is None else tuple(map(str, keys))
        else:
            mode = str(value)
            selected = self._signal_names
        if mode not in {"all", "terminal", "none"}:
            raise ValueError("info_filter mode must be 'all', 'terminal', or 'none'")
        unknown_signals = set(selected) - set(self._signal_names)
        if unknown_signals:
            raise ValueError(f"unknown info keys: {sorted(unknown_signals)}")
        self._info_mode = mode
        self._info_keys = tuple(selected)
        self._info_indices = tuple(self._signal_names.index(key) for key in self._info_keys)
        self._collect_game_variables = mode != "none" and any(
            key in self.game_variable_names for key in self._info_keys
        )
        self._collect_episode_time = mode != "none" and "episode_time" in self._info_keys
        self._collect_episode_return = mode != "none" and "episode_return" in self._info_keys
        self._collect_player_dead = mode != "none" and "player_dead" in self._info_keys
        self._collect_pending_reset = mode != "none" and "pending_reset" in self._info_keys
        self._collect_derived_signals = any(
            (
                self._collect_episode_time,
                self._collect_episode_return,
                self._collect_player_dead,
                self._collect_pending_reset,
            )
        )

    def _configure_info_frame_stacks(
        self,
        value: Sequence[str] | None,
        signal_schema: Mapping[str, Mapping[str, Any]],
    ) -> None:
        if value is None:
            selected: tuple[str, ...] = ()
        else:
            if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
                raise TypeError("info_frame_stack_keys must be a sequence of signal names or None")
            if any(not isinstance(key, str) for key in value):
                raise TypeError("info_frame_stack_keys must contain only strings")
            selected = tuple(value)

        duplicate_keys = sorted({key for key in selected if selected.count(key) > 1})
        if duplicate_keys:
            raise ValueError(f"info_frame_stack_keys generate duplicate fields: {duplicate_keys}")
        unknown = set(selected) - set(self._signal_names)
        if unknown:
            raise ValueError(f"unknown info_frame_stack_keys: {sorted(unknown)}")

        history_names = tuple(f"{key}_frame_stack" for key in selected)
        generated_names = {*history_names, *(f"_{name}" for name in history_names)}
        current_names = {
            *self._signal_names,
            *(f"_{name}" for name in self._signal_names),
        }
        collisions = generated_names & current_names
        if collisions:
            raise ValueError(
                f"generated info frame-stack fields collide with signal names: {sorted(collisions)}"
            )

        incompatible = set(selected) - set(signal_schema)
        if incompatible:
            raise ValueError(
                f"info_frame_stack_keys must be included by info_filter: {sorted(incompatible)}"
            )
        unavailable = [
            key
            for key in selected
            if not signal_schema[key]["available_on_reset"]
            or not signal_schema[key]["available_on_step"]
        ]
        if unavailable:
            raise ValueError(
                "info_frame_stack_keys must be available on reset and every step: "
                f"{sorted(unavailable)}"
            )

        self.info_frame_stack_keys = selected
        self._info_frame_stack_indices = tuple(
            self._signal_names.index(key) for key in self.info_frame_stack_keys
        )
        history_schema = {key: signal_schema[key] for key in self.info_frame_stack_keys}
        self._info_frame_stacks = (
            _SignalFrameStacks(self.num_envs, self.frame_stack, history_schema)
            if history_schema
            else None
        )

    def _info_frame_stack_values(self) -> tuple[np.ndarray, ...]:
        return tuple(self._signals[:, index] for index in self._info_frame_stack_indices)

    def _next_buffers(self):
        index = self._buffer_index
        self._buffer_index = (self._buffer_index + 1) % len(self._obs_buffers)
        return (
            self._obs_buffers[index],
            self._reward_buffers[index],
            self._terminated_buffers[index],
            self._truncated_buffers[index],
        )

    def _returned_obs(self, value: np.ndarray) -> np.ndarray:
        return value.copy() if self.obs_copy == "copy" else value

    def _read_screen(self, lane_game: Any, fallback: np.ndarray) -> np.ndarray:
        state = lane_game.get_state()
        if state is None or state.screen_buffer is None:
            return fallback.copy()
        screen = np.asarray(state.screen_buffer, dtype=np.uint8)
        expected = (self.raw_height, self.raw_width, 3)
        if screen.shape != expected:
            raise RuntimeError(f"ViZDoom returned screen shape {screen.shape}; expected {expected}")
        return np.ascontiguousarray(screen)

    def _raw_signals(self, lane_game: Any) -> np.ndarray | None:
        if not self._collect_game_variables:
            return None
        values = np.empty(len(self._game_variables), dtype=np.float64)
        for index, variable in enumerate(self._game_variables):
            try:
                values[index] = float(lane_game.get_game_variable(variable))
            except Exception as exc:
                name = self.game_variable_names[index]
                if name in self.info_frame_stack_keys:
                    raise RuntimeError(
                        f"selected info history signal {name!r} is unavailable"
                    ) from exc
                values[index] = 0.0
        return values

    def _update_signal_row(self, lane: int) -> None:
        if self._info_mode == "none":
            return
        width = len(self._game_variables)
        lane_game = self._games[lane]
        if self._collect_episode_time:
            self._signals[lane, width] = float(lane_game.get_episode_time())
        if self._collect_episode_return:
            self._signals[lane, width + 1] = self._episode_returns[lane]
        if self._collect_player_dead:
            self._signals[lane, width + 2] = float(lane_game.is_player_dead())
        if self._collect_pending_reset:
            self._signals[lane, width + 3] = float(self._pending_reset[lane])

    def _infos(self, present: np.ndarray | None = None) -> dict[str, np.ndarray]:
        if self._info_mode == "none":
            return {}
        if present is None:
            present = self._all_info_present
        if self._info_mode == "terminal":
            present = present & self._pending_reset
        result: dict[str, np.ndarray] = {}
        for key, index in zip(self._info_keys, self._info_indices, strict=True):
            result[key] = self._signals[:, index].copy()
            result[f"_{key}"] = present.copy()
        if self._info_frame_stacks is not None:
            self._info_frame_stacks.add_infos(result, present)
        return result

    def _save_path(self, lane: int, purpose: str) -> Path:
        return Path(self._tempdir.name) / f"{purpose}-{lane}-{secrets.token_hex(8)}.zds"

    def _load_bytes(
        self,
        lane_game: Any,
        lane: int,
        payload: bytes,
        *,
        episode_time: int = 0,
    ) -> None:
        lane_game.new_episode()
        elapsed = max(0, episode_time - int(lane_game.get_episode_time()))
        if elapsed:
            lane_game.set_action([0.0] * len(self._button_enums))
            lane_game.advance_action(elapsed, False)
        path = self._save_path(lane, "load")
        try:
            path.write_bytes(payload)
            lane_game.load(str(path))
        finally:
            path.unlink(missing_ok=True)

    def _set_enemy_variants(self, lane_game: Any, scenario_indices: Sequence[int]) -> None:
        if self._plus_scenario is not None:
            if len(scenario_indices) != len(self.enemy_variant_roles):
                raise RuntimeError("enemy variant role selection is incomplete")
            for role_index, scenario_index in enumerate(scenario_indices):
                variants = self._enemy_variant_specs[self.enemy_variant_roles[role_index]]
                selector_cvar = variants[0].selector_cvar
                lane_game.send_game_command(f"set {selector_cvar} {int(scenario_index)}")

    def _set_surface_variants(self, lane_game: Any, scenario_indices: Sequence[int]) -> None:
        if self._plus_scenario is not None:
            if len(scenario_indices) != len(self.surface_variant_roles):
                raise RuntimeError("surface variant role selection is incomplete")
            for role_index, scenario_index in enumerate(scenario_indices):
                variants = self._surface_variant_specs[self.surface_variant_roles[role_index]]
                selector_cvar = variants[0].selector_cvar
                lane_game.send_game_command(f"set {selector_cvar} {int(scenario_index)}")

    def _reset_lane(
        self,
        lane: int,
        seed: int | None,
        asset: _StateAsset | None,
        snapshot: _LiveSnapshot | None,
        noop_count: int,
        enemy_variant_indices: Sequence[int],
        surface_variant_indices: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        lane_game = self._games[lane]
        self._set_enemy_variants(lane_game, enemy_variant_indices)
        self._set_surface_variants(lane_game, surface_variant_indices)
        if snapshot is not None:
            origin = snapshot.origin
            lane_game.set_seed(origin.game_seed)
            origin_asset = self._assets[origin.state_index]
            if origin_asset.payload is not None:
                self._load_bytes(lane_game, lane, origin_asset.payload)
            else:
                lane_game.new_episode()
            if origin.noop_count:
                lane_game.set_action([0.0] * len(self._button_enums))
                lane_game.advance_action(origin.noop_count, True)
            for action in snapshot.action_history:
                lane_game.set_action(action.tolist())
                if self.maxpool_last_two and self.frame_skip > 1:
                    lane_game.advance_action(self.frame_skip - 1, True)
                    if not lane_game.is_episode_finished():
                        lane_game.advance_action(1, True)
                else:
                    lane_game.advance_action(self.frame_skip, True)
        else:
            if seed is None:
                raise RuntimeError("static reset requires a resolved lane seed")
            lane_game.set_seed(int(seed) & np.iinfo(np.uint32).max)
            if asset is not None and asset.payload is not None:
                self._load_bytes(lane_game, lane, asset.payload)
            else:
                lane_game.new_episode()
            if noop_count:
                lane_game.set_action([0.0] * len(self._button_enums))
                lane_game.advance_action(noop_count, True)
        if self._native_stepper is not None:
            self._native_stepper.read_lane_into(lane)
            raw = self._raw_frames[lane]
        else:
            raw = self._read_screen(lane_game, self._raw_frames[lane])
        return raw, self._raw_signals(lane_game)

    def seed(self, seed: int | None = None) -> list[int | None]:
        self._seed_values = _normalize_seed(seed, self.num_envs)
        return list(self._seed_values)

    def reset(  # noqa: C901
        self,
        *,
        seed: int | Sequence[int | None] | None = None,
        options: Mapping[str, Any] | None = None,
    ):
        if self.closed:
            raise RuntimeError("cannot reset a closed environment")
        reset_options = dict(options or {})
        mask = reset_options.pop("reset_mask", None)
        if mask is None:
            mask = np.ones(self.num_envs, dtype=np.bool_)
        if not isinstance(mask, np.ndarray):
            raise TypeError("options['reset_mask'] must be a NumPy array")
        if mask.shape != (self.num_envs,):
            raise ValueError(f"options['reset_mask'] must have shape ({self.num_envs},)")
        if mask.dtype != np.bool_:
            raise TypeError("options['reset_mask'] must have dtype np.bool_")
        if not np.any(mask):
            raise ValueError("options['reset_mask'] must select at least one lane")

        snapshots = reset_options.pop("snapshots", None)
        snapshot_values: list[_LiveSnapshot | None]
        if snapshots is None:
            snapshot_values = [None] * self.num_envs
        else:
            if isinstance(snapshots, (str, bytes, bytearray)) or not isinstance(
                snapshots, Sequence
            ):
                raise TypeError("options['snapshots'] must be a lane-aligned sequence")
            if len(snapshots) != self.num_envs:
                raise ValueError(f"options['snapshots'] must have length {self.num_envs}")
            snapshot_values = list(snapshots)
        snapshot_mask = np.asarray([value is not None for value in snapshot_values], dtype=np.bool_)
        if np.any(snapshot_mask & ~mask):
            raise ValueError("snapshots may only be supplied for selected reset lanes")
        for value in (item for item in snapshot_values if item is not None):
            if not isinstance(value, _LiveSnapshot):
                raise TypeError("snapshot values must come from capture_snapshots()")
            if value.owner != self._owner or value.config_hash != self._config_hash:
                raise ValueError("snapshot belongs to a different environment instance/config")

        state_indices = reset_options.pop("state_indices", None)
        if state_indices is None:
            state_indices = np.full(self.num_envs, self._default_state_index, dtype=np.int32)
            state_indices[snapshot_mask] = -1
        if not isinstance(state_indices, np.ndarray):
            raise TypeError("options['state_indices'] must be a NumPy array")
        if state_indices.shape != (self.num_envs,):
            raise ValueError(f"options['state_indices'] must have shape ({self.num_envs},)")
        if state_indices.dtype != np.int32:
            raise TypeError("options['state_indices'] must have dtype np.int32")
        static_mask = mask & ~snapshot_mask
        static_lanes = np.flatnonzero(static_mask)
        reset_lane_values = np.flatnonzero(mask)
        selected = state_indices[static_mask]
        if np.any(selected < 0) or np.any(selected >= len(self.state_catalog)):
            raise ValueError("selected state_indices entries must index state_catalog")
        if np.any(state_indices[snapshot_mask] != -1):
            raise ValueError("snapshot reset lanes must use -1 for state_indices")
        if reset_options:
            raise ValueError(f"unsupported reset options: {sorted(reset_options)}")

        seeds = (
            _normalize_seed(seed, self.num_envs) if seed is not None else list(self._seed_values)
        )
        if any(seeds[lane] is not None for lane in np.flatnonzero(snapshot_mask)):
            raise ValueError("snapshot reset lanes cannot also specify a seed")
        noop_counts = np.zeros(self.num_envs, dtype=np.int64)
        enemy_variant_indices = self._active_enemy_variant_indices.copy()
        surface_variant_indices = self._active_surface_variant_indices.copy()
        game_seeds: list[int | None] = [None] * self.num_envs
        for lane in static_lanes:
            lane_seed = seeds[lane]
            if lane_seed is not None:
                self._rngs[lane] = np.random.default_rng(lane_seed)
                for role_index, role in enumerate(self.enemy_variant_roles):
                    self._enemy_variant_rngs[role_index][lane] = _enemy_variant_rng(lane_seed, role)
                for role_index, role in enumerate(self.surface_variant_roles):
                    self._surface_variant_rngs[role_index][lane] = _surface_variant_rng(
                        lane_seed, role
                    )
            game_seeds[lane] = int(
                self._rngs[lane].integers(
                    0,
                    np.iinfo(np.uint32).max + 1,
                    dtype=np.uint32,
                )
            )
            if self.noop_reset_max:
                noop_counts[lane] = self._rngs[lane].integers(1, self.noop_reset_max + 1)
            for role_index, role in enumerate(self.enemy_variant_roles):
                variants = self._enemy_variant_specs[role]
                selection = int(
                    self._enemy_variant_rngs[role_index][lane].integers(0, len(variants))
                )
                enemy_variant_indices[lane, role_index] = variants[selection].scenario_index
            for role_index, role in enumerate(self.surface_variant_roles):
                variants = self._surface_variant_specs[role]
                selection = int(
                    self._surface_variant_rngs[role_index][lane].integers(0, len(variants))
                )
                surface_variant_indices[lane, role_index] = variants[selection].scenario_index
        for lane in np.flatnonzero(snapshot_mask):
            snapshot = snapshot_values[int(lane)]
            if snapshot is not None:
                enemy_variant_indices[lane] = snapshot.origin.enemy_variant_indices
                surface_variant_indices[lane] = snapshot.origin.surface_variant_indices
        native_static_reset = (
            self._optimized_profile
            and self._native_stepper is not None
            and self._native_reset_api is not None
            and not np.any(snapshot_mask)
            and not np.any(noop_counts[static_mask])
            and all(self._assets[int(state_indices[lane])].payload is None for lane in static_lanes)
        )
        if native_static_reset:
            for lane in static_lanes:
                self._set_enemy_variants(
                    self._games[int(lane)],
                    enemy_variant_indices[lane],
                )
                self._set_surface_variants(
                    self._games[int(lane)],
                    surface_variant_indices[lane],
                )
                self._native_reset_seeds[lane] = game_seeds[lane]
        else:
            pending_snapshot_lanes = np.flatnonzero(snapshot_mask & self._pending_reset)
            if (
                self._native_stepper is not None
                and self._native_reset_start_api is not None
                and hasattr(self._native_stepper, "reset_lane_into")
                and pending_snapshot_lanes.size
            ):
                self._pool.run(
                    [
                        (
                            self._native_stepper.reset_lane_into,
                            (int(lane), int(self._native_reset_seeds[lane])),
                        )
                        for lane in pending_snapshot_lanes
                    ]
                )
            reset_lanes = []
            reset_jobs = []
            for lane in reset_lane_values:
                lane_index = int(lane)
                snapshot = snapshot_values[lane_index]
                asset = (
                    None if snapshot is not None else self._assets[int(state_indices[lane_index])]
                )
                reset_lanes.append(lane_index)
                reset_jobs.append(
                    (
                        self._reset_lane,
                        (
                            lane_index,
                            game_seeds[lane_index],
                            asset,
                            snapshot,
                            int(noop_counts[lane_index]),
                            tuple(int(value) for value in enemy_variant_indices[lane_index]),
                            tuple(int(value) for value in surface_variant_indices[lane_index]),
                        ),
                    )
                )
            for lane, (raw, raw_signals) in zip(
                reset_lanes, self._pool.run(reset_jobs), strict=True
            ):
                self._raw_frames[lane][...] = raw
                if raw_signals is not None:
                    width = len(self._game_variables)
                    self._signals[lane, :width] = raw_signals
        self._active_state_indices.setflags(write=True)
        self._active_enemy_variant_indices.setflags(write=True)
        self._active_surface_variant_indices.setflags(write=True)
        self._action_history.clear(static_mask)
        for lane in reset_lane_values:
            lane_index = int(lane)
            snapshot = snapshot_values[lane_index]
            if snapshot is None:
                self._stack_heads[lane_index] = 0
                self._episode_returns[lane_index] = 0.0
                self._last_actions[lane_index] = 0.0
                self._active_state_indices[lane_index] = state_indices[lane_index]
                self._episode_origins[lane_index] = _EpisodeOrigin(
                    game_seed=int(game_seeds[lane_index]),
                    state_index=int(state_indices[lane_index]),
                    noop_count=int(noop_counts[lane_index]),
                    enemy_variant_indices=tuple(
                        int(value) for value in enemy_variant_indices[lane_index]
                    ),
                    surface_variant_indices=tuple(
                        int(value) for value in surface_variant_indices[lane_index]
                    ),
                )
            else:
                self._stack[lane_index] = snapshot.stack
                self._stack_heads[lane_index] = snapshot.stack_head
                self._raw_frames[lane_index][...] = snapshot.raw_frame
                self._rngs[lane_index].bit_generator.state = copy.deepcopy(snapshot.rng_state)
                self._last_actions[lane_index] = snapshot.last_action
                self._active_state_indices[lane_index] = snapshot.state_index
                for role_index, rng_state in enumerate(snapshot.enemy_variant_rng_states):
                    self._enemy_variant_rngs[role_index][
                        lane_index
                    ].bit_generator.state = copy.deepcopy(rng_state)
                for role_index, rng_state in enumerate(snapshot.surface_variant_rng_states):
                    self._surface_variant_rngs[role_index][
                        lane_index
                    ].bit_generator.state = copy.deepcopy(rng_state)
                self._episode_returns[lane_index] = snapshot.episode_return
                self._episode_origins[lane_index] = snapshot.origin
                self._action_history.replace_lane(lane_index, snapshot.action_history)
            self._active_enemy_variant_indices[lane_index] = enemy_variant_indices[lane_index]
            self._active_surface_variant_indices[lane_index] = surface_variant_indices[lane_index]
        self._active_state_indices.setflags(write=False)
        self._active_enemy_variant_indices.setflags(write=False)
        self._active_surface_variant_indices.setflags(write=False)
        self._initialized[mask] = True
        self._pending_reset[mask] = False
        if not self._all_initialized:
            self._all_initialized = bool(np.all(self._initialized))
        self._has_pending_reset = bool(np.any(self._pending_reset))
        observations, _rewards, _terminated, _truncated = self._next_buffers()
        if self._native_stepper is not None:
            if native_static_reset:
                self._image_processor.reset_native_batch_into(
                    self._native_api[0],
                    self._native_api[3],
                    self._native_api[4],
                    *self._native_error_api,
                    static_mask,
                    self._stack,
                    self._stack_heads,
                    observations,
                    self._native_reset_api,
                    self._native_reset_seeds,
                )
                if self._collect_game_variables:
                    width = len(self._game_variables)
                    self._signals[static_mask, :width] = self._native_game_variables[
                        static_mask, :width
                    ]
            else:
                self._image_processor.reset_native_batch_into(
                    self._native_api[0],
                    self._native_api[3],
                    self._native_api[4],
                    *self._native_error_api,
                    static_mask,
                    self._stack,
                    self._stack_heads,
                    observations,
                )
        else:
            self._image_processor.reset_frames_into(
                self._raw_frames,
                self._stack,
                self._stack_heads,
                observations,
                static_mask,
            )
        if self._collect_derived_signals:
            for lane in reset_lane_values:
                self._update_signal_row(int(lane))
        if self._info_frame_stacks is not None:
            self._info_frame_stacks.reset(self._info_frame_stack_values(), static_mask)
            for lane in np.flatnonzero(snapshot_mask):
                snapshot = snapshot_values[int(lane)]
                if snapshot is not None:
                    self._info_frame_stacks.restore_lane(
                        int(lane),
                        snapshot.info_frame_stacks,
                    )
        if self._native_stepper is not None and self._native_reset_start_api is not None:
            for lane in reset_lane_values:
                lane_index = int(lane)
                generator = self._rngs[lane_index]
                generator_state = copy.deepcopy(generator.bit_generator.state)
                self._native_reset_seeds[lane_index] = generator.integers(
                    0,
                    np.iinfo(np.uint32).max + 1,
                    dtype=np.uint32,
                )
                generator.bit_generator.state = generator_state
        infos = self._infos(mask.copy())
        infos["state_index"] = self._active_state_indices.copy()
        infos["_state_index"] = mask.copy()
        if self._plus_scenario is not None:
            for role_index, role in enumerate(self.enemy_variant_roles):
                infos[f"{role}_variant_index"] = self._active_enemy_variant_indices[
                    :, role_index
                ].copy()
                infos[f"_{role}_variant_index"] = mask.copy()
            for role_index, role in enumerate(self.surface_variant_roles):
                infos[f"{role}_variant_index"] = self._active_surface_variant_indices[
                    :, role_index
                ].copy()
                infos[f"_{role}_variant_index"] = mask.copy()
        infos["start_source"] = snapshot_mask.astype(np.int8, copy=True)
        infos["_start_source"] = mask.copy()
        infos["noop_reset_count"] = noop_counts
        infos["_noop_reset_count"] = static_mask.copy()
        self._seed_values = [None] * self.num_envs
        return self._returned_obs(observations), infos

    def _native_actions(
        self,
        actions: Any,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        if not isinstance(actions, np.ndarray):
            raise TypeError("actions must be a NumPy array")
        if self._custom_actions is not None:
            values = np.asarray(actions, dtype=np.int64).reshape(-1)
            if values.shape != (self.num_envs,):
                raise ValueError(f"actions must have shape ({self.num_envs},)")
            if out is not None and values.flags.c_contiguous:
                self._image_processor.prepare_discrete_actions_into(
                    values, self._custom_actions, out
                )
                return out
            if values.size and int(values.min()) < 0:
                raise ValueError(f"actions must be in [0, {len(self._custom_actions) - 1}]")
            if out is not None:
                try:
                    np.take(self._custom_actions, values, axis=0, out=out)
                except IndexError as error:
                    raise ValueError(
                        f"actions must be in [0, {len(self._custom_actions) - 1}]"
                    ) from error
                return out
            if values.size and int(values.max()) >= len(self._custom_actions):
                raise ValueError(f"actions must be in [0, {len(self._custom_actions) - 1}]")
            return self._custom_actions[values]
        values = np.asarray(actions, dtype=np.float64)
        expected = (self.num_envs, len(self.buttons))
        if values.shape != expected:
            raise ValueError(f"actions must have shape {expected}")
        if np.any(self._binary & np.any((values != 0.0) & (values != 1.0), axis=0)):
            raise ValueError("binary ViZDoom actions must contain only 0 or 1")
        if np.all(self._binary):
            return values
        low = np.asarray(self.single_action_space.low, dtype=np.float64)
        high = np.asarray(self.single_action_space.high, dtype=np.float64)
        if np.any(values < low) or np.any(values > high):
            raise ValueError("actions are outside the declared action space")
        return values

    def _step_lane(
        self,
        lane: int,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, bool, bool, np.ndarray | None]:
        lane_game = self._games[lane]
        fallback = self._raw_frames[lane]
        lane_game.set_action(action.tolist())
        total_before = float(lane_game.get_total_reward())
        previous = fallback
        if self.maxpool_last_two and self.frame_skip > 1:
            lane_game.advance_action(self.frame_skip - 1, True)
            previous = self._read_screen(lane_game, fallback)
            if not lane_game.is_episode_finished():
                lane_game.advance_action(1, True)
        else:
            lane_game.advance_action(self.frame_skip, True)
        reward = float(lane_game.get_total_reward()) - total_before
        raw = self._read_screen(lane_game, previous)
        finished = bool(lane_game.is_episode_finished())
        timeout = bool(lane_game.is_episode_timeout_reached()) if finished else False
        truncated = finished and timeout and self.treat_episode_timeout_as_truncation
        terminated = finished and not truncated
        return (
            raw,
            previous,
            reward,
            terminated,
            truncated,
            self._raw_signals(lane_game),
        )

    def _sync_native_rgb(self, lane: int) -> None:
        if self._native_stepper is not None:
            if self._pending_reset[lane]:
                palette = self._native_terminal_palettes[lane]
                indexed = self._native_terminal_indexed_frames[lane]
            else:
                palette = self._native_palettes[lane]
                indexed = self._native_indexed_frames[lane]
            self._raw_frames[lane][...] = palette[indexed]

    def step(self, actions: Any):
        if self.closed:
            raise RuntimeError("cannot step a closed environment")
        if not self._all_initialized:
            raise RuntimeError("all lanes must be reset before the first step")
        if self._has_pending_reset:
            lanes = np.flatnonzero(self._pending_reset).tolist()
            raise RuntimeError(f"terminal lanes must be reset before step: {lanes}")
        direct_native_actions = (
            self._native_actions_buffer
            if self._native_stepper is not None and not self.sticky_action_prob
            else None
        )
        requested = self._native_actions(actions, direct_native_actions)
        applied = requested if direct_native_actions is not None else requested.copy()
        if self.sticky_action_prob:
            for lane in range(self.num_envs):
                if self._rngs[lane].random() < self.sticky_action_prob:
                    applied[lane] = self._last_actions[lane]
                else:
                    self._last_actions[lane] = requested[lane]
        else:
            self._last_actions[:] = requested
        observations, rewards, terminated, truncated = self._next_buffers()
        if self._native_stepper is not None:
            if applied is not self._native_actions_buffer:
                self._native_actions_buffer[...] = applied
            self._has_pending_reset = self._image_processor.step_native_batch_into(
                *self._native_api,
                *self._native_error_api,
                self._stack,
                self._stack_heads,
                observations,
                self._native_terminal_indexed_frames,
                self._native_terminal_palettes,
                self._native_background_api,
                self._native_reset_start_api,
                self._native_reset_seeds if self._native_reset_start_api is not None else None,
            )
            rewards[...] = self._native_rewards
            terminated[...] = self._native_terminated
            truncated[...] = self._native_truncated
            if self._collect_game_variables:
                width = len(self._game_variables)
                self._signals[:, :width] = self._native_game_variables
        else:
            results = self._pool.run(
                [(self._step_lane, (lane, applied[lane])) for lane in range(self.num_envs)]
            )
            for lane, (
                raw,
                previous,
                reward,
                lane_terminated,
                lane_truncated,
                raw_signals,
            ) in enumerate(results):
                self._raw_frames[lane][...] = raw
                self._previous_raw[lane][...] = previous
                rewards[lane] = reward
                terminated[lane] = lane_terminated
                truncated[lane] = lane_truncated
                if raw_signals is not None:
                    width = len(self._game_variables)
                    self._signals[lane, :width] = raw_signals
        self._action_history.append(applied)
        if self.reward_clip is not None:
            np.clip(rewards, self.reward_clip[0], self.reward_clip[1], out=rewards)
        if self._native_stepper is None:
            self._image_processor.step_frames_into(
                self._raw_frames,
                self._stack,
                self._stack_heads,
                observations,
                self._previous_raw if self.maxpool_last_two else None,
            )
        self._episode_returns += rewards
        np.logical_or(terminated, truncated, out=self._pending_reset)
        if self._native_stepper is None:
            self._has_pending_reset = bool(np.any(self._pending_reset))
        if self._collect_derived_signals:
            for lane in range(self.num_envs):
                self._update_signal_row(lane)
        if self._info_frame_stacks is not None:
            self._info_frame_stacks.append(self._info_frame_stack_values())
        return (
            self._returned_obs(observations),
            rewards,
            terminated,
            truncated,
            self._infos(),
        )

    def active_state_indices(self) -> np.ndarray:
        return self._active_state_indices

    def active_enemy_variant_indices(self) -> np.ndarray:
        """Return read-only ``(lane, role)`` scenario variant indices."""
        return self._active_enemy_variant_indices

    def active_enemy_variant_ids(
        self,
    ) -> Mapping[str, tuple[str | None, ...]]:
        """Return role-keyed selected variant ids for every vector lane."""
        return MappingProxyType(
            {
                role: tuple(
                    (
                        self._enemy_variant_by_scenario_index[role_index][
                            int(self._active_enemy_variant_indices[lane, role_index])
                        ].variant_id
                        if self._initialized[lane]
                        and int(self._active_enemy_variant_indices[lane, role_index])
                        in self._enemy_variant_by_scenario_index[role_index]
                        else None
                    )
                    for lane in range(self.num_envs)
                )
                for role_index, role in enumerate(self.enemy_variant_roles)
            }
        )

    def active_surface_variant_indices(self) -> np.ndarray:
        """Return read-only ``(lane, role)`` surface scenario indices."""
        return self._active_surface_variant_indices

    def active_surface_variant_ids(
        self,
    ) -> Mapping[str, tuple[str | None, ...]]:
        """Return role-keyed selected surface ids for every vector lane."""
        return MappingProxyType(
            {
                role: tuple(
                    (
                        self._surface_variant_by_scenario_index[role_index][
                            int(self._active_surface_variant_indices[lane, role_index])
                        ].variant_id
                        if self._initialized[lane]
                        and int(self._active_surface_variant_indices[lane, role_index])
                        in self._surface_variant_by_scenario_index[role_index]
                        else None
                    )
                    for lane in range(self.num_envs)
                )
                for role_index, role in enumerate(self.surface_variant_roles)
            }
        )

    def capture_snapshots(self, mask: np.ndarray) -> tuple[Any | None, ...]:
        if self.closed:
            raise RuntimeError("cannot capture snapshots from a closed environment")
        if not isinstance(mask, np.ndarray):
            raise TypeError("mask must be a NumPy array")
        if mask.shape != (self.num_envs,):
            raise ValueError(f"mask must have shape ({self.num_envs},)")
        if mask.dtype != np.bool_:
            raise TypeError("mask must have dtype np.bool_")
        if not np.any(mask):
            raise ValueError("mask must select at least one lane")
        if not np.all(self._initialized[mask]):
            raise RuntimeError("cannot capture a lane before its initial reset")
        if np.any(self._pending_reset[mask]):
            raise RuntimeError("cannot capture a terminal lane")
        result: list[_LiveSnapshot | None] = [None] * self.num_envs
        for selected_lane in np.flatnonzero(mask):
            lane = int(selected_lane)
            self._sync_native_rgb(lane)
            history = np.asarray(self._action_history.lane(lane), dtype=np.float64).reshape(
                (-1, len(self.buttons))
            )
            result[lane] = _LiveSnapshot(
                owner=self._owner,
                config_hash=self._config_hash,
                origin=self._episode_origins[lane],
                action_history=history,
                stack=self._stack[lane].copy(),
                stack_head=int(self._stack_heads[lane]),
                raw_frame=self._raw_frames[lane].copy(),
                rng_state=copy.deepcopy(self._rngs[lane].bit_generator.state),
                enemy_variant_rng_states=tuple(
                    copy.deepcopy(self._enemy_variant_rngs[role_index][lane].bit_generator.state)
                    for role_index in range(len(self.enemy_variant_roles))
                ),
                surface_variant_rng_states=tuple(
                    copy.deepcopy(self._surface_variant_rngs[role_index][lane].bit_generator.state)
                    for role_index in range(len(self.surface_variant_roles))
                ),
                last_action=self._last_actions[lane].copy(),
                state_index=int(self._active_state_indices[lane]),
                episode_return=float(self._episode_returns[lane]),
                info_frame_stacks=(
                    self._info_frame_stacks.capture_lane(lane)
                    if self._info_frame_stacks is not None
                    else ()
                ),
            )
        return tuple(result)

    def render_lane(self, lane: int) -> np.ndarray | None:
        if self.closed:
            raise RuntimeError("cannot render a closed environment")
        if isinstance(lane, (bool, np.bool_)):
            raise TypeError("lane must be an integer")
        lane_index = operator.index(lane)
        if not 0 <= lane_index < self.num_envs:
            raise IndexError(f"lane must be in [0, {self.num_envs - 1}]")
        if self.render_mode != "rgb_array":
            return None
        self._sync_native_rgb(lane_index)
        return self._raw_frames[lane_index].copy()

    def render(self):
        return self.render_lane(0)

    def get_images(self) -> list[np.ndarray | None]:
        if self.render_mode != "rgb_array":
            return [None for _ in range(self.num_envs)]
        for lane in range(self.num_envs):
            self._sync_native_rgb(lane)
        return [frame.copy() for frame in self._raw_frames]

    def close(self) -> None:
        if getattr(self, "closed", True):
            return
        self.closed = True
        pool = getattr(self, "_pool", None)
        games = getattr(self, "_games", ())
        if pool is not None:
            try:
                pool.run([(game.close, ()) for game in games])
            except Exception:
                pass
            pool.close()
        tempdir = getattr(self, "_tempdir", None)
        if tempdir is not None:
            tempdir.cleanup()


VizDoomTurboVecEnv = VizdoomTurboVecEnv

__all__ = ["VizDoomTurboVecEnv", "VizdoomTurboVecEnv", "scenario_buttons"]
