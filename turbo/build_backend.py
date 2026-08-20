"""PEP 517 backend that bundles the repository's custom ViZDoom core."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import maturin

from scripts.stage_vizdoom_core import build_and_stage, clean, staged_vizdoom_core


def build_wheel(
    wheel_directory: str,
    config_settings: Mapping[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    with staged_vizdoom_core(profile_guided=True):
        return maturin.build_wheel(
            wheel_directory,
            config_settings,
            metadata_directory,
        )


def build_editable(
    wheel_directory: str,
    config_settings: Mapping[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    build_and_stage()
    try:
        return maturin.build_editable(
            wheel_directory,
            config_settings,
            metadata_directory,
        )
    except BaseException:
        clean()
        raise


def build_sdist(
    sdist_directory: str,
    config_settings: Mapping[str, Any] | None = None,
) -> str:
    raise RuntimeError(
        "env-vizdoom-turbo source distributions cannot contain portable native "
        "ViZDoom binaries; build a wheel from the complete repository checkout"
    )


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: Mapping[str, Any] | None = None,
) -> str:
    return maturin.prepare_metadata_for_build_wheel(
        metadata_directory,
        config_settings,
    )


def get_requires_for_build_wheel(
    config_settings: Mapping[str, Any] | None = None,
) -> list[str]:
    return maturin.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_editable(
    config_settings: Mapping[str, Any] | None = None,
) -> list[str]:
    return maturin.get_requires_for_build_editable(config_settings)
