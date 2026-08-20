#!/usr/bin/env python3
"""Build and stage the custom ViZDoom package for a Turbo wheel."""

from __future__ import annotations

import argparse
import contextlib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
STAGED_PACKAGE = PACKAGE_ROOT / "python" / "vizdoom"
PREBUILT_CORE_ENV = "ENV_VIZDOOM_TURBO_PREBUILT_CORE"
DISABLE_PGO_ENV = "ENV_VIZDOOM_TURBO_DISABLE_PGO"


def built_package(repository: Path) -> Path:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return repository / "bin" / version / "vizdoom"


def clean() -> None:
    shutil.rmtree(STAGED_PACKAGE, ignore_errors=True)


def copy_core_source(destination: Path) -> None:
    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored = {
            name for name in names if name == "__pycache__" or name.endswith((".pyc", ".pyo"))
        }
        if Path(directory) == REPO_ROOT:
            ignored.update(
                {
                    ".git",
                    ".venv",
                    "CMakeCache.txt",
                    "CMakeFiles",
                    "Makefile",
                    "bin",
                    "build",
                    "dist",
                    "turbo",
                    "wheelhouse",
                }
            )
        return ignored

    shutil.copytree(REPO_ROOT, destination, ignore=ignore)


def _install_core(repository: Path, python: Path, environment: dict[str, str]) -> None:
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--editable",
            str(repository),
            "--no-deps",
            "--no-build-isolation",
        ],
        cwd=repository,
        env=environment,
        check=True,
    )


def _append_environment(environment: dict[str, str], name: str, value: str) -> None:
    current = environment.get(name, "").strip()
    environment[name] = f"{current} {value}".strip()


def _is_gcc(environment: dict[str, str]) -> bool:
    compiler = shlex.split(environment.get("CXX", "c++"))
    try:
        result = subprocess.run(
            [*compiler, "--version"],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
    except OSError, subprocess.CalledProcessError:
        return False
    version = f"{result.stdout}\n{result.stderr}".casefold()
    return "free software foundation" in version or "gcc" in version or "g++" in version


def _pgo_environment(
    environment: dict[str, str],
    profile_directory: Path,
    *,
    generate: bool,
) -> dict[str, str]:
    configured = environment.copy()
    if generate:
        compile_flags = (
            f"-fprofile-generate={profile_directory} -fno-semantic-interposition -fno-plt"
        )
        link_flags = f"-fprofile-generate={profile_directory}"
        ipo = "OFF"
    else:
        compile_flags = (
            f"-fprofile-use={profile_directory} -fprofile-correction "
            "-Wno-missing-profile -Wno-error=coverage-mismatch "
            "-fno-semantic-interposition -fno-plt"
        )
        link_flags = f"-fprofile-use={profile_directory} -fprofile-correction"
        ipo = "ON"
    _append_environment(configured, "CFLAGS", compile_flags)
    _append_environment(configured, "CXXFLAGS", compile_flags)
    _append_environment(configured, "LDFLAGS", link_flags)
    _append_environment(
        configured,
        "VIZDOOM_CMAKE_ARGS",
        f"-DCMAKE_INTERPROCEDURAL_OPTIMIZATION={ipo}",
    )
    return configured


def build_core(repository: Path, *, profile_guided: bool = False) -> None:
    environment = os.environ.copy()
    environment.pop("_PYPROJECT_HOOKS_BUILD_BACKEND", None)
    environment.pop("_PYPROJECT_HOOKS_BACKEND_PATH", None)
    build_environment = repository / ".venv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "venv",
            str(build_environment),
        ],
        cwd=repository,
        env=environment,
        check=True,
    )
    python = (
        build_environment / "Scripts" / "python.exe"
        if os.name == "nt"
        else build_environment / "bin" / "python"
    )
    site_packages = (
        build_environment / "Lib" / "site-packages"
        if os.name == "nt"
        else build_environment
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    outer_site_packages = [
        path for path in sys.path if path and Path(path).name in {"site-packages", "dist-packages"}
    ]
    (site_packages / "env_vizdoom_turbo_build_environment.pth").write_text(
        "".join(f"{path}\n" for path in outer_site_packages),
        encoding="utf-8",
    )
    use_pgo = (
        profile_guided
        and sys.platform.startswith("linux")
        and os.environ.get(DISABLE_PGO_ENV) != "1"
        and _is_gcc(environment)
    )
    if use_pgo:
        profile_directory = repository / ".pgo"
        profile_directory.mkdir()
        generate_environment = _pgo_environment(
            environment,
            profile_directory,
            generate=True,
        )
        _install_core(repository, python, generate_environment)
        subprocess.run(
            [
                str(python),
                str(PACKAGE_ROOT / "scripts" / "train_core_pgo.py"),
            ],
            cwd=repository,
            env=generate_environment,
            check=True,
        )
        use_environment = _pgo_environment(
            environment,
            profile_directory,
            generate=False,
        )
        _install_core(repository, python, use_environment)
    else:
        _install_core(repository, python, environment)
    subprocess.run(
        [
            str(python),
            "-c",
            (
                "import vizdoom; "
                "assert hasattr(vizdoom, '_TurboBatchStepper'), "
                "'custom _TurboBatchStepper is missing'"
            ),
        ],
        cwd=repository,
        env=environment,
        check=True,
    )


def build_and_stage(*, profile_guided: bool = False) -> None:
    clean()
    prebuilt = os.environ.get(PREBUILT_CORE_ENV)
    if prebuilt:
        source = Path(prebuilt).expanduser().resolve()
        required = (source / "__init__.py", source / "vizdoom")
        if not source.is_dir() or any(not path.is_file() for path in required):
            raise RuntimeError(f"{PREBUILT_CORE_ENV} must name a built vizdoom package directory")
        if not any(source.glob("vizdoom*.so")) and not any(source.glob("vizdoom*.pyd")):
            raise RuntimeError(f"{PREBUILT_CORE_ENV} does not contain a Python extension")
        shutil.copytree(
            source,
            STAGED_PACKAGE,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        return
    with tempfile.TemporaryDirectory(prefix="env-vizdoom-turbo-core-") as directory:
        repository = Path(directory) / "ViZDoom"
        copy_core_source(repository)
        build_core(repository, profile_guided=profile_guided)
        source = built_package(repository)
        if not source.is_dir():
            raise RuntimeError(f"custom ViZDoom package was not built: {source}")
        shutil.copytree(
            source,
            STAGED_PACKAGE,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )


@contextlib.contextmanager
def staged_vizdoom_core(*, profile_guided: bool = False) -> Iterator[None]:
    build_and_stage(profile_guided=profile_guided)
    try:
        yield
    finally:
        clean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "clean"))
    args = parser.parse_args()
    if args.command == "build":
        build_and_stage()
    else:
        clean()


if __name__ == "__main__":
    main()
