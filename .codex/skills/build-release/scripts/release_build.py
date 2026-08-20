#!/usr/bin/env python3
"""Build and audit env-vizdoom-turbo release distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_ROOT = REPO_ROOT / "turbo"
PACKAGE_NAME = "env-vizdoom-turbo"
CARGO_PACKAGE_NAME = "env-vizdoom-turbo"
IMPORT_NAME = "env_vizdoom_turbo"
EXTENSION_NAME = "_env_vizdoom_turbo"
RELEASE_PLATFORMS = (
    "macos-arm64",
    "linux-x86_64",
)
VERSION_PATTERN = re.compile(
    r"^(?P<base>[0-9]+\.[0-9]+\.[0-9]+)(?:\.post(?P<post>[0-9]+))?$"
)
RELEASE_PYTHON_TAG = "cp314"
RELEASE_REQUIRES_PYTHON = ">=3.14,<3.15"


def run(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path = PACKAGE_ROOT,
    timeout: float | None = None,
) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, env=env, check=True, timeout=timeout)


def read_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def project_version() -> str:
    project = read_toml(PACKAGE_ROOT / "pyproject.toml")["project"]
    assert isinstance(project, dict)
    version = project["version"]
    assert isinstance(version, str)
    return version


def cargo_version() -> str:
    package = read_toml(PACKAGE_ROOT / "Cargo.toml")["package"]
    assert isinstance(package, dict)
    version = package["version"]
    assert isinstance(version, str)
    return version


def cargo_lock_version() -> str:
    lock = read_toml(PACKAGE_ROOT / "Cargo.lock")
    packages = lock.get("package", [])
    assert isinstance(packages, list)
    for package in packages:
        if isinstance(package, dict) and package.get("name") == CARGO_PACKAGE_NAME:
            version = package.get("version")
            assert isinstance(version, str)
            return version
    raise SystemExit(f"{CARGO_PACKAGE_NAME!r} is missing from Cargo.lock")


def parse_version(version: str) -> tuple[str, int]:
    match = VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise SystemExit(
            "release version must be MAJOR.MINOR.PATCH or "
            f"MAJOR.MINOR.PATCH.postN: {version!r}"
        )
    return match.group("base"), int(match.group("post") or 0)


def upstream_vizdoom_version() -> str:
    metadata = read_toml(PACKAGE_ROOT / "pyproject.toml")
    tool = metadata.get("tool", {})
    assert isinstance(tool, dict)
    turbo = tool.get("env-vizdoom-turbo", {})
    assert isinstance(turbo, dict)
    version = turbo.get("upstream-vizdoom-version")
    if not isinstance(version, str) or re.fullmatch(
        r"[0-9]+\.[0-9]+\.[0-9]+",
        version,
    ) is None:
        raise SystemExit(
            "tool.env-vizdoom-turbo.upstream-vizdoom-version must be "
            "MAJOR.MINOR.PATCH"
        )
    return version


def next_post_version(version: str) -> str:
    base, post = parse_version(version)
    return f"{base}.post{post + 1}"


def check_version(args: argparse.Namespace) -> None:
    expected = args.version or project_version()
    expected_base, _ = parse_version(expected)
    upstream = upstream_vizdoom_version()
    project = read_toml(PACKAGE_ROOT / "pyproject.toml")["project"]
    assert isinstance(project, dict)
    actual = {
        "project.name": project.get("name"),
        "project.requires-python": project.get("requires-python"),
        "pyproject.toml": project_version(),
        "Cargo.toml": cargo_version(),
        "Cargo.lock": cargo_lock_version(),
        "upstream ViZDoom": upstream,
    }
    wanted = {
        "project.name": PACKAGE_NAME,
        "project.requires-python": RELEASE_REQUIRES_PYTHON,
        "pyproject.toml": expected,
        "Cargo.toml": expected_base,
        "Cargo.lock": expected_base,
        "upstream ViZDoom": expected_base,
    }
    failures = {
        key: value
        for key, value in actual.items()
        if value != wanted[key]
    }
    if failures:
        raise SystemExit(
            f"release metadata mismatch for {expected}: "
            + ", ".join(
                f"{key}={value!r}, expected {wanted[key]!r}"
                for key, value in failures.items()
            )
        )
    print(
        json.dumps(
            {
                "package": PACKAGE_NAME,
                "version": expected,
                "upstream_vizdoom": upstream,
            },
            indent=2,
        )
    )


def replace_section_version(path: Path, section: str, version: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf'(?ms)(^\[{re.escape(section)}\]\n.*?^version\s*=\s*")[^"]+(")',
    )
    updated, count = pattern.subn(rf"\g<1>{version}\g<2>", text, count=1)
    if count != 1:
        raise SystemExit(f"could not update [{section}] version in {path.name}")
    path.write_text(updated, encoding="utf-8")


def bump_version(args: argparse.Namespace) -> None:
    current = project_version()
    target = args.to or next_post_version(current)
    target_base, _ = parse_version(target)
    if args.write:
        replace_section_version(PACKAGE_ROOT / "pyproject.toml", "project", target)
        replace_section_version(
            PACKAGE_ROOT / "Cargo.toml",
            "package",
            target_base,
        )
    print(target)


def fetch_pypi(package: str = PACKAGE_NAME) -> dict[str, object]:
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {}
        raise
    if not isinstance(data, dict):
        raise SystemExit("unexpected PyPI JSON response")
    return data


def check_pypi(args: argparse.Namespace) -> None:
    parse_version(args.version)
    package = args.package or PACKAGE_NAME
    releases = fetch_pypi(package).get("releases", {})
    if not isinstance(releases, dict):
        raise SystemExit("unexpected PyPI releases payload")
    if releases.get(args.version):
        raise SystemExit(f"{package}=={args.version} already exists on PyPI")
    print(f"{package}=={args.version} is unused on PyPI")


def wheelhouse(version: str, platform: str) -> Path:
    return PACKAGE_ROOT / f"wheelhouse-v{version}-{platform}"


def macos_sdl3_runtime() -> Path:
    try:
        prefix = subprocess.check_output(
            ["brew", "--prefix", "sdl3"],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit("Homebrew SDL3 is required for macOS release wheels") from exc
    runtime = Path(prefix) / "lib" / "libSDL3.dylib"
    if not runtime.is_file():
        raise SystemExit(f"Homebrew SDL3 runtime is missing: {runtime}")
    return runtime


def wheel_dylib_directory(unpacked_root: Path) -> Path:
    directories = sorted(
        path for path in unpacked_root.rglob(".dylibs") if path.is_dir()
    )
    if len(directories) != 1:
        raise SystemExit(
            "delocated wheel must contain exactly one dylib directory; "
            f"found {directories}"
        )
    return directories[0]


def bundle_macos_sdl3(wheel: Path, runtime: Path) -> Path:
    with tempfile.TemporaryDirectory(
        prefix="env-vizdoom-turbo-sdl3-"
    ) as directory:
        work = Path(directory)
        unpacked_output = work / "unpacked"
        repacked_output = work / "repacked"
        run(
            [
                sys.executable,
                "-m",
                "wheel",
                "unpack",
                "--dest",
                str(unpacked_output),
                str(wheel),
            ]
        )
        unpacked = [path for path in unpacked_output.iterdir() if path.is_dir()]
        if len(unpacked) != 1:
            raise SystemExit(
                f"expected one unpacked wheel in {unpacked_output}, found {len(unpacked)}"
            )
        dylibs = wheel_dylib_directory(unpacked[0])
        shutil.copy2(runtime, dylibs / "libSDL3.dylib")
        repacked_output.mkdir()
        run(
            [
                sys.executable,
                "-m",
                "wheel",
                "pack",
                "--dest-dir",
                str(repacked_output),
                str(unpacked[0]),
            ]
        )
        repacked = resolve_wheel(repacked_output)
        bundled = wheel.parent / f".{wheel.name}.sdl3"
        shutil.copy2(repacked, bundled)
    return bundled


def build_platform(args: argparse.Namespace) -> None:
    version = args.version or project_version()
    parse_version(version)
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if python_tag != RELEASE_PYTHON_TAG:
        raise SystemExit(
            f"release builds require {RELEASE_PYTHON_TAG}, found {python_tag}"
        )
    output = wheelhouse(version, args.platform)
    if output.exists():
        raise SystemExit(f"release output already exists: {output}")
    output.mkdir()
    env = os.environ.copy()
    target = PACKAGE_ROOT / f"target-release-{args.platform}"
    env["CARGO_TARGET_DIR"] = str(target)
    if args.platform.startswith("macos-"):
        arch = args.platform.removeprefix("macos-")
        env["ARCHFLAGS"] = f"-arch {arch}"
        env["MACOSX_DEPLOYMENT_TARGET"] = "11.0" if arch == "arm64" else "10.15"
        sdl3_runtime = macos_sdl3_runtime()
        try:
            run([sys.executable, "scripts/stage_vizdoom_core.py", "build"], env=env)
            run(
                [
                    sys.executable,
                    "-m",
                    "maturin",
                    "build",
                    "--release",
                    "--locked",
                    "--interpreter",
                    sys.executable,
                    "--out",
                    str(output),
                ],
                env=env,
            )
            wheel = resolve_wheel(output)
            with tempfile.TemporaryDirectory(
                prefix="env-vizdoom-turbo-delocate-"
            ) as directory:
                repaired_output = Path(directory)
                delocate_env = env.copy()
                macos_version = platform.mac_ver()[0]
                if not macos_version:
                    raise SystemExit("could not determine the macOS runner version")
                delocate_env["MACOSX_DEPLOYMENT_TARGET"] = (
                    f"{macos_version.split('.', maxsplit=1)[0]}.0"
                )
                run(
                    [
                        sys.executable,
                        "-m",
                        "delocate.cmd.delocate_wheel",
                        "--require-archs",
                        arch,
                        "-w",
                        str(repaired_output),
                        "-v",
                        str(wheel),
                    ],
                    env=delocate_env,
                )
                repaired_wheel = resolve_wheel(repaired_output)
                bundled_wheel = bundle_macos_sdl3(repaired_wheel, sdl3_runtime)
                wheel.unlink()
                bundled_wheel.replace(output / repaired_wheel.name)
        finally:
            run([sys.executable, "scripts/stage_vizdoom_core.py", "clean"])
        return
    arch = args.platform.removeprefix("linux-")
    env.update(
        {
            "CIBW_ARCHS_LINUX": arch,
            "CIBW_BEFORE_ALL_LINUX": (
                "yum install -y cmake git boost-devel SDL2-devel "
                "openal-soft-devel && curl https://sh.rustup.rs -sSf | "
                "sh -s -- -y --profile minimal --default-toolchain stable"
            ),
            "CIBW_BUILD": f"{RELEASE_PYTHON_TAG}-manylinux_{arch}",
            "CIBW_BUILD_VERBOSITY": "1",
            "CIBW_ENVIRONMENT_LINUX": (
                'PATH="$HOME/.cargo/bin:$PATH" '
                "CARGO_NET_GIT_FETCH_WITH_CLI=true "
                "CARGO_TARGET_DIR=/tmp/cargo-target"
            ),
            "CIBW_SKIP": "*-musllinux_*",
        }
    )
    run(
        [
            sys.executable,
            "-m",
            "cibuildwheel",
            "turbo",
            "--platform",
            "linux",
            "--output-dir",
            str(output),
        ],
        env=env,
        cwd=REPO_ROOT,
    )


def resolve_wheel(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        return path
    wheels = sorted(path.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one wheel in {path}, found {len(wheels)}")
    return wheels[0]


def wheel_platform(wheel: Path) -> str | None:
    markers = {
        "macos-arm64": ("macosx", "arm64"),
        "linux-x86_64": ("manylinux", "x86_64"),
    }
    for platform_name, required in markers.items():
        if all(marker in wheel.name for marker in required):
            return platform_name
    return None


def wheel_python_tag(wheel: Path) -> str | None:
    if f"-{RELEASE_PYTHON_TAG}-{RELEASE_PYTHON_TAG}-" in wheel.name:
        return RELEASE_PYTHON_TAG
    return None


def audit_wheel(wheel: Path, version: str) -> dict[str, object]:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_name = next(
            (name for name in names if name.endswith(".dist-info/METADATA")),
            None,
        )
        metadata = (
            archive.read(metadata_name).decode("utf-8")
            if metadata_name is not None
            else ""
        )
    extension = [
        name
        for name in names
        if name.startswith(f"{IMPORT_NAME}/{EXTENSION_NAME}")
        and name.endswith((".so", ".pyd"))
    ]
    core_extension = [
        name
        for name in names
        if name.startswith("vizdoom/vizdoom.")
        and name.endswith((".so", ".pyd"))
    ]
    checks = {
        "version_in_filename": version in wheel.name,
        "release_python_abi": wheel_python_tag(wheel) == RELEASE_PYTHON_TAG,
        "known_platform": wheel_platform(wheel) is not None,
        "has_init": f"{IMPORT_NAME}/__init__.py" in names,
        "has_environment": f"{IMPORT_NAME}/env.py" in names,
        "has_action_tables": f"{IMPORT_NAME}/action_tables.py" in names,
        "has_py_typed": f"{IMPORT_NAME}/py.typed" in names,
        "has_extension": len(extension) == 1,
        "has_custom_core_extension": len(core_extension) == 1,
        "has_custom_core_python": "vizdoom/__init__.py" in names,
        "has_custom_core_binary": any(
            name in {"vizdoom/vizdoom", "vizdoom/vizdoom.exe"}
            for name in names
        ),
        "macos_dependencies_vendored": (
            wheel_platform(wheel) != "macos-arm64"
            or any(".dylibs/" in name and name.endswith(".dylib") for name in names)
        ),
        "macos_sdl3_runtime_bundled": (
            wheel_platform(wheel) != "macos-arm64"
            or any(name.endswith("/.dylibs/libSDL3.dylib") for name in names)
        ),
        "no_external_vizdoom_dependency": "Requires-Dist: vizdoom" not in metadata,
        "has_metadata": sum(name.endswith(".dist-info/METADATA") for name in names) == 1,
        "has_license": any(name.endswith(".dist-info/licenses/LICENSE") for name in names),
        "no_cache_files": not any(
            "__pycache__" in Path(name).parts or name.endswith(".pyc") for name in names
        ),
    }
    return {
        "wheel": str(wheel),
        "platform": wheel_platform(wheel),
        "python": wheel_python_tag(wheel),
        "extension": extension,
        "core_extension": core_extension,
        "checks": checks,
    }


def assert_audits(results: list[dict[str, object]]) -> None:
    failures = {}
    for result in results:
        checks = result["checks"]
        assert isinstance(checks, dict)
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            failures[str(result["wheel"])] = failed
    if failures:
        print(json.dumps(results, indent=2), file=sys.stderr)
        raise SystemExit(f"wheel audit failed: {failures}")


def venv_python(root: Path) -> Path:
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def smoke_wheel(args: argparse.Namespace) -> None:
    wheel = resolve_wheel(args.wheel)
    version = args.version or project_version()
    result = audit_wheel(wheel, version)
    assert_audits([result])
    with tempfile.TemporaryDirectory(prefix="env-vizdoom-turbo-smoke-") as directory:
        environment = Path(directory) / "venv"
        run(["uv", "venv", "--python", sys.executable, str(environment)])
        python = venv_python(environment)
        run(["uv", "pip", "install", "--python", str(python), str(wheel)])
        code = """
import numpy as np
from importlib.metadata import version
from env_vizdoom_turbo import EnvViZDoomTurboVecEnv, scenario_buttons

print("smoke: metadata", flush=True)
assert version("env-vizdoom-turbo") == %r
print("smoke: scenario metadata", flush=True)
assert scenario_buttons("VizdoomBasic-v1") == ("MOVE_LEFT", "MOVE_RIGHT", "ATTACK")
print("smoke: construct environment", flush=True)
env = EnvViZDoomTurboVecEnv(
    "VizdoomBasic-v1",
    num_envs=2,
    num_threads=2,
    obs_resize=(32, 40),
    frame_skip=2,
    frame_stack=4,
    use_restricted_actions="minimal",
)
try:
    print("smoke: reset", flush=True)
    observations, _ = env.reset(seed=7)
    assert observations.shape == (2, 4, 32, 40)
    print("smoke: step", flush=True)
    env.step(np.zeros(2, dtype=np.int64))
finally:
    print("smoke: close", flush=True)
    env.close()
print("smoke: complete", flush=True)
""" % version
        run([str(python), "-c", code], timeout=120)
    print(json.dumps(result, indent=2))


def audit_sdist(args: argparse.Namespace) -> None:
    version = args.version or project_version()
    sdist = args.sdist.resolve()
    with tarfile.open(sdist, "r:gz") as archive:
        names = archive.getnames()
    checks = {
        "version_in_filename": version in sdist.name,
        "has_pyproject": any(name.endswith("/pyproject.toml") for name in names),
        "has_cargo_toml": any(name.endswith("/Cargo.toml") for name in names),
        "has_cargo_lock": any(name.endswith("/Cargo.lock") for name in names),
        "has_license": any(name.endswith("/LICENSE") for name in names),
        "has_readme": any(name.endswith("/README.md") for name in names),
        "has_python_package": any(
            name.endswith(f"/{IMPORT_NAME}/env.py") for name in names
        ),
        "has_rust_source": any(name.endswith("/src/lib.rs") for name in names),
        "no_build_outputs": not any(
            part in {"target", "dist", ".venv", ".git"}
            for name in names
            for part in Path(name).parts
        ),
    }
    result = {"sdist": str(sdist), "checks": checks}
    print(json.dumps(result, indent=2))
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"sdist audit failed: {failed}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def final_check(args: argparse.Namespace) -> None:
    version = args.version or project_version()
    wheels = [Path(value).resolve() for value in args.wheels]
    if not wheels:
        raise SystemExit("final-check requires the complete wheel set")
    results = [audit_wheel(wheel, version) for wheel in wheels]
    assert_audits(results)
    seen = {
        (result["platform"], result["python"])
        for result in results
    }
    expected = {
        (platform, RELEASE_PYTHON_TAG) for platform in RELEASE_PLATFORMS
    }
    missing = sorted(expected - seen)
    if missing:
        raise SystemExit(f"release wheel set is missing: {missing}")
    if len(wheels) != len(expected):
        raise SystemExit(f"expected {len(expected)} wheels, found {len(wheels)}")
    run([sys.executable, "-m", "twine", "check", *[str(wheel) for wheel in wheels]])
    print(
        json.dumps(
            {
                "audits": results,
                "sha256": {wheel.name: sha256(wheel) for wheel in wheels},
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check-version")
    check.add_argument("--version")
    check.set_defaults(func=check_version)

    bump = commands.add_parser("bump-version")
    bump.add_argument("--to")
    bump.add_argument("--write", action="store_true")
    bump.set_defaults(func=bump_version)

    pypi = commands.add_parser("check-pypi")
    pypi.add_argument("--version", required=True)
    pypi.add_argument("--package")
    pypi.set_defaults(func=check_pypi)

    build = commands.add_parser("build-platform")
    build.add_argument("--platform", choices=RELEASE_PLATFORMS, required=True)
    build.add_argument("--version")
    build.set_defaults(func=build_platform)

    smoke = commands.add_parser("smoke-wheel")
    smoke.add_argument("wheel", type=Path)
    smoke.add_argument("--version")
    smoke.set_defaults(func=smoke_wheel)

    sdist = commands.add_parser("audit-sdist")
    sdist.add_argument("sdist", type=Path)
    sdist.add_argument("--version")
    sdist.set_defaults(func=audit_sdist)

    final = commands.add_parser("final-check")
    final.add_argument("wheels", nargs="*")
    final.add_argument("--version")
    final.set_defaults(func=final_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
