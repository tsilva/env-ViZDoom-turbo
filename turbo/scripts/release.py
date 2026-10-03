#!/usr/bin/env python3
"""Validate, commit, tag, and push an env-vizdoom-turbo release."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
RELEASE_HELPER = REPO_ROOT / ".codex" / "skills" / "build-release" / "scripts" / "release_build.py"
STAGE_VIZDOOM_CORE = PACKAGE_ROOT / "scripts" / "stage_vizdoom_core.py"
PYTHON = Path(sys.executable)
RELEASE_FILES = (
    PACKAGE_ROOT / "pyproject.toml",
    PACKAGE_ROOT / "Cargo.toml",
    PACKAGE_ROOT / "Cargo.lock",
    PACKAGE_ROOT / "uv.lock",
)
VERSION_RE = re.compile(r"^(?P<base>[0-9]+\.[0-9]+\.[0-9]+)(?:\.post(?P<post>[0-9]+))?$")
PACKAGE_NAME = "env-vizdoom-turbo"
TAG_PREFIX = "env-vizdoom-turbo-v"


def run(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(args))
    environment = os.environ.copy() if env is None else env.copy()
    environment["UV_CONFIG_FILE"] = str(PACKAGE_ROOT / "uv-tool.toml")
    return subprocess.run(args, cwd=PACKAGE_ROOT, env=environment, check=True, text=True)


def capture(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=PACKAGE_ROOT, text=True).strip()


def ensure_clean() -> None:
    status = capture(["git", "status", "--short"])
    if status:
        raise SystemExit(f"release tree must be clean before preparation:\n{status}")


def upstream_ref() -> str:
    try:
        return capture(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    except subprocess.CalledProcessError as exc:
        raise SystemExit("current branch must have an upstream before release") from exc


def ensure_synced() -> tuple[str, str]:
    upstream = upstream_ref()
    if "/" not in upstream:
        raise SystemExit(f"unexpected upstream ref: {upstream}")
    remote, branch = upstream.split("/", 1)
    run(["git", "fetch", "--prune", "--tags", remote])
    counts = capture(["git", "rev-list", "--left-right", "--count", f"HEAD...{upstream}"])
    ahead, behind = [int(value) for value in counts.split()]
    if ahead or behind:
        raise SystemExit(
            f"current branch must be synced with {upstream}; ahead={ahead} behind={behind}"
        )
    if branch != "turbo" or capture(["git", "branch", "--show-current"]) != "turbo":
        raise SystemExit("publication requires synchronized turbo")
    return remote, branch


def helper(*args: str) -> None:
    run([str(PYTHON), str(RELEASE_HELPER), *args])


def helper_capture(*args: str) -> str:
    return capture([str(PYTHON), str(RELEASE_HELPER), *args])


def pypi_version_is_unused(version: str) -> bool:
    args = [str(PYTHON), str(RELEASE_HELPER), "check-pypi", "--version", version]
    result = subprocess.run(
        args,
        cwd=PACKAGE_ROOT,
        text=True,
        capture_output=True,
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode == 0:
        return True
    if f"{PACKAGE_NAME}=={version} already exists on PyPI" in output:
        return False
    raise subprocess.CalledProcessError(
        result.returncode,
        args,
        output=result.stdout,
        stderr=result.stderr,
    )


def tag_exists(tag: str) -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", tag],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def project_version() -> str:
    command = (
        "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"
    )
    return capture([str(PYTHON), "-c", command])


def parse_version(version: str) -> tuple[str, int]:
    match = VERSION_RE.fullmatch(version)
    if match is None:
        raise SystemExit(f"unsupported release version: {version!r}")
    return match.group("base"), int(match.group("post") or 0)


def upstream_vizdoom_version() -> str:
    with (PACKAGE_ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)
    tool = metadata.get("tool", {})
    turbo = tool.get("env-vizdoom-turbo", {})
    version = turbo.get("upstream-vizdoom-version")
    if (
        not isinstance(version, str)
        or re.fullmatch(
            r"[0-9]+\.[0-9]+\.[0-9]+",
            version,
        )
        is None
    ):
        raise SystemExit(
            "tool.env-vizdoom-turbo.upstream-vizdoom-version must be MAJOR.MINOR.PATCH"
        )
    return version


def next_post_version(current: str, upstream_base: str) -> str:
    current_base, current_post = parse_version(current)
    parse_version(upstream_base)
    if current_base != upstream_base:
        return f"{upstream_base}.post1"
    return f"{current_base}.post{current_post + 1}"


def target_version(args: argparse.Namespace) -> str:
    current = project_version()
    upstream_base = upstream_vizdoom_version()
    if args.to:
        version = args.to
    else:
        current_tag = f"{TAG_PREFIX}{current}"
        version = current
        current_base, _ = parse_version(current)
        if current_base != upstream_base:
            version = next_post_version(current, upstream_base)
        elif tag_exists(current_tag) or not pypi_version_is_unused(current):
            version = next_post_version(current, upstream_base)
    target_base, _ = parse_version(version)
    if target_base != upstream_base and not args.allow_upstream_base_mismatch:
        raise SystemExit(
            f"target version {version} is based on {target_base}, but the "
            f"pinned upstream ViZDoom release is {upstream_base}; pass "
            "--allow-upstream-base-mismatch to override"
        )
    helper("check-pypi", "--version", version)
    return version


def refresh_locks() -> None:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", ".uv-cache")
    run(["uv", "lock"], env=env)
    run(["cargo", "metadata", "--no-deps"], env=env)
    run(["uv", "lock", "--check"], env=env)
    run(["cargo", "metadata", "--locked", "--no-deps"], env=env)


def run_pytest(env: dict[str, str]) -> None:
    run([str(PYTHON), str(STAGE_VIZDOOM_CORE), "build"], env=env)
    try:
        run([str(PYTHON), "-m", "pytest", "-q"], env=env)
    finally:
        run([str(PYTHON), str(STAGE_VIZDOOM_CORE), "clean"], env=env)


def create_commit_and_tag(version: str) -> str:
    tag = f"{TAG_PREFIX}{version}"
    if tag_exists(tag):
        raise SystemExit(f"tag already exists: {tag}")
    run(
        [
            "git",
            "add",
            "pyproject.toml",
            "Cargo.toml",
            "Cargo.lock",
            "uv.lock",
        ]
    )
    if (
        subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=REPO_ROOT,
        ).returncode
        != 0
    ):
        run(["git", "commit", "-m", f"Release {tag}"])
    run(["git", "tag", "-a", tag, "-m", f"Release {tag}"])
    return tag


def push_release(remote: str, branch: str, tag: str, dry_run: bool) -> None:
    args = ["git", "push", "--atomic", remote, f"HEAD:{branch}", tag]
    if dry_run:
        args.insert(2, "--dry-run")
    run(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--to",
        help="exact upstream-based release version, for example 1.3.0.post2",
    )
    parser.add_argument(
        "--allow-upstream-base-mismatch",
        action="store_true",
        help="allow a target version whose base differs from the pinned ViZDoom release",
    )
    parser.add_argument("--dry-run-push", action="store_true")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="build the pushed turbo commit in Actions without publication",
    )
    args = parser.parse_args()
    if args.validate and (args.to or args.allow_upstream_base_mismatch or args.dry_run_push):
        parser.error("--validate cannot be combined with release preparation options")
    return args


def validate() -> None:
    upstream = upstream_ref()
    remote, _, branch = upstream.partition("/")
    if branch != "turbo":
        raise SystemExit("validation requires a turbo upstream")
    run(["git", "fetch", remote, "turbo"])
    sha = capture(["git", "rev-parse", f"{remote}/turbo"])
    run(["gh", "workflow", "run", "release.yml", "--ref", "turbo", "-f", f"ref={sha}"])
    print(f"validation-sha\t{sha}")


def main() -> None:
    args = parse_args()
    os.chdir(PACKAGE_ROOT)
    if args.validate:
        validate()
        return
    ensure_clean()
    remote, branch = ensure_synced()
    version = target_version(args)
    snapshots = {path: path.read_bytes() for path in RELEASE_FILES}
    try:
        helper("bump-version", "--to", version, "--write")
        refresh_locks()
        helper("check-version", "--version", version)
        run(["git", "diff", "--check"])
        tag = create_commit_and_tag(version)
    except BaseException:
        for path, contents in snapshots.items():
            path.write_bytes(contents)
        subprocess.run(["git", "reset", "--quiet"], cwd=REPO_ROOT, check=False)
        raise
    push_release(remote, branch, tag, args.dry_run_push)
    print()
    print(f"Released {tag}: pushed {branch} and tag to {remote}.")
    print("GitHub Actions will build, audit, and publish the release distributions from the tag.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
