from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def release_script() -> ModuleType:
    return load_module("env_vizdoom_turbo_release", REPO_ROOT / "turbo/scripts/release.py")


@pytest.fixture(scope="module")
def release_build() -> ModuleType:
    return load_module(
        "env_vizdoom_turbo_release_build",
        REPO_ROOT / ".codex/skills/build-release/scripts/release_build.py",
    )


@pytest.fixture(scope="module")
def build_backend() -> ModuleType:
    package_root = str(REPO_ROOT / "turbo")
    sys.path.insert(0, package_root)
    try:
        return load_module(
            "env_vizdoom_turbo_build_backend",
            REPO_ROOT / "turbo/build_backend.py",
        )
    finally:
        sys.path.remove(package_root)


@pytest.mark.parametrize(
    ("current", "upstream", "expected"),
    [
        ("0.1.3", "1.3.0", "1.3.0.post1"),
        ("1.3.0", "1.3.0", "1.3.0.post1"),
        ("1.3.0.post1", "1.3.0", "1.3.0.post2"),
        ("1.3.0.post35", "1.4.0", "1.4.0.post1"),
    ],
)
def test_next_post_version_matches_upstream(
    release_script: ModuleType,
    current: str,
    upstream: str,
    expected: str,
) -> None:
    assert release_script.next_post_version(current, upstream) == expected


def test_pinned_upstream_version_is_release_base(
    release_script: ModuleType,
    release_build: ModuleType,
) -> None:
    assert release_script.upstream_vizdoom_version() == "1.3.0"
    assert release_build.upstream_vizdoom_version() == "1.3.0"
    assert release_build.parse_version(release_build.project_version())[0] == "1.3.0"
    assert release_build.cargo_version() == "1.3.0"


def test_custom_core_is_bundled_instead_of_a_runtime_dependency(
    release_build: ModuleType,
) -> None:
    metadata = release_build.read_toml(REPO_ROOT / "turbo/pyproject.toml")
    project = metadata["project"]
    assert isinstance(project, dict)
    dependencies = project["dependencies"]
    assert isinstance(dependencies, list)
    assert not any(
        isinstance(dependency, str) and dependency.startswith("vizdoom")
        for dependency in dependencies
    )
    assert metadata["build-system"]["build-backend"] == "build_backend"
    assert "delocate>=0.13,<1" in metadata["dependency-groups"]["release"]
    assert "setuptools>=65" in metadata["dependency-groups"]["release"]


def test_release_workflow_keeps_primary_package_binary_only() -> None:
    workflow = (REPO_ROOT / ".github/workflows/release.yml").read_text(
        encoding="utf-8"
    )

    assert "uv build --sdist" not in workflow
    assert "audit-sdist" not in workflow
    assert "uv run twine check dist/*" in workflow
    assert "packages-dir: turbo/publish/primary" in workflow


@pytest.mark.parametrize("package", ("vizdoom", "env_vizdoom_turbo"))
def test_macos_wheel_repair_accepts_delocates_distribution_derived_layout(
    release_build: ModuleType,
    tmp_path: Path,
    package: str,
) -> None:
    expected = tmp_path / package / ".dylibs"
    expected.mkdir(parents=True)

    assert release_build.wheel_dylib_directory(tmp_path) == expected

    helper = (
        REPO_ROOT
        / ".codex"
        / "skills"
        / "build-release"
        / "scripts"
        / "release_build.py"
    ).read_text(encoding="utf-8")

    assert 'rglob(".dylibs")' in helper
    assert 'name.endswith("/.dylibs/libSDL3.dylib")' in helper


def test_editable_build_keeps_staged_custom_core(
    build_backend: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        build_backend,
        "build_and_stage",
        lambda: events.append("stage"),
    )
    monkeypatch.setattr(build_backend, "clean", lambda: events.append("clean"))

    def build_editable(*args: object) -> str:
        events.append("build")
        return "env_vizdoom_turbo-editable.whl"

    monkeypatch.setattr(build_backend.maturin, "build_editable", build_editable)

    assert (
        build_backend.build_editable(str(tmp_path))
        == "env_vizdoom_turbo-editable.whl"
    )
    assert events == ["stage", "build"]


def test_failed_editable_build_cleans_staged_custom_core(
    build_backend: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        build_backend,
        "build_and_stage",
        lambda: events.append("stage"),
    )
    monkeypatch.setattr(build_backend, "clean", lambda: events.append("clean"))

    def fail_build(*args: object) -> str:
        events.append("build")
        raise RuntimeError("editable build failed")

    monkeypatch.setattr(build_backend.maturin, "build_editable", fail_build)

    with pytest.raises(RuntimeError, match="editable build failed"):
        build_backend.build_editable(str(tmp_path))
    assert events == ["stage", "build", "clean"]


def test_release_pytest_stages_and_cleans_custom_core(
    release_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    environment = {"UV_CACHE_DIR": ".uv-cache"}
    pytest_command = [
        str(release_script.PYTHON),
        "-m",
        "pytest",
        "-q",
    ]

    def fail_pytest(
        args: list[str],
        *,
        env: dict[str, str] | None = None,
    ) -> None:
        assert env is environment
        commands.append(args)
        if args == pytest_command:
            raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(release_script, "run", fail_pytest)

    with pytest.raises(subprocess.CalledProcessError):
        release_script.run_pytest(environment)

    assert commands == [
        [
            str(release_script.PYTHON),
            str(release_script.STAGE_VIZDOOM_CORE),
            "build",
        ],
        pytest_command,
        [
            str(release_script.PYTHON),
            str(release_script.STAGE_VIZDOOM_CORE),
            "clean",
        ],
    ]


def test_release_targets_only_cpython_314(
    release_build: ModuleType,
) -> None:
    metadata = release_build.read_toml(REPO_ROOT / "turbo/pyproject.toml")
    project = metadata["project"]
    assert isinstance(project, dict)
    assert release_build.RELEASE_PYTHON_TAG == "cp314"
    assert project["requires-python"] == release_build.RELEASE_REQUIRES_PYTHON
    classifiers = project["classifiers"]
    assert isinstance(classifiers, list)
    assert {
        classifier
        for classifier in classifiers
        if isinstance(classifier, str)
        and classifier.startswith("Programming Language :: Python")
    } == {
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.14",
    }
    workflow = (REPO_ROOT / ".github/workflows/release.yml").read_text()
    assert "${{ matrix.python-version }}" not in workflow
    assert "python-version: ${{ env.PYTHON_VERSION }}" in workflow
    assert release_build.RELEASE_PLATFORMS == (
        "macos-arm64",
        "linux-x86_64",
    )
    assert "macos-x86_64" not in workflow
    assert "linux-aarch64" not in workflow
