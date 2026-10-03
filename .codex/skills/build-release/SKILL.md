---
name: build-release
description: Launch and monitor an env-vizdoom-turbo release. Use when the user asks to build, tag, publish, or cut a PyPI release; requests a specific version; invokes /build-release; or asks whether an env-vizdoom-turbo release is live.
---

# Build Release

Read and apply the shared `$release-workflow` skill at
`/Users/tsilva/.codex/skills/release-workflow/SKILL.md` before execution.
It owns common preflight, publication safeguards, `$push` integration,
workflow monitoring, verification, and reporting. The rules below are this
project's adapter; they retain its invocation default and required gates.
If the shared skill is unavailable, stop and report the missing dependency.

A bare `$build-release` or `/build-release` invocation requests the full
publication flow. Explicitly local, dry-run, or inspection requests must not
launch `turbo/scripts/release.py`, which commits, tags, and pushes.

Use the repository-owned release path and monitor it until the exact version is
visible on PyPI. Do not manually replay version bumps, tags, builds, or uploads
unless the automated publish job fails and the user explicitly requests
recovery.

The `turbo/` package uses the pinned stable ViZDoom dependency as its release
base and PEP 440 post releases for turbo revisions. For example, turbo releases
against ViZDoom 1.3.0 are `1.3.0.post1`, `1.3.0.post2`, and so on. When the
exact `vizdoom` dependency advances, the next turbo release resets to
`<upstream>.post1`. An untagged, unused version in `turbo/pyproject.toml` is
pending; otherwise the release script selects the next post release.
The release script requires a clean tree synchronized with its upstream, an
unused PyPI version, consistent Python and Rust metadata, locked dependencies,
and metadata checks only. It commits the release metadata,
tags `env-vizdoom-turbo-v<version>`, and atomically pushes the branch and tag.

The tag workflow builds and audits only the two primary CPython 3.14 wheels,
for `macos-arm64` and `linux-x86_64`. For the one-time rename release, it also
builds and audits a metadata-only legacy redirect wheel and source distribution.
It publishes with PyPI trusted publishing and creates a GitHub Release. Never
print, commit, or pass PyPI credentials on a command line.

## Required certification

Before launching a publishing command, confirm the following existing
specification requirements in the repository-owned release path.

The root specification also requires immutable TurboBench parity evidence for
the exact final canonical-host wheel and provider-owned cross-platform
consistency checks. Confirm those release gates in the repository-owned path;
quick or checkout evidence is diagnostic.

## Flow

1. Confirm the worktree is clean and synchronized:

```bash
git status --short --branch
git log --oneline --decorate @{u}..HEAD
```

Stop on dirty or unpublished work. Do not clean, commit, pull, or switch
branches unless the user asked.

2. Launch the metadata-only release operator from
the repository root:

```bash
python3 turbo/scripts/release.py
```

For an exact upstream-based version, invoke the same script with:

```bash
python3 turbo/scripts/release.py --to <version>.post<N>
```

The version base must match the exact `vizdoom` dependency in
`turbo/pyproject.toml`. If preparation fails, report the exact gate and stop.

All source checks, custom-core compilation, Rust builds, platform wheel builds,
installed-wheel smoke, canonical parity, and final audits run only in Actions.
The operator needs Python 3.11+ with uv and Cargo for metadata checks, without
requiring a local virtual environment or compiling any code.

For validation without tags, version changes, publication, or a GradLab update:

```bash
python3 turbo/scripts/release.py --validate
```

It fetches the configured `turbo` upstream and dispatches that exact pushed SHA
on the repository's `turbo` branch. Monitor that SHA, download `release-v<version>`,
audit both existing wheels and parity receipt, and compare downloaded hashes
with the runner log. Dirty local files are excluded. Explicit local diagnosis
remains available through the helpers; normal builds use only Actions.

3. Capture the printed tag and follow shared monitoring.

Follow the shared monitoring and verification procedure for the `release.yml`
tag-push run at the full `env-vizdoom-turbo-v<version>` commit SHA. A `workflow_dispatch` run
validates artifacts but never publishes. Verify PyPI project `env-vizdoom-turbo` and
the GitHub Release for the same tag.

Require the two primary CPython 3.14 wheels for macOS arm64 and Linux x86_64;
include the legacy redirect wheel and sdist only for the one-time rename
release. Allow 60 attempts at 20-second intervals for PyPI visibility, with a
20-second request timeout.

## Update GradLab after successful publication

After the release succeeds and the exact PyPI version and required GitHub
Release artifacts pass external verification, update GradLab to consume the
latest successfully published `env-vizdoom-turbo` version. Complete
this step as part of the full publication flow; local builds, dry runs, and
inspection-only requests do not trigger it.

Read `/Users/tsilva/repos/tsilva/gradlab/AGENTS.md` and its required
specifications before editing. Synchronize GradLab's current branch with its
configured upstream and preserve existing work. Update every matching exact
pin in `pyproject.toml`, including platform-specific project dependencies and
the `train-runtime` dependency group. Use the just-verified release version;
if GradLab already consumes a newer verified publication, do not downgrade it.
Regenerate `uv.lock` with `uv lock --upgrade-package env-vizdoom-turbo`,
preserving unrelated pins, supply-chain constraints, and existing per-package
release-age exceptions. Review the dependency diff, validate lock consistency,
and run GradLab's relevant provider compatibility checks.

Report the GradLab version/pin and lockfile update separately from release
success. If synchronization, resolution, or validation fails, preserve the
published release and report the downstream update as incomplete with its
blocker; do not repeat publication.
