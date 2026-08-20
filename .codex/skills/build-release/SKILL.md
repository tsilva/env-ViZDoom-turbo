---
name: build-release
description: Launch and monitor an env-vizdoom-turbo release. Use when the user asks to build, tag, publish, or cut a PyPI release; requests a specific version; invokes /build-release; or asks whether an env-vizdoom-turbo release is live.
---

# Build Release

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
passing local checks, and a valid changelog. It commits the release metadata,
tags `env-vizdoom-turbo-v<version>`, and atomically pushes the branch and tag.

The tag workflow builds and audits CPython 3.14 wheels for exactly
`macos-arm64` and `linux-x86_64`, plus a source distribution. It publishes with
PyPI trusted publishing and creates a GitHub Release. Never print, commit, or
pass PyPI credentials on a command line.

## Flow

1. Confirm the worktree is clean and synchronized:

```bash
git status --short --branch
git log --oneline --decorate @{u}..HEAD
```

Stop on dirty or unpublished work. Do not clean, commit, pull, or switch
branches unless the user asked.

2. Prepare the frozen release environment and launch the default release from
the repository root:

```bash
UV_CACHE_DIR=turbo/.uv-cache uv sync --project turbo --frozen --all-extras --group release
turbo/scripts/release.py
```

For an exact upstream-based version, invoke the same script with:

```bash
turbo/scripts/release.py --to <version>.post<N>
```

The version base must match the exact `vizdoom` dependency in
`turbo/pyproject.toml`. If preparation fails, report the exact gate and stop.

3. Capture the printed tag, resolve its commit, and monitor the matching
GitHub Actions run:

```bash
release_sha="$(git rev-list -n 1 env-vizdoom-turbo-v<version>)"
gh run list --workflow release.yml --commit "$release_sha" --limit 5 \
  --json databaseId,status,conclusion,event,headBranch,headSha,displayTitle,url
gh run watch <run-id> --exit-status
```

If the commit-filtered query is empty, list the latest release runs and select
the tag-push run. A workflow-dispatch run validates artifacts but does not
publish.

4. After the workflow succeeds, poll PyPI until files exist for the exact
version:

```bash
turbo/.venv/bin/python - <version> <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request

package = "env-vizdoom-turbo"
version = sys.argv[1]
url = f"https://pypi.org/pypi/{package}/json"
for attempt in range(60):
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            files = json.load(response).get("releases", {}).get(version, [])
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        files = []
    if files:
        print(f"https://pypi.org/project/{package}/{version}/")
        for file in files:
            print(file["filename"])
        break
    print(f"waiting for {package} {version} ({attempt + 1}/60)")
    time.sleep(20)
else:
    raise SystemExit(f"{package} {version} did not appear on PyPI")
PY
```

5. If the workflow fails, inspect only failed logs:

```bash
gh run view <run-id> --log-failed
```

Do not report success until PyPI returns files for the version.

## Final Response

Lead with the PyPI version URL. Report the tag, release workflow URL and
conclusion, GitHub Release URL, and every published distribution filename. On
failure, report the exact command, job, or publishing gate and the next safe
recovery action.
