# Changelog

## Unreleased

- Nothing yet.

## 1.3.0.post26 - 2026-08-13

- Migrate `VizdoomTurboVecEnv` to the breaking Turbo Vector API v2 common
  constructor with a required `game`, resolved NumPy transport, exact
  capabilities, and portable signal schema.
- Standardize reset infos to numeric dtypes and expose appearance IDs only
  through immutable catalogs and active-ID lookup methods; transition infos
  now contain numeric variant indices only.
- Remove the catch-all constructor argument path so unknown options receive
  Python's normal `TypeError`.

## 1.3.0.post25 - 2026-08-13

- Add the vector-only Gymnasium factory `vizdoom_turbo:Vizdoom-Turbo-v0`, with
  an explicit `game` argument and the native vector environment as its result.
- Make Turbo vector rendering opt-in with `render_mode="rgb_array"`; the
  default `None` mode performs no RGB synchronization and returns lane-aligned
  `None` values from `get_images()`.
- Validate Stable integration compatibility values instead of silently
  discarding `inttype`.
- Remove unused reset bookkeeping and an unreachable native per-lane job graph
  while retaining the active native batch buffers and episode semantics.

## 1.3.0.post24 - 2026-08-10

- Preserve native Doom failure diagnostics.

## 1.3.0.post23 - 2026-08-06

- Add opt-in frame-stack-aligned histories for selected info signals while
  preserving existing current-transition values and masks.

## 1.3.0.post22 - 2026-08-04

- Keep crop removal and masking on the indexed native preprocessing path,
  including the classic enabled-HUD mask profile.
- Document the shared `(top, bottom, left, right)` crop-coordinate contract.

## 1.3.0.post21 - 2026-08-02

- Optimize turbo vector environment throughput.

## 1.3.0.post20 - 2026-07-31

- Support single-tic native batch steps.

## 1.3.0.post19 - 2026-07-31

- Optimize RLab environment throughput.

## 1.3.0.post18 - 2026-07-29

- Optimize rendered RLab throughput.
- Fix Turbo Python lint.

## 1.3.0.post17 - 2026-07-29

- Improve ViZDoom surface texture generation.

## 1.3.0.post16 - 2026-07-29

- Add `VizdoomBasic-Plus-v1` with seeded per-lane sampling across reusable
  target appearances and coordinated wall/floor/ceiling texture sets.

## 1.3.0.post15 - 2026-07-29

- Add `VizdoomDefendLine-Plus-v1` with seeded per-lane enemy and surface
  appearance variants, provenance-tracked reusable assets, and unchanged
  gameplay contracts.
- Remove the redundant `state_dir` environment constructor option; saved-game
  starts continue to accept direct file paths or byte payloads.

## 1.3.0.post14 - 2026-07-29

- Optimize batched environment throughput.
- Fix Turbo Python import ordering.

## 1.3.0.post13 - 2026-07-28

- Add Turbo dependency validation CI.

## 1.3.0.post12 - 2026-07-28

- +1.5x throughput improvement.
- Merge branch 'turbo' of github.com:tsilva/ViZDoom-turbo into turbo.
- Fix Python import formatting.

## 1.3.0.post11 - 2026-07-28

- Limit release wheels to macOS ARM64 and Linux x86-64.

## 1.3.0.post10 - 2026-07-28

- Target CPython 3.14 and emit one turbo wheel per release platform.
- Bundle SDL3 in macOS wheels for Homebrew's SDL2 compatibility runtime.

## 1.3.0.post9 - 2026-07-28

- Vendor macOS wheel dependencies.

## 1.3.0.post8 - 2026-07-27

- Harden release builds and parallel startup.

## 1.3.0.post7 - 2026-07-27

- Guard cleanup during partial ViZDoom startup.

## 1.3.0.post6 - 2026-07-27

- Load build backend tests with package path.

## 1.3.0.post5 - 2026-07-27

- Checkout release build submodules.

## 1.3.0.post4 - 2026-07-27

- Install native dependencies in release jobs.

## 1.3.0.post3 - 2026-07-27

- Refresh Turbo API documentation.
- Optimize 32-lane ViZDoom throughput.
- Add iterative SPS optimization skill.
- Format Rust preprocessing implementation.
- Address Rust 1.95 Clippy warnings.
- Keep custom core available in editable installs.

## 1.3.0.post2 - 2026-07-27

- Add the immutable Turbo Vector API v1 declaration for capabilities, signals,
  action semantics, observation ownership, state catalogs, and per-lane RGB
  rendering.
- Remove legacy state-name properties and reset selectors in favor of
  `state_catalog`, `active_state_indices()`, and `state_indices`.

## 1.3.0.post1 - 2026-07-27

- Align turbo release versions with upstream ViZDoom using PEP 440 post releases.

## 0.1.3 - 2026-07-27

- Optimize native vector preprocessing throughput.

## 0.1.2 - 2026-07-26

- Move `vizdoom-turbo` into the `turbo` branch of the ViZDoom fork.
- Namespace package release tags separately from upstream ViZDoom tags.
- Exclude generated Python cache files and build artifacts from release distributions.

## 0.1.1 - 2026-07-26

- Install ViZDoom's source-build dependencies for Intel macOS release wheels.
- Refresh the Rust lockfile while preparing version bumps.

## 0.1.0 - 2026-07-26

- Add native-vector Gymnasium environments with concurrent ViZDoom lanes,
  deterministic masked resets, and batched Rust preprocessing.
