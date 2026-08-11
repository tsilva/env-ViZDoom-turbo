---
name: optimize-sps
description: Iteratively optimize ViZDoom-turbo environment SPS to a user-specified multiplier using Goal Mode, TurboBench evidence bundles, exact parity gates, component profiling, and packaging validation. Use when asked to pursue another throughput multiplier, optimize the immutable TurboBench vizdoom/basic-v1 profile, invoke $optimize-sps, or continue SPS performance work until a measured target or defensible upper bound is reached.
---

# Optimize ViZDoom SPS

Pursue a real environment-throughput improvement without training or model
inference. Treat measurement integrity and exact behavior as hard constraints.

## Parse the invocation

Require a numeric target multiplier greater than `1.0`, such as:

```text
$optimize-sps 1.5x
$optimize-sps target=2
```

Interpret the target relative to the frozen control measured at the beginning of
this invocation. Do not use a historical result as the denominator. Ask for the
target only if it is absent or ambiguous.

Use the immutable TurboBench profile below unless the invocation explicitly
supplies another profile. Never silently weaken a setting to obtain the target.

## Start or continue Goal Mode

1. Call `get_goal` before doing substantial work.
2. If no unfinished matching goal exists, call `create_goal`. The explicit skill
   invocation authorizes creating this goal.
3. Set the objective to achieve the requested median speedup in verified
   TurboBench evidence with exact parity and shippable packaging, or prove with
   measured component bounds why exact semantics prevent it.
4. Do not set a token budget unless the user explicitly supplied one.
5. Continue iterating across goal turns; do not stop merely because one
   experiment fails or the work is difficult.
6. Call `update_goal(status="complete")` only after reaching the target and all
   gates, or after producing the required upper-bound proof.
7. Use `blocked` only according to the Goal Mode repeated-blocker rule, never for
   a performance plateau that still has measurable hypotheses.

## Establish repository context

1. Read and follow the applicable `AGENTS.md`.
2. Use `$specs-author` and read the complete root `SPECS.md`. If the skill is
   unavailable, state that and read `SPECS.md` directly.
3. Inspect the dirty worktree before changing anything. Attribute every existing
   change and preserve user or unrelated work.
4. Inspect the TurboBench profile and provider contract, repository-local contract
   checker, tests, native bindings, Rust processor, build, and wheel paths only
   as needed.
5. Inspect these sibling implementations when available:

```text
/Users/tsilva/repos/tsilva/SuperMarioBros-Nes-turbo
/Users/tsilva/repos/tsilva/breakout-turbo-env
/Users/tsilva/repos/tsilva/stable-retro-turbo
```

If a reference path is unavailable, record that once and continue.

## Hold the benchmark contract fixed

Use TurboBench's immutable `vizdoom/basic-v1` profile without diagnostic shape,
step, load, or correctness overrides. It covers shapes 1, 16, and 32; evaluate
the optimization target against the shape-32 result while preserving the other
shape results as regression evidence. Exclude policy inference, training, and
logging work not inherent to the environment.

Install [TurboBench 1.0.0](https://pypi.org/project/turbobench-cli/1.0.0/):

```text
uv tool install \
  --exclude-newer-package turbobench-cli=2026-08-12T00:00:00Z \
  turbobench-cli==1.0.0
```

Pin the same compatible upstream `vizdoom` version for every control and
candidate bundle:

```text
turbobench doctor vizdoom/basic-v1
turbobench compare vizdoom/basic-v1 \
  --left vizdoom-turbo@checkout:/absolute/path/to/checkout \
  --right vizdoom@<compatible-version> \
  --output /absolute/path/to/result
turbobench verify /absolute/path/to/result
```

Do not use `--quick`, `--force-busy`, `--allow-dirty`, `--steps`, or `--shapes`
for final evidence. Those options produce diagnostic results and cannot support
the final claim.

Record the hostname, revision, dirty-state identity, Python environment, build
mode, affinity, and benchmark command. Keep CPU affinity and other execution
conditions identical between control and candidate.

## Freeze and measure the control

Preserve a runnable, clean start-of-goal control checkout before editing:

- Use an isolated worktree or clone so TurboBench can resolve and snapshot the
  exact control revision.
- If the initial tree is dirty, do not pretend the commit alone represents the
  control. Preserve the relevant dirty state non-destructively and label any
  `--allow-dirty` measurement diagnostic.
- Never stash, reset, discard, overwrite, or commit user work to manufacture a
  control.

Create one verified TurboBench bundle for the frozen control and one for each
plausible candidate, always on the same machine and against the same pinned
upstream provider. TurboBench owns warmup, alternating paired measurement,
sample counts, system-load checks, provenance, and correctness gates. Never
reimplement those mechanisms in this repository.

Report the bundle paths, shape-32 raw samples, median SPS, and median vector-step
latency. Compute the optimization speedup from the turbo side of the two bundles:

```text
vector_step_ms = 1000 * num_envs / SPS
speedup = candidate_turbo_median_SPS / control_turbo_median_SPS
```

Rebaseline if the host, pinned upstream provider, Python runtime, build mode,
dependency set, immutable profile, or execution conditions change. Do not
compare results collected on different machines.

## Profile before selecting an experiment

Measure enough components to identify the active bound:

- Doom ticks and waiting
- framebuffer/state extraction
- native/Python boundary calls
- resize and grayscale conversion
- frame-stack maintenance
- output and info writes
- pool scheduling, barriers, and worker idle time
- unavoidable serial and per-vector overhead

Use wall-clock component timings around production paths. Avoid inferring a
bottleneck solely from source inspection. Calculate Amdahl-style ceilings before
large changes.

Prioritize evidence-backed versions of:

- persistent native worker pools
- one native call per vector step
- caller-owned reusable buffers
- direct or indexed framebuffer paths
- fused per-lane step, preprocess, and output pipelines
- specialized grayscale, area-resize, and frame-stack kernels
- removal of phase barriers, allocations, and redundant copies

Preserve exact RGB-to-area-resize-to-grayscale ordering. Do not substitute
ViZDoom `GRAY8` unless parity proves it bit-exact. Handle dynamic palettes when
using indexed framebuffer data.

## Iterate

For each iteration:

1. State one bottleneck hypothesis and its measured maximum contribution.
2. Choose the smallest implementation that can test it.
3. Change only files required for that experiment.
4. Build only what the experiment requires.
5. Run focused parity checks before spending time on a full benchmark.
6. Use a clearly labeled diagnostic TurboBench run to reject obvious regressions.
7. Create and verify a full TurboBench bundle for plausible wins.
8. Keep a change only when its verified result improves the current best without
   violating semantics, generic behavior, or packaging.
9. Remove rejected experimental code that this goal introduced. Never revert
   pre-existing or concurrent user changes.
10. Update component measurements and select the next hypothesis.

Keep the original frozen control as the target denominator. Also compare each
experiment with the current best to avoid accumulating regressions. Do not leave
a slower optimization enabled by default.

Favor overlap over phase-wide batching when synchronization creates bursty
engine contention. Account for tail latency across 32 lanes, not just aggregate
CPU time.

## Enforce semantic gates

Before accepting a final candidate, demonstrate exact observation, reward,
termination, truncation, and info parity against the legacy fallback across:

- seeded random actions
- early deaths
- timeouts
- repeated masked resets
- palette changes when an indexed path is used

Test maxpool and non-fast-path configurations through the generic implementation.
Add regression coverage for native feature detection, incomplete ABI detection,
and fallback behavior.

Run the repository-required validation, including:

```text
turbo/.venv/bin/python -m pytest -q turbo/tests
make -j
```

Follow `AGENTS.md` if it requires additional suites. Do not claim validation that
was not run.

## Enforce the shipping gate

Do not count a speedup that exists only in an editable root install. Build the
actual turbo wheel and install it into a clean environment without relying on
the repository checkout.

Verify that private/custom ViZDoom core functionality used by the fast path is
bundled or otherwise shipped by the wheel. Preserve the PEP 440 post-release
rule whose base matches the pinned stable upstream ViZDoom version. Confirm
native feature detection and fallback behavior in the clean environment.

## Prove an upper bound when the target is unreachable

Do not stop at "no more ideas." Provide a component-level proof containing:

- measured control and best-candidate component medians
- the removable fraction assigned to every remaining hypothesis
- optimistic zero-cost ceilings for those components
- serial, engine, memory-bandwidth, and exact-semantics floors
- the maximum combined SPS implied by those floors
- evidence that the requested multiplier exceeds that ceiling

Treat this proof as completion only when remaining unmeasured time cannot
plausibly bridge the target gap.

## Report completion

Report:

- target multiplier and exact TurboBench commands
- control and final bundle paths and raw samples
- control and final median SPS
- speedup ratio and vector-step latency
- component timings and inferred bottlenecks
- changes retained and rejected, with measured reasons
- parity, fallback, build, and test results
- wheel contents and clean-install packaging status
- any residual risks

Before finishing, recheck `SPECS.md` and the full conversation for changed
stakeholder intent. Do not commit, push, tag, or release unless the user
explicitly requests it.
