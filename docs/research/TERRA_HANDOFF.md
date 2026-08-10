# Terra handoff — use this before further broad research

A pre-researched source/code pack was added on branch:

`research/flightstack-source-pack`

It exists to save model/research time during the FlightStack digital-twin build.

## Read first

1. `docs/research/SOURCE_MANIFEST.md`
2. `docs/research/REFERENCE_CODE.md`
3. existing `docs/frame-conventions.md`
4. existing `README.md`

## How to consume this pack from another working branch

If your implementation branch is already active, do **not** switch away and discard work.

Fetch the reference branch and either:

```bash
git fetch origin research/flightstack-source-pack
```

then inspect files directly:

```bash
git show origin/research/flightstack-source-pack:docs/research/SOURCE_MANIFEST.md
git show origin/research/flightstack-source-pack:docs/research/REFERENCE_CODE.md
```

or cherry-pick the two documentation commits if keeping the research docs in the final implementation branch is useful.

Do not reset/rebase destructively merely to obtain this material.

## Important correction to the original master prompt

The external research listed in the master prompt was already performed in detail before your run. Do not spend a large portion of the task repeating broad ecosystem research.

Use the manifest as a fast index:

- inspect the exact pinned upstream file when implementing that subsystem;
- use the FlightStack-native code seeds in `REFERENCE_CODE.md` immediately where appropriate;
- run new external research only when a concrete integration problem or a clearly superior implementation path warrants it.

The pack intentionally does **not** vendor entire third-party repositories. Mature infrastructure should remain dependencies/references.

## High-value first reads by subsystem

### Physics / motors

Read:

- Elodin AGP `sim/physics.py`
- Elodin AGP `sim/config.py`
- Crazyflow first-principles dynamics only when deeper rotor/aero behavior is needed
- `REFERENCE_CODE.md` sections 3–8

Do not replace existing FlightStack quaternion semantics.

### Racing / gates

Read:

- LSY `lsy_drone_racing/envs/utils.py`
- LSY `lsy_drone_racing/envs/race_core.py`
- LSY `tests/unit/envs/test_race_core.py`
- `REFERENCE_CODE.md` sections 9–10

Use swept plane crossing, not gate-center distance thresholds.

### Pilot/control boundary

Read:

- AGP `solver/api.py`
- SimpleFlight CTBR examples
- `REFERENCE_CODE.md` sections 2 and 12

Preferred default architecture remains:

```text
human / classical / learned
          ↓
collective thrust + body-rate target
          ↓
same FlightStack low-level controller
          ↓
mixer
          ↓
motors
          ↓
plant
```

You may improve this if evidence supports a better shared interface.

### AI training

Read:

- LSY `control/train_rl.py`
- LSY `control/attitude_rl.py`
- Crazyflow parameter/dynamics implementation
- `REFERENCE_CODE.md` sections 12–17

Do not implement PPO from scratch unless there is a specific reason.

Perform the bounded 5-inch training-backend decision described in `SOURCE_MANIFEST.md`:

- extend/adapt Crazyflow cleanly, or
- use a thin custom vectorized JAX parity backend.

Keep one primary backend.

### Frontend

Use:

- Three.js
- browser Gamepad API
- uPlot where live plots are actually useful

Do not build a renderer/charting system from scratch.

### Collision

Use Rapier/Parry-style collision geometry while keeping FlightStack flight equations authoritative.

## Critical frame warning

FlightStack canonical quaternion:

```text
[w,x,y,z]
body -> world
```

Some upstream simulation code uses:

```text
[x,y,z,w]
```

Keep conversion in explicit tested adapters.

If an upstream source uses NED/FRD or another basis, document and test that conversion too.

## Do not interpret this as a limitation on engineering judgment

The reference pack is intended to remove duplicated work, not prevent better ideas.

If you discover:

- a clearly better permissive implementation;
- a cleaner architecture;
- a superior algorithm;
- a high-value feature that fits the build;

use it when justified and document the material decision.

The requirement is not to follow these references mechanically. The requirement is to avoid wasting effort rediscovering already-vetted baseline solutions.

## Priority reminder

Keep the core vertical slice healthy:

```text
manual flight
→ graphics / real 3D simulator
→ correct physical/control flow
→ racing
→ classical baseline
→ actual AI training/inference
→ statistical robustness experiments
```

Do not stop at a planning/documentation milestone.

Implement, test, benchmark, and push working code.
