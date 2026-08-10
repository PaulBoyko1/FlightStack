# AI status and integration contract

## Current status

FlightStack does **not** ship a trained learned policy in this revision.  There
is no `ai/` training package, Gymnasium environment, vectorized training
backend, checkpoint artifact, reward-result table, or learned inference
adapter.  Consequently, the interactive server explicitly declines a
`learned` pilot request instead of silently using the classical baseline.

This is intentional status reporting, not a claim that the classical pilot is
an AI substitute.

## Implemented seam for a future policy

The stable interface is the high-level `Pilot` protocol:

```text
observation/history -> learned pilot -> PilotCommand (CTBR)
                                      |
                                      v
                       rate PID -> mixer -> motor dynamics -> plant
```

`PilotCommand` contains collective thrust in newtons and body-rate targets in
radians per second.  This is the same interface used by `HumanPilot` and
`ClassicalRacePilot`; a learned policy must not bypass motor dynamics or use a
different race judge.  Action scaling and clipping belong at the policy
adapter boundary and must be logged with checkpoint metadata.

The canonical state/frame contract is documented in
[architecture.md](architecture.md) and [frame-conventions.md](frame-conventions.md).
Any scalar-last library or model must use the named quaternion adapter rather
than silently changing the meaning of the state.

## Required work before enabling `LEARNED`

1. Define and test an observation schema using only data available to the
   chosen policy mode, including explicit action/history semantics where used.
2. Implement a Gymnasium-compatible racing environment that calls the
   authoritative vehicle/race contracts or a parity-tested vectorized backend.
3. Choose one primary high-throughput backend: either a thin vectorized JAX
   implementation of FlightStack equations or a clean 5-inch adaptation of a
   vetted backend.  It must share `VehicleConfig` semantics and have focused
   parity scenarios against the reference runtime.
4. Use a maintained PPO implementation rather than a home-grown optimizer;
   record dependency versions, seeds, configuration hashes, reward terms,
   normalization, action scaling, and checkpoint provenance.
5. Evaluate against the classical baseline with paired seeds, report failure
   modes and confidence intervals, and run a documented robustness grid before
   presenting a policy as a result.
6. Add a checkpoint loader that validates its metadata before the server makes
   the learned mode selectable.  Keep the UI notice for absent or incompatible
   checkpoints.

## Source/reuse boundary

The pre-researched [source manifest](research/SOURCE_MANIFEST.md) maps the
relevant pinned LSY Drone Racing, Crazyflow, SimpleFlight, and PPO plumbing
references.  Those projects are implementation references, not vendored
FlightStack code.  The manifest requires an explicit license/reuse update if a
future integration adapts source or adds a model/dataset.

## What a valid AI result must say

A future report may say that a particular checkpoint achieved particular
measured simulator results only when it includes the exact commit/configuration
and evaluation artifact.  It must not call a generic 5-inch configuration a
measured real drone, imply hardware transfer, or compare unequal control
interfaces.  See [experiments.md](experiments.md) for the reporting contract.
