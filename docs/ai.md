# AI status and integration contract

## Current status

FlightStack now has a real, state-based PPO training and inference path.  It
is optional: install `.[train]` for Gymnasium and Stable-Baselines3, while the
manual simulator and reference tests remain usable without those packages.

The path is deliberately narrow:

```text
27-value state observation -> Stable-Baselines3 PPO -> normalized 4-action
                                                        |
                                                        v
                                      LearnedPolicyPilot -> CTBR PilotCommand
                                                        |
                                                        v
                                  rate PID -> mixer -> motors -> 6DOF/race
```

`LearnedPolicyPilot` does not emit raw motor thrust.  It clips a normalized
`[thrust, roll-rate, pitch-rate, yaw-rate]` action to `[-1, 1]`, maps zero
thrust to physical hover, and scales rate commands by the same vehicle limits
used by Human and Classical pilots.  This preserves one actuator/control seam
for all three pilots.  When a 2 ms interactive or headless runtime calls the
pilot, it holds each inferred command for the training-configured 20 ms control
period before requesting the next action.  This preserves the 50 Hz
train/deploy decision rate instead of querying the policy ten times faster.

## Environment and contracts

`FlightStackRaceEnv` is a NumPy-first race environment whose `step()` calls the
existing `FixedStepRuntime` and `RaceState`.  One decision holds its CTBR action
for ten exact 2 ms physics steps (a 50 Hz policy decision rate).  Gate crossing,
ground/gate-frame collision, motor dynamics, and terminal behavior use the
same FlightStack-owned reference path as headless evaluation.

The versioned configuration is
[`config/ai/racing_v1.toml`](../config/ai/racing_v1.toml):

- Reset jitter is seeded and bounded in horizontal position, altitude, and
  yaw.  It is initial-state randomization, **not a curriculum**.
- The observation has 27 normalized values: body velocity/rate; vectors and
  normals for the next two ordered gates; world up in body coordinates; the
  previous normalized action; next-gate distance; and world speed.  It omits a
  raw quaternion to avoid the `q`/`-q` representation ambiguity.
- Reward terms are visible and versioned: progress, gate/lap events, collision
  and out-of-bounds penalties, action-delta and angular-rate penalties, and a
  time penalty.
- A native Gymnasium adapter is created only when the optional extra is
  installed.  The core environment itself has no mandatory Gymnasium import.

`ReferenceVectorEnv` can batch independent exact Python environments with
explicit per-environment seeds.  It is a deterministic reference/evaluation
helper, **not a JAX backend**, a claim of vectorized physics throughput, or a
second simulation model.  PPO training uses Stable-Baselines3's `DummyVecEnv`
around the native Gymnasium adapter.  No JAX backend, visual/sensor policy, or
curriculum scheduler is implemented.

## Training, metadata, and serving a policy

Install the optional dependencies and produce a local checkpoint:

```powershell
python -m pip install -e '.[train]'
flightstack train --output artifacts/training-smoke --smoke

# Default PPO settings use 25,000 timesteps; choose an explicit experimental run.
flightstack train --output artifacts/training-run --timesteps 10000 --seed 17
```

Training delegates PPO and the MLP implementation to Stable-Baselines3; it does
not implement PPO from scratch.  It writes `ppo_model.zip` plus a required
`ppo_model.metadata.json` sidecar.  The metadata records:

- action and observation schema versions;
- the active `VehicleConfig` hash; and
- the complete numeric AI-configuration hash and its 20 ms control period;
- the SHA-256 content hash of `ppo_model.zip`; and
- algorithm, PPO configuration, full AI contract, training track hash, and
  Python/NumPy/SB3 dependency versions.

`LearnedPolicyPilot.from_checkpoint()` validates the archive hash, vehicle,
action/observation schemas, full AI configuration, and decision rate before
loading an SB3 model.  The recorded track identifies the training run; the
track-relative observation contract does not artificially restrict a policy to
that one course.  A missing or incompatible sidecar is an error, never a
fallback to Classical.

To make a compatible local checkpoint selectable in the browser, start the
server with it:

```powershell
flightstack serve --policy artifacts/training-run/ppo_model.zip
```

Without `--policy`, `LEARNED` remains unavailable and the server gives an
explicit notice.  With `--policy`, selection means only that the checkpoint
passed compatibility checks; it is not a performance endorsement.

## Local training evidence and its limitation

Two local PPO runs exercised the full export/load/evaluation plumbing:

| Run | Training setting | Seeded technical-eight evaluation outcome |
| --- | --- | --- |
| Smoke | 256 PPO timesteps, seed 17 | Crashed, 0 gates, did not complete |
| Longer check | 10,000 PPO timesteps, seed 17 | Crashed, 0 gates, did not complete |

Both were evaluated on the named `technical-eight-wind-degraded` scenario at
seed 42.  The local result/checkpoint directories are ignored by Git and are
not shipped as model artifacts.  Therefore FlightStack currently has **no
quality, recommended, or benchmark-winning learned checkpoint**.  Do not use
either run as evidence of successful racing or real-world transfer.

## What remains before a learned result can be claimed

1. Improve the task/training recipe and demonstrate a checkpoint that completes
   a declared evaluation set; retain failures and raw artifacts.
2. Compare it with the Classical pilot on the same seeded scenarios using the
   paired statistics tools, rather than comparing isolated best laps.
3. Run and report a bounded robustness grid (wind and motor-efficiency hooks
   are available) with completion/crash results and uncertainty intervals.
4. Make a deliberate high-throughput-backend decision only after a dedicated
   parity harness exists.  JAX is an option under consideration, not an
   implemented or implied backend.
5. Record dependency versions, checkpoint hashes, source commit, configuration
   hashes, and complete scenario provenance with every report.

## Source/reuse boundary

The pre-researched [source manifest](research/SOURCE_MANIFEST.md) maps pinned
LSY Drone Racing, Crazyflow, SimpleFlight, and PPO-plumbing references.  The
current environment, action/observation/reward contracts, policy adapter, and
experiment code are FlightStack-native; those upstream projects are not
vendored.  See [THIRD_PARTY.md](../THIRD_PARTY.md) and the
[reuse audit](research/reuse-audit.md) for dependency and reuse status.
