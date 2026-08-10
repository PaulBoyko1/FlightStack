# Experiments and reproducibility

## What is implemented

`flightstack.experiments` provides a headless, fixed-step evaluation path over
the canonical Python vehicle, CTBR controller, motor model, gate state, and
collision predicates.  It is not a second or simplified experiment physics
model.

- `Scenario` is a validated TOML-facing record for duration, physics timestep,
  telemetry period, vehicle radius, wind/force/torque, motor efficiency, and
  seed-owned jitter.
- `run_episode()` creates a fresh pilot and runs the same
  `PilotCommand -> rate PID -> mixer -> motors -> 6DOF` chain used by the
  interactive service.
- `EpisodeResult` holds provenance, metrics, event log, sampled telemetry, and
  a versioned FlightStack replay.  `write_artifacts()` writes `result.json`,
  `telemetry.json`, and `replay.json`.
- `paired_evaluate()` runs supplied pilot factories against identical scenario
  seeds.  Its elapsed-time deltas use only pairs where both pilots finish, so
  crashes and timeouts cannot be presented as a faster lap; completion is
  reported separately.
- `build_robustness_grid()` expands named wind-speed, motor-efficiency, and
  seed axes into explicit `Scenario` cases.  It is a grid builder, not an
  unreported benchmark run.

The checked-in
[`technical-eight-wind-degraded`](../config/scenarios/technical-eight-wind-degraded.toml)
scenario is a bounded 20 s, 2 ms reference case with seeded wind and mild
per-motor degradation.  It is useful for reproducible comparison, not a proxy
for hardware conditions.

## Run a headless episode

```powershell
# Defaults to the named technical-eight wind/degradation scenario.
flightstack evaluate --pilot classical --output artifacts/classical-evaluation

# A learned evaluation requires both the optional dependencies and a compatible
# checkpoint plus its metadata sidecar.
flightstack evaluate `
  --pilot learned `
  --policy artifacts/training-run/ppo_model.zip `
  --scenario technical-eight-wind-degraded `
  --output artifacts/learned-evaluation
```

The command prints a JSON summary and, when `--output` is present, writes the
three artifact files.  The summary includes termination type, elapsed time,
completion/lap/gate/collision counts, distance/speed/rate statistics, mixer
saturation count, final gate distance, and full scenario/vehicle provenance.
That provenance includes the full scenario mapping and canonical hash, the
resolved track source path and content hash, vehicle hash, and best-effort Git
revision/dirty status.  Git fields are `null` outside an available checkout.
Callers evaluating a learned artifact can pass
`checkpoint_model_identity(checkpoint_path)` as `pilot_model_identity` to
record model and metadata-sidecar hashes as well.

`replay.json` is a validated FlightStack replay-v1 record, not a browser-only
capture.  It carries sampled canonical state (including motor output), CTBR
command, pilot, race snapshot, and events.  Inspect it or export state frames
without rerunning physics:

```powershell
flightstack replay artifacts/classical-evaluation/replay.json --at 4.0 --interpolate
flightstack replay artifacts/classical-evaluation/replay.json --csv artifacts/replay.csv
```

Interpolation is explicitly for the continuous rendered/debug state; race,
pilot, commands, and events remain discrete recorded values.  There is no
claim yet of deterministic re-simulation from a telemetry-rate command stream.

`artifacts/` is ignored by Git.  Preserve a reviewed artifact set outside a
discardable worktree or attach it to a documented experiment release; no result
is made reproducible merely by pasting a console summary into documentation.

## Paired comparison and robustness APIs

Paired evaluation is exposed as a Python API because callers must choose the
pilot factories and seed set explicitly.  The critical property is that each
pilot receives a fresh instance and the same `Scenario.with_seed(seed)`
realization.  This prevents independently sampled wind/motor variation from
being mistaken for a controller difference.

`matched_pairs` is the number of seed-matched episodes; `completed_pairs` is
the subset on which both pilots finished.  Elapsed-time delta statistics and
their confidence interval are `null` when there are no completed pairs.  In
that case, compare completion outcomes instead of inferring speed from crash or
timeout time-to-termination.

For a valid comparison, report:

| Record | Required content |
| --- | --- |
| Code identity | Git commit and dirty-worktree status |
| Vehicle/track | TOML configuration hash, track name/hash, lap/race rules |
| Runtime | Reference or training backend, timestep/control rate, OS/runtime versions |
| Pilot | Pilot kind, checkpoint hash if any, action scaling and observation schema |
| Randomness | Master seed, per-episode seeds, bootstrap seed/sample count |
| Scenario | Initial-state distribution, wind/force/motor-efficiency disturbances, collision policy |
| Outcomes | Completion/crash rate, gates/laps, lap-time distribution, paired deltas, reward if relevant |
| Artifacts | Configs, summaries, telemetry/replays, and plotting/analysis script version |

The current statistical helper is correct only for the exact supplied samples;
a narrow or all-failure set cannot establish a strong policy claim.  Keep
timeouts, crashes, retries, and invalid episodes in the artifacts.

## Current PPO evidence

The repository has real PPO training/export/inference plumbing, but it does
not contain a successful learned racing result.  A local 256-step smoke run
and a local 10,000-step full-course PPO run (both seed 17) were evaluated on
`technical-eight-wind-degraded`, seed 42.  Both crashed before passing a gate
and did not finish.  The checkpoints and result directories are ignored local
artifacts, not shipped model weights or benchmark evidence.

This is useful negative evidence: the pipeline is exercised, but the training
recipe has not demonstrated competence.  Do not compare either run to the
Classical baseline as if it were a quality policy.

## What remains before reporting a learned result

1. Freeze a declared vehicle/track/collision/CTBR contract for a comparison.
2. Train a candidate whose held-out seeded evaluations meaningfully complete
   the course; record every run and checkpoint hash.
3. Run Classical and learned pilots on matching seeds, then report completion
   differences and dispersion/intervals, not just a best lap.
4. Execute a named robustness grid over bounded wind, parameter, and
   motor-efficiency conditions; report failure boundaries as well as successes.
5. Add a dedicated reference-versus-faster-backend parity harness before
   interpreting results from any future high-throughput backend as reference
   physics results.  `ReferenceVectorEnv` is not such a backend.
