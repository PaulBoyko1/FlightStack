# Experiments and reproducibility

## What is reproducible today

The repository currently provides deterministic regression scenarios, not a
published performance study.  The Python test suite covers the important
contracts below:

- quaternion storage/frame adapters and non-identity attitude targets;
- PID saturation/anti-windup, rotational dynamics, deterministic IMU, and HIL
  frame validation;
- 6DOF free fall, hover, motor response, mixer signs/saturation, disturbance
  hooks, and fixed-step determinism;
- swept gate crossing, gate-local geometry, wrong-order/repeated gate handling,
  collision helpers, and a technical-eight state machine;
- human CTBR mapping, replay serialization, server WebSocket behavior, and a
  conservative classical pilot completing the reference course.

Run the Python checks from the repository root:

```powershell
pytest
ruff check .
mypy src
```

The CI workflow independently runs the C++ checks, Rust format/lint/tests, and
the production web build.  These are correctness gates, not benchmark scores.

## Run provenance available now

`VehicleConfig.config_hash` creates a stable short hash from the canonical
vehicle mapping.  Interactive sessions create a `flightstack-replay-v1`
`ReplayRecorder` containing the vehicle hash, track name, physics timestep,
pilot kind, authoritative state, command, compact race data, and recorded
events.  The recorder's Python `write(path)` method persists JSON for a test or
an external experiment driver.

At present there is no browser export button, no committed replay corpus, and
no experiment-result artifact directory.  A successful unit test or classical
smoke run must not be described as a statistical result.

## Minimum reporting record for future experiments

Every numerical result should retain:

| Record | Required content |
| --- | --- |
| Code identity | Git commit and dirty-worktree status |
| Vehicle/track | TOML configuration hash, track JSON hash/name, lap/race rules |
| Runtime | Reference or training backend, timestep/control rate, OS/runtime versions |
| Pilot | Pilot kind, checkpoint hash if any, action scaling and observation schema |
| Randomness | Master seed and all derived episode seeds |
| Scenario | Initial-state distribution, wind/force/motor-efficiency disturbances, collision policy |
| Outcomes | Completion/crash rate, gates/laps, lap-time distribution, return/reward if applicable |
| Artifacts | Configs, summaries, raw or replay-backed episode records, plotting script version |

## Evaluation protocol to implement before reporting learned results

1. Freeze the vehicle, track, collision, and CTBR contracts for the comparison.
2. Run the classical baseline and learned policy on the same initial states and
   disturbance seeds (paired seeds).
3. Report means with dispersion/confidence intervals, completion/crash rate,
   and a paired difference where appropriate rather than only a best lap.
4. Sweep bounded wind, parameter, motor-efficiency, and sensor/noise conditions
   in a named robustness grid; state precisely which conditions cause failure.
5. Keep failures in the artifact set.  Do not omit crashes, invalid episodes,
   or retry counts.
6. Repeat a small set of deterministic scenarios across Python and the selected
   high-throughput backend before treating faster-training results as reference
   physics results.

This protocol is a requirement for future results; it has not yet produced a
trained-policy report in this repository.
