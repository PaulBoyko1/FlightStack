"""Run the nontrivial reference attitude step."""

from flightstack.sim.scenarios import run_reference_step

telemetry = run_reference_step()
print(f"final error: {telemetry.final_error_deg:.4f} deg")
print(f"peak rate: {telemetry.peak_rate_rad_s:.4f} rad/s")
