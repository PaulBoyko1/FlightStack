# FlightStack Rust runtime

This workspace is the portable, deterministic runtime foundation for
FlightStack.  It deliberately owns the canonical simulation equations rather
than wrapping a general-purpose rigid-body engine.

- `flightstack-core` defines the explicit frame/quaternion contract, canonical
  state and CTBR command types, and validated vehicle configuration.
- `flightstack-sim` implements exact first-order motor lag, geometry-derived
  allocation, and the fixed-step 6DOF reference plant.

The source of physical parameters is the repository-level
[`config/vehicles/flightstack_5in.toml`](../config/vehicles/flightstack_5in.toml).
`flightstack-core` embeds that same tracked file for the `reference_5in()`
convenience loader; it does not duplicate the values in Rust source.

The canonical convention is scalar-first `[w, x, y, z]`, body-to-world, with
world `+Z` up and FLU body axes.  Any scalar-last integration must cross the
named `wxyz_to_xyzw` / `xyzw_to_wxyz` adapter boundary.

Run the Rust checks from this directory:

```powershell
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

The simulator is an engineering reference model, not an authorization to fly
hardware.  Validate controller and actuator paths with appropriate safeguards
before hardware use.
