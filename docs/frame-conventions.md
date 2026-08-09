# Frame and quaternion conventions

FlightStack makes frame semantics explicit because attitude-control bugs are often frame bugs.

- Quaternion storage: scalar-first `[w, x, y, z]`.
- World frame: right-handed local ENU-like frame, with `+Z` up and gravity
  `[0, 0, -g]`.
- Body frame: FLU, with `+X` forward, `+Y` left, and `+Z` up.
- `q_body_to_world` rotates a vector expressed in body coordinates into world coordinates.
- Gyroscope angular velocity is expressed in body coordinates.
- Body-rate integration is right-multiplicative: `q_next = q * Exp(omega_body * dt)`.
- The controller needs a current-to-target error expressed in the current body frame, so the relative quaternion is `conj(q_current) * q_target`.
- The outer loop uses the exact shortest-path quaternion logarithm (rotation vector), not the small-angle `2*q_xyz` approximation.
- Quaternion sign is canonicalized to the positive scalar hemisphere for shortest-path error; `q` and `-q` are treated as identical attitudes.

The regression suite includes arbitrary non-identity current/target attitudes specifically so the multiplication-order bug cannot hide behind an identity target.

## External adapter boundary

FlightStack never silently changes quaternion storage or frame semantics.  APIs
that require scalar-last quaternions must use the named
`wxyz_to_xyzw`/`xyzw_to_wxyz` adapters.  They convert **storage only** and do
not reinterpret the body-to-world rotation.  The adapter tests cover identity,
principal-axis rotations, arbitrary orientations, vector rotation, and the
`q`/`-q` physical equivalence.
