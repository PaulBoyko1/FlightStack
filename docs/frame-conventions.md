# Frame and quaternion conventions

FlightStack makes frame semantics explicit because attitude-control bugs are often frame bugs.

- Quaternion storage: scalar-first `[w, x, y, z]`.
- `q_body_to_world` rotates a vector expressed in body coordinates into world coordinates.
- Gyroscope angular velocity is expressed in body coordinates.
- Body-rate integration is right-multiplicative: `q_next = q * Exp(omega_body * dt)`.
- The controller needs a current-to-target error expressed in the current body frame, so the relative quaternion is `conj(q_current) * q_target`.
- The outer loop uses the exact shortest-path quaternion logarithm (rotation vector), not the small-angle `2*q_xyz` approximation.
- Quaternion sign is canonicalized to the positive scalar hemisphere for shortest-path error; `q` and `-q` are treated as identical attitudes.

The regression suite includes arbitrary non-identity current/target attitudes specifically so the multiplication-order bug cannot hide behind an identity target.
