# Architecture

## Control path

```text
target attitude
      |
      v
quaternion log error (body frame)
      |
      v
attitude P loop -> desired body rate (limited)
      |
      v
rate PID
  - derivative on measured gyro
  - low-pass derivative
  - conditional anti-windup
      |
      v
body torque command
      |
      v
rotational rigid-body plant
```

## Sensor / estimation path

`IMUSimulator` produces timestamped gyro and accelerometer samples with configurable bias/noise. `ComplementaryAttitudeEstimator` propagates attitude using gyro data and uses measured gravity to correct roll/pitch. Yaw is intentionally left gyro-driven because gravity does not make heading observable.

## HIL path

The Python HIL module provides a compact binary packet with a two-byte sync marker, uint8 message id, uint16 payload length, and CRC-16/CCITT. This is deliberately transport-agnostic: a future serial/USB layer can carry the same framed messages between the simulation harness and an MCU.

## Embedded parity

`cpp/` contains dependency-free C++20 quaternion and outer-loop control primitives. Golden-vector tests verify constant-rate quaternion integration, quaternion-log error, and the rate-command path. This is the start of moving timing-critical control code to an MCU without changing frame conventions.
