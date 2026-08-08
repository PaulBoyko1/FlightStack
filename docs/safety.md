# Hardware safety boundary

FlightStack is currently a simulation/HIL engineering project, not a flight-certified autopilot.

Before connecting a controller to a real multirotor:

1. Run controller and estimator tests against recorded/synthetic sensor data.
2. Validate serial/HIL framing and actuator outputs with motors disconnected.
3. Bench-test the flight controller with **propellers removed**.
4. Add actuator saturation, arming state, watchdogs, failsafe behavior, loop-deadline monitoring, sensor health checks, and explicit disarm paths.
5. Use a restrained test rig before free flight.

The repository intentionally does not claim airworthiness or safe free-flight operation.
