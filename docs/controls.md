# Interactive controls

The local simulator is served by `flightstack serve` after building the web
client.  It opens at `http://127.0.0.1:8000`; the browser connects back to the
same process by WebSocket.  The physics and race state remain on the server.

## Keyboard

| Input | Effect |
| --- | --- |
| `W` / `S` | Positive / negative pitch stick |
| `A` / `D` | Negative / positive roll stick |
| `Q` / `E` | Negative / positive yaw stick |
| `Space` | Increase throttle while held |
| Left `Ctrl` | Decrease throttle while held |
| `R` | Reset the vehicle, race, pilots, and in-memory replay capture |
| `C` | Cycle chase, FPV, and spectator cameras |
| `T` | Toggle the technical drawer/body axes/velocity vector |
| `1` | Select Human pilot |
| `2` | Select Classical pilot |
| `3` | Request Learned pilot (currently refused: no checkpoint is loaded) |

Throttle is persistent: releasing `Space` or `Ctrl` holds its current value.
This makes a keyboard usable as a simple throttle slider, but it also means a
reset returns throttle to zero and the pilot must deliberately raise it again.

## Gamepad

The browser uses the first connected Gamepad API device with the common
four-axis layout below.  Controller mappings differ by browser/device, so
verify the live vehicle response in the simulator before relying on a specific
transmitter adapter.

| Browser axis | FlightStack input |
| --- | --- |
| 0 | Yaw |
| 1 | Throttle; top maps to 1, bottom maps to 0 |
| 2 | Roll |
| 3 | Pitch (inverted at the browser edge) |

The server applies the final stick deadzone/expo curve, so hardware-specific
browser noise does not bypass the canonical pilot contract.

## Human-pilot semantics

The browser sends normalized values; `HumanPilot` turns them into the shared
collective-thrust/body-rate (CTBR) command used by every pilot:

- Roll, pitch, and yaw use a continuous 0.07 deadzone and 0.32 cubic expo,
  then scale by the rate limits in the vehicle TOML.  Default yaw uses 82% of
  the configured yaw-rate limit.
- Throttle is clipped to `[0, 1]`.  `0` means zero collective thrust, `0.5`
  maps to physical hover thrust (`mass * gravity`), and `1` maps to the
  configured total motor maximum.
- A new Human run remains in **preflight** until the requested collective is
  at least 95% of hover thrust (approximately a normalized throttle of 0.475
  with the default mapping).  This avoids expending a race or falling before
  an input has arrived.

CTBR is not raw motor control.  Human commands still pass through the same
rate PID, mixer, motor lag, and 6DOF plant as the classical baseline.

## Pilot modes

`HUMAN` is the default and consumes browser input.  `CLASSICAL` starts a
conservative deterministic gate-center guidance baseline; it can be used as a
smoke test for the exact same plant, collision, and race path.  Selecting a
mode does not implicitly reset the session; use `R` when a fresh run is
needed.

`LEARNED` is visible to make the intended comparison seam clear, but it is not
an available mode today.  The server returns a notice rather than pretending a
model exists or falling back to Classical.  See [AI status](ai.md).

## Race behavior

The shipped `technical-eight` course has eight ordered gates.  A pass counts
only when the vehicle's swept path crosses the next gate's plane in the
required direction and inside the aperture.  Merely approaching the gate
center, passing a later gate first, or replaying the same segment cannot add
progress.

Contact with the ground or gate frame is a crash for the current interactive
session.  Press `R` or use the Reset button to begin again.  The technical
drawer displays canonical body rates, total thrust, and individual motor
thrust levels; it is a view of server telemetry, not a separate simulation.
