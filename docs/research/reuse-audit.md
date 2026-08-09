# Reuse audit

## Current implementation

The 6DOF Python reference plant, configuration format, CTBR controller seam,
mixer, tests, and documentation are original FlightStack code.  No external
project source files or visual assets have been copied.

## Planned commodity dependencies

- Three.js for browser rendering, under its MIT license.
- A maintained PPO package selected after the environment contract is stable.
- Rapier/Parry only if its collision primitives materially reduce risk versus a
  small FlightStack-owned race collision layer.

Any direct code adaptation will add the upstream path, commit, license, and
required attribution to this file and `THIRD_PARTY.md` before it is committed.
