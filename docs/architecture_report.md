# THESEUS Phase 1-7 Recovery Architecture Report

## Authoritative data flow

THESEUS is an astrodynamics engine first. Mission configuration is passed to the FastAPI layer, which validates the request and calls core engine modules. The frontend renders only state histories returned by the backend.

```text
User mission configuration
  -> FastAPI request validation
  -> core constants / coordinates / orbital model
  -> propagation / dynamics
  -> Lambert, Hohmann, rendezvous solvers
  -> authoritative state histories
  -> API response schema
  -> React visualization transform
```

## Backend dependency graph

Core modules are organized so the Phase 1-7 engine does not depend on FastAPI, React, reentry, conjunction, uncertainty, or collision probability:

- `theseus.constants`: SI constants and unit helpers.
- `theseus.bodies`: physical body catalog using SI parameters.
- `theseus.coordinates`: frame and coordinate transformations.
- `theseus.orbital`: elements, conversions, Kepler equation, Lambert solver.
- `theseus.propagation`: analytical two-body propagation plus RK4/RKF45 integration.
- `theseus.dynamics`: point mass, J2, drag, SRP, thrust, composite force models.
- `theseus.maneuvers`: Hohmann and burn utilities built on the core.
- `theseus.rendezvous`: rendezvous solver built on Lambert and propagation.
- `theseus.server`: thin API adapter that serializes authoritative histories.
- Phase 8-10 extensions (`reentry`, `conjunction`, `uncertainty`) remain separate and are not required by Phase 1-7.

## Units and frames

- Core physics uses SI: meters, meters per second, seconds, kilograms, radians.
- API request convenience fields ending in `_km`, `_km_s`, `_hours`, or `_deg` are converted once at the server boundary.
- API state histories are serialized in SI meters, meters per second, seconds, and kilograms.
- Interplanetary visualization uses heliocentric inertial coordinates in meters with the Sun at the origin.
- Canvas rendering applies one transform: physics frame `(x, y)` in meters -> camera-relative meters -> screen pixels, with `y` inverted for canvas coordinates.

## Trajectory and body generation policy

- Spacecraft trajectory samples are produced by the backend propagator/solver and returned as `state_history`.
- Planetary states for Lambert interplanetary missions are returned as backend-owned `bodies[].state_history` samples on the same simulation clock.
- React may interpolate or select between backend samples for display, but it must not compute planetary motion, spacecraft trajectories, collision events, TCA, debris paths, or probability of collision.

## Health policy

`/api/health` now reports executable subsystem smoke checks. `ONLINE` means a subsystem completed a basic physical calculation; it no longer means a module merely imported.
