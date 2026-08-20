"""
Analytical two-body orbital propagation.

Propagates a Keplerian orbit by advancing the mean anomaly and solving
Kepler's equation at each output time.

This is exact for the unperturbed two-body problem (no drag, no J2, etc.).

Mathematics
-----------
1. Convert initial state (r₀, v₀) → orbital elements.
2. At each future time t:
       M(t) = M₀ + n·(t − t₀)       (advance mean anomaly)
       Solve M → E (Kepler's equation)
       E → ν (true anomaly)
       ν → (r, v) via elements-to-state conversion
3. This is exact for the two-body problem (energy, angular momentum
   conserved to machine precision).

Reference
---------
Curtis, "Orbital Mechanics for Engineering Students", 4th ed., §3.7.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from theseus.core.state import SimulationState, StateHistory
from theseus.core.events import EventType, EventLog
from theseus.orbital.elements import OrbitalElements
from theseus.orbital.conversions import state_to_elements, elements_to_state
from theseus.orbital.kepler import solve_kepler, eccentric_to_true, hyperbolic_to_true


def propagate_twobody(
    r0: np.ndarray,
    v0: np.ndarray,
    mu: float,
    times: Sequence[float] | np.ndarray,
    *,
    t0: float = 0.0,
    mass: float = 0.0,
) -> StateHistory:
    """
    Analytically propagate a two-body orbit.

    Parameters
    ----------
    r0 : np.ndarray   Initial position (m).
    v0 : np.ndarray   Initial velocity (m/s).
    mu : float         Gravitational parameter (m³/s²).
    times : array-like Output times (s) at which to evaluate the state.
    t0 : float         Epoch of (r0, v0) (s).
    mass : float       Spacecraft mass (kg), carried through for state records.

    Returns
    -------
    StateHistory
        Complete time history.
    """
    r0 = np.asarray(r0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)

    oe0 = state_to_elements(r0, v0, mu)
    M0 = oe0.mean_anomaly
    n = oe0.mean_motion  # rad/s

    if n is None and oe0.e < 1.0:
        raise ValueError("Cannot propagate: mean motion is undefined.")

    history = StateHistory()

    for t in times:
        dt = t - t0
        if oe0.e < 1.0:
            # Elliptic: advance mean anomaly
            M = (M0 + n * dt) % (2.0 * math.pi)
            sol = solve_kepler(M, oe0.e)
            if not sol.converged:
                raise RuntimeError(f"Kepler solver did not converge at t={t}: residual={sol.residual}")
            nu = eccentric_to_true(sol.eccentric_anomaly, oe0.e)
        else:
            # Hyperbolic
            if oe0.a == 0:
                raise ValueError("Semi-major axis is zero; cannot propagate.")
            n_hyp = math.sqrt(mu / abs(oe0.a)**3)
            M = M0 + n_hyp * dt
            sol = solve_kepler(M, oe0.e)
            if not sol.converged:
                raise RuntimeError(f"Kepler solver did not converge at t={t}")
            nu = hyperbolic_to_true(sol.eccentric_anomaly, oe0.e)

        # Build orbital elements at this time (same shape orbit, different ν)
        oe_t = OrbitalElements(
            a=oe0.a, e=oe0.e, i=oe0.i, raan=oe0.raan, argp=oe0.argp,
            nu=nu, mu=mu,
        )
        r, v_vec = elements_to_state(oe_t)

        # Acceleration (two-body only)
        r_mag = float(np.linalg.norm(r))
        accel = -mu / r_mag**3 * r

        state = SimulationState(
            time=t,
            position=r,
            velocity=v_vec,
            acceleration=accel,
            mass=mass,
            metadata={"altitude_m": r_mag - 6_378_137.0},  # rough Earth altitude
        )
        history.append(state)

    return history
