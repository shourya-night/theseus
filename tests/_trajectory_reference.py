"""
Independent reference trajectories for interpolation tests.

Nothing here imports ``theseus.propagation``.  Truth comes from either an
exact analytic two-body solution or a SciPy integration at far tighter
tolerance than the engine uses, evaluated *directly at the requested time* --
never by interpolating between samples.  A test comparing the engine's
interpolant against these values is therefore measuring real trajectory error,
not the difference between two interpolation schemes.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp


MU_EARTH = 3.986004418e14      # m^3/s^2
J2_EARTH = 1.08262668e-3
R_EARTH = 6378137.0


# ---------------------------------------------------------------------------
# Exact analytic two-body propagation
# ---------------------------------------------------------------------------

def _solve_kepler(M: float, e: float, tol: float = 1e-14, max_iter: int = 200) -> float:
    """Eccentric anomaly from mean anomaly by Newton iteration."""
    M = math.fmod(M, 2.0 * math.pi)
    E = M if e < 0.8 else math.pi
    for _ in range(max_iter):
        f = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break
    return E


def kepler_propagator(a: float, e: float, inc_deg: float = 0.0,
                      raan_deg: float = 0.0, argp_deg: float = 0.0,
                      nu0_deg: float = 0.0, mu: float = MU_EARTH):
    """
    Exact two-body state at any time, from classical elements.

    Returns ``state_at(t) -> (position, velocity)``.  This is a closed-form
    solution of the two-body problem, so it carries no integration error at
    all -- only Kepler-solver round-off at the 1e-14 level.
    """
    n = math.sqrt(mu / a ** 3)
    inc = math.radians(inc_deg)
    raan = math.radians(raan_deg)
    argp = math.radians(argp_deg)

    # Mean anomaly at epoch from the initial true anomaly
    nu0 = math.radians(nu0_deg)
    E0 = 2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(0.5 * nu0),
                          math.sqrt(1.0 + e) * math.cos(0.5 * nu0))
    M0 = E0 - e * math.sin(E0)

    cos_raan, sin_raan = math.cos(raan), math.sin(raan)
    cos_i, sin_i = math.cos(inc), math.sin(inc)
    cos_w, sin_w = math.cos(argp), math.sin(argp)

    # Perifocal -> inertial rotation matrix
    R = np.array([
        [cos_raan * cos_w - sin_raan * sin_w * cos_i,
         -cos_raan * sin_w - sin_raan * cos_w * cos_i,
         sin_raan * sin_i],
        [sin_raan * cos_w + cos_raan * sin_w * cos_i,
         -sin_raan * sin_w + cos_raan * cos_w * cos_i,
         -cos_raan * sin_i],
        [sin_w * sin_i, cos_w * sin_i, cos_i],
    ])

    p = a * (1.0 - e * e)

    def state_at(t: float):
        M = M0 + n * float(t)
        E = _solve_kepler(M, e)
        nu = 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(0.5 * E),
                              math.sqrt(1.0 - e) * math.cos(0.5 * E))
        r_mag = p / (1.0 + e * math.cos(nu))

        r_pf = np.array([r_mag * math.cos(nu), r_mag * math.sin(nu), 0.0])
        v_pf = math.sqrt(mu / p) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])
        return R @ r_pf, R @ v_pf

    return state_at


# ---------------------------------------------------------------------------
# High-accuracy numerical reference (for perturbed cases)
# ---------------------------------------------------------------------------

def j2_reference(r0, v0, t_span, mu: float = MU_EARTH,
                 j2: float = J2_EARTH, radius: float = R_EARTH,
                 rtol: float = 1e-13, atol: float = 1e-9):
    """
    Dense-output SciPy reference for two-body + J2 motion.

    Returns ``state_at(t) -> (position, velocity)`` backed by the solver's own
    dense output at rtol 1e-13, which is four to five orders tighter than the
    engine's working tolerance.
    """
    def rhs(t, y):
        r = y[:3]
        rn = float(np.linalg.norm(r))
        a = -mu / rn ** 3 * r
        k = 1.5 * mu * j2 * radius ** 2 / rn ** 5
        z2 = (r[2] / rn) ** 2
        a = a + np.array([
            k * r[0] * (5.0 * z2 - 1.0),
            k * r[1] * (5.0 * z2 - 1.0),
            k * r[2] * (5.0 * z2 - 3.0),
        ])
        return np.concatenate([y[3:], a])

    sol = solve_ivp(rhs, t_span, np.concatenate([np.asarray(r0), np.asarray(v0)]),
                    rtol=rtol, atol=atol, dense_output=True, method="DOP853")
    if not sol.success:
        raise RuntimeError(f"reference integration failed: {sol.message}")

    def state_at(t: float):
        y = sol.sol(float(t))
        return y[:3].copy(), y[3:].copy()

    return state_at


def sample_reference(state_at, times) -> tuple[np.ndarray, np.ndarray]:
    """Sample a reference trajectory onto a node grid."""
    times = np.asarray(times, dtype=np.float64)
    pos = np.empty((len(times), 3))
    vel = np.empty((len(times), 3))
    for i, t in enumerate(times):
        r, v = state_at(float(t))
        pos[i] = r
        vel[i] = v
    return pos, vel


def linear_position(times, positions, t: float) -> np.ndarray:
    """
    The pre-correction interpolation, kept only so tests can quantify how much
    error it carried.  Not used by any engine code path.
    """
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    t_c = min(max(float(t), times[0]), times[-1])
    idx = int(np.searchsorted(times, t_c))
    if idx == 0:
        return positions[0].copy()
    if idx >= len(times):
        return positions[-1].copy()
    dt = times[idx] - times[idx - 1]
    if dt <= 0:
        return positions[idx].copy()
    frac = (t_c - times[idx - 1]) / dt
    return (1.0 - frac) * positions[idx - 1] + frac * positions[idx]
