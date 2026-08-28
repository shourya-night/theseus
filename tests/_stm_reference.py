"""
Independent reference for state-transition-matrix tests.

Nothing in this module imports from ``theseus``.  The dynamics, the dynamics
Jacobian, the variational system and the trajectory propagation are all
written here and integrated with SciPy's DOP853, so a test comparing an engine
STM against these values is not comparing the engine against itself.

The Jacobian is obtained by central differences of the acceleration rather
than analytically, so the reference cannot share an algebraic mistake with the
engine's analytic gravity/J2 Jacobians either.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp


MU_EARTH = 3.986004418e14      # m^3/s^2
J2_EARTH = 1.08262668e-3
RE_EARTH = 6378137.0           # m


def acceleration(r, mu: float = MU_EARTH, j2: float = 0.0,
                 radius: float = RE_EARTH) -> np.ndarray:
    """Point-mass gravity, optionally with the J2 zonal term."""
    r = np.asarray(r, dtype=np.float64)
    r_norm = float(np.linalg.norm(r))
    a = -mu * r / r_norm ** 3
    if j2:
        k = 1.5 * j2 * mu * radius ** 2 / r_norm ** 5
        z2 = r[2] ** 2 / r_norm ** 2
        a = a + k * np.array([
            r[0] * (5.0 * z2 - 1.0),
            r[1] * (5.0 * z2 - 1.0),
            r[2] * (5.0 * z2 - 3.0),
        ])
    return a


def _da_dr(r, mu, j2, radius, h: float = 1.0) -> np.ndarray:
    """d(acceleration)/d(position) by central differences."""
    jac = np.zeros((3, 3), dtype=np.float64)
    for k in range(3):
        step = np.zeros(3)
        step[k] = h
        jac[:, k] = (acceleration(r + step, mu, j2, radius)
                     - acceleration(r - step, mu, j2, radius)) / (2.0 * h)
    return jac


def propagate_state(x0, t0: float, tf: float, mu: float = MU_EARTH,
                    j2: float = 0.0, radius: float = RE_EARTH) -> np.ndarray:
    """Nominal 6-state at ``tf``, integrated from ``x0`` at ``t0``."""
    x0 = np.asarray(x0, dtype=np.float64)
    if tf == t0:
        return x0.copy()

    def deriv(t, y):
        return np.concatenate([y[3:6], acceleration(y[:3], mu, j2, radius)])

    sol = solve_ivp(deriv, (t0, tf), x0, rtol=1e-13, atol=1e-9, method="DOP853")
    return np.asarray(sol.y[:, -1], dtype=np.float64)


def reference_stm(x0, t0: float, tf: float, mu: float = MU_EARTH,
                  j2: float = 0.0, radius: float = RE_EARTH):
    """
    Phi(tf, t0) and the nominal state at tf, from the 42-state variational
    system integrated independently.

    Returns
    -------
    (stm, state_tf)
    """
    x0 = np.asarray(x0, dtype=np.float64)
    if tf == t0:
        return np.eye(6), x0.copy()

    def deriv(t, y):
        r, v, phi = y[:3], y[3:6], y[6:42].reshape(6, 6)
        a_matrix = np.zeros((6, 6), dtype=np.float64)
        a_matrix[:3, 3:] = np.eye(3)
        a_matrix[3:, :3] = _da_dr(r, mu, j2, radius)
        out = np.empty(42, dtype=np.float64)
        out[:3] = v
        out[3:6] = acceleration(r, mu, j2, radius)
        out[6:] = (a_matrix @ phi).ravel()
        return out

    y0 = np.concatenate([x0, np.eye(6).ravel()])
    sol = solve_ivp(deriv, (t0, tf), y0, rtol=1e-12, atol=1e-10, method="DOP853")
    return sol.y[6:42, -1].reshape(6, 6), np.asarray(sol.y[:6, -1])


def finite_difference_column(x0, t0: float, tf: float, index: int,
                             delta: float, mu: float = MU_EARTH,
                             j2: float = 0.0, radius: float = RE_EARTH):
    """
    One column of Phi(tf, t0) by central-difference propagation.

    Uses no variational equations at all: two full nonlinear propagations of
    perturbed initial states.  This is the most independent construction
    available, since it shares nothing with any STM formulation.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    plus, minus = x0.copy(), x0.copy()
    plus[index] += delta
    minus[index] -= delta
    return (propagate_state(plus, t0, tf, mu, j2, radius)
            - propagate_state(minus, t0, tf, mu, j2, radius)) / (2.0 * delta)


def eccentric_state(a_m: float, e: float, inc_deg: float = 35.0,
                    mu: float = MU_EARTH) -> np.ndarray:
    """Periapsis state of an eccentric orbit, as a 6-vector."""
    r_p = a_m * (1.0 - e)
    v_p = math.sqrt(mu * (1.0 + e) / r_p)
    inc = math.radians(inc_deg)
    return np.array([r_p, 0.0, 0.0,
                     0.0, v_p * math.cos(inc), v_p * math.sin(inc)])


def period(a_m: float, mu: float = MU_EARTH) -> float:
    return 2.0 * math.pi * math.sqrt(a_m ** 3 / mu)


def sigma_pos_3d(cov: np.ndarray) -> float:
    """Root-sum-square 1-sigma position uncertainty from a 6x6 covariance."""
    return float(math.sqrt(np.trace(np.asarray(cov)[:3, :3])))
