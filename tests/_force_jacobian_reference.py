"""
Independent reference for force-Jacobian and STM tests.

The rule this module follows: it may call the *production acceleration*
(``CompositeForceModel.compute_acceleration``), because that is the model
under test, but it must never call a production Jacobian.  Every derivative
here comes from central differences of the acceleration itself, and every
reference STM comes either from those finite differences or from nonlinear
propagation of perturbed trajectories.

So a test comparing an engine Jacobian or STM against these values cannot
agree by sharing an algebraic mistake -- there is no algebra to share.
"""

from __future__ import annotations

import math

import numpy as np

from theseus.atmosphere.models import US1976StandardAtmosphere
from theseus.bodies.catalog import EARTH
from theseus.dynamics.drag import DragModel
from theseus.dynamics.force_model import CompositeForceModel
from theseus.dynamics.gravity import J2Perturbation, PointMassGravity
from theseus.propagation.integrators import RKF45Integrator


MU_EARTH = EARTH.mu
RE_EARTH = EARTH.radius
J2_EARTH = EARTH.J2

DEFAULT_MASS = 1000.0
DEFAULT_AREA = 10.0
DEFAULT_CD = 2.2
EARTH_ROTATION_RATE = 7.2921159e-5


def build_force_model(gravity: bool = True, j2: bool = True, drag: bool = False,
                      area: float = DEFAULT_AREA, cd: float = DEFAULT_CD):
    """
    Assemble the same force models ``MultiObjectEnvironment`` assembles.

    Mirrors ``_build_force_model``'s composition so a test exercises the real
    production acceleration, not a stand-in.
    """
    fm = CompositeForceModel()
    if gravity:
        fm.add(PointMassGravity(EARTH))
    if j2:
        fm.add(J2Perturbation(EARTH))
    if drag:
        fm.add(DragModel(atmosphere=US1976StandardAtmosphere(), cd=cd, area=area,
                         body_radius=RE_EARTH,
                         body_rotation_rate=EARTH_ROTATION_RATE))
    return fm


def acceleration_fn(force_model, mass: float = DEFAULT_MASS):
    """``(t, r, v) -> a`` for a composite force model, as the engine uses it."""
    return lambda t, r, v: force_model.compute_acceleration(t, r, v, mass)


def finite_difference_jacobian(acc_fn, t: float, r, v,
                               dr: float = 1.0, dv: float = 1e-3):
    """
    ``(da_dr, da_dv)`` by central differences of *acc_fn* itself.

    One state component is perturbed at a time; no Jacobian implementation
    from the engine is involved.
    """
    r = np.asarray(r, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    da_dr = np.zeros((3, 3), dtype=np.float64)
    da_dv = np.zeros((3, 3), dtype=np.float64)

    for i in range(3):
        r_plus, r_minus = r.copy(), r.copy()
        r_plus[i] += dr
        r_minus[i] -= dr
        da_dr[:, i] = (acc_fn(t, r_plus, v) - acc_fn(t, r_minus, v)) / (2.0 * dr)

        v_plus, v_minus = v.copy(), v.copy()
        v_plus[i] += dv
        v_minus[i] -= dv
        da_dv[:, i] = (acc_fn(t, r, v_plus) - acc_fn(t, r, v_minus)) / (2.0 * dv)

    return da_dr, da_dv


def full_jacobian(acc_fn, t: float, r, v, dr: float = 1.0, dv: float = 1e-3):
    """The 6x6 dynamics Jacobian A, assembled from the finite differences."""
    da_dr, da_dv = finite_difference_jacobian(acc_fn, t, r, v, dr, dv)
    a_matrix = np.zeros((6, 6), dtype=np.float64)
    a_matrix[:3, 3:] = np.eye(3)
    a_matrix[3:, :3] = da_dr
    a_matrix[3:, 3:] = da_dv
    return a_matrix


def propagate_nonlinear(acc_fn, x0, t0: float, tf: float,
                        atol: float = 1e-9, rtol: float = 1e-12) -> np.ndarray:
    """Nonlinear 6-state propagation under the production acceleration."""
    x0 = np.asarray(x0, dtype=np.float64)
    if tf == t0:
        return x0.copy()

    def deriv(t, y):
        return np.concatenate([y[3:6], acc_fn(t, y[:3], y[3:6])])

    result = RKF45Integrator(atol=atol, rtol=rtol,
                             dt_initial=min(1.0, abs(tf - t0))).integrate(
        deriv, x0, (t0, tf))
    return np.asarray(result.states[-1], dtype=np.float64)


def finite_difference_stm(acc_fn, x0, t0: float, tf: float,
                          dr: float = 5.0, dv: float = 5e-3) -> np.ndarray:
    """
    Phi(tf, t0) from twelve nonlinear propagations of perturbed states.

    This uses no variational equations and no Jacobian of any kind, so it is
    the most independent reference available for an STM.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    columns = []
    for i in range(6):
        step = dr if i < 3 else dv
        plus, minus = x0.copy(), x0.copy()
        plus[i] += step
        minus[i] -= step
        columns.append((propagate_nonlinear(acc_fn, plus, t0, tf)
                        - propagate_nonlinear(acc_fn, minus, t0, tf)) / (2.0 * step))
    return np.column_stack(columns)


def variational_stm(acc_fn, x0, t0: float, tf: float,
                    dr: float = 1.0, dv: float = 1e-3) -> np.ndarray:
    """
    Phi(tf, t0) by integrating the variational equations with the
    finite-difference Jacobian above.

    Independent of the engine's analytic Jacobians, but cheaper and smoother
    than :func:`finite_difference_stm`, so it suits the end-to-end covariance
    comparison.
    """
    from scipy.integrate import solve_ivp

    x0 = np.asarray(x0, dtype=np.float64)

    def deriv(t, y):
        r, v, phi = y[:3], y[3:6], y[6:42].reshape(6, 6)
        out = np.empty(42, dtype=np.float64)
        out[:3] = v
        out[3:6] = acc_fn(t, r, v)
        out[6:] = (full_jacobian(acc_fn, t, r, v, dr, dv) @ phi).ravel()
        return out

    sol = solve_ivp(deriv, (t0, tf), np.concatenate([x0, np.eye(6).ravel()]),
                    rtol=1e-11, atol=1e-9, method="DOP853")
    return sol.y[6:42, -1].reshape(6, 6)


def circular_state(altitude_km: float, inclination_deg: float = 51.6) -> np.ndarray:
    """Circular-orbit 6-state at the given altitude above the Earth's radius."""
    radius = RE_EARTH + altitude_km * 1e3
    speed = math.sqrt(MU_EARTH / radius)
    inc = math.radians(inclination_deg)
    return np.array([radius, 0.0, 0.0,
                     0.0, speed * math.cos(inc), speed * math.sin(inc)])


def relative_matrix_error(a, b) -> float:
    """Max-norm relative difference between two matrices."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = np.max(np.abs(b))
    if scale == 0.0:
        return float(np.max(np.abs(a)))
    return float(np.max(np.abs(a - b)) / scale)


def sigma_pos_3d(cov) -> float:
    return float(math.sqrt(np.trace(np.asarray(cov)[:3, :3])))
