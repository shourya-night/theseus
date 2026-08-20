"""
State Transition Matrix (STM) computation for THESEUS.

Implements the variational equations for orbit propagation:
    dΦ(t, t₀)/dt = A(t) Φ(t, t₀)
    Φ(t₀, t₀) = I₆

where A(t) = ∂f/∂x is the 6×6 dynamics Jacobian:
    A = |  0₃×₃       I₃×₃   |
        | ∂a/∂r      ∂a/∂v   |

Supports:
- Analytic Jacobians for point-mass gravity and J₂ oblateness
- Controlled numerical finite-difference Jacobians for composite force models
- Synchronized propagation of nominal trajectory and STM
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from theseus.propagation.integrators import RK4Integrator, RKF45Integrator, DerivFn


def gravity_jacobian(position: np.ndarray, mu: float) -> np.ndarray:
    """
    Analytic Jacobian of Newtonian point-mass gravitational acceleration ∂a_grav/∂r.

    a_grav = -μ / r³ * r

    ∂a/∂r = -μ/r³ * I₃ + 3μ/r⁵ * (r ⊗ r) = μ/r³ * (3 * r_hat ⊗ r_hat - I₃)

    Parameters
    ----------
    position : (3,) array
        Cartesian position vector (m).
    mu : float
        Gravitational parameter (m³/s²).

    Returns
    -------
    np.ndarray (3, 3)
        ∂a/∂r matrix (1/s²).
    """
    r_vec = np.asarray(position, dtype=np.float64)
    r2 = float(np.dot(r_vec, r_vec))
    r_mag = np.sqrt(r2)

    if r_mag < 1.0:
        return np.zeros((3, 3))

    r3 = r_mag * r2
    r5 = r3 * r2

    # (3μ/r⁵) * r r^T - (μ/r³) * I
    outer_rr = np.outer(r_vec, r_vec)
    da_dr = (3.0 * mu / r5) * outer_rr - (mu / r3) * np.eye(3)
    return da_dr


def j2_jacobian(position: np.ndarray, mu: float, j2: float, radius: float) -> np.ndarray:
    """
    Analytic Jacobian of J₂ perturbation acceleration ∂a_J2/∂r.

    Parameters
    ----------
    position : (3,) array
        Cartesian position vector (m).
    mu : float
        Gravitational parameter (m³/s²).
    j2 : float
        J₂ zonal harmonic coefficient.
    radius : float
        Reference equatorial radius (m).

    Returns
    -------
    np.ndarray (3, 3)
        ∂a_J2/∂r matrix (1/s²).
    """
    r_vec = np.asarray(position, dtype=np.float64)
    x, y, z = r_vec[0], r_vec[1], r_vec[2]
    r2 = float(np.dot(r_vec, r_vec))
    r_mag = np.sqrt(r2)

    if r_mag < 1.0 or j2 == 0.0:
        return np.zeros((3, 3))

    k = 1.5 * mu * j2 * (radius ** 2)
    r5 = r_mag ** 5
    r7 = r5 * r2
    r9 = r7 * r2
    z2 = z * z

    da_dr = np.zeros((3, 3), dtype=np.float64)

    # For i = 0 (x) and i = 1 (y):
    # a_i = k * x_i * (5 * z^2 * r^-7 - r^-5)
    # ∂a_i/∂x_j = k * [ δ_ij * (5*z^2*r^-7 - r^-5) + x_i * (-35*z^2*r^-9*x_j + 10*z*δ_3j*r^-7 + 5*x_j*r^-7) ]
    term_xy = 5.0 * z2 / r7 - 1.0 / r5

    # ∂ax/∂x
    da_dr[0, 0] = k * (term_xy + x * (-35.0 * z2 * x / r9 + 5.0 * x / r7))
    # ∂ax/∂y
    da_dr[0, 1] = k * (x * (-35.0 * z2 * y / r9 + 5.0 * y / r7))
    # ∂ax/∂z
    da_dr[0, 2] = k * (x * (-35.0 * z2 * z / r9 + 10.0 * z / r7 + 5.0 * z / r7))

    # ∂ay/∂x
    da_dr[1, 0] = k * (y * (-35.0 * z2 * x / r9 + 5.0 * x / r7))
    # ∂ay/∂y
    da_dr[1, 1] = k * (term_xy + y * (-35.0 * z2 * y / r9 + 5.0 * y / r7))
    # ∂ay/∂z
    da_dr[1, 2] = k * (y * (-35.0 * z2 * z / r9 + 10.0 * z / r7 + 5.0 * z / r7))

    # For i = 2 (z):
    # a_z = k * z * (5 * z^2 * r^-7 - 3 * r^-5)
    # ∂a_z/∂x_j = k * [ δ_3j * (5*z^2*r^-7 - 3*r^-5) + z * (-35*z^2*r^-9*x_j + 10*z*δ_3j*r^-7 + 15*x_j*r^-7) ]
    term_z = 5.0 * z2 / r7 - 3.0 / r5

    # ∂az/∂x
    da_dr[2, 0] = k * (z * (-35.0 * z2 * x / r9 + 15.0 * x / r7))
    # ∂az/∂y
    da_dr[2, 1] = k * (z * (-35.0 * z2 * y / r9 + 15.0 * y / r7))
    # ∂az/∂z
    da_dr[2, 2] = k * (term_z + z * (-35.0 * z2 * z / r9 + 10.0 * z / r7 + 15.0 * z / r7))

    return da_dr


def numerical_jacobian(
    acc_fn: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    t: float,
    r: np.ndarray,
    v: np.ndarray,
    dr: float = 1.0,      # 1 meter perturbation
    dv: float = 1e-3,     # 1 mm/s perturbation
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute ∂a/∂r and ∂a/∂v via central finite differences.

    Parameters
    ----------
    acc_fn : callable
        Function (t, r, v) -> a (3,) array.
    t : float
        Time (s).
    r : (3,) array
        Position (m).
    v : (3,) array
        Velocity (m/s).
    dr : float
        Position perturbation size (m).
    dv : float
        Velocity perturbation size (m/s).

    Returns
    -------
    (da_dr, da_dv) : tuple of (3, 3) arrays
    """
    da_dr = np.zeros((3, 3), dtype=np.float64)
    da_dv = np.zeros((3, 3), dtype=np.float64)

    # ∂a/∂r
    for i in range(3):
        r_plus = r.copy()
        r_minus = r.copy()
        r_plus[i] += dr
        r_minus[i] -= dr
        a_plus = acc_fn(t, r_plus, v)
        a_minus = acc_fn(t, r_minus, v)
        da_dr[:, i] = (a_plus - a_minus) / (2.0 * dr)

    # ∂a/∂v
    for i in range(3):
        v_plus = v.copy()
        v_minus = v.copy()
        v_plus[i] += dv
        v_minus[i] -= dv
        a_plus = acc_fn(t, r, v_plus)
        a_minus = acc_fn(t, r, v_minus)
        da_dv[:, i] = (a_plus - a_minus) / (2.0 * dv)

    return da_dr, da_dv


def build_dynamics_jacobian(
    da_dr: np.ndarray,
    da_dv: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Assemble the 6×6 Jacobian matrix A = ∂f/∂x:
        A = |  0₃×₃       I₃×₃   |
            | ∂a/∂r      ∂a/∂v   |
    """
    A = np.zeros((6, 6), dtype=np.float64)
    A[:3, 3:6] = np.eye(3)
    A[3:6, :3] = da_dr
    if da_dv is not None:
        A[3:6, 3:6] = da_dv
    return A


@dataclass
class STMResult:
    """
    State Transition Matrix computation result.

    Attributes
    ----------
    stm : np.ndarray
        6×6 State Transition Matrix Φ(tf, t0).
    nominal_state_tf : np.ndarray
        Nominal 6-element state [r, v] at tf.
    t0 : float
        Initial time (s).
    tf : float
        Final time (s).
    method : str
        Method used ('analytic_two_body', 'analytic_j2', 'numerical').
    history_times : np.ndarray | None
        Time steps if full history requested.
    history_stms : np.ndarray | None
        (N, 6, 6) array of STMs if full history requested.
    """
    stm: np.ndarray
    nominal_state_tf: np.ndarray
    t0: float
    tf: float
    method: str
    history_times: Optional[np.ndarray] = None
    history_stms: Optional[np.ndarray] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stm": self.stm.tolist(),
            "nominal_state_tf": self.nominal_state_tf.tolist(),
            "t0_s": float(self.t0),
            "tf_s": float(self.tf),
            "method": self.method,
            "determinant": float(np.linalg.det(self.stm)),
            "condition_number": float(np.linalg.cond(self.stm)),
        }


def propagate_stm(
    acc_fn: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    r0: np.ndarray,
    v0: np.ndarray,
    t_span: tuple[float, float],
    mu: Optional[float] = None,
    j2: Optional[float] = None,
    radius: Optional[float] = None,
    integrator: str = "rkf45",
    dt: float = 60.0,
    atol: float = 1e-11,
    rtol: float = 1e-11,
    use_analytic_jacobian: bool = True,
) -> STMResult:
    """
    Propagate the 6-state nominal trajectory and 6×6 STM simultaneously.

    Integrates the 42-element augmented state vector:
        Y = [r (3), v (3), vec(Φ) (36)]
    with initial condition:
        Y(t₀) = [r₀, v₀, vec(I₆)]

    Parameters
    ----------
    acc_fn : callable
        Acceleration function (t, r, v) -> a (3,) [m/s²].
    r0 : (3,) array
        Initial position (m).
    v0 : (3,) array
        Initial velocity (m/s).
    t_span : (t0, tf)
        Integration interval (s).
    mu : float, optional
        Gravitational parameter (m³/s²) for analytic Jacobian.
    j2 : float, optional
        J₂ zonal harmonic for analytic Jacobian.
    radius : float, optional
        Equatorial radius (m) for analytic J₂ Jacobian.
    integrator : str
        'rk4' or 'rkf45'.
    dt : float
        Step size (s).
    atol, rtol : float
        Tolerances for RKF45.
    use_analytic_jacobian : bool
        If True and mu is provided, use analytic gravity (+ J2) Jacobian.

    Returns
    -------
    STMResult
    """
    t0, tf = float(t_span[0]), float(t_span[1])
    r0 = np.asarray(r0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)

    # If t0 == tf, return identity immediately
    if abs(tf - t0) < 1e-12:
        return STMResult(
            stm=np.eye(6, dtype=np.float64),
            nominal_state_tf=np.concatenate([r0, v0]),
            t0=t0,
            tf=tf,
            method="identity_t0",
        )

    # Initial condition: state (6) + vec(I₆) (36)
    phi0 = np.eye(6, dtype=np.float64)
    y0 = np.concatenate([r0, v0, phi0.flatten()])

    can_use_analytic = (
        use_analytic_jacobian and (mu is not None and mu > 0.0)
    )

    method_str = "analytic_jacobian" if can_use_analytic else "numerical_jacobian"

    def augmented_deriv(t: float, y: np.ndarray) -> np.ndarray:
        r = y[:3]
        v = y[3:6]
        phi = y[6:42].reshape((6, 6))

        # Acceleration
        a = acc_fn(t, r, v)

        # Jacobian A = ∂f/∂x
        if can_use_analytic:
            da_dr = gravity_jacobian(r, mu)
            if j2 is not None and j2 != 0.0 and radius is not None:
                da_dr = da_dr + j2_jacobian(r, mu, j2, radius)
            da_dv = np.zeros((3, 3), dtype=np.float64)
        else:
            da_dr, da_dv = numerical_jacobian(acc_fn, t, r, v)

        A = build_dynamics_jacobian(da_dr, da_dv)

        # dΦ/dt = A * Φ
        dphi_dt = A @ phi

        dydt = np.empty(42, dtype=np.float64)
        dydt[:3] = v
        dydt[3:6] = a
        dydt[6:42] = dphi_dt.flatten()

        return dydt

    if integrator == "rk4":
        integ = RK4Integrator(dt=dt)
        res = integ.integrate(augmented_deriv, y0, (t0, tf))
    elif integrator == "rkf45":
        integ = RKF45Integrator(atol=atol, rtol=rtol, dt_initial=min(dt, abs(tf - t0)))
        res = integ.integrate(augmented_deriv, y0, (t0, tf))
    else:
        raise ValueError(f"Unknown integrator: {integrator!r}")

    final_y = res.states[-1]
    final_state = final_y[:6]
    final_stm = final_y[6:42].reshape((6, 6))

    history_stms = res.states[:, 6:42].reshape((-1, 6, 6))

    return STMResult(
        stm=final_stm,
        nominal_state_tf=final_state,
        t0=t0,
        tf=tf,
        method=f"{method_str}_{integrator}",
        history_times=res.times,
        history_stms=history_stms,
    )
