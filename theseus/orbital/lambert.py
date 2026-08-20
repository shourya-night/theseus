"""
Lambert problem solver — Universal-Variable method.

Solves the two-point boundary-value problem:
    Given r₁, r₂, Δt, μ,  find v₁, v₂.

Algorithm
---------
Uses the universal-variable / Stumpff-function formulation as presented
in Curtis, "Orbital Mechanics for Engineering Students", Algorithm 5.2.

1. Compute transfer angle Δθ from r₁, r₂ and transfer direction.
2. Compute auxiliary quantity A.
3. Iterate (Newton-Raphson) on the universal variable z to satisfy
   the time-of-flight equation:
       F(z) = [y(z)/C(z)]^(3/2) S(z) + A√y(z) − √μ Δt = 0
4. From z, compute Lagrange coefficients f, g, ḟ, ġ.
5. v₁ = (r₂ − f r₁) / g,   v₂ = (ġ r₂ − r₁) / g.

Stumpff functions (series valid for all z):
    C(z) = (1 − cos√z) / z           for z > 0
         = (cosh√(−z) − 1) / (−z)    for z < 0
         = 1/2                         for z ≈ 0
    S(z) = (√z − sin√z) / (√z)³      for z > 0
         = (sinh√(−z) − √(−z)) / (−z)^(3/2)  for z < 0
         = 1/6                         for z ≈ 0

References
----------
Curtis, "Orbital Mechanics for Engineering Students", 4th ed., §5.3.
Bate, Mueller, White, "Fundamentals of Astrodynamics", §5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from theseus.core.trace import CalculationTrace, TraceContext


@dataclass
class LambertSolution:
    """Result of solving the Lambert problem."""
    v1: np.ndarray          # departure velocity (m/s)
    v2: np.ndarray          # arrival velocity (m/s)
    transfer_angle: float   # Δθ (rad)
    iterations: int
    residual: float
    converged: bool
    z_final: float          # final universal variable
    trajectory_type: str    # 'elliptic', 'parabolic', or 'hyperbolic'


def _stumpff_c(z: float) -> float:
    """Stumpff function C(z)."""
    if z > 1e-6:
        sz = math.sqrt(z)
        return (1.0 - math.cos(sz)) / z
    elif z < -1e-6:
        sz = math.sqrt(-z)
        return (math.cosh(sz) - 1.0) / (-z)
    else:
        # Taylor series: C(z) = 1/2 − z/24 + z²/720 − …
        return 0.5 - z / 24.0 + z * z / 720.0


def _stumpff_s(z: float) -> float:
    """Stumpff function S(z)."""
    if z > 1e-6:
        sz = math.sqrt(z)
        return (sz - math.sin(sz)) / (sz ** 3)
    elif z < -1e-6:
        sz = math.sqrt(-z)
        return (math.sinh(sz) - sz) / (sz ** 3)
    else:
        # Taylor series: S(z) = 1/6 − z/120 + z²/5040 − …
        return 1.0 / 6.0 - z / 120.0 + z * z / 5040.0


def solve_lambert(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: float,
    mu: float,
    *,
    prograde: bool = True,
    max_iter: int = 100,
    tol: float = 1e-10,
    trace: bool = False,
) -> LambertSolution:
    """
    Solve Lambert's problem using the universal-variable method.

    Parameters
    ----------
    r1 : np.ndarray   Initial position vector (m).
    r2 : np.ndarray   Final position vector (m).
    tof : float        Time of flight (s).  Must be > 0.
    mu : float         Gravitational parameter (m³/s²).
    prograde : bool    True for prograde (short-way) transfer.
    max_iter : int     Maximum Newton-Raphson iterations.
    tol : float        Convergence tolerance on time-of-flight residual (s).

    Returns
    -------
    LambertSolution
    """
    if tof <= 0:
        raise ValueError(f"Time of flight must be positive, got {tof}")

    r1 = np.asarray(r1, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)

    r1_mag = float(np.linalg.norm(r1))
    r2_mag = float(np.linalg.norm(r2))

    if r1_mag < 1.0 or r2_mag < 1.0:
        raise ValueError(f"Position magnitudes too small: |r1|={r1_mag}, |r2|={r2_mag}")

    # --- Transfer angle & geometry ---
    cross = np.cross(r1, r2)
    cross_z = cross[2]
    cos_dtheta = float(np.dot(r1, r2)) / (r1_mag * r2_mag)
    cos_dtheta = max(-1.0, min(1.0, cos_dtheta))

    # Collinear same direction (0 deg or 360 deg) is degenerate
    if abs(1.0 - cos_dtheta) < 1e-12:
        raise ValueError(
            "Transfer angle is near 0° or 360°; two-point boundary problem is degenerate."
        )

    # Collinear opposite direction (180 deg Hohmann transfer)
    if abs(1.0 + cos_dtheta) < 1e-7:
        dtheta = math.pi
        a_t = (r1_mag + r2_mag) / 2.0
        if abs(cross_z) > 1e-10:
            h_dir = cross / np.linalg.norm(cross)
        else:
            h_dir = np.array([0.0, 0.0, 1.0 if prograde else -1.0])
        v1_dir = np.cross(h_dir, r1)
        v1_norm = float(np.linalg.norm(v1_dir))
        if v1_norm < 1e-10:
            v1_dir = np.array([0.0, 1.0 if prograde else -1.0, 0.0])
            v1_norm = 1.0
        v1_dir = v1_dir / v1_norm
        v1_speed = math.sqrt(max(0.0, mu * (2.0 / r1_mag - 1.0 / a_t)))
        v2_speed = math.sqrt(max(0.0, mu * (2.0 / r2_mag - 1.0 / a_t)))
        v1 = v1_speed * v1_dir
        v2 = -v2_speed * v1_dir
        return LambertSolution(
            v1=v1,
            v2=v2,
            transfer_angle=dtheta,
            iterations=1,
            residual=0.0,
            converged=True,
            z_final=0.0,
            trajectory_type="elliptic",
        )

    if prograde:
        dtheta = math.acos(cos_dtheta) if cross_z >= 0 else 2.0 * math.pi - math.acos(cos_dtheta)
    else:
        dtheta = math.acos(cos_dtheta) if cross_z < 0 else 2.0 * math.pi - math.acos(cos_dtheta)

    sin_dtheta = math.sin(dtheta)
    A = sin_dtheta * math.sqrt(r1_mag * r2_mag / (1.0 - cos_dtheta))

    def y_of_z(z_val: float) -> float:
        C = _stumpff_c(z_val)
        S = _stumpff_s(z_val)
        if C <= 1e-12:
            return -1.0
        return r1_mag + r2_mag + A * (z_val * S - 1.0) / math.sqrt(C)

    def tof_of_z(z_val: float) -> float:
        C = _stumpff_c(z_val)
        S = _stumpff_s(z_val)
        y = y_of_z(z_val)
        if y <= 0:
            return float("nan")
        x = math.sqrt(y / C)
        return (x ** 3 * S + A * math.sqrt(y)) / math.sqrt(mu)

    # Bracketing bounds: for single-revolution, z in (z_min, 4*pi^2)
    z_low = -4.0 * math.pi ** 2
    while y_of_z(z_low) <= 0:
        z_low = z_low * 0.5

    z_high = 4.0 * math.pi ** 2 - 1e-4

    # Safe Newton-Raphson with bisection fallback
    z = 0.0
    converged = False
    iteration = 0
    residual = float("inf")

    ct = None
    if trace:
        ct = CalculationTrace(
            operation="lambert_solver",
            equation="F(z) = [y/C]^(3/2) S + A√y − √μ Δt = 0",
            inputs={"r1_m": r1.tolist(), "r2_m": r2.tolist(),
                    "tof_s": tof, "mu": mu, "prograde": prograde},
        )

    for iteration in range(1, max_iter + 1):
        t_z = tof_of_z(z)
        if math.isnan(t_z):
            z = 0.5 * (z_low + z_high)
            continue

        residual = t_z - tof

        if ct:
            ct.add_step("iteration", n=iteration, z=z, tof_z=t_z,
                        residual=residual)

        if abs(residual) < tol:
            converged = True
            break

        if residual > 0:
            z_high = z
        else:
            z_low = z

        # Numerical derivative with central difference
        dz = 1e-6 * max(1.0, abs(z))
        t_z_plus = tof_of_z(z + dz)
        if math.isnan(t_z_plus):
            z = 0.5 * (z_low + z_high)
            continue

        df = (t_z_plus - t_z) / dz
        if abs(df) < 1e-20:
            z = 0.5 * (z_low + z_high)
            continue

        z_new = z - residual / df
        if z_low < z_new < z_high:
            z = z_new
        else:
            z = 0.5 * (z_low + z_high)

    if not converged:
        # If tolerance not strictly met within max_iter, check if residual is reasonable
        if abs(residual) < 1e-4:
            converged = True

    C = _stumpff_c(z)
    S = _stumpff_s(z)
    y = y_of_z(z)

    if y <= 0:
        raise RuntimeError(f"Lambert solver failed to find valid trajectory: y={y}")

    f = 1.0 - y / r1_mag
    g = A * math.sqrt(y / mu)
    g_dot = 1.0 - y / r2_mag

    if abs(g) < 1e-15:
        raise RuntimeError("Lambert solver encountered singular g coefficient.")

    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g

    # Trajectory type
    if z > 1e-6:
        traj_type = "elliptic"
    elif z < -1e-6:
        traj_type = "hyperbolic"
    else:
        traj_type = "parabolic"

    sol = LambertSolution(
        v1=v1,
        v2=v2,
        transfer_angle=dtheta,
        iterations=iteration,
        residual=abs(residual),
        converged=converged,
        z_final=z,
        trajectory_type=traj_type,
    )

    if ct:
        ct.result = {
            "v1_m_s": v1.tolist(),
            "v2_m_s": v2.tolist(),
            "converged": converged,
            "iterations": iteration,
            "z": z,
            "type": traj_type,
        }
        TraceContext.emit(ct)

    return sol
