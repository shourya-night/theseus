"""
Kepler's equation solver.

Solves  M = E − e sin E  (elliptic)
or      M = e sinh H − H  (hyperbolic)
for E (or H) given M and e, using Newton-Raphson iteration.

Convergence diagnostics are always returned.

References
----------
Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., §2.2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from theseus.core.trace import CalculationTrace, TraceContext


@dataclass
class KeplerSolution:
    """Result of solving Kepler's equation."""
    eccentric_anomaly: float   # E (rad) for elliptic, H for hyperbolic
    mean_anomaly: float        # M (rad)
    eccentricity: float        # e
    iterations: int
    residual: float
    converged: bool


def solve_kepler(
    M: float,
    e: float,
    tol: float = 1e-12,
    max_iter: int = 50,
    *,
    trace: bool = False,
) -> KeplerSolution:
    """
    Solve Kepler's equation for eccentric (or hyperbolic) anomaly.

    Parameters
    ----------
    M : float   Mean anomaly (rad).
    e : float   Eccentricity.
    tol : float Convergence tolerance on |f(E)|.
    max_iter : int  Maximum Newton-Raphson iterations.

    Returns
    -------
    KeplerSolution

    Method
    ------
    Newton-Raphson:  E_{n+1} = E_n − f(E_n)/f'(E_n)

    Elliptic (e < 1):
        f(E)  = E − e sin E − M
        f'(E) = 1 − e cos E

    Hyperbolic (e > 1):
        f(H)  = e sinh H − H − M
        f'(H) = e cosh H − 1
    """
    ct = None
    if trace:
        ct = CalculationTrace(
            operation="solve_kepler",
            equation="M = E − e sin E" if e < 1 else "M = e sinh H − H",
            inputs={"M_rad": M, "e": e, "tol": tol},
        )

    if e < 1.0:
        # --- Elliptic ---
        # Initial guess (smart start)
        E = M + e * math.sin(M)  # first-order approximation
        for iteration in range(1, max_iter + 1):
            f = E - e * math.sin(E) - M
            fp = 1.0 - e * math.cos(E)
            if abs(fp) < 1e-30:
                break
            dE = -f / fp
            E += dE
            if ct:
                ct.add_step("iteration", n=iteration, E=E, residual=abs(f))
            if abs(f) < tol:
                sol = KeplerSolution(
                    eccentric_anomaly=E % (2 * math.pi),
                    mean_anomaly=M,
                    eccentricity=e,
                    iterations=iteration,
                    residual=abs(f),
                    converged=True,
                )
                if ct:
                    ct.result = {"E_rad": sol.eccentric_anomaly, "converged": True, "iterations": iteration}
                    TraceContext.emit(ct)
                return sol
        # Did not converge
        sol = KeplerSolution(
            eccentric_anomaly=E % (2 * math.pi),
            mean_anomaly=M,
            eccentricity=e,
            iterations=max_iter,
            residual=abs(E - e * math.sin(E) - M),
            converged=False,
        )
        if ct:
            ct.result = {"E_rad": sol.eccentric_anomaly, "converged": False}
            TraceContext.emit(ct)
        return sol

    else:
        # --- Hyperbolic ---
        H = M  # initial guess
        for iteration in range(1, max_iter + 1):
            f = e * math.sinh(H) - H - M
            fp = e * math.cosh(H) - 1.0
            if abs(fp) < 1e-30:
                break
            dH = -f / fp
            H += dH
            if ct:
                ct.add_step("iteration", n=iteration, H=H, residual=abs(f))
            if abs(f) < tol:
                sol = KeplerSolution(
                    eccentric_anomaly=H,
                    mean_anomaly=M,
                    eccentricity=e,
                    iterations=iteration,
                    residual=abs(f),
                    converged=True,
                )
                if ct:
                    ct.result = {"H": H, "converged": True, "iterations": iteration}
                    TraceContext.emit(ct)
                return sol
        sol = KeplerSolution(
            eccentric_anomaly=H,
            mean_anomaly=M,
            eccentricity=e,
            iterations=max_iter,
            residual=abs(e * math.sinh(H) - H - M),
            converged=False,
        )
        if ct:
            ct.result = {"H": H, "converged": False}
            TraceContext.emit(ct)
        return sol


def eccentric_to_true(E: float, e: float) -> float:
    """
    Convert eccentric anomaly E to true anomaly ν (elliptic).

    tan(ν/2) = √((1+e)/(1−e)) tan(E/2)
    """
    return 2.0 * math.atan2(
        math.sqrt(1.0 + e) * math.sin(E / 2.0),
        math.sqrt(1.0 - e) * math.cos(E / 2.0),
    )


def hyperbolic_to_true(H: float, e: float) -> float:
    """
    Convert hyperbolic anomaly H to true anomaly ν.

    tan(ν/2) = √((e+1)/(e−1)) tanh(H/2)
    """
    return 2.0 * math.atan2(
        math.sqrt(e + 1.0) * math.sinh(H / 2.0),
        math.sqrt(e - 1.0) * math.cosh(H / 2.0),
    )
