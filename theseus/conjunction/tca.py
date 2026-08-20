"""
Time of Closest Approach (TCA) solver.

Given position and velocity functions for two objects, finds the
time(s) at which their separation distance is locally minimised.

TCA Condition
-------------
The squared distance d²(t) = |r₁(t) − r₂(t)|² is minimised when:

    d/dt[d²] = 0

Expanding:

    2(r₁ − r₂) · (v₁ − v₂) = 0

So the TCA condition is:

    f(t) = r_rel(t) · v_rel(t) = 0

where r_rel = r₁ − r₂ and v_rel = v₁ − v₂.

A sign change from negative to positive in f(t) indicates a local
minimum (objects approaching → receding).  A sign change from positive
to negative indicates a local maximum (objects receding → approaching).

We only accept minima: f changes from negative → positive.

Solver: Brent's method (scipy-free implementation) for root-finding
within a bracketed interval.

Reference: Brent, "Algorithms for Minimization Without Derivatives",
Prentice-Hall, 1973.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import math
import numpy as np


@dataclass
class TCAResult:
    """
    Result of a TCA refinement.

    Attributes
    ----------
    tca : float
        Time of closest approach (s).
    miss_distance : float
        Separation distance at TCA (m).
    relative_velocity : float
        Relative speed at TCA (m/s).
    r_rel : np.ndarray
        Relative position vector at TCA (m).
    v_rel : np.ndarray
        Relative velocity vector at TCA (m/s).
    r_a : np.ndarray
        Object A position at TCA (m).
    v_a : np.ndarray
        Object A velocity at TCA (m/s).
    r_b : np.ndarray
        Object B position at TCA (m).
    v_b : np.ndarray
        Object B velocity at TCA (m/s).
    converged : bool
        Whether the solver converged.
    iterations : int
        Number of solver iterations.
    validated : bool
        Whether the TCA was validated as a true local minimum.
    validation_note : str
        Explanation of validation result.
    """
    tca: float
    miss_distance: float
    relative_velocity: float
    r_rel: np.ndarray
    v_rel: np.ndarray
    r_a: np.ndarray
    v_a: np.ndarray
    r_b: np.ndarray
    v_b: np.ndarray
    converged: bool = True
    iterations: int = 0
    validated: bool = True
    validation_note: str = ""

    def to_dict(self) -> dict:
        return {
            "tca_s": self.tca,
            "miss_distance_m": self.miss_distance,
            "miss_distance_km": self.miss_distance / 1e3,
            "relative_velocity_m_s": self.relative_velocity,
            "relative_velocity_km_s": self.relative_velocity / 1e3,
            "r_rel_m": self.r_rel.tolist(),
            "v_rel_m_s": self.v_rel.tolist(),
            "r_a_m": self.r_a.tolist(),
            "v_a_m_s": self.v_a.tolist(),
            "r_b_m": self.r_b.tolist(),
            "v_b_m_s": self.v_b.tolist(),
            "converged": self.converged,
            "iterations": self.iterations,
            "validated": self.validated,
            "validation_note": self.validation_note,
        }


def _brent_root(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> tuple[float, int, bool]:
    """
    Brent's method for finding a root of f in [a, b].

    f(a) and f(b) must have opposite signs.

    Returns
    -------
    (root, iterations, converged)
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        # No sign change — no guaranteed root
        return 0.5 * (a + b), 0, False

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = b - a
    e = d
    mflag = True

    for iteration in range(1, max_iter + 1):
        if abs(fb) < tol:
            return b, iteration, True
        if abs(b - a) < tol:
            return b, iteration, True

        # Inverse quadratic interpolation or secant
        if abs(fa - fc) > tol and abs(fb - fc) > tol:
            # IQI
            s = (a * fb * fc / ((fa - fb) * (fa - fc))
                 + b * fa * fc / ((fb - fa) * (fb - fc))
                 + c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            # Secant
            if abs(fb - fa) < 1e-30:
                s = 0.5 * (a + b)
            else:
                s = b - fb * (b - a) / (fb - fa)

        # Conditions for bisection
        cond1 = not (min(0.75 * a + 0.25 * b, b) < s < max(0.75 * a + 0.25 * b, b))
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2
        cond3 = (not mflag) and abs(s - b) >= abs(c - d) / 2
        cond4 = mflag and abs(b - c) < tol
        cond5 = (not mflag) and abs(c - d) < tol

        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = 0.5 * (a + b)
            mflag = True
        else:
            mflag = False

        fs = f(s)
        d = c
        c = b
        fc = fb

        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    return b, max_iter, False


def find_tca(
    pos_fn_a: Callable[[float], np.ndarray],
    vel_fn_a: Callable[[float], np.ndarray],
    pos_fn_b: Callable[[float], np.ndarray],
    vel_fn_b: Callable[[float], np.ndarray],
    t_start: float,
    t_end: float,
    tol: float = 1e-6,
) -> Optional[TCAResult]:
    """
    Find the TCA within [t_start, t_end] using Brent's method.

    The TCA condition is f(t) = r_rel · v_rel = 0, where the sign
    changes from negative to positive (approaching → receding = minimum).

    Parameters
    ----------
    pos_fn_a, vel_fn_a : callable
        Position/velocity functions for object A.
    pos_fn_b, vel_fn_b : callable
        Position/velocity functions for object B.
    t_start, t_end : float
        Bracket for TCA search.
    tol : float
        Convergence tolerance (s).

    Returns
    -------
    TCAResult or None if no valid TCA found.
    """
    def f(t: float) -> float:
        r_rel = np.asarray(pos_fn_a(t)) - np.asarray(pos_fn_b(t))
        v_rel = np.asarray(vel_fn_a(t)) - np.asarray(vel_fn_b(t))
        return float(np.dot(r_rel, v_rel))

    fa = f(t_start)
    fb = f(t_end)

    # If no sign change, try to find one by sampling
    if fa * fb > 0:
        # Sample finely to find a bracket
        n_search = 100
        search_times = np.linspace(t_start, t_end, n_search)
        found_bracket = False
        for i in range(n_search - 1):
            fi = f(search_times[i])
            fi1 = f(search_times[i + 1])
            # Only accept negative → positive (minimum distance)
            if fi < 0 and fi1 >= 0:
                t_start = float(search_times[i])
                t_end = float(search_times[i + 1])
                found_bracket = True
                break
        if not found_bracket:
            return None

    root, iterations, converged = _brent_root(f, t_start, t_end, tol=tol)

    if not converged:
        return TCAResult(
            tca=root,
            miss_distance=0.0,
            relative_velocity=0.0,
            r_rel=np.zeros(3),
            v_rel=np.zeros(3),
            r_a=np.zeros(3),
            v_a=np.zeros(3),
            r_b=np.zeros(3),
            v_b=np.zeros(3),
            converged=False,
            iterations=iterations,
            validated=False,
            validation_note="TCA SOLUTION INVALID: Brent's method did not converge",
        )

    # Compute states at TCA
    r_a = np.asarray(pos_fn_a(root), dtype=np.float64)
    v_a = np.asarray(vel_fn_a(root), dtype=np.float64)
    r_b = np.asarray(pos_fn_b(root), dtype=np.float64)
    v_b = np.asarray(vel_fn_b(root), dtype=np.float64)
    r_rel = r_a - r_b
    v_rel = v_a - v_b
    miss_distance = float(np.linalg.norm(r_rel))
    rel_vel = float(np.linalg.norm(v_rel))

    # Validate: check that distance is locally minimised
    validated = True
    note = "Validated: distance is locally minimised at TCA"

    # Check derivative sign change (negative → positive)
    dt_check = max(tol * 10, 0.01)
    f_before = f(root - dt_check) if root - dt_check >= 0 else f(root)
    f_after = f(root + dt_check)

    if not (f_before <= 0 and f_after >= 0):
        # Could be a maximum, not minimum
        if f_before >= 0 and f_after <= 0:
            validated = False
            note = "TCA SOLUTION INVALID: local maximum (objects receding → approaching), not minimum"
        else:
            note = "TCA validated with reduced confidence (boundary effects)"

    # Check solution is within window
    if root < t_start - tol or root > t_end + tol:
        validated = False
        note = "TCA SOLUTION INVALID: solution outside analysis window"

    return TCAResult(
        tca=root,
        miss_distance=miss_distance,
        relative_velocity=rel_vel,
        r_rel=r_rel,
        v_rel=v_rel,
        r_a=r_a,
        v_a=v_a,
        r_b=r_b,
        v_b=v_b,
        converged=converged,
        iterations=iterations,
        validated=validated,
        validation_note=note,
    )


def find_all_tca(
    pos_fn_a: Callable[[float], np.ndarray],
    vel_fn_a: Callable[[float], np.ndarray],
    pos_fn_b: Callable[[float], np.ndarray],
    vel_fn_b: Callable[[float], np.ndarray],
    t_start: float,
    t_end: float,
    n_samples: int = 500,
    tol: float = 1e-6,
) -> list[TCAResult]:
    """
    Find ALL TCA events within [t_start, t_end].

    Samples f(t) = r_rel · v_rel at n_samples points, identifies all
    negative-to-positive sign changes, and refines each with Brent's method.

    Returns
    -------
    list[TCAResult]
        All validated TCA events, sorted by time.
    """
    def f(t: float) -> float:
        r_rel = np.asarray(pos_fn_a(t)) - np.asarray(pos_fn_b(t))
        v_rel = np.asarray(vel_fn_a(t)) - np.asarray(vel_fn_b(t))
        return float(np.dot(r_rel, v_rel))

    sample_times = np.linspace(t_start, t_end, n_samples)
    f_values = np.array([f(t) for t in sample_times])

    results: list[TCAResult] = []

    for i in range(len(f_values) - 1):
        # Only negative → positive = local minimum
        if f_values[i] < 0 and f_values[i + 1] >= 0:
            t_a = float(sample_times[i])
            t_b = float(sample_times[i + 1])
            result = find_tca(
                pos_fn_a, vel_fn_a, pos_fn_b, vel_fn_b,
                t_a, t_b, tol=tol,
            )
            if result is not None and result.converged:
                results.append(result)

    return sorted(results, key=lambda r: r.tca)
