"""
Independent reference solutions for conjunction tests.

Nothing in this module imports ``theseus.conjunction``.  Expected times of
closest approach and miss distances are obtained either in closed form
(rectilinear relative motion) or by direct minimisation of |r_a(t) - r_b(t)|
with SciPy, so a test that compares an engine result against these values is
not comparing the engine against itself.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar


MU_EARTH = 3.986004418e14      # m^3/s^2
R_LEO = 6778.137e3             # 400 km altitude


# ---------------------------------------------------------------------------
# Analytic trajectory builders
# ---------------------------------------------------------------------------

def circular_orbit(radius: float, phase_deg: float = 0.0, inc_deg: float = 0.0,
                   mu: float = MU_EARTH):
    """Position/velocity callables for a circular orbit with RAAN = 0."""
    n = math.sqrt(mu / radius ** 3)
    v_circ = math.sqrt(mu / radius)
    phi0 = math.radians(phase_deg)
    inc = math.radians(inc_deg)

    def pos_fn(t: float) -> np.ndarray:
        th = n * t + phi0
        return np.array([
            radius * math.cos(th),
            radius * math.sin(th) * math.cos(inc),
            radius * math.sin(th) * math.sin(inc),
        ])

    def vel_fn(t: float) -> np.ndarray:
        th = n * t + phi0
        return np.array([
            -v_circ * math.sin(th),
            v_circ * math.cos(th) * math.cos(inc),
            v_circ * math.cos(th) * math.sin(inc),
        ])

    return pos_fn, vel_fn


def rectilinear(p0, v) -> tuple:
    """
    Straight-line motion r(t) = p0 + v t.

    Used for adversarial screening cases because the closest approach of two
    rectilinear objects has an exact closed-form solution, so the expected
    answer owes nothing to any solver.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    def pos_fn(t: float) -> np.ndarray:
        return p0 + v * float(t)

    def vel_fn(t: float) -> np.ndarray:
        return v.copy()

    return pos_fn, vel_fn


def rectilinear_closest_approach(p_a, v_a, p_b, v_b):
    """
    Exact closest approach for two rectilinear trajectories.

    With dp = p_a - p_b and dv = v_a - v_b:

        d(t)^2 = |dp + dv t|^2
        d/dt d(t)^2 = 2 (dp + dv t) . dv = 0
        =>  t_ca = -(dp . dv) / |dv|^2

    Returns (t_ca, miss_distance).  t_ca is None when |dv| = 0, in which case
    the separation is constant and there is no closest approach.
    """
    dp = np.asarray(p_a, dtype=np.float64) - np.asarray(p_b, dtype=np.float64)
    dv = np.asarray(v_a, dtype=np.float64) - np.asarray(v_b, dtype=np.float64)
    dv2 = float(np.dot(dv, dv))
    if dv2 <= 0.0:
        return None, float(np.linalg.norm(dp))
    t_ca = -float(np.dot(dp, dv)) / dv2
    miss = float(np.linalg.norm(dp + dv * t_ca))
    return t_ca, miss


# ---------------------------------------------------------------------------
# Independent numerical reference
# ---------------------------------------------------------------------------

def independent_closest_approach(pos_a, pos_b, t_start: float, t_end: float,
                                 n_scan: int = 20000, n_candidates: int = 60):
    """
    Determine (t_ca, miss_distance) by direct minimisation of |r_a - r_b|.

    A single grid scan is not sufficient for high-relative-velocity encounters:
    at 15 km/s a 0.36 s grid step spans 5.4 km, so a metre-scale minimum is
    invisible to the grid.  The search is therefore hierarchical -- coarse scan,
    fine re-scan of the best cells, then a bounded golden-section close-out.
    """
    def dist(t: float) -> float:
        return float(np.linalg.norm(np.asarray(pos_a(t)) - np.asarray(pos_b(t))))

    ts = np.linspace(t_start, t_end, n_scan)
    ds = np.array([dist(t) for t in ts])
    step = ts[1] - ts[0]

    order = np.argsort(ds)[:n_candidates]
    candidate_idx = set(int(i) for i in order) | {0, len(ts) - 1}

    best_t, best_d = float(t_start), float("inf")
    for i in sorted(candidate_idx):
        lo = max(t_start, ts[i] - step)
        hi = min(t_end, ts[i] + step)
        if hi <= lo:
            continue
        sub = np.linspace(lo, hi, 400)
        sub_d = np.array([dist(t) for t in sub])
        j = int(np.argmin(sub_d))
        sub_lo = sub[max(j - 1, 0)]
        sub_hi = sub[min(j + 1, len(sub) - 1)]
        if sub_hi > sub_lo:
            res = minimize_scalar(dist, bounds=(sub_lo, sub_hi), method="bounded",
                                  options={"xatol": 1e-10})
            t_cand, d_cand = float(res.x), float(res.fun)
        else:
            t_cand, d_cand = float(sub[j]), float(sub_d[j])
        if d_cand < best_d:
            best_t, best_d = t_cand, d_cand

    return best_t, best_d


def min_distance_on_interval(pos_a, pos_b, t0: float, t1: float, n: int = 4000) -> float:
    """
    Tight lower reference for the minimum separation on a single interval.

    Dense scan plus a bounded refinement around the best sample.  Used to check
    that the screener's analytic bound never exceeds the true minimum.
    """
    def dist(t: float) -> float:
        return float(np.linalg.norm(np.asarray(pos_a(t)) - np.asarray(pos_b(t))))

    ts = np.linspace(t0, t1, n)
    ds = np.array([dist(t) for t in ts])
    k = int(np.argmin(ds))
    lo = ts[max(k - 1, 0)]
    hi = ts[min(k + 1, n - 1)]
    if hi <= lo:
        return float(ds[k])
    res = minimize_scalar(dist, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-9})
    return float(min(res.fun, ds[k]))
