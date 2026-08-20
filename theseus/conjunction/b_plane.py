"""
B-plane geometry for hyperbolic encounters.

The B-plane is the plane perpendicular to the asymptotic approach
velocity vector, passing through the centre of the target body (or
secondary object).

The B-vector lies in this plane and connects the target centre to
the point where the asymptotic approach velocity line intersects
the B-plane.

    Ŝ = v_∞ / |v_∞|            asymptotic approach direction
    T̂ = Ŝ × ẑ / |Ŝ × ẑ|       transverse B-plane axis
    R̂ = Ŝ × T̂                  radial B-plane axis

    B = r_rel − (r_rel · Ŝ) Ŝ  B-vector (projection into B-plane)
    B·T = B · T̂
    B·R = B · R̂

Frame convention: ẑ = celestial pole ([0, 0, 1] in ICRF).

Reference: Kizner, "A Method of Describing Miss Distances for
Lunar and Interplanetary Trajectories", 1961.

Applicability
-------------
Classical B-plane analysis is meaningful ONLY for:
- Hyperbolic encounters (excess velocity v_∞ > 0)
- Fly-by geometries with well-defined asymptotic velocity

It is NOT applicable to:
- Bound (elliptical/circular) orbit encounters
- Very low relative velocity encounters where the "asymptote"
  concept breaks down
- Conjunction screening between two LEO objects on similar orbits
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class BPlaneResult:
    """
    B-plane analysis result.

    Attributes
    ----------
    applicable : bool
        Whether B-plane analysis is physically meaningful for this encounter.
    reason : str
        Explanation of applicability determination.
    b_vector : np.ndarray | None
        B-vector (m) in the B-plane.
    b_magnitude : float
        |B| (m).
    b_dot_t : float
        B-plane transverse component (m).
    b_dot_r : float
        B-plane radial component (m).
    s_hat : np.ndarray | None
        Approach direction unit vector.
    t_hat : np.ndarray | None
        Transverse basis vector.
    r_hat : np.ndarray | None
        Radial basis vector.
    frame : str
        Reference frame description.
    assumptions : list[str]
        Model assumptions.
    """
    applicable: bool
    reason: str
    b_vector: Optional[np.ndarray] = None
    b_magnitude: float = 0.0
    b_dot_t: float = 0.0
    b_dot_r: float = 0.0
    s_hat: Optional[np.ndarray] = None
    t_hat: Optional[np.ndarray] = None
    r_hat: Optional[np.ndarray] = None
    frame: str = "ICRF (celestial pole = ẑ)"
    assumptions: list[str] = None  # type: ignore

    def __post_init__(self):
        if self.assumptions is None:
            self.assumptions = []

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "applicable": self.applicable,
            "reason": self.reason,
            "frame": self.frame,
        }
        if self.applicable and self.b_vector is not None:
            d.update({
                "b_vector_m": self.b_vector.tolist(),
                "b_magnitude_m": self.b_magnitude,
                "b_magnitude_km": self.b_magnitude / 1e3,
                "b_dot_t_m": self.b_dot_t,
                "b_dot_t_km": self.b_dot_t / 1e3,
                "b_dot_r_m": self.b_dot_r,
                "b_dot_r_km": self.b_dot_r / 1e3,
                "s_hat": self.s_hat.tolist() if self.s_hat is not None else None,
                "t_hat": self.t_hat.tolist() if self.t_hat is not None else None,
                "r_hat": self.r_hat.tolist() if self.r_hat is not None else None,
                "assumptions": self.assumptions,
            })
        else:
            d["note"] = "B-PLANE ANALYSIS NOT APPLICABLE TO THIS ENCOUNTER"
        return d


# Minimum relative velocity for B-plane to be meaningful (m/s)
# Below this, the "asymptotic" concept breaks down.
MIN_V_INF_FOR_BPLANE: float = 100.0  # 100 m/s


def compute_b_plane(
    r_rel: np.ndarray,
    v_rel: np.ndarray,
    v_inf_threshold: float = MIN_V_INF_FOR_BPLANE,
    pole: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> BPlaneResult:
    """
    Compute B-plane parameters for an encounter.

    Parameters
    ----------
    r_rel : (3,) array
        Relative position vector at TCA (m).  r_rel = r₁ − r₂.
    v_rel : (3,) array
        Relative velocity vector at TCA (m/s).  v_rel = v₁ − v₂.
    v_inf_threshold : float
        Minimum |v_rel| for B-plane to be considered applicable (m/s).
    pole : (3,) array
        Reference pole direction.  Default: celestial north [0, 0, 1].

    Returns
    -------
    BPlaneResult
    """
    r_rel = np.asarray(r_rel, dtype=np.float64)
    v_rel = np.asarray(v_rel, dtype=np.float64)
    pole = np.asarray(pole, dtype=np.float64)

    v_rel_mag = float(np.linalg.norm(v_rel))

    # Check applicability
    if v_rel_mag < v_inf_threshold:
        return BPlaneResult(
            applicable=False,
            reason=(
                f"Relative velocity |v_rel| = {v_rel_mag:.1f} m/s is below "
                f"the threshold ({v_inf_threshold:.1f} m/s) for meaningful "
                f"B-plane analysis. The asymptotic-velocity concept requires "
                f"a hyperbolic-like encounter geometry."
            ),
        )

    # Ŝ = approach direction
    s_hat = v_rel / v_rel_mag

    # T̂ = Ŝ × ẑ / |Ŝ × ẑ|  (transverse in B-plane)
    s_cross_z = np.cross(s_hat, pole)
    s_cross_z_mag = float(np.linalg.norm(s_cross_z))

    if s_cross_z_mag < 1e-10:
        # Approach direction is nearly parallel to the pole
        # Use an alternative reference direction
        alt_pole = np.array([1.0, 0.0, 0.0])
        s_cross_z = np.cross(s_hat, alt_pole)
        s_cross_z_mag = float(np.linalg.norm(s_cross_z))
        if s_cross_z_mag < 1e-10:
            return BPlaneResult(
                applicable=False,
                reason="Cannot construct B-plane basis: approach direction is degenerate",
            )

    t_hat = s_cross_z / s_cross_z_mag

    # R̂ = Ŝ × T̂
    r_hat = np.cross(s_hat, t_hat)

    # B = r_rel − (r_rel · Ŝ) Ŝ  (project out the approach-direction component)
    b_vector = r_rel - np.dot(r_rel, s_hat) * s_hat
    b_magnitude = float(np.linalg.norm(b_vector))

    # B·T and B·R
    b_dot_t = float(np.dot(b_vector, t_hat))
    b_dot_r = float(np.dot(b_vector, r_hat))

    return BPlaneResult(
        applicable=True,
        reason="B-plane analysis is applicable: sufficient relative velocity for asymptotic geometry",
        b_vector=b_vector,
        b_magnitude=b_magnitude,
        b_dot_t=b_dot_t,
        b_dot_r=b_dot_r,
        s_hat=s_hat,
        t_hat=t_hat,
        r_hat=r_hat,
        frame="ICRF (celestial pole = ẑ = [0,0,1])",
        assumptions=[
            "Asymptotic approach velocity approximated by v_rel at TCA",
            "B-vector computed from r_rel projection at TCA",
            "Kizner (1961) B-plane definition",
        ],
    )
