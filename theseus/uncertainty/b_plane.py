"""
B-plane uncertainty projection for close encounters.

Projects 3D relative position covariance into the 2D B-plane (encounter plane)
orthogonal to the relative velocity vector at TCA:
    P_B = M P_rr Mᵀ

where M = [T̂ᵀ; R̂ᵀ] is the 2×3 projection matrix.

Outputs:
- 2×2 B-plane covariance matrix P_B
- σ_T, σ_R, cov_TR, and correlation coefficient ρ
- 1-sigma uncertainty ellipse parameters (semi-major axis, semi-minor axis, angle)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from theseus.conjunction.b_plane import BPlaneResult, compute_b_plane


@dataclass
class BPlaneUncertainty:
    """
    Uncertainty representation projected into the 2D B-plane.

    Attributes
    ----------
    b_plane_covariance : np.ndarray
        2×2 covariance matrix in [B·T, B·R] coordinates (m²).
    b_dot_t : float
        Nominal transverse miss component (m).
    b_dot_r : float
        Nominal radial miss component (m).
    sigma_t : float
        1-sigma uncertainty in T-direction (m).
    sigma_r : float
        1-sigma uncertainty in R-direction (m).
    cov_tr : float
        Covariance between T and R components (m²).
    correlation : float
        Correlation coefficient ρ = cov_tr / (σ_t * σ_r).
    sigma_major : float
        Semi-major axis of 1-sigma uncertainty ellipse (m).
    sigma_minor : float
        Semi-minor axis of 1-sigma uncertainty ellipse (m).
    ellipse_angle_deg : float
        Orientation angle of major axis from T̂ toward R̂ (degrees).
    ellipse_angle_rad : float
        Orientation angle (radians).
    eigenvalues : np.ndarray
        Eigenvalues of P_B [λ_min, λ_max] (m²).
    eigenvectors : np.ndarray
        Eigenvectors of P_B (2×2 column vectors).
    b_plane_result : BPlaneResult
        Underlying Phase 9 B-plane geometry result.
    """
    b_plane_covariance: np.ndarray
    b_dot_t: float
    b_dot_r: float
    sigma_t: float
    sigma_r: float
    cov_tr: float
    correlation: float
    sigma_major: float
    sigma_minor: float
    ellipse_angle_deg: float
    ellipse_angle_rad: float
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    b_plane_result: BPlaneResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "b_dot_t_m": float(self.b_dot_t),
            "b_dot_r_m": float(self.b_dot_r),
            "b_dot_t_km": float(self.b_dot_t / 1e3),
            "b_dot_r_km": float(self.b_dot_r / 1e3),
            "b_plane_covariance_m2": self.b_plane_covariance.tolist(),
            "b_plane_covariance_km2": (self.b_plane_covariance / 1e6).tolist(),
            "sigma_t_m": float(self.sigma_t),
            "sigma_r_m": float(self.sigma_r),
            "sigma_t_km": float(self.sigma_t / 1e3),
            "sigma_r_km": float(self.sigma_r / 1e3),
            "cov_tr_m2": float(self.cov_tr),
            "cov_tr_km2": float(self.cov_tr / 1e6),
            "correlation": float(self.correlation),
            "sigma_major_m": float(self.sigma_major),
            "sigma_minor_m": float(self.sigma_minor),
            "sigma_major_km": float(self.sigma_major / 1e3),
            "sigma_minor_km": float(self.sigma_minor / 1e3),
            "ellipse_angle_deg": float(self.ellipse_angle_deg),
            "ellipse_angle_rad": float(self.ellipse_angle_rad),
            "eigenvalues_m2": self.eigenvalues.tolist(),
        }


def project_covariance_to_b_plane(
    rel_pos_cov: np.ndarray,
    r_rel: np.ndarray,
    v_rel: np.ndarray,
    b_plane_result: Optional[BPlaneResult] = None,
    pole: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> BPlaneUncertainty:
    """
    Project 3×3 relative position covariance onto the 2D B-plane.

    Parameters
    ----------
    rel_pos_cov : (3, 3) array
        Relative position covariance matrix P_rr (m²).
    r_rel : (3,) array
        Relative position vector at TCA (m).
    v_rel : (3,) array
        Relative velocity vector at TCA (m/s).
    b_plane_result : BPlaneResult, optional
        Pre-computed BPlaneResult from Phase 9.
    pole : (3,) array
        Celestial reference pole.

    Returns
    -------
    BPlaneUncertainty
    """
    rel_pos_cov = np.asarray(rel_pos_cov, dtype=np.float64)
    r_rel = np.asarray(r_rel, dtype=np.float64)
    v_rel = np.asarray(v_rel, dtype=np.float64)

    if b_plane_result is None or not b_plane_result.applicable:
        b_plane_result = compute_b_plane(r_rel, v_rel, pole=pole)

    if not b_plane_result.applicable or b_plane_result.t_hat is None or b_plane_result.r_hat is None:
        # Fallback: construct orthonormal basis directly around v_rel
        v_mag = float(np.linalg.norm(v_rel))
        if v_mag < 1e-10:
            s_hat = np.array([1.0, 0.0, 0.0])
        else:
            s_hat = v_rel / v_mag

        t_cross = np.cross(s_hat, pole)
        t_mag = float(np.linalg.norm(t_cross))
        if t_mag < 1e-6:
            t_cross = np.cross(s_hat, np.array([1.0, 0.0, 0.0]))
            t_mag = float(np.linalg.norm(t_cross))
        t_hat = t_cross / t_mag
        r_hat = np.cross(s_hat, t_hat)
        b_dot_t = float(np.dot(r_rel, t_hat))
        b_dot_r = float(np.dot(r_rel, r_hat))
    else:
        t_hat = b_plane_result.t_hat
        r_hat = b_plane_result.r_hat
        b_dot_t = b_plane_result.b_dot_t
        b_dot_r = b_plane_result.b_dot_r

    # Projection matrix M = [T̂ᵀ; R̂ᵀ] (2×3)
    M = np.vstack([t_hat, r_hat])  # shape (2, 3)

    # 2×2 B-plane covariance: P_B = M P_rr Mᵀ
    P_B = M @ rel_pos_cov @ M.T

    # Symmetrize explicitly
    P_B = 0.5 * (P_B + P_B.T)

    sigma_t2 = max(0.0, float(P_B[0, 0]))
    sigma_r2 = max(0.0, float(P_B[1, 1]))
    cov_tr = float(P_B[0, 1])

    sigma_t = math.sqrt(sigma_t2)
    sigma_r = math.sqrt(sigma_r2)

    denom = sigma_t * sigma_r
    correlation = (cov_tr / denom) if denom > 1e-15 else 0.0
    correlation = max(-1.0, min(1.0, correlation))

    # Eigendecomposition of 2×2 matrix
    eigvals, eigvecs = np.linalg.eigh(P_B)
    eigvals = np.maximum(0.0, eigvals)

    # Sort so index 1 is major, index 0 is minor
    idx_sort = np.argsort(eigvals)
    lam_min = float(eigvals[idx_sort[0]])
    lam_max = float(eigvals[idx_sort[1]])
    v_major = eigvecs[:, idx_sort[1]]

    sigma_major = math.sqrt(lam_max)
    sigma_minor = math.sqrt(lam_min)

    # Ellipse orientation angle θ relative to T-axis
    ellipse_angle_rad = math.atan2(v_major[1], v_major[0])
    ellipse_angle_deg = math.degrees(ellipse_angle_rad)

    return BPlaneUncertainty(
        b_plane_covariance=P_B,
        b_dot_t=b_dot_t,
        b_dot_r=b_dot_r,
        sigma_t=sigma_t,
        sigma_r=sigma_r,
        cov_tr=cov_tr,
        correlation=correlation,
        sigma_major=sigma_major,
        sigma_minor=sigma_minor,
        ellipse_angle_deg=ellipse_angle_deg,
        ellipse_angle_rad=ellipse_angle_rad,
        eigenvalues=np.array([lam_min, lam_max]),
        eigenvectors=eigvecs,
        b_plane_result=b_plane_result,
    )
