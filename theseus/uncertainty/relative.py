"""
Relative covariance computation for two orbiting objects.

For relative state:
    x_rel = x₁ - x₂

The relative state covariance is:
    P_rel = P₁ + P₂ - P₁₂ - P₂₁

When cross-covariance terms P₁₂ and P₂₁ are unavailable, the objects are
treated as statistically independent:
    P_rel = P₁ + P₂
with this assumption explicitly recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from theseus.uncertainty.covariance import StateCovariance, CovarianceValidationError


@dataclass
class RelativeCovarianceResult:
    """
    Relative state covariance result.

    Attributes
    ----------
    relative_covariance : StateCovariance
        Combined 6×6 relative covariance matrix P_rel.
    cov_a : StateCovariance
        Object A covariance at encounter epoch.
    cov_b : StateCovariance
        Object B covariance at encounter epoch.
    independent : bool
        Whether objects were assumed statistically independent.
    assumptions : list[str]
        Explicit scientific assumptions.
    """
    relative_covariance: StateCovariance
    cov_a: StateCovariance
    cov_b: StateCovariance
    independent: bool = True
    assumptions: list[str] = field(default_factory=list)

    @property
    def position_covariance(self) -> np.ndarray:
        """3×3 relative position covariance (m²)."""
        return self.relative_covariance.position_covariance

    @property
    def velocity_covariance(self) -> np.ndarray:
        """3×3 relative velocity covariance (m²/s²)."""
        return self.relative_covariance.velocity_covariance

    @property
    def sigma_position(self) -> np.ndarray:
        """1-sigma relative position uncertainty [σ_x, σ_y, σ_z] (m)."""
        return self.relative_covariance.sigma_position

    @property
    def sigma_velocity(self) -> np.ndarray:
        """1-sigma relative velocity uncertainty [σ_vx, σ_vy, σ_vz] (m/s)."""
        return self.relative_covariance.sigma_velocity

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_covariance": self.relative_covariance.to_dict(),
            "independent": self.independent,
            "assumptions": self.assumptions,
            "sigma_rel_pos_3d_m": self.relative_covariance.sigma_pos_3d,
            "sigma_rel_pos_3d_km": self.relative_covariance.sigma_pos_3d / 1e3,
            "sigma_rel_vel_3d_m_s": self.relative_covariance.sigma_vel_3d,
            "sigma_rel_vel_3d_km_s": self.relative_covariance.sigma_vel_3d / 1e3,
        }


def compute_relative_covariance(
    cov_a: StateCovariance,
    cov_b: StateCovariance,
    cross_cov_ab: Optional[np.ndarray] = None,
    epoch_tol_s: float = 1.0,
) -> RelativeCovarianceResult:
    """
    Compute relative state covariance for two objects:
        P_rel = P_a + P_b - P_ab - P_ba

    Parameters
    ----------
    cov_a : StateCovariance
        Covariance of object A.
    cov_b : StateCovariance
        Covariance of object B.
    cross_cov_ab : (6, 6) array, optional
        Cross-covariance P_ab = E[(x_a - x̄_a)(x_b - x̄_b)^T].
    epoch_tol_s : float
        Tolerance for epoch matching (s).

    Returns
    -------
    RelativeCovarianceResult
    """
    # Verify frame consistency
    if cov_a.frame != cov_b.frame:
        raise CovarianceValidationError(
            f"FRAME MISMATCH: Object A frame is '{cov_a.frame}', but Object B frame is '{cov_b.frame}'. "
            f"Relative covariance cannot be formed across different reference frames."
        )

    # Verify epoch consistency
    if abs(cov_a.epoch_s - cov_b.epoch_s) > epoch_tol_s:
        raise CovarianceValidationError(
            f"EPOCH MISMATCH: Object A epoch is {cov_a.epoch_s:.3f} s, Object B epoch is {cov_b.epoch_s:.3f} s "
            f"(difference {abs(cov_a.epoch_s - cov_b.epoch_s):.3f} s exceeds tolerance {epoch_tol_s} s)."
        )

    assumptions = []
    independent = (cross_cov_ab is None)

    if independent:
        p_rel_mat = cov_a.matrix + cov_b.matrix
        assumptions.append("ASSUMPTION: OBJECT STATES TREATED AS STATISTICALLY INDEPENDENT (P_ab = 0)")
    else:
        p_ab = np.asarray(cross_cov_ab, dtype=np.float64)
        if p_ab.shape != (6, 6):
            raise CovarianceValidationError(f"Cross-covariance must be 6×6, got shape {p_ab.shape}")
        p_ba = p_ab.T
        p_rel_mat = cov_a.matrix + cov_b.matrix - p_ab - p_ba
        assumptions.append("Cross-covariance P_ab explicitly accounted for")

    # Symmetrize explicitly
    p_rel_mat = 0.5 * (p_rel_mat + p_rel_mat.T)

    rel_cov = StateCovariance(
        matrix=p_rel_mat,
        epoch_s=cov_a.epoch_s,
        frame=cov_a.frame,
        pos_units=cov_a.pos_units,
        vel_units=cov_a.vel_units,
        source="RELATIVE_COMBINATION",
        name=f"Rel({cov_a.name or 'A'}-{cov_b.name or 'B'})",
    )

    return RelativeCovarianceResult(
        relative_covariance=rel_cov,
        cov_a=cov_a,
        cov_b=cov_b,
        independent=independent,
        assumptions=assumptions,
    )
