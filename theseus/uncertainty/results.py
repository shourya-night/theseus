"""
Uncertainty analysis orchestration, complete result containers, and calculation traces.

Generates the complete 14-step structured calculation trace from:
    INPUT STATES & COVARIANCES
             ↓
        VALIDATION
             ↓
     PROPAGATION (STM)
             ↓
        TCA SOLVING
             ↓
     RELATIVE COVARIANCE
             ↓
     B-PLANE PROJECTION
             ↓
      HARD-BODY RADIUS
             ↓
  PROBABILITY OF COLLISION
             ↓
      RISK ASSESSMENT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from theseus.conjunction.analysis import ConjunctionAnalysis, ConjunctionResult, ConjunctionEvent
from theseus.uncertainty.covariance import StateCovariance, CovarianceValidationError
from theseus.uncertainty.state_transition import propagate_stm
from theseus.uncertainty.propagation import propagate_covariance, ProcessNoiseModel
from theseus.uncertainty.relative import compute_relative_covariance, RelativeCovarianceResult
from theseus.uncertainty.b_plane import project_covariance_to_b_plane, BPlaneUncertainty
from theseus.uncertainty.hard_body import compute_hard_body_radius, HardBodyResult, CollisionGeometry
from theseus.uncertainty.collision_probability import compute_collision_probability, CollisionProbabilityResult
from theseus.uncertainty.risk import (
    classify_risk, indeterminate_risk, RiskAssessment, RiskThresholds, PROFILE_STANDARD,
)


# Analysis outcome codes.  These appear verbatim in the serialised result so
# that a consumer can branch on the outcome without inspecting numbers.
STATUS_COMPLETE = "COMPLETE"
STATUS_INDETERMINATE = "INDETERMINATE_NO_CONJUNCTION"


@dataclass
class UncertaintyConjunctionResult:
    """
    Complete Phase 10 result combining nominal conjunction geometry with
    state uncertainty, B-plane covariance, Pc, risk, and calculation traces.

    Two outcomes are possible and they are never conflated:

    ``analysis_status == STATUS_COMPLETE``
        A valid time of closest approach was found and the full pipeline ran.
        Every Phase 10 product below is populated.

    ``analysis_status == STATUS_INDETERMINATE``
        No valid TCA was found inside the analysis window.  ``tca_s``,
        ``miss_distance_m``, the propagated covariances, the B-plane
        projection and the collision probability are all None, and the risk
        assessment is INDETERMINATE with ``action_required=False``.  This is
        deliberately *not* a low-risk result -- it means the encounter was
        not evaluated.

    Attributes
    ----------
    conjunction_result : ConjunctionResult
        Underlying deterministic Phase 9 result.
    initial_cov_a, initial_cov_b : StateCovariance
        Input covariances.  Always populated.
    conjunction_found : bool
        Whether a valid TCA was located inside the window.
    analysis_status : str
        STATUS_COMPLETE or STATUS_INDETERMINATE.
    indeterminate_reason : str
        Why the analysis could not be completed, when applicable.
    tca_s : float | None
        Time of closest approach (s).  None when indeterminate.
    miss_distance_m : float | None
        |r_rel| at TCA (m) -- the true three-dimensional separation, not the
        B-plane projection |b0|.  None when indeterminate.
    relative_velocity_m_s : float | None
        |v_rel| at TCA (m/s).  None when indeterminate.
    cov_a_tca, cov_b_tca : StateCovariance | None
        Covariances propagated to TCA.
    relative_covariance : RelativeCovarianceResult | None
        Relative covariance at TCA.
    b_plane_uncertainty : BPlaneUncertainty | None
        Projected B-plane uncertainty and ellipse.
    hard_body_result : HardBodyResult | None
        Hard-body collision radius model.
    collision_probability : CollisionProbabilityResult | None
        Evaluated probability of collision Pc.
    risk_assessment : RiskAssessment | None
        Risk classification and operational recommendations.
    calculation_steps : list[dict[str, Any]]
        Structured calculation trace: 14 steps when complete, a short
        explicit trace when indeterminate.
    model_metadata : dict[str, Any]
        Physical, mathematical, and numerical metadata.
    """
    conjunction_result: ConjunctionResult
    initial_cov_a: StateCovariance
    initial_cov_b: StateCovariance
    conjunction_found: bool = False
    analysis_status: str = STATUS_INDETERMINATE
    indeterminate_reason: str = ""
    tca_s: Optional[float] = None
    miss_distance_m: Optional[float] = None
    relative_velocity_m_s: Optional[float] = None
    cov_a_tca: Optional[StateCovariance] = None
    cov_b_tca: Optional[StateCovariance] = None
    relative_covariance: Optional[RelativeCovarianceResult] = None
    b_plane_uncertainty: Optional[BPlaneUncertainty] = None
    hard_body_result: Optional[HardBodyResult] = None
    collision_probability: Optional[CollisionProbabilityResult] = None
    risk_assessment: Optional[RiskAssessment] = None
    calculation_steps: list[dict[str, Any]] = field(default_factory=list)
    model_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise the analysis.

        Every quantity that depends on a valid time of closest approach is
        ``None`` when no conjunction was found.  ``analysis_status`` states
        which case this is, so a consumer can never mistake an unevaluated
        encounter for an evaluated safe one.
        """
        summary: dict[str, Any] = {
            "tca_s": None if self.tca_s is None else float(self.tca_s),
            "miss_distance_km": (
                None if self.miss_distance_m is None else float(self.miss_distance_m) / 1e3
            ),
            "miss_distance_m": (
                None if self.miss_distance_m is None else float(self.miss_distance_m)
            ),
            "relative_velocity_km_s": (
                None if self.relative_velocity_m_s is None
                else float(self.relative_velocity_m_s) / 1e3
            ),
            "conjunction_found": bool(self.conjunction_found),
        }

        if self.collision_probability is not None:
            pc_block = self.collision_probability.to_dict()
        else:
            pc_block = {
                "probability": None,
                "probability_scientific": None,
                "method": "NOT_COMPUTED",
                "converged": False,
                "reason": self.indeterminate_reason,
            }

        return {
            "analysis_status": self.analysis_status,
            "conjunction_found": bool(self.conjunction_found),
            "indeterminate_reason": self.indeterminate_reason,
            "conjunction_summary": summary,
            "initial_cov_a": self.initial_cov_a.to_dict(),
            "initial_cov_b": self.initial_cov_b.to_dict(),
            "cov_a_tca": None if self.cov_a_tca is None else self.cov_a_tca.to_dict(),
            "cov_b_tca": None if self.cov_b_tca is None else self.cov_b_tca.to_dict(),
            "relative_covariance": (
                None if self.relative_covariance is None else self.relative_covariance.to_dict()
            ),
            "b_plane_uncertainty": (
                None if self.b_plane_uncertainty is None else self.b_plane_uncertainty.to_dict()
            ),
            "hard_body": (
                None if self.hard_body_result is None else self.hard_body_result.to_dict()
            ),
            "collision_probability": pc_block,
            "risk_assessment": (
                self.risk_assessment.to_dict() if self.risk_assessment is not None
                else indeterminate_risk(self.indeterminate_reason).to_dict()
            ),
            "calculation_steps": self.calculation_steps,
            "model_metadata": self.model_metadata,
            "deterministic_conjunction": self.conjunction_result.to_dict(),
        }


def measure_covariance_validity(cov: StateCovariance) -> dict[str, Any]:
    """
    Measure the three properties trace step 2 claims, without changing them.

    ``StateCovariance.validate()`` cannot serve here: it raises rather than
    reporting, and it *repairs* -- zeroing slightly negative diagonals,
    symmetrising within tolerance and clipping small negative eigenvalues -- so
    by the time a caller could ask it a question the answer is always yes.
    This inspects the matrix as it stands.

    The comparisons use the covariance's own declared ``sym_tol`` and
    ``psd_tol``, applied to the same dimensionless quantities ``validate()``
    tests, so the trace and the class agree on what "valid" means.

    Both are normalised per entry by ``sqrt(P_ii P_jj)`` — the asymmetry and
    the minimum eigenvalue are those of the correlation form.  Before P10-11
    this mirrored a raw-matrix tolerance scaled by the largest entry of the
    6×6, which for a state covariance is a position variance in m², and which
    therefore judged velocity-block defects against an unrelated block's
    magnitude.  Mirroring that is what this function has always done; the
    criterion it mirrors is what changed.

    Returns the measured quantities alongside the three boolean verdicts, so a
    reader can see what was compared rather than being told a conclusion.
    """
    matrix = np.asarray(cov.matrix, dtype=np.float64)

    finite = bool(np.all(np.isfinite(matrix)))
    variances = np.diag(matrix) if finite else np.full(6, float("nan"))
    positive = variances > 0.0 if finite else np.zeros(6, dtype=bool)

    max_asymmetry = float(np.max(np.abs(matrix - matrix.T))) if finite else float("inf")
    if finite and np.count_nonzero(positive) >= 2:
        deviations = np.sqrt(variances[positive])
        relative_asymmetry = float(np.max(
            np.abs(matrix - matrix.T)[np.ix_(positive, positive)]
            / np.outer(deviations, deviations)))
    else:
        relative_asymmetry = 0.0 if max_asymmetry == 0.0 else float("inf")
    symmetric = bool(finite and relative_asymmetry <= cov.sym_tol)

    min_variance = float(np.min(variances)) if finite else float("-inf")
    non_negative_variances = bool(finite and min_variance >= 0.0)

    # A component with zero variance may not covary with anything:
    # |P_ij|² ≤ P_ii P_jj = 0.  The correlation form has no row to normalise
    # there, so it is checked separately, exactly as validate() does.
    zero_variance_coupling = 0.0
    if finite:
        for i in np.flatnonzero(~positive):
            row = np.abs(matrix[i]).copy()
            row[i] = 0.0
            zero_variance_coupling = max(zero_variance_coupling, float(np.max(row)))

    if finite and np.any(positive):
        deviations = np.sqrt(variances[positive])
        correlation = matrix[np.ix_(positive, positive)] / np.outer(deviations, deviations)
        min_eigenvalue = float(np.min(np.linalg.eigvalsh(
            0.5 * (correlation + correlation.T))))
    else:
        min_eigenvalue = 0.0 if finite else float("-inf")
    # A negative variance settles the question on its own: e_iᵀ P e_i = P_ii < 0
    # means P is not positive semi-definite, and the correlation form cannot see
    # it because that row is excluded from the normalisation.
    positive_semidefinite = bool(
        finite
        and non_negative_variances
        and min_eigenvalue >= -cov.psd_tol
        and zero_variance_coupling == 0.0
    )

    return {
        "name": cov.name,
        "finite": finite,
        "max_asymmetry": max_asymmetry,
        "relative_asymmetry": relative_asymmetry,
        "symmetry_tolerance": cov.sym_tol,
        "symmetric": symmetric,
        "min_variance": min_variance,
        "non_negative_variances": non_negative_variances,
        "min_eigenvalue": min_eigenvalue,
        "min_eigenvalue_basis": "correlation form D⁻¹PD⁻¹",
        "zero_variance_coupling": zero_variance_coupling,
        "psd_tolerance": cov.psd_tol,
        "positive_semidefinite": positive_semidefinite,
        "valid": bool(symmetric and non_negative_variances and positive_semidefinite),
    }


def build_phase10_calculation_trace(
    initial_cov_a: StateCovariance,
    initial_cov_b: StateCovariance,
    cov_a_tca: StateCovariance,
    cov_b_tca: StateCovariance,
    rel_cov: RelativeCovarianceResult,
    tca_s: float,
    r_rel_tca: np.ndarray,
    v_rel_tca: np.ndarray,
    b_plane_unc: BPlaneUncertainty,
    hbr_res: HardBodyResult,
    pc_res: CollisionProbabilityResult,
    risk: RiskAssessment,
) -> list[dict[str, Any]]:
    """Build the complete 14-step Phase 10 structured calculation trace."""
    steps = []

    # STEP 01: Acquire Object A Covariance
    steps.append({
        "stepIndex": 1,
        "phase": "PHASE_10_UNCERTAINTY",
        "title": "Acquire Object A State Covariance",
        "status": "completed",
        "equation": "P₁(t₀) = E[(x₁ - x̄₁)(x₁ - x̄₁)^T]",
        "substitutions": {
            "source": initial_cov_a.source,
            "frame": initial_cov_a.frame,
            "sigma_pos_km": (initial_cov_a.sigma_position / 1e3).tolist(),
            "sigma_vel_km_s": (initial_cov_a.sigma_velocity / 1e3).tolist(),
            "sigma_3d_pos_km": initial_cov_a.sigma_pos_3d / 1e3,
        },
        "result": f"σ_pos(3D) = {initial_cov_a.sigma_pos_3d/1e3:.3f} km, σ_vel(3D) = {initial_cov_a.sigma_vel_3d/1e3:.4f} km/s",
        "units": "km, km/s",
        "explanation": "State covariance matrix of Object A characterizing 6-DOF tracking and state uncertainty.",
        "beginnerExplanation": "The size and shape of the initial uncertainty cloud around Object A.",
    })

    # STEP 02: Validate Covariances
    #
    # The three fields below used to be the literals True, True, True, and the
    # result line read "COVARIANCES MATHEMATICALLY VALIDATED" whatever the
    # matrices contained.  The equation named three tests -- symmetry,
    # non-negative eigenvalues, non-negative variances -- none of which this
    # step performed.  Two matrices each asymmetric by 4.0e+03, whose
    # asymmetries cancel in their sum and so pass the downstream relative
    # covariance construction, reached this step and were reported as verified.
    # They are now measured.
    validity_a = measure_covariance_validity(initial_cov_a)
    validity_b = measure_covariance_validity(initial_cov_b)
    both_valid = validity_a["valid"] and validity_b["valid"]
    steps.append({
        "stepIndex": 2,
        "phase": "PHASE_10_UNCERTAINTY",
        "title": "Validate Input Covariances",
        "status": "completed" if both_valid else "warning",
        "equation": "P = P^T,  λ_i(P) ≥ 0,  diag(P) ≥ 0",
        "substitutions": {
            "symmetry_verified": bool(validity_a["symmetric"] and validity_b["symmetric"]),
            "psd_verified": bool(validity_a["positive_semidefinite"]
                                 and validity_b["positive_semidefinite"]),
            "non_negative_variances": bool(validity_a["non_negative_variances"]
                                           and validity_b["non_negative_variances"]),
            "object_a": validity_a,
            "object_b": validity_b,
        },
        "result": (
            "COVARIANCES MATHEMATICALLY VALIDATED" if both_valid else
            "COVARIANCE VALIDATION FAILED — "
            + "; ".join(
                f"{v['name'] or label}: "
                + ", ".join(
                    problem for problem, failed in (
                        (f"asymmetric by {v['max_asymmetry']:.3e} "
                         f"(relative {v['relative_asymmetry']:.3e} > "
                         f"{v['symmetry_tolerance']:.3e})", not v["symmetric"]),
                        (f"negative variance {v['min_variance']:.3e}",
                         not v["non_negative_variances"]),
                        (f"negative eigenvalue {v['min_eigenvalue']:.3e}",
                         not v["positive_semidefinite"]),
                        ("non-finite entries", not v["finite"]),
                    ) if failed
                )
                for label, v in (("Object A", validity_a), ("Object B", validity_b))
                if not v["valid"]
            )
        ),
        "explanation": (
            "Symmetry, non-negative variances and positive semi-definiteness are "
            "measured here on the matrices as supplied, against each covariance's "
            "own declared tolerances. StateCovariance.validate() cannot answer this "
            "question on its own behalf: it repairs what it can and raises on the "
            "rest, so it never has a negative verdict to report."
        ),
        "beginnerExplanation": (
            "Checked that the uncertainty matrices represent valid physical "
            "probabilities."
            if both_valid else
            "One of the uncertainty matrices is not a valid probability "
            "description, so every number derived from it below is suspect."
        ),
    })

    # STEP 03: Propagate Object A Covariance to TCA
    steps.append({
        "stepIndex": 3,
        "phase": "PHASE_10_UNCERTAINTY",
        "title": "Propagate Object A Covariance to TCA",
        "status": "completed",
        "equation": "P₁(TCA) = Φ₁(TCA, t₀) P₁(t₀) Φ₁(TCA, t₀)^T",
        "substitutions": {
            "tca_s": tca_s,
            "initial_sigma_pos_km": initial_cov_a.sigma_pos_3d / 1e3,
            "propagated_sigma_pos_km": cov_a_tca.sigma_pos_3d / 1e3,
        },
        "result": f"P₁(TCA) σ_pos = {cov_a_tca.sigma_pos_3d/1e3:.3f} km",
        "units": "km",
        "explanation": "Propagated state covariance along the nonlinear trajectory using the State Transition Matrix.",
        "beginnerExplanation": "How the uncertainty cloud of Object A expands and rotates over time until closest approach.",
    })

    # STEP 04: Acquire Object B Covariance
    steps.append({
        "stepIndex": 4,
        "phase": "PHASE_10_UNCERTAINTY",
        "title": "Acquire Object B State Covariance",
        "status": "completed",
        "equation": "P₂(t₀) = E[(x₂ - x̄₂)(x₂ - x̄₂)^T]",
        "substitutions": {
            "source": initial_cov_b.source,
            "frame": initial_cov_b.frame,
            "sigma_pos_km": (initial_cov_b.sigma_position / 1e3).tolist(),
            "sigma_vel_km_s": (initial_cov_b.sigma_velocity / 1e3).tolist(),
            "sigma_3d_pos_km": initial_cov_b.sigma_pos_3d / 1e3,
        },
        "result": f"σ_pos(3D) = {initial_cov_b.sigma_pos_3d/1e3:.3f} km, σ_vel(3D) = {initial_cov_b.sigma_vel_3d/1e3:.4f} km/s",
        "units": "km, km/s",
        "explanation": "State covariance matrix of Object B characterizing initial tracking uncertainty.",
        "beginnerExplanation": "The initial uncertainty cloud around Object B.",
    })

    # STEP 05: Propagate Object B Covariance to TCA
    steps.append({
        "stepIndex": 5,
        "phase": "PHASE_10_UNCERTAINTY",
        "title": "Propagate Object B Covariance to TCA",
        "status": "completed",
        "equation": "P₂(TCA) = Φ₂(TCA, t₀) P₂(t₀) Φ₂(TCA, t₀)^T",
        "substitutions": {
            "tca_s": tca_s,
            "initial_sigma_pos_km": initial_cov_b.sigma_pos_3d / 1e3,
            "propagated_sigma_pos_km": cov_b_tca.sigma_pos_3d / 1e3,
        },
        "result": f"P₂(TCA) σ_pos = {cov_b_tca.sigma_pos_3d/1e3:.3f} km",
        "units": "km",
        "explanation": "Propagated state covariance of Object B to the Time of Closest Approach.",
        "beginnerExplanation": "How Object B's uncertainty grows until closest approach.",
    })

    # STEP 06: Construct Relative Covariance
    steps.append({
        "stepIndex": 6,
        "phase": "PHASE_10_UNCERTAINTY",
        "title": "Construct Relative Covariance",
        "status": "completed",
        "equation": "P_rel = P₁(TCA) + P₂(TCA) - P₁₂ - P₂₁",
        "substitutions": {
            "independent_assumed": rel_cov.independent,
            "sigma_rel_pos_km": (rel_cov.sigma_position / 1e3).tolist(),
            "sigma_rel_pos_3d_km": rel_cov.relative_covariance.sigma_pos_3d / 1e3,
        },
        "result": f"σ_rel(3D) = {rel_cov.relative_covariance.sigma_pos_3d/1e3:.3f} km",
        "units": "km",
        "explanation": "Combined relative uncertainty between the two objects under statistical independence.",
        "beginnerExplanation": "Adding the two uncertainty clouds together to see how uncertain their relative distance is.",
    })

    # STEP 07: Evaluate Relative State at TCA
    miss_dist = float(np.linalg.norm(r_rel_tca))
    rel_vel = float(np.linalg.norm(v_rel_tca))
    steps.append({
        "stepIndex": 7,
        "phase": "PHASE_10_UNCERTAINTY",
        "title": "Evaluate Nominal State at TCA",
        "status": "completed",
        "equation": "d(TCA) = |r₁(TCA) - r₂(TCA)|,  v_rel(TCA) = |v₁(TCA) - v₂(TCA)|",
        "substitutions": {
            "tca_s": tca_s,
            "miss_distance_km": miss_dist / 1e3,
            "relative_velocity_km_s": rel_vel / 1e3,
        },
        "result": f"Miss Distance = {miss_dist/1e3:.4f} km, Relative Velocity = {rel_vel/1e3:.3f} km/s",
        "units": "km, km/s",
        "explanation": "Nominal closest approach separation distance and relative velocity vector.",
        "beginnerExplanation": "How far apart the nominal predictions pass each other and how fast.",
    })

    # STEP 08: Construct B-Plane Basis
    steps.append({
        "stepIndex": 8,
        "phase": "PHASE_10_UNCERTAINTY",
        "title": "Construct B-Plane Basis",
        "status": "completed",
        "equation": "Ŝ = v_rel / |v_rel|,  T̂ = Ŝ × ẑ / |Ŝ × ẑ|,  R̂ = Ŝ × T̂",
        "substitutions": {
            "b_dot_t_km": b_plane_unc.b_dot_t / 1e3,
            "b_dot_r_km": b_plane_unc.b_dot_r / 1e3,
        },
        "result": f"B·T = {b_plane_unc.b_dot_t/1e3:.3f} km, B·R = {b_plane_unc.b_dot_r/1e3:.3f} km",
        "units": "km",
        "explanation": "Constructed orthonormal Kizner B-plane coordinate frame perpendicular to relative velocity.",
        "beginnerExplanation": "Setting up a 2D coordinate screen perpendicular to the approach flight direction.",
    })

    # STEP 09: Project Covariance into B-Plane
    steps.append({
        "stepIndex": 9,
        "phase": "PHASE_10_UNCERTAINTY",
        "title": "Project Covariance into B-Plane",
        "status": "completed",
        "equation": "P_B = M P_rel,pos M^T,  where M = [T̂^T; R̂^T]",
        "substitutions": {
            "sigma_T_km": b_plane_unc.sigma_t / 1e3,
            "sigma_R_km": b_plane_unc.sigma_r / 1e3,
            "correlation": b_plane_unc.correlation,
        },
        "result": f"σ_T = {b_plane_unc.sigma_t/1e3:.3f} km, σ_R = {b_plane_unc.sigma_r/1e3:.3f} km, ρ = {b_plane_unc.correlation:.3f}",
        "units": "km",
        "explanation": "Projected the 3D relative position covariance onto the 2D encounter B-plane.",
        "beginnerExplanation": "Flattening the 3D uncertainty cloud onto the 2D encounter screen.",
    })

    # STEP 10: Calculate Uncertainty Ellipse
    steps.append({
        "stepIndex": 10,
        "phase": "PHASE_10_UNCERTAINTY",
        "title": "Calculate Uncertainty Ellipse",
        "status": "completed",
        "equation": "det(P_B - λ I) = 0,  σ_major = √λ_max,  σ_minor = √λ_min",
        "substitutions": {
            "sigma_major_km": b_plane_unc.sigma_major / 1e3,
            "sigma_minor_km": b_plane_unc.sigma_minor / 1e3,
            "ellipse_angle_deg": b_plane_unc.ellipse_angle_deg,
        },
        "result": f"σ_major = {b_plane_unc.sigma_major/1e3:.3f} km, σ_minor = {b_plane_unc.sigma_minor/1e3:.3f} km, θ = {b_plane_unc.ellipse_angle_deg:.1f}°",
        "units": "km, deg",
        "explanation": "Eigendecomposition of 2×2 B-plane covariance yielding principal 1-sigma uncertainty axes.",
        "beginnerExplanation": "Calculating the tilt and major/minor dimensions of the 2D uncertainty oval.",
    })

    # STEP 11: Calculate Hard-Body Radius
    steps.append({
        "stepIndex": 11,
        "phase": "PHASE_10_UNCERTAINTY",
        "title": "Calculate Combined Hard-Body Radius",
        "status": "completed",
        "equation": "HBR = R₁ + R₂",
        "substitutions": {
            "R1_m": hbr_res.object_a.collision_radius_m,
            "R2_m": hbr_res.object_b.collision_radius_m,
            "combined_HBR_m": hbr_res.combined_hbr_m,
        },
        "result": f"HBR = {hbr_res.combined_hbr_m:.1f} m ({hbr_res.combined_hbr_m/1e3:.4f} km)",
        "units": "m",
        "explanation": "Combined collision cross-section radius representing physical spacecraft dimensions and solar arrays.",
        "beginnerExplanation": "The total size of the combined collision target if both objects touch.",
    })

    # STEP 12: Evaluate Collision Probability (Pc)
    steps.append({
        "stepIndex": 12,
        "phase": "PHASE_10_UNCERTAINTY",
        "title": "Evaluate Probability of Collision (Pc)",
        "status": "completed",
        "equation": "Pc = ∬_{|z-b|≤HBR} 1/(2π √|P_B|) exp(-½ z^T P_B^-1 z) dz",
        "substitutions": {
            "method": pc_res.method,
            "hard_body_radius_m": pc_res.hard_body_radius_m,
            "miss_distance_m": pc_res.miss_distance_m,
            "raw_probability": pc_res.probability,
        },
        "result": f"Pc = {pc_res.probability:.6e}",
        "units": "probability [0, 1]",
        "explanation": "Integrated the 2D Gaussian probability density over the circular collision disk.",
        "beginnerExplanation": "The total chance that the two objects collide within their uncertainty clouds.",
    })

    # STEP 13: Numerical Convergence Diagnostics
    #
    # This step used to state "error_estimate <= tolerance" as its equation
    # while the flag it reported was the literal True on every path, so it
    # announced success whatever the quadrature had done -- and the criterion
    # it named was never evaluated, because dblquad's error estimate was bound
    # and discarded.  It now reports the criterion that is actually applied,
    # and says so in both directions.
    pc_diagnostics = pc_res.diagnostics or {}
    steps.append({
        "stepIndex": 13,
        "phase": "PHASE_10_UNCERTAINTY",
        "title": "Check Numerical Stability & Convergence",
        "status": "completed" if pc_res.converged else "warning",
        "equation": "|Pc_i − Pc_j| / max(|Pc_i|, |Pc_j|) ≤ agreement tolerance, "
                    "for at least one pair of three independent evaluations",
        "substitutions": {
            "converged": pc_res.converged,
            "tolerance": pc_res.tolerance,
            "evaluations": pc_res.iterations,
            "condition_number": pc_res.condition_number,
            "method": pc_res.method,
            "quadrature_error_estimate": pc_diagnostics.get("quadrature_error_estimate"),
            "verification_probability": pc_diagnostics.get("verification_probability"),
            "verification_disagreement": pc_diagnostics.get("verification_disagreement"),
            "verification_settled": pc_diagnostics.get("verification_settled"),
            # P10-12: all three constructions, so the verdict can be re-derived
            # rather than taken on trust.
            "certified": pc_diagnostics.get("certified"),
            "certificate_source": pc_diagnostics.get("certificate_source"),
            "polar_quadrature_probability": pc_diagnostics.get("raw_probability"),
            "polar_quadrature_agrees": pc_diagnostics.get("polar_quadrature_agrees"),
            "polar_quadrature_superseded": pc_diagnostics.get(
                "polar_quadrature_superseded"),
            "reduction_minor_axis_probability": pc_diagnostics.get(
                "reduction_minor_axis_probability"),
            "reduction_minor_axis_settled": pc_diagnostics.get(
                "reduction_minor_axis_settled"),
            "reduction_major_axis_probability": pc_diagnostics.get(
                "reduction_major_axis_probability"),
            "reduction_major_axis_settled": pc_diagnostics.get(
                "reduction_major_axis_settled"),
            "convergence_criterion": pc_diagnostics.get("convergence_criterion"),
        },
        "result": (
            f"Converged = {pc_res.converged} (tol = {pc_res.tolerance:.1e}, "
            f"iterations = {pc_res.iterations})"
        ),
        "explanation": (
            "Convergence is decided by agreement between three independent "
            "evaluations of the same encounter-plane integral, not by the "
            "quadrature's own error estimate: on a density whose ridge is much "
            "narrower than the collision disk, adaptive two-dimensional quadrature "
            "can step over the ridge and return a confidently wrong value with a "
            "small error estimate. Where the quadrature is contradicted by two "
            "constructions that agree with each other, the value it produced is "
            "discarded and the method field names what replaced it. "
            + (pc_diagnostics.get("convergence_note") or "")
        ),
        "beginnerExplanation": (
            "Independent calculations of the same probability agree, so the "
            "number above can be trusted."
            if pc_res.converged else
            "Independent calculations of the same probability disagree, so the "
            "number above is NOT reliable and should not be acted on."
        ),
    })

    # STEP 14: Classify Risk
    steps.append({
        "stepIndex": 14,
        "phase": "PHASE_10_UNCERTAINTY",
        "title": "Classify Conjunction Risk",
        "status": "completed",
        "equation": f"Pc = {risk.probability:.2e} vs thresholds [Low: {risk.thresholds.low_threshold:.1e}, Elevated: {risk.thresholds.elevated_threshold:.1e}, High: {risk.thresholds.high_threshold:.1e}]",
        "substitutions": {
            "risk_level": risk.level.value,
            "action_required": risk.action_required,
            "threshold_profile": risk.thresholds.name,
        },
        "result": f"RISK LEVEL: {risk.level.value} — Action Required: {risk.action_required}",
        "explanation": risk.recommendation,
        "beginnerExplanation": f"This conjunction is classified as {risk.level.value} risk. {'A collision avoidance maneuver is recommended.' if risk.action_required else 'No immediate maneuver is needed.'}",
    })

    return steps


def build_indeterminate_calculation_trace(
    initial_cov_a: StateCovariance,
    initial_cov_b: StateCovariance,
    t_start: float,
    t_end: float,
    screening_threshold_m: float,
    coarse_dt: float,
    candidate_interval_count: int,
    reason: str,
) -> list[dict[str, Any]]:
    """
    Build the short, explicit trace for an analysis that produced no TCA.

    The trace deliberately stops at the point where the pipeline stopped.  It
    does not continue with placeholder numbers, and it states plainly that the
    absence of a result is not a safety finding.
    """
    return [
        {
            "stepIndex": 1,
            "phase": "PHASE_10_UNCERTAINTY",
            "title": "Acquire Input Covariances",
            "status": "completed",
            "equation": "P₁(t₀), P₂(t₀)",
            "substitutions": {
                "sigma_a_pos_3d_km": initial_cov_a.sigma_pos_3d / 1e3,
                "sigma_b_pos_3d_km": initial_cov_b.sigma_pos_3d / 1e3,
            },
            "result": (
                f"σ_A = {initial_cov_a.sigma_pos_3d/1e3:.3f} km, "
                f"σ_B = {initial_cov_b.sigma_pos_3d/1e3:.3f} km"
            ),
            "units": "km",
            "explanation": "Input state covariances for both objects were accepted and validated.",
            "beginnerExplanation": "We read in how uncertain each object's position is.",
        },
        {
            "stepIndex": 2,
            "phase": "PHASE_10_UNCERTAINTY",
            "title": "Locate Time of Closest Approach",
            "status": "failed",
            "equation": "(r₁ − r₂) · (v₁ − v₂) = 0",
            "substitutions": {
                "window_start_s": t_start,
                "window_end_s": t_end,
                "screening_threshold_km": screening_threshold_m / 1e3,
                "coarse_dt_s": coarse_dt,
                "candidate_intervals": candidate_interval_count,
                "validated_tca_count": 0,
            },
            "result": "NO VALIDATED TCA FOUND — ANALYSIS INDETERMINATE",
            "units": "s",
            "explanation": reason,
            "beginnerExplanation": (
                "We could not find a moment when these two objects are closest, so there "
                "is nothing to assess. This does not mean they are safe — it means we did "
                "not manage to check."
            ),
        },
        {
            "stepIndex": 3,
            "phase": "PHASE_10_UNCERTAINTY",
            "title": "No Conjunction — Pipeline Halted",
            "status": "halted",
            "equation": "",
            "result": (
                "Covariance propagation, B-plane projection, collision probability and "
                "risk classification were NOT performed."
            ),
            "explanation": (
                "Every downstream Phase 10 product is defined at the time of closest "
                "approach. Without a validated TCA there is no encounter plane to project "
                "into and no collision region to integrate over, so no probability and no "
                "risk level are reported."
            ),
            "beginnerExplanation": (
                "Because there was no closest-approach moment to analyse, we stopped here "
                "rather than producing numbers that would look meaningful but would not be."
            ),
        },
    ]


def _indeterminate_metadata(reason: str) -> dict[str, Any]:
    """Model metadata for an indeterminate analysis."""
    return {
        "analysis_status": STATUS_INDETERMINATE,
        "reason": reason,
        "computed": [],
        "not_computed": [
            "covariance propagation to TCA",
            "relative covariance",
            "B-plane projection",
            "uncertainty ellipse",
            "probability of collision",
            "risk classification",
        ],
        "scientific_honesty_note": (
            "An indeterminate analysis is not a low-risk finding. No collision "
            "probability was computed, so no statement about collision risk is being "
            "made either way."
        ),
    }


def run_uncertainty_conjunction_analysis(
    pos_fn_a: Callable[[float], np.ndarray],
    vel_fn_a: Callable[[float], np.ndarray],
    pos_fn_b: Callable[[float], np.ndarray],
    vel_fn_b: Callable[[float], np.ndarray],
    initial_cov_a: StateCovariance,
    initial_cov_b: StateCovariance,
    t_start: float,
    t_end: float,
    acc_fn_a: Optional[Callable[[float, np.ndarray, np.ndarray], np.ndarray]] = None,
    acc_fn_b: Optional[Callable[[float, np.ndarray, np.ndarray], np.ndarray]] = None,
    mu: float = 3.986004418e14,
    j2: float = 1.08262668e-3,
    radius: float = 6378137.0,
    hbr_m: Optional[float] = None,
    obj_a_geom: Optional[CollisionGeometry] = None,
    obj_b_geom: Optional[CollisionGeometry] = None,
    risk_thresholds: Optional[RiskThresholds] = None,
    screening_threshold_m: float = 100_000.0,
    coarse_dt: float = 60.0,
) -> UncertaintyConjunctionResult:
    """
    Run complete end-to-end Phase 10 uncertainty conjunction analysis.

    Orchestrates:
    1. Phase 9 Conjunction Screening & TCA finding
    2. Covariance propagation for both objects to TCA
    3. Relative covariance combination
    4. B-plane projection and uncertainty ellipse calculation
    5. Hard-body radius evaluation
    6. Probability of collision (Pc) integration
    7. Risk classification
    8. 14-step structured calculation trace generation
    """
    # 1. Deterministic Conjunction Analysis (Phase 9)
    conj_analysis = ConjunctionAnalysis(
        screening_threshold_m=screening_threshold_m,
        coarse_dt=coarse_dt,
    )
    conj_res = conj_analysis.analyse(pos_fn_a, vel_fn_a, pos_fn_b, vel_fn_b, t_start, t_end)

    # A valid TCA is the precondition for every Phase 10 product.  Without
    # one there is nothing to project into an encounter plane and nothing to
    # integrate a probability over, so the analysis terminates as explicitly
    # indeterminate rather than substituting an arbitrary evaluation point.
    valid_events = [e for e in conj_res.events if e.tca_result.validated]
    if not valid_events:
        reason = (
            f"No validated time of closest approach was found in the analysis window "
            f"[{t_start:.1f}, {t_end:.1f}] s using a screening threshold of "
            f"{screening_threshold_m / 1e3:.1f} km and a coarse step of {coarse_dt:.1f} s. "
            f"A conjunction may still exist: coarse screening can step over an encounter "
            f"when the relative velocity is high (detection requires roughly "
            f"coarse_dt < 2 * threshold / |v_rel|). No collision probability or risk "
            f"level has been computed."
        )
        return UncertaintyConjunctionResult(
            conjunction_result=conj_res,
            initial_cov_a=initial_cov_a,
            initial_cov_b=initial_cov_b,
            conjunction_found=False,
            analysis_status=STATUS_INDETERMINATE,
            indeterminate_reason=reason,
            risk_assessment=indeterminate_risk(reason, risk_thresholds or PROFILE_STANDARD),
            calculation_steps=build_indeterminate_calculation_trace(
                initial_cov_a=initial_cov_a,
                initial_cov_b=initial_cov_b,
                t_start=t_start,
                t_end=t_end,
                screening_threshold_m=screening_threshold_m,
                coarse_dt=coarse_dt,
                candidate_interval_count=len(conj_res.candidate_intervals),
                reason=reason,
            ),
            model_metadata=_indeterminate_metadata(reason),
        )

    # Use the closest of the validated conjunction events.
    event = min(valid_events, key=lambda e: e.tca_result.miss_distance)
    tca_s = event.tca_result.tca
    r_a = event.tca_result.r_a
    v_a = event.tca_result.v_a
    r_b = event.tca_result.r_b
    v_b = event.tca_result.v_b

    r_rel = r_a - r_b
    v_rel = v_a - v_b

    # Reported geometry comes from the TCA state itself, never from a
    # downstream projection.  |r_rel| is the physical separation; |b0| (the
    # B-plane projection computed later) coincides with it only at a true TCA.
    miss_distance_m = float(np.linalg.norm(r_rel))
    relative_velocity_m_s = float(np.linalg.norm(v_rel))

    # Default two-body + J2 acceleration if not supplied
    def default_acc(t: float, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        r_mag = float(np.linalg.norm(r))
        if r_mag < 1.0:
            return np.zeros(3)
        a_grav = -mu / (r_mag ** 3) * r
        # J2
        k = 1.5 * mu * j2 * (radius ** 2) / (r_mag ** 5)
        z2_over_r2 = (r[2] ** 2) / (r_mag ** 2)
        ax = k * r[0] * (5.0 * z2_over_r2 - 1.0)
        ay = k * r[1] * (5.0 * z2_over_r2 - 1.0)
        az = k * r[2] * (5.0 * z2_over_r2 - 3.0)
        return a_grav + np.array([ax, ay, az])

    afn_a = acc_fn_a or default_acc
    afn_b = acc_fn_b or default_acc

    # 2. Propagate Covariances to TCA
    r0_a = np.asarray(pos_fn_a(t_start), dtype=np.float64)
    v0_a = np.asarray(vel_fn_a(t_start), dtype=np.float64)
    stm_a_res = propagate_stm(
        afn_a, r0_a, v0_a, (t_start, tca_s), mu=mu, j2=j2, radius=radius
    )
    cov_a_tca = propagate_covariance(initial_cov_a, stm_a_res.stm, tca_s)

    r0_b = np.asarray(pos_fn_b(t_start), dtype=np.float64)
    v0_b = np.asarray(vel_fn_b(t_start), dtype=np.float64)
    stm_b_res = propagate_stm(
        afn_b, r0_b, v0_b, (t_start, tca_s), mu=mu, j2=j2, radius=radius
    )
    cov_b_tca = propagate_covariance(initial_cov_b, stm_b_res.stm, tca_s)

    # 3. Relative Covariance
    rel_cov = compute_relative_covariance(cov_a_tca, cov_b_tca)

    # 4. B-Plane Uncertainty
    b_plane_unc = project_covariance_to_b_plane(
        rel_pos_cov=rel_cov.position_covariance,
        r_rel=r_rel,
        v_rel=v_rel,
    )

    # 5. Hard-Body Radius
    hbr_res = compute_hard_body_radius(
        obj_a=obj_a_geom,
        obj_b=obj_b_geom,
        custom_hbr_m=hbr_m,
    )

    # 6. Probability of Collision
    pc_res = compute_collision_probability(b_plane_unc, hbr_res.combined_hbr_m)

    # 7. Risk Assessment
    risk_assessment = classify_risk(pc_res.probability, risk_thresholds or PROFILE_STANDARD)

    # 8. Structured Calculation Trace
    calc_steps = build_phase10_calculation_trace(
        initial_cov_a=initial_cov_a,
        initial_cov_b=initial_cov_b,
        cov_a_tca=cov_a_tca,
        cov_b_tca=cov_b_tca,
        rel_cov=rel_cov,
        tca_s=tca_s,
        r_rel_tca=r_rel,
        v_rel_tca=v_rel,
        b_plane_unc=b_plane_unc,
        hbr_res=hbr_res,
        pc_res=pc_res,
        risk=risk_assessment,
    )

    # 9. Model Metadata & Scientific Honesty
    model_meta = {
        "mathematical_model": {
            "covariance_propagation": "P(t) = Φ(t, t₀) P₀ Φ(t, t₀)ᵀ",
            "stm_variational_equation": "dΦ/dt = A(t) Φ, Φ(t₀, t₀) = I₆",
            "b_plane_projection": "P_B = M P_rel,pos Mᵀ",
            "collision_probability": "Pc = ∬_D f(x, y) dx dy (Alfriend/Akella/Chan 2D Gaussian Model)",
        },
        "engineering_interpretation": (
            "The covariance models quantify the spatial dispersion of orbital states at closest approach. "
            "The 2D encounter plane projection decouples the fast along-track motion from the transverse miss "
            "geometry, enabling rigorous integration of collision risk over the combined hard-body cross-section."
        ),
        "beginner_explanation": (
            "Instead of assuming exact orbits, THESEUS simulates uncertainty clouds around both spacecraft. "
            "When the objects fly past each other, we calculate the exact mathematical overlap of these clouds "
            "with the spacecraft sizes to find the true odds of an accidental collision."
        ),
        "assumptions": [
            "Linearized covariance mapping via State Transition Matrix across analysis interval",
            "Short-duration encounter: rectilinear relative trajectory during close approach",
            "Gaussian probability distribution in position and velocity errors",
            "Statistically independent objects (no shared tracking cross-correlations)",
            "Spherical bounding hard-body collision cross-section",
        ],
        "limitations": [
            "Non-Gaussian error distributions (e.g. initial launch insertion) not modeled",
            "Nonlinearities during long propagation intervals (> a few days) may require higher-order tensor methods or EnKF",
            "Atmospheric density fluctuations and solar flare events require empirical process noise models",
        ],
        "scientific_honesty_note": (
            "Covariance matrices represent estimated state uncertainties from tracking or simulation models, "
            "not numerical integration tolerances. The calculated Pc is a conditional probability under the stated "
            "Gaussian and rectilinear encounter assumptions."
        ),
    }

    return UncertaintyConjunctionResult(
        conjunction_result=conj_res,
        initial_cov_a=initial_cov_a,
        initial_cov_b=initial_cov_b,
        conjunction_found=True,
        analysis_status=STATUS_COMPLETE,
        indeterminate_reason="",
        tca_s=float(tca_s),
        miss_distance_m=miss_distance_m,
        relative_velocity_m_s=relative_velocity_m_s,
        cov_a_tca=cov_a_tca,
        cov_b_tca=cov_b_tca,
        relative_covariance=rel_cov,
        b_plane_uncertainty=b_plane_unc,
        hard_body_result=hbr_res,
        collision_probability=pc_res,
        risk_assessment=risk_assessment,
        calculation_steps=calc_steps,
        model_metadata=model_meta,
    )
