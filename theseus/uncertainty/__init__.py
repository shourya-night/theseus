"""
THESEUS Phase 10: Uncertainty Propagation, Covariance Analysis & Probability of Collision.

Provides:
- StateCovariance: Rigorous 6×6 Cartesian state covariance representation and validation
- State Transition Matrix (STM) propagation with analytic and numerical Jacobians
- Covariance propagation synchronized with physical force models
- Relative covariance computation with statistical independence tracking
- B-plane uncertainty projection and uncertainty ellipse analysis
- Hard-body radius modeling with object presets
- Probability of Collision (Pc) 2D Gaussian integral computation
- Deterministic risk classification with configurable thresholds
- Full 14-step progressive calculation traces and scientific model metadata
"""

from theseus.uncertainty.covariance import (
    StateCovariance,
    CovarianceValidationError,
)
from theseus.uncertainty.state_transition import (
    STMResult,
    propagate_stm,
    gravity_jacobian,
    j2_jacobian,
    numerical_jacobian,
    build_dynamics_jacobian,
)
from theseus.uncertainty.propagation import (
    ProcessNoiseModel,
    CovariancePropagationResult,
    CovariancePropagator,
    propagate_covariance,
)
from theseus.uncertainty.relative import (
    RelativeCovarianceResult,
    compute_relative_covariance,
)
from theseus.uncertainty.b_plane import (
    BPlaneUncertainty,
    project_covariance_to_b_plane,
)
from theseus.uncertainty.hard_body import (
    CollisionGeometry,
    HardBodyResult,
    compute_hard_body_radius,
    PRESET_ISS,
    PRESET_LARGE_SAT,
    PRESET_MEDIUM_SAT,
    PRESET_CUBESAT,
    PRESET_ROCKET_BODY,
    PRESET_DEBRIS_SMALL,
)
from theseus.uncertainty.collision_probability import (
    CollisionProbabilityResult,
    MonteCarloValidationResult,
    compute_collision_probability,
    monte_carlo_pc_validation,
)
from theseus.uncertainty.risk import (
    RiskLevel,
    RiskThresholds,
    RiskAssessment,
    classify_risk,
    PROFILE_CONSERVATIVE,
    PROFILE_STANDARD,
    PROFILE_PERMISSIVE,
)
from theseus.uncertainty.results import (
    UncertaintyConjunctionResult,
    run_uncertainty_conjunction_analysis,
    build_phase10_calculation_trace,
)

__all__ = [
    # Covariance
    "StateCovariance",
    "CovarianceValidationError",
    # STM
    "STMResult",
    "propagate_stm",
    "gravity_jacobian",
    "j2_jacobian",
    "numerical_jacobian",
    "build_dynamics_jacobian",
    # Propagation
    "ProcessNoiseModel",
    "CovariancePropagationResult",
    "CovariancePropagator",
    "propagate_covariance",
    # Relative
    "RelativeCovarianceResult",
    "compute_relative_covariance",
    # B-Plane
    "BPlaneUncertainty",
    "project_covariance_to_b_plane",
    # Hard Body
    "CollisionGeometry",
    "HardBodyResult",
    "compute_hard_body_radius",
    "PRESET_ISS",
    "PRESET_LARGE_SAT",
    "PRESET_MEDIUM_SAT",
    "PRESET_CUBESAT",
    "PRESET_ROCKET_BODY",
    "PRESET_DEBRIS_SMALL",
    # Collision Probability
    "CollisionProbabilityResult",
    "MonteCarloValidationResult",
    "compute_collision_probability",
    "monte_carlo_pc_validation",
    # Risk
    "RiskLevel",
    "RiskThresholds",
    "RiskAssessment",
    "classify_risk",
    "PROFILE_CONSERVATIVE",
    "PROFILE_STANDARD",
    "PROFILE_PERMISSIVE",
    # Results & Trace
    "UncertaintyConjunctionResult",
    "run_uncertainty_conjunction_analysis",
    "build_phase10_calculation_trace",
]
