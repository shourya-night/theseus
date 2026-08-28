"""
Conjunction (close-approach) analysis subpackage.

Phase 9 of the THESEUS astrodynamics engine.

Provides:
- Conjunction screening (coarse pass)
- TCA (Time of Closest Approach) refinement via Brent's method
- Miss-distance and relative-velocity computation
- B-plane geometry (when applicable to hyperbolic encounters)
- Encounter classification (head-on, overtaking, crossing)
- Full calculation traces with scientific transparency

All analysis operates on actual propagated/interpolated state histories.
"""

from theseus.conjunction.state_validation import (
    NonFiniteStateError,
    guard_state_function,
    validate_state_vector,
)
from theseus.conjunction.screening import ConjunctionScreener, ScreeningDiagnostics
from theseus.conjunction.tca import (
    find_tca, find_all_tca, find_all_tca_with_diagnostics, TCASearchDiagnostics,
)
from theseus.conjunction.b_plane import compute_b_plane, BPlaneResult
from theseus.conjunction.geometry import (
    CollisionGeometry,
    CollisionAssessment,
    CollisionStatus,
    assess_collision_geometry,
    combined_hard_body_radius,
)
from theseus.conjunction.analysis import (
    ConjunctionAccounting,
    ConjunctionEvent,
    ConjunctionAnalysis,
    ConjunctionResult,
)

__all__ = [
    "NonFiniteStateError",
    "guard_state_function",
    "validate_state_vector",
    "ConjunctionScreener",
    "ScreeningDiagnostics",
    "find_tca",
    "find_all_tca",
    "find_all_tca_with_diagnostics",
    "TCASearchDiagnostics",
    "compute_b_plane",
    "BPlaneResult",
    "CollisionGeometry",
    "CollisionAssessment",
    "CollisionStatus",
    "assess_collision_geometry",
    "combined_hard_body_radius",
    "ConjunctionAccounting",
    "ConjunctionEvent",
    "ConjunctionAnalysis",
    "ConjunctionResult",
]
