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

from theseus.conjunction.screening import ConjunctionScreener
from theseus.conjunction.tca import find_tca, find_all_tca
from theseus.conjunction.b_plane import compute_b_plane, BPlaneResult
from theseus.conjunction.analysis import (
    ConjunctionEvent,
    ConjunctionAnalysis,
    ConjunctionResult,
)

__all__ = [
    "ConjunctionScreener",
    "find_tca",
    "find_all_tca",
    "compute_b_plane",
    "BPlaneResult",
    "ConjunctionEvent",
    "ConjunctionAnalysis",
    "ConjunctionResult",
]
