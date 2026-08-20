"""
Reentry dynamics subpackage.

Phase 8 of the THESEUS astrodynamics engine.

Provides atmospheric reentry simulation with:
- Configurable reentry vehicle models
- Sutton-Graves convective heating
- 2D planar entry equations of motion
- Event detection (peak-Q, peak heating, peak-G, impact, skip-out)
- Full calculation traces with scientific transparency

All models clearly distinguish numerical accuracy from physical
model fidelity and engineering approximation.
"""

from theseus.reentry.vehicle import ReentryVehicle
from theseus.reentry.heating import sutton_graves_convective, stagnation_temperature
from theseus.reentry.results import ReentryEvent, ReentryResult
from theseus.reentry.simulator import ReentrySimulator

__all__ = [
    "ReentryVehicle",
    "sutton_graves_convective",
    "stagnation_temperature",
    "ReentryEvent",
    "ReentryResult",
    "ReentrySimulator",
]
