"""
Reentry vehicle model.

Describes the aerodynamic, geometric, and mass properties of a vehicle
undergoing atmospheric entry.

    β = m / (C_D A)       ballistic coefficient
    L/D = C_L / C_D       lift-to-drag ratio

Limitations
-----------
- C_D and C_L are treated as constants (no Mach/Reynolds dependence).
- No attitude dynamics; reference area is fixed.
- No ablation or mass loss during entry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class ReentryVehicle:
    """
    Immutable reentry-vehicle descriptor.

    All quantities in SI units (kg, m, m²).

    Attributes
    ----------
    name : str
        Vehicle identifier.
    mass : float
        Total entry mass (kg).  Must be > 0.
    reference_area : float
        Aerodynamic reference area (m²).  Must be > 0.
    nose_radius : float
        Effective nose radius for heating calculations (m).  Must be > 0.
    cd : float
        Drag coefficient (dimensionless).  Must be >= 0.
    cl : float
        Lift coefficient (dimensionless).  0 for ballistic entry.
    vehicle_type : str
        Descriptor ('capsule', 'winged', 'ballistic', etc.).
    atmospheric_body : str
        Name of the body whose atmosphere is being entered.
    source : str
        Data provenance / reference.
    extra : dict[str, Any]
        Additional configuration metadata.
    """
    name: str
    mass: float                              # kg
    reference_area: float                    # m²
    nose_radius: float                       # m
    cd: float                                # dimensionless
    cl: float = 0.0                          # dimensionless  (0 = ballistic)
    vehicle_type: str = "capsule"
    atmospheric_body: str = "Earth"
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate physical constraints."""
        if self.mass <= 0:
            raise ValueError(f"mass must be > 0, got {self.mass}")
        if self.reference_area <= 0:
            raise ValueError(f"reference_area must be > 0, got {self.reference_area}")
        if self.nose_radius <= 0:
            raise ValueError(f"nose_radius must be > 0, got {self.nose_radius}")
        if self.cd < 0:
            raise ValueError(f"cd must be >= 0, got {self.cd}")
        if abs(self.cl) > 5.0:
            raise ValueError(
                f"cl = {self.cl} is outside physically plausible range [-5, 5]"
            )

    # -- derived properties ---------------------------------------------------

    @property
    def ballistic_coefficient(self) -> float:
        """
        Ballistic coefficient β = m / (C_D A)  [kg/m²].

        A higher β means the vehicle decelerates more slowly in the
        atmosphere — it "penetrates" deeper before slowing down.

        Returns infinity if C_D is zero (drag-free body).
        """
        if self.cd <= 0:
            return math.inf
        return self.mass / (self.cd * self.reference_area)

    @property
    def lift_to_drag_ratio(self) -> float:
        """
        Aerodynamic lift-to-drag ratio L/D = C_L / C_D.

        Returns 0.0 if C_D is zero (undefined, not physical).
        """
        if self.cd <= 0:
            return 0.0
        return self.cl / self.cd

    @property
    def is_lifting(self) -> bool:
        """True if the vehicle has non-zero lift coefficient."""
        return abs(self.cl) > 1e-15

    @property
    def entry_type(self) -> str:
        """'ballistic' or 'lifting'."""
        return "lifting" if self.is_lifting else "ballistic"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "name": self.name,
            "mass_kg": self.mass,
            "reference_area_m2": self.reference_area,
            "nose_radius_m": self.nose_radius,
            "cd": self.cd,
            "cl": self.cl,
            "ballistic_coefficient_kg_m2": self.ballistic_coefficient,
            "lift_to_drag_ratio": self.lift_to_drag_ratio,
            "entry_type": self.entry_type,
            "vehicle_type": self.vehicle_type,
            "atmospheric_body": self.atmospheric_body,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Pre-configured reference vehicles
# ---------------------------------------------------------------------------

APOLLO_CM = ReentryVehicle(
    name="Apollo Command Module",
    mass=5424.0,            # kg (entry mass)
    reference_area=12.017,  # m²  (heat-shield diameter 3.912 m)
    nose_radius=4.694,      # m   (effective blunt-body radius)
    cd=1.29,                # hypersonic continuum Cd (Hillje 1969)
    cl=0.368,               # trim L/D ≈ 0.285 at α ≈ -27° (Moseley et al.)
    vehicle_type="capsule",
    atmospheric_body="Earth",
    source="NASA TN D-6792 (Hillje 1969); Moseley et al. 1969; JSC-09133",
)

SOYUZ_SA = ReentryVehicle(
    name="Soyuz SA (descent module)",
    mass=2900.0,
    reference_area=3.80,    # m²  (heat shield ~2.2 m diameter)
    nose_radius=2.235,      # m
    cd=1.3,
    cl=0.18,                # low L/D capsule
    vehicle_type="capsule",
    atmospheric_body="Earth",
    source="RSC Energia published data; Soyuz Users Manual",
)

GENERIC_BALLISTIC = ReentryVehicle(
    name="Generic Ballistic Entry Body",
    mass=1000.0,
    reference_area=1.0,
    nose_radius=0.5,
    cd=2.0,
    cl=0.0,
    vehicle_type="ballistic",
    atmospheric_body="Earth",
    source="Textbook example (no specific vehicle)",
)
