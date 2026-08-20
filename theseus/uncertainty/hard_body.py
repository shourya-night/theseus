"""
Hard-body radius modeling for collision probability analysis.

For two close-approaching space objects:
    HBR = R₁ + R₂

where R₁ and R₂ are the collision radii (hard-body bounding radii) of
the primary and secondary objects.

Distinguishes:
- Physical radius (geometric dimension)
- Hard-body radius (sphere enclosing active appendages / solar arrays)
- Combined collision radius (HBR)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CollisionGeometry:
    """
    Collision geometry model for a single space object.

    Attributes
    ----------
    name : str
        Object name or identifier.
    physical_radius_m : float
        Nominal body radius (m).
    collision_radius_m : float
        Effective hard-body collision radius (m) enclosing appendages.
    object_type : str
        'payload', 'rocket_body', 'debris', 'generic'.
    shape : str
        'spherical', 'box_wing', 'cylinder'.
    """
    name: str = "Generic Object"
    physical_radius_m: float = 1.0
    collision_radius_m: float = 1.0
    object_type: str = "generic"
    shape: str = "spherical"

    def __post_init__(self) -> None:
        if self.physical_radius_m < 0.0:
            raise ValueError(f"Physical radius must be non-negative, got {self.physical_radius_m}")
        if self.collision_radius_m < 0.0:
            raise ValueError(f"Collision radius must be non-negative, got {self.collision_radius_m}")
        if self.collision_radius_m < self.physical_radius_m:
            raise ValueError(
                f"Collision radius ({self.collision_radius_m} m) cannot be smaller than physical radius ({self.physical_radius_m} m)"
            )


# Common reference presets
PRESET_ISS = CollisionGeometry(
    name="International Space Station",
    physical_radius_m=15.0,
    collision_radius_m=54.0,  # 108m wingspan enclosing sphere
    object_type="payload",
    shape="box_wing",
)

PRESET_LARGE_SAT = CollisionGeometry(
    name="Large GEO Satellite",
    physical_radius_m=2.5,
    collision_radius_m=12.0,  # Solar array span
    object_type="payload",
    shape="box_wing",
)

PRESET_MEDIUM_SAT = CollisionGeometry(
    name="Medium LEO Satellite",
    physical_radius_m=1.0,
    collision_radius_m=3.0,
    object_type="payload",
    shape="box_wing",
)

PRESET_CUBESAT = CollisionGeometry(
    name="3U CubeSat",
    physical_radius_m=0.17,
    collision_radius_m=0.3,
    object_type="payload",
    shape="box_wing",
)

PRESET_ROCKET_BODY = CollisionGeometry(
    name="Upper Stage Rocket Body",
    physical_radius_m=2.0,
    collision_radius_m=6.0,
    object_type="rocket_body",
    shape="cylinder",
)

PRESET_DEBRIS_SMALL = CollisionGeometry(
    name="Trackable Debris (Small)",
    physical_radius_m=0.1,
    collision_radius_m=0.1,
    object_type="debris",
    shape="spherical",
)


@dataclass
class HardBodyResult:
    """
    Combined Hard-Body Radius calculation result.

    Attributes
    ----------
    combined_hbr_m : float
        Combined hard-body collision radius HBR = R₁ + R₂ (m).
    object_a : CollisionGeometry
        Object A geometry.
    object_b : CollisionGeometry
        Object B geometry.
    assumptions : list[str]
        Modeling assumptions.
    """
    combined_hbr_m: float
    object_a: CollisionGeometry
    object_b: CollisionGeometry
    assumptions: list[str] = field(default_factory=lambda: [
        "Spherical hard-body approximation: objects treated as spheres of radius R₁ and R₂",
        "Combined collision cross-section disk of radius HBR = R₁ + R₂ in the encounter plane",
        "Orientation-averaged or conservative bounding radius",
    ])

    def to_dict(self) -> dict[str, Any]:
        return {
            "combined_hbr_m": float(self.combined_hbr_m),
            "combined_hbr_km": float(self.combined_hbr_m / 1e3),
            "object_a": {
                "name": self.object_a.name,
                "physical_radius_m": float(self.object_a.physical_radius_m),
                "collision_radius_m": float(self.object_a.collision_radius_m),
                "type": self.object_a.object_type,
            },
            "object_b": {
                "name": self.object_b.name,
                "physical_radius_m": float(self.object_b.physical_radius_m),
                "collision_radius_m": float(self.object_b.collision_radius_m),
                "type": self.object_b.object_type,
            },
            "assumptions": self.assumptions,
        }


def compute_hard_body_radius(
    obj_a: Optional[CollisionGeometry] = None,
    obj_b: Optional[CollisionGeometry] = None,
    custom_hbr_m: Optional[float] = None,
    radius_a_m: Optional[float] = None,
    radius_b_m: Optional[float] = None,
) -> HardBodyResult:
    """
    Compute the combined hard-body collision radius.

    Parameters
    ----------
    obj_a : CollisionGeometry, optional
    obj_b : CollisionGeometry, optional
    custom_hbr_m : float, optional
        Directly specified combined HBR.
    radius_a_m : float, optional
        Collision radius for object A.
    radius_b_m : float, optional
        Collision radius for object B.

    Returns
    -------
    HardBodyResult
    """
    if custom_hbr_m is not None:
        if custom_hbr_m < 0.0:
            raise ValueError(f"Custom HBR must be non-negative, got {custom_hbr_m}")
        geom_a = CollisionGeometry(name="Object A", collision_radius_m=custom_hbr_m / 2.0)
        geom_b = CollisionGeometry(name="Object B", collision_radius_m=custom_hbr_m / 2.0)
        return HardBodyResult(combined_hbr_m=custom_hbr_m, object_a=geom_a, object_b=geom_b)

    geom_a = obj_a or CollisionGeometry(
        name="Object A",
        collision_radius_m=radius_a_m if radius_a_m is not None else 5.0,
        physical_radius_m=radius_a_m if radius_a_m is not None else 2.0,
    )
    geom_b = obj_b or CollisionGeometry(
        name="Object B",
        collision_radius_m=radius_b_m if radius_b_m is not None else 5.0,
        physical_radius_m=radius_b_m if radius_b_m is not None else 2.0,
    )

    combined_hbr = geom_a.collision_radius_m + geom_b.collision_radius_m
    return HardBodyResult(combined_hbr_m=combined_hbr, object_a=geom_a, object_b=geom_b)
