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

# CollisionGeometry describes an object's *deterministic* physical dimensions,
# which is a Phase 9 concept, so it now lives with the rest of the conjunction
# geometry.  It is re-exported here so that Phase 10 code and any existing
# `from theseus.uncertainty.hard_body import CollisionGeometry` keep working.
# Phase 10 layers the probabilistic hard-body *disk radius* on top of it.
from theseus.conjunction.geometry import CollisionGeometry  # noqa: F401


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
        Directly specified **combined** HBR, i.e. R_A + R_B for the pair, not
        a per-object radius.  Any non-negative value is accepted, including
        values below one metre and zero.
    radius_a_m : float, optional
        Collision radius for object A.
    radius_b_m : float, optional
        Collision radius for object B.

    Returns
    -------
    HardBodyResult
        ``combined_hbr_m`` is the quantity the collision-probability
        integration uses.  On the ``custom_hbr_m`` path it equals the supplied
        value exactly; the two per-object geometries carry an equal split of
        it, marked as derived in their ``source``.

    Raises
    ------
    ValueError
        If ``custom_hbr_m`` is negative, or if a supplied geometry is itself
        internally inconsistent.

    Notes
    -----
    **Precedence, pre-existing and unchanged:** when ``custom_hbr_m`` is not
    None it takes priority and ``obj_a``/``obj_b`` are ignored entirely.  A
    caller that supplies both real geometries *and* a combined HBR gets the
    combined HBR.  This is stated here rather than left implicit; whether the
    precedence is the right one is a separate question from this function's
    correctness and has not been changed.
    """
    if custom_hbr_m is not None:
        if custom_hbr_m < 0.0:
            raise ValueError(f"Custom HBR must be non-negative, got {custom_hbr_m}")

        # A caller-supplied combined HBR carries no per-object information: it
        # is one number for the pair.  The two geometries below exist only so
        # the result can report a consistent pair; the equal split is a
        # reporting convention and nothing downstream uses it -- the collision
        # probability integrates over a disk of radius `combined_hbr_m`, which
        # is the supplied value exactly, whatever the split.
        #
        # `physical_radius_m` must therefore be derived from the supplied
        # number too, not left at the dataclass default.  Leaving it at 1.0 m
        # was the defect: for any combined HBR below 2 m the derived collision
        # radius falls under that fixed default and the geometry's own
        # consistency check -- collision radius may not be smaller than
        # physical radius -- rejects a perfectly valid request.  Every CubeSat,
        # small-debris and fragment conjunction was unanalysable because of a
        # placeholder, and for HBR >= 2 m the reported 1.0 m was a fabricated
        # structural dimension that no caller had supplied.
        #
        # A supplied combined HBR is by definition an enclosing radius: it
        # already includes appendages, so there is no separate structural
        # figure to distinguish.  The structural radius is therefore set equal
        # to the collision radius rather than invented below it, and `source`
        # records that both are derived, not measured.
        half_hbr = custom_hbr_m / 2.0
        provenance = (
            f"Derived from a caller-supplied combined hard-body radius of "
            f"{custom_hbr_m} m, split equally between the two objects. "
            f"Neither object's individual dimensions are known; only the sum "
            f"is physically meaningful."
        )
        geom_a = CollisionGeometry(
            name="Object A",
            collision_radius_m=half_hbr,
            physical_radius_m=half_hbr,
            source=provenance,
        )
        geom_b = CollisionGeometry(
            name="Object B",
            collision_radius_m=half_hbr,
            physical_radius_m=half_hbr,
            source=provenance,
        )
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
