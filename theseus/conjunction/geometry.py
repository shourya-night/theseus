"""
Deterministic collision geometry for conjunction analysis.

This module answers one question, and only that question:

    Given the nominal trajectories, did the two bodies physically intersect?

That is a geometric question about the trajectories the propagator produced.
It is **not** the question Phase 10 asks, which is:

    Given that the trajectories are uncertain, what is the probability of
    collision?

The two must not be conflated.  A close approach is not a collision; a
conjunction is not a collision; a non-zero collision probability is not a
collision.  Only ``clearance <= 0`` at the time of closest approach means the
nominal bodies touched.

Definitions
-----------
For two objects with hard-body radii R_A and R_B, approaching to a miss
distance d at the time of closest approach:

    combined_hard_body_radius  =  R_A + R_B
    clearance                  =  d − (R_A + R_B)

so that

    clearance  <  0   →  the bodies interpenetrate      (INTERSECTION)
    clearance  =  0   →  surfaces touch exactly         (GRAZING)
    clearance  >  0   →  the bodies pass clear          (NO_INTERSECTION)

``miss_distance`` is the separation of the two *centres*.  It is never itself
a collision criterion: a 9 m miss between two CubeSats is a clean pass, while
a 60 m miss between two objects with 40 m arrays is a strike.  Keeping the
three quantities distinct is the whole point of this module.

Grazing is a measure-zero condition in floating point, so it is reported only
when the miss distance equals the combined radius exactly, or within an
explicitly supplied ``grazing_tolerance_m``.  There is no hidden default
tolerance: contact is decided by the numbers, not by a fudge factor.

Hard-body radius
----------------
The hard-body radius is the radius of a sphere enclosing the object including
anything that would be destroyed on contact -- solar arrays, booms, antennas --
not the radius of its structural bus.  For the ISS the bus-scale radius is
about 15 m while the enclosing sphere is about 54 m; using the former would
under-report strikes by a factor of three in linear dimension.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class CollisionStatus(str, Enum):
    """
    Deterministic outcome of the collision-geometry test.

    ``UNKNOWN`` is returned when no body geometry was supplied.  It means the
    question was not evaluated -- it does **not** mean the bodies passed
    clear, and must never be rendered as a safe result.
    """
    NO_INTERSECTION = "NO_INTERSECTION"
    GRAZING = "GRAZING"
    INTERSECTION = "INTERSECTION"
    UNKNOWN = "UNKNOWN"


@dataclass
class CollisionGeometry:
    """
    Collision geometry model for a single space object.

    Attributes
    ----------
    name : str
        Object name or identifier.
    physical_radius_m : float
        Nominal structural body radius (m).
    collision_radius_m : float
        Effective hard-body collision radius (m), enclosing appendages.
        Must be at least ``physical_radius_m``.
    object_type : str
        'payload', 'rocket_body', 'debris', 'generic'.
    shape : str
        'spherical', 'box_wing', 'cylinder'.
    source : str
        Provenance of the dimensions, so a reader can tell a published value
        from an assumed one.
    """
    name: str = "Generic Object"
    physical_radius_m: float = 1.0
    collision_radius_m: float = 1.0
    object_type: str = "generic"
    shape: str = "spherical"
    source: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.physical_radius_m) or not math.isfinite(self.collision_radius_m):
            raise ValueError("collision geometry radii must be finite")
        if self.physical_radius_m < 0.0:
            raise ValueError(
                f"Physical radius must be non-negative, got {self.physical_radius_m}"
            )
        if self.collision_radius_m < 0.0:
            raise ValueError(
                f"Collision radius must be non-negative, got {self.collision_radius_m}"
            )
        if self.collision_radius_m < self.physical_radius_m:
            raise ValueError(
                f"Collision radius ({self.collision_radius_m} m) cannot be smaller "
                f"than physical radius ({self.physical_radius_m} m)"
            )

    @classmethod
    def from_radius(cls, radius_m: float, name: str = "Object",
                    object_type: str = "generic", source: str = "") -> CollisionGeometry:
        """
        Build a spherical geometry from a single hard-body radius.

        Use this when an object model carries one radius and draws no
        distinction between structure and enclosing sphere.
        """
        return cls(
            name=name,
            physical_radius_m=float(radius_m),
            collision_radius_m=float(radius_m),
            object_type=object_type,
            shape="spherical",
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "physical_radius_m": float(self.physical_radius_m),
            "collision_radius_m": float(self.collision_radius_m),
            "object_type": self.object_type,
            "shape": self.shape,
            "source": self.source,
        }


@dataclass(frozen=True)
class CollisionAssessment:
    """
    Deterministic collision-geometry result at the time of closest approach.

    Attributes
    ----------
    status : CollisionStatus
        INTERSECTION, GRAZING, NO_INTERSECTION, or UNKNOWN.
    miss_distance_m : float
        Centre-to-centre separation at TCA (m).
    combined_hard_body_radius_m : float
        R_A + R_B (m).  None of the objects' geometry is re-derived here; this
        is the sum of the supplied collision radii.
    clearance_m : float
        ``miss_distance_m - combined_hard_body_radius_m`` (m).  Negative means
        the bodies interpenetrate.
    grazing_tolerance_m : float
        Half-width of the band around zero clearance reported as GRAZING.
    object_a, object_b : CollisionGeometry | None
        The geometry used, echoed back for traceability.
    """
    status: CollisionStatus
    miss_distance_m: float
    combined_hard_body_radius_m: float
    clearance_m: float
    grazing_tolerance_m: float = 0.0
    object_a: Optional[CollisionGeometry] = None
    object_b: Optional[CollisionGeometry] = None

    @property
    def is_physical_intersection(self) -> bool:
        """
        True when the nominal bodies made contact.

        Grazing counts as contact: surfaces touching is a strike, not a pass.
        UNKNOWN is False, so always check :attr:`is_evaluated` before treating
        a False as a safe result.
        """
        return self.status in (CollisionStatus.INTERSECTION, CollisionStatus.GRAZING)

    @property
    def is_evaluated(self) -> bool:
        """False when no geometry was supplied and the test was not performed."""
        return self.status is not CollisionStatus.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "evaluated": self.is_evaluated,
            "is_physical_intersection": self.is_physical_intersection,
            "miss_distance_m": float(self.miss_distance_m),
            "miss_distance_km": float(self.miss_distance_m) / 1e3,
            "combined_hard_body_radius_m": float(self.combined_hard_body_radius_m),
            "clearance_m": float(self.clearance_m),
            "clearance_km": float(self.clearance_m) / 1e3,
            "grazing_tolerance_m": float(self.grazing_tolerance_m),
            "object_a": self.object_a.to_dict() if self.object_a is not None else None,
            "object_b": self.object_b.to_dict() if self.object_b is not None else None,
        }


def combined_hard_body_radius(geom_a: CollisionGeometry,
                              geom_b: CollisionGeometry) -> float:
    """Combined hard-body radius R_A + R_B (m)."""
    return float(geom_a.collision_radius_m) + float(geom_b.collision_radius_m)


def assess_collision_geometry(
    miss_distance_m: float,
    geom_a: Optional[CollisionGeometry] = None,
    geom_b: Optional[CollisionGeometry] = None,
    *,
    grazing_tolerance_m: float = 0.0,
) -> CollisionAssessment:
    """
    Classify a close approach as intersection, grazing contact, or clear pass.

    Parameters
    ----------
    miss_distance_m : float
        Centre-to-centre separation at the validated time of closest approach.
        This must come from the TCA solution, not from a coarse screening
        sample or an arbitrary display time.
    geom_a, geom_b : CollisionGeometry, optional
        Body geometry.  When either is missing the result is UNKNOWN: the
        question is reported as unevaluated rather than answered "no".
    grazing_tolerance_m : float
        Clearances within +/- this value of zero are reported as GRAZING.
        Defaults to 0.0, so grazing is only reported on exact contact.

    Returns
    -------
    CollisionAssessment
    """
    d = float(miss_distance_m)
    if not math.isfinite(d) or d < 0.0:
        raise ValueError(f"miss_distance_m must be finite and non-negative, got {miss_distance_m}")
    if grazing_tolerance_m < 0.0:
        raise ValueError(
            f"grazing_tolerance_m must be non-negative, got {grazing_tolerance_m}"
        )

    if geom_a is None or geom_b is None:
        return CollisionAssessment(
            status=CollisionStatus.UNKNOWN,
            miss_distance_m=d,
            combined_hard_body_radius_m=float("nan"),
            clearance_m=float("nan"),
            grazing_tolerance_m=float(grazing_tolerance_m),
            object_a=geom_a,
            object_b=geom_b,
        )

    hbr = combined_hard_body_radius(geom_a, geom_b)
    clearance = d - hbr

    tol = float(grazing_tolerance_m)
    if abs(clearance) <= tol:
        status = CollisionStatus.GRAZING
    elif clearance < 0.0:
        status = CollisionStatus.INTERSECTION
    else:
        status = CollisionStatus.NO_INTERSECTION

    return CollisionAssessment(
        status=status,
        miss_distance_m=d,
        combined_hard_body_radius_m=hbr,
        clearance_m=clearance,
        grazing_tolerance_m=tol,
        object_a=geom_a,
        object_b=geom_b,
    )
