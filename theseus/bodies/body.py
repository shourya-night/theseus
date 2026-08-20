"""
Celestial body data model.

All values stored in SI units (m, kg, s, rad).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class AtmosphereParams:
    """
    Basic atmospheric descriptor for a celestial body.

    Attributes
    ----------
    has_atmosphere : bool
    surface_pressure : float | None
        Pa.
    surface_density : float | None
        kg/m³.
    scale_height : float | None
        m.  Characteristic exponential scale height (approximate).
    composition : dict[str, float]
        Major species and their volume fractions.
    """
    has_atmosphere: bool = False
    surface_pressure: Optional[float] = None
    surface_density: Optional[float] = None
    scale_height: Optional[float] = None
    composition: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CelestialBody:
    """
    Immutable celestial-body descriptor.

    Attributes
    ----------
    name : str
        Common name.
    mu : float
        Standard gravitational parameter GM (m³/s²).
        Preferred over separate G and M because GM is known to much
        higher precision than either G or M individually.
    radius : float
        Mean or equatorial radius (m).
    radius_polar : float | None
        Polar radius (m), if different from equatorial.
    mass : float | None
        Mass (kg).  Derived from mu / G; lower precision than mu.
    rotation_period : float | None
        Sidereal rotation period (s).
    axial_tilt : float | None
        Obliquity to orbit (rad).
    J2 : float
        Second zonal harmonic (dimensionless).
    J3 : float
        Third zonal harmonic (dimensionless).
    atmosphere : AtmosphereParams
        Atmospheric descriptor.
    parent_name : str | None
        Name of parent body (e.g. 'Sun' for planets, 'Earth' for Moon).
    data_source : str
        Provenance string for the numerical values.
    extra : dict[str, Any]
        Additional properties.
    """
    name: str
    mu: float                                       # m³/s²
    radius: float                                   # m (equatorial)
    radius_polar: Optional[float] = None            # m
    mass: Optional[float] = None                    # kg
    rotation_period: Optional[float] = None         # s (sidereal)
    axial_tilt: Optional[float] = None              # rad
    J2: float = 0.0
    J3: float = 0.0
    atmosphere: AtmosphereParams = field(default_factory=AtmosphereParams)
    parent_name: Optional[str] = None
    data_source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    # -- derived properties --------------------------------------------------

    @property
    def surface_gravity(self) -> float:
        """Mean surface gravitational acceleration (m/s²)."""
        return self.mu / (self.radius ** 2)

    @property
    def escape_velocity(self) -> float:
        """Surface escape velocity (m/s)."""
        import math
        return math.sqrt(2.0 * self.mu / self.radius)

    @property
    def rotation_rate(self) -> Optional[float]:
        """Sidereal angular rotation rate (rad/s), or None."""
        if self.rotation_period is not None and self.rotation_period > 0:
            import math
            return 2.0 * math.pi / self.rotation_period
        return None
