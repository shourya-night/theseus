"""
Classical orbital elements and derived quantities.

All angular quantities in radians.  All lengths in metres.
All times in seconds.

Definitions
-----------
a  : semi-major axis (m).  Negative for hyperbolic orbits.
e  : eccentricity (dimensionless).  0 = circular, 0 < e < 1 elliptic,
     e = 1 parabolic, e > 1 hyperbolic.
i  : inclination (rad), [0, π].
raan : right ascension of ascending node Ω (rad), [0, 2π).
argp : argument of periapsis ω (rad), [0, 2π).
nu   : true anomaly ν (rad), [0, 2π).

Derived
-------
p      : semi-latus rectum  p = a(1 − e²)
r_peri : periapsis radius   r_p = a(1 − e)
r_apo  : apoapsis radius    r_a = a(1 + e)   (elliptic only)
T      : orbital period     T = 2π √(a³/μ)   (elliptic only)
E      : specific orbital energy  ε = −μ/(2a)
h      : specific angular momentum magnitude  h = √(μp)

References
----------
Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed.
Curtis, "Orbital Mechanics for Engineering Students", 4th ed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class OrbitalElements:
    """
    Classical (Keplerian) orbital elements.

    Parameters
    ----------
    a    : semi-major axis (m).
    e    : eccentricity.
    i    : inclination (rad).
    raan : right ascension of ascending node (rad).
    argp : argument of periapsis (rad).
    nu   : true anomaly (rad).
    mu   : gravitational parameter of central body (m³/s²).
    """
    a: float
    e: float
    i: float
    raan: float
    argp: float
    nu: float
    mu: float

    # -- Geometry --------------------------------------------------------

    @property
    def semi_latus_rectum(self) -> float:
        """p = a(1 − e²)  (m)."""
        return self.a * (1.0 - self.e ** 2)

    @property
    def periapsis_radius(self) -> float:
        """r_p = a(1 − e)  (m)."""
        return self.a * (1.0 - self.e)

    @property
    def apoapsis_radius(self) -> Optional[float]:
        """r_a = a(1 + e)  (m).  None for parabolic/hyperbolic."""
        if self.e >= 1.0:
            return None
        return self.a * (1.0 + self.e)

    @property
    def radius(self) -> float:
        """Orbital radius at current true anomaly (m)."""
        p = self.semi_latus_rectum
        return p / (1.0 + self.e * math.cos(self.nu))

    # -- Energy & momentum -----------------------------------------------

    @property
    def specific_energy(self) -> float:
        """ε = −μ / (2a)  (m²/s²)."""
        return -self.mu / (2.0 * self.a)

    @property
    def specific_angular_momentum(self) -> float:
        """h = √(μ p)  (m²/s)."""
        return math.sqrt(self.mu * self.semi_latus_rectum)

    # -- Period ----------------------------------------------------------

    @property
    def period(self) -> Optional[float]:
        """T = 2π √(a³/μ)  (s).  None for non-elliptic orbits."""
        if self.e >= 1.0 or self.a <= 0:
            return None
        return 2.0 * math.pi * math.sqrt(self.a ** 3 / self.mu)

    @property
    def mean_motion(self) -> Optional[float]:
        """n = √(μ/a³)  (rad/s).  None for non-elliptic orbits."""
        if self.a <= 0:
            return None
        return math.sqrt(self.mu / self.a ** 3)

    # -- Anomalies -------------------------------------------------------

    @property
    def eccentric_anomaly(self) -> float:
        """
        Eccentric anomaly E (rad) from true anomaly ν.

        Elliptic:    tan(E/2) = √((1−e)/(1+e)) tan(ν/2)
        Hyperbolic:  tanh(H/2) = √((e−1)/(e+1)) tan(ν/2)
        """
        if self.e < 1.0:
            # Elliptic
            E = 2.0 * math.atan2(
                math.sqrt(1.0 - self.e) * math.sin(self.nu / 2.0),
                math.sqrt(1.0 + self.e) * math.cos(self.nu / 2.0),
            )
            return E % (2.0 * math.pi)
        else:
            # Hyperbolic anomaly H
            tan_half_nu = math.tan(self.nu / 2.0)
            tanh_half_H = math.sqrt((self.e - 1.0) / (self.e + 1.0)) * tan_half_nu
            # Clamp for numerical safety
            tanh_half_H = max(-0.999999999, min(0.999999999, tanh_half_H))
            return 2.0 * math.atanh(tanh_half_H)

    @property
    def mean_anomaly(self) -> float:
        """
        Mean anomaly M (rad).

        Elliptic:    M = E − e sin(E)
        Hyperbolic:  M = e sinh(H) − H
        """
        if self.e < 1.0:
            E = self.eccentric_anomaly
            return (E - self.e * math.sin(E)) % (2.0 * math.pi)
        else:
            H = self.eccentric_anomaly
            return self.e * math.sinh(H) - H

    # -- Summary ---------------------------------------------------------

    def summary(self) -> dict:
        return {
            "a_m": self.a,
            "e": self.e,
            "i_rad": self.i,
            "raan_rad": self.raan,
            "argp_rad": self.argp,
            "nu_rad": self.nu,
            "mu_m3s2": self.mu,
            "period_s": self.period,
            "r_periapsis_m": self.periapsis_radius,
            "r_apoapsis_m": self.apoapsis_radius,
            "specific_energy_m2s2": self.specific_energy,
            "h_m2s": self.specific_angular_momentum,
        }
