"""
Gravitational acceleration models.

PointMassGravity
    Newtonian point-mass: a = −μ/r³ × r

J2Perturbation
    Oblateness perturbation (zonal harmonic J₂).
    a_J2 = (3μJ₂R²)/(2r⁵) × [x(5z²/r²−1), y(5z²/r²−1), z(5z²/r²−3)]

NBodyGravity
    Sum of gravitational accelerations from multiple bodies using an
    ephemeris provider for time-dependent positions.

Mathematics (J2)
----------------
The gravitational potential including J₂:

    U = μ/r [1 − J₂(R_e/r)² P₂(sin φ)]

where P₂(x) = (3x²−1)/2 and sin φ = z/r.

Taking the gradient yields the perturbation acceleration above.

Reference
---------
Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., §8.6.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from theseus.dynamics.force_model import ForceModel
from theseus.bodies.body import CelestialBody
from theseus.ephemeris.provider import EphemerisProvider
from theseus.core.fidelity import (
    ModelFidelity, FidelityLevel, Assumption, FidelityRegistry,
)
from theseus.time.epochs import JD_J2000


class PointMassGravity(ForceModel):
    """
    Newtonian point-mass gravitational acceleration.

    a = −μ / |r|³ × r

    Parameters
    ----------
    body : CelestialBody
        Central body.
    """

    def __init__(self, body: CelestialBody, enabled: bool = True) -> None:
        super().__init__(name=f"PointMassGravity({body.name})", enabled=enabled)
        self.body = body
        self._fidelity = ModelFidelity(
            model_name=self.name,
            level=FidelityLevel.SIMPLIFIED,
            assumptions=[
                Assumption("point_mass", "Body treated as a point mass (no oblateness)"),
                Assumption("body_centred", "Position is relative to this body's centre"),
            ],
            source="Newton's law of gravitation",
        )
        FidelityRegistry.get().register(self._fidelity)

    def compute_acceleration(
        self, t: float, position: np.ndarray, velocity: np.ndarray, mass: float,
    ) -> np.ndarray:
        r_mag = float(np.linalg.norm(position))
        if r_mag < 1.0:  # avoid division by zero
            return np.zeros(3)
        return -self.body.mu / (r_mag ** 3) * position


class J2Perturbation(ForceModel):
    """
    J₂ oblateness perturbation acceleration.

    Perturbation only — does NOT include the point-mass term.
    Must be used together with PointMassGravity for a complete model.

    Parameters
    ----------
    body : CelestialBody
        Must have J2 and radius defined.
    """

    def __init__(self, body: CelestialBody, enabled: bool = True) -> None:
        super().__init__(name=f"J2Perturbation({body.name})", enabled=enabled)
        self.body = body
        if body.J2 == 0.0:
            raise ValueError(f"{body.name} has J2=0; J2 perturbation is meaningless")
        self._fidelity = ModelFidelity(
            model_name=self.name,
            level=FidelityLevel.MODERATE,
            assumptions=[
                Assumption("J2_only", "Only the J₂ zonal harmonic; higher-order terms ignored",
                           "Neglects J₃, J₄, tesseral harmonics"),
                Assumption("body_centred", "Position relative to body centre, body-fixed z = spin axis"),
            ],
            source="Vallado §8.6, derived from geopotential expansion",
            valid_domain=f"Near {body.name}; r > {body.radius:.0f} m",
        )
        FidelityRegistry.get().register(self._fidelity)

    def compute_acceleration(
        self, t: float, position: np.ndarray, velocity: np.ndarray, mass: float,
    ) -> np.ndarray:
        x, y, z = position[0], position[1], position[2]
        r_mag = float(np.linalg.norm(position))
        if r_mag < 1.0:
            return np.zeros(3)

        mu = self.body.mu
        J2 = self.body.J2
        Re = self.body.radius
        r2 = r_mag * r_mag
        r5 = r_mag ** 5
        z2_over_r2 = (z * z) / r2

        factor = 1.5 * mu * J2 * Re * Re / r5

        ax = factor * x * (5.0 * z2_over_r2 - 1.0)
        ay = factor * y * (5.0 * z2_over_r2 - 1.0)
        az = factor * z * (5.0 * z2_over_r2 - 3.0)

        return np.array([ax, ay, az])


class ThirdBodyGravity(ForceModel):
    """
    Gravitational acceleration from a third body (e.g. Moon, Sun)
    using ephemeris data.

    Uses the full third-body perturbation formula:

        a = μ_3 × [ (r_3 − r_sc)/|r_3 − r_sc|³  −  r_3/|r_3|³ ]

    where r_3 is the third body position (geocentric ICRF) and
    r_sc is the spacecraft position (geocentric ICRF).

    Parameters
    ----------
    body : CelestialBody
        The perturbing body.
    ephemeris : EphemerisProvider
        Provides the perturbing body's position.
    epoch_jd_t0 : float
        Julian Date corresponding to simulation time t=0.
    """

    def __init__(
        self,
        body: CelestialBody,
        ephemeris: EphemerisProvider,
        epoch_jd_t0: float = JD_J2000,
        enabled: bool = True,
    ) -> None:
        super().__init__(name=f"ThirdBodyGravity({body.name})", enabled=enabled)
        self.body = body
        self.ephemeris = ephemeris
        self.epoch_jd_t0 = epoch_jd_t0

    def compute_acceleration(
        self, t: float, position: np.ndarray, velocity: np.ndarray, mass: float,
    ) -> np.ndarray:
        jd = self.epoch_jd_t0 + t / 86400.0
        r_body = self.ephemeris.get_position(self.body.name, jd)  # geocentric

        # Vector from spacecraft to third body
        r_rel = r_body - position
        r_rel_mag = float(np.linalg.norm(r_rel))
        r_body_mag = float(np.linalg.norm(r_body))

        if r_rel_mag < 1.0 or r_body_mag < 1.0:
            return np.zeros(3)

        # Third-body perturbation
        return self.body.mu * (
            r_rel / r_rel_mag ** 3 - r_body / r_body_mag ** 3
        )
