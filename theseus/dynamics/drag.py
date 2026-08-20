"""
Atmospheric drag force model.

F_D = ½ ρ v_rel² C_D A
a_D = F_D / m    (opposing velocity relative to atmosphere)

The relative velocity accounts for Earth's rotation:
    v_atm = ω_earth × r    (atmosphere co-rotates with Earth)
    v_rel = v_inertial − v_atm

Reference
---------
Vallado §8.6.2.
"""

from __future__ import annotations

import numpy as np

from theseus.dynamics.force_model import ForceModel
from theseus.atmosphere.models import AtmosphereModel
from theseus.bodies.catalog import EARTH
from theseus.core.fidelity import (
    ModelFidelity, FidelityLevel, Assumption, FidelityRegistry,
)


class DragModel(ForceModel):
    """
    Atmospheric drag as a force model.

    Parameters
    ----------
    atmosphere : AtmosphereModel
    cd : float     Drag coefficient (dimensionless).
    area : float   Reference cross-section area (m²).
    body_radius : float  Central body radius (m), for altitude computation.
    body_rotation_rate : float  Body rotation rate (rad/s).
    """

    def __init__(
        self,
        atmosphere: AtmosphereModel,
        cd: float = 2.2,
        area: float = 10.0,
        body_radius: float = EARTH.radius,
        body_rotation_rate: float = 7.2921159e-5,  # Earth
        enabled: bool = True,
    ) -> None:
        super().__init__(name="AtmosphericDrag", enabled=enabled)
        self.atmosphere = atmosphere
        self.cd = cd
        self.area = area
        self.body_radius = body_radius
        self.omega = body_rotation_rate
        self._fidelity = ModelFidelity(
            model_name="DragModel",
            level=FidelityLevel.MODERATE,
            assumptions=[
                Assumption("co_rotating_atmosphere",
                           "Atmosphere co-rotates rigidly with Earth"),
                Assumption("constant_cd",
                           "Drag coefficient is constant (no Mach/Reynolds dependence)"),
                Assumption("flat_plate",
                           "Drag area is constant regardless of attitude"),
            ],
            source="Vallado §8.6.2",
        )
        FidelityRegistry.get().register(self._fidelity)

    def compute_acceleration(
        self, t: float, position: np.ndarray, velocity: np.ndarray, mass: float,
    ) -> np.ndarray:
        # Altitude
        r_mag = float(np.linalg.norm(position))
        alt = r_mag - self.body_radius
        if alt < 0:
            alt = 0.0

        # Atmospheric density
        rho = self.atmosphere.density(alt)
        if rho < 1e-30:
            return np.zeros(3)

        # Atmospheric velocity (co-rotating): v_atm = ω × r
        omega_vec = np.array([0.0, 0.0, self.omega])
        v_atm = np.cross(omega_vec, position)

        # Relative velocity
        v_rel = velocity - v_atm
        v_rel_mag = float(np.linalg.norm(v_rel))
        if v_rel_mag < 1e-10:
            return np.zeros(3)

        # Drag acceleration: a_D = −½ ρ v² Cd A / m × v̂_rel
        factor = -0.5 * rho * v_rel_mag * self.cd * self.area / mass
        return factor * v_rel


class LiftModel(ForceModel):
    """
    Aerodynamic lift force model.

    F_L = ½ ρ v_rel² C_L A
    Lift direction: perpendicular to v_rel, in the plane defined by
    v_rel and the local vertical (radial direction).

    Parameters
    ----------
    atmosphere : AtmosphereModel
    cl : float     Lift coefficient.
    area : float   Reference area (m²).
    body_radius : float
    body_rotation_rate : float
    """

    def __init__(
        self,
        atmosphere: AtmosphereModel,
        cl: float = 0.0,
        area: float = 10.0,
        body_radius: float = EARTH.radius,
        body_rotation_rate: float = 7.2921159e-5,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="AerodynamicLift", enabled=enabled)
        self.atmosphere = atmosphere
        self.cl = cl
        self.area = area
        self.body_radius = body_radius
        self.omega = body_rotation_rate
        self._fidelity = ModelFidelity(
            model_name="LiftModel",
            level=FidelityLevel.SIMPLIFIED,
            assumptions=[
                Assumption("constant_cl", "Lift coefficient constant"),
                Assumption("lift_in_vertical_plane",
                           "Lift perpendicular to velocity in velocity-vertical plane"),
            ],
            source="Standard aerodynamic lift equation",
        )
        FidelityRegistry.get().register(self._fidelity)

    def compute_acceleration(
        self, t: float, position: np.ndarray, velocity: np.ndarray, mass: float,
    ) -> np.ndarray:
        if abs(self.cl) < 1e-15:
            return np.zeros(3)

        r_mag = float(np.linalg.norm(position))
        alt = r_mag - self.body_radius
        if alt < 0:
            alt = 0.0

        rho = self.atmosphere.density(alt)
        if rho < 1e-30:
            return np.zeros(3)

        omega_vec = np.array([0.0, 0.0, self.omega])
        v_atm = np.cross(omega_vec, position)
        v_rel = velocity - v_atm
        v_rel_mag = float(np.linalg.norm(v_rel))
        if v_rel_mag < 1e-10:
            return np.zeros(3)

        # Lift direction: perpendicular to v_rel, toward the "up" side
        # Compute as: L̂ = (v̂_rel × (r̂ × v̂_rel)) normalised
        r_hat = position / r_mag
        v_hat = v_rel / v_rel_mag
        # Cross product to find the component of r̂ perpendicular to v̂
        perp = r_hat - np.dot(r_hat, v_hat) * v_hat
        perp_mag = float(np.linalg.norm(perp))
        if perp_mag < 1e-15:
            return np.zeros(3)
        lift_dir = perp / perp_mag

        factor = 0.5 * rho * v_rel_mag * v_rel_mag * self.cl * self.area / mass
        return factor * lift_dir
