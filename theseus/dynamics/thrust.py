"""
Thrust force model.

Computes acceleration from spacecraft engine thrust, including
propellant consumption.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Callable

import numpy as np

from theseus.dynamics.force_model import ForceModel
from theseus.spacecraft.vehicle import Spacecraft
from theseus.core.fidelity import (
    ModelFidelity, FidelityLevel, Assumption, FidelityRegistry,
)


class ThrustDirection(Enum):
    """Pre-defined thrust direction modes."""
    PROGRADE = "prograde"
    RETROGRADE = "retrograde"
    NORMAL = "normal"
    ANTI_NORMAL = "anti_normal"
    RADIAL_IN = "radial_in"
    RADIAL_OUT = "radial_out"
    CUSTOM = "custom"


class ThrustModel(ForceModel):
    """
    Engine thrust as a force model.

    Parameters
    ----------
    spacecraft : Spacecraft
    direction : ThrustDirection
        Pre-defined direction mode.
    custom_direction_fn : callable or None
        If direction is CUSTOM, this function ``f(t, r, v) -> unit_vec``
        supplies the thrust direction as a unit vector.
    burn_start : float
        Simulation time when thrust begins (s).
    burn_end : float
        Simulation time when thrust ends (s).
    throttle : float
        Throttle level (0.0–1.0).
    """

    def __init__(
        self,
        spacecraft: Spacecraft,
        direction: ThrustDirection = ThrustDirection.PROGRADE,
        custom_direction_fn: Optional[Callable] = None,
        burn_start: float = 0.0,
        burn_end: float = float("inf"),
        throttle: float = 1.0,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="Thrust", enabled=enabled)
        self.spacecraft = spacecraft
        self.direction = direction
        self.custom_direction_fn = custom_direction_fn
        self.burn_start = burn_start
        self.burn_end = burn_end
        self.throttle = max(0.0, min(1.0, throttle))
        self._fidelity = ModelFidelity(
            model_name="ThrustModel",
            level=FidelityLevel.MODERATE,
            assumptions=[
                Assumption("constant_thrust", "Thrust magnitude constant during burn"),
                Assumption("constant_isp", "Specific impulse constant during burn"),
            ],
            source="Tsiolkovsky rocket equation",
        )

    def _get_direction(self, t: float, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Compute unit thrust direction vector in ICRF."""
        v_mag = np.linalg.norm(velocity)
        r_mag = np.linalg.norm(position)

        if self.direction == ThrustDirection.PROGRADE:
            if v_mag < 1e-10:
                return np.array([1.0, 0.0, 0.0])
            return velocity / v_mag

        elif self.direction == ThrustDirection.RETROGRADE:
            if v_mag < 1e-10:
                return np.array([-1.0, 0.0, 0.0])
            return -velocity / v_mag

        elif self.direction == ThrustDirection.NORMAL:
            h = np.cross(position, velocity)
            h_mag = np.linalg.norm(h)
            if h_mag < 1e-10:
                return np.array([0.0, 0.0, 1.0])
            return h / h_mag

        elif self.direction == ThrustDirection.ANTI_NORMAL:
            h = np.cross(position, velocity)
            h_mag = np.linalg.norm(h)
            if h_mag < 1e-10:
                return np.array([0.0, 0.0, -1.0])
            return -h / h_mag

        elif self.direction == ThrustDirection.RADIAL_OUT:
            if r_mag < 1e-10:
                return np.array([1.0, 0.0, 0.0])
            return position / r_mag

        elif self.direction == ThrustDirection.RADIAL_IN:
            if r_mag < 1e-10:
                return np.array([-1.0, 0.0, 0.0])
            return -position / r_mag

        elif self.direction == ThrustDirection.CUSTOM:
            if self.custom_direction_fn is None:
                raise ValueError("Custom direction function not provided")
            d = self.custom_direction_fn(t, position, velocity)
            d_mag = np.linalg.norm(d)
            if d_mag < 1e-15:
                return np.array([1.0, 0.0, 0.0])
            return d / d_mag

        raise ValueError(f"Unknown direction: {self.direction}")

    def compute_acceleration(
        self, t: float, position: np.ndarray, velocity: np.ndarray, mass: float,
    ) -> np.ndarray:
        # Check if within burn window and fuel available
        if t < self.burn_start or t > self.burn_end:
            return np.zeros(3)
        if self.spacecraft.fuel_mass <= 0:
            return np.zeros(3)
        if mass < 1e-10:
            return np.zeros(3)

        F = self.spacecraft.max_thrust * self.throttle
        direction = self._get_direction(t, position, velocity)
        return (F / mass) * direction
