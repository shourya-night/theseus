"""
Abstract force-model interface and composite force model.

Every physical force/acceleration in THESEUS implements ForceModel.
CompositeForceModel sums contributions from individually-toggleable models.

    a_total = a_gravity + a_J2 + a_drag + a_thrust + a_SRP + …
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from theseus.core.fidelity import ModelFidelity


class ForceModel(ABC):
    """
    Base class for all force/acceleration models.

    Subclasses implement ``compute_acceleration`` and may optionally
    declare their fidelity descriptor.
    """

    def __init__(self, name: str, enabled: bool = True) -> None:
        self.name = name
        self.enabled = enabled
        self._fidelity: Optional[ModelFidelity] = None

    @abstractmethod
    def compute_acceleration(
        self,
        t: float,
        position: np.ndarray,
        velocity: np.ndarray,
        mass: float,
    ) -> np.ndarray:
        """
        Compute acceleration contribution.

        Parameters
        ----------
        t : float           Simulation time (s).
        position : (3,)     Position in ICRF (m).
        velocity : (3,)     Velocity in ICRF (m/s).
        mass : float        Current spacecraft mass (kg).

        Returns
        -------
        np.ndarray  (3,)    Acceleration (m/s²).
        """
        ...

    @property
    def fidelity(self) -> Optional[ModelFidelity]:
        return self._fidelity


class CompositeForceModel:
    """
    Aggregates multiple ForceModel instances.

    Only enabled models contribute to the total acceleration.
    """

    def __init__(self, models: list[ForceModel] | None = None) -> None:
        self._models: list[ForceModel] = list(models) if models else []

    def add(self, model: ForceModel) -> None:
        """Add a force model."""
        self._models.append(model)

    def remove(self, name: str) -> None:
        """Remove a force model by name."""
        self._models = [m for m in self._models if m.name != name]

    def enable(self, name: str) -> None:
        """Enable a force model by name."""
        for m in self._models:
            if m.name == name:
                m.enabled = True

    def disable(self, name: str) -> None:
        """Disable a force model by name."""
        for m in self._models:
            if m.name == name:
                m.enabled = False

    @property
    def models(self) -> list[ForceModel]:
        return list(self._models)

    @property
    def active_models(self) -> list[ForceModel]:
        return [m for m in self._models if m.enabled]

    def compute_acceleration(
        self,
        t: float,
        position: np.ndarray,
        velocity: np.ndarray,
        mass: float,
    ) -> np.ndarray:
        """Total acceleration from all enabled models."""
        a_total = np.zeros(3, dtype=np.float64)
        for model in self._models:
            if model.enabled:
                a_total += model.compute_acceleration(t, position, velocity, mass)
        return a_total

    def summary(self) -> list[dict]:
        """Return a summary of all models and their status."""
        return [
            {"name": m.name, "enabled": m.enabled, "type": type(m).__name__}
            for m in self._models
        ]
