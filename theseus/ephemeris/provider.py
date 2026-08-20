"""
Ephemeris provider interface.

All ephemeris providers implement this ABC, so the physics engine never
depends on a specific data source.

Contract
--------
* ``get_position(body_name, epoch_jd)`` → position in ICRF (m).
* ``get_state(body_name, epoch_jd)`` → (position, velocity) in ICRF (m, m/s).
* Reference frame: ICRF / J2000, geocentric unless otherwise noted.
* Time scale: TDB (Barycentric Dynamical Time) for JD input.
* Units: metres, metres/second.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EphemerisProvider(ABC):
    """
    Abstract ephemeris provider.

    Subclasses supply positions and velocities of celestial bodies
    at arbitrary epochs.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""
        ...

    @property
    @abstractmethod
    def source(self) -> str:
        """Data source / reference (e.g. 'DE430', 'circular-approx')."""
        ...

    @property
    @abstractmethod
    def precision_description(self) -> str:
        """Human-readable precision statement."""
        ...

    @abstractmethod
    def get_position(self, body_name: str, epoch_jd: float) -> np.ndarray:
        """
        Position of *body_name* at *epoch_jd* (TDB).

        Returns
        -------
        np.ndarray (3,)   Position in ICRF (m).
        """
        ...

    @abstractmethod
    def get_state(self, body_name: str, epoch_jd: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Position and velocity of *body_name* at *epoch_jd*.

        Returns
        -------
        (position, velocity)   ICRF, (m, m/s).
        """
        ...
