"""
Spacecraft vehicle model.

All values in SI units (kg, m, m², s, N).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from theseus.constants.physical import G0_VAL


@dataclass
class Spacecraft:
    """
    Spacecraft configuration and state.

    Parameters
    ----------
    name : str
    dry_mass : float           Dry mass (kg).
    fuel_mass : float          Propellant mass (kg).
    cross_section_area : float Reference cross-section area (m²).
    drag_coefficient : float   Cd (dimensionless).
    reflectivity_coefficient : float  Cr (dimensionless, 1.0–2.0 typical).
    lift_coefficient : float   Cl (dimensionless).
    thrust : float             Maximum thrust per engine (N).
    specific_impulse : float   Isp (s).
    num_engines : int          Number of engines.
    """
    name: str = "spacecraft"
    dry_mass: float = 500.0
    fuel_mass: float = 200.0
    cross_section_area: float = 10.0       # m²
    drag_coefficient: float = 2.2          # typical for LEO
    reflectivity_coefficient: float = 1.5  # 1.0 = absorb, 2.0 = full specular reflection
    lift_coefficient: float = 0.0
    thrust: float = 500.0                  # N
    specific_impulse: float = 300.0        # s
    num_engines: int = 1

    @property
    def total_mass(self) -> float:
        """Total mass = dry + fuel (kg)."""
        return self.dry_mass + self.fuel_mass

    @property
    def exhaust_velocity(self) -> float:
        """Effective exhaust velocity  v_e = Isp × g₀  (m/s)."""
        return self.specific_impulse * G0_VAL

    @property
    def max_thrust(self) -> float:
        """Total available thrust (N)."""
        return self.thrust * self.num_engines

    @property
    def mass_flow_rate(self) -> float:
        """ṁ = F / (Isp × g₀)  (kg/s) at maximum thrust."""
        ve = self.exhaust_velocity
        if ve < 1e-10:
            return 0.0
        return self.max_thrust / ve

    def delta_v_available(self) -> float:
        """
        Available Δv from Tsiolkovsky rocket equation (m/s).

        Δv = v_e × ln(m₀ / m_f)
        """
        if self.fuel_mass <= 0 or self.dry_mass <= 0:
            return 0.0
        ve = self.exhaust_velocity
        return ve * math.log(self.total_mass / self.dry_mass)

    def fuel_required(self, delta_v: float) -> float:
        """
        Propellant mass required for a given Δv (kg).

        m_fuel = m₀ (1 − exp(−Δv/v_e))

        where m₀ = total current mass.
        """
        ve = self.exhaust_velocity
        if ve < 1e-10:
            return float("inf")
        return self.total_mass * (1.0 - math.exp(-delta_v / ve))

    def consume_fuel(self, dm: float) -> float:
        """
        Consume *dm* kg of fuel.  Returns actual amount consumed
        (may be less if fuel runs out).
        """
        actual = min(dm, self.fuel_mass)
        self.fuel_mass -= actual
        return actual
