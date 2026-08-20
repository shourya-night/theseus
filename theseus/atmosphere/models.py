"""
Atmospheric density, pressure, and temperature models.

ExponentialAtmosphere
    ρ = ρ₀ exp(−h/H)     Simple single-scale-height model.

US1976StandardAtmosphere
    US Standard Atmosphere 1976.  Piecewise temperature profile with
    lapse-rate and isothermal layers from 0 to 86 km geopotential altitude.

Both implement a common interface:  get_properties(altitude_m) → (ρ, P, T).

Source
------
US Standard Atmosphere, 1976.  NOAA/NASA/USAF.
https://ntrs.nasa.gov/api/citations/19770009539/downloads/19770009539.pdf

Limitations
-----------
- US1976 is valid 0–86 km geopotential altitude.
- Does not model thermospheric density (> 86 km); returns exponential
  extrapolation above 86 km.
- No solar-activity or geomagnetic-storm variations.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from theseus.constants.physical import G0_VAL, R_GAS_VAL, M_AIR_VAL
from theseus.core.fidelity import (
    ModelFidelity, FidelityLevel, Assumption, FidelityRegistry,
)


@dataclass(frozen=True)
class AtmosphericProperties:
    """Atmospheric state at a given altitude."""
    density: float       # kg/m³
    pressure: float      # Pa
    temperature: float   # K
    altitude: float      # m (geopotential)


class AtmosphereModel(ABC):
    """Abstract atmospheric model interface."""

    @abstractmethod
    def get_properties(self, altitude_m: float) -> AtmosphericProperties:
        """
        Compute atmospheric properties at a given altitude.

        Parameters
        ----------
        altitude_m : float   Geometric altitude above mean sea level (m).

        Returns
        -------
        AtmosphericProperties
        """
        ...

    @abstractmethod
    def density(self, altitude_m: float) -> float:
        """Density (kg/m³) at altitude."""
        ...


class ExponentialAtmosphere(AtmosphereModel):
    """
    Simple exponential atmosphere:  ρ = ρ₀ exp(−h/H).

    Parameters
    ----------
    rho0 : float   Sea-level density (kg/m³).  Default 1.225.
    scale_height : float  Scale height (m).  Default 8500.
    """

    def __init__(self, rho0: float = 1.225, scale_height: float = 8500.0) -> None:
        self.rho0 = rho0
        self.H = scale_height
        self._fidelity = ModelFidelity(
            model_name="ExponentialAtmosphere",
            level=FidelityLevel.SIMPLIFIED,
            assumptions=[
                Assumption("single_scale_height", "Constant scale height (isothermal)",
                           "Density profile deviates from reality at all altitudes by up to ~50%"),
            ],
            valid_domain="0–1000 km (order-of-magnitude)",
            source="Vallado, simplified exponential model",
            limitations="Single-layer approximation; no temperature/pressure profile",
        )
        FidelityRegistry.get().register(self._fidelity)

    def density(self, altitude_m: float) -> float:
        if altitude_m < 0:
            altitude_m = 0.0
        return self.rho0 * math.exp(-altitude_m / self.H)

    def get_properties(self, altitude_m: float) -> AtmosphericProperties:
        rho = self.density(altitude_m)
        T = 288.15  # assumed constant (isothermal approximation)
        P = rho * R_GAS_VAL * T / M_AIR_VAL
        return AtmosphericProperties(density=rho, pressure=P, temperature=T, altitude=altitude_m)


# US Standard Atmosphere 1976 layer definitions
# Each tuple: (base_altitude_m, base_temperature_K, lapse_rate_K_per_m)
_US76_LAYERS = [
    (0.0,     288.15,  -0.0065),
    (11000.0, 216.65,   0.0),
    (20000.0, 216.65,   0.001),
    (32000.0, 228.65,   0.0028),
    (47000.0, 270.65,   0.0),
    (51000.0, 270.65,  -0.0028),
    (71000.0, 214.65,  -0.002),
]

# Precomputed base pressures for each layer (Pa)
# P at h=0: 101325 Pa.  Subsequent layers computed from barometric formula.
_US76_BASE_PRESSURES: list[float] = []


def _compute_base_pressures() -> list[float]:
    """Compute base pressures for each US76 layer."""
    if _US76_BASE_PRESSURES:
        return _US76_BASE_PRESSURES

    pressures = [101325.0]
    for i in range(len(_US76_LAYERS) - 1):
        h_b, T_b, L = _US76_LAYERS[i]
        h_next = _US76_LAYERS[i + 1][0]
        dh = h_next - h_b
        P_b = pressures[-1]

        if abs(L) < 1e-10:
            # Isothermal
            P_next = P_b * math.exp(-G0_VAL * M_AIR_VAL * dh / (R_GAS_VAL * T_b))
        else:
            # Lapse rate
            T_next = T_b + L * dh
            exponent = G0_VAL * M_AIR_VAL / (R_GAS_VAL * L)
            P_next = P_b * (T_next / T_b) ** (-exponent)

        pressures.append(P_next)

    _US76_BASE_PRESSURES.extend(pressures)
    return _US76_BASE_PRESSURES


class US1976StandardAtmosphere(AtmosphereModel):
    """
    US Standard Atmosphere 1976.

    Valid 0 to 86 km geopotential altitude.
    Above 86 km: exponential extrapolation from top-layer density.
    """

    def __init__(self) -> None:
        _compute_base_pressures()
        self._fidelity = ModelFidelity(
            model_name="US1976StandardAtmosphere",
            level=FidelityLevel.MODERATE,
            assumptions=[
                Assumption("us76_model", "US Standard Atmosphere 1976 profile",
                           "Does not include solar-activity or weather variations"),
                Assumption("geopotential_altitude",
                           "Uses geopotential altitude (≈ geometric for h < 86 km)"),
            ],
            valid_domain="0–86 km geopotential altitude",
            source="US Standard Atmosphere, 1976 (NOAA/NASA/USAF)",
            limitations="No thermospheric model; exponential extrapolation above 86 km",
        )
        FidelityRegistry.get().register(self._fidelity)

    def _find_layer(self, h: float) -> int:
        """Find the layer index for altitude h."""
        for i in range(len(_US76_LAYERS) - 1, -1, -1):
            if h >= _US76_LAYERS[i][0]:
                return i
        return 0

    def get_properties(self, altitude_m: float) -> AtmosphericProperties:
        if altitude_m < 0:
            altitude_m = 0.0

        # Above 86 km: exponential extrapolation
        if altitude_m > 86000.0:
            # Get properties at 86 km, then exponential decay
            props_86 = self._compute(86000.0)
            H = 6000.0  # approximate scale height above 86 km
            rho = props_86.density * math.exp(-(altitude_m - 86000.0) / H)
            T = 186.87  # approximate at 86 km
            P = rho * R_GAS_VAL * T / M_AIR_VAL
            return AtmosphericProperties(density=rho, pressure=P, temperature=T, altitude=altitude_m)

        return self._compute(altitude_m)

    def _compute(self, h: float) -> AtmosphericProperties:
        """Compute properties for 0 ≤ h ≤ 86000 m."""
        layer = self._find_layer(h)
        h_b, T_b, L = _US76_LAYERS[layer]
        P_b = _US76_BASE_PRESSURES[layer]
        dh = h - h_b

        # Temperature
        T = T_b + L * dh

        # Pressure
        if abs(L) < 1e-10:
            P = P_b * math.exp(-G0_VAL * M_AIR_VAL * dh / (R_GAS_VAL * T_b))
        else:
            exponent = G0_VAL * M_AIR_VAL / (R_GAS_VAL * L)
            P = P_b * (T / T_b) ** (-exponent)

        # Density from ideal gas law
        rho = P * M_AIR_VAL / (R_GAS_VAL * T)

        return AtmosphericProperties(density=rho, pressure=P, temperature=T, altitude=h)

    def density(self, altitude_m: float) -> float:
        return self.get_properties(altitude_m).density
