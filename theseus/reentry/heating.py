"""
Aerothermal heating models for atmospheric reentry.

Sutton-Graves Convective Heating
    q̇_conv = k √(ρ / r_n) V³       [W/m²]

    k = 1.7415×10⁻⁴   for Earth air (N₂/O₂)   [kg^0.5 / m]

    Source: Sutton, K. and Graves, R.A.,
    "A General Stagnation-Point Convective-Heating Equation for
    Arbitrary Gas Mixtures", NASA TR R-376, 1971.

Stagnation Temperature
    T_s = T_∞ (1 + (γ−1)/2 · M²)    [K]

    Isentropic stagnation temperature for a calorically-perfect gas.

Limitations
-----------
- Sutton-Graves is an *engineering estimate*, not a CFD result.
  Valid order-of-magnitude for continuum hypersonic flow (Kn < 0.01).
- Assumes equilibrium boundary-layer chemistry.
- No catalytic-wall effects.
- Stagnation temperature assumes calorically-perfect gas (γ = const),
  which breaks down above ~2000 K due to molecular dissociation.
- Radiative heating is NOT implemented (see module docstring below).

Radiative Heating — Status
--------------------------
Radiative heating (significant above ~10 km/s for Earth) is **not enabled**
in this release.  Implementing it requires validated coefficients for a
specific radiative-transfer model (e.g. Tauber-Sutton correlation).
Fabricating coefficients would violate THESEUS's scientific-honesty policy.

When radiative heating is disabled, the API will report:

    RADIATIVE HEATING: NOT ENABLED
"""

from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sutton-Graves constant for Earth atmosphere (N₂/O₂ mixture)
# Units: kg^0.5 / m   (such that q̇ [W/m²] = k √(ρ [kg/m³] / r_n [m]) V³ [m/s]³)
SUTTON_GRAVES_K_EARTH: float = 1.7415e-4

# Ratio of specific heats for air at moderate temperatures
GAMMA_AIR: float = 1.4

# Mean molar mass of dry air (kg/mol) — used for speed-of-sound
M_AIR: float = 0.0289644

# Universal gas constant (J/(mol·K))
R_GAS: float = 8.314462618


def sutton_graves_convective(
    rho: float,
    velocity: float,
    nose_radius: float,
    k: float = SUTTON_GRAVES_K_EARTH,
) -> float:
    """
    Sutton-Graves stagnation-point convective heating rate.

    q̇ = k √(ρ / r_n) V³     [W/m²]

    ENGINEERING ESTIMATE — NOT CFD.

    Parameters
    ----------
    rho : float
        Atmospheric density (kg/m³).
    velocity : float
        Vehicle speed relative to atmosphere (m/s).
    nose_radius : float
        Effective nose radius (m).  Must be > 0.
    k : float
        Sutton-Graves constant.  Default = 1.7415e-4 (Earth air).

    Returns
    -------
    float
        Convective heat flux at the stagnation point (W/m²).

    Notes
    -----
    - Source: Sutton & Graves, NASA TR R-376 (1971).
    - Valid for continuum hypersonic regime (Kn ≪ 1).
    - Larger nose radius → lower heat flux (blunt bodies spread heat).
    - Heat flux scales as V³ — very sensitive to entry speed.
    """
    if rho < 0 or velocity < 0 or nose_radius <= 0:
        return 0.0
    return k * math.sqrt(rho / nose_radius) * velocity ** 3


def stagnation_temperature(
    freestream_temp: float,
    mach: float,
    gamma: float = GAMMA_AIR,
) -> float:
    """
    Isentropic stagnation temperature.

    T_s = T_∞ (1 + (γ−1)/2 · M²)     [K]

    Parameters
    ----------
    freestream_temp : float
        Freestream static temperature (K).
    mach : float
        Mach number (dimensionless).
    gamma : float
        Ratio of specific heats.  Default 1.4 (diatomic air).

    Returns
    -------
    float
        Stagnation temperature (K).

    Limitations
    -----------
    - Assumes calorically-perfect gas (constant γ).
    - Breaks down above ~2000 K due to dissociation.
    """
    if freestream_temp <= 0 or mach < 0:
        return freestream_temp
    return freestream_temp * (1.0 + 0.5 * (gamma - 1.0) * mach * mach)


def speed_of_sound(temperature: float, gamma: float = GAMMA_AIR) -> float:
    """
    Speed of sound in an ideal gas.

    a = √(γ R T / M)     [m/s]

    Parameters
    ----------
    temperature : float
        Static temperature (K).
    gamma : float
        Ratio of specific heats.
    """
    if temperature <= 0:
        return 0.0
    return math.sqrt(gamma * R_GAS * temperature / M_AIR)


def mach_number(velocity: float, temperature: float) -> float:
    """
    Mach number  M = V / a.

    Parameters
    ----------
    velocity : float
        Speed (m/s).
    temperature : float
        Atmospheric temperature (K).

    Returns
    -------
    float
        Mach number (dimensionless).
    """
    a = speed_of_sound(temperature)
    if a < 1e-10:
        return 0.0
    return velocity / a


def dynamic_pressure(rho: float, velocity: float) -> float:
    """
    Dynamic pressure q = ½ ρ V²   [Pa].
    """
    if rho < 0 or velocity < 0:
        return 0.0
    return 0.5 * rho * velocity * velocity


def aerodynamic_force(
    rho: float, velocity: float, coefficient: float, area: float,
) -> float:
    """
    Aerodynamic force  F = ½ ρ V² C A   [N].

    Used for both drag (C = C_D) and lift (C = C_L).
    """
    return 0.5 * rho * velocity * velocity * coefficient * area


def heating_model_metadata() -> dict[str, Any]:
    """Return metadata describing the active heating models."""
    return {
        "convective": {
            "model": "Sutton-Graves stagnation-point correlation",
            "equation": "q̇ = k √(ρ/r_n) V³",
            "constant_k": SUTTON_GRAVES_K_EARTH,
            "constant_k_units": "kg^0.5 / m",
            "source": "Sutton & Graves, NASA TR R-376 (1971)",
            "validity": "Continuum hypersonic flow (Kn << 1), Earth N₂/O₂ atmosphere",
            "classification": "ENGINEERING ESTIMATE — NOT CFD",
        },
        "radiative": {
            "model": "NOT ENABLED",
            "reason": "No validated radiative-transfer correlation coefficients available",
            "note": "Radiative heating is significant above ~10 km/s for Earth entry; "
                    "omission underestimates total heating at lunar/Mars-return speeds",
        },
    }
