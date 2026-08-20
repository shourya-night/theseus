"""
Impulsive and finite burn models.

Impulsive burns: instantaneous Δv application (no mass change during burn).
Finite burns:  continuous thrust over a duration, with mass depletion.

Fuel accounting uses the Tsiolkovsky rocket equation:
    Δv = v_e × ln(m₀ / m_f)
    m_f = m₀ × exp(−Δv / v_e)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from theseus.constants.physical import G0_VAL
from theseus.core.trace import CalculationTrace, TraceContext


@dataclass
class BurnResult:
    """Result of a burn computation."""
    delta_v: float                # m/s (scalar magnitude)
    delta_v_vector: np.ndarray    # m/s (3-vector)
    fuel_consumed: float          # kg
    burn_duration: float          # s (0 for impulsive)
    initial_mass: float           # kg
    final_mass: float             # kg
    specific_impulse: float       # s


def impulsive_burn(
    velocity: np.ndarray,
    delta_v_vector: np.ndarray,
    mass: float,
    specific_impulse: float,
    *,
    trace: bool = False,
) -> tuple[np.ndarray, BurnResult]:
    """
    Apply an instantaneous Δv to a velocity vector.

    Parameters
    ----------
    velocity : np.ndarray     Current velocity (m/s).
    delta_v_vector : np.ndarray  Δv to apply (m/s), 3-vector.
    mass : float              Current total mass (kg).
    specific_impulse : float  Engine Isp (s).

    Returns
    -------
    (new_velocity, BurnResult)
    """
    dv_mag = float(np.linalg.norm(delta_v_vector))
    ve = specific_impulse * G0_VAL

    # Fuel consumed: m_fuel = m₀ (1 − exp(−Δv/v_e))
    if ve > 1e-10 and dv_mag > 0:
        fuel = mass * (1.0 - math.exp(-dv_mag / ve))
    else:
        fuel = 0.0

    new_vel = velocity + delta_v_vector

    result = BurnResult(
        delta_v=dv_mag,
        delta_v_vector=delta_v_vector.copy(),
        fuel_consumed=fuel,
        burn_duration=0.0,
        initial_mass=mass,
        final_mass=mass - fuel,
        specific_impulse=specific_impulse,
    )

    if trace:
        ct = CalculationTrace(
            operation="impulsive_burn",
            equation="Δv applied instantaneously; m_f = m₀ exp(−Δv/v_e)",
            inputs={"v_m_s": velocity.tolist(), "dv_m_s": delta_v_vector.tolist(),
                    "mass_kg": mass, "Isp_s": specific_impulse},
            result={"new_v": new_vel.tolist(), "fuel_kg": fuel},
        )
        TraceContext.emit(ct)

    return new_vel, result


def finite_burn_duration(
    delta_v: float,
    thrust: float,
    mass: float,
    specific_impulse: float,
) -> float:
    """
    Compute burn duration for a given Δv with constant thrust.

    Uses  t_burn = (m₀ v_e / F) × (1 − exp(−Δv/v_e))

    Parameters
    ----------
    delta_v : float   Desired Δv (m/s).
    thrust : float    Engine thrust (N).
    mass : float      Initial total mass (kg).
    specific_impulse : float  Isp (s).

    Returns
    -------
    float   Burn duration (s).
    """
    ve = specific_impulse * G0_VAL
    if thrust < 1e-10 or ve < 1e-10:
        return float("inf")
    return (mass * ve / thrust) * (1.0 - math.exp(-delta_v / ve))


def fuel_for_delta_v(
    delta_v: float,
    mass: float,
    specific_impulse: float,
) -> float:
    """
    Propellant required for a given Δv.

    m_fuel = m₀ (1 − exp(−Δv/v_e))
    """
    ve = specific_impulse * G0_VAL
    if ve < 1e-10:
        return float("inf")
    return mass * (1.0 - math.exp(-delta_v / ve))


def delta_v_from_fuel(
    fuel_mass: float,
    total_mass: float,
    specific_impulse: float,
) -> float:
    """
    Δv obtainable from a given amount of fuel.

    Δv = v_e × ln(m₀ / (m₀ − m_fuel))
    """
    ve = specific_impulse * G0_VAL
    dry = total_mass - fuel_mass
    if dry <= 0 or total_mass <= 0:
        return 0.0
    return ve * math.log(total_mass / dry)
