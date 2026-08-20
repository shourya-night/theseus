"""
Orbital transfer computations.

hohmann_transfer     : Two-impulse minimum-energy coplanar transfer.
bielliptic_transfer  : Three-impulse transfer via intermediate altitude.
plane_change         : Δv for simple inclination change.
combined_maneuver    : Combined altitude + inclination change.

All computations assume circular initial and final orbits unless
otherwise noted.

References
----------
Curtis, "Orbital Mechanics for Engineering Students", 4th ed., §6.
Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., §6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from theseus.core.trace import CalculationTrace, TraceContext


@dataclass
class TransferResult:
    """Result of a transfer computation."""
    delta_v1: float          # first burn Δv (m/s)
    delta_v2: float          # second burn Δv (m/s)
    delta_v3: float          # third burn Δv (m/s), 0 for Hohmann
    total_delta_v: float     # total Δv (m/s)
    transfer_time: float     # transfer duration (s)
    transfer_a: float        # transfer orbit semi-major axis (m)
    description: str = ""


def hohmann_transfer(
    r1: float,
    r2: float,
    mu: float,
    *,
    trace: bool = False,
) -> TransferResult:
    """
    Hohmann transfer between two coplanar circular orbits.

    Parameters
    ----------
    r1 : float   Initial circular orbit radius (m).
    r2 : float   Final circular orbit radius (m).
    mu : float   Gravitational parameter (m³/s²).

    Returns
    -------
    TransferResult

    Mathematics
    -----------
    Transfer semi-major axis:  a_t = (r1 + r2) / 2
    v_circ(r) = √(μ/r)
    v_transfer_peri = √(2μ/r1 − μ/a_t)
    v_transfer_apo  = √(2μ/r2 − μ/a_t)

    Δv₁ = v_transfer_peri − v_circ(r1)   (at periapsis)
    Δv₂ = v_circ(r2) − v_transfer_apo    (at apoapsis)
    """
    v1_circ = math.sqrt(mu / r1)
    v2_circ = math.sqrt(mu / r2)
    a_t = (r1 + r2) / 2.0

    v_transfer_1 = math.sqrt(2.0 * mu / r1 - mu / a_t)
    v_transfer_2 = math.sqrt(2.0 * mu / r2 - mu / a_t)

    dv1 = abs(v_transfer_1 - v1_circ)
    dv2 = abs(v2_circ - v_transfer_2)

    T_transfer = math.pi * math.sqrt(a_t ** 3 / mu)

    result = TransferResult(
        delta_v1=dv1,
        delta_v2=dv2,
        delta_v3=0.0,
        total_delta_v=dv1 + dv2,
        transfer_time=T_transfer,
        transfer_a=a_t,
        description=f"Hohmann transfer: r1={r1/1e3:.0f} km → r2={r2/1e3:.0f} km",
    )

    if trace:
        ct = CalculationTrace(
            operation="hohmann_transfer",
            equation="a_t = (r1+r2)/2; Δv1 = v_t1−v_c1; Δv2 = v_c2−v_t2",
            inputs={"r1_m": r1, "r2_m": r2, "mu": mu},
            result={
                "dv1_m_s": dv1, "dv2_m_s": dv2,
                "total_dv_m_s": dv1 + dv2,
                "transfer_time_s": T_transfer,
                "a_t_m": a_t,
            },
        )
        ct.add_step("circular_velocities", v1=v1_circ, v2=v2_circ)
        ct.add_step("transfer_velocities", v_t1=v_transfer_1, v_t2=v_transfer_2)
        TraceContext.emit(ct)

    return result


def bielliptic_transfer(
    r1: float,
    r2: float,
    r_intermediate: float,
    mu: float,
    *,
    trace: bool = False,
) -> TransferResult:
    """
    Bi-elliptic transfer via an intermediate radius.

    Three burns:
    1. r1 → transfer ellipse 1 (periapsis r1, apoapsis r_intermediate)
    2. At r_intermediate: circularise / transition to transfer ellipse 2
    3. At r2: circularise

    This can be more efficient than Hohmann when r2/r1 > 11.94.
    """
    v1_circ = math.sqrt(mu / r1)
    v2_circ = math.sqrt(mu / r2)

    a_t1 = (r1 + r_intermediate) / 2.0
    a_t2 = (r2 + r_intermediate) / 2.0

    # Burn 1: at r1, enter first transfer ellipse
    v_t1_peri = math.sqrt(2.0 * mu / r1 - mu / a_t1)
    dv1 = abs(v_t1_peri - v1_circ)

    # At intermediate: velocity on first transfer ellipse
    v_t1_apo = math.sqrt(2.0 * mu / r_intermediate - mu / a_t1)
    # Velocity needed on second transfer ellipse at intermediate
    v_t2_apo = math.sqrt(2.0 * mu / r_intermediate - mu / a_t2)
    dv2 = abs(v_t2_apo - v_t1_apo)

    # Burn 3: at r2
    v_t2_peri = math.sqrt(2.0 * mu / r2 - mu / a_t2)
    dv3 = abs(v2_circ - v_t2_peri)

    T1 = math.pi * math.sqrt(a_t1 ** 3 / mu)
    T2 = math.pi * math.sqrt(a_t2 ** 3 / mu)

    return TransferResult(
        delta_v1=dv1, delta_v2=dv2, delta_v3=dv3,
        total_delta_v=dv1 + dv2 + dv3,
        transfer_time=T1 + T2,
        transfer_a=a_t1,
        description=f"Bi-elliptic: r1={r1/1e3:.0f} → r_i={r_intermediate/1e3:.0f} → r2={r2/1e3:.0f} km",
    )


def plane_change(
    v: float,
    delta_i: float,
) -> float:
    """
    Δv for a simple inclination change at constant altitude.

    Δv = 2v sin(Δi/2)

    Parameters
    ----------
    v : float       Orbital velocity (m/s).
    delta_i : float Inclination change (rad).

    Returns
    -------
    float   Δv (m/s).
    """
    return 2.0 * v * abs(math.sin(delta_i / 2.0))


def combined_maneuver(
    r1: float,
    r2: float,
    delta_i: float,
    mu: float,
) -> TransferResult:
    """
    Combined altitude change + inclination change.

    The inclination change is performed at the apoapsis of the
    Hohmann transfer (where velocity is lowest) for minimum Δv.

    Δv₂ = √(v_t2² + v_c2² − 2 v_t2 v_c2 cos(Δi))
    """
    v1_circ = math.sqrt(mu / r1)
    a_t = (r1 + r2) / 2.0
    v_transfer_1 = math.sqrt(2.0 * mu / r1 - mu / a_t)
    dv1 = abs(v_transfer_1 - v1_circ)

    v2_circ = math.sqrt(mu / r2)
    v_transfer_2 = math.sqrt(2.0 * mu / r2 - mu / a_t)
    # Combined burn at apoapsis
    dv2 = math.sqrt(
        v_transfer_2 ** 2 + v2_circ ** 2
        - 2.0 * v_transfer_2 * v2_circ * math.cos(delta_i)
    )

    T_transfer = math.pi * math.sqrt(a_t ** 3 / mu)

    return TransferResult(
        delta_v1=dv1, delta_v2=dv2, delta_v3=0.0,
        total_delta_v=dv1 + dv2,
        transfer_time=T_transfer,
        transfer_a=a_t,
        description=f"Combined: r1={r1/1e3:.0f}→r2={r2/1e3:.0f} km, Δi={math.degrees(delta_i):.1f}°",
    )
