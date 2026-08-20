"""
Rendezvous solver.

Given chaser and target states, computes the transfer trajectory
and maneuver plan using the Lambert solver.

Input:  chaser state (r, v), target state (r, v), rendezvous time.
Output: transfer trajectory, Δv budget, maneuver sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from theseus.orbital.lambert import solve_lambert, LambertSolution
from theseus.propagation.analytical import propagate_twobody
from theseus.core.state import StateHistory


@dataclass
class RendezvousResult:
    """Result of a rendezvous computation."""
    lambert_solution: LambertSolution
    delta_v_departure: np.ndarray     # m/s
    delta_v_arrival: np.ndarray       # m/s
    delta_v_total: float              # m/s (scalar total)
    time_of_flight: float             # s
    target_position_at_arrival: np.ndarray
    target_velocity_at_arrival: np.ndarray
    relative_velocity_at_arrival: float   # m/s
    transfer_trajectory: StateHistory | None = None
    description: str = ""


def solve_rendezvous(
    chaser_r: np.ndarray,
    chaser_v: np.ndarray,
    target_r: np.ndarray,
    target_v: np.ndarray,
    tof: float,
    mu: float,
    *,
    prograde: bool = True,
    compute_trajectory: bool = True,
    n_trajectory_points: int = 100,
) -> RendezvousResult:
    """
    Solve the rendezvous problem using Lambert's method.

    Parameters
    ----------
    chaser_r, chaser_v : np.ndarray
        Chaser state at departure (m, m/s).
    target_r, target_v : np.ndarray
        Target state at departure (m, m/s).
        The target is propagated forward by *tof* to find its position at arrival.
    tof : float
        Time of flight / time to rendezvous (s).
    mu : float
        Gravitational parameter (m³/s²).
    prograde : bool
        Transfer direction.
    compute_trajectory : bool
        If True, compute and return the transfer trajectory.
    n_trajectory_points : int
        Number of points in the transfer trajectory.

    Returns
    -------
    RendezvousResult
    """
    chaser_r = np.asarray(chaser_r, dtype=np.float64)
    chaser_v = np.asarray(chaser_v, dtype=np.float64)
    target_r = np.asarray(target_r, dtype=np.float64)
    target_v = np.asarray(target_v, dtype=np.float64)

    # Propagate target to arrival time (analytical two-body)
    target_at_arrival = propagate_twobody(
        target_r, target_v, mu, [tof], t0=0.0,
    )
    r2 = target_at_arrival[0].position
    v2_target = target_at_arrival[0].velocity

    # Solve Lambert problem
    lambert = solve_lambert(chaser_r, r2, tof, mu, prograde=prograde)

    if not lambert.converged:
        raise RuntimeError(
            f"Lambert solver did not converge: residual={lambert.residual:.3e}, "
            f"iterations={lambert.iterations}"
        )

    # Departure Δv
    dv_depart = lambert.v1 - chaser_v
    # Arrival Δv (to match target velocity)
    dv_arrive = v2_target - lambert.v2
    dv_total = float(np.linalg.norm(dv_depart)) + float(np.linalg.norm(dv_arrive))

    # Relative velocity at arrival
    rel_v = float(np.linalg.norm(lambert.v2 - v2_target))

    # Transfer trajectory
    trajectory = None
    if compute_trajectory:
        times = np.linspace(0.0, tof, n_trajectory_points)
        trajectory = propagate_twobody(chaser_r, lambert.v1, mu, times)

    return RendezvousResult(
        lambert_solution=lambert,
        delta_v_departure=dv_depart,
        delta_v_arrival=dv_arrive,
        delta_v_total=dv_total,
        time_of_flight=tof,
        target_position_at_arrival=r2,
        target_velocity_at_arrival=v2_target,
        relative_velocity_at_arrival=rel_v,
        transfer_trajectory=trajectory,
        description=f"Rendezvous: TOF={tof:.0f} s, Δv_total={dv_total:.1f} m/s",
    )
