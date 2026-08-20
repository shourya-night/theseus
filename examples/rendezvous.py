"""
Example: Rendezvous in LEO

A chaser spacecraft intercepts a target spacecraft in a slightly
higher orbit using the Lambert solver.
"""

import math
import numpy as np

from theseus.bodies.catalog import EARTH
from theseus.rendezvous.solver import solve_rendezvous
from theseus.constants.units import m_to_km


def main():
    print("=" * 70)
    print("THESEUS -- LEO Rendezvous Example")
    print("=" * 70)

    mu = EARTH.mu

    # Chaser: 400 km circular orbit
    r1 = EARTH.radius + 400_000.0
    v1 = math.sqrt(mu / r1)

    # Target: 420 km circular orbit, 60 deg ahead
    r2 = EARTH.radius + 420_000.0
    v2 = math.sqrt(mu / r2)
    angle = math.radians(60)

    chaser_r = np.array([r1, 0.0, 0.0])
    chaser_v = np.array([0.0, v1, 0.0])
    target_r = np.array([r2 * math.cos(angle), r2 * math.sin(angle), 0.0])
    target_v = np.array([-v2 * math.sin(angle), v2 * math.cos(angle), 0.0])

    print(f"\n--- Initial Conditions ---")
    print(f"  Chaser altitude:  {(r1 - EARTH.radius)/1e3:.0f} km")
    print(f"  Target altitude:  {(r2 - EARTH.radius)/1e3:.0f} km")
    print(f"  Target lead:      {math.degrees(angle):.0f} deg")
    print(f"  Chaser velocity:  {v1:.1f} m/s")
    print(f"  Target velocity:  {v2:.1f} m/s")

    # Try several time-of-flight options
    for tof_hours in [1.0, 2.0, 4.0]:
        tof = tof_hours * 3600
        try:
            result = solve_rendezvous(
                chaser_r, chaser_v,
                target_r, target_v,
                tof, mu,
                compute_trajectory=True,
                n_trajectory_points=50,
            )

            print(f"\n--- TOF = {tof_hours:.0f} hours ---")
            print(f"  Departure Delta-v:{np.linalg.norm(result.delta_v_departure):.2f} m/s")
            print(f"  Arrival Delta-v:  {np.linalg.norm(result.delta_v_arrival):.2f} m/s")
            print(f"  Total Delta-v:    {result.delta_v_total:.2f} m/s")
            print(f"  Relative vel:     {result.relative_velocity_at_arrival:.2f} m/s")
            print(f"  Lambert converged: {result.lambert_solution.converged}")
            print(f"  Lambert iters:    {result.lambert_solution.iterations}")
            print(f"  Transfer type:    {result.lambert_solution.trajectory_type}")
            if result.transfer_trajectory:
                print(f"  Trajectory pts:   {len(result.transfer_trajectory)}")
        except Exception as e:
            print(f"\n--- TOF = {tof_hours:.0f} hours: FAILED ({e}) ---")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
