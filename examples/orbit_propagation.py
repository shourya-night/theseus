"""
Example: Orbit Propagation

Propagates an ISS-like orbit (400 km altitude, 51.6° inclination)
for 2 orbital periods using both analytical and numerical methods.
Compares energy conservation between the two approaches.
"""

import math
import numpy as np

from theseus.bodies.catalog import EARTH
from theseus.orbital.conversions import state_to_elements
from theseus.propagation.analytical import propagate_twobody
from theseus.propagation.numerical import NumericalPropagator
from theseus.constants.units import m_to_km, rad_to_deg


def main():
    print("=" * 70)
    print("THESEUS -- Orbit Propagation Example")
    print("=" * 70)

    # --- Initial conditions: ISS-like orbit ---
    altitude = 400_000.0   # m (400 km)
    r0_mag = EARTH.radius + altitude
    inclination = math.radians(51.6)

    # Circular velocity
    v_circ = math.sqrt(EARTH.mu / r0_mag)

    # Position at ascending node, velocity inclined
    r0 = np.array([r0_mag, 0.0, 0.0])
    v0 = np.array([0.0, v_circ * math.cos(inclination), v_circ * math.sin(inclination)])

    # Orbital elements
    oe = state_to_elements(r0, v0, EARTH.mu)
    T = oe.period

    print(f"\nInitial Conditions (ISS-like orbit):")
    print(f"  Altitude:         {altitude / 1e3:.0f} km")
    print(f"  Orbital radius:   {m_to_km(r0_mag):.1f} km")
    print(f"  Velocity:         {v_circ:.1f} m/s")
    print(f"  Inclination:      {rad_to_deg(oe.i):.1f} deg")
    print(f"  Eccentricity:     {oe.e:.6f}")
    print(f"  Semi-major axis:  {m_to_km(oe.a):.1f} km")
    print(f"  Period:           {T:.1f} s ({T/60:.1f} min)")
    print(f"  Specific energy:  {oe.specific_energy:.2f} J/kg")
    print(f"  Angular momentum: {oe.specific_angular_momentum:.2f} m^2/s")

    # --- Analytical propagation (2 periods) ---
    n_points = 200
    times = np.linspace(0, 2 * T, n_points)
    history_analytical = propagate_twobody(r0, v0, EARTH.mu, times)

    print(f"\n--- Analytical Propagation (2 periods, {n_points} points) ---")
    print(f"  Final position:   [{', '.join(f'{x/1e3:.1f}' for x in history_analytical[-1].position)}] km")
    print(f"  Final velocity:   [{', '.join(f'{x:.2f}' for x in history_analytical[-1].velocity)}] m/s")

    # Energy conservation check
    e0 = 0.5 * np.dot(v0, v0) - EARTH.mu / np.linalg.norm(r0)
    e_final = 0.5 * np.dot(history_analytical[-1].velocity, history_analytical[-1].velocity) \
              - EARTH.mu / np.linalg.norm(history_analytical[-1].position)
    print(f"  Energy drift:     {abs((e_final - e0) / abs(e0)):.2e} (relative)")

    # --- Numerical propagation ---
    def accel(t, pos, vel, mass):
        r_mag = np.linalg.norm(pos)
        return -EARTH.mu / r_mag**3 * pos

    prop = NumericalPropagator(
        acceleration_fn=accel,
        integrator="rkf45",
        dt=30.0,
        atol=1e-12,
        rtol=1e-12,
        mu=EARTH.mu,
    )
    history_num, events, diag = prop.propagate(r0, v0, (0, 2 * T))

    print(f"\n--- Numerical Propagation (RKF45 adaptive) ---")
    print(f"  Steps taken:      {len(history_num)}")
    print(f"  Final position:   [{', '.join(f'{x/1e3:.1f}' for x in history_num[-1].position)}] km")
    print(f"  Energy drift:     {diag.max_energy_drift():.2e} (relative)")
    print(f"  h drift:          {diag.max_angular_momentum_drift():.2e} (relative)")

    # Position difference after 2 orbits
    pos_diff = np.linalg.norm(history_analytical[-1].position - history_num[-1].position)
    print(f"\n  Position difference (analytical vs numerical): {pos_diff:.3f} m")
    print(f"  Events logged: {len(events)}")

    print("\n" + "=" * 70)
    print("Propagation complete.")


if __name__ == "__main__":
    main()
