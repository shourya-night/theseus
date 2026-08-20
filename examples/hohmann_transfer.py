"""
Example: Hohmann Transfer — LEO to GEO

Computes the classic LEO → GEO Hohmann transfer and prints
the Δv budget, transfer time, and fuel requirements.
"""

import math
import numpy as np

from theseus.bodies.catalog import EARTH
from theseus.maneuvers.transfers import hohmann_transfer, bielliptic_transfer, combined_maneuver
from theseus.spacecraft.vehicle import Spacecraft
from theseus.maneuvers.burns import fuel_for_delta_v
from theseus.constants.units import m_to_km


def main():
    print("=" * 70)
    print("THESEUS -- Hohmann Transfer: LEO -> GEO")
    print("=" * 70)

    # Orbit parameters
    r_leo = EARTH.radius + 300_000.0   # 300 km altitude
    r_geo = 42_164_000.0               # GEO radius

    mu = EARTH.mu

    # --- Hohmann transfer ---
    hoh = hohmann_transfer(r_leo, r_geo, mu, trace=True)

    print(f"\n--- Hohmann Transfer ---")
    print(f"  LEO radius:       {m_to_km(r_leo):.0f} km  (altitude {(r_leo - EARTH.radius)/1e3:.0f} km)")
    print(f"  GEO radius:       {m_to_km(r_geo):.0f} km  (altitude {(r_geo - EARTH.radius)/1e3:.0f} km)")
    print(f"  Transfer SMA:     {m_to_km(hoh.transfer_a):.0f} km")
    print(f"  Delta-v1 (LEO burn): {hoh.delta_v1:.1f} m/s")
    print(f"  Delta-v2 (GEO burn): {hoh.delta_v2:.1f} m/s")
    print(f"  Total Delta-v:       {hoh.total_delta_v:.1f} m/s  ({hoh.total_delta_v/1e3:.3f} km/s)")
    print(f"  Transfer time:       {hoh.transfer_time:.0f} s  ({hoh.transfer_time/3600:.2f} hours)")

    # --- Fuel requirements ---
    sc = Spacecraft(
        name="Transfer Vehicle",
        dry_mass=2000.0,      # 2 tonnes dry
        fuel_mass=3000.0,     # 3 tonnes fuel
        specific_impulse=316, # bipropellant engine
    )

    fuel_needed = fuel_for_delta_v(hoh.total_delta_v, sc.total_mass, sc.specific_impulse)

    print(f"\n--- Fuel Budget ---")
    print(f"  Spacecraft:       {sc.name}")
    print(f"  Dry mass:         {sc.dry_mass:.0f} kg")
    print(f"  Fuel mass:        {sc.fuel_mass:.0f} kg")
    print(f"  Isp:              {sc.specific_impulse:.0f} s")
    print(f"  Available Delta-v:{sc.delta_v_available():.1f} m/s")
    print(f"  Fuel required:    {fuel_needed:.1f} kg")
    print(f"  Fuel margin:      {sc.fuel_mass - fuel_needed:.1f} kg")

    # --- Combined transfer with inclination change ---
    delta_i = math.radians(28.5)  # Cape Canaveral launch inclination
    combined = combined_maneuver(r_leo, r_geo, delta_i, mu)

    print(f"\n--- Combined Transfer (with 28.5 deg plane change) ---")
    print(f"  Delta-v1 (LEO burn): {combined.delta_v1:.1f} m/s")
    print(f"  Delta-v2 (GEO burn): {combined.delta_v2:.1f} m/s")
    print(f"  Total Delta-v:       {combined.total_delta_v:.1f} m/s  ({combined.total_delta_v/1e3:.3f} km/s)")
    print(f"  Transfer time:       {combined.transfer_time/3600:.2f} hours")

    # --- Bi-elliptic comparison ---
    r_int = 100_000_000.0  # 100,000 km intermediate
    bie = bielliptic_transfer(r_leo, r_geo, r_int, mu)

    print(f"\n--- Bi-elliptic Transfer (r_int = {m_to_km(r_int):.0f} km) ---")
    print(f"  Delta-v1:         {bie.delta_v1:.1f} m/s")
    print(f"  Delta-v2:         {bie.delta_v2:.1f} m/s")
    print(f"  Delta-v3:         {bie.delta_v3:.1f} m/s")
    print(f"  Total Delta-v:    {bie.total_delta_v:.1f} m/s")
    print(f"  (Hohmann is {'better' if hoh.total_delta_v < bie.total_delta_v else 'worse'} "
          f"for this ratio r2/r1 = {r_geo/r_leo:.1f})")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
