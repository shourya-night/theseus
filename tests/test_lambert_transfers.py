"""Tests for Lambert solver, transfers, and rendezvous."""

import math

import numpy as np
import pytest

from theseus.orbital.lambert import solve_lambert
from theseus.maneuvers.transfers import (
    hohmann_transfer, bielliptic_transfer, plane_change, combined_maneuver,
)
from theseus.rendezvous.solver import solve_rendezvous


MU_EARTH = 3.986004418e14  # m³/s²


# ===================================================================
# Lambert solver
# ===================================================================

class TestLambert:

    def test_half_orbit_transfer(self):
        """180° transfer (Hohmann-like): from [r1,0,0] to [-r2,0,0]."""
        r1 = 7_000_000.0
        r2 = 42_164_000.0  # GEO
        # Use Hohmann TOF
        a_t = (r1 + r2) / 2
        tof = math.pi * math.sqrt(a_t**3 / MU_EARTH)

        pos1 = np.array([r1, 0.0, 0.0])
        pos2 = np.array([-r2, 0.0, 0.0])  # opposite side

        sol = solve_lambert(pos1, pos2, tof, MU_EARTH, prograde=True)
        assert sol.converged
        assert sol.iterations > 0
        # Departure velocity should be larger than circular
        v_circ = math.sqrt(MU_EARTH / r1)
        assert np.linalg.norm(sol.v1) > v_circ

    def test_known_circular_orbit(self):
        """Quarter orbit on a circular orbit should recover circular velocity."""
        r = 7_000_000.0
        v_circ = math.sqrt(MU_EARTH / r)
        T = 2 * math.pi * math.sqrt(r**3 / MU_EARTH)
        tof = T / 4.0  # quarter orbit

        pos1 = np.array([r, 0.0, 0.0])
        pos2 = np.array([0.0, r, 0.0])  # 90° ahead

        sol = solve_lambert(pos1, pos2, tof, MU_EARTH, prograde=True)
        assert sol.converged
        # Velocities should match circular orbit velocities
        assert np.linalg.norm(sol.v1) == pytest.approx(v_circ, rel=1e-4)
        assert np.linalg.norm(sol.v2) == pytest.approx(v_circ, rel=1e-4)

    def test_convergence_diagnostics(self):
        """Verify convergence info is present."""
        r1 = np.array([5000e3, 10000e3, 2100e3])
        r2 = np.array([-14600e3, 2500e3, 7000e3])
        tof = 3600.0
        sol = solve_lambert(r1, r2, tof, MU_EARTH)
        assert sol.converged
        assert sol.residual < 1e-6
        assert sol.trajectory_type in ("elliptic", "hyperbolic", "parabolic")

    def test_short_transfer(self):
        """Short time of flight (close positions)."""
        r1 = np.array([7000e3, 0.0, 0.0])
        r2 = np.array([7000e3 * math.cos(0.1), 7000e3 * math.sin(0.1), 0.0])
        tof = 300.0  # 5 minutes
        sol = solve_lambert(r1, r2, tof, MU_EARTH)
        assert sol.converged


# ===================================================================
# Transfers
# ===================================================================

class TestTransfers:

    def test_hohmann_leo_to_geo(self):
        """
        LEO (r=6678 km, 300 km alt) → GEO (r=42164 km).
        Exact analytical: Δv = 3892.61 m/s, T = 18974.77 s (~5.271 hrs).
        """
        r1 = 6_678_000.0   # LEO (300 km altitude)
        r2 = 42_164_000.0  # GEO

        result = hohmann_transfer(r1, r2, MU_EARTH)

        # Exact analytical derivation:
        # vc1 = sqrt(mu/r1) = 7725.82 m/s, vc2 = sqrt(mu/r2) = 3074.66 m/s
        # at = (r1+r2)/2 = 24421000 m
        # vt1 = sqrt(mu*(2/r1 - 1/at)) = 10145.42 m/s -> dv1 = 2419.60 m/s
        # vt2 = sqrt(mu*(2/r2 - 1/at)) = 1607.65 m/s -> dv2 = 1467.01 m/s
        # total_dv = 3892.61 m/s
        assert result.total_delta_v == pytest.approx(3892.61, rel=1e-3)
        assert result.delta_v1 == pytest.approx(2425.77, rel=1e-3)
        assert result.delta_v2 == pytest.approx(1466.84, rel=1e-3)
        assert result.transfer_time == pytest.approx(18974.77, rel=1e-3)

    def test_hohmann_200km_leo_to_geo(self):
        """
        LEO (r=6578 km, 200 km alt) → GEO (r=42164 km).
        Curtis example: Δv ≈ 3935.84 m/s.
        """
        r1 = 6_578_000.0   # LEO (200 km altitude)
        r2 = 42_164_000.0  # GEO
        result = hohmann_transfer(r1, r2, MU_EARTH)
        assert result.total_delta_v == pytest.approx(3935.84, rel=1e-3)

    def test_hohmann_symmetric(self):
        """Hohmann Δv is the same whether going up or down (magnitudes)."""
        r1 = 7_000_000.0
        r2 = 14_000_000.0
        up = hohmann_transfer(r1, r2, MU_EARTH)
        down = hohmann_transfer(r2, r1, MU_EARTH)
        assert up.total_delta_v == pytest.approx(down.total_delta_v, rel=1e-8)

    def test_bielliptic_vs_hohmann(self):
        """
        For r2/r1 > 11.94, bi-elliptic is more efficient.
        For r2/r1 < 11.94, Hohmann is more efficient.
        """
        r1 = 7_000_000.0
        r2 = r1 * 15  # ratio = 15 > 11.94 → bi-elliptic better
        r_int = r1 * 50

        hoh = hohmann_transfer(r1, r2, MU_EARTH)
        bie = bielliptic_transfer(r1, r2, r_int, MU_EARTH)

        # Bi-elliptic should have lower total Δv for this ratio
        assert bie.total_delta_v < hoh.total_delta_v

    def test_plane_change(self):
        """Δv for 28.5° inclination change at circular velocity."""
        v = 7_700.0  # typical LEO velocity
        di = math.radians(28.5)
        dv = plane_change(v, di)
        expected = 2 * v * math.sin(di / 2)
        assert dv == pytest.approx(expected)

    def test_combined_maneuver(self):
        """Combined altitude + inclination is cheaper than separate."""
        r1 = 6_678_000.0
        r2 = 42_164_000.0
        di = math.radians(28.5)

        combined = combined_maneuver(r1, r2, di, MU_EARTH)
        hohmann_dv = hohmann_transfer(r1, r2, MU_EARTH).total_delta_v
        pc_dv = plane_change(math.sqrt(MU_EARTH / r2), di)

        # Combined should be cheaper than Hohmann + separate plane change
        assert combined.total_delta_v < hohmann_dv + pc_dv


# ===================================================================
# Rendezvous
# ===================================================================

class TestRendezvous:

    def test_coplanar_rendezvous(self):
        """Chaser in lower orbit intercepts target in higher orbit."""
        r1 = 6_778_000.0  # 400 km altitude
        r2 = 6_878_000.0  # 500 km altitude
        v1 = math.sqrt(MU_EARTH / r1)
        v2 = math.sqrt(MU_EARTH / r2)

        chaser_r = np.array([r1, 0.0, 0.0])
        chaser_v = np.array([0.0, v1, 0.0])
        target_r = np.array([0.0, r2, 0.0])  # 90° ahead
        target_v = np.array([-v2, 0.0, 0.0])

        tof = 3600.0  # 1 hour

        result = solve_rendezvous(
            chaser_r, chaser_v, target_r, target_v,
            tof, MU_EARTH,
        )

        assert result.delta_v_total > 0
        assert result.delta_v_total < 5000  # should be reasonable
        assert result.time_of_flight == tof
        if result.transfer_trajectory is not None:
            assert len(result.transfer_trajectory) > 10
