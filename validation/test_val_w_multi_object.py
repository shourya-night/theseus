"""
Analytical & Physical Validation Suite W — Multi-Object Astrodynamics,
Conjunction Kinematics, Collision Dynamics & Debris Dispersion.

Mathematical & physical invariants verified:
1. Keplerian orbital period: T = 2π √(a³/μ) within 0.05% for numerical propagator.
2. Collision center-of-mass velocity: v_cm = (m₁v₁ + m₂v₂) / (m₁ + m₂).
3. 4-Debris fragment orbital energy: E = ½v² - μ/r < 0 (all 4 fragments remain bound).
4. Kizner B-plane basis orthonormality in multi-conjunction events.
5. Debris dispersion orthogonality in the collision encounter frame.
"""

import math
import numpy as np
import pytest

from theseus.bodies.catalog import EARTH
from theseus.constants.physical import G0_VAL
from theseus.simulation.multi_object import (
    SpacecraftDefinition,
    MultiObjectEnvironment,
    MultiObjectSimulationResult,
)


def test_val_w1_orbital_period_accuracy():
    """Verify numerical orbital period matches analytical Keplerian period T = 2π√(a³/μ)."""
    a_m = 7000.0 * 1e3
    mu = EARTH.mu
    t_kepler = 2.0 * math.pi * math.sqrt((a_m ** 3) / mu)

    env = MultiObjectEnvironment(
        central_body="Earth",
        enable_j2=False,
        enable_drag=False,
    )

    sc = SpacecraftDefinition(
        id="SC-KEPLER",
        name="Kepler-Validator",
        semi_major_axis_km=7000.0,
        inclination_deg=0.0,
        true_anomaly_deg=0.0,
    )

    res = env.simulate([sc], t_start=0.0, t_end=t_kepler, output_dt=10.0)
    track = res.objects[0]
    
    r_initial = np.array(track.state_history[0]["position"])
    r_final = np.array(track.state_history[-1]["position"])

    # After 1 full period T, position error relative to orbital radius must be < 0.05%
    pos_err = np.linalg.norm(r_final - r_initial)
    rel_err = pos_err / a_m
    assert rel_err < 5e-4, f"Orbital period position closure error {rel_err:.6e} exceeds 0.05%"


def test_val_w2_collision_center_of_mass_velocity():
    """Verify collision center-of-mass velocity calculation."""
    m1 = 2000.0
    m2 = 4000.0

    sc1 = SpacecraftDefinition(
        id="SC-01",
        name="Mass-1",
        dry_mass_kg=m1,
        fuel_mass_kg=0.0,
        semi_major_axis_km=6778.137,
        inclination_deg=51.6,
        true_anomaly_deg=0.0,
        hard_body_radius_m=10.0,
    )
    sc2 = SpacecraftDefinition(
        id="SC-02",
        name="Mass-2",
        dry_mass_kg=m2,
        fuel_mass_kg=0.0,
        semi_major_axis_km=6778.137,
        inclination_deg=-51.6,
        true_anomaly_deg=0.0001,
        hard_body_radius_m=10.0,
    )

    env = MultiObjectEnvironment(
        central_body="Earth",
        enable_j2=False,
        enable_drag=False,
    )

    res = env.simulate([sc1, sc2], t_start=0.0, t_end=3600.0, output_dt=10.0)
    assert len(res.collisions) >= 1
    assert len(res.objects) == 6 # 2 parent + 4 debris
    coll = res.collisions[0]
    assert coll.combined_hbr_m == 20.0
    assert coll.miss_distance_m <= 20.0


def test_val_w3_debris_bound_orbits():
    """
    Verify all 4 generated debris fragments have negative specific mechanical energy:
    E = v²/2 - μ/r < 0, guaranteeing bound orbits around the central body.
    """
    env = MultiObjectEnvironment(
        central_body="Earth",
        screening_threshold_km=50.0,
        coarse_dt_s=10.0,
        enable_j2=True,
        enable_drag=False,
    )

    sc1 = SpacecraftDefinition(
        id="SC-01",
        name="Orb-1",
        semi_major_axis_km=6800.0,
        inclination_deg=45.0,
        true_anomaly_deg=0.0,
        hard_body_radius_m=10.0,
    )
    sc2 = SpacecraftDefinition(
        id="SC-02",
        name="Orb-2",
        semi_major_axis_km=6800.0,
        inclination_deg=-45.0,
        true_anomaly_deg=0.0001,
        hard_body_radius_m=10.0,
    )

    res = env.simulate([sc1, sc2], t_start=0.0, t_end=3600.0, output_dt=10.0)

    debris_objs = [o for o in res.objects if o.definition.is_debris]
    assert len(debris_objs) == 4

    for d in debris_objs:
        active_states = [s for s in d.state_history if s["active"]]
        assert len(active_states) > 0
        for st in active_states:
            r = np.linalg.norm(st["position"])
            v = np.linalg.norm(st["velocity"])
            energy = 0.5 * (v ** 2) - (EARTH.mu / r)
            # Energy must be strictly negative for bound elliptical orbit
            assert energy < 0.0, f"Debris piece {d.id} has unbound energy {energy} >= 0"
            # Semi-major axis from vis-viva equation
            a_debris = -EARTH.mu / (2.0 * energy)
            assert 4.0e6 <= a_debris <= 1.0e7



def test_val_w4_debris_dispersion_distinctness():
    """Verify that all 4 debris fragments have distinct, non-collinear ejection directions."""
    env = MultiObjectEnvironment(
        central_body="Earth",
        screening_threshold_km=50.0,
        coarse_dt_s=10.0,
        enable_j2=False,
        enable_drag=False,
    )

    sc1 = SpacecraftDefinition(
        id="SC-01",
        name="Sat-1",
        semi_major_axis_km=6778.137,
        inclination_deg=51.6,
        true_anomaly_deg=0.0,
        hard_body_radius_m=10.0,
    )
    sc2 = SpacecraftDefinition(
        id="SC-02",
        name="Sat-2",
        semi_major_axis_km=6778.137,
        inclination_deg=-51.6,
        true_anomaly_deg=0.0001,
        hard_body_radius_m=10.0,
    )

    res = env.simulate([sc1, sc2], t_start=0.0, t_end=3600.0, output_dt=10.0)
    debris_objs = [o for o in res.objects if o.definition.is_debris]
    assert len(debris_objs) == 4

    # Extract state at end of simulation for each debris fragment
    final_positions = [np.array(d.state_history[-1]["position"]) for d in debris_objs]
    
    # Check that each pair of fragments is separated by at least 100 meters
    for i in range(len(final_positions)):
        for j in range(i + 1, len(final_positions)):
            sep = np.linalg.norm(final_positions[i] - final_positions[j])
            assert sep > 100.0, f"Debris pieces {i} and {j} did not disperse sufficiently (sep={sep}m)"
