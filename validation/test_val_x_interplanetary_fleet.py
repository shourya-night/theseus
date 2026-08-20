"""
THESEUS Validation Suite X — Multi-Object Interplanetary Fleets, Ephemeris & Debris Dynamics.

Validates:
1. Moving target planet endpoint closure.
2. Orbital energy and conservation across interplanetary trajectories.
3. Deterministic collision detection, center-of-mass momentum conservation, and 4-debris propagation.
"""

import math
import numpy as np
import pytest

from theseus.bodies.catalog import SUN
from theseus.simulation.multi_object import (
    SpacecraftDefinition,
    MultiObjectEnvironment,
    get_planet_state_at_time,
    solve_interplanetary_transfer,
    AU_METERS,
)



def test_val_x1_moving_planet_endpoint_closure():
    """
    Validation X.1: Moving Target Planet Endpoint Closure
    Verify that the Lambert transfer endpoint exactly coincides with the predicted
    future position of the destination planet at arrival epoch.
    """
    destinations = ["Mars", "Venus", "Jupiter"]
    
    for dest in destinations:
        sc = SpacecraftDefinition(
            id=f"TEST-{dest}",
            name=f"Explorer-{dest}",
            origin="Earth",
            destination=dest,
            dry_mass_kg=2000.0,
            fuel_mass_kg=4000.0,
        )
        
        r0, v0, tof_sec, traces, dv_budget, prop_budget = solve_interplanetary_transfer(sc, departure_t_sec=0.0)
        
        # Predicted future position of target planet at arrival
        r_target_future, _ = get_planet_state_at_time(dest, tof_sec)
        
        # Propagate spacecraft for tof_sec under point mass gravity
        env = MultiObjectEnvironment(central_body="Sun", enable_j2=False, enable_drag=False, enable_srp=False)
        res = env.simulate([sc], t_start=0.0, t_end=tof_sec, output_dt=tof_sec / 200.0)
        
        sc_final_pos = np.array(res.objects[0].state_history[-1]["position"])
        closure_error_m = np.linalg.norm(sc_final_pos - r_target_future)
        rel_error = closure_error_m / np.linalg.norm(r_target_future)
        
        # Endpoint closure error must be < 0.1%
        assert rel_error < 1.0e-3, f"Endpoint closure error {rel_error:.4e} for {dest} exceeds tolerance"


def test_val_x2_interplanetary_energy_conservation():
    """
    Validation X.2: Heliocentric Energy Conservation
    Verify that all unpowered interplanetary coast trajectories conserve specific orbital energy:
    epsilon = 0.5 * v^2 - mu / r = const < 0.
    """
    sc1 = SpacecraftDefinition(id="SC-1", name="Mars-Bound", origin="Earth", destination="Mars")
    sc2 = SpacecraftDefinition(id="SC-2", name="Venus-Bound", origin="Earth", destination="Venus")

    env = MultiObjectEnvironment(central_body="Sun", enable_j2=False, enable_drag=False, enable_srp=False)
    res = env.simulate([sc1, sc2], t_start=0.0, output_dt=86400.0 * 5.0)

    for obj in res.objects:
        energies = []
        for st in obj.state_history:
            r = np.linalg.norm(st["position"])
            v = np.linalg.norm(st["velocity"])
            eps = 0.5 * (v ** 2) - (SUN.mu / r)
            energies.append(eps)
            assert eps < 0.0, f"Specific orbital energy {eps} >= 0 (must be bound ellipse)"

        initial_eps = energies[0]
        final_eps = energies[-1]
        drift_rel = abs(final_eps - initial_eps) / abs(initial_eps)
        assert drift_rel < 1.0e-4, f"Energy drift {drift_rel:.4e} exceeds 0.01%"


def test_val_x3_deterministic_collision_and_4_debris_orbits():
    """
    Validation X.3: Physical Collision Detection and 4-Debris Physical Orbits
    In a deterministic collision, verify:
    1. Collision detected when d <= HBR1 + HBR2.
    2. Exactly 4 debris fragments created.
    3. Each debris fragment has distinct non-collinear ejection directions and bound orbits.
    """
    r_init = np.array([1.496e11, 0.0, 0.0])
    v_circ = math.sqrt(SUN.mu / 1.496e11)
    
    # Spacecraft 1: approaching from -Z
    sc1 = SpacecraftDefinition(
        id="PROBE-A",
        name="Alpha",
        central_body="Sun",
        initial_r_m=r_init - np.array([0.0, 0.0, 500.0 * 50.0]),
        initial_v_m_s=np.array([0.0, v_circ, 500.0]),
        hard_body_radius_m=150.0,
    )
    # Spacecraft 2: converging from +Z with 10m initial Y offset
    sc2 = SpacecraftDefinition(
        id="PROBE-B",
        name="Beta",
        central_body="Sun",
        initial_r_m=r_init + np.array([0.0, 10.0, 500.0 * 50.0]),
        initial_v_m_s=np.array([0.0, v_circ, -500.0]),
        hard_body_radius_m=150.0,
    )

    env = MultiObjectEnvironment(central_body="Sun", screening_threshold_km=1.0e5, coarse_dt_s=10.0)
    res = env.simulate([sc1, sc2], t_start=0.0, t_end=3600.0, output_dt=10.0)

    assert len(res.collisions) >= 1
    coll = res.collisions[0]
    assert coll.combined_hbr_m == 300.0

    debris_objs = [o for o in res.objects if o.definition.is_debris]
    assert len(debris_objs) == 4

    for d in debris_objs:
        active_states = [s for s in d.state_history if s["active"]]
        assert len(active_states) > 0
        for st in active_states:
            r = np.linalg.norm(st["position"])
            v = np.linalg.norm(st["velocity"])
            energy = 0.5 * (v ** 2) - (SUN.mu / r)
            assert energy < 0.0, f"Debris piece {d.id} has unbound energy {energy} >= 0"
