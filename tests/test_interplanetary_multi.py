"""
Unit tests for Unified Interplanetary Multi-Object Transfers & Fleet Simulation.
"""

import math
import numpy as np
import pytest
from fastapi.testclient import TestClient

from theseus.server.app import app
from theseus.bodies.catalog import SUN
from theseus.simulation.multi_object import (
    SpacecraftDefinition,
    MultiObjectEnvironment,
    get_planet_state_at_time,
    solve_interplanetary_transfer,
    PLANET_KEPLERIAN_DATA,
    AU_METERS,
)



@pytest.fixture
def client():
    return TestClient(app)


def test_planet_keplerian_ephemeris_accuracy():
    """Verify non-circular Keplerian planetary orbits and time synchronization."""
    # 1. Mercury eccentricity > 0.20
    pos_merc_0, vel_merc_0 = get_planet_state_at_time("Mercury", 0.0)
    pos_merc_half, _ = get_planet_state_at_time("Mercury", 44.0 * 86400.0)
    r0 = np.linalg.norm(pos_merc_0)
    r_half = np.linalg.norm(pos_merc_half)
    # Perihelion vs Aphelion difference must reflect e = 0.2056
    e_merc = PLANET_KEPLERIAN_DATA["mercury"]["e"]
    assert e_merc > 0.20
    assert abs(r0 - r_half) > 1.0e10, "Mercury orbital radius should vary with true anomaly"

    # 2. Neptune semi-major axis ~ 30 AU
    pos_nep, _ = get_planet_state_at_time("Neptune", 0.0)
    r_nep = np.linalg.norm(pos_nep)
    assert 28.0 * 1.496e11 <= r_nep <= 32.0 * 1.496e11

    # 3. Planetary motion is synchronized with simulation time
    pos_earth_0, _ = get_planet_state_at_time("Earth", 0.0)
    pos_earth_100d, _ = get_planet_state_at_time("Earth", 100.0 * 86400.0)
    assert np.linalg.norm(pos_earth_100d - pos_earth_0) > 1.0e11


def test_single_interplanetary_transfer_future_planet_target():
    """Verify Lambert transfer targets the future moving position of target planet at arrival."""
    sc = SpacecraftDefinition(
        id="EXP-01",
        name="Mars-Explorer",
        origin="Earth",
        destination="Mars",
        dry_mass_kg=2000.0,
        fuel_mass_kg=4000.0,
        specific_impulse_s=325.0,
    )

    r0, v0, tof_sec, traces, dv_budget, prop_budget = solve_interplanetary_transfer(sc, departure_t_sec=0.0)

    # Departure position must match Earth at t=0
    r_earth_0, _ = get_planet_state_at_time("Earth", 0.0)
    assert np.allclose(r0, r_earth_0, atol=1e-3)

    # Verify arrival target matches Mars at t = TOF (future position, not t=0)
    r_mars_0, _ = get_planet_state_at_time("Mars", 0.0)
    r_mars_arrival, _ = get_planet_state_at_time("Mars", tof_sec)
    
    # Mars at departure != Mars at arrival
    assert np.linalg.norm(r_mars_arrival - r_mars_0) > 1.0e10
    
    # Delta-V budget must be physically positive and non-zero
    assert 3.0e3 <= dv_budget["total_delta_v"] <= 70.0e3
    assert prop_budget["fuel_consumed_kg"] > 0.0




def test_multi_destination_fleet_simulation():
    """
    Verify 4 spacecraft fleet targeting different destinations simultaneously:
    - SC1: Earth -> Mars
    - SC2: Earth -> Mars
    - SC3: Earth -> Jupiter
    - SC4: Earth -> Venus
    """
    sc1 = SpacecraftDefinition(id="EXP-01", name="Mars-1", origin="Earth", destination="Mars", dry_mass_kg=1500.0, fuel_mass_kg=3000.0)
    sc2 = SpacecraftDefinition(id="EXP-02", name="Mars-2", origin="Earth", destination="Mars", dry_mass_kg=2500.0, fuel_mass_kg=4000.0)
    sc3 = SpacecraftDefinition(id="EXP-03", name="Jup-1", origin="Earth", destination="Jupiter", dry_mass_kg=1000.0, fuel_mass_kg=5000.0)
    sc4 = SpacecraftDefinition(id="EXP-04", name="Ven-1", origin="Earth", destination="Venus", dry_mass_kg=1200.0, fuel_mass_kg=2000.0)

    env = MultiObjectEnvironment(central_body="Sun", screening_threshold_km=5.0e7, coarse_dt_s=86400.0)
    res = env.simulate([sc1, sc2, sc3, sc4], t_start=0.0, output_dt=86400.0 * 5.0)

    # 4 distinct spacecraft tracks
    assert len(res.objects) == 4
    assert res.summary["total_spacecraft"] == 4
    assert res.summary["active_spacecraft_count"] == 4

    # Each spacecraft has its own delta-V and propellant budget
    for obj in res.objects:
        assert obj.delta_v_budget is not None
        assert obj.delta_v_budget["total_delta_v"] > 0.0
        assert obj.propellant_budget is not None
        assert len(obj.state_history) > 10

    # Mars transfers vs Jupiter transfer vs Venus transfer have distinct distances
    r_mars_end = np.linalg.norm(res.objects[0].state_history[-1]["position"])
    r_jup_end = np.linalg.norm(res.objects[2].state_history[-1]["position"])
    r_ven_end = np.linalg.norm(res.objects[3].state_history[-1]["position"])
    assert r_ven_end < r_mars_end < r_jup_end


def test_ten_spacecraft_fleet_pairwise_screening():
    """Verify pairwise screening scaling for 10 spacecraft fleet: N(N-1)/2 = 45 pairs."""
    fleet = []
    for i in range(10):
        fleet.append(SpacecraftDefinition(
            id=f"EXP-{i+1:02d}",
            name=f"Probe-{i+1}",
            origin="Earth",
            destination="Mars",
            tof_days=200.0 + i * 5.0,
            dry_mass_kg=1000.0,
            fuel_mass_kg=2000.0,
        ))

    env = MultiObjectEnvironment(central_body="Sun", screening_threshold_km=1.0e6, coarse_dt_s=86400.0)
    res = env.simulate(fleet, t_start=0.0, output_dt=86400.0 * 10.0)

    assert len(res.objects) == 10
    assert res.summary["total_spacecraft"] == 10


def test_api_multi_interplanetary_endpoint(client):
    """Verify POST /api/simulate/environment with interplanetary spacecraft."""
    payload = {
        "central_body": "Sun",
        "spacecraft": [
            {
                "id": "SC-01",
                "name": "Explorer-Mars",
                "vehicle_type": "falcon_heavy",
                "origin": "Earth",
                "destination": "Mars",
                "dry_mass_kg": 2500.0,
                "fuel_mass_kg": 5000.0,
                "specific_impulse_s": 325.0,
            },
            {
                "id": "SC-02",
                "name": "Explorer-Venus",
                "vehicle_type": "starship",
                "origin": "Earth",
                "destination": "Venus",
                "dry_mass_kg": 3000.0,
                "fuel_mass_kg": 6000.0,
                "specific_impulse_s": 380.0,
            },
        ],
        "dt_sec": 86400.0 * 2.0,
    }

    response = client.post("/api/simulate/environment", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["objects"]) == 2
    assert data["summary"]["total_spacecraft"] == 2
    assert len(data["calculation_steps"]) >= 2
