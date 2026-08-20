"""
Unit tests for THESEUS Multi-Object Simulation Environment.

Tests:
1. Multi-spacecraft propagation across N objects.
2. Pairwise conjunction screening and TCA refinement.
3. Physical collision threshold detection (miss <= HBR_comb).
4. Generation and numerical propagation of exactly 4 debris fragments.
5. FastAPI endpoints /api/simulate/environment and /api/simulate/multi.
"""

import math
import numpy as np
import pytest
from fastapi.testclient import TestClient

from theseus.bodies.catalog import EARTH
from theseus.server.app import app
from theseus.simulation.multi_object import (
    SpacecraftDefinition,
    MultiObjectEnvironment,
    MultiConjunctionEvent,
    PhysicalCollisionEvent,
    MultiObjectSimulationResult,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_spacecraft_definition_initial_state():
    """Verify state vector generation from Keplerian orbital elements."""
    sc = SpacecraftDefinition(
        id="SC-01",
        name="Test-Sat",
        semi_major_axis_km=6778.137,
        eccentricity=0.0,
        inclination_deg=51.6,
        raan_deg=0.0,
        arg_periapsis_deg=0.0,
        true_anomaly_deg=0.0,
        hard_body_radius_m=6.0,
    )
    r, v = sc.get_initial_state(EARTH.mu)
    
    assert len(r) == 3
    assert len(v) == 3
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    
    # 400 km LEO altitude: r ≈ 6778137 m, v ≈ 7672 m/s
    assert math.isclose(r_mag, 6778137.0, rel_tol=1e-4)
    v_expected = math.sqrt(EARTH.mu / 6778137.0)
    assert math.isclose(v_mag, v_expected, rel_tol=1e-4)


def test_multi_object_environment_propagation():
    """Verify simultaneous physical propagation of 3 spacecraft."""
    env = MultiObjectEnvironment(
        central_body="Earth",
        screening_threshold_km=50.0,
        coarse_dt_s=30.0,
        enable_j2=True,
        enable_drag=False,
    )

    sc1 = SpacecraftDefinition(
        id="SC-01",
        name="Sat-A",
        semi_major_axis_km=6778.137,
        inclination_deg=51.6,
        true_anomaly_deg=0.0,
    )
    sc2 = SpacecraftDefinition(
        id="SC-02",
        name="Sat-B",
        semi_major_axis_km=7000.0,
        inclination_deg=28.5,
        true_anomaly_deg=45.0,
    )
    sc3 = SpacecraftDefinition(
        id="SC-03",
        name="Sat-C",
        semi_major_axis_km=7200.0,
        inclination_deg=98.0,
        true_anomaly_deg=90.0,
    )

    res = env.simulate([sc1, sc2, sc3], t_start=0.0, t_end=1800.0, output_dt=30.0)

    assert isinstance(res, MultiObjectSimulationResult)
    assert len(res.objects) == 3
    assert res.summary["total_spacecraft"] == 3
    assert res.summary["total_debris"] == 0
    assert len(res.calculation_steps) >= 2

    # Check each object has populated state history
    for obj in res.objects:
        assert len(obj.state_history) > 10
        assert not obj.destroyed
        # Positions are within valid Earth orbital distances
        for st in obj.state_history:
            r_norm = np.linalg.norm(st["position"])
            assert 6.5e6 <= r_norm <= 8.0e6


def test_physical_collision_and_4_debris_propagation():
    """
    Test authoritative physical collision condition (miss <= HBR_comb)
    and physical numerical propagation of exactly 4 debris fragments.
    """
    env = MultiObjectEnvironment(
        central_body="Earth",
        screening_threshold_km=50.0,
        coarse_dt_s=10.0,
        enable_j2=False,
        enable_drag=False,
    )

    # 2 Spacecraft on intersecting orbits configured for direct impact
    sc1 = SpacecraftDefinition(
        id="SC-01",
        name="Interceptor-A",
        color="#ff9900",
        semi_major_axis_km=6778.137,
        inclination_deg=51.6,
        true_anomaly_deg=0.0,
        hard_body_radius_m=10.0,
    )
    sc2 = SpacecraftDefinition(
        id="SC-02",
        name="Interceptor-B",
        color="#3388ff",
        semi_major_axis_km=6778.137,
        inclination_deg=-51.6,
        true_anomaly_deg=0.0001, # Intersects within 5 meters
        hard_body_radius_m=10.0,
    )

    res = env.simulate([sc1, sc2], t_start=0.0, t_end=3600.0, output_dt=10.0)

    # Must detect conjunction and physical collision
    assert len(res.conjunctions) >= 1
    assert len(res.collisions) >= 1

    coll = res.collisions[0]
    assert coll.miss_distance_m <= coll.combined_hbr_m
    assert len(coll.debris_ids) == 4

    # Both parent spacecraft must be marked as destroyed
    sc1_track = next(o for o in res.objects if o.definition.id == "SC-01")
    sc2_track = next(o for o in res.objects if o.definition.id == "SC-02")
    assert sc1_track.destroyed
    assert sc2_track.destroyed

    # Must generate exactly 4 distinct physical debris tracks
    debris_tracks = [o for o in res.objects if o.definition.is_debris]
    assert len(debris_tracks) == 4

    debris_types = {d.definition.debris_type for d in debris_tracks}
    assert debris_types == {"solar_panel", "truss", "nozzle", "body"}

    # Debris must have valid physical orbits propagated under gravity
    for d in debris_tracks:
        active_states = [s for s in d.state_history if s["active"]]
        assert len(active_states) > 0
        for s in active_states:
            r_norm = np.linalg.norm(s["position"])
            assert 0.0 < r_norm <= 8.0e6



            v_norm = np.linalg.norm(s["velocity"])
            assert 3.0e3 <= v_norm <= 12.0e3



def test_api_simulate_environment(client):
    """Test POST /api/simulate/environment endpoint."""
    payload = {
        "spacecraft": [
            {
                "id": "SC-01",
                "name": "Alpha-Sat",
                "vehicle_type": "falcon9",
                "color": "#ff9900",
                "semi_major_axis_km": 6778.137,
                "inclination_deg": 51.6,
                "true_anomaly_deg": 0.0,
                "hard_body_radius_m": 5.0,
            },
            {
                "id": "SC-02",
                "name": "Beta-Sat",
                "vehicle_type": "isro-pslv",
                "color": "#3388ff",
                "semi_major_axis_km": 6878.137,
                "inclination_deg": 28.5,
                "true_anomaly_deg": 30.0,
                "hard_body_radius_m": 5.0,
            },
        ],
        "central_body": "Earth",
        "duration_hours": 0.5,
        "dt_sec": 30.0,
        "screening_threshold_km": 100.0,
    }

    res = client.post("/api/simulate/environment", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert "objects" in data
    assert "conjunctions" in data
    assert "collisions" in data
    assert "summary" in data
    assert data["summary"]["total_spacecraft"] == 2
    assert len(data["objects"]) >= 2


def test_api_simulate_multi_alias(client):
    """Test POST /api/simulate/multi alias endpoint."""
    payload = {
        "spacecraft": [
            {
                "id": "SC-01",
                "name": "Sat-1",
                "semi_major_axis_km": 6778.137,
                "inclination_deg": 0.0,
                "true_anomaly_deg": 0.0,
            }
        ],
        "duration_hours": 0.25,
        "dt_sec": 30.0,
    }
    res = client.post("/api/simulate/multi", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data["objects"]) == 1
