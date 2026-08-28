"""
Unit tests for THESEUS FastAPI server endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from theseus.server.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"
    assert data["subsystems"]["orbital_mechanics"] == "VALIDATED"
    assert data["subsystems"]["lambert_solver"] == "VALIDATED (Universal Variables)"
    assert "VALIDATED" in data["subsystems"]["phase_8_reentry"]
    assert "VALIDATED" in data["subsystems"]["phase_9_collision"]
    assert "VALIDATED" in data["subsystems"]["phase_10_uncertainty"]


def test_bodies_endpoint():
    response = client.get("/api/bodies")
    assert response.status_code == 200
    data = response.json()
    assert "earth" in data["bodies"]
    assert "mars" in data["bodies"]
    assert "sun" in data["bodies"]
    assert data["bodies"]["earth"]["radius_km"] == pytest.approx(6378.137, abs=1e-2)


def test_presets_endpoint():
    response = client.get("/api/presets")
    assert response.status_code == 200
    data = response.json()
    presets = data["presets"]
    assert len(presets) >= 8
    pslv = next(p for p in presets if p["id"] == "isro-pslv-xl")
    assert pslv["operator"] == "ISRO"
    saturn5 = next(p for p in presets if p["id"] == "nasa-saturn-v")
    assert saturn5["operator"] == "NASA"


def test_hohmann_simulation():
    payload = {
        "r1_km": 6678.137,
        "r2_km": 42164.0,
        "origin_body": "Earth",
        "plane_change_deg": 28.5,
        "dry_mass_kg": 2000.0,
        "fuel_mass_kg": 6000.0,
        "specific_impulse_s": 450.0,
        "thrust_n": 1000.0,
    }
    response = client.post("/api/simulate/hohmann", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["status"] == "SUCCESS"
    assert len(data["state_history"]) > 50
    assert len(data["calculation_trace"]) >= 4
    assert len(data["events"]) >= 4
    assert data["delta_v_budget"]["total_delta_v"] == pytest.approx(4255.956, abs=1.0)


def test_lambert_simulation():
    payload = {
        "r1_km": [6678.137, 0.0, 0.0],
        "r2_km": [0.0, 42164.0, 0.0],
        "tof_hours": 5.27,
        "central_body": "Earth",
        "prograde": True,
        "dry_mass_kg": 2000.0,
        "fuel_mass_kg": 3000.0,
        "specific_impulse_s": 316.0,
        "thrust_n": 500.0,
    }
    response = client.post("/api/simulate/lambert", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["status"] == "SUCCESS"
    assert len(data["state_history"]) > 20
    assert data["diagnostics"]["endpoint_miss_distance_m"] < 1.0


def test_rendezvous_simulation():
    payload = {
        "chaser_alt_km": 400.0,
        "target_alt_km": 420.0,
        "target_lead_deg": 60.0,
        "tof_hours": 1.0,
        "central_body": "Earth",
        "dry_mass_kg": 1000.0,
        "fuel_mass_kg": 500.0,
        "specific_impulse_s": 300.0,
        "thrust_n": 200.0,
    }
    response = client.post("/api/simulate/rendezvous", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["status"] == "SUCCESS"
    assert len(data["chaser_state_history"]) > 20


def test_demo_missions():
    for demo_id in ["earth-moon", "leo-geo", "leo-rendezvous", "earth-mars"]:
        response = client.get(f"/api/demo/{demo_id}")
        assert response.status_code == 200
        data = response.json()
        assert len(data["state_history"]) > 0
        assert len(data["calculation_trace"]) > 0


def _risk_payload(**overrides):
    """
    Baseline Phase 10 risk request.

    Object B crosses A's plane (51.6 deg vs 55.0 deg) so the pair has a
    genuine node conjunction inside the two-hour window.  The relative speed
    at the node is a few hundred m/s, well within what a 30 s coarse
    screening step resolves for a 100 km threshold.

    Object B carries a small phase offset.  With both phases at exactly zero
    the two objects sit on their shared ascending node at t = 0, separated
    only by their 50 m radius difference, and r_rel is exactly perpendicular
    to v_rel there -- so r_rel . v_rel is a floating-point zero at the window
    edge and whether that stationary point gets bracketed depends on its last
    bit.  The offset moves the fixture off that degeneracy so the test
    exercises the interior conjunction it is meant to.
    """
    payload = {
        "object_a_alt_km": 400.0,
        "object_a_inc_deg": 51.6,
        "object_a_phase_deg": 0.0,
        "object_b_alt_km": 400.05,
        "object_b_inc_deg": 55.0,
        "object_b_phase_deg": 0.02,
        "central_body": "Earth",
        "analysis_duration_hours": 2.0,
        "screening_threshold_km": 100.0,
        "coarse_dt_s": 30.0,
        "cov_a": {
            "sigma_pos_km": [0.5, 0.5, 0.5],
            "sigma_vel_km_s": [0.0005, 0.0005, 0.0005],
        },
        "cov_b": {
            "sigma_pos_km": [0.5, 0.5, 0.5],
            "sigma_vel_km_s": [0.0005, 0.0005, 0.0005],
        },
        "hard_body_radius_m": 15.0,
    }
    payload.update(overrides)
    return payload


def test_conjunction_risk_simulation():
    """A real conjunction must produce a complete, fully populated analysis."""
    response = client.post("/api/simulate/conjunction/risk", json=_risk_payload())
    assert response.status_code == 200
    data = response.json()

    assert data["analysis_status"] == "COMPLETE"
    assert data["conjunction_found"] is True

    assert "collision_probability" in data
    assert "probability" in data["collision_probability"]
    assert 0.0 <= data["collision_probability"]["probability"] <= 1.0
    assert "risk_assessment" in data
    assert "level" in data["risk_assessment"]
    assert len(data["calculation_steps"]) >= 12
    assert "b_plane_uncertainty" in data

    # The summary must describe the real encounter, not a placeholder.
    summary = data["conjunction_summary"]
    assert isinstance(summary["tca_s"], float)
    assert not isinstance(summary["tca_s"], bool)
    assert 0.0 < summary["tca_s"] < 2.0 * 3600.0
    assert summary["miss_distance_km"] is not None
    assert summary["relative_velocity_km_s"] > 0.0


def test_conjunction_risk_reports_indeterminate_when_no_tca_found():
    """
    Two objects on nearly identical co-planar orbits never reach a closest
    approach inside the window.  The API must say so explicitly rather than
    evaluating an arbitrary point and returning a risk level.

    This is the scenario that previously produced a fabricated Pc and an
    actionable risk classification.
    """
    payload = _risk_payload(
        object_b_alt_km=400.01,
        object_b_inc_deg=51.6,
        object_b_phase_deg=0.01,
    )
    response = client.post("/api/simulate/conjunction/risk", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["analysis_status"] == "INDETERMINATE_NO_CONJUNCTION"
    assert data["conjunction_found"] is False
    assert data["indeterminate_reason"]

    assert data["collision_probability"]["probability"] is None
    assert data["collision_probability"]["method"] == "NOT_COMPUTED"

    assert data["risk_assessment"]["level"] == "INDETERMINATE"
    assert data["risk_assessment"]["action_required"] is False
    assert data["risk_assessment"]["probability"] is None

    summary = data["conjunction_summary"]
    assert summary["tca_s"] is None
    assert summary["miss_distance_km"] is None
    assert summary["relative_velocity_km_s"] is None
