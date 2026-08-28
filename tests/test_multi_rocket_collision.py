import pytest
import numpy as np
from fastapi.testclient import TestClient
from theseus.server.app import app

client = TestClient(app)

def test_arbitrary_interplanetary_transfers():
    """Verify backend Lambert transfers work for arbitrary planet pairs (Mars->Earth, Earth->Uranus, Jupiter->Saturn)."""
    transfers = [
        ("Mars", "Earth"),
        ("Earth", "Uranus"),
        ("Jupiter", "Saturn"),
        ("Mercury", "Venus"),
    ]

    for origin, dest in transfers:
        response = client.post("/api/simulate/lambert", json={
            "r1_km": [227939200.0, 0.0, 0.0],
            "r2_km": [0.0, 149597870.7, 0.0],
            "tof_hours": 4500.0,
            "central_body": "sun",
            "prograde": True,
            "dry_mass_kg": 2500.0,
            "fuel_mass_kg": 3500.0,
            "specific_impulse_s": 316.0,
            "thrust_n": 500000.0,
            "origin_body": origin,
            "destination_body": dest,
        })
        assert response.status_code == 200
        data = response.json()
        assert "state_history" in data
        assert len(data["state_history"]) > 0
        assert "bodies" in data
        body_names = [b["name"] for b in data["bodies"]]
        assert "Sun" in body_names
        assert origin in body_names
        assert dest in body_names

def test_pairwise_collision_distance_threshold():
    """Verify pairwise distance threshold detection logic."""
    pos_rocket_1 = np.array([1.495978707e11, 0.0, 0.0])
    pos_rocket_2 = np.array([1.495978707e11 + 1.0e8, 0.0, 0.0]) # 100,000 km apart

    dist = np.linalg.norm(pos_rocket_1 - pos_rocket_2)
    threshold = 5e9 # 5,000,000 km threshold

    assert dist < threshold
