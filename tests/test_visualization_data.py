"""
Unit tests for THESEUS visualization data pipeline and physical co-location assertions.
Verifies Phase 14 visualization data criteria for Earth -> Mars transfer.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from theseus.server.app import app

client = TestClient(app)


def test_earth_mars_visualization_data():
    """Verify backend returns Sun, Earth, Mars body histories with exact co-location."""
    response = client.get("/api/demo/earth-mars")
    assert response.status_code == 200
    data = response.json()

    # 1. State history & body histories exist
    assert "state_history" in data
    assert len(data["state_history"]) > 20
    assert "bodies" in data
    
    body_dict = {b["id"]: b for b in data["bodies"]}
    assert "sun" in body_dict
    assert "earth" in body_dict
    assert "mars" in body_dict

    earth_hist = body_dict["earth"]["state_history"]
    mars_hist = body_dict["mars"]["state_history"]
    sc_hist = data["state_history"]

    # 2. Compatible timestamps
    assert len(earth_hist) == len(sc_hist)
    assert len(mars_hist) == len(sc_hist)
    assert earth_hist[0]["time_seconds"] == sc_hist[0]["time_seconds"]
    assert earth_hist[-1]["time_seconds"] == sc_hist[-1]["time_seconds"]

    # 3. Spacecraft initial state is physically colocated with Earth (t0)
    sc_pos_0 = np.array(sc_hist[0]["position"])
    earth_pos_0 = np.array(earth_hist[0]["position"])
    dep_distance_m = float(np.linalg.norm(sc_pos_0 - earth_pos_0))
    assert dep_distance_m < 1.0, f"Spacecraft departure distance from Earth is {dep_distance_m} m (expected < 1.0 m)"

    # 4. Spacecraft arrival approaches Mars (t_final)
    sc_pos_f = np.array(sc_hist[-1]["position"])
    mars_pos_f = np.array(mars_hist[-1]["position"])
    arr_distance_m = float(np.linalg.norm(sc_pos_f - mars_pos_f))
    assert arr_distance_m < 1e9, f"Spacecraft arrival distance from Mars is {arr_distance_m} m"

    # 5. Planetary motion evolves over time
    earth_pos_f = np.array(earth_hist[-1]["position"])
    earth_displacement = float(np.linalg.norm(earth_pos_f - earth_pos_0))
    assert earth_displacement > 1e10, "Earth should move significantly over ~250 day transfer"

    mars_pos_0 = np.array(mars_hist[0]["position"])
    mars_displacement = float(np.linalg.norm(mars_pos_f - mars_pos_0))
    assert mars_displacement > 1e10, "Mars should move significantly over ~250 day transfer"
