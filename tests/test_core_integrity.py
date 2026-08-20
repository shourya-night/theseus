"""Core integrity checks for authoritative API data flow."""

from theseus.server.app import LambertRequest, get_health, simulate_lambert


def test_health_executes_core_smoke_checks():
    health = get_health()
    assert health["status"] == "ONLINE"
    assert health["subsystems"]["core_engine"] == "ONLINE"
    assert health["subsystems"]["propagator"] == "ONLINE"
    assert health["subsystems"]["transfers"] == "ONLINE"
    assert health["subsystems"]["rendezvous"] == "ONLINE"


def test_interplanetary_lambert_returns_authoritative_bodies_and_trajectory():
    result = simulate_lambert(
        LambertRequest(
            central_body="sun",
            origin_body="Earth",
            destination_body="Mars",
            tof_hours=6240.0,
        )
    )

    assert result["metadata"]["status"] == "SUCCESS"
    assert result["diagnostics"]["endpoint_miss_distance_m"] < 1.0
    assert len(result["state_history"]) > 20

    bodies = {body["name"]: body for body in result["bodies"]}
    assert {"Sun", "Earth", "Mars"}.issubset(bodies)
    assert len(bodies["Earth"]["state_history"]) == len(result["state_history"])
    assert len(bodies["Mars"]["state_history"]) == len(result["state_history"])
    assert bodies["Earth"]["state_history"][0]["position"] != bodies["Earth"]["state_history"][-1]["position"]
    assert bodies["Mars"]["state_history"][0]["position"] != bodies["Mars"]["state_history"][-1]["position"]

    trajectory = result["trajectories"][0]
    assert trajectory["source"].startswith("RKF45 propagation")
    assert trajectory["state_history"] == result["state_history"]
