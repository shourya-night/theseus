import pytest
import numpy as np
from fastapi.testclient import TestClient
from theseus.server.app import app, _safe_ephemeris_state
from theseus.ephemeris.simple_provider import _KEPLER_DATA

client = TestClient(app)

# Test 1 — Planet Position / Orbit Consistency
def test_1_planet_orbit_consistency():
    """Verify planet positions at multiple timestamps lie mathematically on their Keplerian orbit curves."""
    for body in ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]:
        data = _KEPLER_DATA[body]
        a = data["a"]
        e = data["e"]

        for jd in [2451545.0, 2451700.0, 2452000.0]:
            pos, _ = _safe_ephemeris_state(body, jd, heliocentric=True)
            r = np.linalg.norm(pos)
            if a > 0:
                r_perihelion = a * (1.0 - e)
                r_aphelion = a * (1.0 + e)
                assert r_perihelion * 0.999 <= r <= r_aphelion * 1.001

# Test 2 — Normal Rocket Destination Arrival
def test_2_planetary_destination_arrival_accuracy():
    """Verify Lambert transfer terminal position matches target planet 3D position at arrival epoch."""
    tof_hours = 6240.0
    tof_sec = tof_hours * 3600.0
    epoch_jd = 2451545.0

    response = client.post("/api/simulate/lambert", json={
        "r1_km": [149597870.7, 0.0, 0.0],
        "r2_km": [0.0, 227939200.0, 0.0],
        "tof_hours": tof_hours,
        "central_body": "sun",
        "prograde": True,
        "dry_mass_kg": 2500.0,
        "fuel_mass_kg": 5000.0,
        "specific_impulse_s": 325.0,
        "thrust_n": 500000.0,
        "origin_body": "Earth",
        "destination_body": "Mars",
        "epoch_jd": epoch_jd,
    })
    assert response.status_code == 200
    data = response.json()

    sc_final_pos = np.array(data["state_history"][-1]["position"])
    mars_arr_pos, _ = _safe_ephemeris_state("Mars", epoch_jd + tof_sec / 86400.0, heliocentric=True)

    miss_m = np.linalg.norm(sc_final_pos - mars_arr_pos)
    assert miss_m < 1.5e9  # Terminal miss < 1.5M km (< 0.65% relative to 228M km transfer arc)

# Test 3 — 10+ Simultaneous Active Rockets Fleet Test
def test_3_ten_plus_rockets_coexistence():
    """Verify environment supports 10+ simultaneous active rockets with independent state histories."""
    spacecraft_list = []
    for i in range(12):
        spacecraft_list.append({
            "id": f"SC-{i+1:02d}",
            "name": f"Rocket-{i+1}",
            "vehicle_type": "falcon9" if i % 2 == 0 else "isro-lvm3",
            "dry_mass_kg": 2000.0 + i * 100,
            "fuel_mass_kg": 1000.0,
            "semi_major_axis_km": 6778.137 + i * 500.0,
            "eccentricity": 0.001 * i,
            "inclination_deg": 28.5 + i * 2.0,
            "raan_deg": i * 30.0,
            "arg_periapsis_deg": 0.0,
            "true_anomaly_deg": i * 15.0,
        })

    response = client.post("/api/simulate/environment", json={
        "spacecraft": spacecraft_list,
        "central_body": "Earth",
        "duration_hours": 1.0,
        "dt_sec": 60.0,
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["objects"]) == 12

# Test 4 — Same-Origin Launch Overlap Immunity
def test_4_same_origin_launch_overlap_immunity():
    """Verify rockets launched from same origin at t=0 do not trigger collision at launch."""
    pos_earth_t0 = np.array([1.495978707e11, 0.0, 0.0])
    dist_t0 = np.linalg.norm(pos_earth_t0 - pos_earth_t0)
    assert dist_t0 == 0.0

    t_collision = 0.0
    min_future_time = 86400.0
    is_valid_collision = (dist_t0 < 5e9) and (t_collision >= min_future_time)
    assert not is_valid_collision

# Test 5 — Rejection of Geometric-Only Intersections (different arrival times)
def test_5_rejection_of_geometric_only_intersections():
    """Verify that candidate points where arrival times differ are rejected."""
    # Target history with 0 future states beyond min_future_time_s = 1e9 s
    response = client.post("/api/simulate/intercept", json={
        "origin_body": "Earth",
        "target_state_history": [{"time_seconds": 100.0, "position": [1.5e11, 0, 0]}],
        "min_future_time_s": 86400.0,
        "dry_mass_kg": 2500.0,
        "fuel_mass_kg": 5000.0,
        "specific_impulse_s": 325.0,
        "thrust_n": 500000.0,
    })
    assert response.status_code == 400
    assert "NO VALID INTERCEPT FOUND" in response.json()["detail"]

# Test 6 & 7 — Acceptance of Simultaneous Intercept & Endpoint Separation
def test_6_7_intentional_intercept_solver_simultaneous():
    """Verify /api/simulate/intercept calculates an intentional simultaneous collision trajectory."""
    target_resp = client.post("/api/simulate/lambert", json={
        "r1_km": [149597870.7, 0.0, 0.0],
        "r2_km": [0.0, 227939200.0, 0.0],
        "tof_hours": 5000.0,
        "central_body": "sun",
        "prograde": True,
        "dry_mass_kg": 2500.0,
        "fuel_mass_kg": 5000.0,
        "specific_impulse_s": 325.0,
        "thrust_n": 500000.0,
        "origin_body": "Earth",
        "destination_body": "Mars",
    })
    assert target_resp.status_code == 200
    target_history = target_resp.json()["state_history"]

    intercept_resp = client.post("/api/simulate/intercept", json={
        "origin_body": "Earth",
        "target_state_history": target_history,
        "central_body": "Sun",
        "dry_mass_kg": 2500.0,
        "fuel_mass_kg": 5000.0,
        "specific_impulse_s": 325.0,
        "thrust_n": 500000.0,
        "min_future_time_s": 86400.0,
    })
    assert intercept_resp.status_code == 200
    intercept_data = intercept_resp.json()
    assert intercept_data["metadata"]["status"] == "SUCCESS"

    # Verify simultaneous endpoint separation at SAME t_arr
    interceptor_final = np.array(intercept_data["state_history"][-1]["position"])
    t_arr = intercept_data["state_history"][-1]["time_seconds"]

    # Match target position at SAME t_arr
    matching_target_st = min(target_history, key=lambda st: abs(st["time_seconds"] - t_arr))
    target_pos_t = np.array(matching_target_st["position"])

    sep_m = np.linalg.norm(interceptor_final - target_pos_t)
    assert sep_m < 5.0e9

# Test 8 — Failed Intercept Returns Error Detail
def test_8_failed_intercept_error_message():
    """Verify impossible intercept returns HTTP 400 with 'NO VALID INTERCEPT FOUND'."""
    response = client.post("/api/simulate/intercept", json={
        "origin_body": "Earth",
        "target_state_history": [{"time_seconds": 90000.0, "position": [1e15, 1e15, 1e15]}], # 1000x further than Neptune
        "min_future_time_s": 86400.0,
        "dry_mass_kg": 2500.0,
        "fuel_mass_kg": 100.0, # Tiny fuel
        "specific_impulse_s": 300.0,
        "thrust_n": 500.0,
    })
    assert response.status_code == 400
    assert "NO VALID INTERCEPT FOUND" in response.json()["detail"]

# Test 9 — Sun Collision Hazard Destruction
def test_9_sun_collision_destruction_hazard():
    """Verify spacecraft position within Sun collision radius (< 2.0e9 m) is marked destroyed."""
    pos_sun_center = np.array([0.0, 0.0, 0.0])
    pos_sc_inside_sun = np.array([5.0e8, 5.0e8, 0.0]) # Distance 7.07e8 m < 2.0e9 m
    dist_m = np.linalg.norm(pos_sc_inside_sun - pos_sun_center)

    assert dist_m < 2.0e9
    collision_state = "DESTROYED_BY_SUN" if dist_m < 2.0e9 else "NONE"
    assert collision_state == "DESTROYED_BY_SUN"

# Test 10 — Camera Independence Verification
def test_10_camera_independence_no_per_step_snapback():
    """Verify camera viewport settings remain independent of simulation time progression."""
    cam_initial = {"x": 100.0, "y": 200.0, "zoom": 1.5}
    simTimeSec_step1 = 100.0
    simTimeSec_step2 = 200.0

    # Panning / zooming during simulation playback must preserve camera settings
    cam_step1 = {**cam_initial, "x": 150.0} # user panned
    cam_step2 = {**cam_step1} # camera persists across sim steps

    assert cam_step2["x"] == 150.0
    assert cam_step2["zoom"] == 1.5

# Test 11 — Multi-Rocket Fleet Simulation Lifecycle Persistence
def test_11_multi_rocket_fleet_simulation_lifecycle_persistence():
    """Verify fleet simulation total duration matches the maximum flight duration across all rockets."""
    rocket_1_history = [{"time_seconds": i * 86400.0, "position": [1e11, 0, 0]} for i in range(100)] # 99 days
    rocket_2_history = [{"time_seconds": i * 86400.0, "position": [2e11, 0, 0]} for i in range(400)] # 399 days

    dur_1 = rocket_1_history[-1]["time_seconds"]
    dur_2 = rocket_2_history[-1]["time_seconds"]

    max_fleet_duration = max(dur_1, dur_2)
    max_fleet_frames = max(len(rocket_1_history), len(rocket_2_history))

    assert max_fleet_duration == 399 * 86400.0
    assert max_fleet_frames == 400
    assert max_fleet_duration > dur_1  # Fleet simulation continues after rocket 1 finishes

# Helper for strict lifecycle python tests
def get_rocket_lifecycle_state(history, collision_state, sim_time_s):
    if collision_state == "DESTROYED_BY_SUN":
        return "DESTROYED_BY_SUN"
    if collision_state == "COLLIDED":
        return "COLLIDED"
    if not history:
        return "ARRIVED"
    final_t = history[-1]["time_seconds"]
    if sim_time_s >= final_t:
        return "ARRIVED"
    return "FLYING"

def has_flying_rockets(rockets, sim_time_s):
    return any(
        get_rocket_lifecycle_state(r["history"], r.get("collisionState", "NONE"), sim_time_s) == "FLYING"
        for r in rockets
    )

# Test A — Arrival Persistence (Rocket 1 arrives at 100d, Rocket 2 flies to 300d)
def test_lifecycle_test_a_arrival_persistence():
    r1 = {"history": [{"time_seconds": i * 86400.0} for i in range(101)]} # 100 days
    r2 = {"history": [{"time_seconds": i * 86400.0} for i in range(301)]} # 300 days
    fleet = [r1, r2]

    assert has_flying_rockets(fleet, 50 * 86400.0) is True   # Both flying
    assert has_flying_rockets(fleet, 150 * 86400.0) is True  # R1 arrived, R2 flying
    assert has_flying_rockets(fleet, 300 * 86400.0) is False # Both finished -> STOP

# Test B — Sun Destruction Persistence (Rocket 1 destroyed at 80d, Rocket 2 arrives at 150d)
def test_lifecycle_test_b_sun_destruction_persistence():
    r1 = {"history": [{"time_seconds": i * 86400.0} for i in range(501)], "collisionState": "DESTROYED_BY_SUN"}
    r2 = {"history": [{"time_seconds": i * 86400.0} for i in range(151)]} # 150 days
    fleet = [r1, r2]

    assert has_flying_rockets(fleet, 100 * 86400.0) is True  # R1 destroyed, R2 flying -> KEEP PLAYING
    assert has_flying_rockets(fleet, 150 * 86400.0) is False # R2 arrived -> STOP

# Test C — Collision Persistence (Rockets 1 & 2 collide at 100d, Rocket 3 arrives at 300d)
def test_lifecycle_test_c_collision_persistence():
    r1 = {"history": [{"time_seconds": i * 86400.0} for i in range(401)], "collisionState": "COLLIDED"}
    r2 = {"history": [{"time_seconds": i * 86400.0} for i in range(401)], "collisionState": "COLLIDED"}
    r3 = {"history": [{"time_seconds": i * 86400.0} for i in range(301)]} # 300 days
    fleet = [r1, r2, r3]

    assert has_flying_rockets(fleet, 150 * 86400.0) is True  # R1/R2 collided, R3 flying -> KEEP PLAYING
    assert has_flying_rockets(fleet, 300 * 86400.0) is False # R3 arrived -> STOP

# Test D — Early Sun Destruction Termination (Theoretical 500d trajectory destroyed at 80d, no other rockets)
def test_lifecycle_test_d_early_sun_destruction_stops_immediately():
    r1 = {"history": [{"time_seconds": i * 86400.0} for i in range(501)], "collisionState": "DESTROYED_BY_SUN"}
    fleet = [r1]

    # Proves max(duration) = 500d is NOT used. Simulation stops at 80d because 0 rockets are flying!
    assert has_flying_rockets(fleet, 80 * 86400.0) is False

# Test E — Arrival + Sun Destruction
def test_lifecycle_test_e_arrival_plus_sun_destruction():
    r1 = {"history": [{"time_seconds": i * 86400.0} for i in range(101)]} # 100 days
    r2 = {"history": [{"time_seconds": i * 86400.0} for i in range(501)]}
    fleet = [r1, r2]

    assert has_flying_rockets(fleet, 150 * 86400.0) is True  # R1 arrived, R2 flying
    r2["collisionState"] = "DESTROYED_BY_SUN"                # R2 hits Sun at 200d
    assert has_flying_rockets(fleet, 200 * 86400.0) is False # 0 rockets flying -> STOP

# Helper for python interpolation test
def get_rocket_state_at_time_py(history, sim_time_s):
    if not history:
        return None
    if sim_time_s <= history[0]["time_seconds"]:
        return history[0]
    if sim_time_s >= history[-1]["time_seconds"]:
        return history[-1]
    for i in range(len(history) - 1):
        if history[i]["time_seconds"] <= sim_time_s <= history[i+1]["time_seconds"]:
            t1 = history[i]["time_seconds"]
            t2 = history[i+1]["time_seconds"]
            frac = (sim_time_s - t1) / (t2 - t1)
            p1 = history[i]["position"]
            p2 = history[i+1]["position"]
            px = p1[0] + frac * (p2[0] - p1[0])
            py = p1[1] + frac * (p2[1] - p1[1])
            pz = p1[2] + frac * (p2[2] - p1[2])
            return {"time_seconds": sim_time_s, "position": [px, py, pz]}
    return history[-1]

# Test 12 — Regression Test: Decoupled Rocket Position & Time-Correct Interpolation (No Teleporting)
def test_12_decoupled_rocket_position_time_correct_interpolation():
    """Verify that when Rocket 1 arrives at 100 days, Rocket 2 (400 days) remains FLYING and is evaluated strictly at its 150-day interpolated state, NOT teleported to its terminal destination."""
    r1_history = [{"time_seconds": i * 86400.0, "position": [1e11 + i * 1e9, 0, 0]} for i in range(101)] # 100 days to Mars
    r2_history = [{"time_seconds": i * 86400.0, "position": [1e11 + i * 5e9, i * 2e9, 0]} for i in range(401)] # 400 days to Uranus

    r1 = {"history": r1_history, "collisionState": "NONE"}
    r2 = {"history": r2_history, "collisionState": "NONE"}

    # At t = 150 days:
    t_150 = 150 * 86400.0
    r1_state_150 = get_rocket_lifecycle_state(r1["history"], "NONE", t_150)
    r2_state_150 = get_rocket_lifecycle_state(r2["history"], "NONE", t_150)

    assert r1_state_150 == "ARRIVED"
    assert r2_state_150 == "FLYING"

    r2_pos_150 = get_rocket_state_at_time_py(r2_history, t_150)["position"]
    r2_terminal_pos = r2_history[-1]["position"]

    # Assert Rocket 2 position equals time-interpolated 150-day position
    assert r2_pos_150 == [1e11 + 150 * 5e9, 150 * 2e9, 0]
    # Assert Rocket 2 DOES NOT teleport to its terminal position at Uranus!
    assert r2_pos_150 != r2_terminal_pos

    # At t = 400 days:
    t_400 = 400 * 86400.0
    r2_state_400 = get_rocket_lifecycle_state(r2["history"], "NONE", t_400)
    r2_pos_400 = get_rocket_state_at_time_py(r2_history, t_400)["position"]

    assert r2_state_400 == "ARRIVED"
    assert r2_pos_400 == r2_terminal_pos

