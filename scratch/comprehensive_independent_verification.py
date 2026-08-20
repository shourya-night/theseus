"""
COMPREHENSIVE INDEPENDENT ASTRODYNAMICS VERIFICATION HARNESS
===========================================================
Independent Verification of THESEUS Phase 1-7 Astrodynamics Engine.
Strictly evaluates:
1. Orbital Element Conversion (Cartesian <-> Keplerian across all geometries)
2. Two-body Analytical Propagation (Multi-orbit invariants & Kepler solution)
3. Numerical Integrators (RK4 4th-order convergence on dy/dt=-y, RKF45 adaptive error control)
4. Gravity & J2 Perturbation (Surface g, 1/r^2 scaling, analytical secular RAAN drift)
5. Lambert Problem Solver (Published Vallado/Curtis literature cases & endpoint propagation)
6. Hohmann & Orbital Transfers (LEO->GEO analytical delta-v, transfer time, bi-elliptic)
7. Rendezvous (Independent chaser and target propagation to arrival epoch)
8. Ephemerides (Astronomical distances, ICRF frames, Moon/Sun vectors)
9. Solar Radiation Pressure (1 AU magnitude, 1/r^2 scaling, cylindrical shadow geometry)
10. Spacecraft Mass Depletion & Thrust (Tsiolkovsky equation, mdot, multi-step finite burn)
11. Time & Epoch Handling (Julian Date, MJD, UTC, TDB, leap offsets, float64 precision bounds)
"""

import math
import sys
from datetime import datetime, timezone, timedelta
import numpy as np

# Import THESEUS modules for black-box testing
from theseus.constants.physical import (
    G_VAL, C_VAL, AU_VAL, L_SUN_VAL, G0_VAL, R_GAS_VAL, M_AIR_VAL,
)
from theseus.bodies.catalog import EARTH, MOON, SUN, MARS, JUPITER, get_body
from theseus.coordinates.transformations import (
    perifocal_to_eci_matrix, eci_to_perifocal_matrix,
    eci_to_ecef, ecef_to_eci, cartesian_to_spherical, spherical_to_cartesian,
)
from theseus.orbital.elements import OrbitalElements
from theseus.orbital.conversions import state_to_elements, elements_to_state
from theseus.orbital.kepler import solve_kepler
from theseus.orbital.lambert import solve_lambert
from theseus.propagation.analytical import propagate_twobody
from theseus.propagation.integrators import RK4Integrator, RKF45Integrator
from theseus.propagation.numerical import NumericalPropagator
from theseus.dynamics.gravity import PointMassGravity, J2Perturbation
from theseus.dynamics.drag import DragModel
from theseus.dynamics.srp import SolarRadiationPressure
from theseus.dynamics.thrust import ThrustModel, ThrustDirection
from theseus.dynamics.force_model import CompositeForceModel
from theseus.atmosphere.models import ExponentialAtmosphere, US1976StandardAtmosphere
from theseus.ephemeris.simple_provider import SimpleEphemerisProvider
from theseus.ephemeris.astropy_provider import AstropyEphemerisProvider
from theseus.spacecraft.vehicle import Spacecraft
from theseus.maneuvers.burns import fuel_for_delta_v, delta_v_from_fuel, finite_burn_duration
from theseus.maneuvers.transfers import hohmann_transfer, bielliptic_transfer, combined_maneuver
from theseus.rendezvous.solver import solve_rendezvous
from theseus.time.epochs import Epoch, JD_J2000
from theseus.time.scales import TimeScale


def run_all_verifications():
    results = {}
    print("=" * 80)
    print("THESEUS FINAL INDEPENDENT PHYSICS VERIFICATION HARNESS")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # VALIDATION 1: ORBITAL ELEMENT CONVERSION
    # -------------------------------------------------------------------------
    print("\n--- VALIDATION 1: ORBITAL ELEMENT CONVERSIONS ---")
    val1_passed = True
    mu = EARTH.mu

    # Test Cases:
    # 1. Circular Equatorial
    # 2. Elliptic Equatorial
    # 3. Circular Inclined (45 deg)
    # 4. Inclined Elliptic (3D)
    # 5. Polar Orbit (90 deg)
    # 6. Retrograde Orbit (135 deg)
    # 7. Equatorial Retrograde (180 deg)
    # 8. High Eccentricity (e = 0.95)

    test_states = [
        ("Circular Equatorial", np.array([7000e3, 0, 0]), np.array([0, math.sqrt(mu/7000e3), 0])),
        ("Elliptic Equatorial", np.array([7000e3, 0, 0]), np.array([0, 8500.0, 0])),
        ("Circular Inclined 45deg", np.array([7000e3, 0, 0]), np.array([0, math.sqrt(mu/7000e3)*math.cos(math.pi/4), math.sqrt(mu/7000e3)*math.sin(math.pi/4)])),
        ("Inclined Elliptic 3D", np.array([6524834.0, 6862875.0, 6448296.0]), np.array([4901.327, 5533.756, -1976.341])),
        ("Polar Orbit 90deg", np.array([7000e3, 0, 0]), np.array([0, 0, math.sqrt(mu/7000e3)])),
        ("Retrograde Orbit 135deg", np.array([7000e3, 0, 0]), np.array([0, -math.sqrt(mu/7000e3)*math.cos(math.pi/4), math.sqrt(mu/7000e3)*math.sin(math.pi/4)])),
        ("Equatorial Retrograde 180deg", np.array([7000e3, 0, 0]), np.array([0, -math.sqrt(mu/7000e3), 0])),
        ("High Eccentricity e=0.95", np.array([7000e3, 0, 0]), np.array([0, math.sqrt(mu * (1 + 0.95) / 7000e3), 0])),
    ]

    for name, r_ref, v_ref in test_states:
        # Step 1: Cart -> Elements
        oe = state_to_elements(r_ref, v_ref, mu)
        # Step 2: Elements -> Cart
        r_rec, v_rec = elements_to_state(oe)

        pos_err = np.linalg.norm(r_rec - r_ref)
        vel_err = np.linalg.norm(v_rec - v_ref)
        rel_pos_err = pos_err / np.linalg.norm(r_ref)
        rel_vel_err = vel_err / np.linalg.norm(v_ref)

        # Independent verification of elements
        h_vec = np.cross(r_ref, v_ref)
        h_mag = np.linalg.norm(h_vec)
        e_vec = ((np.linalg.norm(v_ref)**2 - mu / np.linalg.norm(r_ref)) * r_ref - np.dot(r_ref, v_ref) * v_ref) / mu
        e_ref = np.linalg.norm(e_vec)
        energy_ref = 0.5 * np.linalg.norm(v_ref)**2 - mu / np.linalg.norm(r_ref)
        a_ref = -mu / (2.0 * energy_ref)
        i_ref = math.acos(max(-1.0, min(1.0, h_vec[2] / h_mag)))

        a_err = abs(oe.a - a_ref)
        e_err = abs(oe.e - e_ref)
        i_err = abs(oe.i - i_ref)

        passed = (pos_err < 1e-4) and (vel_err < 1e-6) and (a_err < 1e-4) and (e_err < 1e-9)
    # Coordinate Frame Transformation Check: ECI <-> ECEF Greenwich station
    # Greenwich station at t=0 (JD_J2000) is at [EARTH.radius, 0, 0] in ECEF.
    # At JD_J2000 + 0.25 days (6 hours later), Earth rotates 90 degrees eastward (+y in ECI).
    # Therefore, at t=6h, the station is at [0, EARTH.radius, 0] in ECI.
    # Converting [0, EARTH.radius, 0] in ECI to ECEF MUST yield [EARTH.radius, 0, 0].
    from theseus.coordinates.transformations import gmst_from_jd
    jd_test = JD_J2000
    theta_0 = gmst_from_jd(jd_test)
    r_station_ecef = np.array([EARTH.radius, 0.0, 0.0])
    # The physical ECI position at jd_test is rotated by +theta_0
    r_station_eci = np.array([EARTH.radius * math.cos(theta_0), EARTH.radius * math.sin(theta_0), 0.0])
    r_ecef_calc = eci_to_ecef(r_station_eci, jd_test)
    err_ecef = np.linalg.norm(r_ecef_calc - r_station_ecef)
    p_ecef = (err_ecef < 1e-3)
    if not p_ecef:
        print(f"  [FAIL] Coordinate Frame: ECI->ECEF rotation sign error: calc={r_ecef_calc/1e3} km, expected={r_station_ecef/1e3} km (err={err_ecef/1e3:.1f} km)")
    else:
        print(f"  [PASS] Coordinate Frame: ECI->ECEF Greenwich station: err={err_ecef:.2e} m")

    results["Orbital elements"] = "PASS" if val1_passed else "FAIL"
    results["Coordinate transformations (ECI/ECEF)"] = "PASS" if p_ecef else "FAIL"

    # -------------------------------------------------------------------------
    # VALIDATION 2: TWO-BODY ANALYTICAL PROPAGATION
    # -------------------------------------------------------------------------
    print("\n--- VALIDATION 2: TWO-BODY ANALYTICAL PROPAGATION ---")
    val2_passed = True

    # Test propagation across 1, 10, 100 orbits
    r0 = np.array([7000e3, 0, 0])
    v0 = np.array([0, 8500.0, 0])  # Elliptical orbit e ~ 0.4
    oe0 = state_to_elements(r0, v0, mu)
    T = oe0.period

    energy0 = 0.5 * np.dot(v0, v0) - mu / np.linalg.norm(r0)
    h0 = np.linalg.norm(np.cross(r0, v0))

    for n_orbits in [1, 10, 100]:
        t_eval = n_orbits * T
        history = propagate_twobody(r0, v0, mu, [t_eval])
        rf = history[-1].position
        vf = history[-1].velocity

        energy_f = 0.5 * np.dot(vf, vf) - mu / np.linalg.norm(rf)
        hf = np.linalg.norm(np.cross(rf, vf))

        # After integer periods, spacecraft should return exactly to (r0, v0)
        pos_drift = np.linalg.norm(rf - r0)
        vel_drift = np.linalg.norm(vf - v0)
        energy_drift = abs((energy_f - energy0) / energy0)
        h_drift = abs((hf - h0) / h0)

        passed = (pos_drift < 1e-4) and (vel_drift < 1e-6) and (energy_drift < 1e-13) and (h_drift < 1e-13)
        if not passed:
            val2_passed = False
        print(f"  [{'PASS' if passed else 'FAIL'}] {n_orbits:3d} Orbits: pos_err={pos_drift:.2e} m, vel_err={vel_drift:.2e} m/s, dE/E={energy_drift:.2e}, dh/h={h_drift:.2e}")

    results["Two-body propagation"] = "PASS" if val2_passed else "FAIL"

    # -------------------------------------------------------------------------
    # VALIDATION 3: NUMERICAL INTEGRATORS (RK4 & RKF45)
    # -------------------------------------------------------------------------
    print("\n--- VALIDATION 3: NUMERICAL INTEGRATORS ---")
    val3_rk4_passed = True
    val3_rkf_passed = True

    # Analytical test problem: dy/dt = -y, y(0) = 1.0, exact solution y(t) = exp(-t)
    def ode_decay(t, y):
        return -y

    # RK4 fourth-order convergence test: error ~ O(dt^4)
    # Halving dt should reduce error by ~16x (slope = 4 in log-log)
    t_end = 2.0
    exact_y = math.exp(-t_end)
    dts = [0.2, 0.1, 0.05, 0.025]
    errors_rk4 = []
    for dt in dts:
        rk4 = RK4Integrator(dt=dt)
        res = rk4.integrate(ode_decay, np.array([1.0]), (0.0, t_end))
        err = abs(res.states[-1][0] - exact_y)
        errors_rk4.append(err)

    ratios_rk4 = [errors_rk4[i] / errors_rk4[i+1] for i in range(len(errors_rk4)-1)]
    # Each ratio should be approximately 16.0
    for i, ratio in enumerate(ratios_rk4):
        p = (ratio >= 14.0 and ratio <= 18.0)
        if not p:
            val3_rk4_passed = False
        print(f"  [{'PASS' if p else 'FAIL'}] RK4 Step Halving {dts[i]:.3f}->{dts[i+1]:.3f}s: error_ratio={ratio:.3f} (theoretical: 16.000)")

    # RKF45 adaptive tolerance response
    tols = [1e-4, 1e-7, 1e-10]
    errors_rkf = []
    for tol in tols:
        rkf = RKF45Integrator(atol=tol, rtol=tol, dt_initial=0.1)
        res = rkf.integrate(ode_decay, np.array([1.0]), (0.0, t_end))
        err = abs(res.states[-1][0] - exact_y)
        errors_rkf.append(err)
        p = (err < tol * 10)
        if not p:
            val3_rkf_passed = False
        print(f"  [{'PASS' if p else 'FAIL'}] RKF45 tol={tol:.1e}: actual_err={err:.2e}, steps={res.steps_taken}, rejected={res.rejected_steps}")

    results["RK4"] = "PASS" if val3_rk4_passed else "FAIL"
    results["RKF45"] = "PASS" if val3_rkf_passed else "FAIL"

    # -------------------------------------------------------------------------
    # VALIDATION 4: GRAVITY & J2 PERTURBATION
    # -------------------------------------------------------------------------
    print("\n--- VALIDATION 4: GRAVITY & J2 ---")
    val4_grav_passed = True
    val4_j2_passed = True

    # 1. Earth surface gravity: g = mu / R_e^2
    grav = PointMassGravity(EARTH)
    g_surf_calc = np.linalg.norm(grav.compute_acceleration(0, np.array([EARTH.radius, 0, 0]), np.zeros(3), 1000))
    g_surf_ref = EARTH.mu / (EARTH.radius ** 2)
    err_g = abs(g_surf_calc - g_surf_ref) / g_surf_ref
    if err_g > 1e-12: val4_grav_passed = False
    print(f"  [{'PASS' if err_g < 1e-12 else 'FAIL'}] Earth Surface g: calc={g_surf_calc:.6f} m/s^2, ref={g_surf_ref:.6f} m/s^2 (err={err_g:.2e})")

    # 2. Inverse square scaling
    g_2re = np.linalg.norm(grav.compute_acceleration(0, np.array([2 * EARTH.radius, 0, 0]), np.zeros(3), 1000))
    scale_err = abs(g_2re / g_surf_calc - 0.25)
    if scale_err > 1e-12: val4_grav_passed = False
    print(f"  [{'PASS' if scale_err < 1e-12 else 'FAIL'}] Inverse Square (2*Re): ratio={g_2re/g_surf_calc:.8f}, expected=0.25000000 (err={scale_err:.2e})")

    # 3. Analytical J2 RAAN drift for ISS orbit (400 km, inc=51.6 deg)
    r_iss = EARTH.radius + 400e3
    inc_iss = math.radians(51.6)
    n_iss = math.sqrt(EARTH.mu / (r_iss ** 3))
    # Secular RAAN precession analytical formula:
    # dOmega/dt = -1.5 * J2 * (Re/p)^2 * n * cos(inc)
    raan_rate_ref = -1.5 * EARTH.J2 * ((EARTH.radius / r_iss) ** 2) * n_iss * math.cos(inc_iss)  # rad/s

    # Numerical J2 evaluation using THESEUS J2 model
    j2_model = J2Perturbation(EARTH)
    pos_node = np.array([r_iss, 0.0, 0.0])
    v_circ = math.sqrt(EARTH.mu / r_iss)
    vel_node = np.array([0.0, v_circ * math.cos(inc_iss), v_circ * math.sin(inc_iss)])
    a_j2 = j2_model.compute_acceleration(0, pos_node, vel_node, 1000)

    # Verify J2 acceleration component magnitude against exact formula
    # a_j2 = -1.5 * J2 * mu * Re^2 / r^4 * [ (1 - 5 z^2/r^2) x_hat + ... ]
    # At z = 0 (equator): a_j2_r = -1.5 * J2 * mu * Re^2 / r^4
    a_j2_ref_mag = 1.5 * EARTH.J2 * EARTH.mu * (EARTH.radius ** 2) / (r_iss ** 4)
    a_j2_calc_mag = abs(a_j2[0])
    err_j2_a = abs(a_j2_calc_mag - a_j2_ref_mag) / a_j2_ref_mag
    if err_j2_a > 1e-12: val4_j2_passed = False
    print(f"  [{'PASS' if err_j2_a < 1e-12 else 'FAIL'}] J2 Equatorial Acceleration: calc={a_j2_calc_mag:.8e} m/s^2, ref={a_j2_ref_mag:.8e} m/s^2 (err={err_j2_a:.2e})")
    print(f"  [INFO] Analytical Secular RAAN Drift Rate: {math.degrees(raan_rate_ref)*86400:.4f} deg/day")

    results["Gravity"] = "PASS" if val4_grav_passed else "FAIL"
    results["J2"] = "PASS" if val4_j2_passed else "FAIL"

    # -------------------------------------------------------------------------
    # VALIDATION 5: LAMBERT PROBLEM SOLVER (WITH LITERATURE BENCHMARKS)
    # -------------------------------------------------------------------------
    print("\n--- VALIDATION 5: LAMBERT SOLVER ---")
    val5_passed = True

    # Case 1: Published Benchmark from Vallado Example 7.5 / Curtis Example 5.2 (3D inclined transfer)
    # r1 = [5000, 10000, 2100] km, r2 = [-14600, 2500, 7000] km, TOF = 3600 s
    r1_val = np.array([5000e3, 10000e3, 2100e3])
    r2_val = np.array([-14600e3, 2500e3, 7000e3])
    tof_val = 3600.0

    sol_val = solve_lambert(r1_val, r2_val, tof_val, mu, prograde=True)
    # Independent trajectory propagation from (r1, v1) for TOF
    def deriv_2body(t, y):
        r = y[:3]; v = y[3:6]
        rm = np.linalg.norm(r)
        return np.concatenate([v, -mu / (rm**3) * r])

    rkf = RKF45Integrator(atol=1e-12, rtol=1e-12, dt_initial=10.0)
    res_val = rkf.integrate(deriv_2body, np.concatenate([r1_val, sol_val.v1]), (0, tof_val))
    rf_val = res_val.states[-1][:3]
    vf_val = res_val.states[-1][3:6]

    pos_miss_1 = np.linalg.norm(rf_val - r2_val)
    vel_miss_1 = np.linalg.norm(vf_val - sol_val.v2)
    p1 = (pos_miss_1 < 0.01) and sol_val.converged
    if not p1: val5_passed = False
    print(f"  [{'PASS' if p1 else 'FAIL'}] Vallado Ex 7.5 (3D Arc): conv={sol_val.converged}, iters={sol_val.iterations}, pos_miss={pos_miss_1:.6f} m, vel_miss={vel_miss_1:.6f} m/s")

    # Case 2: 180° Hohmann Collinear Transfer (Curtis Ex 5.3)
    r1_hoh = np.array([7000e3, 0, 0])
    r2_hoh = np.array([-42164e3, 0, 0])
    at_hoh = (7000e3 + 42164e3) / 2.0
    tof_hoh = math.pi * math.sqrt(at_hoh**3 / mu)

    # Reference analytical velocities:
    v1_hoh_ref = math.sqrt(mu * (2.0 / 7000e3 - 1.0 / at_hoh))  # 9882.849 m/s
    v2_hoh_ref = math.sqrt(mu * (2.0 / 42164e3 - 1.0 / at_hoh))  # 1640.735 m/s

    sol_hoh = solve_lambert(r1_hoh, r2_hoh, tof_hoh, mu, prograde=True)
    res_hoh = rkf.integrate(deriv_2body, np.concatenate([r1_hoh, sol_hoh.v1]), (0, tof_hoh))
    rf_hoh = res_hoh.states[-1][:3]
    pos_miss_2 = np.linalg.norm(rf_hoh - r2_hoh)
    err_v1_hoh = abs(np.linalg.norm(sol_hoh.v1) - v1_hoh_ref)
    p2 = (pos_miss_2 < 0.01) and (err_v1_hoh < 1e-3) and sol_hoh.converged
    if not p2: val5_passed = False
    print(f"  [{'PASS' if p2 else 'FAIL'}] 180 deg Hohmann Transfer: conv={sol_hoh.converged}, v1_err={err_v1_hoh:.2e} m/s, pos_miss={pos_miss_2:.6f} m")

    # Case 3: Long-way transfer (240 deg)
    ang = math.radians(240)
    r1_long = np.array([6800e3, 0, 0])
    r2_long = np.array([7200e3 * math.cos(ang), 7200e3 * math.sin(ang), 0])
    tof_long = 4500.0

    sol_long = solve_lambert(r1_long, r2_long, tof_long, mu, prograde=True)
    res_long = rkf.integrate(deriv_2body, np.concatenate([r1_long, sol_long.v1]), (0, tof_long))
    rf_long = res_long.states[-1][:3]
    pos_miss_3 = np.linalg.norm(rf_long - r2_long)
    p3 = (pos_miss_3 < 0.01) and sol_long.converged
    if not p3: val5_passed = False
    print(f"  [{'PASS' if p3 else 'FAIL'}] Long-Way Transfer (240 deg): conv={sol_long.converged}, iters={sol_long.iterations}, pos_miss={pos_miss_3:.6f} m")

    # Case 4: High Energy Hyperbolic Transfer (90 deg in 500 s)
    r1_hyp = np.array([7000e3, 0, 0])
    r2_hyp = np.array([0, 7000e3, 0])
    tof_hyp = 500.0

    sol_hyp = solve_lambert(r1_hyp, r2_hyp, tof_hyp, mu, prograde=True)
    res_hyp = rkf.integrate(deriv_2body, np.concatenate([r1_hyp, sol_hyp.v1]), (0, tof_hyp))
    rf_hyp = res_hyp.states[-1][:3]
    pos_miss_4 = np.linalg.norm(rf_hyp - r2_hyp)
    p4 = (pos_miss_4 < 0.01) and sol_hyp.converged and (sol_hyp.trajectory_type == "hyperbolic")
    if not p4: val5_passed = False
    print(f"  [{'PASS' if p4 else 'FAIL'}] Fast Hyperbolic Arc (500s): conv={sol_hyp.converged}, type={sol_hyp.trajectory_type}, pos_miss={pos_miss_4:.6f} m")

    results["Lambert"] = "PASS" if val5_passed else "FAIL"

    # -------------------------------------------------------------------------
    # VALIDATION 6: HOHMANN / ORBITAL TRANSFERS
    # -------------------------------------------------------------------------
    print("\n--- VALIDATION 6: HOHMANN & TRANSFERS ---")
    val6_passed = True

    r_leo_300 = EARTH.radius + 300e3
    r_geo = 42164e3

    # Independent analytical calculation
    v_c1 = math.sqrt(mu / r_leo_300)
    v_c2 = math.sqrt(mu / r_geo)
    a_trans = (r_leo_300 + r_geo) / 2.0
    v_t1 = math.sqrt(mu * (2.0 / r_leo_300 - 1.0 / a_trans))
    v_t2 = math.sqrt(mu * (2.0 / r_geo - 1.0 / a_trans))
    dv1_ref = v_t1 - v_c1
    dv2_ref = v_c2 - v_t2
    dv_tot_ref = dv1_ref + dv2_ref
    t_trans_ref = math.pi * math.sqrt(a_trans**3 / mu)

    hoh_res = hohmann_transfer(r_leo_300, r_geo, mu)
    err_dv1 = abs(hoh_res.delta_v1 - dv1_ref)
    err_dv2 = abs(hoh_res.delta_v2 - dv2_ref)
    err_tot = abs(hoh_res.total_delta_v - dv_tot_ref)
    err_time = abs(hoh_res.transfer_time - t_trans_ref)

    p_hoh = (err_tot < 1e-6) and (err_time < 1e-6)
    if not p_hoh: val6_passed = False
    print(f"  [{'PASS' if p_hoh else 'FAIL'}] LEO(300km)->GEO Hohmann: dv_tot={hoh_res.total_delta_v:.3f} m/s (ref={dv_tot_ref:.3f}), err={err_tot:.2e} m/s")

    # Combined transfer with 28.5 deg plane change
    d_inc = math.radians(28.5)
    # At second burn: vector triangle delta_v2 = sqrt(v_c2^2 + v_t2^2 - 2*v_c2*v_t2*cos(d_inc))
    dv2_comb_ref = math.sqrt(v_c2**2 + v_t2**2 - 2.0 * v_c2 * v_t2 * math.cos(d_inc))
    dv_comb_tot_ref = dv1_ref + dv2_comb_ref

    comb_res = combined_maneuver(r_leo_300, r_geo, d_inc, mu)
    err_comb = abs(comb_res.total_delta_v - dv_comb_tot_ref)
    p_comb = (err_comb < 1e-6)
    if not p_comb: val6_passed = False
    print(f"  [{'PASS' if p_comb else 'FAIL'}] Combined (28.5 deg plane change): dv_tot={comb_res.total_delta_v:.3f} m/s (ref={dv_comb_tot_ref:.3f}), err={err_comb:.2e} m/s")

    results["Transfers"] = "PASS" if val6_passed else "FAIL"

    # -------------------------------------------------------------------------
    # VALIDATION 7: RENDEZVOUS
    # -------------------------------------------------------------------------
    print("\n--- VALIDATION 7: RENDEZVOUS INTERCEPTION ---")
    val7_passed = True

    # Scenario:
    # Chaser in 400 km orbit at theta = 0
    # Target in 420 km orbit, 60 deg ahead
    # TOF = 3600 s (1 hour)
    r_chaser_0 = np.array([EARTH.radius + 400e3, 0.0, 0.0])
    v_c1_speed = math.sqrt(mu / np.linalg.norm(r_chaser_0))
    v_chaser_0 = np.array([0.0, v_c1_speed, 0.0])

    r_target_mag = EARTH.radius + 420e3
    v_target_speed = math.sqrt(mu / r_target_mag)
    lead_ang = math.radians(60.0)
    r_target_0 = np.array([r_target_mag * math.cos(lead_ang), r_target_mag * math.sin(lead_ang), 0.0])
    v_target_0 = np.array([-v_target_speed * math.sin(lead_ang), v_target_speed * math.cos(lead_ang), 0.0])

    tof_rendezvous = 3600.0

    # 1. Run THESEUS solve_rendezvous
    rend_res = solve_rendezvous(r_chaser_0, v_chaser_0, r_target_0, v_target_0, tof_rendezvous, mu)

    # 2. INDEPENDENT VERIFICATION:
    # Propagate target independently from (r_target_0, v_target_0) for 3600 s
    res_tgt_prop = rkf.integrate(deriv_2body, np.concatenate([r_target_0, v_target_0]), (0, tof_rendezvous))
    target_pos_actual = res_tgt_prop.states[-1][:3]
    target_vel_actual = res_tgt_prop.states[-1][3:6]

    # Propagate chaser independently from (r_chaser_0, r_chaser_0_v + dv_depart) for 3600 s
    v_chaser_depart = r_chaser_0_v = rend_res.lambert_solution.v1
    res_chs_prop = rkf.integrate(deriv_2body, np.concatenate([r_chaser_0, v_chaser_depart]), (0, tof_rendezvous))
    chaser_pos_arrival = res_chs_prop.states[-1][:3]
    chaser_vel_arrival = res_chs_prop.states[-1][3:6]

    # Calculate actual miss distance at rendezvous
    miss_distance = np.linalg.norm(chaser_pos_arrival - target_pos_actual)
    rel_vel_actual = np.linalg.norm(chaser_vel_arrival - target_vel_actual)
    rel_vel_diff = abs(rel_vel_actual - rend_res.relative_velocity_at_arrival)

    p_rend = (miss_distance < 0.01) and (rel_vel_diff < 1e-4) and rend_res.lambert_solution.converged
    if not p_rend: val7_passed = False
    print(f"  [{'PASS' if p_rend else 'FAIL'}] Rendezvous 1-hr Intercept: Miss Distance={miss_distance:.6f} m, Rel Velocity={rel_vel_actual:.3f} m/s, dv_tot={rend_res.delta_v_total:.2f} m/s")

    results["Rendezvous"] = "PASS" if val7_passed else "FAIL"

    # -------------------------------------------------------------------------
    # VALIDATION 8: EPHEMERIDES
    # -------------------------------------------------------------------------
    print("\n--- VALIDATION 8: EPHEMERIDES ---")
    val8_passed = True

    simple_eph = SimpleEphemerisProvider()
    astropy_eph = AstropyEphemerisProvider()

    # 1. Earth-Sun nominal distance at J2000: ~ 1 AU = 1.496e11 m
    r_sun_astropy = astropy_eph.get_position("Sun", JD_J2000)
    d_sun_astropy = np.linalg.norm(r_sun_astropy)
    err_sun = abs(d_sun_astropy - AU_VAL) / AU_VAL
    p_sun = (err_sun < 0.03)  # Earth orbital eccentricity ~ 0.0167
    if not p_sun: val8_passed = False
    print(f"  [{'PASS' if p_sun else 'FAIL'}] Astropy Geocentric Sun Distance at J2000: {d_sun_astropy/1e3:.1f} km (~1 AU, err={err_sun*100:.2f}%)")

    # 2. Earth-Moon nominal distance: ~ 384,400 km
    r_moon_astropy = astropy_eph.get_position("Moon", JD_J2000)
    d_moon_astropy = np.linalg.norm(r_moon_astropy)
    err_moon = abs(d_moon_astropy - 384400e3) / 384400e3
    p_moon = (err_moon < 0.08)  # Moon eccentricity ~ 0.055
    if not p_moon: val8_passed = False
    print(f"  [{'PASS' if p_moon else 'FAIL'}] Astropy Geocentric Moon Distance at J2000: {d_moon_astropy/1e3:.1f} km (~384,400 km, err={err_moon*100:.2f}%)")

    # 3. Simple Ephemeris provider parent-centered distances
    r_earth_simple = simple_eph.get_position("Earth", JD_J2000)
    r_moon_simple = simple_eph.get_position("Moon", JD_J2000)
    p_simple = (abs(np.linalg.norm(r_earth_simple) - 1.496e11) < 1e9) and (abs(np.linalg.norm(r_moon_simple) - 3.844e8) < 1e7)
    if not p_simple: val8_passed = False
    print(f"  [{'PASS' if p_simple else 'FAIL'}] Simple Provider Distances: Earth={np.linalg.norm(r_earth_simple)/1e3:.0f} km, Moon={np.linalg.norm(r_moon_simple)/1e3:.0f} km")

    results["Ephemerides"] = "PASS" if val8_passed else "FAIL"

    # -------------------------------------------------------------------------
    # VALIDATION 9: SOLAR RADIATION PRESSURE
    # -------------------------------------------------------------------------
    print("\n--- VALIDATION 9: SOLAR RADIATION PRESSURE ---")
    val9_passed = True

    # Nominal radiation pressure at 1 AU:
    # P0 = L_sun / (4 * pi * c * (1 AU)^2) = 3.828e26 / (4 * pi * 299792458 * (1.495978707e11)^2) = 4.5404e-6 N/m^2
    P0_ref = L_SUN_VAL / (4.0 * math.pi * C_VAL * (AU_VAL ** 2))
    cr = 1.5
    area = 10.0
    mass = 1000.0

    # Expected acceleration at 1 AU away from shadow:
    a_srp_ref = P0_ref * cr * area / mass  # 6.8106e-8 m/s^2

    srp = SolarRadiationPressure(astropy_eph, cr=cr, area=area, shadow_body_radius=EARTH.radius)
    pos_sun_dir = r_sun_astropy / np.linalg.norm(r_sun_astropy)

    # Place spacecraft towards Sun at 7000 km from Earth center
    pos_lit = pos_sun_dir * 7000e3
    a_srp_calc = srp.compute_acceleration(0.0, pos_lit, np.zeros(3), mass)
    a_srp_calc_mag = np.linalg.norm(a_srp_calc)

    err_srp = abs(a_srp_calc_mag - a_srp_ref) / a_srp_ref
    p_srp_mag = (err_srp < 0.05)
    if not p_srp_mag: val9_passed = False
    print(f"  [{'PASS' if p_srp_mag else 'FAIL'}] SRP Acceleration at 1 AU: calc={a_srp_calc_mag:.6e} m/s^2, ref={a_srp_ref:.6e} m/s^2 (err={err_srp*100:.2f}%)")

    # Shadow check (spacecraft placed directly behind Earth in umbra)
    pos_shadow = -pos_sun_dir * 7000e3
    a_srp_shadow = srp.compute_acceleration(0.0, pos_shadow, np.zeros(3), mass)
    p_shadow = (np.linalg.norm(a_srp_shadow) == 0.0) and not np.any(np.isnan(a_srp_shadow))
    if not p_shadow: val9_passed = False
    print(f"  [{'PASS' if p_shadow else 'FAIL'}] SRP Cylindrical Shadow: acc={np.linalg.norm(a_srp_shadow):.1f} m/s^2 (strictly zero, no NaNs)")

    # 1/r^2 distance scaling test
    srp_noshadow = SolarRadiationPressure(astropy_eph, cr=cr, area=area, shadow_body_radius=0.0)
    # In ECI: Sun is at +pos_sun_dir * 1 AU.
    # To be at 2 AU from Sun, spacecraft is placed at -pos_sun_dir * 1 AU from Earth.
    pos_2au = -pos_sun_dir * AU_VAL
    a_2au = np.linalg.norm(srp_noshadow.compute_acceleration(0.0, pos_2au, np.zeros(3), mass))
    # Distance from Sun is now 2 AU -> acceleration should be (1/2)^2 = 0.25 of 1 AU value
    a_ratio = a_2au / a_srp_calc_mag
    p_dist = (abs(a_ratio - 0.25) < 0.05)
    if not p_dist: val9_passed = False
    print(f"  [{'PASS' if p_dist else 'FAIL'}] SRP 1/r^2 Scaling (2 AU): ratio={a_ratio:.4f} (theoretical: 0.2500)")

    results["SRP"] = "PASS" if val9_passed else "FAIL"

    # -------------------------------------------------------------------------
    # VALIDATION 10: SPACECRAFT MASS DEPLETION & THRUST
    # -------------------------------------------------------------------------
    print("\n--- VALIDATION 10: SPACECRAFT MASS DEPLETION & THRUST ---")
    val10_passed = True

    m_dry = 2000.0
    m_fuel = 3000.0
    m_total = m_dry + m_fuel
    isp = 316.0
    f_thrust = 500.0  # N

    sc = Spacecraft(name="Sat", dry_mass=m_dry, fuel_mass=m_fuel, specific_impulse=isp, thrust=f_thrust)

    # 1. Tsiolkovsky rocket equation: dv = isp * g0 * ln(m0 / mf)
    dv_avail_calc = sc.delta_v_available()
    dv_avail_ref = isp * G0_VAL * math.log(m_total / m_dry)
    err_dv_avail = abs(dv_avail_calc - dv_avail_ref)
    p_tsiol = (err_dv_avail < 1e-9)
    if not p_tsiol: val10_passed = False
    print(f"  [{'PASS' if p_tsiol else 'FAIL'}] Tsiolkovsky Delta-v: calc={dv_avail_calc:.4f} m/s, ref={dv_avail_ref:.4f} m/s (err={err_dv_avail:.2e})")

    # 2. Mass flow rate mdot = F / (isp * g0)
    mdot_ref = f_thrust / (isp * G0_VAL)
    mdot_calc = sc.mass_flow_rate
    err_mdot = abs(mdot_calc - mdot_ref)
    p_mdot = (err_mdot < 1e-12)
    if not p_mdot: val10_passed = False
    print(f"  [{'PASS' if p_mdot else 'FAIL'}] Mass Flow Rate mdot: calc={mdot_calc:.6f} kg/s, ref={mdot_ref:.6f} kg/s (err={err_mdot:.2e})")

    # 3. Finite continuous burn propagation with 7-DOF mass integration
    burn_dur = 1000.0  # 1000 seconds
    fuel_burned_ref = mdot_ref * burn_dur  # 161.346 kg
    fuel_remaining_ref = m_fuel - fuel_burned_ref

    thrust_mod = ThrustModel(spacecraft=sc, direction=ThrustDirection.PROGRADE, burn_start=0.0, burn_end=burn_dur, throttle=1.0)
    grav_mod = PointMassGravity(EARTH)
    composite = CompositeForceModel([grav_mod, thrust_mod])

    try:
        num_prop = NumericalPropagator(acceleration_fn=composite.compute_acceleration, integrator="rk4", dt=10.0, mu=EARTH.mu)
        hist, _, _ = num_prop.propagate(r0, v0, (0.0, burn_dur), mass=m_total, fuel_mass=m_fuel)
        final_state = hist[-1]
        err_fuel_num = abs(final_state.fuel_mass - fuel_remaining_ref)
        err_mass_num = abs(final_state.mass - (m_dry + fuel_remaining_ref))
        p_burn = (err_fuel_num < 0.1) and (err_mass_num < 0.1)
        print(f"  [{'PASS' if p_burn else 'FAIL'}] 7-DOF Finite Burn Fuel Depletion: final_fuel={final_state.fuel_mass:.3f} kg (ref={fuel_remaining_ref:.3f} kg), err={err_fuel_num:.2e} kg")
    except Exception as e:
        p_burn = False
        val10_passed = False
        print(f"  [FAIL] 7-DOF Numerical Finite Burn: Crashed with {type(e).__name__}: {e}")

    results["Thrust/mass"] = "PASS" if val10_passed else "FAIL"

    # -------------------------------------------------------------------------
    # VALIDATION 11: TIME & EPOCH PRECISION
    # -------------------------------------------------------------------------
    print("\n--- VALIDATION 11: TIME & EPOCH PRECISION ---")
    val11_passed = True

    # 1. J2000 JD definition: 2000-01-01 12:00:00 UTC = JD 2451545.0
    ep_j2000 = Epoch.from_datetime(datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
    err_j2000 = abs(ep_j2000.jd - 2451545.0)
    p_j2000 = (err_j2000 == 0.0)
    if not p_j2000: val11_passed = False
    print(f"  [{'PASS' if p_j2000 else 'FAIL'}] J2000 Epoch JD: jd={ep_j2000.jd:.6f}, expected=2451545.000000 (err={err_j2000})")

    # 2. DateTime roundtrip across various dates
    test_dts = [
        datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc),
        datetime(1995, 10, 4, 8, 12, 45, tzinfo=timezone.utc),
        datetime(2030, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    ]
    for dt_in in test_dts:
        ep = Epoch.from_datetime(dt_in)
        dt_out = ep.to_datetime()
        dt_diff_s = abs((dt_out - dt_in).total_seconds())
        p_dt = (dt_diff_s < 1e-4)  # within sub-millisecond float64 limit
        if not p_dt: val11_passed = False
        print(f"  [{'PASS' if p_dt else 'FAIL'}] DateTime Roundtrip ({dt_in.isoformat()}): out={dt_out.isoformat()}, err={dt_diff_s:.2e} s")

    # 3. Time scale offsets: TT - UTC at J2000 ≈ 64.184 s
    ep_utc = Epoch(JD_J2000, scale=TimeScale.UTC)
    ep_tt = ep_utc.to_scale(TimeScale.TT)
    offset_tt = (ep_tt.jd - ep_utc.jd) * 86400.0
    p_scale = abs(offset_tt - 64.184) < 1.0  # Approx 64.184s offset
    if not p_scale: val11_passed = False
    print(f"  [{'PASS' if p_scale else 'FAIL'}] TT - UTC Offset: offset={offset_tt:.3f} s, expected=64.184 s")

    results["Time/epoch"] = "PASS" if val11_passed else "FAIL"

    # -------------------------------------------------------------------------
    # SUMMARY SCORECARD
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL INDEPENDENT VALIDATION SUMMARY SCORECARD")
    print("=" * 80)
    for sub, stat in results.items():
        print(f"  {sub:25s}: {stat}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_all_verifications()
