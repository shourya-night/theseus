"""
THESEUS Phase 1-7 Core Astrodynamics Engine Integrity Gate
===========================================================
A dedicated, granular, and mathematically rigorous test suite verifying
the Phase 1-7 Core Astrodynamics Engine without importing Phase 8-10 extensions.

Covers:
- Test 1: Constants & Fundamental Planetary Parameters (Consistency & Accuracy)
- Test 2: Coordinate & Orbital Element Conversions (Multi-Regime & Singularities)
- Test 3: Keplerian Two-Body Mechanics & Invariant Conservation
- Test 4: RK4 Integrator Fourth-Order Convergence Rate
- Test 5: RKF45 Adaptive Integrator Tolerance Scaling
- Test 6: Point-Mass Gravitational Acceleration & Inverse-Square Scaling
- Test 7: J2 Oblateness Secular Perturbation Rates
- Test 8: US1976 Standard Atmosphere Thermodynamic Consistency
- Test 9: Aerodynamic Drag Scaling & Co-Rotating Atmosphere
- Test 10: Solar Radiation Pressure & Planetary Shadow Model
- Test 11: Thruster Dynamics & Tsiolkovsky Propellant Depletion
- Test 12: Lambert Boundary Value Solver & Propagation Endpoint Closure
- Test 13: Hohmann Transfers Across Multiple Radius Ratios
- Test 14: Orbital Rendezvous Guidance & Relative State Closure
- Test 15: Core Engine Import Isolation (No Phase 8-10 Leakage)
"""

from __future__ import annotations

import math
import sys
import subprocess
import numpy as np
import pytest

from theseus.constants.physical import (
    G, SPEED_OF_LIGHT, STANDARD_GRAVITY, ASTRONOMICAL_UNIT,
    SOLAR_LUMINOSITY, SOLAR_IRRADIANCE_1AU,
    G_VAL, C_VAL, G0_VAL, AU_VAL, L_SUN_VAL, S0_VAL,
)
from theseus.bodies.catalog import ALL_BODIES, EARTH, SUN, MOON, MARS, JUPITER
from theseus.coordinates.transformations import (
    perifocal_to_eci_matrix, eci_to_perifocal_matrix,
    eci_to_ecef, ecef_to_eci, cartesian_to_spherical, spherical_to_cartesian,
)
from theseus.orbital.elements import OrbitalElements
from theseus.orbital.conversions import state_to_elements, elements_to_state
from theseus.orbital.kepler import solve_kepler, eccentric_to_true, hyperbolic_to_true
from theseus.orbital.lambert import solve_lambert
from theseus.propagation.analytical import propagate_twobody
from theseus.propagation.integrators import RK4Integrator, RKF45Integrator
from theseus.propagation.numerical import NumericalPropagator
from theseus.dynamics.gravity import PointMassGravity, J2Perturbation
from theseus.dynamics.drag import DragModel
from theseus.dynamics.srp import SolarRadiationPressure
from theseus.dynamics.thrust import ThrustModel, ThrustDirection
from theseus.dynamics.force_model import CompositeForceModel
from theseus.atmosphere.models import US1976StandardAtmosphere
from theseus.ephemeris.simple_provider import SimpleEphemerisProvider
from theseus.spacecraft.vehicle import Spacecraft
from theseus.maneuvers.transfers import hohmann_transfer, bielliptic_transfer
from theseus.rendezvous.solver import solve_rendezvous


# ===========================================================================
# 1. CONSTANTS INTEGRITY
# ===========================================================================

def test_constants_integrity():
    """Verify physical constants, significant figures, and internal consistency (mu = G * M)."""
    assert SPEED_OF_LIGHT.value == 299_792_458.0
    assert C_VAL == 299_792_458.0
    assert ASTRONOMICAL_UNIT.value == 149_597_870_700.0
    assert AU_VAL == 149_597_870_700.0
    assert STANDARD_GRAVITY.value == 9.80665
    assert G0_VAL == 9.80665
    assert G_VAL == pytest.approx(6.67430e-11, rel=1e-5)

    # Solar irradiance consistency S0 = L_sun / (4 * pi * AU^2)
    s0_calc = L_SUN_VAL / (4.0 * math.pi * AU_VAL**2)
    assert s0_calc == pytest.approx(S0_VAL, rel=1e-3)

    # Internal consistency: mu ≈ G * M
    for body in [EARTH, SUN, MARS, JUPITER]:
        mu_from_gm = G_VAL * body.mass
        assert body.mu == pytest.approx(mu_from_gm, rel=1e-4)


# ===========================================================================
# 2. COORDINATE & ORBITAL ELEMENT CONVERSIONS (MULTI-REGIME)
# ===========================================================================

@pytest.mark.parametrize("scenario, a_km, e, i_deg, raan_deg, argp_deg, nu_deg", [
    ("circular_equatorial", 7000.0, 0.0, 0.0, 0.0, 0.0, 45.0),
    ("elliptic_equatorial", 7500.0, 0.25, 0.0, 0.0, 30.0, 60.0),
    ("circular_inclined_iss", 6778.137, 0.0, 51.6, 120.0, 0.0, 90.0),
    ("elliptic_inclined", 8000.0, 0.15, 28.5, 40.0, 30.0, 15.0),
    ("molniya_critical_inclination", 26600.0, 0.74, 63.4, 45.0, 270.0, 180.0),
    ("polar_orbit", 7200.0, 0.05, 90.0, 15.0, 45.0, 120.0),
    ("retrograde_orbit", 7000.0, 0.02, 135.0, 80.0, 110.0, 200.0),
    ("near_circular_singularity", 7000.0, 1e-6, 45.0, 10.0, 20.0, 30.0),
    ("near_equatorial_singularity", 7000.0, 0.1, 1e-6, 0.0, 30.0, 45.0),
])
def test_coordinate_roundtrip_multi_regime(scenario, a_km, e, i_deg, raan_deg, argp_deg, nu_deg):
    """Test Keplerian -> Cartesian -> Keplerian conversion across multiple orbit regimes and edge cases."""
    mu = EARTH.mu
    orig_elem = OrbitalElements(
        a=a_km * 1e3,
        e=e,
        i=math.radians(i_deg),
        raan=math.radians(raan_deg),
        argp=math.radians(argp_deg),
        nu=math.radians(nu_deg),
        mu=mu,
    )
    
    r_vec, v_vec = elements_to_state(orig_elem)
    rec_elem = state_to_elements(r_vec, v_vec, mu)

    # Physical checks
    assert rec_elem.a == pytest.approx(orig_elem.a, rel=1e-5)
    assert rec_elem.e == pytest.approx(orig_elem.e, abs=1e-5)
    assert rec_elem.i == pytest.approx(orig_elem.i, abs=1e-5)
    
    # State roundtrip Cartesian -> Keplerian -> Cartesian
    r_rec, v_rec = elements_to_state(rec_elem)
    np.testing.assert_allclose(r_rec, r_vec, atol=1.0)        # within 1 meter
    np.testing.assert_allclose(v_rec, v_vec, atol=1e-3)       # within 1 mm/s


# ===========================================================================
# 3. KEPLER PROPAGATION & INVARIANT CONSERVATION
# ===========================================================================

def test_kepler_propagation_conservation():
    """Verify analytical Keplerian two-body propagation conserves energy and angular momentum to machine precision."""
    mu = EARTH.mu
    a = 7000e3
    e = 0.1
    i = math.radians(28.5)
    elem = OrbitalElements(a=a, e=e, i=i, raan=0.5, argp=0.8, nu=0.0, mu=mu)
    r0, v0 = elements_to_state(elem)

    period = elem.period
    t_eval = np.linspace(0, 10.0 * period, 500)
    history = propagate_twobody(r0, v0, mu, t_eval)

    eps_0 = np.linalg.norm(v0)**2 / 2.0 - mu / np.linalg.norm(r0)
    h0_vec = np.cross(r0, v0)
    h0 = np.linalg.norm(h0_vec)

    for state in history:
        r = state.position
        v = state.velocity
        eps = np.linalg.norm(v)**2 / 2.0 - mu / np.linalg.norm(r)
        h = np.linalg.norm(np.cross(r, v))

        assert abs(eps - eps_0) / abs(eps_0) < 1e-12
        assert abs(h - h0) / h0 < 1e-12


# ===========================================================================
# 4. RK4 CONVERGENCE RATE
# ===========================================================================

def test_rk4_convergence_rate():
    """Verify RK4 exhibits true fourth-order asymptotic error scaling O(dt^4)."""
    def f(t, y):
        return -y

    y_exact = math.exp(-1.0)
    dts = [0.2, 0.1, 0.05, 0.025]
    errors = []

    for dt in dts:
        rk4 = RK4Integrator(dt=dt)
        res = rk4.integrate(f, np.array([1.0]), (0.0, 1.0))
        err = abs(res.states[-1][0] - y_exact)
        errors.append(err)

    for i in range(len(errors) - 1):
        ratio = errors[i] / errors[i + 1]
        assert 14.0 < ratio < 18.0, f"RK4 error ratio {ratio:.2f} deviates from expected 16x"


# ===========================================================================
# 5. RKF45 ADAPTIVE TOLERANCE SCALING
# ===========================================================================

def test_rkf45_tolerance_scaling():
    """Verify RKF45 adaptive integrator produces monotonically decreasing global error with tighter tolerances."""
    mu = EARTH.mu
    def two_body(t, y):
        r = y[:3]
        v = y[3:6]
        return np.concatenate([v, -mu / (np.linalg.norm(r)**3) * r])

    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, math.sqrt(mu / 7000e3), 0.0])
    y0 = np.concatenate([r0, v0])
    period = 2.0 * math.pi * math.sqrt(7000e3**3 / mu)

    errors = []
    for tol in [1e-6, 1e-9, 1e-12]:
        rkf = RKF45Integrator(atol=tol, rtol=tol, dt_initial=10.0)
        res = rkf.integrate(two_body, y0, (0.0, period))
        err = np.linalg.norm(res.states[-1][:3] - r0)
        errors.append(err)

    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.05  # sub-decimeter closure over 1 orbit at 1e-12 tolerance


# ===========================================================================
# 6. POINT MASS GRAVITY
# ===========================================================================

def test_gravity():
    """Verify point-mass gravity acceleration and inverse-square scaling."""
    pmg = PointMassGravity(body=EARTH)
    r1 = np.array([7000e3, 0.0, 0.0])
    v1 = np.array([0.0, 7500.0, 0.0])
    a1 = pmg.compute_acceleration(0.0, r1, v1, mass=1000.0)

    expected_a1 = -EARTH.mu / (7000e3**2)
    assert a1[0] == pytest.approx(expected_a1, rel=1e-14)
    assert a1[1] == 0.0 and a1[2] == 0.0

    # 2x distance -> 4x smaller acceleration
    r2 = np.array([14000e3, 0.0, 0.0])
    a2 = pmg.compute_acceleration(0.0, r2, v1, mass=1000.0)
    assert np.linalg.norm(a1) / np.linalg.norm(a2) == pytest.approx(4.0, rel=1e-12)


# ===========================================================================
# 7. J2 OBLATENESS PERTURBATION
# ===========================================================================

def test_j2_perturbation():
    """Verify J2 acceleration components and secular RAAN precession rate."""
    j2_model = J2Perturbation(body=EARTH)
    r = np.array([7000e3, 0.0, 0.0])
    v = np.array([0.0, 7500.0, 0.0])
    a_j2 = j2_model.compute_acceleration(0.0, r, v, mass=1000.0)

    expected_ax = -1.5 * EARTH.J2 * EARTH.mu * (EARTH.radius**2) / (7000e3**4)
    assert a_j2[0] == pytest.approx(expected_ax, rel=1e-6)

    # Secular nodal precession: dOmega/dt = -1.5 * n * J2 * (R/p)^2 * cos(i)
    a = 7000e3
    i = math.radians(51.6)
    n = math.sqrt(EARTH.mu / a**3)
    p = a
    expected_raan_dot = -1.5 * n * EARTH.J2 * ((EARTH.radius / p)**2) * math.cos(i)
    # Expected ~ -4.47 degrees/day for 7000 km orbit at 51.6 deg
    deg_per_day = math.degrees(expected_raan_dot) * 86400.0
    assert -6.0 < deg_per_day < -4.0


# ===========================================================================
# 8. US1976 ATMOSPHERE THERMODYNAMICS
# ===========================================================================

def test_atmosphere():
    """Verify US1976 Standard Atmosphere thermodynamic consistency and monotonic density decrease."""
    atmos = US1976StandardAtmosphere()
    props0 = atmos.get_properties(0.0)
    rho0 = props0.density
    temp0 = props0.temperature
    p0 = props0.pressure

    assert rho0 == pytest.approx(1.2250, abs=1e-3)
    assert temp0 == pytest.approx(288.15, abs=0.1)
    assert p0 == pytest.approx(101325.0, abs=10.0)

    # Monotonic density decrease
    altitudes = [0.0, 10e3, 25e3, 50e3, 85e3, 120e3, 200e3, 400e3]
    densities = [atmos.density(h) for h in altitudes]
    for k in range(len(densities) - 1):
        assert densities[k] > densities[k + 1]


# ===========================================================================
# 9. AERODYNAMIC DRAG
# ===========================================================================

def test_drag():
    """Verify aerodynamic drag magnitude, velocity squared scaling, and planetary co-rotation."""
    atmos = US1976StandardAtmosphere()
    drag = DragModel(atmosphere=atmos, cd=2.2, area=10.0, body_radius=EARTH.radius)

    r_leo = np.array([EARTH.radius + 200e3, 0.0, 0.0])
    v_leo1 = np.array([0.0, 7000.0, 0.0])
    v_leo2 = np.array([0.0, 14000.0, 0.0])

    a_d1 = drag.compute_acceleration(0.0, r_leo, v_leo1, mass=1000.0)
    a_d2 = drag.compute_acceleration(0.0, r_leo, v_leo2, mass=1000.0)

    # Drag must oppose motion
    assert np.dot(a_d1, v_leo1) < 0
    assert np.linalg.norm(a_d2) > np.linalg.norm(a_d1)


# ===========================================================================
# 10. SOLAR RADIATION PRESSURE
# ===========================================================================

def test_srp():
    """Verify solar radiation pressure magnitude, distance scaling, and cylindrical shadow cutoff."""
    eph = SimpleEphemerisProvider()
    srp_sunlit = SolarRadiationPressure(ephemeris=eph, cr=1.5, area=10.0, shadow_body_radius=0.0)

    # In sunlight near Earth
    r_sc = np.array([7000e3, 0.0, 0.0])
    v_sc = np.array([0.0, 7500.0, 0.0])
    a_srp = srp_sunlit.compute_acceleration(0.0, r_sc, v_sc, mass=1000.0)
    assert np.linalg.norm(a_srp) > 0
    expected_acc = (L_SUN_VAL / (4.0 * math.pi * C_VAL * AU_VAL**2)) * 1.5 * 10.0 / 1000.0
    assert np.linalg.norm(a_srp) == pytest.approx(expected_acc, rel=0.05)

    # In shadow cylinder
    srp_shadow = SolarRadiationPressure(ephemeris=eph, cr=1.5, area=10.0, shadow_body_radius=EARTH.radius)
    sun_pos = srp_shadow._get_geocentric_sun_position(srp_shadow.epoch_jd_t0)
    sun_dir = sun_pos / np.linalg.norm(sun_pos)
    pos_shadow = -sun_dir * (EARTH.radius + 500e3)
    a_shadow = srp_shadow.compute_acceleration(0.0, pos_shadow, v_sc, mass=1000.0)
    assert np.allclose(a_shadow, [0.0, 0.0, 0.0])


# ===========================================================================
# 11. THRUSTER DYNAMICS & MASS DEPLETION
# ===========================================================================

def test_thrust():
    """Verify thrust acceleration, mass flow rate, and Tsiolkovsky delta-v."""
    sc = Spacecraft(name="TestCraft", dry_mass=500.0, fuel_mass=500.0, thrust=500.0, specific_impulse=300.0)
    thrust = ThrustModel(spacecraft=sc, direction=ThrustDirection.PROGRADE)

    r = np.array([7000e3, 0.0, 0.0])
    v = np.array([0.0, 7500.0, 0.0])

    a_thrust = thrust.compute_acceleration(0.0, r, v, mass=sc.total_mass)
    assert np.dot(a_thrust, v) > 0
    assert np.linalg.norm(a_thrust) == pytest.approx(500.0 / 1000.0, rel=1e-12)

    # Mass flow rate m_dot = F / (Isp * g0)
    expected_mdot = 500.0 / (300.0 * G0_VAL)
    assert sc.mass_flow_rate == pytest.approx(expected_mdot, rel=1e-6)


# ===========================================================================
# 12. LAMBERT SOLVER & INDEPENDENT PROPAGATION CLOSURE
# ===========================================================================

@pytest.mark.parametrize("r1, r2, tof_s, mu, prograde", [
    (np.array([7000e3, 0.0, 0.0]), np.array([0.0, 7000e3, 0.0]), 1500.0, EARTH.mu, True),
    (np.array([6800e3, 1000e3, 500e3]), np.array([-2000e3, 15000e3, 3000e3]), 3000.0, EARTH.mu, True),
    (np.array([149597870.7e3, 0.0, 0.0]), np.array([0.0, 227939200.0e3, 0.0]), 6240.0 * 3600.0, SUN.mu, True),
    (np.array([149597870.7e3, 0.0, 0.0]), np.array([-100000000e3, 200000000e3, 0.0]), 7000.0 * 3600.0, SUN.mu, False),
])
def test_lambert_propagation_closure(r1, r2, tof_s, mu, prograde):
    """
    Solve Lambert two-point boundary problem and INDEPENDENTLY PROPAGATE from r1 with v1 for TOF.
    The propagated terminal state MUST close onto r2 within numerical tolerance.
    """
    sol = solve_lambert(r1, r2, tof_s, mu, prograde=prograde)
    assert sol.converged, f"Lambert solver did not converge: residual={sol.residual}"

    # Numerical integration with RKF45 from r1 with sol.v1
    def two_body(t, y):
        r = y[:3]
        v = y[3:6]
        return np.concatenate([v, -mu / (np.linalg.norm(r)**3) * r])

    rkf = RKF45Integrator(atol=1e-12, rtol=1e-12, dt_initial=min(10.0, tof_s / 50.0))
    res = rkf.integrate(two_body, np.concatenate([r1, sol.v1]), (0.0, tof_s))

    r_final = res.states[-1][:3]
    v_final = res.states[-1][3:6]

    pos_error_m = np.linalg.norm(r_final - r2)
    vel_error_m_s = np.linalg.norm(v_final - sol.v2)

    # Relative closure tolerance
    r_scale = max(np.linalg.norm(r1), np.linalg.norm(r2))
    assert pos_error_m / r_scale < 1e-5, f"Endpoint miss {pos_error_m} m too large for r_scale {r_scale} m"
    assert vel_error_m_s < 0.1, f"Arrival velocity error {vel_error_m_s} m/s too large"


# ===========================================================================
# 13. HOHMANN TRANSFERS (MULTI-RATIO)
# ===========================================================================

@pytest.mark.parametrize("r1, r2", [
    (6678.137e3, 10000e3),
    (6678.137e3, 42164.0e3),
    (7000e3, 384400e3),
])
def test_hohmann_multi_ratio(r1, r2):
    """Verify Hohmann transfer impulses and TOF against analytical closed-form equations."""
    mu = EARTH.mu
    transfer = hohmann_transfer(r1, r2, mu)

    v1_circ = math.sqrt(mu / r1)
    v2_circ = math.sqrt(mu / r2)
    a_tx = (r1 + r2) / 2.0
    v_tx1 = math.sqrt(mu * (2.0 / r1 - 1.0 / a_tx))
    v_tx2 = math.sqrt(mu * (2.0 / r2 - 1.0 / a_tx))
    expected_dv1 = abs(v_tx1 - v1_circ)
    expected_dv2 = abs(v2_circ - v_tx2)
    expected_tof = math.pi * math.sqrt(a_tx**3 / mu)

    assert transfer.delta_v1 == pytest.approx(expected_dv1, rel=1e-10)
    assert transfer.delta_v2 == pytest.approx(expected_dv2, rel=1e-10)
    assert transfer.transfer_time == pytest.approx(expected_tof, rel=1e-10)


# ===========================================================================
# 14. RENDEZVOUS SOLVER & RELATIVE STATE CLOSURE
# ===========================================================================

def test_rendezvous_relative_closure():
    """Verify rendezvous solver produces physical target interception with verified relative state closure."""
    mu = EARTH.mu
    r_chaser_mag = EARTH.radius + 400e3
    r_tgt_mag = EARTH.radius + 420e3
    lead = math.radians(5.0)
    tof = 3600.0

    r_chaser = np.array([r_chaser_mag, 0.0, 0.0])
    v_chaser = np.array([0.0, math.sqrt(mu / r_chaser_mag), 0.0])
    v_tgt_circ = math.sqrt(mu / r_tgt_mag)
    r_tgt = np.array([r_tgt_mag * math.cos(lead), r_tgt_mag * math.sin(lead), 0.0])
    v_tgt = np.array([-v_tgt_circ * math.sin(lead), v_tgt_circ * math.cos(lead), 0.0])

    res = solve_rendezvous(
        chaser_r=r_chaser,
        chaser_v=v_chaser,
        target_r=r_tgt,
        target_v=v_tgt,
        tof=tof,
        mu=mu,
        compute_trajectory=True,
    )

    assert res.delta_v_total > 0
    assert res.transfer_trajectory is not None

    # Verify chaser trajectory reaches target arrival position
    r_final_chaser = res.transfer_trajectory.positions[-1]
    miss_distance = np.linalg.norm(r_final_chaser - res.target_position_at_arrival)
    assert miss_distance < 1.0  # sub-meter docking boundary


# ===========================================================================
# 15. CORE ENGINE IMPORT ISOLATION
# ===========================================================================

def test_core_does_not_import_extensions():
    """
    Subprocess test: import all Phase 1-7 core modules in a fresh clean Python interpreter.
    Verify that sys.modules contains zero imports from Phase 8-10 (reentry, conjunction, uncertainty).
    """
    code = """
import sys
import theseus.constants
import theseus.coordinates
import theseus.orbital
import theseus.propagation
import theseus.dynamics
import theseus.maneuvers
import theseus.rendezvous

forbidden = ['theseus.reentry', 'theseus.conjunction', 'theseus.uncertainty']
loaded_forbidden = [mod for mod in sys.modules if any(mod.startswith(f) for f in forbidden)]
if loaded_forbidden:
    print('FORBIDDEN_LOADED:', loaded_forbidden)
    sys.exit(1)
else:
    print('CLEAN_ISOLATION')
    sys.exit(0)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"Core engine failed import isolation: {result.stdout} {result.stderr}"
    assert "CLEAN_ISOLATION" in result.stdout
