"""Tests for dynamics, ephemeris, gravity, spacecraft, thrust, atmosphere, drag, SRP."""

import math

import numpy as np
import pytest

from theseus.bodies.catalog import EARTH, MOON, SUN
from theseus.dynamics.force_model import CompositeForceModel
from theseus.dynamics.gravity import PointMassGravity, J2Perturbation, ThirdBodyGravity
from theseus.dynamics.thrust import ThrustModel, ThrustDirection
from theseus.dynamics.drag import DragModel, LiftModel
from theseus.dynamics.srp import SolarRadiationPressure
from theseus.ephemeris.simple_provider import SimpleEphemerisProvider
from theseus.atmosphere.models import ExponentialAtmosphere, US1976StandardAtmosphere
from theseus.spacecraft.vehicle import Spacecraft
from theseus.maneuvers.burns import impulsive_burn, fuel_for_delta_v, delta_v_from_fuel
from theseus.constants.physical import G0_VAL, AU_VAL


MU = EARTH.mu


# ===================================================================
# Ephemeris
# ===================================================================

class TestSimpleEphemeris:

    def test_earth_at_origin(self):
        eph = SimpleEphemerisProvider()
        # Earth orbits Sun at ~1 AU
        pos = eph.get_position("Earth", 2_451_545.0)
        r = np.linalg.norm(pos)
        assert r == pytest.approx(1.496e11, rel=0.05)  # ~1 AU

    def test_moon_distance(self):
        eph = SimpleEphemerisProvider()
        pos = eph.get_position("Moon", 2_451_545.0)
        r = np.linalg.norm(pos)
        assert r == pytest.approx(3.844e8, rel=0.01)  # ~384 400 km

    def test_sun_at_origin(self):
        eph = SimpleEphemerisProvider()
        pos = eph.get_position("Sun", 2_451_545.0)
        np.testing.assert_allclose(pos, [0, 0, 0])


# ===================================================================
# Gravity
# ===================================================================

class TestGravity:

    def test_point_mass_surface(self):
        """g at Earth surface ≈ 9.798 m/s² (equatorial, geometric)."""
        grav = PointMassGravity(EARTH)
        pos = np.array([EARTH.radius, 0.0, 0.0])
        a = grav.compute_acceleration(0.0, pos, np.zeros(3), 1.0)
        g = np.linalg.norm(a)
        assert g == pytest.approx(9.798, abs=0.05)

    def test_point_mass_inverse_square(self):
        """Gravity falls off as 1/r²."""
        grav = PointMassGravity(EARTH)
        a1 = np.linalg.norm(grav.compute_acceleration(0, np.array([7e6, 0, 0]), np.zeros(3), 1))
        a2 = np.linalg.norm(grav.compute_acceleration(0, np.array([14e6, 0, 0]), np.zeros(3), 1))
        assert a1 / a2 == pytest.approx(4.0, rel=1e-8)

    def test_j2_equatorial_vs_polar(self):
        """J2 perturbation should differ between equatorial and polar positions."""
        j2 = J2Perturbation(EARTH)
        r = EARTH.radius + 500_000.0  # 500 km altitude
        a_eq = j2.compute_acceleration(0, np.array([r, 0, 0]), np.zeros(3), 1)
        a_pol = j2.compute_acceleration(0, np.array([0, 0, r]), np.zeros(3), 1)
        # Equatorial and polar J2 accelerations should be different
        assert not np.allclose(a_eq, a_pol)
        # Both should be non-zero
        assert np.linalg.norm(a_eq) > 0
        assert np.linalg.norm(a_pol) > 0

    def test_j2_magnitude_order(self):
        """J2 acceleration should be ~1000x smaller than point-mass at LEO."""
        r = 7_000_000.0
        pos = np.array([r, 0, 0])
        pm = PointMassGravity(EARTH)
        j2 = J2Perturbation(EARTH)
        a_pm = np.linalg.norm(pm.compute_acceleration(0, pos, np.zeros(3), 1))
        a_j2 = np.linalg.norm(j2.compute_acceleration(0, pos, np.zeros(3), 1))
        ratio = a_j2 / a_pm
        # J2 ≈ 1e-3, so ratio should be O(1e-3)
        assert 1e-4 < ratio < 1e-2

    def test_composite_force_model(self):
        """Composite model sums contributions."""
        pm = PointMassGravity(EARTH)
        j2 = J2Perturbation(EARTH)
        composite = CompositeForceModel([pm, j2])
        pos = np.array([7_000_000.0, 0.0, 0.0])
        a_total = composite.compute_acceleration(0, pos, np.zeros(3), 1)
        a_pm = pm.compute_acceleration(0, pos, np.zeros(3), 1)
        a_j2 = j2.compute_acceleration(0, pos, np.zeros(3), 1)
        np.testing.assert_allclose(a_total, a_pm + a_j2, atol=1e-15)

    def test_disable_force(self):
        """Disabled models should not contribute."""
        pm = PointMassGravity(EARTH)
        j2 = J2Perturbation(EARTH, enabled=False)
        composite = CompositeForceModel([pm, j2])
        pos = np.array([7_000_000.0, 0.0, 0.0])
        a_total = composite.compute_acceleration(0, pos, np.zeros(3), 1)
        a_pm = pm.compute_acceleration(0, pos, np.zeros(3), 1)
        np.testing.assert_allclose(a_total, a_pm)


# ===================================================================
# Spacecraft & burns
# ===================================================================

class TestSpacecraft:

    def test_total_mass(self):
        sc = Spacecraft(dry_mass=500, fuel_mass=200)
        assert sc.total_mass == 700

    def test_delta_v_available(self):
        """Tsiolkovsky: Δv = v_e ln(m0/mf)."""
        sc = Spacecraft(dry_mass=500, fuel_mass=200, specific_impulse=300)
        ve = 300 * G0_VAL
        expected = ve * math.log(700 / 500)
        assert sc.delta_v_available() == pytest.approx(expected)

    def test_fuel_consumption(self):
        sc = Spacecraft(dry_mass=500, fuel_mass=200)
        consumed = sc.consume_fuel(50)
        assert consumed == 50
        assert sc.fuel_mass == 150

    def test_fuel_exhaustion(self):
        sc = Spacecraft(dry_mass=500, fuel_mass=100)
        consumed = sc.consume_fuel(150)
        assert consumed == 100
        assert sc.fuel_mass == 0


class TestBurns:

    def test_impulsive_burn_velocity(self):
        v0 = np.array([0.0, 7500.0, 0.0])
        dv = np.array([0.0, 500.0, 0.0])
        v_new, result = impulsive_burn(v0, dv, 700.0, 300.0)
        np.testing.assert_allclose(v_new, [0, 8000, 0])
        assert result.delta_v == pytest.approx(500.0)
        assert result.fuel_consumed > 0

    def test_fuel_for_delta_v_consistency(self):
        """fuel_for_delta_v and delta_v_from_fuel should be inverses."""
        mass = 1000.0
        isp = 300.0
        dv = 500.0
        fuel = fuel_for_delta_v(dv, mass, isp)
        dv_back = delta_v_from_fuel(fuel, mass, isp)
        assert dv_back == pytest.approx(dv, rel=1e-8)


# ===================================================================
# Atmosphere
# ===================================================================

class TestAtmosphere:

    def test_exponential_sea_level(self):
        atm = ExponentialAtmosphere()
        assert atm.density(0) == pytest.approx(1.225)

    def test_exponential_decreasing(self):
        atm = ExponentialAtmosphere()
        assert atm.density(10000) < atm.density(0)
        assert atm.density(50000) < atm.density(10000)

    def test_us76_sea_level(self):
        atm = US1976StandardAtmosphere()
        props = atm.get_properties(0)
        assert props.temperature == pytest.approx(288.15)
        assert props.pressure == pytest.approx(101325.0)
        assert props.density == pytest.approx(1.225, abs=0.01)

    def test_us76_tropopause(self):
        """At 11 km: T ≈ 216.65 K (isothermal layer begins)."""
        atm = US1976StandardAtmosphere()
        props = atm.get_properties(11000)
        assert props.temperature == pytest.approx(216.65, abs=1.0)

    def test_us76_density_decreases(self):
        atm = US1976StandardAtmosphere()
        d0 = atm.density(0)
        d10 = atm.density(10000)
        d50 = atm.density(50000)
        d80 = atm.density(80000)
        assert d0 > d10 > d50 > d80 > 0

    def test_us76_above_86km(self):
        """Above 86 km: should return small but positive density."""
        atm = US1976StandardAtmosphere()
        d = atm.density(100_000)
        assert d > 0
        assert d < 1e-5


# ===================================================================
# Drag
# ===================================================================

class TestDrag:

    def test_drag_proportional_to_density_and_v2(self):
        """F_D ∝ ρ v²."""
        atm = ExponentialAtmosphere()
        drag = DragModel(atm, cd=2.2, area=10.0)

        r = EARTH.radius + 400_000.0
        pos = np.array([r, 0.0, 0.0])
        v1 = np.array([0.0, 7500.0, 0.0])
        v2 = np.array([0.0, 15000.0, 0.0])

        a1 = np.linalg.norm(drag.compute_acceleration(0, pos, v1, 1000))
        a2 = np.linalg.norm(drag.compute_acceleration(0, pos, v2, 1000))

        # Should scale roughly as v², but atmospheric velocity subtraction
        # means it's not exactly 4x
        assert a2 > a1
        assert a2 / a1 > 3.0  # should be close to 4


# ===================================================================
# SRP
# ===================================================================

class TestSRP:

    def test_srp_inverse_square(self):
        """SRP should scale as 1/r² from Sun."""
        eph = SimpleEphemerisProvider()
        srp = SolarRadiationPressure(eph, cr=1.5, area=10.0, shadow_body_radius=0.0)

        # Spacecraft near Earth (approx 1 AU from Sun)
        pos_1au = np.array([7_000_000.0, 0.0, 0.0])
        a1 = np.linalg.norm(srp.compute_acceleration(0, pos_1au, np.zeros(3), 1000.0))

        # Nominal SRP acceleration at 1 AU: P = 4.54e-6 * 1.5 * 10 / 1000 = 6.81e-8 m/s^2
        assert a1 == pytest.approx(6.81e-8, rel=0.05)

    def test_srp_in_shadow_is_zero(self):
        """SRP should be zero when in Earth's shadow."""
        eph = SimpleEphemerisProvider()
        srp = SolarRadiationPressure(
            eph, cr=1.5, area=10.0,
            shadow_body_radius=EARTH.radius,
        )
        # Geocentric Sun position
        sun_pos = srp._get_geocentric_sun_position(2_451_545.0)
        # Anti-Sun direction from Earth
        anti_sun = -sun_pos / np.linalg.norm(sun_pos)
        pos = anti_sun * (EARTH.radius + 100_000.0)  # 100 km behind Earth in shadow
        a = srp.compute_acceleration(0, pos, np.zeros(3), 100)
        np.testing.assert_allclose(a, [0, 0, 0])


# ===================================================================
# End-to-End Finite Burn Propagation
# ===================================================================

class TestFiniteBurnPropagation:

    def test_7dof_continuous_burn_depletion(self):
        """Verify 7-DOF numerical propagation depletes propellant according to rocket equation."""
        from theseus.propagation.numerical import NumericalPropagator

        sc = Spacecraft(
            name="BurnSat",
            dry_mass=1000.0,
            fuel_mass=500.0,
            specific_impulse=300.0,
            thrust=100.0,
        )
        thrust_mod = ThrustModel(
            spacecraft=sc,
            direction=ThrustDirection.PROGRADE,
            burn_start=0.0,
            burn_end=500.0,
            throttle=1.0,
        )
        grav = PointMassGravity(EARTH)
        force_model = CompositeForceModel([grav, thrust_mod])

        prop = NumericalPropagator(
            acceleration_fn=force_model.compute_acceleration,
            integrator="rk4",
            dt=10.0,
            mu=EARTH.mu,
        )

        r0 = np.array([EARTH.radius + 500_000.0, 0.0, 0.0])
        v0 = np.array([0.0, math.sqrt(EARTH.mu / np.linalg.norm(r0)), 0.0])

        history, events, diag = prop.propagate(
            r0, v0, (0.0, 500.0),
            mass=sc.total_mass,
            fuel_mass=sc.fuel_mass,
        )

        # Expected fuel consumption: mdot * delta_t = (F / (Isp * g0)) * 500
        expected_mdot = 100.0 / (300.0 * G0_VAL)
        expected_fuel_consumed = expected_mdot * 500.0
        expected_final_fuel = 500.0 - expected_fuel_consumed
        expected_final_mass = 1000.0 + expected_final_fuel

        final_state = history[-1]
        assert final_state.fuel_mass == pytest.approx(expected_final_fuel, rel=1e-3)
        assert final_state.mass == pytest.approx(expected_final_mass, rel=1e-3)
        assert final_state.fuel_mass > 0.0
        assert final_state.mass > sc.dry_mass
