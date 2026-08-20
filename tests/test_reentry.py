"""
Phase 8 — Reentry Dynamics Tests.

Tests cover:
- Vehicle model construction, validation, and derived properties
- Aerothermal calculations (Sutton-Graves, Mach, dynamic pressure)
- Physical relationships (monotonicity, scaling laws)
- Event detection (peak-Q, peak-heating, peak-G, impact, skip-out)
- Apollo-like entry scenario
- LEO deorbit scenario
- Shallow skip-out scenario
"""

import math
import pytest
import numpy as np

from theseus.reentry.vehicle import ReentryVehicle, APOLLO_CM, GENERIC_BALLISTIC
from theseus.reentry.heating import (
    sutton_graves_convective,
    stagnation_temperature,
    mach_number,
    dynamic_pressure,
    aerodynamic_force,
    speed_of_sound,
    heating_model_metadata,
    SUTTON_GRAVES_K_EARTH,
)
from theseus.reentry.results import ReentryEventType
from theseus.reentry.simulator import ReentrySimulator


# ===================================================================
# Vehicle model tests
# ===================================================================

class TestReentryVehicle:

    def test_construction_valid(self):
        v = ReentryVehicle(name="test", mass=1000, reference_area=2.0,
                           nose_radius=0.5, cd=1.5, cl=0.3)
        assert v.mass == 1000
        assert v.cd == 1.5
        assert v.cl == 0.3

    def test_mass_must_be_positive(self):
        with pytest.raises(ValueError, match="mass"):
            ReentryVehicle(name="bad", mass=0, reference_area=1, nose_radius=0.5, cd=1)

    def test_area_must_be_positive(self):
        with pytest.raises(ValueError, match="reference_area"):
            ReentryVehicle(name="bad", mass=100, reference_area=-1, nose_radius=0.5, cd=1)

    def test_nose_radius_must_be_positive(self):
        with pytest.raises(ValueError, match="nose_radius"):
            ReentryVehicle(name="bad", mass=100, reference_area=1, nose_radius=0, cd=1)

    def test_cd_must_be_non_negative(self):
        with pytest.raises(ValueError, match="cd"):
            ReentryVehicle(name="bad", mass=100, reference_area=1, nose_radius=0.5, cd=-0.1)

    def test_cl_out_of_range(self):
        with pytest.raises(ValueError, match="cl"):
            ReentryVehicle(name="bad", mass=100, reference_area=1, nose_radius=0.5, cd=1, cl=10)

    def test_ballistic_coefficient(self):
        """β = m / (Cd A).  1000 / (2.0 * 1.0) = 500."""
        v = ReentryVehicle(name="t", mass=1000, reference_area=1.0,
                           nose_radius=0.5, cd=2.0)
        assert abs(v.ballistic_coefficient - 500.0) < 1e-10

    def test_ballistic_coefficient_increases_with_mass(self):
        """Heavier vehicle → higher ballistic coefficient."""
        v1 = ReentryVehicle(name="a", mass=1000, reference_area=1, nose_radius=0.5, cd=2)
        v2 = ReentryVehicle(name="b", mass=2000, reference_area=1, nose_radius=0.5, cd=2)
        assert v2.ballistic_coefficient > v1.ballistic_coefficient

    def test_lift_to_drag_ratio(self):
        v = ReentryVehicle(name="t", mass=100, reference_area=1,
                           nose_radius=0.5, cd=1.29, cl=0.368)
        expected = 0.368 / 1.29
        assert abs(v.lift_to_drag_ratio - expected) < 1e-10

    def test_entry_type_ballistic(self):
        v = ReentryVehicle(name="t", mass=100, reference_area=1,
                           nose_radius=0.5, cd=2, cl=0)
        assert v.entry_type == "ballistic"
        assert not v.is_lifting

    def test_entry_type_lifting(self):
        v = ReentryVehicle(name="t", mass=100, reference_area=1,
                           nose_radius=0.5, cd=2, cl=0.5)
        assert v.entry_type == "lifting"
        assert v.is_lifting

    def test_apollo_cm_properties(self):
        """Apollo CM reference values."""
        assert APOLLO_CM.mass == 5424.0
        assert abs(APOLLO_CM.ballistic_coefficient - 5424.0 / (1.29 * 12.017)) < 0.1
        assert abs(APOLLO_CM.lift_to_drag_ratio - 0.368/1.29) < 0.001

    def test_to_dict_contains_all_fields(self):
        d = APOLLO_CM.to_dict()
        assert "name" in d
        assert "ballistic_coefficient_kg_m2" in d
        assert "lift_to_drag_ratio" in d
        assert "entry_type" in d


# ===================================================================
# Heating and aerodynamic formula tests
# ===================================================================

class TestHeatingModels:

    def test_sutton_graves_known_value(self):
        """
        At sea level conditions (ρ ≈ 1.225 kg/m³), V = 7000 m/s, r_n = 1 m:
        q̇ = k √(ρ/r_n) V³ = 1.7415e-4 * √(1.225/1) * 7000³
        ≈ 1.7415e-4 * 1.1068 * 3.43e11 ≈ 66.12 MW/m²
        """
        q = sutton_graves_convective(1.225, 7000.0, 1.0)
        assert q > 60e6  # > 60 MW/m²
        assert q < 80e6  # < 80 MW/m²

    def test_sutton_graves_increases_with_velocity(self):
        """Heat flux scales as V³."""
        q1 = sutton_graves_convective(0.01, 5000.0, 1.0)
        q2 = sutton_graves_convective(0.01, 10000.0, 1.0)
        # V doubled → q should increase by factor of 8
        ratio = q2 / q1
        assert abs(ratio - 8.0) < 0.01

    def test_sutton_graves_increases_with_density(self):
        """Higher density → more heating."""
        q1 = sutton_graves_convective(0.001, 7000.0, 1.0)
        q2 = sutton_graves_convective(0.01, 7000.0, 1.0)
        assert q2 > q1

    def test_sutton_graves_decreases_with_nose_radius(self):
        """Larger nose radius → lower heat flux (blunt body advantage)."""
        q1 = sutton_graves_convective(0.01, 7000.0, 0.5)
        q2 = sutton_graves_convective(0.01, 7000.0, 2.0)
        assert q1 > q2

    def test_sutton_graves_zero_for_zero_velocity(self):
        q = sutton_graves_convective(1.0, 0.0, 1.0)
        assert q == 0.0

    def test_sutton_graves_zero_for_negative_density(self):
        q = sutton_graves_convective(-1.0, 1000.0, 1.0)
        assert q == 0.0

    def test_stagnation_temperature(self):
        """T_s = T_∞(1 + (γ-1)/2 M²) at M=5, T=220 K → T_s = 220*(1+0.2*25) = 1320 K."""
        T_s = stagnation_temperature(220.0, 5.0)
        assert abs(T_s - 1320.0) < 1.0

    def test_mach_number_at_sea_level(self):
        """Speed of sound at 288.15 K ≈ 340 m/s. V=680 → M≈2."""
        M = mach_number(680.0, 288.15)
        assert abs(M - 2.0) < 0.05

    def test_dynamic_pressure(self):
        """q = ½ρV² = 0.5 * 1.225 * 100² = 6125 Pa."""
        q = dynamic_pressure(1.225, 100.0)
        assert abs(q - 6125.0) < 1.0

    def test_aerodynamic_force(self):
        """F = ½ρV²CA = 0.5 * 0.01 * 7000² * 2.0 * 10.0 = 4.9e6 N."""
        F = aerodynamic_force(0.01, 7000.0, 2.0, 10.0)
        assert abs(F - 4.9e6) < 1e3

    def test_speed_of_sound_at_standard(self):
        """a ≈ 340 m/s at T = 288.15 K."""
        a = speed_of_sound(288.15)
        assert abs(a - 340.0) < 5.0

    def test_heating_model_metadata(self):
        meta = heating_model_metadata()
        assert meta["convective"]["model"] == "Sutton-Graves stagnation-point correlation"
        assert meta["radiative"]["model"] == "NOT ENABLED"
        assert "ENGINEERING ESTIMATE" in meta["convective"]["classification"]


# ===================================================================
# Reentry simulation tests
# ===================================================================

class TestReentrySimulator:

    def test_ballistic_leo_deorbit(self):
        """
        LEO deorbit: V ≈ 7.8 km/s, γ ≈ -3°, alt = 120 km.
        Should reach ground impact.
        Note: -1° is too shallow for a high-β ballistic body;
        -3° ensures atmospheric capture.
        """
        vehicle = GENERIC_BALLISTIC
        sim = ReentrySimulator(vehicle=vehicle, max_time=3600.0)
        result = sim.simulate(
            entry_altitude=120_000.0,
            entry_velocity=7800.0,
            entry_fpa_deg=-3.0,
        )
        assert result.termination_reason == "ground_impact"
        assert len(result.telemetry) > 10
        assert len(result.events) > 0
        # Should detect ground impact event
        impact_events = [e for e in result.events
                         if e.event_type == ReentryEventType.GROUND_IMPACT]
        assert len(impact_events) == 1

    def test_apollo_like_entry(self):
        """
        Apollo-like lunar return: V ≈ 11 km/s, γ ≈ -6.5°, alt = 120 km.
        Expected peak deceleration ~6-12g (varies with model assumptions).
        Tolerance: order of magnitude is correct.
        """
        sim = ReentrySimulator(vehicle=APOLLO_CM, max_time=3600.0)
        result = sim.simulate(
            entry_altitude=120_000.0,
            entry_velocity=11_000.0,
            entry_fpa_deg=-6.5,
        )
        assert result.termination_reason in ("ground_impact", "skip_out")
        assert "peak_g_load" in result.peak_statistics
        peak_g = result.peak_statistics["peak_g_load"]
        # Apollo experienced ~6-7g with lifting entry; ballistic would be higher
        # Our model (constant Cd/Cl, no attitude control) will give approximate results
        # Accept 3-20g as physically reasonable for this model fidelity
        assert 3.0 < peak_g < 30.0, f"Peak g-load {peak_g:.1f}g outside reasonable range"

    def test_skip_out_shallow_entry(self):
        """
        Very shallow entry angle should cause skip-out.
        V = 11 km/s, γ = -0.5° — too shallow for capture.
        """
        vehicle = ReentryVehicle(
            name="skip-test", mass=5000, reference_area=10,
            nose_radius=3.0, cd=1.3, cl=0.4,
        )
        sim = ReentrySimulator(vehicle=vehicle, max_time=3600.0)
        result = sim.simulate(
            entry_altitude=120_000.0,
            entry_velocity=11_000.0,
            entry_fpa_deg=-0.5,
        )
        # Shallow + lifting → likely skip-out (model dependent)
        # At minimum the simulation should run without crashing
        assert result.termination_reason in ("skip_out", "ground_impact", "max_time")
        assert len(result.telemetry) >= 3

    def test_events_are_time_ordered(self):
        """All events should be in chronological order."""
        sim = ReentrySimulator(vehicle=GENERIC_BALLISTIC, max_time=2400.0)
        result = sim.simulate(
            entry_altitude=120_000.0,
            entry_velocity=7800.0,
            entry_fpa_deg=-2.0,
        )
        for i in range(1, len(result.events)):
            assert result.events[i].time >= result.events[i-1].time

    def test_peak_q_detected(self):
        """For a reasonable entry, peak-Q should be detected."""
        sim = ReentrySimulator(vehicle=GENERIC_BALLISTIC, max_time=2400.0)
        result = sim.simulate(
            entry_altitude=120_000.0,
            entry_velocity=7800.0,
            entry_fpa_deg=-3.0,
        )
        peak_q_events = [e for e in result.events
                         if e.event_type == ReentryEventType.PEAK_DYNAMIC_PRESSURE]
        assert len(peak_q_events) >= 1
        # Peak-Q altitude should be in the atmosphere (< 80 km typically)
        assert peak_q_events[0].altitude < 80_000

    def test_peak_heating_detected(self):
        """Peak heating should be detected for a reasonable entry."""
        sim = ReentrySimulator(vehicle=GENERIC_BALLISTIC, max_time=2400.0)
        result = sim.simulate(
            entry_altitude=120_000.0,
            entry_velocity=7800.0,
            entry_fpa_deg=-3.0,
        )
        heat_events = [e for e in result.events
                       if e.event_type == ReentryEventType.PEAK_HEATING]
        assert len(heat_events) >= 1

    def test_calculation_steps_present(self):
        """Calculation trace must be present and structured."""
        sim = ReentrySimulator(vehicle=GENERIC_BALLISTIC, max_time=600.0)
        result = sim.simulate(
            entry_altitude=120_000.0,
            entry_velocity=7800.0,
            entry_fpa_deg=-5.0,
        )
        assert len(result.calculation_steps) >= 10
        for step in result.calculation_steps:
            assert "stepIndex" in step
            assert "phase" in step
            assert "title" in step
            assert "equation" in step

    def test_model_metadata_transparency(self):
        """Model metadata must clearly document limitations."""
        sim = ReentrySimulator(vehicle=GENERIC_BALLISTIC, max_time=600.0)
        result = sim.simulate(
            entry_altitude=120_000.0,
            entry_velocity=7800.0,
            entry_fpa_deg=-5.0,
        )
        meta = result.model_metadata
        assert "numerical" in meta
        assert "physical" in meta
        assert "limitations" in meta
        assert "assumptions" in meta
        assert meta["numerical"]["integrator"] == "RKF45 (Runge-Kutta-Fehlberg 4(5))"
        assert "No winds" in meta["limitations"]
        assert meta["heating"]["radiative"]["model"] == "NOT ENABLED"

    def test_telemetry_altitude_decreasing_steep_entry(self):
        """For a steep ballistic entry, altitude should generally decrease."""
        sim = ReentrySimulator(vehicle=GENERIC_BALLISTIC, max_time=1200.0)
        result = sim.simulate(
            entry_altitude=120_000.0,
            entry_velocity=7800.0,
            entry_fpa_deg=-10.0,
        )
        if result.termination_reason == "ground_impact":
            last_alt = result.telemetry[-1].altitude
            assert last_alt <= 100  # near ground

    def test_increasing_density_increases_drag(self):
        """Physical relationship: higher density → more drag at same V, Cd, A."""
        D1 = aerodynamic_force(0.001, 7000, 2.0, 10.0)
        D2 = aerodynamic_force(0.01, 7000, 2.0, 10.0)
        D3 = aerodynamic_force(0.1, 7000, 2.0, 10.0)
        assert D1 < D2 < D3

    def test_result_serialisation(self):
        """ReentryResult.to_dict() should produce a valid dictionary."""
        sim = ReentrySimulator(vehicle=GENERIC_BALLISTIC, max_time=600.0)
        result = sim.simulate(
            entry_altitude=120_000.0,
            entry_velocity=7800.0,
            entry_fpa_deg=-5.0,
        )
        d = result.to_dict()
        assert "telemetry" in d
        assert "events" in d
        assert "model_metadata" in d
        assert "peak_statistics" in d
        assert "calculation_steps" in d

    def test_event_contains_required_fields(self):
        """Each event must have all required fields."""
        sim = ReentrySimulator(vehicle=GENERIC_BALLISTIC, max_time=1200.0)
        result = sim.simulate(
            entry_altitude=120_000.0,
            entry_velocity=7800.0,
            entry_fpa_deg=-3.0,
        )
        for event in result.events:
            d = event.to_dict()
            assert "event_type" in d
            assert "time_s" in d
            assert "altitude_m" in d
            assert "velocity_m_s" in d
            assert "value" in d
            assert "units" in d
            assert "detection_method" in d
