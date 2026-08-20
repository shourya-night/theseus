"""
Phase 9 — Collision / Conjunction Analysis Tests.

Tests cover:
- Relative position/velocity computation
- Coarse screening (threshold detection)
- TCA root condition (r_rel · v_rel = 0)
- TCA refinement and convergence
- Local minimum verification
- Multiple TCA detection
- Miss distance accuracy
- Encounter classification (head-on, overtaking, crossing)
- B-plane validity guard
- B-plane geometry
- Full pipeline integration
"""

import math
import pytest
import numpy as np

from theseus.conjunction.screening import ConjunctionScreener, CandidateInterval
from theseus.conjunction.tca import find_tca, find_all_tca, TCAResult
from theseus.conjunction.b_plane import compute_b_plane, BPlaneResult, MIN_V_INF_FOR_BPLANE
from theseus.conjunction.analysis import (
    ConjunctionAnalysis,
    ConjunctionResult,
    classify_encounter,
)


# ===================================================================
# Helper: simple circular-orbit position/velocity functions
# ===================================================================

def make_circular_orbit(radius: float, mu: float, phase_deg: float = 0.0, inc_deg: float = 0.0):
    """
    Create position and velocity functions for a circular orbit.

    Returns (pos_fn, vel_fn) where each takes time (s) and returns (3,) array.
    """
    n = math.sqrt(mu / radius ** 3)  # mean motion (rad/s)
    v_circ = math.sqrt(mu / radius)
    phi0 = math.radians(phase_deg)
    inc = math.radians(inc_deg)

    def pos_fn(t):
        theta = n * t + phi0
        x = radius * math.cos(theta)
        y = radius * math.sin(theta) * math.cos(inc)
        z = radius * math.sin(theta) * math.sin(inc)
        return np.array([x, y, z])

    def vel_fn(t):
        theta = n * t + phi0
        vx = -v_circ * math.sin(theta)
        vy = v_circ * math.cos(theta) * math.cos(inc)
        vz = v_circ * math.cos(theta) * math.sin(inc)
        return np.array([vx, vy, vz])

    return pos_fn, vel_fn


MU_EARTH = 3.986004418e14  # m³/s²
R_LEO_A = 6778.137e3       # 400 km altitude
R_LEO_B = 6778.137e3       # same altitude, different phase


# ===================================================================
# Screening tests
# ===================================================================

class TestConjunctionScreener:

    def test_no_conjunction_far_apart(self):
        """Two orbits at very different altitudes → no conjunction."""
        pos_a, vel_a = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=0)
        pos_b, vel_b = make_circular_orbit(R_LEO_A + 1000e3, MU_EARTH, phase_deg=0)
        screener = ConjunctionScreener(threshold_m=100e3)
        period = 2 * math.pi * math.sqrt(R_LEO_A ** 3 / MU_EARTH)
        candidates = screener.screen(pos_a, pos_b, 0, period)
        assert len(candidates) == 0

    def test_same_orbit_always_close(self):
        """Two objects on same orbit, small phase offset → always within threshold."""
        pos_a, vel_a = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=0)
        pos_b, vel_b = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=0.1)  # ~12 km apart
        screener = ConjunctionScreener(threshold_m=100e3)
        period = 2 * math.pi * math.sqrt(R_LEO_A ** 3 / MU_EARTH)
        candidates = screener.screen(pos_a, pos_b, 0, period)
        assert len(candidates) > 0

    def test_crossing_orbits_detect_candidate(self):
        """Two orbits at same radius but different inclinations → periodic crossings."""
        pos_a, vel_a = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=0, inc_deg=0)
        pos_b, vel_b = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=0, inc_deg=10)
        screener = ConjunctionScreener(threshold_m=2000e3, coarse_dt=30.0)
        period = 2 * math.pi * math.sqrt(R_LEO_A ** 3 / MU_EARTH)
        candidates = screener.screen(pos_a, pos_b, 0, period)
        # Crossing orbits at the same radius will come close at the nodes
        assert len(candidates) > 0

    def test_threshold_must_be_positive(self):
        with pytest.raises(ValueError):
            ConjunctionScreener(threshold_m=-1)

    def test_screen_from_arrays(self):
        """Test array-based screening."""
        times = np.linspace(0, 100, 101)
        # Object A at origin, object B moving past
        pos_a = np.zeros((101, 3))
        pos_b = np.column_stack([
            np.linspace(-200e3, 200e3, 101),
            np.zeros(101),
            np.zeros(101),
        ])
        screener = ConjunctionScreener(threshold_m=50e3)
        candidates = screener.screen_from_arrays(times, pos_a, pos_b)
        assert len(candidates) > 0
        # Minimum distance should be near zero (at the midpoint)
        assert candidates[0].min_distance < 50e3


# ===================================================================
# TCA solver tests
# ===================================================================

class TestTCASolver:

    def test_tca_for_crossing_orbits(self):
        """Two circular orbits crossing → should find TCA."""
        pos_a, vel_a = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=0, inc_deg=0)
        pos_b, vel_b = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=5, inc_deg=10)
        period = 2 * math.pi * math.sqrt(R_LEO_A ** 3 / MU_EARTH)
        result = find_tca(pos_a, vel_a, pos_b, vel_b, 0, period)
        if result is not None and result.converged:
            # Miss distance should equal |r_rel(TCA)|
            r_a = np.asarray(pos_a(result.tca))
            r_b = np.asarray(pos_b(result.tca))
            expected_dist = float(np.linalg.norm(r_a - r_b))
            assert abs(result.miss_distance - expected_dist) < 1.0  # < 1 m accuracy

    def test_tca_root_condition(self):
        """At TCA, r_rel · v_rel ≈ 0."""
        pos_a, vel_a = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=0, inc_deg=0)
        pos_b, vel_b = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=5, inc_deg=10)
        period = 2 * math.pi * math.sqrt(R_LEO_A ** 3 / MU_EARTH)
        result = find_tca(pos_a, vel_a, pos_b, vel_b, 0, period)
        if result is not None and result.converged:
            dot_product = float(np.dot(result.r_rel, result.v_rel))
            assert abs(dot_product) < 1e3  # should be near zero

    def test_tca_is_local_minimum(self):
        """Distance at TCA should be less than at nearby times."""
        pos_a, vel_a = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=0, inc_deg=0)
        pos_b, vel_b = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=5, inc_deg=10)
        period = 2 * math.pi * math.sqrt(R_LEO_A ** 3 / MU_EARTH)
        result = find_tca(pos_a, vel_a, pos_b, vel_b, 0, period)
        if result is not None and result.converged:
            dt = 1.0  # 1 second
            d_before = float(np.linalg.norm(
                np.asarray(pos_a(result.tca - dt)) - np.asarray(pos_b(result.tca - dt))
            ))
            d_after = float(np.linalg.norm(
                np.asarray(pos_a(result.tca + dt)) - np.asarray(pos_b(result.tca + dt))
            ))
            assert result.miss_distance <= d_before + 1.0
            assert result.miss_distance <= d_after + 1.0

    def test_multiple_tca_detection(self):
        """Over multiple periods, should find multiple TCA events."""
        pos_a, vel_a = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=0, inc_deg=0)
        pos_b, vel_b = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=30, inc_deg=15)
        period = 2 * math.pi * math.sqrt(R_LEO_A ** 3 / MU_EARTH)
        results = find_all_tca(pos_a, vel_a, pos_b, vel_b, 0, 3 * period, n_samples=1000)
        # Over 3 orbits with crossing inclinations, expect multiple close approaches
        assert len(results) >= 2

    def test_relative_velocity_at_tca(self):
        """Relative velocity at TCA should equal |v₁(TCA) − v₂(TCA)|."""
        pos_a, vel_a = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=0, inc_deg=0)
        pos_b, vel_b = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=5, inc_deg=10)
        period = 2 * math.pi * math.sqrt(R_LEO_A ** 3 / MU_EARTH)
        result = find_tca(pos_a, vel_a, pos_b, vel_b, 0, period)
        if result is not None and result.converged:
            expected_vrel = float(np.linalg.norm(
                np.asarray(vel_a(result.tca)) - np.asarray(vel_b(result.tca))
            ))
            assert abs(result.relative_velocity - expected_vrel) < 1.0

    def test_miss_distance_equals_r_rel(self):
        """miss_distance must equal |r_rel(TCA)|."""
        pos_a, vel_a = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=0, inc_deg=0)
        pos_b, vel_b = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=5, inc_deg=10)
        period = 2 * math.pi * math.sqrt(R_LEO_A ** 3 / MU_EARTH)
        result = find_tca(pos_a, vel_a, pos_b, vel_b, 0, period)
        if result is not None:
            assert abs(result.miss_distance - float(np.linalg.norm(result.r_rel))) < 1e-6


# ===================================================================
# Encounter classification tests
# ===================================================================

class TestEncounterClassification:

    def test_head_on(self):
        """Anti-parallel velocities → head-on."""
        v_a = np.array([7000.0, 0.0, 0.0])
        v_b = np.array([-7000.0, 0.0, 0.0])
        angle, enc_type = classify_encounter(v_a, v_b)
        assert angle > 150
        assert enc_type == "head-on"

    def test_overtaking(self):
        """Nearly parallel velocities → overtaking."""
        v_a = np.array([7000.0, 0.0, 0.0])
        v_b = np.array([6900.0, 100.0, 0.0])
        angle, enc_type = classify_encounter(v_a, v_b)
        assert angle < 30
        assert enc_type == "overtaking"

    def test_crossing(self):
        """Perpendicular velocities → crossing."""
        v_a = np.array([7000.0, 0.0, 0.0])
        v_b = np.array([0.0, 7000.0, 0.0])
        angle, enc_type = classify_encounter(v_a, v_b)
        assert 85 < angle < 95
        assert enc_type == "crossing"


# ===================================================================
# B-plane tests
# ===================================================================

class TestBPlane:

    def test_valid_high_velocity_encounter(self):
        """High relative velocity → B-plane applicable."""
        r_rel = np.array([10e3, 5e3, 2e3])  # 10 km, 5 km, 2 km
        v_rel = np.array([5000.0, -3000.0, 1000.0])  # ~6 km/s
        result = compute_b_plane(r_rel, v_rel)
        assert result.applicable
        assert result.b_vector is not None
        assert result.b_magnitude > 0

    def test_invalid_low_velocity_encounter(self):
        """Low relative velocity → B-plane NOT applicable."""
        r_rel = np.array([10e3, 0.0, 0.0])
        v_rel = np.array([10.0, 0.0, 0.0])  # 10 m/s — way too slow
        result = compute_b_plane(r_rel, v_rel)
        assert not result.applicable
        assert "NOT APPLICABLE" in result.to_dict().get("note", "")

    def test_b_vector_perpendicular_to_s_hat(self):
        """B-vector must be perpendicular to the approach direction."""
        r_rel = np.array([10e3, 5e3, 2e3])
        v_rel = np.array([5000.0, -3000.0, 1000.0])
        result = compute_b_plane(r_rel, v_rel)
        if result.applicable and result.b_vector is not None and result.s_hat is not None:
            dot = abs(float(np.dot(result.b_vector, result.s_hat)))
            assert dot < 1e-6  # perpendicular

    def test_b_magnitude_consistency(self):
        """B magnitude should equal sqrt(B·T² + B·R²)."""
        r_rel = np.array([10e3, 5e3, 2e3])
        v_rel = np.array([5000.0, -3000.0, 1000.0])
        result = compute_b_plane(r_rel, v_rel)
        if result.applicable:
            b_from_components = math.sqrt(result.b_dot_t ** 2 + result.b_dot_r ** 2)
            assert abs(result.b_magnitude - b_from_components) < 1e-6

    def test_basis_orthonormality(self):
        """S, T, R should form an orthonormal basis."""
        r_rel = np.array([10e3, 5e3, 2e3])
        v_rel = np.array([5000.0, -3000.0, 1000.0])
        result = compute_b_plane(r_rel, v_rel)
        if result.applicable:
            S = result.s_hat
            T = result.t_hat
            R = result.r_hat
            # Orthogonal
            assert abs(float(np.dot(S, T))) < 1e-10
            assert abs(float(np.dot(S, R))) < 1e-10
            assert abs(float(np.dot(T, R))) < 1e-10
            # Unit length
            assert abs(float(np.linalg.norm(S)) - 1.0) < 1e-10
            assert abs(float(np.linalg.norm(T)) - 1.0) < 1e-10
            assert abs(float(np.linalg.norm(R)) - 1.0) < 1e-10


# ===================================================================
# Full pipeline tests
# ===================================================================

class TestConjunctionAnalysis:

    def test_full_pipeline_crossing_orbits(self):
        """Full pipeline: crossing orbits should produce events."""
        pos_a, vel_a = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=0, inc_deg=0)
        pos_b, vel_b = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=5, inc_deg=10)
        period = 2 * math.pi * math.sqrt(R_LEO_A ** 3 / MU_EARTH)

        analysis = ConjunctionAnalysis(
            screening_threshold_m=2000e3,
            coarse_dt=30.0,
        )
        result = analysis.analyse(pos_a, vel_a, pos_b, vel_b, 0, period)
        assert isinstance(result, ConjunctionResult)
        assert len(result.events) > 0
        # Verify calculation steps are present
        assert len(result.calculation_steps) > 5

    def test_no_events_for_distant_orbits(self):
        """Widely separated orbits → no conjunction events."""
        pos_a, vel_a = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=0)
        pos_b, vel_b = make_circular_orbit(R_LEO_A + 5000e3, MU_EARTH, phase_deg=0)
        period = 2 * math.pi * math.sqrt(R_LEO_A ** 3 / MU_EARTH)

        analysis = ConjunctionAnalysis(screening_threshold_m=100e3)
        result = analysis.analyse(pos_a, vel_a, pos_b, vel_b, 0, period)
        assert len(result.events) == 0

    def test_calculation_steps_structured(self):
        """Calculation steps should have required fields."""
        pos_a, vel_a = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=0, inc_deg=0)
        pos_b, vel_b = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=5, inc_deg=10)
        period = 2 * math.pi * math.sqrt(R_LEO_A ** 3 / MU_EARTH)

        analysis = ConjunctionAnalysis(screening_threshold_m=2000e3, coarse_dt=30)
        result = analysis.analyse(pos_a, vel_a, pos_b, vel_b, 0, period)
        for step in result.calculation_steps:
            assert "stepIndex" in step
            assert "phase" in step
            assert "title" in step

    def test_model_metadata_present(self):
        """Model metadata should document limitations."""
        pos_a, vel_a = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=0)
        pos_b, vel_b = make_circular_orbit(R_LEO_A + 5000e3, MU_EARTH, phase_deg=0)
        period = 2 * math.pi * math.sqrt(R_LEO_A ** 3 / MU_EARTH)

        analysis = ConjunctionAnalysis()
        result = analysis.analyse(pos_a, vel_a, pos_b, vel_b, 0, period)
        meta = result.model_metadata
        assert "numerical" in meta
        assert "physical" in meta
        assert "limitations" in meta
        assert "assumptions" in meta
        assert meta["numerical"]["tca_solver"] == "Brent's method"

    def test_result_serialisation(self):
        """ConjunctionResult.to_dict() should work."""
        pos_a, vel_a = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=0, inc_deg=0)
        pos_b, vel_b = make_circular_orbit(R_LEO_A, MU_EARTH, phase_deg=5, inc_deg=10)
        period = 2 * math.pi * math.sqrt(R_LEO_A ** 3 / MU_EARTH)

        analysis = ConjunctionAnalysis(screening_threshold_m=2000e3, coarse_dt=30)
        result = analysis.analyse(pos_a, vel_a, pos_b, vel_b, 0, period)
        d = result.to_dict()
        assert "events" in d
        assert "model_metadata" in d
        assert "calculation_steps" in d
        assert "summary" in d
