"""Tests for core infrastructure: state, events, fidelity, diagnostics, health, trace."""

import math

import numpy as np
import pytest

from theseus.core.state import SimulationState, StateHistory
from theseus.core.events import EventType, SimulationEvent, EventLog
from theseus.core.fidelity import (
    FidelityLevel, Assumption, ModelFidelity, FidelityRegistry,
)
from theseus.core.trace import CalculationTrace, TraceContext
from theseus.core.health import NumericalHealthChecker, NumericalInstabilityError
from theseus.core.diagnostics import ConservationDiagnostics


# ===================================================================
# SimulationState
# ===================================================================

class TestSimulationState:

    def test_basic_construction(self):
        s = SimulationState(
            time=0.0,
            position=np.array([7000e3, 0.0, 0.0]),
            velocity=np.array([0.0, 7500.0, 0.0]),
            mass=1000.0,
        )
        assert s.speed == pytest.approx(7500.0)
        assert s.r_mag == pytest.approx(7000e3)

    def test_to_dict(self):
        s = SimulationState(
            time=10.0,
            position=np.array([1.0, 2.0, 3.0]),
            velocity=np.array([4.0, 5.0, 6.0]),
        )
        d = s.to_dict()
        assert d["time"] == 10.0
        assert len(d["position"]) == 3

    def test_metadata(self):
        s = SimulationState(
            time=0.0,
            position=np.zeros(3),
            velocity=np.zeros(3),
            metadata={"altitude": 400_000.0},
        )
        assert s.altitude == 400_000.0


class TestStateHistory:

    def test_append_and_access(self):
        h = StateHistory()
        for i in range(5):
            h.append(SimulationState(
                time=float(i),
                position=np.array([float(i), 0, 0]),
                velocity=np.zeros(3),
            ))
        assert len(h) == 5
        assert h[0].time == 0.0
        assert h[4].time == 4.0

    def test_times_array(self):
        h = StateHistory()
        for t in [0.0, 1.0, 2.0]:
            h.append(SimulationState(time=t, position=np.zeros(3), velocity=np.zeros(3)))
        np.testing.assert_allclose(h.times, [0, 1, 2])


# ===================================================================
# Events
# ===================================================================

class TestEvents:

    def test_event_creation(self):
        e = SimulationEvent(time=100.0, event_type=EventType.BURN_START, description="Main engine ignition")
        assert e.event_type == EventType.BURN_START
        assert e.to_dict()["event_type"] == "BURN_START"

    def test_event_log(self):
        log = EventLog()
        log.emit(0.0, EventType.INITIALIZATION, "Sim start")
        log.emit(100.0, EventType.BURN_START, "Burn 1")
        log.emit(200.0, EventType.BURN_END, "Burn 1 end")
        assert len(log) == 3
        burns = log.filter(EventType.BURN_START)
        assert len(burns) == 1


# ===================================================================
# Fidelity & assumptions
# ===================================================================

class TestFidelity:

    def setup_method(self):
        FidelityRegistry.reset()

    def test_register_and_audit(self):
        reg = FidelityRegistry.get()
        mf = ModelFidelity(
            model_name="Exponential Atmosphere",
            level=FidelityLevel.SIMPLIFIED,
            assumptions=[
                Assumption("isothermal", "Constant temperature per layer"),
            ],
            valid_domain="altitude 0–1000 km",
            source="Vallado",
        )
        reg.register(mf)
        assert reg.get_model("Exponential Atmosphere") is mf
        audit = reg.audit()
        assert len(audit) == 1
        assert audit[0]["level"] == "simplified"

    def test_singleton(self):
        r1 = FidelityRegistry.get()
        r2 = FidelityRegistry.get()
        assert r1 is r2


# ===================================================================
# Trace
# ===================================================================

class TestTrace:

    def test_trace_creation(self):
        t = CalculationTrace(
            operation="specific_energy",
            equation="ε = v²/2 − μ/r",
            inputs={"v": 7500.0, "mu": 3.986e14, "r": 7000e3},
        )
        t.result = -2.5e7
        d = t.to_dict()
        assert d["operation"] == "specific_energy"
        assert d["result"] == -2.5e7

    def test_trace_context(self):
        ctx = TraceContext()
        with ctx:
            TraceContext.emit(CalculationTrace(operation="op1"))
            TraceContext.emit(CalculationTrace(operation="op2"))
        assert len(ctx.traces) == 2
        # Outside context, emit is no-op
        TraceContext.emit(CalculationTrace(operation="op3"))
        assert len(ctx.traces) == 2


# ===================================================================
# Health
# ===================================================================

class TestHealth:

    def test_nan_detected(self):
        hc = NumericalHealthChecker()
        with pytest.raises(NumericalInstabilityError):
            hc.check_state(
                np.array([float("nan"), 0, 0]),
                np.zeros(3),
                1000.0,
            )

    def test_inf_detected(self):
        hc = NumericalHealthChecker()
        with pytest.raises(NumericalInstabilityError):
            hc.check_state(
                np.array([float("inf"), 0, 0]),
                np.zeros(3),
                1000.0,
            )

    def test_negative_mass_detected(self):
        hc = NumericalHealthChecker()
        with pytest.raises(NumericalInstabilityError):
            hc.check_state(np.zeros(3), np.zeros(3), -1.0)

    def test_divergence_detected(self):
        hc = NumericalHealthChecker(max_position=1e10)
        with pytest.raises(NumericalInstabilityError):
            hc.check_state(np.array([2e10, 0, 0]), np.zeros(3), 100.0)

    def test_valid_state_passes(self):
        hc = NumericalHealthChecker()
        # Should not raise
        hc.check_state(
            np.array([7000e3, 0, 0]),
            np.array([0, 7500, 0]),
            1000.0,
        )


# ===================================================================
# Conservation diagnostics
# ===================================================================

class TestConservationDiagnostics:

    def test_circular_orbit_conservation(self):
        """For a circular orbit, energy and |h| should be constant."""
        mu = 3.986004418e14  # Earth
        r0 = 7_000_000.0    # m
        v0 = math.sqrt(mu / r0)  # circular velocity

        diag = ConservationDiagnostics(mu=mu)

        # Simulate 10 points around a circular orbit (analytical — perfect)
        n = 10
        for i in range(n):
            theta = 2 * math.pi * i / n
            pos = np.array([r0 * math.cos(theta), r0 * math.sin(theta), 0.0])
            vel = np.array([-v0 * math.sin(theta), v0 * math.cos(theta), 0.0])
            diag.record(float(i), pos, vel)

        assert diag.max_energy_drift() < 1e-12
        assert diag.max_angular_momentum_drift() < 1e-12

    def test_summary(self):
        diag = ConservationDiagnostics(mu=3.986e14)
        diag.record(0.0, np.array([7e6, 0, 0]), np.array([0, 7500, 0]))
        diag.record(1.0, np.array([7e6, 0, 0]), np.array([0, 7500, 0]))
        s = diag.summary()
        assert "max_energy_drift_relative" in s
        assert s["num_samples"] == 2
