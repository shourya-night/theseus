"""
General-purpose numerical orbit propagator.

Integrates the equations of motion dy/dt = f(t, y) where the derivative
function is assembled from a collection of force models (gravity, drag,
thrust, SRP, …).

The propagator returns a complete StateHistory suitable for visualisation.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from theseus.core.state import SimulationState, StateHistory
from theseus.core.events import EventType, EventLog
from theseus.core.health import NumericalHealthChecker
from theseus.core.diagnostics import ConservationDiagnostics
from theseus.constants.physical import G0_VAL
from theseus.dynamics.force_model import CompositeForceModel
from theseus.propagation.integrators import RK4Integrator, RKF45Integrator, IntegrationResult


class NumericalPropagator:
    """
    Numerical orbit propagator driven by pluggable force models.

    Parameters
    ----------
    acceleration_fn : callable
        Function ``a(t, position, velocity, mass) -> np.ndarray(3)``
        returning total acceleration (m/s²).
    integrator : str
        'rk4' or 'rkf45'.
    dt : float
        Step size for RK4, or initial step for RKF45 (s).
    atol, rtol : float
        Tolerances for RKF45.
    health_checker : NumericalHealthChecker or None
    record_diagnostics : bool
        If True, track conservation-law diagnostics.
    mu : float
        Central-body GM (m³/s²), required for conservation diagnostics.
    """

    def __init__(
        self,
        acceleration_fn: Callable[[float, np.ndarray, np.ndarray, float], np.ndarray],
        integrator: str = "rkf45",
        dt: float = 60.0,
        atol: float = 1e-10,
        rtol: float = 1e-10,
        health_checker: Optional[NumericalHealthChecker] = None,
        record_diagnostics: bool = True,
        mu: float = 3.986004418e14,
    ) -> None:
        self.acceleration_fn = acceleration_fn
        self.health_checker = health_checker or NumericalHealthChecker()
        self.record_diagnostics = record_diagnostics
        self.mu = mu

        if integrator == "rk4":
            self._integrator = RK4Integrator(dt=dt)
        elif integrator == "rkf45":
            self._integrator = RKF45Integrator(atol=atol, rtol=rtol, dt_initial=dt)
        else:
            raise ValueError(f"Unknown integrator: {integrator!r}")

    def propagate(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        t_span: tuple[float, float],
        mass: float = 1000.0,
        fuel_mass: float = 0.0,
    ) -> tuple[StateHistory, EventLog, Optional[ConservationDiagnostics]]:
        """
        Propagate from initial state over [t0, tf].

        The state vector for the integrator is [x, y, z, vx, vy, vz].
        Mass is updated externally (for thrust burns).

        Returns
        -------
        (history, events, diagnostics)
        """
        r0 = np.asarray(r0, dtype=np.float64)
        v0 = np.asarray(v0, dtype=np.float64)

        # Detect any active thrust model for mass flow calculation
        thrust_models = []
        fm = getattr(self.acceleration_fn, "__self__", None)
        if isinstance(fm, CompositeForceModel):
            for model in fm.models:
                if model.name == "Thrust" and hasattr(model, "spacecraft") and model.enabled:
                    thrust_models.append(model)
        elif hasattr(self, "force_model") and isinstance(self.force_model, CompositeForceModel):
            for model in self.force_model.models:
                if model.name == "Thrust" and hasattr(model, "spacecraft") and model.enabled:
                    thrust_models.append(model)

        has_thrust = len(thrust_models) > 0 and fuel_mass > 0.0
        m_dry = max(0.0, mass - fuel_mass)

        if has_thrust:
            y0 = np.concatenate([r0, v0, [fuel_mass]])
        else:
            y0 = np.concatenate([r0, v0])

        events = EventLog()
        diag = ConservationDiagnostics(mu=self.mu) if self.record_diagnostics else None

        def deriv(t: float, y: np.ndarray) -> np.ndarray:
            pos = y[:3]
            vel = y[3:6]
            if has_thrust:
                curr_fuel = max(0.0, float(y[6]))
                curr_m = m_dry + curr_fuel
                acc = self.acceleration_fn(t, pos, vel, curr_m)
                dm_fuel = 0.0
                if curr_fuel > 0.0:
                    for tm in thrust_models:
                        if tm.burn_start <= t <= tm.burn_end and tm.enabled:
                            F = tm.spacecraft.max_thrust * tm.throttle
                            isp = max(1.0, tm.spacecraft.specific_impulse)
                            dm_fuel -= F / (isp * G0_VAL)
                return np.concatenate([vel, acc, [dm_fuel]])
            else:
                acc = self.acceleration_fn(t, pos, vel, mass)
                return np.concatenate([vel, acc])

        result: IntegrationResult = self._integrator.integrate(deriv, y0, t_span)

        events.emit(t_span[0], EventType.INITIALIZATION, "Propagation started")

        history = StateHistory()
        for i in range(len(result.times)):
            t = result.times[i]
            y = result.states[i]
            pos = y[:3]
            vel = y[3:6]
            if has_thrust:
                curr_fuel = max(0.0, float(y[6]))
                curr_mass = m_dry + curr_fuel
            else:
                curr_fuel = fuel_mass
                curr_mass = mass

            acc = self.acceleration_fn(t, pos, vel, curr_mass)

            # Health check
            self.health_checker.check_state(pos, vel, curr_mass, acc)

            # Conservation diagnostics
            if diag is not None:
                diag.record(t, pos, vel)

            r_mag = float(np.linalg.norm(pos))
            state = SimulationState(
                time=t,
                position=pos.copy(),
                velocity=vel.copy(),
                acceleration=acc.copy(),
                mass=curr_mass,
                fuel_mass=curr_fuel,
                metadata={
                    "r_mag_m": r_mag,
                    "speed_m_s": float(np.linalg.norm(vel)),
                },
            )
            history.append(state)

        # Update spacecraft fuel mass if applicable
        if has_thrust and len(history) > 0:
            final_fuel = history[-1].fuel_mass
            for tm in thrust_models:
                tm.spacecraft.fuel_mass = final_fuel

        events.emit(t_span[1], EventType.STATE_UPDATE, "Propagation completed",
                     steps=result.steps_taken, rejected=result.rejected_steps,
                     method=result.method)

        return history, events, diag
