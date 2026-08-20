"""
Numerical integrators: RK4 and RKF45 (adaptive).

Both integrators operate on a generic ODE system  dy/dt = f(t, y)
where y is an arbitrary-length state vector (numpy array).

RK4  — Classical 4th-order Runge-Kutta (fixed step)
RKF45 — Runge-Kutta-Fehlberg 4(5) with embedded error estimation
         and automatic step-size control.

Mathematics
-----------
RK4 Butcher tableau:

    0   |
    1/2 | 1/2
    1/2 | 0    1/2
    1   | 0    0    1
    ----|------------------
        | 1/6  1/3  1/3  1/6

RKF45 uses a 6-stage tableau with 4th- and 5th-order solutions
for embedded error estimation.  See Fehlberg (1969).

References
----------
Butcher, "Numerical Methods for Ordinary Differential Equations", 3rd ed.
Fehlberg, "Low-order classical Runge-Kutta formulas with stepsize control
           and their application to some heat transfer problems", NASA TR R-315, 1969.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# Type alias for the derivative function
DerivFn = Callable[[float, np.ndarray], np.ndarray]


@dataclass
class IntegrationResult:
    """Result of a numerical integration."""
    times: np.ndarray
    states: np.ndarray          # shape (N, state_dim)
    steps_taken: int
    rejected_steps: int = 0
    method: str = ""


# ===================================================================
# RK4 — Classical fourth-order Runge-Kutta
# ===================================================================

class RK4Integrator:
    """
    Fixed-step classical 4th-order Runge-Kutta integrator.

    Parameters
    ----------
    dt : float
        Fixed time step (s).
    """

    def __init__(self, dt: float) -> None:
        if dt <= 0:
            raise ValueError(f"Step size must be positive, got {dt}")
        self.dt = dt

    def step(self, f: DerivFn, t: float, y: np.ndarray) -> np.ndarray:
        """
        Advance one RK4 step.

        Parameters
        ----------
        f : callable   dy/dt = f(t, y).
        t : float      Current time.
        y : np.ndarray Current state.

        Returns
        -------
        np.ndarray     State at t + dt.
        """
        h = self.dt
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def integrate(
        self,
        f: DerivFn,
        y0: np.ndarray,
        t_span: tuple[float, float],
    ) -> IntegrationResult:
        """
        Integrate dy/dt = f(t, y) over [t0, tf].

        Returns states at every step (including t0 and tf).
        """
        t0, tf = t_span
        if tf < t0:
            raise ValueError(f"t_span end ({tf}) must be >= start ({t0})")

        direction = 1.0
        t = t0
        y = np.array(y0, dtype=np.float64)
        times = [t]
        states = [y.copy()]
        steps = 0

        while t < tf - 1e-12 * abs(self.dt):
            h = min(self.dt, tf - t)
            k1 = f(t, y)
            k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
            k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
            k4 = f(t + h, y + h * k3)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
            times.append(t)
            states.append(y.copy())
            steps += 1

        return IntegrationResult(
            times=np.array(times),
            states=np.array(states),
            steps_taken=steps,
            method="RK4",
        )


# ===================================================================
# RKF45 — Runge-Kutta-Fehlberg adaptive
# ===================================================================

# RKF45 Butcher tableau coefficients
_A2 = 1.0 / 4.0
_A3 = 3.0 / 8.0
_A4 = 12.0 / 13.0
_A5 = 1.0
_A6 = 1.0 / 2.0

_B21 = 1.0 / 4.0
_B31 = 3.0 / 32.0;   _B32 = 9.0 / 32.0
_B41 = 1932.0 / 2197.0; _B42 = -7200.0 / 2197.0; _B43 = 7296.0 / 2197.0
_B51 = 439.0 / 216.0; _B52 = -8.0; _B53 = 3680.0 / 513.0; _B54 = -845.0 / 4104.0
_B61 = -8.0 / 27.0; _B62 = 2.0; _B63 = -3544.0 / 2565.0; _B64 = 1859.0 / 4104.0; _B65 = -11.0 / 40.0

# 4th-order weights
_C1 = 25.0 / 216.0; _C3 = 1408.0 / 2565.0; _C4 = 2197.0 / 4104.0; _C5 = -1.0 / 5.0
# 5th-order weights
_D1 = 16.0 / 135.0; _D3 = 6656.0 / 12825.0; _D4 = 28561.0 / 56430.0; _D5 = -9.0 / 50.0; _D6 = 2.0 / 55.0


class RKF45Integrator:
    """
    Runge-Kutta-Fehlberg 4(5) adaptive-step integrator.

    Parameters
    ----------
    atol : float
        Absolute error tolerance.
    rtol : float
        Relative error tolerance.
    dt_min : float
        Minimum allowed step size (s).
    dt_max : float
        Maximum allowed step size (s).
    dt_initial : float
        Initial step size (s).
    safety : float
        Safety factor for step-size control (0.8–0.9 typical).
    max_steps : int
        Maximum number of integration steps (safety limit).
    """

    def __init__(
        self,
        atol: float = 1e-10,
        rtol: float = 1e-10,
        dt_min: float = 1e-6,
        dt_max: float = 1e4,
        dt_initial: float = 60.0,
        safety: float = 0.84,
        max_steps: int = 1_000_000,
    ) -> None:
        self.atol = atol
        self.rtol = rtol
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_initial = dt_initial
        self.safety = safety
        self.max_steps = max_steps

    def integrate(
        self,
        f: DerivFn,
        y0: np.ndarray,
        t_span: tuple[float, float],
        output_dt: float | None = None,
    ) -> IntegrationResult:
        """
        Integrate dy/dt = f(t, y) with adaptive step-size control.

        Parameters
        ----------
        f : callable
        y0 : np.ndarray
        t_span : (t0, tf)
        output_dt : float or None
            If given, interpolate output at uniform intervals of this size.
            Otherwise return every accepted step.

        Returns
        -------
        IntegrationResult
        """
        t0, tf = t_span
        t = t0
        y = np.array(y0, dtype=np.float64)
        h = min(self.dt_initial, tf - t0)

        times = [t]
        states = [y.copy()]
        steps = 0
        rejected = 0

        while t < tf - 1e-15 * abs(h):
            if steps >= self.max_steps:
                break

            h = min(h, tf - t)
            h = max(h, self.dt_min)

            # RKF45 stages
            k1 = f(t, y)
            k2 = f(t + _A2 * h, y + h * _B21 * k1)
            k3 = f(t + _A3 * h, y + h * (_B31 * k1 + _B32 * k2))
            k4 = f(t + _A4 * h, y + h * (_B41 * k1 + _B42 * k2 + _B43 * k3))
            k5 = f(t + _A5 * h, y + h * (_B51 * k1 + _B52 * k2 + _B53 * k3 + _B54 * k4))
            k6 = f(t + _A6 * h, y + h * (_B61 * k1 + _B62 * k2 + _B63 * k3 + _B64 * k4 + _B65 * k5))

            # 4th- and 5th-order solutions
            y4 = y + h * (_C1 * k1 + _C3 * k3 + _C4 * k4 + _C5 * k5)
            y5 = y + h * (_D1 * k1 + _D3 * k3 + _D4 * k4 + _D5 * k5 + _D6 * k6)

            # Error estimate
            err_vec = y5 - y4
            scale = self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y5))
            err = float(np.sqrt(np.mean((err_vec / scale) ** 2)))

            if err <= 1.0 or h <= self.dt_min:
                # Accept step (use 5th-order solution)
                t += h
                y = y5
                times.append(t)
                states.append(y.copy())
                steps += 1
            else:
                rejected += 1

            # Adjust step size
            if err > 1e-30:
                h_new = h * self.safety * (1.0 / err) ** 0.2
            else:
                h_new = h * 5.0
            h = max(self.dt_min, min(h_new, self.dt_max))

        return IntegrationResult(
            times=np.array(times),
            states=np.array(states),
            steps_taken=steps,
            rejected_steps=rejected,
            method="RKF45",
        )
