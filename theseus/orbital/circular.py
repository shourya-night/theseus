"""
Analytic circular-orbit ephemeris.

Produces position and velocity as functions of time for an unperturbed
circular orbit.  This exists so that callers which need a simple analytic
two-body ephemeris -- the conjunction API endpoints, test fixtures -- have one
authoritative construction to use instead of open-coding the rotation from
orbital elements to inertial coordinates.

Convention
----------
A circular orbit is fully specified here by radius, inclination, right
ascension of the ascending node, and a phase angle.  The phase angle is the
**true anomaly at the epoch**, measured in the orbital plane from the
ascending node, because argument of periapsis is undefined for a circular
orbit and is therefore fixed at zero.

With the default ``raan_rad = 0`` the ascending node lies along +x, so two
orbits of differing inclination built with this default cross each other on
the x axis.  That is a real modelling choice, not an accident: it is what
makes a pair of such orbits produce node conjunctions at all.  Callers that
want independent node geometry must pass distinct ``raan_rad`` values.

Mathematics
-----------
For eccentricity zero the state at true anomaly ν is

    r(ν) = a ( P̂ cos ν + Q̂ sin ν )
    v(ν) = √(μ/a) ( −P̂ sin ν + Q̂ cos ν )

where P̂ and Q̂ are the perifocal basis vectors expressed in the inertial
frame.  Rather than re-deriving that rotation, the basis is read back from
:func:`theseus.orbital.conversions.elements_to_state` — evaluating it at
ν = 0 gives a·P̂ and at ν = π/2 gives a·Q̂.  The rotation convention is
therefore whatever the engine's element conversion says it is, and cannot
drift away from it.

True anomaly advances uniformly for a circular orbit,

    ν(t) = ν₀ + n (t − t_epoch),    n = √(μ / a³)

so no Kepler equation has to be solved and the ephemeris is exact for the
unperturbed two-body problem.

Limitations
-----------
Two-body only: no J2, no drag, no third bodies.  Use the numerical propagator
when perturbations matter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from theseus.orbital.elements import OrbitalElements
from theseus.orbital.conversions import elements_to_state


@dataclass(frozen=True)
class CircularOrbitStates:
    """
    Analytic position/velocity functions for a circular two-body orbit.

    Parameters
    ----------
    radius_m : float
        Orbit radius from the central body's centre (m).  Must be > 0.
    inclination_rad : float
        Inclination (rad).
    phase_rad : float
        True anomaly at ``epoch_s`` (rad), measured from the ascending node.
    mu : float
        Gravitational parameter of the central body (m³/s²).  Must be > 0.
    raan_rad : float
        Right ascension of the ascending node (rad).  Default 0.
    epoch_s : float
        Time at which the phase angle applies (s).  Default 0.

    All quantities are SI, and the produced states are inertial (ECI/ICRF),
    consistent with the rest of the engine.
    """
    radius_m: float
    inclination_rad: float
    phase_rad: float
    mu: float
    raan_rad: float = 0.0
    epoch_s: float = 0.0

    _p_hat: np.ndarray = field(init=False, repr=False, compare=False)
    _q_hat: np.ndarray = field(init=False, repr=False, compare=False)
    _cache: list = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError(f"radius_m must be finite and > 0, got {self.radius_m}")
        if not math.isfinite(self.mu) or self.mu <= 0.0:
            raise ValueError(f"mu must be finite and > 0, got {self.mu}")
        for name in ("inclination_rad", "phase_rad", "raan_rad", "epoch_s"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")

        # Read the perifocal basis back out of the engine's own element
        # conversion, so this class never re-implements the rotation.
        r_at_0, _ = elements_to_state(self._elements(0.0))
        r_at_90, _ = elements_to_state(self._elements(0.5 * math.pi))
        object.__setattr__(self, "_p_hat", np.asarray(r_at_0, dtype=np.float64) / self.radius_m)
        object.__setattr__(self, "_q_hat", np.asarray(r_at_90, dtype=np.float64) / self.radius_m)

        # Single-entry memo.  Consumers such as the conjunction pipeline ask
        # for the position and then the velocity at the same instant; without
        # this the trigonometry runs twice for every time.  Keyed on the exact
        # float, so it can only return the state it would have recomputed.
        object.__setattr__(self, "_cache", [None, None, None])

    def _elements(self, nu: float) -> OrbitalElements:
        """Classical elements at true anomaly *nu*.  Circular: e = 0, argp = 0."""
        return OrbitalElements(
            a=self.radius_m,
            e=0.0,
            i=self.inclination_rad,
            raan=self.raan_rad,
            argp=0.0,
            nu=nu,
            mu=self.mu,
        )

    # -- derived quantities --------------------------------------------------

    @property
    def mean_motion_rad_s(self) -> float:
        """n = √(μ / a³) (rad/s).  Equal to the true-anomaly rate when e = 0."""
        return math.sqrt(self.mu / self.radius_m ** 3)

    @property
    def speed_m_s(self) -> float:
        """Circular orbital speed √(μ / a) (m/s), constant along the orbit."""
        return math.sqrt(self.mu / self.radius_m)

    @property
    def period_s(self) -> float:
        """Orbital period (s)."""
        return 2.0 * math.pi / self.mean_motion_rad_s

    def true_anomaly_at(self, t: float) -> float:
        """True anomaly (rad) at time *t* (s)."""
        return self.phase_rad + self.mean_motion_rad_s * (float(t) - self.epoch_s)

    # -- evaluation ----------------------------------------------------------

    def state_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Inertial position (m) and velocity (m/s) at time *t* (s)."""
        t = float(t)
        cache = self._cache
        if cache[0] is not None and t == cache[0]:
            return cache[1], cache[2]

        nu = self.true_anomaly_at(t)
        cos_nu = math.cos(nu)
        sin_nu = math.sin(nu)
        position = self.radius_m * (cos_nu * self._p_hat + sin_nu * self._q_hat)
        velocity = self.speed_m_s * (-sin_nu * self._p_hat + cos_nu * self._q_hat)
        position.flags.writeable = False
        velocity.flags.writeable = False

        cache[0], cache[1], cache[2] = t, position, velocity
        return position, velocity

    def position_at(self, t: float) -> np.ndarray:
        return self.state_at(t)[0]

    def velocity_at(self, t: float) -> np.ndarray:
        return self.state_at(t)[1]

    def as_callables(self) -> tuple[Callable[[float], np.ndarray], Callable[[float], np.ndarray]]:
        """
        Return ``(position_fn, velocity_fn)`` for APIs that take separate
        position and velocity callables, such as the conjunction pipeline.
        """
        return self.position_at, self.velocity_at

    def to_dict(self) -> dict[str, Any]:
        """Serialise the defining parameters, for traceability in a result."""
        return {
            "radius_m": float(self.radius_m),
            "radius_km": float(self.radius_m) / 1e3,
            "inclination_deg": math.degrees(self.inclination_rad),
            "raan_deg": math.degrees(self.raan_rad),
            "phase_deg": math.degrees(self.phase_rad),
            "epoch_s": float(self.epoch_s),
            "mu_m3_s2": float(self.mu),
            "mean_motion_rad_s": self.mean_motion_rad_s,
            "period_s": self.period_s,
            "speed_m_s": self.speed_m_s,
            "eccentricity": 0.0,
            "arg_periapsis_deg": 0.0,
            "frame": "ECI (ICRF)",
            "model": "analytic two-body circular",
        }


def circular_orbit_from_altitude(
    altitude_m: float,
    body_radius_m: float,
    inclination_rad: float,
    phase_rad: float,
    mu: float,
    raan_rad: float = 0.0,
    epoch_s: float = 0.0,
) -> CircularOrbitStates:
    """
    Build a circular orbit from an altitude above the central body's surface.

    ``radius = body_radius_m + altitude_m``.  The body radius used is whatever
    the caller supplies; for the engine's body catalogue that is the mean
    equatorial radius, so the altitude is a spherical-Earth altitude and not a
    geodetic one.
    """
    return CircularOrbitStates(
        radius_m=float(body_radius_m) + float(altitude_m),
        inclination_rad=float(inclination_rad),
        phase_rad=float(phase_rad),
        mu=float(mu),
        raan_rad=float(raan_rad),
        epoch_s=float(epoch_s),
    )
