"""
Astropy-backed ephemeris provider.

Uses ``astropy.coordinates`` with the built-in solar-system ephemeris
(low-precision analytic series, based on Meeus) or optionally JPL
kernels (DE430/DE440) if available.

Precision (built-in)
--------------------
~1000 km (1e6 m) for inner planets; ~10 000 km for outer planets.
Adequate for trajectory design to ~km level with JPL kernels.

Reference frame : ICRF (≈ J2000 equatorial).
Time scale      : TDB via Astropy's Time.
Units           : metres, metres per second.
"""

from __future__ import annotations

import numpy as np

from theseus.ephemeris.provider import EphemerisProvider
from theseus.core.fidelity import (
    ModelFidelity, FidelityLevel, Assumption, FidelityRegistry,
)

# Lazy import astropy — only fail when actually used
_astropy_available: bool | None = None


def _ensure_astropy() -> None:
    global _astropy_available
    if _astropy_available is None:
        try:
            import astropy  # noqa: F401
            _astropy_available = True
        except ImportError:
            _astropy_available = False
    if not _astropy_available:
        raise ImportError(
            "Astropy is required for AstropyEphemerisProvider.  "
            "Install with:  pip install astropy"
        )


_FIDELITY_BUILTIN = ModelFidelity(
    model_name="AstropyBuiltinEphemeris",
    level=FidelityLevel.MODERATE,
    assumptions=[
        Assumption("analytic_series",
                   "Uses Meeus-based analytic series for planet positions",
                   "~1000 km accuracy for inner planets"),
        Assumption("no_asteroids",
                   "Only major solar-system bodies supported"),
    ],
    valid_domain="Solar system major bodies, years ~1900–2100",
    source="Astropy built-in (Meeus analytic series)",
    limitations="~1000 km for inner planets, ~10000 km for outer planets",
)

# Mapping from THESEUS body names to Astropy body names
_ASTROPY_NAMES: dict[str, str] = {
    "sun": "sun",
    "mercury": "mercury",
    "venus": "venus",
    "earth": "earth",
    "moon": "moon",
    "mars": "mars",
    "jupiter": "jupiter barycenter",
    "saturn": "saturn barycenter",
    "uranus": "uranus barycenter",
    "neptune": "neptune barycenter",
}


class AstropyEphemerisProvider(EphemerisProvider):
    """
    Ephemeris provider backed by Astropy.

    Parameters
    ----------
    ephemeris : str
        Astropy ephemeris to use.  'builtin' (default) or a JPL kernel
        name like 'de430' or 'de440' (requires downloading the kernel).
    """

    def __init__(self, ephemeris: str = "builtin") -> None:
        _ensure_astropy()
        self._ephemeris = ephemeris

        import astropy.coordinates as coord
        coord.solar_system_ephemeris.set(ephemeris)

        fidelity = ModelFidelity(
            model_name=f"AstropyEphemeris({ephemeris})",
            level=FidelityLevel.MODERATE if ephemeris == "builtin" else FidelityLevel.HIGH,
            assumptions=_FIDELITY_BUILTIN.assumptions,
            valid_domain="Solar system major bodies",
            source=f"Astropy, ephemeris={ephemeris}",
            limitations="~1000 km (builtin) or ~m (JPL)" if ephemeris == "builtin" else "~1 m (JPL kernels)",
        )
        FidelityRegistry.get().register(fidelity)

    @property
    def name(self) -> str:
        return f"AstropyEphemeris({self._ephemeris})"

    @property
    def source(self) -> str:
        return f"Astropy {self._ephemeris}"

    @property
    def precision_description(self) -> str:
        if self._ephemeris == "builtin":
            return "~1000 km for inner planets (Meeus analytic series)"
        return f"~1 m (JPL {self._ephemeris} kernels)"

    def _resolve_name(self, body_name: str) -> str:
        key = body_name.strip().lower()
        if key in _ASTROPY_NAMES:
            return _ASTROPY_NAMES[key]
        raise KeyError(f"Unknown body for Astropy ephemeris: {body_name!r}")

    def get_position(self, body_name: str, epoch_jd: float) -> np.ndarray:
        """Position in ICRF (m), geocentric."""
        from astropy.time import Time
        import astropy.coordinates as coord
        import astropy.units as u

        t = Time(epoch_jd, format="jd", scale="tdb")
        astro_name = self._resolve_name(body_name)

        if astro_name == "earth":
            return np.zeros(3)

        pos = coord.get_body_barycentric(astro_name, t)
        earth_pos = coord.get_body_barycentric("earth", t)
        # Geocentric
        dx = pos.x.to(u.m).value - earth_pos.x.to(u.m).value
        dy = pos.y.to(u.m).value - earth_pos.y.to(u.m).value
        dz = pos.z.to(u.m).value - earth_pos.z.to(u.m).value
        return np.array([dx, dy, dz])

    def get_state(self, body_name: str, epoch_jd: float) -> tuple[np.ndarray, np.ndarray]:
        """Position and velocity in ICRF (m, m/s), geocentric."""
        from astropy.time import Time
        import astropy.coordinates as coord
        import astropy.units as u

        t = Time(epoch_jd, format="jd", scale="tdb")
        astro_name = self._resolve_name(body_name)

        if astro_name == "earth":
            return np.zeros(3), np.zeros(3)

        pos_vel = coord.get_body_barycentric_posvel(astro_name, t)
        earth_pv = coord.get_body_barycentric_posvel("earth", t)

        pos = np.array([
            pos_vel[0].x.to(u.m).value - earth_pv[0].x.to(u.m).value,
            pos_vel[0].y.to(u.m).value - earth_pv[0].y.to(u.m).value,
            pos_vel[0].z.to(u.m).value - earth_pv[0].z.to(u.m).value,
        ])
        vel = np.array([
            pos_vel[1].x.to(u.m / u.s).value - earth_pv[1].x.to(u.m / u.s).value,
            pos_vel[1].y.to(u.m / u.s).value - earth_pv[1].y.to(u.m / u.s).value,
            pos_vel[1].z.to(u.m / u.s).value - earth_pv[1].z.to(u.m / u.s).value,
        ])
        return pos, vel
