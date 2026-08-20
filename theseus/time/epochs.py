"""
Astronomical epoch representation.

Internal storage: Julian Date (float64).
Provides conversions between UTC datetime, JD, MJD, and elapsed seconds.

The J2000.0 epoch (2000-01-01T12:00:00 TDB, JD 2451545.0) is the
standard reference epoch for most astrodynamical calculations.

Notes
-----
* Julian Date arithmetic avoids the pitfalls of naïve datetime arithmetic
  for astronomical time spans.
* For sub-millisecond precision, a split-JD (integer + fractional day)
  representation would be needed.  The current float64 JD gives ~20 μs
  precision near J2000, which is adequate for the initial engine.

References
----------
Meeus, "Astronomical Algorithms", 2nd ed., Chapter 7.
USNO Circular 179.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from typing import Optional

from theseus.time.scales import TimeScale

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JD_J2000: float = 2_451_545.0
"""Julian Date of J2000.0 epoch (2000-01-01T12:00:00 TDB)."""

MJD_OFFSET: float = 2_400_000.5
"""MJD = JD − 2400000.5."""

# TAI − UTC offset table would be needed for full leap-second support.
# For now we use a simplified constant offset for TT↔UTC.
# TT = TAI + 32.184 s;  TAI = UTC + ΔAT (leap seconds).
# As of 2024-12 ΔAT = 37 s → TT − UTC ≈ 69.184 s.
_TT_MINUS_UTC_APPROX: float = 69.184  # seconds (valid 2017–2026+)


class Epoch:
    """
    A single instant in astronomical time.

    Parameters
    ----------
    jd : float
        Julian Date.
    scale : TimeScale
        Time scale of the supplied JD.  Default UTC.
    """

    __slots__ = ("_jd", "_scale")

    def __init__(self, jd: float, scale: TimeScale = TimeScale.UTC) -> None:
        self._jd = jd
        self._scale = scale

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_jd(cls, jd: float, scale: TimeScale = TimeScale.UTC) -> Epoch:
        """Create from Julian Date."""
        return cls(jd, scale)

    @classmethod
    def from_mjd(cls, mjd: float, scale: TimeScale = TimeScale.UTC) -> Epoch:
        """Create from Modified Julian Date (MJD = JD − 2400000.5)."""
        return cls(mjd + MJD_OFFSET, scale)

    @classmethod
    def from_datetime(cls, dt: datetime, scale: TimeScale = TimeScale.UTC) -> Epoch:
        """
        Create from a Python datetime.

        If *dt* is naïve (no tzinfo), it is assumed to be UTC.
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # Convert to UTC
        dt_utc = dt.astimezone(timezone.utc)
        jd = cls._datetime_to_jd(dt_utc)
        return cls(jd, scale)

    @classmethod
    def j2000(cls) -> Epoch:
        """Return the J2000.0 epoch (TDB)."""
        return cls(JD_J2000, TimeScale.TDB)

    @classmethod
    def from_elapsed(cls, seconds: float, reference: Optional[Epoch] = None) -> Epoch:
        """
        Create from elapsed seconds after a reference epoch.

        Parameters
        ----------
        seconds : float
            Elapsed time in seconds.
        reference : Epoch or None
            Reference epoch.  Defaults to J2000.0 TDB.
        """
        if reference is None:
            reference = cls.j2000()
        jd = reference._jd + seconds / 86400.0
        return cls(jd, reference._scale)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def jd(self) -> float:
        """Julian Date in the stored time scale."""
        return self._jd

    @property
    def mjd(self) -> float:
        """Modified Julian Date."""
        return self._jd - MJD_OFFSET

    @property
    def scale(self) -> TimeScale:
        return self._scale

    def to_datetime(self) -> datetime:
        """Convert to a Python datetime (UTC)."""
        return self._jd_to_datetime(self._jd)

    def elapsed_since(self, other: Epoch) -> float:
        """Seconds elapsed since *other* epoch (self − other)."""
        return (self._jd - other._jd) * 86400.0

    def elapsed_since_j2000(self) -> float:
        """Seconds elapsed since J2000.0."""
        return (self._jd - JD_J2000) * 86400.0

    def julian_centuries_since_j2000(self) -> float:
        """Julian centuries (36525 days) since J2000.0."""
        return (self._jd - JD_J2000) / 36525.0

    # ------------------------------------------------------------------
    # Time-scale conversion (simplified)
    # ------------------------------------------------------------------

    def to_scale(self, target: TimeScale) -> Epoch:
        """
        Convert to a different time scale.

        .. warning::
           Currently uses an approximate TT−UTC offset (valid 2017-2026).
           For high-precision work, a full leap-second table is needed.
        """
        if target == self._scale:
            return Epoch(self._jd, self._scale)

        # Convert to TT first, then from TT to target
        jd_tt = self._to_tt()
        jd_target = self._from_tt(jd_tt, target)
        return Epoch(jd_target, target)

    def _to_tt(self) -> float:
        """Convert stored JD to TT."""
        if self._scale == TimeScale.TT:
            return self._jd
        elif self._scale == TimeScale.TAI:
            return self._jd + 32.184 / 86400.0
        elif self._scale == TimeScale.TDB:
            try:
                from astropy.time import Time
                t = Time(self._jd, format="jd", scale="tdb")
                return float(t.tt.jd)
            except Exception:
                return self._jd
        elif self._scale == TimeScale.UTC:
            try:
                from astropy.time import Time
                t = Time(self._jd, format="jd", scale="utc")
                return float(t.tt.jd)
            except Exception:
                return self._jd + _TT_MINUS_UTC_APPROX / 86400.0
        raise ValueError(f"Unsupported time scale: {self._scale}")

    @staticmethod
    def _from_tt(jd_tt: float, target: TimeScale) -> float:
        """Convert TT JD to target scale."""
        if target == TimeScale.TT:
            return jd_tt
        elif target == TimeScale.TAI:
            return jd_tt - 32.184 / 86400.0
        elif target == TimeScale.TDB:
            try:
                from astropy.time import Time
                t = Time(jd_tt, format="jd", scale="tt")
                return float(t.tdb.jd)
            except Exception:
                return jd_tt
        elif target == TimeScale.UTC:
            try:
                from astropy.time import Time
                t = Time(jd_tt, format="jd", scale="tt")
                return float(t.utc.jd)
            except Exception:
                return jd_tt - _TT_MINUS_UTC_APPROX / 86400.0
        raise ValueError(f"Unsupported time scale: {target}")

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, seconds: float) -> Epoch:
        """epoch + seconds → new Epoch."""
        return Epoch(self._jd + seconds / 86400.0, self._scale)

    def __sub__(self, other: Epoch | float) -> Epoch | float:
        """
        epoch − epoch → seconds.
        epoch − seconds → new Epoch.
        """
        if isinstance(other, Epoch):
            return (self._jd - other._jd) * 86400.0
        return Epoch(self._jd - other / 86400.0, self._scale)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Epoch):
            return NotImplemented
        return self._jd == other._jd and self._scale == other._scale

    def __lt__(self, other: Epoch) -> bool:
        return self._jd < other._jd

    def __le__(self, other: Epoch) -> bool:
        return self._jd <= other._jd

    def __repr__(self) -> str:
        return f"Epoch(jd={self._jd:.8f}, scale={self._scale.value})"

    def __str__(self) -> str:
        dt = self.to_datetime()
        return f"{dt.strftime('%Y-%m-%dT%H:%M:%S.%f')} {self._scale.value}"

    # ------------------------------------------------------------------
    # Internal JD ↔ datetime helpers  (Meeus Ch. 7)
    # ------------------------------------------------------------------

    @staticmethod
    def _datetime_to_jd(dt: datetime) -> float:
        """Convert a UTC datetime to Julian Date."""
        y = dt.year
        m = dt.month
        d = (
            dt.day
            + dt.hour / 24.0
            + dt.minute / 1440.0
            + dt.second / 86400.0
            + dt.microsecond / 86400e6
        )
        if m <= 2:
            y -= 1
            m += 12
        A = math.floor(y / 100)
        B = 2 - A + math.floor(A / 4)
        return math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d + B - 1524.5

    @staticmethod
    def _jd_to_datetime(jd: float) -> datetime:
        """Convert Julian Date to a UTC datetime."""
        jd = jd + 0.5
        Z = math.floor(jd)
        F = jd - Z
        if Z < 2299161:
            A = Z
        else:
            alpha = math.floor((Z - 1867216.25) / 36524.25)
            A = Z + 1 + alpha - math.floor(alpha / 4)
        B = A + 1524
        C = math.floor((B - 122.1) / 365.25)
        D = math.floor(365.25 * C)
        E = math.floor((B - D) / 30.6001)

        day_frac = B - D - math.floor(30.6001 * E) + F
        day = int(day_frac)
        frac = day_frac - day

        month = E - 1 if E < 14 else E - 13
        year = C - 4716 if month > 2 else C - 4715

        sec_raw = frac * 86400.0
        sec_nearest = round(sec_raw)
        if abs(sec_raw - sec_nearest) < 1e-4:
            sec = float(sec_nearest)
        else:
            sec = round(sec_raw, 6)

        base_dt = datetime(year, month, day, tzinfo=timezone.utc)
        return base_dt + timedelta(seconds=sec)
