"""Tests for astronomical time system."""

import math
from datetime import datetime, timezone

import pytest

from theseus.time.epochs import Epoch, JD_J2000, MJD_OFFSET
from theseus.time.scales import TimeScale


class TestEpoch:

    # -- J2000 reference -------------------------------------------------

    def test_j2000_jd(self):
        """J2000.0 = JD 2 451 545.0."""
        e = Epoch.j2000()
        assert e.jd == 2_451_545.0

    def test_j2000_datetime(self):
        """J2000.0 = 2000-01-01T12:00:00 UTC (approximately)."""
        e = Epoch.j2000()
        dt = e.to_datetime()
        assert dt.year == 2000
        assert dt.month == 1
        assert dt.day == 1
        # TDB vs UTC offset means this is approximately noon
        assert abs(dt.hour - 12) <= 1

    def test_j2000_elapsed(self):
        """Elapsed seconds since J2000 at J2000 is zero."""
        e = Epoch.j2000()
        assert e.elapsed_since_j2000() == pytest.approx(0.0, abs=1e-6)

    # -- JD ↔ MJD -------------------------------------------------------

    def test_jd_to_mjd(self):
        """MJD = JD − 2 400 000.5."""
        e = Epoch.from_jd(2_451_545.0)
        assert e.mjd == pytest.approx(51_544.5)

    def test_mjd_roundtrip(self):
        mjd = 51544.5
        e = Epoch.from_mjd(mjd)
        assert e.mjd == pytest.approx(mjd)

    # -- datetime ↔ JD ---------------------------------------------------

    def test_known_datetime_to_jd(self):
        """2000-01-01T12:00:00 UTC → JD 2451545.0."""
        dt = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        e = Epoch.from_datetime(dt)
        assert e.jd == pytest.approx(2_451_545.0, abs=1e-4)

    def test_datetime_roundtrip(self):
        dt_in = datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        e = Epoch.from_datetime(dt_in)
        dt_out = e.to_datetime()
        assert dt_out.year == dt_in.year
        assert dt_out.month == dt_in.month
        assert dt_out.day == dt_in.day
        assert dt_out.hour == dt_in.hour
        assert dt_out.minute == dt_in.minute
        assert abs(dt_out.second - dt_in.second) <= 1

    # -- Arithmetic -------------------------------------------------------

    def test_add_seconds(self):
        """epoch + 86400 s = next day."""
        e0 = Epoch.from_jd(2_451_545.0)
        e1 = e0 + 86400.0
        assert e1.jd == pytest.approx(2_451_546.0)

    def test_subtract_epochs(self):
        """epoch2 − epoch1 = seconds between them."""
        e0 = Epoch.from_jd(2_451_545.0)
        e1 = Epoch.from_jd(2_451_546.0)
        assert (e1 - e0) == pytest.approx(86400.0)

    def test_subtract_seconds(self):
        e = Epoch.from_jd(2_451_546.0)
        e2 = e - 86400.0
        assert e2.jd == pytest.approx(2_451_545.0)

    # -- from_elapsed -----------------------------------------------------

    def test_from_elapsed(self):
        ref = Epoch.j2000()
        e = Epoch.from_elapsed(3600.0, ref)
        assert e.jd == pytest.approx(ref.jd + 3600.0 / 86400.0)

    # -- Comparison -------------------------------------------------------

    def test_ordering(self):
        e1 = Epoch.from_jd(2_451_545.0)
        e2 = Epoch.from_jd(2_451_546.0)
        assert e1 < e2
        assert e2 > e1  # type: ignore[operator]
        assert e1 <= e2
        assert e1 == Epoch.from_jd(2_451_545.0)

    # -- Time-scale conversion (smoke test) --------------------------------

    def test_utc_to_tt(self):
        """TT − UTC = 64.184 s at J2000, 69.184 s in 2024."""
        # J2000 (32 leap seconds)
        e_utc_j2000 = Epoch.from_jd(2_451_545.0, TimeScale.UTC)
        e_tt_j2000 = e_utc_j2000.to_scale(TimeScale.TT)
        diff_j2000 = (e_tt_j2000.jd - e_utc_j2000.jd) * 86400.0
        assert diff_j2000 == pytest.approx(64.184, abs=0.01)

        # 2024 (37 leap seconds)
        e_utc_2024 = Epoch.from_jd(2_460_310.5, TimeScale.UTC)
        e_tt_2024 = e_utc_2024.to_scale(TimeScale.TT)
        diff_2024 = (e_tt_2024.jd - e_utc_2024.jd) * 86400.0
        assert diff_2024 == pytest.approx(69.184, abs=0.01)

    def test_same_scale_conversion(self):
        e = Epoch.from_jd(2_451_545.0, TimeScale.TDB)
        e2 = e.to_scale(TimeScale.TDB)
        assert e2.jd == e.jd
