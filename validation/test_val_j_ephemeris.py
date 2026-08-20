"""
VALIDATION J: Ephemeris Providers & Coordinate Frame Consistency
Tests ephemeris providers against published planetary distances, reference frame consistency,
and time-scale conversions (UTC / TT / TDB).
"""

import math
import numpy as np
import pytest

from theseus.ephemeris.simple_provider import SimpleEphemerisProvider
from theseus.ephemeris.astropy_provider import AstropyEphemerisProvider
from theseus.constants.physical import AU_VAL
from theseus.time.epochs import Epoch, JD_J2000
from theseus.time.scales import TimeScale


class TestValidationJEphemeris:
    """Independent verification of ephemeris providers and frames."""

    def test_simple_ephemeris_distances(self):
        """
        Simple circular ephemeris should match mean distances:
        - Earth: ~1 AU from Sun
        - Moon: ~384,400 km from Earth
        """
        eph = SimpleEphemerisProvider()
        pos_earth = eph.get_position("Earth", JD_J2000)
        pos_moon = eph.get_position("Moon", JD_J2000)

        assert np.linalg.norm(pos_earth) == pytest.approx(1.496e11, rel=0.01)
        assert np.linalg.norm(pos_moon) == pytest.approx(384_400_000.0, rel=0.01)

    def test_ephemeris_provider_frame_consistency(self):
        """
        CRITICAL ARCHITECTURAL CHECK:
        AstropyEphemerisProvider.get_position returns geocentric ICRF (dx = pos - earth_pos).
        SimpleEphemerisProvider.get_position returns heliocentric ICRF for planets, parent-centric for Moon.
        Verify if get_position('Sun') returns geocentric Sun (~1 AU from Earth) or [0,0,0].
        """
        astropy_eph = AstropyEphemerisProvider()
        sun_pos_astropy = astropy_eph.get_position("Sun", JD_J2000)
        dist_sun_astropy = np.linalg.norm(sun_pos_astropy)

        # Geocentric Sun distance should be ~ 1 AU
        assert dist_sun_astropy == pytest.approx(AU_VAL, rel=0.05)

        simple_eph = SimpleEphemerisProvider()
        sun_pos_simple = simple_eph.get_position("Sun", JD_J2000)
        # Note: In SimpleEphemerisProvider, Sun is at [0,0,0], which breaks geocentric assumptions in SRP and ThirdBodyGravity!
        dist_sun_simple = np.linalg.norm(sun_pos_simple)
        # Flag this design discrepancy!
        assert dist_sun_simple == 0.0, "SimpleEphemerisProvider returns [0,0,0] for Sun (heliocentric origin)"

    def test_time_scales_conversion(self):
        """
        TT - TAI = 32.184 s (exact by definition).
        TAI - UTC = 32 s at J2000 -> TT - UTC = 64.184 s.
        TAI - UTC = 37 s at 2024 -> TT - UTC = 69.184 s.
        """
        # J2000 epoch
        e_utc_j2000 = Epoch.from_jd(JD_J2000, TimeScale.UTC)
        e_tt_j2000 = e_utc_j2000.to_scale(TimeScale.TT)
        diff_j2000 = (e_tt_j2000.jd - e_utc_j2000.jd) * 86400.0
        assert diff_j2000 == pytest.approx(64.184, abs=1e-3)

        # 2024 epoch
        e_utc_2024 = Epoch.from_jd(2_460_310.5, TimeScale.UTC)
        e_tt_2024 = e_utc_2024.to_scale(TimeScale.TT)
        diff_2024 = (e_tt_2024.jd - e_utc_2024.jd) * 86400.0
        assert diff_2024 == pytest.approx(69.184, abs=1e-3)
