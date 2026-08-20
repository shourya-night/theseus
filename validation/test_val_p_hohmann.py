"""
VALIDATION P: Hohmann Transfer (LEO -> GEO)
Independently calculates LEO -> GEO Hohmann burns, transfer time, and Delta-v budget.
"""

import math
import pytest

from theseus.bodies.catalog import EARTH
from theseus.maneuvers.transfers import hohmann_transfer


MU = EARTH.mu


class TestValidationPHohmann:
    """Independent verification of Hohmann transfer analytics."""

    def test_hohmann_leo_to_geo_exact_analytical(self):
        """
        LEO: r1 = 6678137.0 m (300 km altitude)
        GEO: r2 = 42164137.0 m (35786 km altitude)
        Independently calculated reference values:
        - vc1 = sqrt(mu / r1) = 7725.74 m/s
        - vc2 = sqrt(mu / r2) = 3074.66 m/s
        - a_t = (r1 + r2) / 2 = 24421137.0 m
        - vt1 = sqrt(mu * (2/r1 - 1/a_t)) = 10145.37 m/s
        - vt2 = sqrt(mu * (2/r2 - 1/a_t)) = 1607.72 m/s
        - dv1 = vt1 - vc1 = 2419.63 m/s
        - dv2 = vc2 - vt2 = 1466.94 m/s
        - dv_total = 3886.57 m/s
        - T_transfer = pi * sqrt(a_t^3 / mu) = 18974.77 s (~5.27 hours)
        """
        r1 = 6_678_137.0
        r2 = 42_164_137.0

        vc1 = math.sqrt(MU / r1)
        vc2 = math.sqrt(MU / r2)
        a_t = (r1 + r2) / 2.0
        vt1 = math.sqrt(MU * (2.0 / r1 - 1.0 / a_t))
        vt2 = math.sqrt(MU * (2.0 / r2 - 1.0 / a_t))

        dv1_ref = vt1 - vc1
        dv2_ref = vc2 - vt2
        dv_tot_ref = dv1_ref + dv2_ref
        t_transfer_ref = math.pi * math.sqrt(a_t**3 / MU)

        res = hohmann_transfer(r1, r2, MU)

        assert res.delta_v1 == pytest.approx(dv1_ref, rel=1e-12)
        assert res.delta_v2 == pytest.approx(dv2_ref, rel=1e-12)
        assert res.total_delta_v == pytest.approx(dv_tot_ref, rel=1e-12)
        assert res.transfer_time == pytest.approx(t_transfer_ref, rel=1e-12)
        assert res.transfer_a == pytest.approx(a_t, rel=1e-12)
