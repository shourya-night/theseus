"""Tests for coordinate systems and transformations."""

import math

import numpy as np
import pytest

from theseus.coordinates.frames import ReferenceFrame
from theseus.coordinates.states import StateVector
from theseus.coordinates.transformations import (
    rotation_x, rotation_y, rotation_z,
    gmst_from_jd,
    eci_to_ecef, ecef_to_eci,
    cartesian_to_spherical, spherical_to_cartesian,
    perifocal_to_eci_matrix, eci_to_perifocal_matrix,
    eci_to_rtn_matrix,
)


class TestRotationMatrices:

    def test_rotation_identity(self):
        """Rotation by 0 is identity."""
        np.testing.assert_allclose(rotation_x(0), np.eye(3), atol=1e-15)
        np.testing.assert_allclose(rotation_y(0), np.eye(3), atol=1e-15)
        np.testing.assert_allclose(rotation_z(0), np.eye(3), atol=1e-15)

    def test_rotation_orthogonality(self):
        """R · R^T = I for arbitrary angle."""
        for angle in [0.3, 1.0, math.pi / 4, math.pi]:
            for R_fn in [rotation_x, rotation_y, rotation_z]:
                R = R_fn(angle)
                np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)

    def test_rotation_determinant(self):
        """det(R) = 1 (proper rotation)."""
        for angle in [0.5, 1.5, -0.7]:
            for R_fn in [rotation_x, rotation_y, rotation_z]:
                assert np.linalg.det(R_fn(angle)) == pytest.approx(1.0, abs=1e-14)

    def test_rotation_z_90(self):
        """Rz(90°) maps x̂ to ŷ."""
        R = rotation_z(math.pi / 2)
        result = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-14)

    def test_rotation_x_90(self):
        """Rx(90°) maps ŷ to ẑ."""
        R = rotation_x(math.pi / 2)
        result = R @ np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], atol=1e-14)


class TestEciEcef:

    def test_roundtrip(self):
        """ECI → ECEF → ECI is identity."""
        jd = 2_451_545.0
        pos_eci = np.array([7000e3, 1000e3, 0.0])
        pos_ecef = eci_to_ecef(pos_eci, jd)
        pos_back = ecef_to_eci(pos_ecef, jd)
        np.testing.assert_allclose(pos_back, pos_eci, atol=1e-6)

    def test_magnitude_preserved(self):
        """Rotation preserves vector magnitude."""
        jd = 2_460_000.0
        pos = np.array([42164e3, 0.0, 0.0])
        pos_ecef = eci_to_ecef(pos, jd)
        assert np.linalg.norm(pos_ecef) == pytest.approx(np.linalg.norm(pos), rel=1e-12)

    def test_physical_eastward_direction(self):
        """
        Verify physical orientation: Greenwich station on Earth equator rotates eastward.
        At GMST theta, Greenwich is at [R*cos(theta), R*sin(theta), 0] in ECI.
        Converting this ECI vector to ECEF must yield [R, 0, 0].
        """
        from theseus.coordinates.transformations import gmst_from_jd
        jd = 2_451_545.0 + 0.25  # 6 hours after J2000
        theta = gmst_from_jd(jd)
        r = 6378137.0
        # Greenwich in ECI is at +theta relative to vernal equinox
        pos_eci = np.array([r * math.cos(theta), r * math.sin(theta), 0.0])
        pos_ecef = eci_to_ecef(pos_eci, jd)
        np.testing.assert_allclose(pos_ecef, [r, 0.0, 0.0], atol=1e-6)

        # Conversely, Greenwich in ECEF is [r, 0, 0] -> ECI must be [r*cos(theta), r*sin(theta), 0]
        pos_eci_back = ecef_to_eci(np.array([r, 0.0, 0.0]), jd)
        np.testing.assert_allclose(pos_eci_back, pos_eci, atol=1e-6)


class TestCartesianSpherical:

    def test_x_axis(self):
        """(r, 0, 0) → r=r, θ=0, φ=0."""
        r, theta, phi = cartesian_to_spherical(np.array([5.0, 0.0, 0.0]))
        assert r == pytest.approx(5.0)
        assert theta == pytest.approx(0.0)
        assert phi == pytest.approx(0.0)

    def test_z_axis(self):
        """(0, 0, r) → θ = π/2."""
        r, theta, phi = cartesian_to_spherical(np.array([0.0, 0.0, 10.0]))
        assert r == pytest.approx(10.0)
        assert theta == pytest.approx(math.pi / 2)

    def test_roundtrip(self):
        xyz = np.array([3.0, 4.0, 5.0])
        r, theta, phi = cartesian_to_spherical(xyz)
        xyz_back = spherical_to_cartesian(r, theta, phi)
        np.testing.assert_allclose(xyz_back, xyz, atol=1e-12)

    def test_roundtrip_negative_z(self):
        xyz = np.array([1.0, -2.0, -3.0])
        r, theta, phi = cartesian_to_spherical(xyz)
        xyz_back = spherical_to_cartesian(r, theta, phi)
        np.testing.assert_allclose(xyz_back, xyz, atol=1e-12)


class TestPerifocalEci:

    def test_identity_for_zero_angles(self):
        """When Ω=i=ω=0, perifocal = ECI."""
        R = perifocal_to_eci_matrix(0.0, 0.0, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-14)

    def test_orthogonality(self):
        R = perifocal_to_eci_matrix(0.3, 0.5, 0.7)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)

    def test_inverse_is_transpose(self):
        raan, inc, argp = 1.0, 0.8, 0.5
        R_fwd = perifocal_to_eci_matrix(raan, inc, argp)
        R_inv = eci_to_perifocal_matrix(raan, inc, argp)
        np.testing.assert_allclose(R_fwd @ R_inv, np.eye(3), atol=1e-14)


class TestRTN:

    def test_rtn_orthonormal(self):
        """RTN axes form an orthonormal basis."""
        r = np.array([7000e3, 0.0, 0.0])
        v = np.array([0.0, 7500.0, 0.0])
        M = eci_to_rtn_matrix(r, v)
        np.testing.assert_allclose(M @ M.T, np.eye(3), atol=1e-12)

    def test_radial_direction(self):
        """R̂ should be along the position vector."""
        r = np.array([7000e3, 0.0, 0.0])
        v = np.array([0.0, 7500.0, 0.0])
        M = eci_to_rtn_matrix(r, v)
        r_hat = M[0]
        expected = r / np.linalg.norm(r)
        np.testing.assert_allclose(r_hat, expected, atol=1e-12)


class TestStateVector:

    def test_construction(self):
        sv = StateVector(
            position=np.array([7000e3, 0.0, 0.0]),
            velocity=np.array([0.0, 7500.0, 0.0]),
            frame=ReferenceFrame.ICRF,
        )
        assert sv.r == pytest.approx(7000e3)
        assert sv.v == pytest.approx(7500.0)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            StateVector(position=np.array([1, 2]), velocity=np.array([3, 4, 5]))

    def test_nan_raises(self):
        with pytest.raises(ValueError):
            StateVector(position=np.array([float("nan"), 0, 0]), velocity=np.array([0, 0, 0]))
