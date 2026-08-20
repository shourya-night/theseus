"""
State covariance representation and validation for THESEUS.

Provides the 6×6 Cartesian state covariance class with strict mathematical
validation:
- Shape and finiteness (no NaN/Inf)
- Matrix symmetry (with explicit numerical symmetrization for float noise)
- Positive semi-definiteness (eigenvalue check)
- Physical variance non-negativity
- Reference frame, epoch, and unit metadata tracking
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


class CovarianceValidationError(ValueError):
    """Raised when a covariance matrix violates physical or mathematical validity."""
    pass


@dataclass
class StateCovariance:
    """
    Rigorous 6×6 Cartesian state covariance matrix representation.

    State vector definition:
        x = [rx, ry, rz, vx, vy, vz]^T

    Covariance definition:
        P = E[(x - x̄)(x - x̄)^T] =
        | P_rr (3×3)  P_rv (3×3) |
        | P_vr (3×3)  P_vv (3×3) |

    Attributes
    ----------
    matrix : np.ndarray
        6×6 symmetric positive-semidefinite covariance matrix.
    epoch_s : float
        Epoch time in seconds from simulation origin.
    frame : str
        Reference frame (e.g. 'ICRF', 'GCRF', 'TEME', 'RIC').
    pos_units : str
        Position variance units (default: 'm^2', position standard deviation in 'm').
    vel_units : str
        Velocity variance units (default: 'm^2/s^2', velocity standard deviation in 'm/s').
    source : str
        Provenance of covariance: 'USER_PROVIDED', 'SIMULATION_GENERATED',
        'ESTIMATION_OUTPUT', 'REFERENCE_PRESET'.
    name : str
        Optional identifier for the tracked object.
    sym_tol : float
        Relative tolerance for symmetry verification.
    psd_tol : float
        Absolute tolerance for negative eigenvalues due to floating-point precision.
    """
    matrix: np.ndarray
    epoch_s: float = 0.0
    frame: str = "ICRF"
    pos_units: str = "m"
    vel_units: str = "m/s"
    source: str = "USER_PROVIDED"
    name: Optional[str] = None
    sym_tol: float = 1e-7
    psd_tol: float = 1e-9

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=np.float64)
        self.validate()

    def validate(self) -> None:
        """
        Perform strict validation of the 6×6 covariance matrix.

        Raises
        ------
        CovarianceValidationError
            If the matrix is not 6×6, contains non-finite values, has negative
            diagonal entries, is asymmetric beyond tolerance, or has negative
            eigenvalues beyond numerical tolerance.
        """
        # 1. Dimension check
        if self.matrix.shape != (6, 6):
            raise CovarianceValidationError(
                f"COVARIANCE INVALID: Expected 6×6 matrix, got shape {self.matrix.shape}"
            )

        # 2. Finite values check
        if not np.all(np.isfinite(self.matrix)):
            nan_count = int(np.isnan(self.matrix).sum())
            inf_count = int(np.isinf(self.matrix).sum())
            raise CovarianceValidationError(
                f"COVARIANCE INVALID: Matrix contains non-finite values ({nan_count} NaNs, {inf_count} Infs)"
            )

        # 3. Non-negative diagonal variances check
        diag = np.diag(self.matrix)
        for i, val in enumerate(diag):
            var_name = ["rx", "ry", "rz", "vx", "vy", "vz"][i]
            if val < -1e-15:
                raise CovarianceValidationError(
                    f"COVARIANCE INVALID: Negative variance on diagonal for {var_name}: {val:.6e}"
                )
            if val < 0.0:  # slight negative float noise
                self.matrix[i, i] = 0.0

        # 4. Symmetry check
        max_abs = float(np.max(np.abs(self.matrix)))
        scale = max(max_abs, 1.0)
        asym = np.abs(self.matrix - self.matrix.T)
        max_asym = float(np.max(asym))
        rel_asym = max_asym / scale

        if rel_asym > self.sym_tol:
            raise CovarianceValidationError(
                f"COVARIANCE INVALID: Gross asymmetry detected. Max asymmetry |P - P^T| = {max_asym:.6e} "
                f"(relative: {rel_asym:.6e}, tolerance: {self.sym_tol:.6e})"
            )

        # Explicitly symmetrize within tolerance to eliminate floating-point shear
        self.matrix = 0.5 * (self.matrix + self.matrix.T)

        # 5. Positive semi-definiteness (eigenvalues >= -effective_psd_tol)
        eigenvalues, eigvecs = np.linalg.eigh(self.matrix)
        min_eig = float(np.min(eigenvalues))
        effective_psd_tol = max(self.psd_tol, scale * 1e-9)

        if min_eig < -effective_psd_tol:
            raise CovarianceValidationError(
                f"COVARIANCE INVALID: Matrix is not positive semi-definite. "
                f"Negative eigenvalue detected: λ_min = {min_eig:.6e} (tolerance: {-effective_psd_tol:.6e})"
            )
        
        # Eliminate tiny negative numerical roundoff within tolerance
        if min_eig < 0.0:
            eigenvalues = np.maximum(eigenvalues, 0.0)
            self.matrix = eigvecs @ np.diag(eigenvalues) @ eigvecs.T
            self.matrix = 0.5 * (self.matrix + self.matrix.T)


    # -----------------------------------------------------------------------
    # Sub-block extractors
    # -----------------------------------------------------------------------

    @property
    def position_covariance(self) -> np.ndarray:
        """3×3 position-position covariance P_rr (m²)."""
        return self.matrix[:3, :3].copy()

    @property
    def velocity_covariance(self) -> np.ndarray:
        """3×3 velocity-velocity covariance P_vv (m²/s²)."""
        return self.matrix[3:6, 3:6].copy()

    @property
    def pos_vel_covariance(self) -> np.ndarray:
        """3×3 position-velocity cross-covariance P_rv (m²/s)."""
        return self.matrix[:3, 3:6].copy()

    @property
    def vel_pos_covariance(self) -> np.ndarray:
        """3×3 velocity-position cross-covariance P_vr (m²/s)."""
        return self.matrix[3:6, :3].copy()

    # -----------------------------------------------------------------------
    # Standard deviation statistics
    # -----------------------------------------------------------------------

    @property
    def sigma_position(self) -> np.ndarray:
        """1-sigma position standard deviations [σ_x, σ_y, σ_z] (m)."""
        return np.sqrt(np.maximum(0.0, np.diag(self.matrix[:3, :3])))

    @property
    def sigma_velocity(self) -> np.ndarray:
        """1-sigma velocity standard deviations [σ_vx, σ_vy, σ_vz] (m/s)."""
        return np.sqrt(np.maximum(0.0, np.diag(self.matrix[3:6, 3:6])))

    @property
    def sigma_pos_3d(self) -> float:
        """Scalar 3D position uncertainty: sqrt(Tr(P_rr)) (m)."""
        return float(np.sqrt(max(0.0, np.trace(self.matrix[:3, :3]))))

    @property
    def sigma_vel_3d(self) -> float:
        """Scalar 3D velocity uncertainty: sqrt(Tr(P_vv)) (m/s)."""
        return float(np.sqrt(max(0.0, np.trace(self.matrix[3:6, 3:6]))))

    # -----------------------------------------------------------------------
    # Construction helpers
    # -----------------------------------------------------------------------

    @classmethod
    def from_diagonal(
        cls,
        sigma_pos: np.ndarray | list[float],
        sigma_vel: np.ndarray | list[float],
        epoch_s: float = 0.0,
        frame: str = "ICRF",
        source: str = "USER_PROVIDED",
        name: Optional[str] = None,
    ) -> StateCovariance:
        """
        Create a diagonal covariance matrix from 1-sigma standard deviations.

        Parameters
        ----------
        sigma_pos : 3-element sequence
            Position 1-sigma standard deviations [σ_x, σ_y, σ_z] (m).
        sigma_vel : 3-element sequence
            Velocity 1-sigma standard deviations [σ_vx, σ_vy, σ_vz] (m/s).
        """
        sp = np.asarray(sigma_pos, dtype=np.float64)
        sv = np.asarray(sigma_vel, dtype=np.float64)
        if len(sp) != 3 or len(sv) != 3:
            raise ValueError("sigma_pos and sigma_vel must each contain exactly 3 values")

        diag_vars = np.concatenate([sp ** 2, sv ** 2])
        matrix = np.diag(diag_vars)

        return cls(
            matrix=matrix,
            epoch_s=epoch_s,
            frame=frame,
            source=source,
            name=name,
        )

    @classmethod
    def from_isotropic(
        cls,
        sigma_pos_m: float,
        sigma_vel_m_s: float,
        epoch_s: float = 0.0,
        frame: str = "ICRF",
        source: str = "USER_PROVIDED",
        name: Optional[str] = None,
    ) -> StateCovariance:
        """Create an isotropic diagonal covariance matrix."""
        return cls.from_diagonal(
            sigma_pos=[sigma_pos_m, sigma_pos_m, sigma_pos_m],
            sigma_vel=[sigma_vel_m_s, sigma_vel_m_s, sigma_vel_m_s],
            epoch_s=epoch_s,
            frame=frame,
            source=source,
            name=name,
        )

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-safe dictionary with SI and km units."""
        return {
            "matrix_si": self.matrix.tolist(),
            "epoch_s": float(self.epoch_s),
            "frame": self.frame,
            "source": self.source,
            "name": self.name,
            "sigma_position_m": self.sigma_position.tolist(),
            "sigma_velocity_m_s": self.sigma_velocity.tolist(),
            "sigma_position_km": (self.sigma_position / 1e3).tolist(),
            "sigma_velocity_km_s": (self.sigma_velocity / 1e3).tolist(),
            "sigma_pos_3d_m": self.sigma_pos_3d,
            "sigma_pos_3d_km": self.sigma_pos_3d / 1e3,
            "sigma_vel_3d_m_s": self.sigma_vel_3d,
            "sigma_vel_3d_km_s": self.sigma_vel_3d / 1e3,
            "trace_pos_m2": float(np.trace(self.matrix[:3, :3])),
            "trace_vel_m2_s2": float(np.trace(self.matrix[3:6, 3:6])),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateCovariance:
        """Reconstruct from dictionary."""
        matrix = np.array(data["matrix_si"], dtype=np.float64)
        return cls(
            matrix=matrix,
            epoch_s=float(data.get("epoch_s", 0.0)),
            frame=str(data.get("frame", "ICRF")),
            source=str(data.get("source", "USER_PROVIDED")),
            name=data.get("name"),
        )
