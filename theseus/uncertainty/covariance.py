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


#: Default tolerance on the smallest eigenvalue of the *correlation* form
#: D⁻¹PD⁻¹, where D = diag(sqrt(P_ii)).
#:
#: Dimensionless and invariant under a component-wise rescaling of the state,
#: which the previous raw-eigenvalue tolerance was not.  Calibrated: 4 000
#: exactly-singular correlation matrices (smallest eigenvalue zero by
#: construction) produced no measured value below -1.22e-15, matching the
#: backward-error bound eps·‖C‖₂ = 1.33e-15 for a 6×6; 35 genuine Phase 10
#: covariances sampled from this repository sit no closer than +3.67e-06.
PSD_CORRELATION_TOL = 1e-12

#: Relative floor below which a negative diagonal entry is treated as roundoff
#: rather than as an invalid variance, measured against the largest variance in
#: the same block (position or velocity).
#:
#: Dimensionless by construction, so the verdict cannot depend on whether the
#: caller works in metres or kilometres.  Same order as PSD_CORRELATION_TOL and
#: justified the same way: forming P = Phi P0 Phi^T commits a backward error
#: bounded by a small multiple of eps * ||P||, so a diagonal entry below this
#: relative to its own block carries no information about sign.
DIAGONAL_NOISE_RTOL = 1e-12


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
        Relative tolerance for symmetry verification, applied to each entry
        against its own scale sqrt(P_ii P_jj) — i.e. to the asymmetry of the
        correlation matrix. Dimensionless.
    psd_tol : float
        Tolerance for negative eigenvalues due to floating-point precision,
        applied to the **correlation form** D⁻¹PD⁻¹ rather than to the raw
        matrix, so it is dimensionless and unchanged by a rescaling of the
        state components.

        Its default follows from measurement rather than preference: over 4 000
        exactly-singular correlation matrices, whose smallest eigenvalue is zero
        by construction, `eigvalsh` returned no value below -1.22e-15, in
        agreement with the backward-error bound eps·‖C‖₂ = 1.33e-15 for a 6×6
        correlation matrix. Genuine Phase 10 covariances measured in this
        repository sit no closer to the boundary than +3.67e-06. A tolerance of
        1e-12 therefore sits about 750× above the roundoff floor and about
        3.7e6× below real data, so it can neither reject legitimate roundoff nor
        admit a correlation exceeding unity by more than 1e-12.
    """
    matrix: np.ndarray
    epoch_s: float = 0.0
    frame: str = "ICRF"
    pos_units: str = "m"
    vel_units: str = "m/s"
    source: str = "USER_PROVIDED"
    name: Optional[str] = None
    sym_tol: float = 1e-7
    psd_tol: float = PSD_CORRELATION_TOL

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
        #
        # The threshold used to be the literal -1e-15, which is an absolute
        # value in whatever units the caller chose: m^2 for the position block,
        # (m/s)^2 for the velocity block.  A variance is not dimensionless, so
        # no absolute number can be right for both, and the same physical
        # covariance changed verdict with the unit system.  Measured on one
        # matrix with sigma_r = 10 m and a genuinely negative P[0,0] of
        # -1e-18 sigma^2:
        #
        #     length unit  metres     P[0,0] = -1.0e-14  -> REJECTED
        #     length unit  kilometres P[0,0] = -1.0e-20  -> accepted
        #     length unit  megametres P[0,0] = -1.0e-26  -> accepted
        #
        # The same defect P10-10 removed from the collision-probability branch
        # criteria and P10-11 removed from the PSD test, still present here.
        #
        # The scale that makes it dimensionless is the covariance's own: a
        # diagonal entry is compared against the largest variance in its own
        # block, which is what the correlation form does.  Roundoff in forming
        # P = Phi P0 Phi^T is bounded by a small multiple of eps times the norm
        # of the result, so DIAGONAL_NOISE_RTOL is the same order as the
        # correlation-form PSD tolerance and justified the same way.
        diag = np.diag(self.matrix)
        block_scale = self._diagonal_block_scales(diag)
        for i, val in enumerate(diag):
            var_name = ["rx", "ry", "rz", "vx", "vy", "vz"][i]
            noise_floor = DIAGONAL_NOISE_RTOL * block_scale[i]
            if val < -noise_floor:
                raise CovarianceValidationError(
                    f"COVARIANCE INVALID: Negative variance on diagonal for "
                    f"{var_name}: {val:.6e} "
                    f"(more negative than {noise_floor:.6e}, which is "
                    f"{DIAGONAL_NOISE_RTOL:.0e} of the largest variance in its "
                    f"block, {block_scale[i]:.6e})"
                )
            if val < 0.0:  # slight negative float noise
                self.matrix[i, i] = 0.0

        # 4. Symmetry check, normalised per entry
        #
        # This used to be `max|P - P^T| / max(max|P|, 1.0)`.  The denominator is
        # the largest entry of the whole 6x6, which for a state covariance is a
        # position variance in m^2, while the numerator may be an asymmetry in
        # the velocity block in (m/s)^2.  With sigma_r = 1 km and
        # sigma_v = 1e-4 m/s that let an asymmetry of 10 000 % of the velocity
        # variance pass, while at sigma_r = sigma_v = 1 a 1 % asymmetry was
        # rejected -- the same physical defect judged differently according to
        # an unrelated block's magnitude.
        #
        # Each entry is now compared against its own scale, sqrt(P_ii P_jj),
        # which is the only quantity with the units of P_ij.  The result is the
        # asymmetry of the correlation matrix, so `sym_tol` keeps its documented
        # meaning of a *relative* tolerance and simply becomes relative to the
        # right thing.
        max_asym, rel_asym = self._normalised_asymmetry()

        if rel_asym > self.sym_tol:
            raise CovarianceValidationError(
                f"COVARIANCE INVALID: Gross asymmetry detected. Max asymmetry |P - P^T| = {max_asym:.6e} "
                f"(relative: {rel_asym:.6e}, tolerance: {self.sym_tol:.6e})"
            )

        # Explicitly symmetrize within tolerance to eliminate floating-point shear
        self.matrix = 0.5 * (self.matrix + self.matrix.T)

        # 5. Positive semi-definiteness, tested in correlation form
        #
        # The previous test was `min_eig < -max(psd_tol, max(max|P|, 1.0) * 1e-9)`
        # on the raw eigenvalues.  Positive semi-definiteness is invariant under
        # congruence P -> S P S^T, so expressing the same physical covariance in
        # different state units must not change the verdict -- but that test
        # made it do exactly that.  Measured: a covariance with a correlation
        # coefficient of 2.0 between r_x and v_x (impossible for any
        # distribution) was accepted with sigma_r = 1 km, sigma_v = 1e-4 m/s,
        # because the tolerance was 1e-3 m^2 while the offending eigenvalue was
        # -3.0e-8.  The identical matrix in km and mm/s was rejected.  At
        # sigma_r = 1 km the raw eigenvalue could not even be resolved: at a
        # correlation of 1.00001 `eigh` returned +1.15e-11, positive, so no
        # absolute tolerance however tight would have caught it.
        #
        # The test therefore runs on the correlation form C = D^-1 P D^-1,
        # D = diag(sqrt(P_ii)).  D is nonsingular where the variances are
        # positive, so C is PSD exactly when P is; C is dimensionless; and
        # normalising by P's own diagonal introduces no scale from outside the
        # matrix.  A zero variance is handled separately below, because
        # Cauchy-Schwarz then forces the whole row to vanish.
        self._reject_covariance_with_zero_variance()
        min_corr_eig = self._correlation_min_eigenvalue()

        # Eigenvalues without eigenvectors: the raw spectrum is needed only to
        # report alongside a rejection, and to decide whether the roundoff
        # repair below has anything to do.  The eigenvectors are computed only
        # if that repair actually runs.
        min_eig = float(np.min(np.linalg.eigvalsh(self.matrix)))

        if min_corr_eig < -self.psd_tol:
            raise CovarianceValidationError(
                f"COVARIANCE INVALID: Matrix is not positive semi-definite. "
                f"Negative eigenvalue detected: λ_min(correlation form) = "
                f"{min_corr_eig:.6e} (tolerance: {-self.psd_tol:.6e}); "
                f"λ_min(raw) = {min_eig:.6e}"
            )

        # Eliminate tiny negative numerical roundoff within tolerance
        if min_eig < 0.0:
            eigenvalues, eigvecs = np.linalg.eigh(self.matrix)
            eigenvalues = np.maximum(eigenvalues, 0.0)
            self.matrix = eigvecs @ np.diag(eigenvalues) @ eigvecs.T
            self.matrix = 0.5 * (self.matrix + self.matrix.T)


    # -----------------------------------------------------------------------
    # Dimensionless validity helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _diagonal_block_scales(diag: np.ndarray) -> np.ndarray:
        """
        For each diagonal entry, the largest variance in its own block.

        Position and velocity variances carry different units, so a single
        scale for the whole 6x6 would reintroduce exactly the unit dependence
        this replaces -- with sigma_r = 1 km and sigma_v = 1e-4 m/s the
        position block is twenty orders of magnitude larger, and a velocity
        variance would be compared against a position one.

        A block whose variances are all zero or negative has no scale of its
        own; the other block's is not meaningful for it either, so such an
        entry falls back to a bare zero floor -- any negative value there is
        reported rather than absorbed.
        """
        scales = np.zeros(6, dtype=np.float64)
        for lo, hi in ((0, 3), (3, 6)):
            block = np.asarray(diag[lo:hi], dtype=np.float64)
            positive = block[block > 0.0]
            scales[lo:hi] = float(np.max(positive)) if positive.size else 0.0
        return scales

    def _positive_variance_mask(self) -> np.ndarray:
        """Indices whose variance is strictly positive, so a correlation exists."""
        return np.diag(self.matrix) > 0.0

    def _normalised_asymmetry(self) -> tuple[float, float]:
        """
        ``(max |P_ij - P_ji|, max |P_ij - P_ji| / sqrt(P_ii P_jj))``.

        The second is the asymmetry of the correlation matrix: dimensionless,
        and unchanged when the state is rescaled component-wise.
        """
        asym = np.abs(self.matrix - self.matrix.T)
        max_asym = float(np.max(asym))

        keep = self._positive_variance_mask()
        if np.count_nonzero(keep) < 2:
            return max_asym, 0.0 if max_asym == 0.0 else float("inf")

        deviations = np.sqrt(np.diag(self.matrix)[keep])
        normalised = asym[np.ix_(keep, keep)] / np.outer(deviations, deviations)
        return max_asym, float(np.max(normalised))

    def _reject_covariance_with_zero_variance(self) -> None:
        """
        A component with zero variance may not covary with anything.

        Cauchy-Schwarz gives |P_ij|^2 <= P_ii P_jj, so P_ii = 0 forces P_ij = 0
        for every j; any other value makes the matrix indefinite.  The bound is
        exact and carries no tolerance, because the only quantity with the units
        of P_ij here is sqrt(P_ii P_jj), which is zero.  The correlation form
        cannot see this case -- the row has no correlation to normalise -- so it
        is tested explicitly.
        """
        variances = np.diag(self.matrix)
        for i in np.flatnonzero(variances <= 0.0):
            row = np.abs(self.matrix[i]).copy()
            row[i] = 0.0
            worst = float(np.max(row))
            if worst > 0.0:
                name = ["rx", "ry", "rz", "vx", "vy", "vz"][int(i)]
                partner = ["rx", "ry", "rz", "vx", "vy", "vz"][int(np.argmax(row))]
                raise CovarianceValidationError(
                    f"COVARIANCE INVALID: {name} has zero variance but covaries "
                    f"with {partner} (P = {worst:.6e}). Cauchy-Schwarz requires "
                    f"|P_ij|² ≤ P_ii P_jj = 0, so this matrix is not positive "
                    f"semi-definite."
                )

    def _correlation_min_eigenvalue(self) -> float:
        """
        Smallest eigenvalue of the correlation form ``D⁻¹ P D⁻¹``.

        Positive semi-definiteness is invariant under congruence by the
        nonsingular diagonal ``D = diag(sqrt(P_ii))``, so this has the same sign
        as the raw minimum eigenvalue in exact arithmetic — but it is
        dimensionless, it is unchanged by rescaling the state components, and it
        is resolvable in floating point when the blocks span many orders of
        magnitude, which the raw eigenvalues are not.
        """
        keep = self._positive_variance_mask()
        if not np.any(keep):
            return 0.0

        deviations = np.sqrt(np.diag(self.matrix)[keep])
        correlation = self.matrix[np.ix_(keep, keep)] / np.outer(deviations, deviations)
        correlation = 0.5 * (correlation + correlation.T)
        return float(np.min(np.linalg.eigvalsh(correlation)))

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
