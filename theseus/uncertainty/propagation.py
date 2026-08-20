"""
Covariance propagation module for THESEUS.

Propagates 6×6 state covariance through nonlinear astrodynamics using
the State Transition Matrix:
    P(t) = Φ(t, t₀) P₀ Φ(t, t₀)ᵀ + Q(t, t₀)

Synchronized with nominal trajectory force models, central body, and epoch.
Includes an optional, explicit process noise model Q (disabled by default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from theseus.uncertainty.covariance import StateCovariance
from theseus.uncertainty.state_transition import propagate_stm, STMResult


@dataclass
class ProcessNoiseModel:
    """
    Optional process-noise model for unmodeled physical perturbations.

    Attributes
    ----------
    enabled : bool
        Whether process noise is added (default: False).
    q_matrix : np.ndarray | None
        6×6 process noise covariance matrix Q (SI units).
    q_spectral_density : np.ndarray | None
        3×3 acceleration noise spectral density Q_acc (m²/s³).
    source : str
        Source or justification for noise parameters.
    assumptions : list[str]
        Explicit scientific assumptions.
    """
    enabled: bool = False
    q_matrix: Optional[np.ndarray] = None
    q_spectral_density: Optional[np.ndarray] = None
    source: str = "Unspecified / User Supplied"
    assumptions: list[str] = field(default_factory=lambda: [
        "Process noise represents unmodeled atmospheric density fluctuations and solar activity",
        "White-noise acceleration assumption",
    ])

    def compute_q(self, dt: float, stm: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the discrete process noise matrix Q for a time step dt (s)."""
        if not self.enabled:
            return np.zeros((6, 6), dtype=np.float64)

        if self.q_matrix is not None:
            return np.asarray(self.q_matrix, dtype=np.float64)

        if self.q_spectral_density is not None:
            # Discrete approximation for white acceleration noise:
            # Q_rr = (1/3) * Q_acc * dt^3
            # Q_rv = (1/2) * Q_acc * dt^2
            # Q_vv = Q_acc * dt
            q_acc = np.asarray(self.q_spectral_density, dtype=np.float64)
            q = np.zeros((6, 6), dtype=np.float64)
            q[:3, :3] = (1.0 / 3.0) * (dt ** 3) * q_acc
            q[:3, 3:6] = (0.5) * (dt ** 2) * q_acc
            q[3:6, :3] = (0.5) * (dt ** 2) * q_acc
            q[3:6, 3:6] = dt * q_acc
            return q

        return np.zeros((6, 6), dtype=np.float64)


@dataclass
class CovariancePropagationResult:
    """
    Result of a covariance propagation.

    Attributes
    ----------
    initial_covariance : StateCovariance
        Covariance at t0.
    propagated_covariance : StateCovariance
        Covariance at tf.
    stm_result : STMResult
        Underlying State Transition Matrix solution.
    process_noise : ProcessNoiseModel
        Process noise configuration used.
    times : np.ndarray | None
        Time history points (s).
    history_covariances : list[StateCovariance] | None
        Covariance history along trajectory if computed.
    """
    initial_covariance: StateCovariance
    propagated_covariance: StateCovariance
    stm_result: STMResult
    process_noise: ProcessNoiseModel
    times: Optional[np.ndarray] = None
    history_covariances: Optional[list[StateCovariance]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_covariance": self.initial_covariance.to_dict(),
            "propagated_covariance": self.propagated_covariance.to_dict(),
            "stm": self.stm_result.to_dict(),
            "process_noise_enabled": self.process_noise.enabled,
            "process_noise_source": self.process_noise.source,
            "initial_sigma_pos_km": self.initial_covariance.sigma_pos_3d / 1e3,
            "final_sigma_pos_km": self.propagated_covariance.sigma_pos_3d / 1e3,
            "initial_sigma_vel_km_s": self.initial_covariance.sigma_vel_3d / 1e3,
            "final_sigma_vel_km_s": self.propagated_covariance.sigma_vel_3d / 1e3,
        }


def propagate_covariance(
    initial_covariance: StateCovariance,
    stm: np.ndarray,
    epoch_tf_s: float,
    process_noise: Optional[ProcessNoiseModel] = None,
    name: Optional[str] = None,
) -> StateCovariance:
    """
    Map a covariance matrix across a time step using a known State Transition Matrix.

        P(t) = Φ P₀ Φᵀ + Q

    Parameters
    ----------
    initial_covariance : StateCovariance
        Covariance P₀ at t₀.
    stm : (6, 6) array
        State Transition Matrix Φ(t, t₀).
    epoch_tf_s : float
        Final epoch time (s).
    process_noise : ProcessNoiseModel, optional
        Process noise model.
    name : str, optional
        Target object identifier.

    Returns
    -------
    StateCovariance
    """
    phi = np.asarray(stm, dtype=np.float64)
    p0 = initial_covariance.matrix

    # Linear covariance mapping
    p_prop = phi @ p0 @ phi.T

    # Add process noise if active
    dt = epoch_tf_s - initial_covariance.epoch_s
    if process_noise is not None and process_noise.enabled:
        q = process_noise.compute_q(dt, phi)
        p_prop = p_prop + q

    # Symmetrize explicitly to prevent floating-point asymmetry build-up
    p_prop = 0.5 * (p_prop + p_prop.T)

    # Ensure numerical positive semi-definiteness against float precision roundoff
    eigvals, eigvecs = np.linalg.eigh(p_prop)
    if np.any(eigvals < 0.0):
        eigvals = np.maximum(eigvals, 0.0)
        p_prop = eigvecs @ np.diag(eigvals) @ eigvecs.T
        p_prop = 0.5 * (p_prop + p_prop.T)

    return StateCovariance(

        matrix=p_prop,
        epoch_s=epoch_tf_s,
        frame=initial_covariance.frame,
        pos_units=initial_covariance.pos_units,
        vel_units=initial_covariance.vel_units,
        source="SIMULATION_PROPAGATED",
        name=name or initial_covariance.name,
    )


class CovariancePropagator:
    """
    High-level covariance propagator tied to orbit dynamics.

    Parameters
    ----------
    acc_fn : callable
        Function (t, r, v) -> a (3,) [m/s²].
    mu : float, optional
        Central body gravitational parameter.
    j2 : float, optional
        Zonal harmonic J2.
    radius : float, optional
        Body equatorial radius.
    process_noise : ProcessNoiseModel, optional
        Process noise configuration.
    integrator : str
        'rkf45' or 'rk4'.
    """

    def __init__(
        self,
        acc_fn: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
        mu: Optional[float] = None,
        j2: Optional[float] = None,
        radius: Optional[float] = None,
        process_noise: Optional[ProcessNoiseModel] = None,
        integrator: str = "rkf45",
        dt: float = 60.0,
        atol: float = 1e-11,
        rtol: float = 1e-11,
    ) -> None:
        self.acc_fn = acc_fn
        self.mu = mu
        self.j2 = j2
        self.radius = radius
        self.process_noise = process_noise or ProcessNoiseModel(enabled=False)
        self.integrator = integrator
        self.dt = dt
        self.atol = atol
        self.rtol = rtol

    def propagate(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        initial_covariance: StateCovariance,
        t_span: tuple[float, float],
    ) -> CovariancePropagationResult:
        """
        Propagate nominal state and covariance from t0 to tf.

        Returns
        -------
        CovariancePropagationResult
        """
        t0, tf = float(t_span[0]), float(t_span[1])

        stm_res = propagate_stm(
            acc_fn=self.acc_fn,
            r0=r0,
            v0=v0,
            t_span=(t0, tf),
            mu=self.mu,
            j2=self.j2,
            radius=self.radius,
            integrator=self.integrator,
            dt=self.dt,
            atol=self.atol,
            rtol=self.rtol,
        )

        p_final = propagate_covariance(
            initial_covariance=initial_covariance,
            stm=stm_res.stm,
            epoch_tf_s=tf,
            process_noise=self.process_noise,
        )

        # If time history is available in stm_res, also compute covariance history
        history_covs: Optional[list[StateCovariance]] = None
        if stm_res.history_times is not None and stm_res.history_stms is not None:
            history_covs = []
            for i, t_step in enumerate(stm_res.history_times):
                phi_step = stm_res.history_stms[i]
                cov_step = propagate_covariance(
                    initial_covariance=initial_covariance,
                    stm=phi_step,
                    epoch_tf_s=t_step,
                    process_noise=self.process_noise,
                )
                history_covs.append(cov_step)

        return CovariancePropagationResult(
            initial_covariance=initial_covariance,
            propagated_covariance=p_final,
            stm_result=stm_res,
            process_noise=self.process_noise,
            times=stm_res.history_times,
            history_covariances=history_covs,
        )
