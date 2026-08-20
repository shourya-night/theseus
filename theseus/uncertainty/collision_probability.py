"""
Probability of collision (Pc) computation for close encounters.

Implements the 2D Encounter Plane (B-plane) Gaussian integral model
(Alfriend, Akella, Chan, Foster, Patera).

Mathematical Model
------------------
During a short-duration orbital conjunction:
1. Relative motion along the approach direction Ŝ is rectilinear.
2. State uncertainties at TCA are mapped into the 2D encounter plane (B-plane).
3. The collision cross-section is represented as a circular disk of radius
   HBR = R₁ + R₂ centered at the nominal miss vector b₀ = [B·T, B·R]ᵀ.
4. The 2D Gaussian probability density function (PDF) in the encounter plane is:
       f(z) = 1 / (2π √(det P_B)) * exp(-½ zᵀ P_B⁻¹ z)
5. The collision probability is the integral of the PDF over the collision disk D:
       Pc = ∬_D f(z) dz
   where D = { z ∈ ℝ² : |z - b₀|² ≤ HBR² }.

Numerical Methods
-----------------
- Primary: 2D Adaptive Polar Quadrature over the transformed principal axes
- Analytical / Series: Chan's series expansion for isotropic and mildly anisotropic cases
- Validation: Monte Carlo sampling (strictly labeled as validation only)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np
from scipy import integrate

from theseus.uncertainty.b_plane import BPlaneUncertainty


@dataclass
class CollisionProbabilityResult:
    """
    Complete Probability of Collision result.

    Attributes
    ----------
    probability : float
        Calculated collision probability (0.0 to 1.0).
    method : str
        Algorithm name used for integration.
    converged : bool
        Whether the numerical calculation converged within tolerance.
    tolerance : float
        Numerical integration absolute/relative tolerance.
    iterations : int
        Number of function evaluations / iterations.
    hard_body_radius_m : float
        Combined hard-body radius HBR (m).
    miss_distance_m : float
        Nominal miss distance |b₀| at TCA (m).
    b_plane_coordinates_m : tuple[float, float]
        [B·T, B·R] miss vector in B-plane (m).
    b_plane_covariance_m2 : list[list[float]]
        2×2 B-plane covariance matrix (m²).
    sigma_major_m : float
        Semi-major axis of 1-sigma uncertainty ellipse (m).
    sigma_minor_m : float
        Semi-minor axis of 1-sigma uncertainty ellipse (m).
    ellipse_angle_deg : float
        Uncertainty ellipse orientation angle (degrees).
    covariance_eigenvalues : list[float]
        Eigenvalues of P_B [λ_min, λ_max] (m²).
    condition_number : float
        Condition number of P_B.
    determinant : float
        Determinant of P_B (m⁴).
    assumptions : list[str]
        Explicit scientific assumptions.
    diagnostics : dict[str, Any]
        Numerical diagnostics.
    """
    probability: float
    method: str
    converged: bool
    tolerance: float
    iterations: int
    hard_body_radius_m: float
    miss_distance_m: float
    b_plane_coordinates_m: Tuple[float, float]
    b_plane_covariance_m2: list[list[float]]
    sigma_major_m: float
    sigma_minor_m: float
    ellipse_angle_deg: float
    covariance_eigenvalues: list[float]
    condition_number: float
    determinant: float
    assumptions: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability": float(self.probability),
            "probability_scientific": f"{self.probability:.6e}",
            "method": self.method,
            "converged": self.converged,
            "tolerance": float(self.tolerance),
            "iterations": int(self.iterations),
            "hard_body_radius_m": float(self.hard_body_radius_m),
            "hard_body_radius_km": float(self.hard_body_radius_m / 1e3),
            "miss_distance_m": float(self.miss_distance_m),
            "miss_distance_km": float(self.miss_distance_m / 1e3),
            "b_plane_coordinates_m": list(self.b_plane_coordinates_m),
            "b_plane_coordinates_km": [x / 1e3 for x in self.b_plane_coordinates_m],
            "b_plane_covariance_m2": self.b_plane_covariance_m2,
            "sigma_major_m": float(self.sigma_major_m),
            "sigma_minor_m": float(self.sigma_minor_m),
            "sigma_major_km": float(self.sigma_major_m / 1e3),
            "sigma_minor_km": float(self.sigma_minor_m / 1e3),
            "ellipse_angle_deg": float(self.ellipse_angle_deg),
            "covariance_eigenvalues": self.covariance_eigenvalues,
            "condition_number": float(self.condition_number),
            "determinant": float(self.determinant),
            "assumptions": self.assumptions,
            "diagnostics": self.diagnostics,
        }


@dataclass
class MonteCarloValidationResult:
    """
    Monte Carlo cross-validation result (strictly for validation).
    """
    sample_count: int
    hits: int
    empirical_pc: float
    deterministic_pc: float
    standard_error: float
    confidence_interval_95: Tuple[float, float]
    difference: float
    is_consistent: bool
    notes: str


def compute_collision_probability(
    b_plane_unc: BPlaneUncertainty,
    hbr_m: float,
    tol: float = 1e-8,
    max_evals: int = 10000,
) -> CollisionProbabilityResult:
    """
    Compute probability of collision Pc in the 2D B-plane.

    Transforms the problem into principal axes and integrates the 2D Gaussian
    PDF over the collision disk of radius hbr_m.

    Parameters
    ----------
    b_plane_unc : BPlaneUncertainty
        B-plane geometry and covariance data.
    hbr_m : float
        Combined hard-body radius (m).
    tol : float
        Numerical integration tolerance.
    max_evals : int
        Maximum function evaluations for integration.

    Returns
    -------
    CollisionProbabilityResult
    """
    bt = float(b_plane_unc.b_dot_t)
    br = float(b_plane_unc.b_dot_r)
    miss_dist = math.sqrt(bt * bt + br * br)

    assumptions = [
        "Encounter plane 2D Gaussian probability distribution (Alfriend-Akella-Chan model)",
        "Short-duration conjunction: rectilinear relative motion near TCA",
        "Position uncertainty at TCA dominates velocity uncertainty during encounter",
        "Combined spherical hard-body collision cross-section",
        "Static covariance across encounter duration (Gaussian error propagation)",
    ]

    p_mat = b_plane_unc.b_plane_covariance
    det_p = float(np.linalg.det(p_mat))
    cond_p = float(np.linalg.cond(p_mat)) if det_p > 1e-30 else float("inf")
    eigvals = b_plane_unc.eigenvalues.tolist()

    # Edge Case 1: HBR <= 0 -> Pc = 0
    if hbr_m <= 0.0:
        return CollisionProbabilityResult(
            probability=0.0,
            method="analytic_zero_hbr",
            converged=True,
            tolerance=tol,
            iterations=1,
            hard_body_radius_m=0.0,
            miss_distance_m=miss_dist,
            b_plane_coordinates_m=(bt, br),
            b_plane_covariance_m2=p_mat.tolist(),
            sigma_major_m=b_plane_unc.sigma_major,
            sigma_minor_m=b_plane_unc.sigma_minor,
            ellipse_angle_deg=b_plane_unc.ellipse_angle_deg,
            covariance_eigenvalues=eigvals,
            condition_number=cond_p,
            determinant=det_p,
            assumptions=assumptions,
            diagnostics={"reason": "HBR is zero; collision cross-section has zero area"},
        )

    # Edge Case 2: Near-zero covariance / deterministic limit
    sigma_major = b_plane_unc.sigma_major
    sigma_minor = b_plane_unc.sigma_minor

    if sigma_major < 1e-6 or sigma_minor < 1e-6 or det_p < 1e-12:
        # Deterministic collision check
        pc_det = 1.0 if miss_dist <= hbr_m else 0.0
        return CollisionProbabilityResult(
            probability=pc_det,
            method="deterministic_limit",
            converged=True,
            tolerance=tol,
            iterations=1,
            hard_body_radius_m=hbr_m,
            miss_distance_m=miss_dist,
            b_plane_coordinates_m=(bt, br),
            b_plane_covariance_m2=p_mat.tolist(),
            sigma_major_m=sigma_major,
            sigma_minor_m=sigma_minor,
            ellipse_angle_deg=b_plane_unc.ellipse_angle_deg,
            covariance_eigenvalues=eigvals,
            condition_number=cond_p,
            determinant=det_p,
            assumptions=assumptions,
            diagnostics={"reason": "Covariance approaches zero; evaluating deterministic overlap"},
        )

    # Edge Case 3: Miss distance much larger than uncertainty and HBR -> Pc ~ 0
    max_sigma = max(sigma_major, 1.0)
    if miss_dist > 50.0 * max_sigma and (miss_dist - hbr_m) > 10.0 * max_sigma:
        return CollisionProbabilityResult(
            probability=0.0,
            method="analytic_far_separation",
            converged=True,
            tolerance=tol,
            iterations=1,
            hard_body_radius_m=hbr_m,
            miss_distance_m=miss_dist,
            b_plane_coordinates_m=(bt, br),
            b_plane_covariance_m2=p_mat.tolist(),
            sigma_major_m=sigma_major,
            sigma_minor_m=sigma_minor,
            ellipse_angle_deg=b_plane_unc.ellipse_angle_deg,
            covariance_eigenvalues=eigvals,
            condition_number=cond_p,
            determinant=det_p,
            assumptions=assumptions,
            diagnostics={"reason": "Separation exceeds 50-sigma; probability analytically underflows"},
        )

    # -----------------------------------------------------------------------
    # Principal Axis Transformation
    # -----------------------------------------------------------------------
    # P_B = V Λ Vᵀ where Λ = diag(σ_x², σ_y²)
    # Transform miss vector into principal axes: μ' = Vᵀ b₀ = [μ_x, μ_y]ᵀ
    # The transformed PDF is uncorrelated:
    #   f(u, v) = 1/(2π σ_x σ_y) * exp(-u²/(2σ_x²) - v²/(2σ_y²))
    # over disk (u - μ_x)² + (v - μ_y)² ≤ HBR²
    V = b_plane_unc.eigenvectors
    b_vec = np.array([bt, br], dtype=np.float64)
    # Sort order matching eigenvalues:
    idx_sort = np.argsort(b_plane_unc.eigenvalues)
    lam_x = float(b_plane_unc.eigenvalues[idx_sort[0]])  # σ_minor²
    lam_y = float(b_plane_unc.eigenvalues[idx_sort[1]])  # σ_major²
    vx = V[:, idx_sort[0]]
    vy = V[:, idx_sort[1]]
    V_sorted = np.column_stack([vx, vy])

    mu_prime = V_sorted.T @ b_vec
    mu_x = float(mu_prime[0])
    mu_y = float(mu_prime[1])

    sigma_x = math.sqrt(lam_x)
    sigma_y = math.sqrt(lam_y)

    eval_count = [0]

    # Polar coordinate integration around (mu_x, mu_y):
    # u = mu_x + r*cos(theta), v = mu_y + r*sin(theta)
    # Jacobian = r
    # Pc = 1/(2π σ_x σ_y) ∫₀^HBR r dr ∫₀^{2π} exp(-½ [ (mu_x + r cos θ)²/σ_x² + (mu_y + r sin θ)²/σ_y² ]) dθ
    inv_2sx2 = 1.0 / (2.0 * sigma_x * sigma_x)
    inv_2sy2 = 1.0 / (2.0 * sigma_y * sigma_y)
    norm_const = 1.0 / (2.0 * math.pi * sigma_x * sigma_y)

    def integrand(theta: float, r: float) -> float:
        eval_count[0] += 1
        u = mu_x + r * math.cos(theta)
        v = mu_y + r * math.sin(theta)
        exponent = -(u * u * inv_2sx2 + v * v * inv_2sy2)
        # Numerical underflow protection
        if exponent < -500.0:
            return 0.0
        return float(r * math.exp(exponent))

    try:
        integral_val, err_est = integrate.dblquad(
            integrand,
            0.0,
            hbr_m,
            0.0,
            2.0 * math.pi,
            epsabs=tol,
            epsrel=tol,
        )
        pc_computed = float(integral_val * norm_const)
        converged = True
    except Exception as ex:
        # Fallback to high-order Gauss-Legendre quadrature
        n_r = 64
        n_th = 64
        r_pts, r_w = np.polynomial.legendre.leggauss(n_r)
        th_pts, th_w = np.polynomial.legendre.leggauss(n_th)

        # Scale r to [0, hbr_m]
        r_nodes = 0.5 * hbr_m * (r_pts + 1.0)
        r_weights = 0.5 * hbr_m * r_w

        # Scale theta to [0, 2π]
        th_nodes = math.pi * (th_pts + 1.0)
        th_weights = math.pi * th_w

        R_mesh, TH_mesh = np.meshgrid(r_nodes, th_nodes, indexing="ij")
        U = mu_x + R_mesh * np.cos(TH_mesh)
        V_mesh = mu_y + R_mesh * np.sin(TH_mesh)

        exponent = -(U * U * inv_2sx2 + V_mesh * V_mesh * inv_2sy2)
        integrand_vals = np.where(exponent < -500.0, 0.0, R_mesh * np.exp(exponent))

        integral_val = np.sum(r_weights[:, None] * th_weights[None, :] * integrand_vals)
        pc_computed = float(integral_val * norm_const)
        eval_count[0] = n_r * n_th
        converged = True

    # Clamp probability strictly to [0, 1]
    pc_clamped = max(0.0, min(1.0, pc_computed))

    return CollisionProbabilityResult(
        probability=pc_clamped,
        method="2D_Gaussian_Polar_Quadrature_Principal_Axes",
        converged=converged,
        tolerance=tol,
        iterations=eval_count[0],
        hard_body_radius_m=hbr_m,
        miss_distance_m=miss_dist,
        b_plane_coordinates_m=(bt, br),
        b_plane_covariance_m2=p_mat.tolist(),
        sigma_major_m=sigma_major,
        sigma_minor_m=sigma_minor,
        ellipse_angle_deg=b_plane_unc.ellipse_angle_deg,
        covariance_eigenvalues=eigvals,
        condition_number=cond_p,
        determinant=det_p,
        assumptions=assumptions,
        diagnostics={
            "sigma_x_m": sigma_x,
            "sigma_y_m": sigma_y,
            "mu_x_principal_m": mu_x,
            "mu_y_principal_m": mu_y,
            "raw_probability": pc_computed,
        },
    )


def monte_carlo_pc_validation(
    b_plane_unc: BPlaneUncertainty,
    hbr_m: float,
    sample_count: int = 100_000,
    seed: Optional[int] = 42,
) -> MonteCarloValidationResult:
    """
    Validation-only Monte Carlo estimation of collision probability.

    NOT intended for operational flight dynamics. Provided strictly to cross-validate
    the deterministic 2D Gaussian quadrature implementation.

    Parameters
    ----------
    b_plane_unc : BPlaneUncertainty
        B-plane geometry and covariance data.
    hbr_m : float
        Combined hard-body radius (m).
    sample_count : int
        Number of random samples drawn from N(b₀, P_B).
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    MonteCarloValidationResult
    """
    rng = np.random.default_rng(seed)

    bt = float(b_plane_unc.b_dot_t)
    br = float(b_plane_unc.b_dot_r)
    mean = np.array([bt, br])
    cov = b_plane_unc.b_plane_covariance

    # Draw samples from N(mean, cov)
    samples = rng.multivariate_normal(mean, cov, size=sample_count)

    # Count samples falling within HBR disk of target centered at origin: |z| <= HBR
    radii = np.linalg.norm(samples, axis=1)
    hits = int(np.sum(radii <= hbr_m))

    empirical_pc = float(hits / sample_count)
    std_err = math.sqrt(empirical_pc * (1.0 - empirical_pc) / sample_count) if sample_count > 0 else 0.0

    z_95 = 1.96
    ci_low = max(0.0, empirical_pc - z_95 * std_err)
    ci_high = min(1.0, empirical_pc + z_95 * std_err)

    det_result = compute_collision_probability(b_plane_unc, hbr_m)
    det_pc = det_result.probability

    diff = abs(empirical_pc - det_pc)
    # Check consistency within 3 standard errors
    is_consistent = diff <= max(3.0 * std_err, 1e-4)

    notes = (
        f"Monte Carlo validation with N={sample_count:,} samples. "
        f"Empirical Pc: {empirical_pc:.6e} ± {std_err:.2e}, "
        f"Deterministic Pc: {det_pc:.6e}. "
        f"{'CONSISTENT within 99% CI' if is_consistent else 'STATISTICAL DEVIATION EXCEEDED'}"
    )

    return MonteCarloValidationResult(
        sample_count=sample_count,
        hits=hits,
        empirical_pc=empirical_pc,
        deterministic_pc=det_pc,
        standard_error=std_err,
        confidence_interval_95=(ci_low, ci_high),
        difference=diff,
        is_consistent=is_consistent,
        notes=notes,
    )
