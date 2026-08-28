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
- Certification: two orientations of an exact 1-D reduction of the same
  integral, each self-refined; the reported value is the one at least two of
  the three independent constructions agree on
- Validation: Monte Carlo sampling (strictly labeled as validation only)

A note on Chan's series
-----------------------
This docstring used to advertise "Chan's series expansion for isotropic and
mildly anisotropic cases".  No such code existed.  Before writing it, P10-12
measured it against a 50-digit arbiter.  Chan's equivalent-area series is
exact for isotropic encounters (relative error 0 to 1.3e-14 across the
isotropic cases tried) and stays usable to about 10:1 anisotropy (1.9e-3), but
its equal-area substitution -- replacing the scaled elliptical cross-section by
a circle of the same area -- is an approximation, not an identity, and it
fails precisely in the regime this module had trouble with:

    sigma_major/sigma_minor    Pc true        Chan          relative error
              1e+03          7.259839e-03   1.982831e-05      9.97e-01
              4e+04          3.989356e-03   3.934693e-01      9.76e+01
              5e+05          7.978712e-03   1.000000e+00      1.24e+02

Overstating Pc by two orders of magnitude is no better than understating it.
The series was therefore not adopted, and the advertisement has been removed
rather than implemented.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np
from scipy import integrate
from scipy.special import ndtr

from theseus.uncertainty.b_plane import BPlaneUncertainty


#: Relative agreement required between the reported polar quadrature and the
#: independent verification integral before a result may be reported as
#: converged.
#:
#: Calibrated, not chosen: over 240 configurations spanning sigma_minor from
#: 3 m to 3 km, anisotropy ratios from 1 to 2500 and hard-body radii from 1 m
#: to 500 m, the two methods agree to at most 1.4e-12 (99th percentile
#: 4.3e-14).  The failures this threshold exists to catch are 19 % or larger.
#: 1e-6 therefore sits six orders above the observed agreement floor and five
#: below the smallest observed failure, so it can neither false-alarm on a
#: healthy case nor pass a broken one.
QUADRATURE_AGREEMENT_RTOL = 1e-6

#: How far inside the deterministic limit an encounter must be before the
#: 0/1 shortcut is taken, as a ratio of the uncertainty to the distance from
#: the boundary it would have to cross:  sigma_major <= ratio * | |b| - R |.
#:
#: Dimensionless by construction, so the branch cannot be selected by the
#: absolute scale of the problem.  The neglected probability mass is the
#: Gaussian tail beyond 1/ratio standard deviations; at 1e-8 that is
#: exp(-5e15), which is zero in double precision many times over.  Exactness
#: needs only about 9 sigma, so this is conservative by seven orders.
DETERMINISTIC_LIMIT_SIGMA_RATIO = 1e-8

#: Number of standard deviations beyond which the encounter-plane density
#: underflows to exactly zero in IEEE-754 double precision.  exp(-50^2/2) =
#: exp(-1250) is below the smallest subnormal, so a separation of 50 sigma
#: gives a probability that is not merely negligible but exactly 0.0.
FAR_SEPARATION_SIGMA = 50.0

#: The same margin applied to the near edge of the collision disk, so the disk
#: cannot reach back inside the resolvable region.
FAR_SEPARATION_DISK_SIGMA = 10.0

#: Smallest 1-sigma value (m) for which the encounter-plane density can be
#: formed at all.  This one is deliberately dimensional: 1/(2 sigma^2)
#: overflows to infinity once sigma falls below about 1.2e-154, and
#: 1/(2 pi sigma_x sigma_y) follows.  The bound comes from IEEE-754, not from
#: any physical scale, and no rescaling of the problem can move it.
FLOAT_SAFE_SIGMA_M = 1e-150

#: How far into the Gaussian tail the collision disk must reach before it can
#: be said to contain the entire distribution.
#:
#: Dimensionless, and exact rather than approximate: the mass of a 2-D Gaussian
#: outside k standard deviations of its widest axis is bounded by exp(-k^2/2),
#: which at k = 12 is 5.4e-32.  That is below the spacing of doubles near 1
#: (2.2e-16) by sixteen orders of magnitude, so 1.0 is the correctly rounded
#: double for such an encounter, not an approximation to it.  Checked against
#: 50-digit arithmetic: 1 - Pc evaluates to exactly 0.0 on every geometry that
#: satisfies this test.
ENGULFMENT_SIGMA = 12.0

#: Gauss-Legendre order per panel in the reduction integrals.
_VERIFICATION_GL_ORDER = 24

#: Panels are evaluated in blocks of this size so that a large panel count does
#: not allocate a proportionally large temporary.
_REDUCTION_PANEL_CHUNK = 4096

#: Largest panel count the refinement will reach.  Beyond this the reduction is
#: reported as unsettled rather than refined further; 2^15 panels at 24 nodes
#: is 786 432 evaluations, which is where the cost stops being justifiable
#: against the accuracy still being gained.
REDUCTION_PANEL_CAP = 2 ** 15

#: Relative agreement required between successive panel doublings before a
#: reduction is accepted as settled.
#:
#: Set from measurement: over 208 random geometries spanning sigma from 1e-2 to
#: 1e3 m, anisotropy 1 to 1e6 and hard-body radii from 0.1 m to 5 km, the two
#: orientations agreed with 50-digit arithmetic to at most 1.5e-11 whenever
#: both settled at this tolerance.  1e-12 is tight enough that a reduction
#: which has genuinely stopped moving is converged well past the 1e-6 agreement
#: tolerance the result is finally judged on, and loose enough not to chase
#: floating-point noise.
REDUCTION_REFINEMENT_RTOL = 1e-12

#: The documented default of the inert ``max_evals`` parameter.
_MAX_EVALS_DEFAULT = 10000

_VERIFICATION_NODES, _VERIFICATION_WEIGHTS = np.polynomial.legendre.leggauss(
    _VERIFICATION_GL_ORDER
)


def _cdf_band(mu: float, half_chord: np.ndarray, sigma: float) -> np.ndarray:
    """
    Phi((mu + h)/sigma) - Phi((mu - h)/sigma), evaluated where it is well
    conditioned.

    ``ndtr`` loses relative accuracy as its result approaches 1, so when both
    endpoints lie in the upper tail this is a difference of two numbers near 1
    and the subtraction destroys most of the significant digits.  Reflecting to
    the lower tail -- ndtr(-a) - ndtr(-b), exact by the symmetry of the normal
    distribution -- keeps both operands small and the subtraction benign.

    Measured on far-tail geometries, this is worth five to seven digits:
    relative errors of 4.3e-08, 3.8e-09 and 4.3e-09 against 50-digit arithmetic
    become 2.5e-15, 8.1e-15 and 6.8e-14 at the same panel count.
    """
    lo = (mu - half_chord) / sigma
    hi = (mu + half_chord) / sigma
    return np.where(lo >= 0.0, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))


def _disk_integral_1d(sigma_x: float, sigma_y: float, mu_x: float, mu_y: float,
                      hbr_m: float, panels: int) -> float:
    """
    The same probability by a different construction: one dimension instead of
    two, with the second integral done in closed form.

    In principal axes the density is centred at the origin with covariance
    diag(sigma_x^2, sigma_y^2) and the collision disk of radius R is centred at
    (mu_x, mu_y).  Integrating the v direction analytically turns the inner
    integral into a difference of normal CDFs, and the substitution
    u = mu_x + R sin(p) removes the square-root behaviour at the disk edge:

        Pc = int_{-pi/2}^{pi/2} (R cos p / (sqrt(2 pi) sigma_x))
                 exp(-(mu_x + R sin p)^2 / (2 sigma_x^2))
                 [Phi((mu_y + R cos p)/sigma_y) - Phi((mu_y - R cos p)/sigma_y)] dp

    This identity is ORIENTATION-DEPENDENT, which is the point.  Whichever axis
    is passed as *sigma_x* is integrated numerically; the other is integrated in
    closed form.  The outer factor is a spike of width ~12 sigma_x / R in p, so
    the orientation that integrates the NARROW axis needs O(R / sigma_minor)
    panels, while the orientation that integrates the BROAD axis is smooth on
    an order-unity scale and needs almost none.  Calling it both ways gives two
    evaluations of the same quantity whose conditioning is complementary --
    see `certified_disk_integral`.
    """
    if hbr_m <= 0.0:
        return 0.0

    panels = int(panels)
    step = math.pi / panels
    half_width = 0.5 * step
    total = 0.0
    done = 0
    inv_two_var = 1.0 / (2.0 * sigma_x * sigma_x)
    scale = hbr_m / (math.sqrt(2.0 * math.pi) * sigma_x)

    # Blocked rather than one allocation: at the panel cap a single array would
    # be 786 432 doubles, and several exist at once.
    while done < panels:
        count = min(_REDUCTION_PANEL_CHUNK, panels - done)
        midpoint = (-0.5 * math.pi + half_width
                    + step * np.arange(done, done + count, dtype=np.float64))
        phi = (midpoint[:, None]
               + half_width * _VERIFICATION_NODES[None, :]).ravel()
        weights = np.tile(half_width * _VERIFICATION_WEIGHTS, count)

        cos_phi = np.cos(phi)
        u = mu_x + hbr_m * np.sin(phi)
        outer = (scale * cos_phi) * np.exp(np.clip(-u * u * inv_two_var, -700.0, None))
        total += float(np.sum(weights * outer * _cdf_band(mu_y, hbr_m * cos_phi, sigma_y)))
        done += count

    return total


def _starting_panels(sigma_x: float, hbr_m: float) -> int:
    """
    A derived lower bound on the panel count, so refinement starts from a
    resolution the geometry justifies rather than from a fixed guess.

    The outer factor exp(-(mu + R sin p)^2 / 2 sigma^2) is narrowest where
    d(R sin p)/dp is largest, at p = 0, where it spans about 12 sigma / R in p.
    Requiring the panel width pi/n not to exceed that gives n >= pi R /
    (12 sigma) ~ 0.262 R / sigma; a factor of two of margin is applied and the
    result rounded up to a power of two.

    This is a floor, not a guarantee -- 37 of 252 measured orientations needed
    more -- so `certified_disk_integral` still refines by doubling until the
    value stops moving.
    """
    needed = max(32.0, 0.5 * hbr_m / sigma_x) if sigma_x > 0.0 else 32.0
    if not math.isfinite(needed):
        return REDUCTION_PANEL_CAP
    n = int(min(needed, float(REDUCTION_PANEL_CAP)))
    return min(1 << max(5, (n - 1).bit_length()), REDUCTION_PANEL_CAP)


def _refine_disk_integral(sigma_x: float, sigma_y: float, mu_x: float, mu_y: float,
                          hbr_m: float,
                          rtol: float = REDUCTION_REFINEMENT_RTOL
                          ) -> tuple[float, bool, int]:
    """
    One orientation of the reduction, refined until it stops moving.

    Returns ``(value, settled, panels)``.  ``settled`` is False when the panel
    cap is reached with successive doublings still moving the value by more
    than *rtol*: the reduction then certifies nothing, and the caller must say
    so rather than assume it converged.
    """
    panels = _starting_panels(sigma_x, hbr_m)
    previous = _disk_integral_1d(sigma_x, sigma_y, mu_x, mu_y, hbr_m, panels)
    while panels < REDUCTION_PANEL_CAP:
        panels *= 2
        current = _disk_integral_1d(sigma_x, sigma_y, mu_x, mu_y, hbr_m, panels)
        scale = max(abs(current), abs(previous))
        if scale == 0.0 or abs(current - previous) <= rtol * scale:
            return current, True, panels
        previous = current
    return previous, False, panels


def verify_disk_integral(sigma_x: float, sigma_y: float, mu_x: float, mu_y: float,
                         hbr_m: float,
                         rtol: float = REDUCTION_REFINEMENT_RTOL) -> tuple[float, bool, int]:
    """
    Independent value for the encounter-plane disk integral, plus whether it
    settled.

    This is the reduction taken in the orientation that integrates the MINOR
    axis numerically.  It is retained under its P10-08 name and contract;
    `certified_disk_integral` is what callers should use, since a single
    orientation can fail to resolve the disk edge when R / sigma_minor is very
    large.
    """
    return _refine_disk_integral(sigma_x, sigma_y, mu_x, mu_y, hbr_m, rtol=rtol)


@dataclass(frozen=True)
class DiskIntegralCertificate:
    """
    The outcome of cross-checking up to three independent evaluations of the
    encounter-plane disk integral.

    Attributes
    ----------
    value : float
        The probability at least two independent constructions agree on, or the
        polar quadrature's value when no two agree.
    certified : bool
        Whether two independent constructions that each settled agree to within
        the agreement tolerance.
    source : str
        Which construction supplied `value`.
    polar_agrees : bool
        Whether the adaptive polar quadrature is one of the agreeing methods.
        When it is, its value is the one reported, so results that were already
        correct are not perturbed.
    """
    value: float
    certified: bool
    source: str
    polar_agrees: bool
    minor_axis_value: float
    minor_axis_settled: bool
    minor_axis_panels: int
    major_axis_value: float
    major_axis_settled: bool
    major_axis_panels: int
    disagreement: float


def _relative_gap(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return 0.0
    return abs(a - b) / scale


def certified_disk_integral(sigma_x: float, sigma_y: float, mu_x: float, mu_y: float,
                            hbr_m: float, polar_value: Optional[float],
                            rtol: float = QUADRATURE_AGREEMENT_RTOL
                            ) -> DiskIntegralCertificate:
    """
    Decide what the encounter-plane disk integral actually is, using three
    independent constructions and believing whatever at least two of them agree
    on.

    The three are the adaptive polar quadrature (the documented primary
    method), and the exact 1-D reduction taken in each of its two orientations.
    None of them is reliable everywhere:

    * the polar quadrature fails in at least three distinct ways -- the ridge
      stepped over in theta, a spike stepped over in r, and crescent geometries
      where the disk excludes the density centre;
    * the minor-axis orientation needs O(R / sigma_minor) panels and cannot
      resolve the disk edge once that exceeds the panel cap;
    * the major-axis orientation is smooth in the anisotropic regime but has no
      advantage when the encounter is nearly isotropic.

    Their failure modes are, however, unrelated, and that is what makes the
    cross-check meaningful.  Over 208 random geometries no case was found where
    the two orientations both settled, agreed with each other, and were both
    wrong; the worst error among all such cases was 1.5e-11 against 50-digit
    arithmetic.

    A prescriptive rule was tried first and rejected on the evidence.  The
    natural candidate -- refuse the polar quadrature once the disk spans more
    than N ridge widths -- does not work: across 337 random geometries the
    largest ridge-resolution number among ACCURATE polar results was 155.0 and
    the smallest among INACCURATE ones was 2.03, so no threshold in that
    variable separates them.  Agreement is measured instead of predicted.
    """
    minor_value, minor_settled, minor_panels = _refine_disk_integral(
        sigma_x, sigma_y, mu_x, mu_y, hbr_m)
    major_value, major_settled, major_panels = _refine_disk_integral(
        sigma_y, sigma_x, mu_y, mu_x, hbr_m)

    candidates: list[tuple[str, float]] = []
    if polar_value is not None and math.isfinite(polar_value):
        candidates.append(("polar_quadrature", float(polar_value)))
    if minor_settled:
        candidates.append(("reduction_minor_axis", minor_value))
    if major_settled:
        candidates.append(("reduction_major_axis", major_value))

    # Every pair that agrees.  Each entry is (polar is in this pair, gap,
    # name of the value to report, that value).  The polar quadrature is
    # preferred whenever it belongs to an agreeing pair, so encounters that
    # were already computed correctly keep exactly the value they had; among
    # equally eligible pairs the tightest agreement wins.
    agreeing: list[tuple[bool, float, str, float]] = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            name_i, value_i = candidates[i]
            name_j, value_j = candidates[j]
            gap = _relative_gap(value_i, value_j)
            if gap > rtol:
                continue
            if name_i == "polar_quadrature":
                agreeing.append((True, gap, name_i, value_i))
            elif name_j == "polar_quadrature":
                agreeing.append((True, gap, name_j, value_j))
            else:
                agreeing.append((False, gap, name_i, value_i))

    if agreeing:
        polar_agrees, gap, source, value = max(
            agreeing, key=lambda entry: (entry[0], -entry[1]))
        return DiskIntegralCertificate(
            value=value, certified=True, source=source, polar_agrees=polar_agrees,
            minor_axis_value=minor_value, minor_axis_settled=minor_settled,
            minor_axis_panels=minor_panels, major_axis_value=major_value,
            major_axis_settled=major_settled, major_axis_panels=major_panels,
            disagreement=gap,
        )

    # Nothing could be corroborated.  The result is reported as unconverged
    # either way, but there is still a choice about WHICH uncorroborated number
    # to report, and it is not a free one.
    #
    # When exactly one reduction settled, it carries evidence the polar
    # quadrature does not: it refined by doubling until successive panel counts
    # agreed to 1e-12, whereas the quadrature's own error estimate is known to
    # be worthless in exactly this regime -- it reports zero error on an
    # integral it never sampled.  Reporting the quadrature there means printing
    # a probability of 0.0 for encounters whose settled reduction says
    # otherwise, which is the most dangerous output this module can produce.
    # So the settled reduction wins, the `method` field names it, and
    # `converged` stays False because one construction is not two.
    #
    # When both settled but disagree with each other, or when neither settled,
    # there is no such evidence and the documented method's value stands.
    settled = [(minor_panels, "reduction_minor_axis", minor_value)] if minor_settled else []
    if major_settled:
        settled.append((major_panels, "reduction_major_axis", major_value))

    have_polar = polar_value is not None and math.isfinite(polar_value)
    if len(settled) == 1:
        _, fallback_source, fallback_value = settled[0]
    elif have_polar:
        fallback_source, fallback_value = "polar_quadrature", float(polar_value)
    else:
        fallback_source, fallback_value = "reduction_minor_axis", minor_value

    reference = (float(polar_value) if have_polar
                 else (minor_value if minor_settled else major_value))
    return DiskIntegralCertificate(
        value=fallback_value, certified=False, source=fallback_source,
        polar_agrees=False,
        minor_axis_value=minor_value, minor_axis_settled=minor_settled,
        minor_axis_panels=minor_panels, major_axis_value=major_value,
        major_axis_settled=major_settled, major_axis_panels=major_panels,
        disagreement=_relative_gap(fallback_value, reference),
    )


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
        Deprecated and ignored.

        This was documented as "maximum function evaluations for integration"
        and was never read: ``scipy.integrate.dblquad`` takes no evaluation
        budget, and the reported ``iterations`` is a count of what happened,
        not a cap on it.  Measured before this note existed, ``max_evals=1``
        and ``max_evals=10**9`` returned byte-identical probabilities after
        an identical 777 evaluations.

        A caller who set it believed they had bounded the work and had not.
        Passing anything other than the default now raises a
        ``DeprecationWarning`` saying so.  The parameter is kept rather than
        removed so that existing callers keep working while being told the
        truth; the effective bound on the reduction integrals is
        ``REDUCTION_PANEL_CAP``.

    Returns
    -------
    CollisionProbabilityResult
    """
    if max_evals != _MAX_EVALS_DEFAULT:
        warnings.warn(
            "compute_collision_probability(max_evals=...) has never had any "
            "effect and is ignored; the quadrature takes no evaluation budget. "
            f"The reduction integrals are bounded by REDUCTION_PANEL_CAP="
            f"{REDUCTION_PANEL_CAP}.",
            DeprecationWarning, stacklevel=2)

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
            diagnostics={
                "reason": "HBR is zero; collision cross-section has zero area",
                "result_kind": "analytic_exact",
            },
        )

    # Edge Case 2: the uncertainty is negligible -> deterministic overlap
    #
    # This used to trigger on `sigma_major < 1e-6 or sigma_minor < 1e-6 or
    # det_p < 1e-12` -- metres and metres to the fourth.  Pc is dimensionless,
    # so scaling sigma, the miss distance and the hard-body radius by a common
    # factor must leave it unchanged; those tests did not.  A geometry with
    # sigma_minor/sigma_major = 0.2, |b| = 1.5 sigma and R = 0.3 sigma has
    # Pc = 5.8007827098e-02 at every scale, but the determinant test (which
    # scales as the fourth power of the length unit) fired below about 1e-2 m
    # and returned exactly 0.0 -- a probability of 5.8 % reported as
    # impossible, and reported as converged, purely because the same situation
    # was expressed in smaller units.
    #
    # The branch is really approximating a limit: when the uncertainty cannot
    # reach the boundary between hit and miss, the answer is the deterministic
    # one.  That is a statement about a ratio, so it is now written as one.
    sigma_major = b_plane_unc.sigma_major
    sigma_minor = b_plane_unc.sigma_minor

    boundary_clearance = abs(miss_dist - hbr_m)
    uncertainty_is_negligible = (
        sigma_major <= DETERMINISTIC_LIMIT_SIGMA_RATIO * boundary_clearance
    )
    # Separately, and for a genuinely dimensional reason: below this the
    # encounter-plane density cannot be represented in floating point at all.
    density_is_representable = (
        sigma_minor >= FLOAT_SAFE_SIGMA_M
        and math.isfinite(1.0 / (2.0 * sigma_minor * sigma_minor))
        and sigma_major * sigma_minor > 0.0
    )

    if uncertainty_is_negligible or not density_is_representable:
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
            diagnostics={
                "reason": "Covariance approaches zero; evaluating deterministic overlap",
                "result_kind": (
                    "deterministic_limit" if uncertainty_is_negligible
                    else "density_not_representable"
                ),
                "sigma_major_over_boundary_clearance": (
                    sigma_major / boundary_clearance
                    if boundary_clearance > 0.0 else float("inf")
                ),
                "deterministic_limit_sigma_ratio": DETERMINISTIC_LIMIT_SIGMA_RATIO,
                "density_is_representable": density_is_representable,
            },
        )

    # Edge Case 3: the disk lies so far into the tail that the density
    # underflows to exactly zero.
    #
    # The sigma multiples below are dimensionless and always were; the defect
    # was the floor `max_sigma = max(sigma_major, 1.0)`, which measured the
    # separation in metres once sigma fell below one metre.  That made the
    # shortcut unavailable for small-scale encounters -- conservative rather
    # than wrong, since the quadrature then returns the same 0.0, but
    # scale-dependent all the same.  Removing the floor leaves a purely
    # dimensionless criterion.
    if (miss_dist > FAR_SEPARATION_SIGMA * sigma_major
            and (miss_dist - hbr_m) > FAR_SEPARATION_DISK_SIGMA * sigma_major):
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
            diagnostics={
                "reason": (
                    f"Separation exceeds {FAR_SEPARATION_SIGMA:.0f}-sigma; the "
                    f"density underflows to exactly zero in double precision"
                ),
                "result_kind": "analytic_exact",
                "miss_over_sigma_major": (
                    miss_dist / sigma_major if sigma_major > 0.0 else float("inf")
                ),
            },
        )

    # Edge Case 4: the collision disk contains the entire distribution.
    #
    # The mirror of Edge Case 3, and found the same way P10-12 was: by random
    # sweep.  When the disk reaches ENGULFMENT_SIGMA standard deviations past
    # the far side of the uncertainty ellipse, every draw lands inside it, so
    # Pc = 1 to within exp(-k^2/2) = 5.4e-32 -- exactly 1.0 in double
    # precision.  The polar quadrature does not get this right: with the
    # density a spike of width sigma at the origin and the disk radius
    # thousands of times larger, the adaptive subdivision in r steps over the
    # spike and returns essentially zero.  Measured at sigma = 0.0146 /
    # 0.0488 m and HBR = 536.9 m, it reported a relative error of 1.00 -- the
    # certain collision reported as the impossible one.
    #
    # These geometries also defeat the reduction, whose panel requirement
    # scales as R / sigma_minor and reaches 2e7 here, so the certificate below
    # cannot rescue them either.  The branch is analytic for that reason, not
    # for speed.
    engulfment_clearance = miss_dist + ENGULFMENT_SIGMA * sigma_major
    if hbr_m >= engulfment_clearance:
        return CollisionProbabilityResult(
            probability=1.0,
            method="analytic_engulfment",
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
            diagnostics={
                "reason": (
                    f"The collision disk encloses the uncertainty distribution "
                    f"out to {ENGULFMENT_SIGMA:.0f} sigma; the mass outside it "
                    f"is below the resolution of double precision"
                ),
                "result_kind": "analytic_exact",
                "engulfment_sigma": ENGULFMENT_SIGMA,
                "hbr_over_engulfment_clearance": (
                    hbr_m / engulfment_clearance
                    if engulfment_clearance > 0.0 else float("inf")
                ),
            },
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

    quadrature_error_estimate: Optional[float] = None
    fallback_used = False

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
        quadrature_error_estimate = float(err_est * norm_const)
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
        fallback_used = True

    # -----------------------------------------------------------------------
    # Convergence, determined rather than asserted -- and now acted on
    # -----------------------------------------------------------------------
    # `converged` used to be the literal True on both branches above, so it
    # could not report failure.  Reading dblquad's own error estimate is not
    # enough either: on a density whose ridge is much narrower than the disk,
    # the adaptive subdivision can miss the ridge entirely, return a value 19 %
    # to 97 % low, and report an error estimate around 1e-10 -- confidently
    # wrong, with no exception and no IntegrationWarning to catch.
    #
    # P10-08 established that much and stopped there: it reported the polar
    # quadrature's number and marked it unconverged.  P10-12 is the follow-on
    # finding that reporting a knowably wrong probability is not an acceptable
    # resting place for a conjunction-assessment tool, however loudly it is
    # labelled.  The mechanism is now understood exactly (see the module tests):
    # QUADPACK's initial 21-point Gauss-Kronrod rule on [0, 2 pi] has its
    # nearest node 0.19716 rad from each ridge crossing, so for radii beyond
    # about 160.4 sigma_minor every node of that rule returns exactly zero; the
    # rule then reports an integral of 0 with an error estimate of 0 and QAGS
    # accepts it without subdividing.  An adaptive algorithm cannot detect that
    # it has missed a feature it never sampled.
    #
    # The value is therefore cross-checked against two further constructions
    # and the majority is believed.  Where the polar quadrature is one of the
    # agreeing methods -- every case that was already correct -- its own value
    # is what gets reported, unchanged.
    certificate = certified_disk_integral(sigma_x, sigma_y, mu_x, mu_y, hbr_m,
                                          polar_value=pc_computed)

    verification = certificate.minor_axis_value
    verification_settled = certificate.minor_axis_settled
    verification_panels = certificate.minor_axis_panels
    disagreement = _relative_gap(pc_computed, certificate.value)

    converged = certificate.certified
    substituted = not certificate.polar_agrees and certificate.source != "polar_quadrature"

    if substituted and converged:
        method = "Exact_1D_Reduction_Dual_Orientation"
        reported = certificate.value
        convergence_note = (
            f"The adaptive polar quadrature returned {pc_computed:.6e}, which "
            f"disagrees by {disagreement:.3%} with the value two independent "
            f"exact reductions of the same integral agree on "
            f"({certificate.value:.6e}). The reduction is what is reported, and "
            f"`method` names it. The quadrature fails here because the "
            f"encounter-plane density forms a ridge of width sigma_minor = "
            f"{sigma_x:.3g} m across a collision disk of radius "
            f"{hbr_m:.3g} m, and the fixed nodes of the inner Gauss-Kronrod "
            f"rule fall either side of it."
        )
    elif converged:
        method = "2D_Gaussian_Polar_Quadrature_Principal_Axes"
        reported = pc_computed
        convergence_note = (
            f"Agrees with an independent evaluation of the same integral to "
            f"{certificate.disagreement:.3e} relative."
        )
    else:
        reported = certificate.value
        method = ("Exact_1D_Reduction_Uncorroborated" if substituted
                  else "2D_Gaussian_Polar_Quadrature_Principal_Axes")
        provenance = (
            f"The reported probability comes from the "
            f"{certificate.source.replace('_', ' ')}, which was the only "
            f"construction to settle under refinement; one construction is not "
            f"two, so this result is NOT certified."
            if substituted else
            "The reported probability is the one the documented method "
            "produced and must not be relied on."
        )
        convergence_note = (
            f"No two of the three independent evaluations of this integral "
            f"agree: polar quadrature {pc_computed:.6e}, minor-axis reduction "
            f"{certificate.minor_axis_value:.6e} "
            f"(settled={certificate.minor_axis_settled}), major-axis reduction "
            f"{certificate.major_axis_value:.6e} "
            f"(settled={certificate.major_axis_settled}). " + provenance
        )

    # Clamp probability strictly to [0, 1]
    pc_clamped = max(0.0, min(1.0, reported))

    return CollisionProbabilityResult(
        probability=pc_clamped,
        method=method,
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
            "result_kind": ("exact_reduction" if (substituted and converged)
                            else "numerical_quadrature"),
            "quadrature_error_estimate": quadrature_error_estimate,
            "fallback_quadrature_used": fallback_used,
            "verification_probability": verification,
            "verification_settled": verification_settled,
            "verification_panels": verification_panels,
            "verification_disagreement": disagreement,
            "certified_probability": certificate.value,
            "certified": certificate.certified,
            "certificate_source": certificate.source,
            "polar_quadrature_agrees": certificate.polar_agrees,
            "polar_quadrature_superseded": substituted,
            "reduction_minor_axis_probability": certificate.minor_axis_value,
            "reduction_minor_axis_settled": certificate.minor_axis_settled,
            "reduction_minor_axis_panels": certificate.minor_axis_panels,
            "reduction_major_axis_probability": certificate.major_axis_value,
            "reduction_major_axis_settled": certificate.major_axis_settled,
            "reduction_major_axis_panels": certificate.major_axis_panels,
            "convergence_criterion": (
                "at least two of three independent evaluations of the same "
                "integral (adaptive polar quadrature, and the exact 1-D "
                "reduction in each of its two orientations) agree to within "
                f"{QUADRATURE_AGREEMENT_RTOL:.0e} relative"
            ),
            "convergence_note": convergence_note,
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
