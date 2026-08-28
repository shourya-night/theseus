"""
P10-08 — the collision-probability convergence flag must mean something.

The defect
----------
``compute_collision_probability`` set ``converged = True`` as a literal on both
branches of its ``try/except``; the string ``converged = False`` did not occur
in the function.  ``dblquad``'s error estimate was bound to ``err_est`` and
never read, and ``dblquad`` reports accuracy trouble through a warning rather
than an exception, so the ``except`` clause could not catch it either.  Trace
step 13 -- "Check Numerical Stability & Convergence" -- named
``error_estimate <= tolerance`` as its equation and then reported success
unconditionally.

The audit called this "a diagnostic honesty problem rather than a numerical
one".  Measured against an independent evaluation of the same integral, that
is too generous: the quadrature does become inaccurate under valid
configurations, and it does so silently.

    sigma ratio   HBR      Pc engine      Pc independent   error     converged
        1e+03    20 m    1.593665e-02    1.593665e-02    6.5e-16      True
        3e+03    60 m    1.595441e-02    1.595441e-02    3.0e-15      True
        1e+04   200 m    1.288004e-02    1.595643e-02     19.3 %      True  <-
        3e+04   600 m    4.293511e-03    1.595661e-02     73.1 %      True  <-
        1e+05  2000 m    1.288226e-03    1.595663e-02     91.9 %      True  <-
        5e+05  5000 m    2.576128e-04    7.978712e-03     96.8 %      True  <-

scipy raised **zero** IntegrationWarnings in those rows, and its own error
estimate was around 1e-10.  When the encounter-plane density forms a ridge far
narrower than the collision disk, the adaptive subdivision steps over the ridge
and converges confidently to the wrong answer.  Understating Pc by a factor of
31 is the dangerous direction.

Reading ``err_est`` -- the fix the finding implies -- would therefore not have
worked.  Convergence is now decided by agreement with an independent
construction of the same integral, which reduces one dimension analytically and
so does not share the failure mode.

What these tests pin
--------------------
A. ``converged`` can be False, and is False exactly where the reported value
   cannot be corroborated.
B. The flag tracks measured accuracy in both directions, against Monte Carlo.
C. The error estimate is now read and reported rather than discarded.
D. No value is silently substituted: the reported probability is always the
   value of the method ``result.method`` names.
E. Trace step 13 states the criterion it actually applies.
F. P10-04, P10-05, P10-06 and P10-07 are unaffected.

The independent reference here is Monte Carlo sampling plus a dense polar
Gauss-Legendre grid.  Neither is the production verification integral: Monte
Carlo shares no quadrature at all, and the polar grid uses two dimensions
where production's verification uses one.

Amended by P10-12
-----------------
P10-08 detected the failure and stopped there, reporting the quadrature's
wrong number with ``converged = False``.  P10-12 found the mechanism -- the
initial 21-point Gauss-Kronrod rule of the inner QAGS call has its nearest
node 0.19716 rad from each ridge crossing, so beyond about 160.4 sigma_minor
every node of that rule returns exactly zero, the rule reports an integral of
0 with an error estimate of 0, and QAGS accepts it without subdividing -- and
made the result correct rather than merely labelled.  The four rows tabulated
above are now computed exactly by an independent construction.

Nothing P10-08 established has been given up.  The changes to this file are:

* ``BROKEN`` is renamed ``SUPERSEDED``: these geometries still defeat the
  polar quadrature by 19 % to 97 %, and the tests still prove that the polar
  value is wrong and that the check catches it.  What has changed is that the
  wrong value is no longer the one reported.
* ``UNCERTIFIABLE`` is new.  ``converged`` must still be capable of being
  False, and these geometries are the witnesses: at R / sigma_minor of 1e5 to
  1e6 the minor-axis reduction cannot resolve the disk edge at any practical
  panel count, so only one construction settles, and one is not two.
* D is stated the way it was always meant: the guarantee is that the reported
  number and the named method match, not that one particular method always
  wins.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy import integrate

from theseus.uncertainty.b_plane import project_covariance_to_b_plane
from theseus.uncertainty.collision_probability import (
    QUADRATURE_AGREEMENT_RTOL,
    compute_collision_probability,
)


def encounter(sigma_major_m: float, sigma_minor_m: float,
              miss_t_m: float = 0.0, miss_r_m: float = 0.0):
    """
    A B-plane uncertainty with a prescribed ellipse, built through the real
    projection so the input is a genuine production object.
    """
    return project_covariance_to_b_plane(
        rel_pos_cov=np.diag([sigma_major_m ** 2, sigma_minor_m ** 2, 1.0]),
        r_rel=np.array([miss_t_m, miss_r_m, 0.0]),
        v_rel=np.array([0.0, 0.0, 1.0e4]),
    )


def monte_carlo_pc(bpu, hbr_m: float, samples: int = 2_000_000, seed: int = 7):
    """
    Pc by direct sampling of the encounter-plane Gaussian.  No quadrature of
    any kind, so it cannot share a quadrature blind spot.

    Returns ``(estimate, standard_error)``.
    """
    covariance = np.asarray(bpu.b_plane_covariance, dtype=float)
    mean = np.array([bpu.b_dot_t, bpu.b_dot_r], dtype=float)
    chol = np.linalg.cholesky(covariance)
    rng = np.random.default_rng(seed)

    hits = 0
    drawn = 0
    while drawn < samples:
        batch = min(500_000, samples - drawn)
        points = (chol @ rng.standard_normal((2, batch))).T + mean
        hits += int(np.count_nonzero(
            np.einsum("ij,ij->i", points, points) <= hbr_m * hbr_m))
        drawn += batch

    estimate = hits / samples
    error = math.sqrt(max(estimate, 1.0 / samples) * (1.0 - estimate) / samples)
    return estimate, error


def dense_polar_pc(bpu, hbr_m: float, nodes: int = 1200):
    """
    Pc by a dense two-dimensional polar Gauss-Legendre grid about the disk
    centre.  Reliable where the ridge is resolvable; used only for the benign
    cases, where its own refinement is checked.
    """
    covariance = np.asarray(bpu.b_plane_covariance, dtype=float)
    mean = np.array([bpu.b_dot_t, bpu.b_dot_r], dtype=float)
    inverse = np.linalg.inv(covariance)
    scale = 1.0 / (2.0 * math.pi * math.sqrt(np.linalg.det(covariance)))

    r_pts, r_w = np.polynomial.legendre.leggauss(nodes)
    t_pts, t_w = np.polynomial.legendre.leggauss(nodes)
    radius = 0.5 * hbr_m * (r_pts + 1.0)
    radius_w = 0.5 * hbr_m * r_w
    theta = math.pi * (t_pts + 1.0)
    theta_w = math.pi * t_w

    r_mesh, t_mesh = np.meshgrid(radius, theta, indexing="ij")
    x = r_mesh * np.cos(t_mesh) - mean[0]
    y = r_mesh * np.sin(t_mesh) - mean[1]
    quad = inverse[0, 0] * x * x + 2.0 * inverse[0, 1] * x * y + inverse[1, 1] * y * y
    values = scale * np.exp(np.clip(-0.5 * quad, -700.0, None)) * r_mesh
    return float(np.sum(radius_w[:, None] * theta_w[None, :] * values))


#: (sigma_major, sigma_minor, hbr) where the quadrature is sound.
HEALTHY = [
    (5.0e2, 5.0e1, 10.0),
    (5.0e3, 5.0e2, 50.0),
    (1.0e3, 1.0e0, 20.0),
    (3.0e3, 1.0e0, 60.0),
    (2.0e2, 2.0e2, 15.0),
]

#: (sigma_major, sigma_minor, hbr) where the polar quadrature is not sound.
#: Under P10-12 these are still detected, and the reported value now comes from
#: the exact reduction instead.
SUPERSEDED = [
    (1.0e4, 1.0e0, 200.0),
    (3.0e4, 1.0e0, 600.0),
    (1.0e5, 1.0e0, 2000.0),
    (5.0e5, 1.0e0, 5000.0),
]

#: Retained under the old name so the P10-08 intent stays greppable.
BROKEN = SUPERSEDED

#: (sigma_major, sigma_minor, hbr) where nothing can be corroborated: the disk
#: spans 1e5 to 1e6 minor-axis widths, so the minor-axis reduction cannot
#: resolve its edge within the panel cap and only one construction settles.
#: These are the witnesses that ``converged`` can still be False.
UNCERTIFIABLE = [
    (1.0e2, 1.0e-3, 1000.0),
    (1.0e3, 1.0e-2, 5000.0),
    (1.0e1, 1.0e-4, 100.0),
]


# ---------------------------------------------------------------------------
# A. The flag can be False, and is False where the quadrature is wrong
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma_major,sigma_minor,hbr", UNCERTIFIABLE)
def test_convergence_flag_is_false_when_nothing_can_be_corroborated(
        sigma_major, sigma_minor, hbr):
    """
    The literal claim of P10-08: ``converged`` must be capable of being False.

    Under P10-12 the geometries that used to witness this are computed
    correctly, so the witness moves to the geometries where corroboration is
    genuinely unavailable.  If this ever passes vacuously -- because every
    encounter became certifiable -- the flag has stopped meaning anything and
    P10-08 has been silently undone.
    """
    result = compute_collision_probability(encounter(sigma_major, sigma_minor), hbr)
    assert result.converged is False
    diagnostics = result.diagnostics
    assert diagnostics["certified"] is False
    # Exactly one construction settled; that is why it cannot be certified.
    settled = [diagnostics["reduction_minor_axis_settled"],
               diagnostics["reduction_major_axis_settled"]]
    assert settled.count(True) == 1


@pytest.mark.parametrize("sigma_major,sigma_minor,hbr", SUPERSEDED)
def test_the_polar_quadrature_is_still_detected_as_wrong(
        sigma_major, sigma_minor, hbr):
    """
    P10-08's detection, unchanged: these are the configurations where the polar
    quadrature is 19 % to 97 % low.  It is still computed, still compared, and
    still found wanting -- the only difference is that its value is no longer
    the one reported.
    """
    result = compute_collision_probability(encounter(sigma_major, sigma_minor), hbr)
    diagnostics = result.diagnostics

    assert diagnostics["polar_quadrature_agrees"] is False
    assert diagnostics["polar_quadrature_superseded"] is True
    polar = diagnostics["raw_probability"]
    assert polar == pytest.approx(polar)          # it is still recorded
    assert abs(polar - result.probability) / result.probability > 0.15


@pytest.mark.parametrize("sigma_major,sigma_minor,hbr", HEALTHY)
def test_convergence_flag_is_true_when_the_quadrature_succeeds(
        sigma_major, sigma_minor, hbr):
    """The complement: no false alarms on cases that were always correct."""
    result = compute_collision_probability(encounter(sigma_major, sigma_minor), hbr)
    assert result.converged is True


def test_the_failure_is_silent_without_the_check():
    """
    Why reading scipy's own signals would not have been enough: the failing
    cases raise no exception and emit no IntegrationWarning, and dblquad's
    error estimate is tiny while its answer is grossly wrong.

    P10-12 note: the assertion now targets ``raw_probability`` -- the polar
    quadrature's own value -- because that is the number P10-08 was talking
    about.  It is still silent, still wrong, and still caught.  What changed is
    that it is no longer what the caller receives.
    """
    sigma_major, sigma_minor, hbr = SUPERSEDED[2]
    bpu = encounter(sigma_major, sigma_minor)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_collision_probability(bpu, hbr)
        integration_warnings = [w for w in caught
                                if issubclass(w.category, integrate.IntegrationWarning)]

    assert integration_warnings == []
    error_estimate = result.diagnostics["quadrature_error_estimate"]
    assert error_estimate is not None

    polar = result.diagnostics["raw_probability"]
    assert error_estimate < 1e-6 * max(polar, 1e-30) or error_estimate < 1e-9

    reference, standard_error = monte_carlo_pc(bpu, hbr)
    assert abs(polar - reference) > 20.0 * standard_error, \
        "the polar quadrature must still be measurably wrong here"
    assert result.diagnostics["polar_quadrature_superseded"] is True
    # ...and the value the caller actually gets is right.
    assert abs(result.probability - reference) < 5.0 * standard_error


# ---------------------------------------------------------------------------
# B. The flag tracks measured accuracy, in both directions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma_major,sigma_minor,hbr", HEALTHY[:3])
def test_converged_results_agree_with_monte_carlo(sigma_major, sigma_minor, hbr):
    """A converged result must actually be right, checked by sampling."""
    bpu = encounter(sigma_major, sigma_minor)
    result = compute_collision_probability(bpu, hbr)
    assert result.converged is True

    estimate, standard_error = monte_carlo_pc(bpu, hbr)
    assert abs(result.probability - estimate) < 5.0 * standard_error


@pytest.mark.parametrize("sigma_major,sigma_minor,hbr", SUPERSEDED[:2])
def test_superseded_polar_values_disagree_with_monte_carlo(
        sigma_major, sigma_minor, hbr):
    """
    And a value must be superseded for a real reason, not out of caution: the
    polar quadrature is measurably wrong on these, and the replacement is
    measurably right.  Both directions are checked against sampling, which
    shares no quadrature with either.
    """
    bpu = encounter(sigma_major, sigma_minor)
    result = compute_collision_probability(bpu, hbr)
    assert result.diagnostics["polar_quadrature_superseded"] is True

    estimate, standard_error = monte_carlo_pc(bpu, hbr)
    assert abs(result.diagnostics["raw_probability"] - estimate) > 20.0 * standard_error
    assert abs(result.probability - estimate) < 5.0 * standard_error


@pytest.mark.parametrize("sigma_major,sigma_minor,hbr", HEALTHY[:3])
def test_converged_results_agree_with_a_dense_polar_grid(sigma_major, sigma_minor, hbr):
    """
    A second independent check on the healthy cases, using a two-dimensional
    grid rather than the one-dimensional reduction production verifies with.
    """
    bpu = encounter(sigma_major, sigma_minor)
    result = compute_collision_probability(bpu, hbr)

    coarse = dense_polar_pc(bpu, hbr, nodes=600)
    fine = dense_polar_pc(bpu, hbr, nodes=1200)
    assert abs(fine - coarse) / fine < 1e-6, "the grid reference must itself be settled"
    assert result.probability == pytest.approx(fine, rel=1e-8)


# ---------------------------------------------------------------------------
# C. The error estimate is read, not discarded
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma_major,sigma_minor,hbr", HEALTHY[:2] + SUPERSEDED[:1])
def test_quadrature_error_estimate_is_reported(sigma_major, sigma_minor, hbr):
    """``err_est`` used to be bound and never referenced again."""
    result = compute_collision_probability(encounter(sigma_major, sigma_minor), hbr)
    estimate = result.diagnostics["quadrature_error_estimate"]
    assert estimate is not None
    assert math.isfinite(estimate)
    assert estimate >= 0.0


@pytest.mark.parametrize("sigma_major,sigma_minor,hbr",
                         HEALTHY[:1] + SUPERSEDED[:1] + UNCERTIFIABLE[:1])
def test_convergence_diagnostics_explain_the_verdict(sigma_major, sigma_minor, hbr):
    """
    A flag with no explanation is only marginally better than a hard-coded
    one: the caller must be able to see what was compared.
    """
    result = compute_collision_probability(encounter(sigma_major, sigma_minor), hbr)
    diagnostics = result.diagnostics

    assert math.isfinite(diagnostics["verification_probability"])
    assert math.isfinite(diagnostics["verification_disagreement"])
    assert "1e-06" in diagnostics["convergence_criterion"]
    assert "two of three" in diagnostics["convergence_criterion"]
    assert diagnostics["convergence_note"]

    # Every one of the three constructions is reported, with its own settled
    # flag, so the verdict can be re-derived by the caller.
    for key in ("raw_probability", "certified_probability",
                "reduction_minor_axis_probability",
                "reduction_major_axis_probability"):
        assert math.isfinite(diagnostics[key])
    assert isinstance(diagnostics["reduction_minor_axis_settled"], bool)
    assert isinstance(diagnostics["reduction_major_axis_settled"], bool)
    assert result.converged is diagnostics["certified"]


# ---------------------------------------------------------------------------
# D. The reported probability is never silently replaced
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma_major,sigma_minor,hbr",
                         HEALTHY + SUPERSEDED + UNCERTIFIABLE)
def test_the_reported_value_is_always_the_named_methods_value(
        sigma_major, sigma_minor, hbr):
    """
    P10-08's anti-substitution rule, stated the way it was always meant.

    The prohibition is on changing the answer on the strength of a method the
    ``method`` field does not name -- not on ever preferring a different
    method.  So whichever construction supplies the number, ``method`` must say
    so, and the discarded candidates must remain visible in the diagnostics for
    the caller to check.
    """
    result = compute_collision_probability(encounter(sigma_major, sigma_minor), hbr)
    diagnostics = result.diagnostics

    if result.method == "2D_Gaussian_Polar_Quadrature_Principal_Axes":
        assert result.probability == pytest.approx(diagnostics["raw_probability"])
        assert diagnostics["polar_quadrature_superseded"] is False
    elif result.method == "Exact_1D_Reduction_Dual_Orientation":
        assert result.converged is True
        assert result.probability == pytest.approx(diagnostics["certified_probability"])
        assert diagnostics["polar_quadrature_superseded"] is True
        assert diagnostics["certificate_source"].startswith("reduction_")
    elif result.method == "Exact_1D_Reduction_Uncorroborated":
        assert result.converged is False
        assert result.probability == pytest.approx(diagnostics["certified_probability"])
        assert diagnostics["certificate_source"].startswith("reduction_")
    else:  # pragma: no cover - a new method must be added here deliberately
        pytest.fail(f"unrecognised method {result.method!r}")

    # The value the polar quadrature produced is retained either way, so the
    # substitution can never be invisible.
    assert math.isfinite(diagnostics["raw_probability"])


@pytest.mark.parametrize("sigma_major,sigma_minor,hbr", SUPERSEDED)
def test_a_superseded_result_names_the_method_that_replaced_it(
        sigma_major, sigma_minor, hbr):
    """The specific case P10-12 exists for."""
    result = compute_collision_probability(encounter(sigma_major, sigma_minor), hbr)

    assert result.method == "Exact_1D_Reduction_Dual_Orientation"
    assert result.converged is True
    assert result.diagnostics["result_kind"] == "exact_reduction"
    assert "disagrees" in result.diagnostics["convergence_note"]
    assert result.probability != pytest.approx(
        result.diagnostics["raw_probability"], rel=1e-3)


@pytest.mark.parametrize("sigma_major,sigma_minor,hbr", HEALTHY)
def test_healthy_case_probabilities_are_unchanged_by_the_correction(
        sigma_major, sigma_minor, hbr):
    """
    Behaviour preservation where nothing was broken: the value still comes
    from the polar quadrature and still lies in [0, 1].
    """
    result = compute_collision_probability(encounter(sigma_major, sigma_minor), hbr)
    assert result.probability == pytest.approx(result.diagnostics["raw_probability"])
    assert 0.0 <= result.probability <= 1.0
    assert result.iterations > 0


# ---------------------------------------------------------------------------
# E. Boundary and early-return branches
# ---------------------------------------------------------------------------

def test_zero_hard_body_radius_is_exact_and_reports_converged():
    """
    A zero-area cross-section has probability zero exactly; there is no
    quadrature to converge, and reporting True is honest.  Reachable since
    P10-04.
    """
    result = compute_collision_probability(encounter(5.0e2, 5.0e1), 0.0)
    assert result.probability == 0.0
    assert result.method == "analytic_zero_hbr"
    assert result.converged is True


def test_far_separation_branch_reports_converged():
    """Analytic underflow, not a quadrature result."""
    result = compute_collision_probability(encounter(1.0e2, 1.0e2, miss_t_m=1.0e5), 10.0)
    assert result.probability == 0.0
    assert result.converged is True


@pytest.mark.parametrize("hbr", (0.05, 0.12, 0.6, 1.9))
def test_small_hard_body_radii_still_converge(hbr):
    """P10-04 interaction: the small-HBR range must not be flagged spuriously."""
    result = compute_collision_probability(encounter(5.0e2, 3.0e2), hbr)
    assert result.converged is True
    assert result.probability > 0.0


def test_verification_is_reported_as_unsettled_rather_than_guessed():
    """
    If the verification integral itself will not settle, the honest answer is
    "cannot certify", not "converged".  Exercised directly on the helper so
    both branches are covered without contriving a pathological encounter.

    P10-12 note: the panel schedule is now derived from the geometry and
    refined by doubling rather than being the fixed (64, 256, 1024), so the
    assertion is on the contract -- a power of two within the cap -- rather
    than on which fixed step happened to succeed.
    """
    from theseus.uncertainty.collision_probability import (
        REDUCTION_PANEL_CAP, verify_disk_integral)

    value, settled, panels = verify_disk_integral(
        sigma_x=5.0e2, sigma_y=5.0e2, mu_x=0.0, mu_y=0.0, hbr_m=10.0)
    assert settled is True
    assert value > 0.0
    assert panels & (panels - 1) == 0
    assert panels <= REDUCTION_PANEL_CAP

    # And the other branch: a disk spanning a million minor-axis widths cannot
    # have its edge resolved within the cap, so the helper must say so.
    _, unsettled, capped = verify_disk_integral(
        sigma_x=1.0e-3, sigma_y=1.0e2, mu_x=0.0, mu_y=0.0, hbr_m=1000.0)
    assert unsettled is False
    assert capped == REDUCTION_PANEL_CAP


# ---------------------------------------------------------------------------
# F. Trace step 13 tells the truth
# ---------------------------------------------------------------------------

def _phase10_trace(sigma_major, sigma_minor, hbr):
    """Build the 14-step trace around a prescribed encounter geometry."""
    from theseus.uncertainty.covariance import StateCovariance
    from theseus.uncertainty.results import build_phase10_calculation_trace
    from theseus.uncertainty.hard_body import compute_hard_body_radius
    from theseus.uncertainty.risk import PROFILE_STANDARD, classify_risk
    from theseus.uncertainty.relative import compute_relative_covariance

    bpu = encounter(sigma_major, sigma_minor)
    pc_res = compute_collision_probability(bpu, hbr)
    hbr_res = compute_hard_body_radius(custom_hbr_m=hbr)
    cov = StateCovariance.from_diagonal([100.0] * 3, [0.1] * 3, name="test")
    rel_cov = compute_relative_covariance(cov, cov)
    return pc_res, build_phase10_calculation_trace(
        initial_cov_a=cov, initial_cov_b=cov,
        cov_a_tca=cov, cov_b_tca=cov, rel_cov=rel_cov,
        tca_s=100.0,
        r_rel_tca=np.array([bpu.b_dot_t, bpu.b_dot_r, 0.0]),
        v_rel_tca=np.array([0.0, 0.0, 1.0e4]),
        b_plane_unc=bpu, hbr_res=hbr_res, pc_res=pc_res,
        risk=classify_risk(pc_res.probability, PROFILE_STANDARD),
    )


def _step_13(steps):
    return next(s for s in steps if s["stepIndex"] == 13)


def test_trace_step_13_reports_success_only_when_converged():
    pc_res, steps = _phase10_trace(*HEALTHY[0])
    step = _step_13(steps)

    assert pc_res.converged is True
    assert step["status"] == "completed"
    assert step["substitutions"]["converged"] is True
    assert "agree" in step["beginnerExplanation"].lower()
    assert "not reliable" not in step["beginnerExplanation"].lower()


def test_trace_step_13_reports_failure_when_not_converged():
    """
    The step used to say "Verified that the math solver converged cleanly with
    zero errors" regardless.  It must now say the opposite when that is what
    happened.
    """
    pc_res, steps = _phase10_trace(*UNCERTIFIABLE[0])
    step = _step_13(steps)

    assert pc_res.converged is False
    assert step["status"] == "warning"
    assert step["substitutions"]["converged"] is False
    assert "not reliable" in step["beginnerExplanation"].lower()
    assert step["substitutions"]["certified"] is False


def test_trace_step_13_states_the_criterion_it_applies():
    """
    The old equation named ``error_estimate <= tolerance`` -- a test the code
    never performed, since the estimate was discarded.
    """
    _, steps = _phase10_trace(*HEALTHY[0])
    step = _step_13(steps)

    assert "error_estimate ≤ tolerance" != step["equation"]
    assert "independent" in step["equation"].lower() or "Pc_independent" in step["equation"]
    assert step["substitutions"]["quadrature_error_estimate"] is not None
    assert step["substitutions"]["convergence_criterion"]


# ---------------------------------------------------------------------------
# G. The closed findings this touches
# ---------------------------------------------------------------------------

def test_multi_object_phase10_path_still_converges():
    """
    P10-05 / P10-06 / P10-07 interaction: an ordinary conjunction must still
    produce a converged, usable probability through the whole chain.
    """
    from theseus.simulation.multi_object import (
        MultiObjectEnvironment, SpacecraftDefinition,
    )

    common = dict(semi_major_axis_km=6778.137, eccentricity=0.0,
                  dry_mass_kg=1000.0, fuel_mass_kg=0.0, payload_mass_kg=0.0,
                  hard_body_radius_m=5.0, sigma_pos_m=[300.0] * 3,
                  sigma_vel_m_s=[0.3] * 3)
    a = SpacecraftDefinition(id="A", name="A", color="#ff9900",
                             inclination_deg=51.6, true_anomaly_deg=0.0, **common)
    b = SpacecraftDefinition(id="B", name="B", color="#3388ff",
                             inclination_deg=-51.6, true_anomaly_deg=0.05, **common)
    env = MultiObjectEnvironment(central_body="Earth", screening_threshold_km=50.0,
                                 coarse_dt_s=5.0, enable_j2=True, enable_drag=True)
    result = env.simulate([a, b], t_start=0.0, t_end=3000.0, output_dt=5.0)

    assert result.conjunctions
    event = min(result.conjunctions, key=lambda e: e.miss_distance_m)
    assert event.collision_probability is not None
    assert 0.0 <= event.collision_probability <= 1.0


def test_risk_api_still_reports_a_converged_probability():
    """The single-pair Phase 10 path, end to end."""
    from fastapi.testclient import TestClient
    from theseus.server.app import app

    client = TestClient(app)
    response = client.post("/api/simulate/conjunction/risk", json={
        "object_a_alt_km": 400.0, "object_a_inc_deg": 51.6, "object_a_phase_deg": 0.0,
        "object_b_alt_km": 400.05, "object_b_inc_deg": 55.0, "object_b_phase_deg": 0.02,
        "analysis_duration_hours": 2.0, "screening_threshold_km": 100.0,
        "coarse_dt_s": 30.0, "hard_body_radius_m": 15.0,
    })

    assert response.status_code == 200
    data = response.json()
    assert data["analysis_status"] == "COMPLETE"
    assert data["collision_probability"]["converged"] is True

    step_13 = next(s for s in data["calculation_steps"] if s["stepIndex"] == 13)
    assert step_13["status"] == "completed"
    assert step_13["substitutions"]["converged"] is True
