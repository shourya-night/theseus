# THESEUS — P10-09 + P10-10 CORRECTION REPORT

Scope: `theseus/uncertainty/results.py` (trace step 2), `theseus/uncertainty/collision_probability.py`
(early-return branch criteria). One new test module. P10-04 through P10-08, P9-01
through P9-05, B-1, B-2 and Phase 1–7 untouched.

---

### 1. P10-09 DEFECT REPRODUCED FIRST

**The defect reproduces.**

Trace step 2 named three tests as its equation — `P = Pᵀ`, `λ_i(P) ≥ 0`, `diag(P) ≥ 0` —
and then reported them as literals:

```python
"substitutions": {
    "symmetry_verified": True,
    "psd_verified": True,
    "non_negative_variances": True,
},
"result": "COVARIANCES MATHEMATICALLY VALIDATED",
```

None of the three was computed. Two reachable demonstrations, measured against the
unfixed tree:

**(a) Cancelling asymmetries — reachable through the orchestration.**

| quantity | Object A | Object B |
|---|---|---|
| max abs asymmetry | 4.000e+03 | 4.000e+03 |
| relative asymmetry | 4.000e-01 | 4.000e-01 |
| symmetry tolerance | 1.000e-07 | 1.000e-07 |
| `validate()` verdict | REJECTS | REJECTS |

Their asymmetries cancel in the sum, so `compute_relative_covariance` produced a
matrix with `max|P − Pᵀ| = 0.000e+00` and did not object. Step 2 then reported
`symmetry_verified: True, psd_verified: True, non_negative_variances: True`,
status `completed`, `COVARIANCES MATHEMATICALLY VALIDATED`.

**(b) Direct call** with a covariance whose minimum eigenvalue and minimum variance
are both −9.000e+03 — step 2 again reported all three claims verified.

Honest qualification: gross single-sided corruption *is* caught downstream, because
`compute_relative_covariance` constructs a `StateCovariance` and re-validates. The
defect is not that invalid data always flows freely; it is that step 2's specific
claims about the two input matrices were never earned, and case (a) shows the gap is
reachable rather than hypothetical.

### 2. P10-09 ROOT CAUSE

`theseus/uncertainty/results.py`, `build_phase10_calculation_trace`, step 2.

Data flow: `initial_cov_a/b` → (never inspected here) → step 2's three literals →
reported trace. Everything else in the chain — B-plane projection, principal-axis
transform, Pc inputs — is computed; only step 2's verdict is asserted.

`StateCovariance.validate()` cannot be the source of the claim even though it is real:

- it **raises** rather than reporting, so it has no negative verdict to return;
- it **repairs** — zeroes slightly negative diagonals, symmetrises within `sym_tol`,
  clips small negative eigenvalues — so by the time anything could ask, the answer is
  always yes, and the repair is never disclosed;
- the matrix is a mutable array on a non-frozen dataclass, so validity at construction
  is not validity at trace time.

Classification: **D — reporting-only defect** (fabricated confidence), same class as
step 13 before P10-08. The underlying validation is genuine; the trace was not.

### 3. P10-09 INDEPENDENT REFERENCE

The three claims recomputed directly with numpy — `max|P − Pᵀ|`,
`min λ(½(P + Pᵀ))`, `min diag(P)` — with no production validator involved, so the
reference can disagree with production. Checked over 60 random matrices including
borderline perturbations (1e-9 and 1e3 asymmetries, negative variances), plus the two
reproduction cases.

### 4. P10-09 CORRECTION

`theseus/uncertainty/results.py`:

- new `measure_covariance_validity(cov)` — inspects the matrix **as it stands**,
  returns the measured quantities alongside the three verdicts, mutates nothing, raises
  nothing. Comparisons use the covariance's own `sym_tol` / `psd_tol` and the same
  effective-tolerance scaling `validate()` applies, so the trace and the class agree on
  what "valid" means. Whether that scaling is itself right is **P10-11**, untouched.
- step 2's three fields now come from that measurement; `status` becomes `warning` on
  failure; the result line names the failing quantity and its number; the
  beginner-facing text says the opposite when the check fails.
- the three original keys are preserved as booleans (schema stable), with per-object
  detail added alongside.

After the fix, case (a) reports `symmetry_verified: False`, status `warning`,
`COVARIANCE VALIDATION FAILED — A: asymmetric by 4.000e+03 (relative 4.000e-01 >
1.000e-07); B: …`. Case (b) reports `psd_verified: False, non_negative_variances:
False`, `symmetry_verified: True` — each claim discriminated separately.

### 5. P10-09 REGRESSION TESTS

11 tests. Against the unfixed implementation (measurement forced all-True):
**7 failed, 29 passed.**

| Test | Proves |
|---|---|
| `test_valid_covariances_report_all_three_claims_verified` | no false alarm |
| `test_cancelling_asymmetries_reach_the_trace_and_are_reported` | the reachable case |
| `test_each_claim_is_reported_independently` | a negative variance is not reported as an asymmetry |
| `test_the_verdict_depends_on_the_measured_property` ×3 | each verdict moves only with its own quantity |
| `test_measurement_matches_an_independent_computation` | 60 random matrices vs direct numpy |
| `test_measurement_does_not_repair_the_matrix` | unlike `validate()` |
| `test_step_2_schema_is_preserved` | existing consumers keep working |

---

### 6. P10-10 DEFECT REPRODUCED FIRST

**The defect reproduces, and it produces a wrong probability, not merely an
inelegant branch.**

Pc is dimensionless. For fixed dimensionless geometry — σ_minor/σ_major = 0.2,
|b| = 1.5 σ_major, R = 0.3 σ_major — Pc must not depend on the scale:

| scale | Pc (before) | branch | Pc (after) |
|---|---|---|---|
| 1e+06 m | 5.80078271e-02 | quadrature | 5.80078271e-02 |
| 1e+03 m | 5.80078271e-02 | quadrature | 5.80078271e-02 |
| 1e+00 m | 5.80078271e-02 | quadrature | 5.80078271e-02 |
| 1e-02 m | 5.80078271e-02 | quadrature | 5.80078271e-02 |
| **1e-04 m** | **0.00000000e+00** | **deterministic_limit** | 5.80078271e-02 |
| **1e-06 m** | **0.00000000e+00** | **deterministic_limit** | 5.80078271e-02 |
| **1e-09 m** | **0.00000000e+00** | **deterministic_limit** | 5.80078275e-02 |

A 5.8 % probability reported as exactly impossible, and as `converged = True`, purely
because the same encounter was expressed at a smaller scale.

### 7. P10-10 ROOT CAUSE

`theseus/uncertainty/collision_probability.py`, Edge Cases 2 and 3.

| constant | unit | branch | purpose | dimensionally valid | scale-invariant | required |
|---|---|---|---|---|---|---|
| `hbr_m <= 0.0` | m | zero cross-section | exact | yes (vs zero) | yes | yes |
| `sigma_major < 1e-6` | m | deterministic | avoid 1/σ | **no** | **no** | no |
| `sigma_minor < 1e-6` | m | deterministic | avoid 1/σ | **no** | **no** | no |
| `det_p < 1e-12` | **m⁴** | deterministic | avoid 1/√det | **no** | **no (k⁴)** | no |
| `max(sigma_major, 1.0)` | m | far separation | floor on the σ multiple | **no** | **no** | no |
| `50.0 × max_sigma` | σ | far separation | underflow cutoff | yes | yes given no floor | optimisation |
| `10.0 × max_sigma` | σ | far separation | disk near-edge margin | yes | yes given no floor | optimisation |
| `exponent < -500.0` | — | underflow guard | float protection | yes | yes | yes |
| `QUADRATURE_AGREEMENT_RTOL` | — | P10-08 | agreement | yes | yes | yes |

The determinant test dominates: `det P` has units of m⁴ and scales as the fourth power
of the length unit, so it fires first and hardest. The `max(σ_major, 1.0)` floor is a
separate, milder issue: below one metre it measured separation in metres, disabling the
shortcut. That is **conservative rather than wrong** — the quadrature then returns the
same 0.0 — so it changed the branch, never the answer.

Classification: **A — wrong physics/model in the branch selection**, since a
dimensionless quantity was being decided by a dimensional test.

### 8. P10-10 INDEPENDENT REFERENCE

Monte Carlo sampling of the encounter-plane Gaussian (no quadrature at all) and a dense
two-dimensional polar Gauss-Legendre grid, both refined until settled. Neither is
production's one-dimensional verification integral, so neither can inherit its
behaviour. They agree on the scale-free value:

- dense polar grid, 600 → 1200 nodes: relative change < 1e-8, value 5.8007827e-02;
- Monte Carlo, 2×10⁶ draws: within 5 standard errors of the same value.

### 9. P10-10 CORRECTION

`theseus/uncertainty/collision_probability.py`:

**Deterministic limit** — the branch approximates a limit, so it is now written as a
ratio:

```
boundary_clearance = | |b| − R |
uncertainty_is_negligible = sigma_major <= DETERMINISTIC_LIMIT_SIGMA_RATIO * boundary_clearance
```

`DETERMINISTIC_LIMIT_SIGMA_RATIO = 1e-8`. Derived, not chosen: the neglected mass is the
Gaussian tail beyond 1/ratio standard deviations; at 1e-8 that is exp(−5e15). Exactness
needs about 9σ, so this is conservative by seven orders. When the uncertainty *can* reach
the hit/miss boundary the shortcut is not a limit of anything and the quadrature runs.

**Float safety** — kept dimensional, for a stated reason:
`FLOAT_SAFE_SIGMA_M = 1e-150`. Below roughly 1.2e-154, σ² underflows to zero and
`1/(2σ²)` raises; the bound comes from IEEE-754, not from any physical scale, and no
rescaling can move it. Short-circuit evaluation keeps the division from being reached.

**Far separation** — the floor removed, leaving purely dimensionless multiples:
`FAR_SEPARATION_SIGMA = 50.0`, `FAR_SEPARATION_DISK_SIGMA = 10.0`. exp(−50²/2) is below
the smallest subnormal double, so the branch returns an exact zero rather than a cutoff.

**Cross-finding**: every result now carries `result_kind` in its diagnostics —
`analytic_exact`, `deterministic_limit`, `density_not_representable`, or
`numerical_quadrature` — so an exact analytic `converged=True` is distinguishable from a
P10-08-verified quadrature.

No Pc value changed except where the old branch was demonstrably wrong. No new
dimensional constant was introduced. P10-12's integrator repair was not attempted.

### 10. P10-10 REGRESSION TESTS

19 tests. Against the unfixed branch conditions: **3 failed, 33 passed** — the
scale-invariance cases at 1e-4, 1e-6 and 1e-9, each returning 0.0 against 5.8e-2.

The far-separation change cannot be made to fail by restoring the old floor, because the
old floor produced the same probability (0.0) by a slower route; those tests assert the
criterion via `method` and `result_kind` instead, and the probability assertion pins that
the value did not move.

| Group | Tests | Proves |
|---|---|---|
| Scale invariance | 8 | Pc constant over 15 decades; value matches MC and a dense grid |
| Deterministic limit | 3 | fires on a ratio, agrees with direct integration, does not fire near the boundary |
| Far separation | 2 | scale-free criterion; the density really has underflowed |
| No false early returns | 1 | 40 random ordinary geometries all reach the quadrature |
| Float safety | 1 | the surviving dimensional bound, justified by IEEE-754 |
| P10-04 | 4 | small HBR still reaches the quadrature and converges |

---

### 11. CROSS-FINDING VALIDATION

1. **An early return does not suppress a validation failure** — with both inputs
   asymmetric and a far-separation geometry, Pc takes `analytic_far_separation` and step 2
   still reports `warning` / `symmetry_verified: False`. Pinned by test.
2. **Exact vs converged is distinguishable** — analytic branches report
   `result_kind: analytic_exact` and carry no `verification_disagreement`; the quadrature
   reports `numerical_quadrature` and does. Pinned by test.
3. **P10-08 remains authoritative** — a high-anisotropy case is still flagged
   `converged=False`; a healthy one still is not.
4. P10-09 was not "fixed" by forcing fields to False, nor by making validation depend on
   P10-08: the fields are measured.
5. P10-10 was not "fixed" by adding another arbitrary threshold: two dimensionless
   criteria plus one IEEE-754 bound.
6. P10-12 not implemented.
7. P10-05, P10-06, P10-07 behaviour unchanged.

### 12. FULL REGRESSION

| Suite | Baseline | After |
|---|---|---|
| `tests/` | 823 passed | **859 passed** (+36) |
| `validation/` | 158 passed, 1 failed, 5 skipped | **158 passed, 1 failed, 5 skipped** |

Same known failure, untouched: `test_val_j_ephemeris.py::test_simple_ephemeris_distances`,
147 098 826 715.3002 vs 149 600 000 000.0 ± 1.5e9.

Closed-finding suites: **576 tests, all pass**. Phase 9 fingerprint **bit-identical**.
B-1 unchanged (`f(0) = 0.0`, 2 events at the window start, 3 when shifted 1e-3 s).

### 13. PERFORMANCE

| Measurement | Value |
|---|---|
| `measure_covariance_validity` (one 6×6 `eigvalsh`) | 25.88 µs |
| Full 14-step trace build, before ≈ | 0.100 ms |
| Full 14-step trace build, after | 0.152 ms |
| Pc, quadrature path | 0.806 ms |
| Pc, deterministic path | 0.005 ms |
| Pc, far-separation path | 0.016 ms |

P10-09 adds two 6×6 eigen-decompositions per trace (+0.05 ms) against a Phase 10 chain of
roughly 420 ms — 0.012 %. P10-10 adds no computation at all; the shortcuts still
short-circuit, and the branch test is a handful of comparisons.

### 14. REMAINING ISSUES — REPORTED, NOT FIXED

1. **P10-11 — the PSD tolerance is scaled by the largest matrix entry**
   (`effective_psd_tol = max(psd_tol, scale × 1e-9)` in `covariance.py`). Since the
   position block dominates a state covariance, the velocity block is judged against a
   tolerance set by position variances. `measure_covariance_validity` deliberately mirrors
   this so the trace and the class agree; changing it is P10-11's business.
2. **P10-12 — the quadrature is still inaccurate for high-anisotropy covariances**
   (19–97 % low above σ ratio ≈ 3×10³). P10-08 reports it honestly; the integrator is
   unrepaired. `collision_probability.py`'s own docstring still advertises Chan's series,
   which does not exist in the file.
3. **`max_evals` is an unused parameter** of `compute_collision_probability`.
4. **The deterministic branch is a fallback, not an exact model, when the covariance is
   rank-deficient** (σ_minor → 0). The true answer there is a one-dimensional integral. No
   aspect-ratio branch was added, deliberately: that would collide with P10-08's authority
   over the quadrature and with P10-12.
5. Carried forward unchanged: B-1; the SRP construction crash in the multi-object path;
   the atmosphere model's thermospheric deficiency; backward STM integration; non-finite
   API request fields surfacing as 500; custom-HBR precedence; the unconditional `j2`
   argument at the multi-object STM call site.

### 15. FINAL STATUS

**P10-09 — CLOSED — CORRECTED.** The three validation fields were literals; two reachable
cases reported success for matrices carrying a 4.000e+03 asymmetry and a −9.000e+03
eigenvalue. They are now measured, per claim, without repairing the matrix, with the
schema preserved. 7 of 11 tests fail against the unfixed implementation.

**P10-10 — CLOSED — CORRECTED.** The branch criteria were dimensional; the same encounter
returned 5.80078271e-02 at metre scale and exactly 0.0 below 1e-2 m, reported as
converged. The deterministic limit is now a ratio, the far-separation criterion is a pure
σ multiple, and the one surviving dimensional bound is justified by IEEE-754 and
documented as such. Pc is now invariant over 15 decades of scale, matching Monte Carlo and
a dense independent grid. 3 of 19 tests fail against the unfixed conditions.

Stopped after P10-09 and P10-10. **Not proceeding to P10-11 or P10-12.**
