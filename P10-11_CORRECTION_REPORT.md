# THESEUS — P10-11 CORRECTION REPORT

Scope: `theseus/uncertainty/covariance.py` (validation criteria),
`theseus/uncertainty/results.py` (P10-09 measurement, updated consistently),
`tests/test_phase10_psd_tolerance.py` (new). P10-04 … P10-10, P9-01 … P9-05,
B-1, B-2 and Phase 1–7 untouched. P10-12 not started.

---

### 1. DEFECT REPRODUCED FIRST

**Reproduced, and it reaches a reported risk classification.**

`StateCovariance.validate()` judged both symmetry and positive semi-definiteness
against one scalar derived from the whole matrix:

```python
scale             = max(max|P_ij|, 1.0)
rel_asym          = max|P - P^T| / scale
effective_psd_tol = max(psd_tol, scale * 1e-9)
reject iff  lambda_min(P) < -effective_psd_tol
```

For a state covariance the largest entry is a position variance in m², while the
quantity being judged may live in the velocity block in (m/s)² or in a
position–velocity correlation in m²/s.

**Minimal reproduction** — σ_r = 1 km, σ_v = 1e-4 m/s, correlation ρ between
r_x and v_x. The 2×2 sub-block has determinant σr²σv²(1 − ρ²), so the matrix is
PSD **if and only if |ρ| ≤ 1** — analytic truth, no numerics involved:

| ρ | raw λ_min | λ_min(correlation) | truly PSD | verdict (before) |
|---|---|---|---|---|
| 0.99999 | +1.1918e-11 | +1.0000e-05 | yes | ACCEPT |
| 1.00000 | +1.1718e-11 | 0.0000e+00 | yes | ACCEPT |
| **1.00001** | **+1.1518e-11** | −1.0000e-05 | **NO** | **ACCEPT** |
| **1.00100** | −8.2923e-12 | −1.0000e-03 | **NO** | **ACCEPT** |
| **1.01000** | −1.8928e-10 | −1.0000e-02 | **NO** | **ACCEPT** |
| **1.50000** | −1.2488e-08 | −5.0000e-01 | **NO** | **ACCEPT** |
| **2.00000** | −2.9988e-08 | −1.0000e+00 | **NO** | **ACCEPT** |

A correlation coefficient of **2.0** — impossible for any probability
distribution — was accepted, because the tolerance was 1e-3 m² while the
offending eigenvalue was −3.0e-8. Note ρ = 1.00001: `eigh` returned a
**positive** minimum eigenvalue. The sign is not resolvable in raw coordinates
when the blocks span 1e6 and 1e-8, so no absolute tolerance, however tight,
would have caught it.

**Congruence:** PSD is invariant under `P → S P Sᵀ`. The identical physical
covariance (ρ = 1.01) in five unit systems:

| position unit | velocity unit | raw λ_min | tolerance | verdict (before) |
|---|---|---|---|---|
| m | m/s | −1.8928e-10 | 1.0000e-03 | ACCEPT |
| km | m/s | −2.0100e-10 | 1.0000e-09 | ACCEPT |
| m | mm/s | −2.0100e-04 | 1.0000e-03 | ACCEPT |
| **km** | **mm/s** | −1.9897e-04 | 1.0000e-09 | **REJECT** |
| Mm | µm/s | −2.0100e-08 | 1.0000e-05 | ACCEPT |

Five expressions of one covariance, two different answers.

**Reachable through Phase 10.** The risk API takes `cov_a.matrix_si` directly
(`app.py:1489`). Supplying the ρ = 2.0 matrix returned:

```
HTTP 200   analysis_status: COMPLETE
step 2:    completed, psd_verified=True
Pc:        1.101625e-05
risk:      HIGH
```

A risk classification derived from something that is not a probability
distribution.

**Two further instances of the same defect**, found while measuring:

* the **symmetry** tolerance shares the scalar. At σ_r = 1 km, σ_v = 1e-4 m/s an
  asymmetry of **10 000 %** of the velocity variance passed (rel_asym = 1e-12
  against 1e-7), while at σ_r = σ_v = 1 a 1 % asymmetry was rejected;
* a component with **zero variance covarying with another** was accepted
  (P_vz,vz = 0 with Cov(r_z, v_z) = 1, raw λ_min = −1e-6). Cauchy–Schwarz gives
  |P_ij|² ≤ P_ii P_jj = 0, so this cannot be PSD.

### 2. ROOT CAUSE

`theseus/uncertainty/covariance.py::StateCovariance.validate`, checks 4 and 5,
both using `scale = max(max|P_ij|, 1.0)` computed on line 113.

Data flow: user or propagated matrix → `__post_init__` → `validate()` →
(repair) → B-plane projection → principal axes → Pc → risk → trace step 2.
`validate()` is the only gate, and it is called from every construction site:
`propagation.py:166`, `relative.py:138`, `app.py:1489/1501`,
`multi_object.py:252`.

The declared contract is a floating-point allowance — *"psd_tol: Absolute
tolerance for negative eigenvalues due to floating-point precision"*. The
implementation inflated it to `scale × 1e-9`, about 1e7 times the actual
backward-error bound eps·‖P‖, and tied it to whichever block happened to be
largest. So the implementation had already departed from its own stated intent.

**Classification: A — genuine dimensional/unit inconsistency affecting
correctness.** Not merely "position² and velocity² have different units": the
verdict is demonstrably not congruence-invariant, and the failure is reachable
through the public API all the way to a risk level.

### 3. INDEPENDENT REFERENCE

Three references, none of which can share the production mistake:

1. **Analytic truth.** For the 2×2 embedding used throughout, PSD ⟺ |ρ| ≤ 1.
   This is algebra, not a computation.
2. **Rank-deficient construction.** `A A^T` with `A` 6×5 is singular by
   construction, so the smallest correlation eigenvalue is exactly zero
   mathematically — this calibrates the roundoff floor without reference to any
   tolerance.
3. **`numpy.longdouble`** cross-check of the eigenvalue sign, at higher
   precision than the float64 path production uses.

**Roundoff floor measured:** 4 000 exactly-singular correlation matrices gave no
value below **−1.22e-15**, against the backward-error bound
eps·‖C‖₂ = **1.33e-15** for a 6×6 — agreement to within 10 %.

**Real data measured:** 35 genuine Phase 10 covariances, sampled across LEO,
eccentric, drag-on and drag-off configurations with σ_pos from 10 m to 5 km and
σ_vel from 1 mm/s to 5 m/s, have λ_min(correlation) between **+3.67e-06** and
1.8e-03; none below 1e-6, while their raw condition numbers reach 3.25e+11.

### 4. CORRECTION

**`theseus/uncertainty/covariance.py`**

* New module constant `PSD_CORRELATION_TOL = 1e-12`, with its calibration in the
  comment: ~750× above the measured roundoff floor, ~3.7e6× below the closest
  real datum.
* Check 4 (symmetry) now normalises each entry by `sqrt(P_ii P_jj)` — the only
  quantity with the units of `P_ij`. This is the asymmetry of the correlation
  matrix, so `sym_tol` keeps its documented meaning of a *relative* tolerance and
  becomes relative to the right thing.
* Check 5 (PSD) now tests `λ_min(D⁻¹PD⁻¹)`, `D = diag(sqrt(P_ii))`. `D` is
  nonsingular where variances are positive, so `C` is PSD exactly when `P` is;
  `C` is dimensionless; and normalising by `P`'s own diagonal introduces no scale
  from outside the matrix. No unique external normalisation had to be invented.
* New `_reject_covariance_with_zero_variance()`: Cauchy–Schwarz makes
  `P_ii = 0 ⇒ P_ij = 0`, exactly and without any tolerance, since the only
  quantity with the units of `P_ij` there is zero.
* `psd_tol` keeps its field name and place in the API; its meaning is now
  explicitly the tolerance on the correlation form, documented in full, with the
  default moved from 1e-9 to 1e-12. **No caller anywhere in the repository sets
  `psd_tol` or `sym_tol`**, so no behaviour depends on the old interpretation.
* The raw spectrum is now obtained with `eigvalsh` and the eigenvectors computed
  only when the roundoff repair actually runs — one decomposition instead of two.
* The repair step, the exception type and the message shape are unchanged.

**`theseus/uncertainty/results.py`** — `measure_covariance_validity` mirrors the
new criteria. It remains an independent measurement: it computes the quantities
itself, never calls `validate()`, and still does not mutate the matrix. Two new
reported fields, `min_eigenvalue_basis` and `zero_variance_coupling`, make the
basis explicit rather than implicit.

**One regression I introduced and then caught:** excluding zero/negative-variance
rows from the correlation form meant a matrix with a *negative* variance had a
positive correlation eigenvalue, so the measurement briefly reported
`psd_verified: True` for it. A P10-09 test failed on exactly that. Fixed:
`e_iᵀ P e_i = P_ii < 0` is itself a proof of non-PSD, so a negative variance now
fails the PSD claim directly. `validate()` was never affected — check 3 rejects
negative variances before check 5 runs.

### 5. REGRESSION TESTS

**101 new tests** in `tests/test_phase10_psd_tolerance.py`. Against the restored
pre-P10-11 behaviour: **36 failed, 65 passed** (the 65 being cases the old
criterion got right — small scale ratios, basic PSD, roundoff absorption, real
Phase 10 data, unchanged production outputs).

| Category | Tests | Covers |
|---|---|---|
| A. Basic PSD | 20 | positive definite, exactly PSD (ρ = 1), clearly indefinite (ρ = 2), negative variance, non-finite — at six position/velocity scale pairs |
| B. Near-PSD | 37 | ρ = 1 + ε for ε inside and outside tolerance at every scale; the roundoff floor against 400 rank-deficient constructions |
| C. Block scaling | 10 | indefiniteness confined to the velocity block, the position block, and a mixed direction; no false rejection at any scale ratio |
| D. Unit scaling | 11 | `P' = S P Sᵀ` across five unit systems, valid and invalid, plus a single property test asserting one verdict per ρ |
| E. False accept / reject | 8 | the ρ = 2.0 case; 200 random indefinite matrices; zero-variance coupling; symmetry normalisation both directions |
| F. P10-09 integration | 12 | measurement and validator agree; basis reported; no repair; zero-variance coupling flagged |
| G. Real Phase 10 | 3 | every covariance a real run builds still validates with ≥1e4× tolerance margin; risk-API outputs unchanged; the impossible user covariance yields no risk level |

Two P10-09 tests were updated: one reference recomputed on the correlation basis
(still direct numpy, still independent), one expectation corrected after the
regression above was fixed.

### 6. NUMERICAL VALIDATION

**Verdicts** (σ_r = 1 km, σ_v = 1e-4 m/s):

| ρ | truly PSD | λ_min(corr) | before | after |
|---|---|---|---|---|
| 0.999 | yes | +1.0000e-03 | ACCEPT | ACCEPT |
| 1.000 | yes | 0.0000e+00 | ACCEPT | ACCEPT |
| 1.00001 | NO | −1.0000e-05 | ACCEPT | **REJECT** |
| 1.01 | NO | −1.0000e-02 | ACCEPT | **REJECT** |
| 2.00 | NO | −1.0000e+00 | ACCEPT | **REJECT** |

**Unit systems**, ρ = 1.01: before ACCEPT/ACCEPT/ACCEPT/REJECT/ACCEPT; after
**REJECT in all five**. ρ = 0.99: ACCEPT in all five, before and after.

**Production outputs — unchanged.** Risk API, identical to the values recorded
under P10-04 and re-verified under P10-08 and P10-10:

| HBR | σ_major | σ_minor | Pc | risk | converged | step 2 |
|---|---|---|---|---|---|---|
| 15.0 m | 35554.426534 m | 936.651236 m | 3.34376486e-06 | ELEVATED | True | completed |
| 1.9 m | 35554.426534 m | 936.651236 m | 5.36505133e-08 | LOW | True | completed |
| 0.3 m | 35554.426534 m | 936.651236 m | 1.33754808e-09 | LOW | True | completed |

No collision probability, B-plane quantity or risk classification moved. The only
outputs that changed are verdicts on covariances that were never valid.

### 7. FULL REGRESSION

| Suite | Before | After |
|---|---|---|
| `tests/` | 859 passed | **960 passed** (+101) |
| `validation/` | 158 passed, 1 failed, 5 skipped | **158 passed, 1 failed, 5 skipped** |

Known failure unchanged and untouched:
`test_val_j_ephemeris.py::test_simple_ephemeris_distances`,
147 098 826 715.3002 vs 149 600 000 000.0 ± 1.5e9.

Closed-finding suites: **612 tests, all pass** — Phase 1–7 integrity, P9-01…P9-05,
B-2, P10-04, P10-05, P10-06, P10-07, P10-08, P10-09, P10-10. Phase 9 fingerprint
**bit-identical**. B-1 unchanged (`f(0) = 0.0`, 2 events at the window start, 3
when shifted 1e-3 s).

### 8. PERFORMANCE

| Measurement | Before | After |
|---|---|---|
| One covariance construction + validation | 24.89 µs | **72.41 µs** |
| `measure_covariance_validity` | 25.88 µs | 50.55 µs |
| Risk API, median of 9 | ~230 ms | **229.6 ms** |

The validator now forms the correlation matrix and takes its eigenvalues in
addition to the raw spectrum — +47 µs. Computing the raw eigenvalues with
`eigvalsh` and deferring the eigenvectors to the repair branch recovered 13 µs of
that. A Phase 10 analysis constructs a handful of covariances, so the end-to-end
cost is ~0.3 ms against 230 ms — **0.13 %, within measurement noise**. The cost
buys a verdict that no longer depends on the units the caller chose.

### 9. REMAINING ISSUES — REPORTED, NOT FIXED

1. **P10-12** — the high-anisotropy collision-probability quadrature (19–97 % low
   above σ ratio ≈ 3×10³); `collision_probability.py` still advertises Chan's
   series in its docstring and still does not contain it. Not started.
2. **Check 3's `-1e-15` negative-variance threshold** is itself dimensional: for
   σ_v = 1e-9 m/s it exceeds the entire velocity variance. Left alone — it fires
   before the PSD test, and any matrix it clips to zero is then caught by the new
   Cauchy–Schwarz test, so the composition is safe. Worth a future look.
3. **The API surfaces an invalid `matrix_si` as HTTP 500** rather than a 4xx with
   a diagnostic. Pre-existing, carried forward; what P10-11 changed is that such
   a matrix no longer produces a risk level at all.
4. Carried forward unchanged: unused `max_evals`; the SRP construction crash in
   the multi-object path; the atmosphere model's thermospheric deficiency;
   backward STM integration; non-finite API request fields surfacing as 500;
   custom-HBR precedence; the unconditional `j2` argument at the multi-object STM
   call site; B-1.

### 10. FINAL STATUS

**P10-11 CLOSED — CORRECTED**

The audit's hypothesis held and understated the consequence. The tolerance was
not merely conservative or ill-scaled: the verdict was not invariant under a
change of state units, a correlation coefficient of 2.0 was accepted, and that
matrix produced `risk = HIGH` through the public API. Both the PSD and the
symmetry criteria now run on the correlation form — dimensionless, invariant
under component-wise rescaling, and resolvable in floating point where the raw
eigenvalues are not — with the tolerance calibrated against a measured roundoff
floor rather than chosen. No production Pc, B-plane quantity or risk
classification changed.

Stopped after P10-11. **Not proceeding to P10-12.**
