# P10-12 CORRECTION REPORT

**Finding:** High-anisotropy collision-probability quadrature
**Scope:** P10-12, plus the mandatory final sweep of all remaining P10 issues
**Status:** Closed. This is the final P10 task.

---

## 1. Summary

The collision-probability quadrature understated Pc by 19 % to 97 % on valid
encounter geometries, silently. P10-08 established that the failure was real
and made the `converged` flag capable of reporting it, but left the wrong
number as the one returned to the caller. P10-12 finds the mechanism exactly,
fixes the value, and keeps every geometry that was already correct byte-for-byte
unchanged.

The mechanism is not "adaptive quadrature is inexact". QUADPACK's initial
21-point Gauss–Kronrod rule on `[0, 2π]` has its nearest node **0.1971573532 rad**
from each crossing of the probability ridge. Beyond a radius of about
**160.4 σ_minor**, every node of that rule returns exactly zero, so the rule
reports an integral of `0` with an error estimate of `0`, and QAGS accepts it
without subdividing. An adaptive algorithm cannot detect a feature it never
sampled. The resulting loss is predictable in closed form and matches
measurement to a constant factor of 0.9935 across five decades of anisotropy.

The fix evaluates the same integral three independent ways and reports what at
least two of them agree on. Where the polar quadrature is in the agreeing pair —
every geometry that was already right — its own value is what is reported.

The investigation also turned up a second instance of the same defect class (a
disk that engulfs the distribution, returning ~0 where the truth is exactly 1.0),
and the mandatory sweep that followed found five further issues classified
FIX NOW.

**Test suite: 960 → 1058 passing. `validation/` unchanged at 158 passed / 1
failed (pre-existing) / 5 skipped. Phase 9 fingerprint bit-identical. Recorded
production probabilities unchanged to 1e-12. Phase 11 was not started.**

---

## 2. Reproduction, before any edit

Measured against a composite Gauss–Legendre reference, self-converged to 2e-15
and cross-checked by Monte Carlo:

| σ ratio | HBR | Pc engine | Pc reference | rel err | converged |
|---|---|---|---|---|---|
| 1e+03 | 20 m | 1.593665e-02 | 1.593665e-02 | 6.5e-16 | True |
| 3e+03 | 60 m | 1.595441e-02 | 1.595441e-02 | 3.0e-15 | True |
| **1e+04** | **200 m** | **1.288004e-02** | **1.595643e-02** | **19.3 %** | False |
| **3e+04** | **600 m** | **4.293511e-03** | **1.595661e-02** | **73.1 %** | False |
| **1e+05** | **2000 m** | **1.288226e-03** | **1.595663e-02** | **91.9 %** | False |
| **5e+05** | **5000 m** | **2.576128e-04** | **7.978712e-03** | **96.8 %** | False |

All four tabulated failures reproduced exactly. P10-08's convergence flag
reported `False` on precisely those four and `True` on the four healthy rows, so
P10-08 was doing its job — the defect that remained was that the reported *value*
was still the wrong one.

---

## 3. Root cause

### 3.1 The decisive observation

Instrumenting the production integrand and running QUADPACK's inner `quad` call
at a fixed radius:

```
  ratio 1e5 (92% low)   (sigma_x = 1 m, HBR = 2000 m)
           r      inner quad      inner true      rel err   quad err est   nevals
       200.0    0.000000e+00    5.013309e+00    1.000e+00      0.000e+00       21
       600.0    0.000000e+00    5.013173e+00    1.000e+00      0.000e+00       21
      1000.0    0.000000e+00    5.013008e+00    1.000e+00      0.000e+00       21
      2000.0    0.000000e+00    5.012255e+00    1.000e+00      0.000e+00       21
```

Exactly `0.0`, with an error estimate of exactly `0.0`, after exactly **21**
function evaluations. 21 is the size of QUADPACK's initial Gauss–Kronrod rule.
The rule never subdivided because a rule that samples only zeros reports zero
error.

### 3.2 The mechanism, stated precisely

1. In principal axes the density is a ridge along the major axis whose
   half-width in θ at radius `r` is `σ_minor / r`.
2. The GK21 rule on `[0, 2π]` places its nearest node **0.1971573532 rad** from
   each ridge crossing — a fixed offset, independent of the integrand
   (recovered from the rule itself, not quoted).
3. The production integrand returns `0` when its exponent falls below `-500`,
   i.e. when `|u| > √1000 · σ_minor`. Every node is therefore zeroed once
   `r · 0.1971573532 > √1000 · σ_minor`, that is once

   ```
   r > r_crit = √1000 / 0.1971573532 · σ_minor = 160.39 · σ_minor
   ```

4. The outer `r`-integral accumulates nothing beyond `r_crit`, so

   ```
   Pc_reported / Pc_true  ≈  160.39 · σ_minor / HBR
   ```

### 3.3 The prediction, checked

| σ ratio | HBR | measured ratio | predicted | pred/meas |
|---|---|---|---|---|
| 1e+04 | 200 | 0.807200 | 0.801968 | 0.9935 |
| 3e+04 | 600 | 0.269074 | 0.267323 | 0.9935 |
| 1e+05 | 2000 | 0.080733 | 0.080197 | 0.9934 |
| 3e+05 | 5000 | 0.032288 | 0.032079 | 0.9935 |
| 5e+05 | 5000 | 0.032288 | 0.032079 | 0.9935 |

A constant factor of 0.9935 across five decades. The root cause is quantitative,
not narrative.

### 3.4 The `-500` clamp is the trigger, not the cause

Removing the clamp entirely moves the cliff to IEEE underflow near `-745`; it
does not remove it. Relative error without the clamp: 1.5e-2, **67.2 %**,
**90.2 %**, **96.1 %** at ratios 1e4, 3e4, 1e5, 5e5. "Widen the clamp" is the
obvious wrong fix, and it is measurably wrong.

### 3.5 The failure is independent of the miss distance

Sweeping `miss / HBR` from 0 to 3 changes the relative error by less than 0.01.
It is a property of the ridge geometry, so a fix keyed to the miss distance would
be keyed to the wrong thing.

---

## 4. Chan's series: evaluated, then rejected

The module docstring advertised *"Chan's series expansion for isotropic and
mildly anisotropic cases."* No such code existed. Rather than write what was
advertised, it was implemented and measured against 50-digit arithmetic.

The implementation is correct — it reproduces isotropic encounters to
0 – 1.3e-14, which establishes that what follows is a property of the method and
not of a mis-implementation:

| σ_major/σ_minor | Pc true | Chan | rel err |
|---|---|---|---|
| 1 | 8.646647e-01 | 8.646647e-01 | 0.00e+00 |
| 1 | 6.679970e-04 | 6.679970e-04 | 1.30e-14 |
| 10 | 1.392876e-04 | 1.390284e-04 | 1.86e-03 |
| **1e+03** | **7.259839e-03** | **1.982831e-05** | **9.97e-01** |
| **4e+04** | **3.989356e-03** | **3.934693e-01** | **9.76e+01** |
| **5e+05** | **7.978712e-03** | **1.000000e+00** | **1.24e+02** |

Chan's equal-area substitution — replacing the scaled elliptical cross-section
by a circle of the same area — is an approximation, not an identity, and it fails
precisely in the regime this module had trouble with. It overstates Pc by up to
two orders of magnitude, which is no better than understating it.

**The series was not adopted. The advertisement was removed rather than
implemented.**

---

## 5. A predictive switching rule was tried and rejected on evidence

The natural design — refuse the polar quadrature once the disk spans more than
N ridge widths — does not work. Over 337 random geometries:

- largest ridge-resolution number among **accurate** polar results: **155.0**
- smallest among **inaccurate** ones: **2.03**

No threshold in that variable separates them, because the polar quadrature has
at least three unrelated failure modes:

1. the ridge stepped over in θ (the P10-12 case),
2. a spike stepped over in `r` (near-isotropic tiny σ with a large disk),
3. crescent geometries where the disk excludes the density centre.

So the switch is **measured, not predicted**. This is stronger than the brief's
requested dimensionless criterion: it catches all three failure modes rather
than the one whose geometry was known in advance, and it introduces no tuned
constant at all.

---

## 6. The correction

### 6.1 The orientation insight

The exact 1-D reduction is **orientation-dependent**. Whichever axis is passed
as `sigma_x` is integrated numerically; the other is integrated in closed form.
The outer factor is a spike of width `~12 σ_int / R`, so the orientation that
integrates the **narrow** axis needs `O(R / σ_minor)` panels while the one that
integrates the **broad** axis is smooth on an order-unity scale.

Measured at needle geometry (σ_minor = 1 m, σ_major = 5e5 m, HBR = 5 km),
against a closed form:

| panels | minor-axis orientation | major-axis orientation |
|---|---|---|
| 64 | 3.78e-02 | 2.07e-08 (at the reference's own floor) |
| 256 | 3.13e-04 | 2.00e-08 |
| 1024 | 2.00e-08 | 2.00e-08 |

Two evaluations of the same identity with complementary conditioning.

### 6.2 The certificate

`certified_disk_integral` evaluates three independent constructions — the
adaptive polar quadrature, and the reduction in each orientation, each
self-refined until successive panel doublings agree — and reports the value at
least two of them agree on. The polar quadrature is preferred whenever it
belongs to an agreeing pair, so results that were already correct are not
perturbed.

Safety, measured over 208 random geometries spanning σ from 1e-2 to 1e3 m,
anisotropy 1 to 1e6, HBR 0.1 m to 5 km:

- both orientations settled and **disagreed with each other**: **0**
- both settled, agreed, and were **both wrong**: **0**
- worst error where both settled and agreed: **1.5e-11**

### 6.3 Supporting numerical repairs

**Cancellation-safe inner integral.** `Φ((μ+h)/σ) − Φ((μ−h)/σ)` with both
endpoints in the same tail is a difference of numbers near 1. Reflecting to the
lower tail is exact by symmetry and buys five to seven digits on far-tail
encounters — the regime where Pc actually matters:

| geometry | before | after |
|---|---|---|
| σ=(0.5, 40), μ_y=260, R=3 | 4.31e-08 | 2.52e-15 |
| σ=(1, 1e4), μ_y=6e4, R=200 | 3.84e-09 | 8.07e-15 |
| σ=(100, 1e6), μ_y=5e6, R=50 | 4.31e-09 | 6.82e-14 |

**Derived starting panel count.** `n ≥ π R / (12 σ) ≈ 0.262 R/σ`, with a factor
of two of margin, rounded to a power of two. A floor, not a guarantee — 37 of
252 measured orientations needed more — so refinement by doubling still runs.

**Blocked evaluation.** Panels are evaluated in chunks of 4096 so a large panel
count does not allocate a proportionally large temporary.

### 6.4 The engulfment branch

The same defect class, found by random sweep during this investigation: with the
density a spike of width centimetres and the disk hundreds of metres across, the
adaptive subdivision in `r` steps over the spike and returns essentially zero —
**the certain collision reported as the impossible one** (measured relative
error 1.00 at σ = 0.0146 / 0.0488 m, HBR = 536.9 m). These geometries also
defeat the reduction, whose panel requirement reaches 2.3e7 here.

The branch is analytic and exact: the mass outside `ENGULFMENT_SIGMA = 12`
standard deviations is bounded by `exp(-k²/2) = 5.4e-32`, sixteen orders below
the spacing of doubles near 1, so `1.0` is the correctly rounded double.
Verified in 50-digit arithmetic: `1 − Pc` evaluates to exactly `0.0` on every
geometry satisfying the test. It is the exact mirror of the existing
`analytic_far_separation` branch.

### 6.5 New constants, each justified

| constant | value | basis |
|---|---|---|
| `ENGULFMENT_SIGMA` | 12.0 | `exp(-72) = 5.4e-32` < eps near 1 by 16 orders; dimensionless |
| `REDUCTION_REFINEMENT_RTOL` | 1e-12 | 208 geometries: worst agreement-vs-truth 1.5e-11 when settled at this tolerance |
| `REDUCTION_PANEL_CAP` | 2^15 | 786 432 evaluations; where cost stops being justifiable against accuracy still gained |
| `DIAGONAL_NOISE_RTOL` | 1e-12 | same order and justification as `PSD_CORRELATION_TOL`; dimensionless |

No dimensional threshold was introduced.

---

## 7. Result after the correction

| case | reported | 50-digit arbiter | rel err | converged | method |
|---|---|---|---|---|---|
| healthy 5e2/5e1 | 1.989950e-03 | 1.989950e-03 | 0.00e+00 | True | polar quadrature |
| healthy 1e3/1 | 1.593665e-02 | 1.593665e-02 | 6.5e-16 | True | polar quadrature |
| healthy 2e2/2e2 | 2.808549e-03 | 2.808549e-03 | 0.00e+00 | True | polar quadrature |
| **1e4/1** | **1.595643e-02** | 1.595643e-02 | **2.4e-15** | True | exact reduction |
| **3e4/1** | **1.595661e-02** | 1.595661e-02 | **2.4e-14** | True | exact reduction |
| **1e5/1** | **1.595663e-02** | 1.595663e-02 | **4.8e-14** | True | exact reduction |
| **5e5/1** | **7.978712e-03** | 7.978712e-03 | **1.1e-13** | True | exact reduction |
| **sub-metre σ** | **1.060702e-01** | 1.060702e-01 | **3.2e-14** | True | exact reduction |
| **engulfment** | **1.000000e+00** | 1.000000e+00 | **0.00e+00** | True | analytic engulfment |

Every healthy case keeps its exact previous value and its previous method name.
Every failure is now correct to at least 13 significant digits.

---

## 8. P10-08 was not weakened

P10-08's contract is that `converged` is **measured**, and that a result which
cannot be corroborated **says so**. Both halves survive, and both were checked
adversarially.

- **`converged` can still be False.** The old witnesses are now computed
  correctly, so new ones were located by random sweep: at `HBR / σ_minor` of 1e5
  to 1e6 the minor-axis reduction cannot resolve the disk edge within the panel
  cap, so only one construction settles. One is not two, and the result is
  reported uncertified.
- **Proof of teeth:** hard-coding `converged = True` again breaks **7** tests,
  including the new witnesses.
- **The anti-substitution rule is preserved, stated the way it was meant.** The
  prohibition is on changing the answer on the strength of a method the `method`
  field does not name. Whichever construction supplies the number, `method` says
  so, and every discarded candidate stays visible in the diagnostics.
- **The polar quadrature is still computed, still compared, still found
  wanting** on the superseded geometries. Its value is retained under
  `raw_probability` so the substitution can never be invisible.

One judgement call, recorded explicitly: when exactly **one** construction
settles and it contradicts the polar quadrature, the settled reduction is what
is reported, with `converged = False` and `method = Exact_1D_Reduction_Uncorroborated`.
On the witness geometry the polar quadrature returns `0.0` for an encounter that
is very nearly certain; printing that, however loudly labelled, would be the
worst output this module can produce. The reduction refined by doubling until
successive panel counts agreed to 1e-12; the quadrature's own error estimate is
known to be worthless in exactly this regime.

---

## 9. Final P10 sweep

Every remaining known issue was **reproduced, not recalled**, before being
classified.

| # | Issue | Classification | Evidence |
|---|---|---|---|
| A | Unconditional `j2=self.body.J2` at the multi-object STM call site | **FIX NOW** | Gravity-only model + `j2=1.08e-3` → analytic Jacobian rejected by P10-06's guard, numerical path taken unnecessarily. Never wrong, always slower. |
| B | `max_evals` declared, documented, never read | **FIX NOW** | `max_evals=1` and `max_evals=1e9` → byte-identical Pc after an identical 777 evaluations. |
| C | Non-finite request fields at the API | **FIX NOW** | `screening_threshold_km = NaN` → **HTTP 200, `"events": []`**. B-2's exact defect at the boundary B-2 did not cover. |
| D | Custom hard-body radius precedence | **ALREADY CLOSED** (P10-04) | Documented precedence with provenance on both objects; verified unchanged. |
| E | SRP construction crash | **FIX NOW** | `enable_srp=True` → `TypeError: missing 'ephemeris'` → bare HTTP 500 on a documented API option. |
| F | Backward STM integration | **FIX NOW** | `‖Φ_back − I‖ = 0.0`, `‖Φ_back Φ_fwd − I‖ = 1046`. Returned the identity — a covariance came back unpropagated. |
| G | Covariance check 3's `-1e-15` diagonal threshold | **FIX NOW** | Same physical covariance: rejected in metres, accepted in km and Mm. |
| H | B-1 | **REMAINING KNOWN LIMITATION** | Unfixed by explicit instruction; pinned by test so "unfixed" stays a decision. |
| I | Atmosphere thermospheric deficiency | **OUT OF SCOPE FOR P10** | Classified under Phase 8; propagation physics, untouched. |

### The six fixes

**A — the STM asks for the J2 its force model applies.** `_analytic_j2_for`
returns `body.J2` only when a `J2Perturbation` is actually in the model. A
gravity-only model (`enable_j2=False`, exposed by `/api/simulate/environment`)
now reaches the analytic Jacobian instead of being downgraded. Correctness was
never affected — P10-06's guard caught the mismatch every time — but the guard
was firing on a case it was not written for, which obscures the cases it was.
The two Jacobians are asserted to agree to 1e-7.

**B — `max_evals` says what it does.** Kept for compatibility; passing anything
other than the default now raises a `DeprecationWarning` naming the real bound
(`REDUCTION_PANEL_CAP`). The default path stays silent.

**C — B-2's guarantee extended to the API boundary.** `FiniteFieldsModel`
rejects non-finite floats in every simulation request model, walking nested
sub-models and lists. A `RequestValidationError` handler sanitises FastAPI's
echo of the rejected input — without it the 422 became a 500, because the body
could not encode the NaN that caused the rejection. Before/after:

| field | before | after |
|---|---|---|
| `object_a_alt_km = NaN` | HTTP 500 | HTTP 422 `NON_FINITE_REQUEST_FIELD` |
| `object_a_inc_deg = Infinity` | HTTP 500 | HTTP 422 `NON_FINITE_REQUEST_FIELD` |
| `screening_threshold_km = NaN` | **HTTP 200, `"events": []`** | HTTP 422 `NON_FINITE_REQUEST_FIELD` |
| `analysis_duration_hours = -Infinity` | **HTTP 200, `"events": []`** | HTTP 422 `NON_FINITE_REQUEST_FIELD` |
| `cov_a.sigma_pos_km = [.., NaN]` | HTTP 200 | HTTP 422, names `sigma_pos_km[2]` |

The response carries no analysis payload: a non-finite request has no analysis,
not an empty one. Ordinary type errors keep their ordinary 422 shape.

**E — SRP refuses instead of crashing.** `SolarRadiationPressureUnavailable`
(a `NotImplementedError`) → HTTP **501**. Supplying an ephemeris here would make
the flag "work" while switching on a perturbation this project has never
validated end to end, silently changing every trajectory of anyone who sets it.
A documented option that has never run is not made trustworthy by making it run;
it is made honest by saying it is not available. **The wiring remains a known
limitation.**

**F — backward STM propagation refuses.** Raises `ValueError` naming the
identity trap and pointing to propagate-forward-and-invert. The Phase 10
pipeline always propagates from the covariance epoch forward to TCA, so this
costs no production behaviour — asserted by test. `t0 == tf` is genuinely the
identity and stays allowed.

**G — the covariance diagonal test is dimensionless.** Compared against the
largest variance in its **own block**, so position and velocity are judged on
their own scales (with σ_r = 1 km and σ_v = 1e-4 m/s a single scale would absorb
any negative velocity variance as position-block roundoff). Verdicts are now
identical across four unit systems; a test proves the retired absolute rule
disagreed with itself.

---

## 10. Verification

| check | result |
|---|---|
| `tests/` | **1058 passed** (960 before P10-12) |
| `validation/` | **158 passed, 1 failed, 5 skipped** — unchanged; the failure is the pre-existing `test_val_j_ephemeris.py::test_simple_ephemeris_distances` |
| Closed-finding suites (B-2, P10-04 … P10-11) | **536 passed** |
| Phase 9 fingerprint | **bit-identical** |
| Recorded production Pc (HBR 15.0 / 1.9 / 0.3 m) | **3.3437648624900262e-06 / 5.365051327225372e-08 / 1.3375480817909169e-09** — unchanged to 1e-12 |
| Healthy-geometry Pc values | unchanged, value and method name |
| B-1 | unchanged |

### New tests

- `tests/test_phase10_pc_anisotropy.py` — **52 tests**
- `tests/test_phase10_final_sweep.py` — **30 tests**
- `tests/test_phase10_pc_convergence.py` — updated to preserve every P10-08
  invariant under the new behaviour

### Proof of teeth

Reverting only the P10-12 *decision* — keeping the certificate computed but
ignoring its verdict, and removing the engulfment branch, i.e. exactly the
P10-08 resting point — fails **19 of 52** anisotropy tests: all four tabulated
cases against Monte Carlo *and* against an independent ridge-aligned 2-D grid,
the scaling law, the closed-form needle case, the sub-metre case, both
engulfment cases, and the trace.

Hard-coding `converged = True` fails **7** tests, so P10-08's flag still has
teeth.

The 33 anisotropy tests that pass against the unfixed code are the ones that
*describe the defect* — by design, they must pass in both states.

### Independent references used

- **50-digit tanh-sinh arbiter** (arbitrary precision, different arithmetic,
  different quadrature family, different substitution). Its own first version
  missed the ridge on offset geometries and was corrected before being trusted —
  the arbiter is not automatically immune to the defect it arbitrates.
- **Monte Carlo**, 2e6–8e6 draws. No quadrature at all. Agreement `|z| ≤ 1.8` on
  every case it can reach.
- **Dense ridge-aligned 2-D Cartesian grid.** Two dimensions where the fix uses
  one, no analytic inner integral.
- **Closed forms** where the geometry admits them: `2Φ(HBR/σ_major) − 1` for the
  needle case, the small-disk expansion to second order for tiny HBR.

---

## 11. Performance

| geometry | cost |
|---|---|
| Recorded production encounter (HBR 15 m) | 0.89 ms |
| Healthy 5e2/5e1, HBR 10 m | 0.77 ms |
| Superseded 1e5/1, HBR 2000 m | 106 ms |
| Superseded 5e5/1, HBR 5000 m | 116 ms |

The superseded geometries are dominated by the polar `dblquad` itself
(333 000 – 677 000 evaluations), which is the pre-existing cost — the adaptive
quadrature is still run so that its disagreement can be *measured* rather than
predicted, since §5 shows prediction does not work. The two reduction
orientations are vectorised and negligible beside it. The healthy path is
unchanged.

---

## 12. Remaining known limitations

1. **B-1** — unfixed by explicit instruction. Pinned by test.
2. **Solar radiation pressure is not wired up.** Now refuses honestly (501)
   instead of crashing. Making it work needs an ephemeris provider at the
   force-model assembly site *and* end-to-end validation of the perturbation.
3. **Geometries with `HBR / σ_minor ≳ 1e5` cannot be certified.** Only one
   construction settles; the result is reported from it with `converged = False`.
   Conservative by design — one construction is not two.
4. **The polar `dblquad` is still run on geometries where it is known to fail**,
   costing 300 000+ evaluations for a value that is then discarded. A predictive
   skip was measured and rejected (§5); a *sufficient-condition* skip would be
   safe but would change reported values in the last ulps and was judged not to
   earn that risk in the final task.
5. **Atmosphere thermospheric deficiency** — Phase 8, out of scope, untouched.
6. **Known `validation/` failure** — `test_val_j_ephemeris.py::test_simple_ephemeris_distances`
   (1.47e11 vs 1.496e11 ± 1.5e9). Pre-existing, explicitly out of scope, not
   touched.

---

## 13. Final status

P10-12 is closed. The high-anisotropy quadrature defect is corrected at its
root, with the mechanism understood to a predictive closed form rather than
described, and with the correction verified against three references that share
nothing with it.

The mandatory final sweep is complete: nine issues classified on measured
evidence, six fixed with regression tests, one confirmed already closed, one
out of scope, one deliberately unchanged. Every remaining limitation is recorded
above rather than left implicit.

No dimensional threshold was introduced. No production value moved. No
architectural refactor was performed. P10-08 was strengthened, not weakened.
The Phase 9 fingerprint is bit-identical.

**Phase 11 was not started because of the project time constraint.**

**Confirmed: Phase 11 was not started.**
