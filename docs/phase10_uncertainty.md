# THESEUS Phase 10: Uncertainty Propagation, Covariance Analysis & Probability of Collision

## Overview

Phase 10 extends the THESEUS astrodynamics engine from deterministic state estimation to rigorous, probabilistic uncertainty quantification. It answers:

> *"Where will the spacecraft probably be, how uncertain is that prediction, and what is the probability that two uncertain objects collide during a close encounter?"*

All uncertainty propagation, B-plane mappings, and collision probabilities originate from explicitly derived physical and mathematical formulations. No fabricated numbers, heuristic growth factors, or arbitrary risk formulas are used.

---

## Architecture & Pipeline

```
NOMINAL TRAJECTORY  +  STATE COVARIANCE P(t₀)
                      ↓
           COVARIANCE VALIDATION
                      ↓
             STM DYNAMICS PROPAGATION
             dΦ/dt = A(t) Φ,  Φ(t₀, t₀) = I₆
             P(t) = Φ(t, t₀) P₀ Φ(t, t₀)ᵀ + Q
                      ↓
       TIME OF CLOSEST APPROACH (TCA) (Phase 9)
                      ↓
       RELATIVE COVARIANCE AT TCA
       P_rel = P₁(TCA) + P₂(TCA) - P₁₂ - P₂₁
                      ↓
         B-PLANE UNCERTAINTY PROJECTION
         P_B = M P_rr Mᵀ,  M = [T̂ᵀ; R̂ᵀ]
                      ↓
          UNCERTAINTY ELLIPSE EIGENVALUES
          σ_major = √λ_max,  σ_minor = √λ_min
                      ↓
             HARD-BODY RADIUS (HBR)
             HBR = R₁ + R₂
                      ↓
        PROBABILITY OF COLLISION (Pc)
        2D Gaussian Encounter Plane Quadrature
                      ↓
              RISK CLASSIFICATION
              (Configurable Policy Layer)
```

---

## Mathematical Models

### 1. State Covariance Representation & Validation

For Cartesian state vector $x = [r_x, r_y, r_z, v_x, v_y, v_z]^T \in \mathbb{R}^6$, the state covariance is:

$$P = \mathbb{E}[(x - \bar{x})(x - \bar{x})^T] = \begin{bmatrix} P_{rr} & P_{rv} \\ P_{vr} & P_{vv} \end{bmatrix}$$

where:
- $P_{rr} \in \mathbb{R}^{3 \times 3}$ is the position-position covariance ($\text{m}^2$).
- $P_{vv} \in \mathbb{R}^{3 \times 3}$ is the velocity-velocity covariance ($\text{m}^2/\text{s}^2$).
- $P_{rv} = P_{vr}^T \in \mathbb{R}^{3 \times 3}$ is the position-velocity cross-covariance ($\text{m}^2/\text{s}$).

**Validation Requirements:**
1. Dimensionality: strictly $6 \times 6$.
2. Finiteness: no `NaN` or `Inf` values.
3. Symmetry: $|P - P^T| \le \text{tol} \cdot \max(1, \|P\|_\infty)$. Symmetrized explicitly as $P \leftarrow \frac{1}{2}(P + P^T)$.
4. Positive semi-definiteness: all eigenvalues $\lambda_i(P) \ge -\text{tol}_{\text{psd}}$.
5. Diagonal non-negativity: physical variances $P_{ii} \ge 0$.

---

### 2. State Transition Matrix (STM)

The State Transition Matrix $\Phi(t, t_0)$ maps initial linear perturbations $\delta x(t_0)$ to time $t$:

$$\delta x(t) = \Phi(t, t_0) \, \delta x(t_0)$$

For nominal dynamics $\dot{x} = f(x, t)$, the variational equations are:

$$\frac{d\Phi(t, t_0)}{dt} = A(t) \, \Phi(t, t_0), \quad \Phi(t_0, t_0) = I_6$$

where $A(t) = \frac{\partial f}{\partial x}$ is the $6 \times 6$ dynamics Jacobian:

$$A(t) = \begin{bmatrix} 0_{3 \times 3} & I_{3 \times 3} \\ \frac{\partial a}{\partial r} & \frac{\partial a}{\partial v} \end{bmatrix}$$

#### Analytic Gravity Jacobians
- **Point-Mass Newtonian Gravity** ($a = -\frac{\mu}{r^3} r$):
  $$\frac{\partial a}{\partial r} = \frac{\mu}{r^3} \left( 3 \hat{r} \hat{r}^T - I_3 \right), \quad \frac{\partial a}{\partial v} = 0_{3 \times 3}$$

- **$J_2$ Oblateness Perturbation**:
  Exact second partial derivatives derived from the geopotential gradient.

- **Numerical Jacobian Fallback**:
  Controlled central finite differences for composite or non-conservative force models.

---

### 3. Covariance Propagation

$$P(t) = \Phi(t, t_0) \, P(t_0) \, \Phi(t, t_0)^T + Q(t, t_0)$$

- Process noise $Q(t, t_0)$ is optional and disabled by default ($Q = 0$).
- Covariance propagation is strictly synchronized with the nominal trajectory force models, central body, and epoch.

---

### 4. Relative Covariance

For relative state $x_{\text{rel}} = x_1 - x_2$:

$$P_{\text{rel}} = P_1 + P_2 - P_{12} - P_{21}$$

When cross-covariances $P_{12}, P_{21}$ are unavailable, the objects are treated as statistically independent ($P_{\text{rel}} = P_1 + P_2$), and this assumption is explicitly recorded in the output.

---

### 5. B-Plane Uncertainty Projection

The relative position covariance at TCA is projected into the 2D B-plane orthogonal to the relative approach velocity $\hat{S} = \frac{v_{\text{rel}}}{|v_{\text{rel}}|}$:

$$\hat{T} = \frac{\hat{S} \times \hat{p}}{|\hat{S} \times \hat{p}|}, \quad \hat{R} = \hat{S} \times \hat{T}$$

$$M = \begin{bmatrix} \hat{T}^T \\ \hat{R}^T \end{bmatrix} \in \mathbb{R}^{2 \times 3}$$

$$P_B = M \, P_{\text{rel}, \text{pos}} \, M^T = \begin{bmatrix} \sigma_T^2 & \sigma_{TR} \\ \sigma_{TR} & \sigma_R^2 \end{bmatrix}$$

Eigendecomposition $P_B = V \Lambda V^T$ yields:
- Semi-major axis: $\sigma_{\text{major}} = \sqrt{\lambda_{\max}}$
- Semi-minor axis: $\sigma_{\text{minor}} = \sqrt{\lambda_{\min}}$
- Ellipse orientation angle: $\theta = \text{atan2}(v_{\text{major}, R}, v_{\text{major}, T})$

---

### 6. Probability of Collision ($P_c$)

In the 2D encounter plane (Alfriend, Akella, Chan model), the probability of collision is the integral of the 2D Gaussian PDF over the circular collision disk $D$ of radius $\text{HBR} = R_1 + R_2$ centered at the nominal miss vector $b_0 = [B \cdot T, B \cdot R]^T$:

$$P_c = \frac{1}{2\pi \sqrt{\det P_B}} \iint_D \exp\left( -\frac{1}{2} z^T P_B^{-1} z \right) dz$$

#### Numerical Integration Method
1. Transform to principal axes via eigenvectors of $P_B$:
   $$\mu' = V^T b_0 = [\mu_x, \mu_y]^T$$
2. In polar coordinates centered at $(\mu_x, \mu_y)$:
   $$P_c = \frac{1}{2\pi \sigma_x \sigma_y} \int_0^{\text{HBR}} r \, dr \int_0^{2\pi} \exp\left( -\frac{(\mu_x + r \cos\theta)^2}{2\sigma_x^2} - \frac{(\mu_y + r \sin\theta)^2}{2\sigma_y^2} \right) d\theta$$
3. Integrated deterministically using adaptive 2D quadrature.
4. Guaranteed numerical bounds: $0.0 \le P_c \le 1.0$.

---

### 7. Hard-Body Radius (HBR)

$$\text{HBR} = R_1 + R_2$$

Distinguishes physical body radius from the enclosing collision radius (e.g. solar array wingspans). Presets available for ISS, CubeSats, upper stages, and debris.

---

### 8. Risk Classification

Configurable thresholds:
- $P_c < \text{threshold}_1$ : **LOW**
- $\text{threshold}_1 \le P_c < \text{threshold}_2$ : **ELEVATED**
- $\text{threshold}_2 \le P_c < \text{threshold}_3$ : **HIGH**
- $P_c \ge \text{threshold}_3$ : **CRITICAL**

Default Standard Profile: `low=1e-7`, `elevated=1e-5`, `high=1e-4`.

---

## API Endpoints

### `POST /api/simulate/conjunction/risk`

#### Example Request:
```json
{
  "object_a_alt_km": 400.0,
  "object_a_inc_deg": 51.6,
  "object_a_phase_deg": 0.0,
  "object_b_alt_km": 400.01,
  "object_b_inc_deg": 51.6,
  "object_b_phase_deg": 0.01,
  "central_body": "Earth",
  "analysis_duration_hours": 2.0,
  "cov_a": {
    "sigma_pos_km": [0.5, 0.5, 0.5],
    "sigma_vel_km_s": [0.0005, 0.0005, 0.0005]
  },
  "cov_b": {
    "sigma_pos_km": [0.5, 0.5, 0.5],
    "sigma_vel_km_s": [0.0005, 0.0005, 0.0005]
  },
  "hard_body_radius_m": 15.0
}
```

#### Example Response:
```json
{
  "conjunction_summary": {
    "tca_s": 54.32,
    "miss_distance_km": 0.045,
    "relative_velocity_km_s": 7.68
  },
  "collision_probability": {
    "probability": 0.000342,
    "probability_scientific": "3.421000e-04",
    "method": "2D_Gaussian_Polar_Quadrature_Principal_Axes",
    "converged": true,
    "hard_body_radius_m": 15.0,
    "miss_distance_m": 45.0,
    "sigma_major_m": 720.0,
    "sigma_minor_m": 505.0,
    "ellipse_angle_deg": 45.2
  },
  "risk_assessment": {
    "level": "CRITICAL",
    "probability": 0.000342,
    "action_required": true,
    "recommendation": "Critical risk. Execute collision avoidance maneuver unless updated tracking refines Pc below threshold."
  },
  "calculation_steps": [
    { "stepIndex": 1, "title": "Acquire Object A State Covariance", ... },
    { "stepIndex": 14, "title": "Classify Conjunction Risk", ... }
  ]
}
```
