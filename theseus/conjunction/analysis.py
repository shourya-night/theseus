"""
Conjunction analysis — full pipeline.

Orchestrates:
    screening → TCA refinement → encounter characterisation → optional B-plane

Produces a ConjunctionResult containing all ConjunctionEvents,
encounter geometry, and calculation traces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from theseus.conjunction.screening import ConjunctionScreener, CandidateInterval
from theseus.conjunction.tca import find_tca, find_all_tca, TCAResult
from theseus.conjunction.b_plane import compute_b_plane, BPlaneResult


@dataclass
class ConjunctionEvent:
    """
    A single conjunction event (close approach).

    Attributes
    ----------
    tca_result : TCAResult
        Full TCA solution.
    encounter_angle_deg : float
        Angle between velocity vectors at TCA (0° = same direction,
        180° = head-on).
    encounter_type : str
        'head-on', 'overtaking', or 'crossing'.
    b_plane : BPlaneResult | None
        B-plane analysis (if applicable).
    """
    tca_result: TCAResult
    encounter_angle_deg: float
    encounter_type: str
    b_plane: Optional[BPlaneResult] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tca": self.tca_result.to_dict(),
            "encounter_angle_deg": self.encounter_angle_deg,
            "encounter_type": self.encounter_type,
            "b_plane": self.b_plane.to_dict() if self.b_plane is not None else None,
        }


@dataclass
class ConjunctionResult:
    """
    Complete result of a conjunction analysis.

    Attributes
    ----------
    events : list[ConjunctionEvent]
        All detected conjunction events.
    screening_threshold_m : float
        Threshold used for coarse screening (m).
    analysis_window : tuple[float, float]
        [t_start, t_end] of the analysis (s).
    candidate_intervals : list[CandidateInterval]
        Raw screening results.
    model_metadata : dict[str, Any]
        Physical model and numerical method metadata.
    calculation_steps : list[dict[str, Any]]
        Structured calculation trace.
    """
    events: list[ConjunctionEvent] = field(default_factory=list)
    screening_threshold_m: float = 0.0
    analysis_window: tuple[float, float] = (0.0, 0.0)
    candidate_intervals: list[CandidateInterval] = field(default_factory=list)
    model_metadata: dict[str, Any] = field(default_factory=dict)
    calculation_steps: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [e.to_dict() for e in self.events],
            "screening_threshold_km": self.screening_threshold_m / 1e3,
            "analysis_window_s": list(self.analysis_window),
            "candidate_intervals": [
                {
                    "t_start_s": c.t_start,
                    "t_end_s": c.t_end,
                    "min_distance_km": c.min_distance / 1e3,
                }
                for c in self.candidate_intervals
            ],
            "model_metadata": self.model_metadata,
            "calculation_steps": self.calculation_steps,
            "summary": {
                "total_events": len(self.events),
                "closest_approach_km": (
                    min(e.tca_result.miss_distance for e in self.events) / 1e3
                    if self.events else None
                ),
            },
        }


def classify_encounter(v_a: np.ndarray, v_b: np.ndarray) -> tuple[float, str]:
    """
    Classify the encounter geometry.

    Parameters
    ----------
    v_a, v_b : (3,) arrays
        Velocity vectors of objects A and B at TCA.

    Returns
    -------
    (angle_deg, classification)
        angle_deg : angle between velocity vectors (0–180°).
        classification : 'head-on', 'overtaking', or 'crossing'.
    """
    va_mag = float(np.linalg.norm(v_a))
    vb_mag = float(np.linalg.norm(v_b))

    if va_mag < 1e-10 or vb_mag < 1e-10:
        return 0.0, "indeterminate"

    cos_angle = float(np.dot(v_a, v_b) / (va_mag * vb_mag))
    cos_angle = max(-1.0, min(1.0, cos_angle))  # clamp for numerical safety
    angle_deg = math.degrees(math.acos(cos_angle))

    if angle_deg > 150:
        return angle_deg, "head-on"
    elif angle_deg < 30:
        return angle_deg, "overtaking"
    else:
        return angle_deg, "crossing"


class ConjunctionAnalysis:
    """
    Full conjunction analysis pipeline.

    Parameters
    ----------
    screening_threshold_m : float
        Distance threshold for coarse screening (m).
    coarse_dt : float
        Coarse time step for screening (s).
    tca_tol : float
        TCA solver convergence tolerance (s).
    """

    def __init__(
        self,
        screening_threshold_m: float = 100_000.0,
        coarse_dt: float = 60.0,
        tca_tol: float = 1e-6,
    ) -> None:
        self.screener = ConjunctionScreener(
            threshold_m=screening_threshold_m,
            coarse_dt=coarse_dt,
        )
        self.tca_tol = tca_tol
        self.threshold = screening_threshold_m

    def analyse(
        self,
        pos_fn_a: Callable[[float], np.ndarray],
        vel_fn_a: Callable[[float], np.ndarray],
        pos_fn_b: Callable[[float], np.ndarray],
        vel_fn_b: Callable[[float], np.ndarray],
        t_start: float,
        t_end: float,
    ) -> ConjunctionResult:
        """
        Run the full conjunction analysis pipeline.

        Parameters
        ----------
        pos_fn_a, vel_fn_a : callable
            Position and velocity functions for object A.
        pos_fn_b, vel_fn_b : callable
            Position and velocity functions for object B.
        t_start, t_end : float
            Analysis window (s).

        Returns
        -------
        ConjunctionResult
        """
        calc_steps: list[dict] = []
        step_idx = 1

        # --- STEP 1: Acquire Object A state ---
        r_a_0 = np.asarray(pos_fn_a(t_start))
        v_a_0 = np.asarray(vel_fn_a(t_start))
        calc_steps.append({
            "stepIndex": step_idx,
            "phase": "PHASE_09",
            "title": "Acquire Object A State",
            "status": "completed",
            "equation": "r₁(t₀), v₁(t₀)",
            "substitutions": {
                "r1_km": (r_a_0 / 1e3).tolist(),
                "v1_km_s": (v_a_0 / 1e3).tolist(),
            },
            "result": f"r₁ = [{r_a_0[0]/1e3:.1f}, {r_a_0[1]/1e3:.1f}, {r_a_0[2]/1e3:.1f}] km",
            "units": "km, km/s",
            "explanation": "Initial state vector of Object A at the start of the analysis window.",
            "beginnerExplanation": "Where Object A is and how fast it's moving at the start.",
        })
        step_idx += 1

        # --- STEP 2: Acquire Object B state ---
        r_b_0 = np.asarray(pos_fn_b(t_start))
        v_b_0 = np.asarray(vel_fn_b(t_start))
        calc_steps.append({
            "stepIndex": step_idx,
            "phase": "PHASE_09",
            "title": "Acquire Object B State",
            "status": "completed",
            "equation": "r₂(t₀), v₂(t₀)",
            "substitutions": {
                "r2_km": (r_b_0 / 1e3).tolist(),
                "v2_km_s": (v_b_0 / 1e3).tolist(),
            },
            "result": f"r₂ = [{r_b_0[0]/1e3:.1f}, {r_b_0[1]/1e3:.1f}, {r_b_0[2]/1e3:.1f}] km",
            "units": "km, km/s",
            "explanation": "Initial state vector of Object B at the start of the analysis window.",
            "beginnerExplanation": "Where Object B is and how fast it's moving at the start.",
        })
        step_idx += 1

        # --- STEP 3: Initial relative state ---
        r_rel_0 = r_a_0 - r_b_0
        v_rel_0 = v_a_0 - v_b_0
        d_0 = float(np.linalg.norm(r_rel_0))
        calc_steps.append({
            "stepIndex": step_idx,
            "phase": "PHASE_09",
            "title": "Calculate Relative Position",
            "status": "completed",
            "equation": "r_rel = r₁ − r₂",
            "substitutions": {
                "r_rel_km": (r_rel_0 / 1e3).tolist(),
            },
            "result": f"r_rel = [{r_rel_0[0]/1e3:.1f}, {r_rel_0[1]/1e3:.1f}, {r_rel_0[2]/1e3:.1f}] km",
            "units": "km",
            "explanation": "Vector from Object B to Object A.",
            "beginnerExplanation": "The direction and distance between the two objects.",
        })
        step_idx += 1

        # --- STEP 4: Relative velocity ---
        calc_steps.append({
            "stepIndex": step_idx,
            "phase": "PHASE_09",
            "title": "Calculate Relative Velocity",
            "status": "completed",
            "equation": "v_rel = v₁ − v₂",
            "substitutions": {
                "v_rel_km_s": (v_rel_0 / 1e3).tolist(),
            },
            "result": f"|v_rel| = {float(np.linalg.norm(v_rel_0))/1e3:.3f} km/s",
            "units": "km/s",
            "explanation": "How fast the objects are moving relative to each other.",
            "beginnerExplanation": "The closing speed between the two objects.",
        })
        step_idx += 1

        # --- STEP 5: Initial separation ---
        calc_steps.append({
            "stepIndex": step_idx,
            "phase": "PHASE_09",
            "title": "Calculate Separation",
            "status": "completed",
            "equation": "d = |r_rel|",
            "substitutions": {"d_km": d_0 / 1e3},
            "result": f"d = {d_0/1e3:.3f} km",
            "units": "km",
            "explanation": "Scalar distance between the two objects at the window start.",
            "beginnerExplanation": "How far apart the objects are right now.",
        })
        step_idx += 1

        # --- STEP 6: Coarse screening ---
        candidates = self.screener.screen(pos_fn_a, pos_fn_b, t_start, t_end)
        calc_steps.append({
            "stepIndex": step_idx,
            "phase": "PHASE_09",
            "title": "Coarse Screening",
            "status": "completed",
            "equation": "|r₁(t) − r₂(t)| < threshold",
            "substitutions": {
                "threshold_km": self.threshold / 1e3,
                "window_start_s": t_start,
                "window_end_s": t_end,
                "coarse_dt_s": self.screener.coarse_dt,
            },
            "result": f"{len(candidates)} candidate interval(s) detected",
            "units": "km",
            "explanation": (
                "Coarse pass: sample separation at regular intervals and flag "
                "periods where distance drops below the screening threshold."
            ),
            "beginnerExplanation": (
                "We quickly scan the entire time window to find periods when "
                "the objects come close enough to warrant a closer look."
            ),
        })
        step_idx += 1

        if not candidates:
            calc_steps.append({
                "stepIndex": step_idx,
                "phase": "PHASE_09",
                "title": "No Conjunction Detected",
                "status": "completed",
                "equation": "",
                "result": f"No approaches within {self.threshold/1e3:.0f} km threshold",
                "explanation": "The objects remain well-separated throughout the analysis window.",
                "beginnerExplanation": "The objects don't come close to each other during this time.",
            })

            return ConjunctionResult(
                events=[],
                screening_threshold_m=self.threshold,
                analysis_window=(t_start, t_end),
                candidate_intervals=[],
                model_metadata=self._model_metadata(),
                calculation_steps=calc_steps,
            )

        # --- STEP 7: TCA refinement ---
        all_tcas: list[TCAResult] = []
        for ci in candidates:
            tcas = find_all_tca(
                pos_fn_a, vel_fn_a, pos_fn_b, vel_fn_b,
                ci.t_start, ci.t_end,
                n_samples=200,
                tol=self.tca_tol,
            )
            all_tcas.extend(tcas)

        tca_step = {
            "stepIndex": step_idx,
            "phase": "PHASE_09",
            "title": "TCA Condition",
            "status": "completed",
            "equation": "(r₁ − r₂) · (v₁ − v₂) = 0",
            "result": f"{len(all_tcas)} TCA(s) found",
            "units": "s",
            "explanation": (
                "At the time of closest approach, the inner product of the "
                "relative position and relative velocity vectors equals zero. "
                "This is because the distance is stationary (neither increasing "
                "nor decreasing) at the minimum."
            ),
            "beginnerExplanation": (
                "We find the exact moment when the objects are closest by "
                "solving for when they stop getting closer and start moving apart."
            ),
        }
        if all_tcas:
            tca_step["iterations"] = [
                {"tca_s": r.tca, "converged": r.converged, "iterations": r.iterations}
                for r in all_tcas
            ]
        calc_steps.append(tca_step)
        step_idx += 1

        # --- Process each TCA ---
        conjunction_events: list[ConjunctionEvent] = []

        for tca in all_tcas:
            if not tca.validated:
                continue

            # Miss distance
            calc_steps.append({
                "stepIndex": step_idx,
                "phase": "PHASE_09",
                "title": "Miss Distance",
                "status": "completed",
                "equation": "d(TCA) = |r₁(TCA) − r₂(TCA)|",
                "substitutions": {
                    "tca_s": tca.tca,
                    "r_rel_km": (tca.r_rel / 1e3).tolist(),
                },
                "result": f"d(TCA) = {tca.miss_distance/1e3:.6f} km",
                "units": "km",
                "explanation": "The closest distance between the two objects.",
                "beginnerExplanation": "How close the objects get at their nearest point.",
            })
            step_idx += 1

            # Relative velocity at TCA
            calc_steps.append({
                "stepIndex": step_idx,
                "phase": "PHASE_09",
                "title": "Relative Velocity at TCA",
                "status": "completed",
                "equation": "|v_rel(TCA)| = |v₁(TCA) − v₂(TCA)|",
                "substitutions": {
                    "v_rel_km_s": (tca.v_rel / 1e3).tolist(),
                },
                "result": f"|v_rel(TCA)| = {tca.relative_velocity/1e3:.3f} km/s",
                "units": "km/s",
                "explanation": "How fast the objects are moving relative to each other at closest approach.",
                "beginnerExplanation": "The speed at which the objects pass each other.",
            })
            step_idx += 1

            # Encounter geometry
            angle_deg, enc_type = classify_encounter(tca.v_a, tca.v_b)
            calc_steps.append({
                "stepIndex": step_idx,
                "phase": "PHASE_09",
                "title": "Encounter Geometry",
                "status": "completed",
                "equation": "θ = arccos(v̂₁ · v̂₂)",
                "substitutions": {
                    "angle_deg": angle_deg,
                    "classification": enc_type,
                },
                "result": f"θ = {angle_deg:.1f}° — {enc_type}",
                "units": "degrees",
                "explanation": (
                    f"Angle between velocity vectors: {angle_deg:.1f}°. "
                    f"Classification: {enc_type}. "
                    f"Head-on (>150°): objects approach from opposite directions. "
                    f"Overtaking (<30°): objects move in nearly the same direction. "
                    f"Crossing (30°–150°): objects cross at an angle."
                ),
                "beginnerExplanation": (
                    f"The objects {'are heading straight toward each other' if enc_type == 'head-on' else 'are moving in similar directions' if enc_type == 'overtaking' else 'cross paths at an angle'}."
                ),
            })
            step_idx += 1

            # B-plane (if applicable)
            b_plane = compute_b_plane(tca.r_rel, tca.v_rel)
            if b_plane.applicable:
                calc_steps.append({
                    "stepIndex": step_idx,
                    "phase": "PHASE_09",
                    "title": "B-Plane Analysis",
                    "status": "completed",
                    "equation": "B = r_rel − (r_rel · Ŝ)Ŝ\nB·T = B · T̂\nB·R = B · R̂",
                    "substitutions": {
                        "b_magnitude_km": b_plane.b_magnitude / 1e3,
                        "b_dot_t_km": b_plane.b_dot_t / 1e3,
                        "b_dot_r_km": b_plane.b_dot_r / 1e3,
                    },
                    "result": f"|B| = {b_plane.b_magnitude/1e3:.3f} km, B·T = {b_plane.b_dot_t/1e3:.3f} km, B·R = {b_plane.b_dot_r/1e3:.3f} km",
                    "units": "km",
                    "explanation": (
                        "B-plane analysis decomposes the miss vector into components "
                        "transverse (T) and radial (R) to the approach direction."
                    ),
                    "beginnerExplanation": (
                        "The B-plane shows exactly how the objects miss each other: "
                        "sideways (T) and up/down (R) relative to their approach path."
                    ),
                    "modelName": "Kizner B-plane (1961)",
                    "assumptions": b_plane.assumptions,
                })
            else:
                calc_steps.append({
                    "stepIndex": step_idx,
                    "phase": "PHASE_09",
                    "title": "B-Plane Analysis",
                    "status": "completed",
                    "equation": "",
                    "result": "B-PLANE ANALYSIS NOT APPLICABLE TO THIS ENCOUNTER",
                    "explanation": b_plane.reason,
                    "beginnerExplanation": (
                        "The B-plane concept only works for high-speed fly-by encounters. "
                        "These objects are moving too slowly relative to each other."
                    ),
                })
            step_idx += 1

            conjunction_events.append(ConjunctionEvent(
                tca_result=tca,
                encounter_angle_deg=angle_deg,
                encounter_type=enc_type,
                b_plane=b_plane,
            ))

        return ConjunctionResult(
            events=conjunction_events,
            screening_threshold_m=self.threshold,
            analysis_window=(t_start, t_end),
            candidate_intervals=candidates,
            model_metadata=self._model_metadata(),
            calculation_steps=calc_steps,
        )

    def _model_metadata(self) -> dict[str, Any]:
        return {
            "numerical": {
                "tca_solver": "Brent's method",
                "tca_tolerance_s": self.tca_tol,
                "screening_coarse_dt_s": self.screener.coarse_dt,
                "tca_validation": "derivative sign-change verification",
            },
            "physical": {
                "analysis_type": "Deterministic conjunction analysis",
                "state_source": "Propagated/interpolated state histories",
                "screening_method": "Relative-position distance threshold",
                "tca_condition": "(r₁−r₂)·(v₁−v₂) = 0",
            },
            "limitations": [
                "No covariance / probability of collision",
                "Deterministic trajectories only",
                "No manoeuvre planning or avoidance",
                "B-plane analysis only for high-velocity encounters",
            ],
            "assumptions": [
                "Objects follow their predicted trajectories exactly",
                "No trajectory uncertainties considered",
                "Single-body gravity during screening (if using Keplerian propagation)",
            ],
        }
