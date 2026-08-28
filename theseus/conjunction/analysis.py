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
from theseus.conjunction.tca import (
    find_tca, find_all_tca, find_all_tca_with_diagnostics, TCAResult,
)
from theseus.conjunction.b_plane import compute_b_plane, BPlaneResult
from theseus.conjunction.geometry import (
    CollisionAssessment,
    CollisionGeometry,
    CollisionStatus,
    assess_collision_geometry,
)
from theseus.conjunction.state_validation import (
    NonFiniteStateError,
    guard_position_functions,
    guard_velocity_functions,
)


@dataclass
class ConjunctionEvent:
    """
    A single conjunction event (close approach).

    A conjunction is a close approach.  Whether the bodies actually touched is
    a separate, geometric question answered by :attr:`collision`; whether they
    are *likely* to touch given uncertainty is a Phase 10 question that this
    object deliberately does not address.

    Attributes
    ----------
    tca_result : TCAResult
        Full TCA solution: time, relative position and velocity, miss distance.
    encounter_angle_deg : float
        Angle between velocity vectors at TCA (0° = same direction,
        180° = head-on).
    encounter_type : str
        'head-on', 'overtaking', or 'crossing'.
    b_plane : BPlaneResult | None
        B-plane analysis (if applicable).
    collision : CollisionAssessment | None
        Deterministic collision geometry evaluated at the validated TCA.
        None when no body geometry was supplied to the analysis -- which
        means the question was not asked, not that the bodies passed clear.
    object_a_id, object_b_id : str | None
        Identifiers of the two objects, when the caller supplied them.
    """
    tca_result: TCAResult
    encounter_angle_deg: float
    encounter_type: str
    b_plane: Optional[BPlaneResult] = None
    collision: Optional[CollisionAssessment] = None
    object_a_id: Optional[str] = None
    object_b_id: Optional[str] = None

    @property
    def miss_distance_m(self) -> float:
        """Centre-to-centre separation at TCA (m)."""
        return float(self.tca_result.miss_distance)

    @property
    def clearance_m(self) -> Optional[float]:
        """
        Surface-to-surface clearance at TCA (m), or None when geometry is
        unknown.  Negative means the bodies interpenetrated.
        """
        if self.collision is None or not self.collision.is_evaluated:
            return None
        return float(self.collision.clearance_m)

    @property
    def is_physical_intersection(self) -> Optional[bool]:
        """
        True/False when geometry was supplied, None when it was not.

        Returning None rather than False for the unevaluated case keeps a
        missing-geometry analysis from reading as a clean pass.
        """
        if self.collision is None or not self.collision.is_evaluated:
            return None
        return self.collision.is_physical_intersection

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_a_id": self.object_a_id,
            "object_b_id": self.object_b_id,
            "tca": self.tca_result.to_dict(),
            "encounter_angle_deg": self.encounter_angle_deg,
            "encounter_type": self.encounter_type,
            "b_plane": self.b_plane.to_dict() if self.b_plane is not None else None,
            "collision": self.collision.to_dict() if self.collision is not None else None,
        }


@dataclass(frozen=True)
class ConjunctionAccounting:
    """
    Truthful accounting of one conjunction analysis.

    Every field is taken from execution state, never inferred from another
    field, and the vocabulary is kept distinct throughout:

    ``intervals_screened``
        Coarse intervals examined by the screen.
    ``intervals_accepted``
        Intervals the screen could not prove clear of the threshold.
    ``candidate_intervals``
        Merged spans handed to TCA refinement.  Consecutive accepted intervals
        merge, so this is at most ``intervals_accepted`` -- a candidate
        interval is *not* a TCA.
    ``tca_attempts``
        Refinements attempted, one per sign change of r_rel · v_rel found
        inside a candidate span.  A single span may contain several, so this
        can exceed ``candidate_intervals``.
    ``tca_converged``
        Refinements whose solver converged.  A converged solve is not yet a
        usable TCA.
    ``tca_validated``
        Converged solutions that passed the local-minimum validation.
    ``tca_rejected_not_converged`` / ``tca_rejected_not_validated``
        Why the rest were dropped.
    ``accepted_conjunctions``
        Events actually present in :attr:`ConjunctionResult.events`.
    ``accepted_beyond_screening_threshold``
        Accepted events whose miss distance exceeds the screening threshold.
        Not a rejection -- the screen is deliberately conservative and admits
        intervals it cannot prove clear -- but worth surfacing to a reader.
    ``rejection_reasons``
        Validation notes from every dropped solution, verbatim.

    Invariants, which hold for this architecture and are asserted in the
    Phase 9 accounting tests:

        intervals_accepted     <= intervals_screened
        candidate_intervals    <= intervals_accepted
        tca_converged          <= tca_attempts
        tca_validated          <= tca_converged
        accepted_conjunctions  == tca_validated
        tca_attempts is NOT bounded by candidate_intervals
    """
    intervals_screened: int = 0
    intervals_accepted: int = 0
    candidate_intervals: int = 0
    tca_attempts: int = 0
    tca_converged: int = 0
    tca_validated: int = 0
    tca_rejected_not_converged: int = 0
    tca_rejected_not_validated: int = 0
    accepted_conjunctions: int = 0
    accepted_beyond_screening_threshold: int = 0
    rejection_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "intervals_screened": int(self.intervals_screened),
            "intervals_accepted": int(self.intervals_accepted),
            "candidate_intervals": int(self.candidate_intervals),
            "tca_attempts": int(self.tca_attempts),
            "tca_converged": int(self.tca_converged),
            "tca_validated": int(self.tca_validated),
            "tca_rejected_not_converged": int(self.tca_rejected_not_converged),
            "tca_rejected_not_validated": int(self.tca_rejected_not_validated),
            "accepted_conjunctions": int(self.accepted_conjunctions),
            "accepted_beyond_screening_threshold": int(
                self.accepted_beyond_screening_threshold
            ),
            "rejection_reasons": list(self.rejection_reasons),
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
    accounting: ConjunctionAccounting = field(default_factory=ConjunctionAccounting)

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
            "accounting": self.accounting.to_dict(),
            "summary": {
                "total_events": len(self.events),
                "closest_approach_km": (
                    min(e.tca_result.miss_distance for e in self.events) / 1e3
                    if self.events else None
                ),
                "smallest_clearance_m": (
                    min(c for c in (e.clearance_m for e in self.events) if c is not None)
                    if any(e.clearance_m is not None for e in self.events) else None
                ),
                "physical_intersections": sum(
                    1 for e in self.events if e.is_physical_intersection is True
                ),
                "collision_geometry_evaluated": all(
                    e.is_physical_intersection is not None for e in self.events
                ) if self.events else False,
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
        *,
        geometry_a: Optional[CollisionGeometry] = None,
        geometry_b: Optional[CollisionGeometry] = None,
        grazing_tolerance_m: float = 0.0,
        object_a_id: Optional[str] = None,
        object_b_id: Optional[str] = None,
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
        geometry_a, geometry_b : CollisionGeometry, optional
            Body geometry.  When both are supplied, every event carries a
            deterministic collision assessment evaluated at its validated TCA.
            When either is missing the assessment is UNKNOWN -- the question
            is reported as unevaluated, never as "no intersection".
        grazing_tolerance_m : float
            Clearance band around zero reported as grazing contact (m).
        object_a_id, object_b_id : str, optional
            Identifiers carried through onto each event.

        Returns
        -------
        ConjunctionResult

        Raises
        ------
        NonFiniteStateError
            If any state the pipeline evaluates is not finite.  This is the
            boundary at which trajectory functions enter Phase 9, so it is
            where they are checked.  A non-finite state can never produce a
            ``ConjunctionResult`` -- an empty event list from this method
            always means "the finite trajectories supplied contained no
            validated closest approach", never "the input was unusable".
        """
        # Guard the four trajectory functions before anything consumes them.
        # Every later stage -- screening, the TCA search, the states written
        # into the trace and into each event -- reads through these wrappers,
        # so no unvalidated state can enter the pipeline behind them.  The
        # guard is idempotent, so the screener and TCA solver re-guarding the
        # same functions costs nothing.
        pos_fn_a, pos_fn_b = guard_position_functions(
            pos_fn_a, pos_fn_b,
            object_a_id=object_a_id or "A", object_b_id=object_b_id or "B",
        )
        vel_fn_a, vel_fn_b = guard_velocity_functions(
            vel_fn_a, vel_fn_b,
            object_a_id=object_a_id or "A", object_b_id=object_b_id or "B",
        )

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
        candidates, screen_diag = self.screener.screen_with_diagnostics(
            pos_fn_a, pos_fn_b, t_start, t_end,
            vel_fn_a=vel_fn_a, vel_fn_b=vel_fn_b,
            object_a_id=object_a_id or "A", object_b_id=object_b_id or "B",
        )
        calc_steps.append({
            "stepIndex": step_idx,
            "phase": "PHASE_09",
            "title": "Coarse Screening",
            "status": "completed",
            "equation": "min |r_rel| ≥ (d₀ + d₁ − V h) / 2  <  threshold ?",
            "substitutions": {
                "threshold_km": self.threshold / 1e3,
                "window_start_s": t_start,
                "window_end_s": t_end,
                "coarse_dt_s": screen_diag.coarse_dt_s,
                "samples": screen_diag.samples,
                "intervals_screened": screen_diag.intervals_screened,
                "intervals_accepted": screen_diag.intervals_accepted,
                "intervals_rejected": screen_diag.intervals_rejected,
                "candidate_intervals": screen_diag.candidate_intervals,
                "min_sampled_distance_km": screen_diag.min_sampled_distance_m / 1e3,
                "min_lower_bound_km": screen_diag.min_lower_bound_m / 1e3,
            },
            "result": (
                f"{screen_diag.intervals_screened} interval(s) screened, "
                f"{screen_diag.intervals_accepted} could not be proved clear, "
                f"merged into {screen_diag.candidate_intervals} candidate interval(s)"
            ),
            "units": "km",
            "explanation": (
                "Coarse pass: for each interval, compute an analytic lower bound on "
                "the separation across the whole interval and discard it only if that "
                "bound stays above the threshold. A candidate interval is a period that "
                "could not be ruled out — it is not itself a close approach."
            ),
            "beginnerExplanation": (
                "We scan the whole time window and rule out the stretches where the "
                "objects definitely stayed far apart. What is left gets a closer look."
            ),
        })
        step_idx += 1

        if not candidates:
            calc_steps.append({
                "stepIndex": step_idx,
                "phase": "PHASE_09",
                "title": "No Candidate Intervals — Analysis Complete",
                "status": "completed",
                "equation": "",
                "substitutions": {
                    "intervals_screened": screen_diag.intervals_screened,
                    "intervals_rejected": screen_diag.intervals_rejected,
                    "candidate_intervals": 0,
                    "tca_attempts": 0,
                    "accepted_conjunctions": 0,
                },
                "result": (
                    f"All {screen_diag.intervals_screened} interval(s) proved clear of the "
                    f"{self.threshold/1e3:.0f} km threshold; no TCA refinement was attempted"
                ),
                "explanation": (
                    "The separation bound stayed above the screening threshold across every "
                    "interval, so no period could contain a sub-threshold approach and no "
                    "refinement was needed."
                ),
                "beginnerExplanation": "The objects never came close during this time.",
            })

            return ConjunctionResult(
                events=[],
                screening_threshold_m=self.threshold,
                analysis_window=(t_start, t_end),
                candidate_intervals=[],
                model_metadata=self._model_metadata(),
                calculation_steps=calc_steps,
                accounting=ConjunctionAccounting(
                    intervals_screened=screen_diag.intervals_screened,
                    intervals_accepted=screen_diag.intervals_accepted,
                    candidate_intervals=0,
                ),
            )

        # --- STEP 7: TCA refinement ---
        # Counts come from the search itself.  An attempt is counted when the
        # refinement is made, and a success only once the solver reports it.
        all_tcas: list[TCAResult] = []
        tca_attempts = 0
        tca_not_converged = 0
        for ci in candidates:
            tcas, tca_diag = find_all_tca_with_diagnostics(
                pos_fn_a, vel_fn_a, pos_fn_b, vel_fn_b,
                ci.t_start, ci.t_end,
                n_samples=200,
                tol=self.tca_tol,
                object_a_id=object_a_id or "A", object_b_id=object_b_id or "B",
            )
            all_tcas.extend(tcas)
            tca_attempts += tca_diag.attempts
            tca_not_converged += tca_diag.non_converged

        validated_tcas = [t for t in all_tcas if t.validated]
        rejected_tcas = [t for t in all_tcas if not t.validated]
        rejection_reasons = tuple(
            t.validation_note for t in rejected_tcas if t.validation_note
        )

        tca_step = {
            "stepIndex": step_idx,
            "phase": "PHASE_09",
            "title": "TCA Refinement",
            "status": "completed" if validated_tcas else "no_solution",
            "equation": "(r₁ − r₂) · (v₁ − v₂) = 0",
            "substitutions": {
                "candidate_intervals": len(candidates),
                "tca_attempts": tca_attempts,
                "tca_converged": len(all_tcas),
                "tca_not_converged": tca_not_converged,
                "tca_validated": len(validated_tcas),
                "tca_rejected_not_validated": len(rejected_tcas),
            },
            "result": (
                f"{tca_attempts} refinement attempt(s) across {len(candidates)} candidate "
                f"interval(s) → {len(all_tcas)} converged → {len(validated_tcas)} validated "
                f"as true minima"
            ),
            "units": "s",
            "explanation": (
                "Each sign change of r_rel · v_rel inside a candidate interval brackets one "
                "closest approach, and Brent's method refines it. A candidate interval may "
                "contain more than one such bracket, so attempts are not bounded by the "
                "number of candidate intervals. A converged solve is only accepted once it "
                "is verified to be a local minimum rather than a maximum."
            ),
            "beginnerExplanation": (
                "For each stretch worth examining, we solve for the exact moment the objects "
                "stop approaching and start receding, then check that moment really is the "
                "closest one."
            ),
        }
        if all_tcas:
            tca_step["iterations"] = [
                {
                    "tca_s": r.tca,
                    "converged": r.converged,
                    "iterations": r.iterations,
                    "validated": r.validated,
                    "validation_note": r.validation_note,
                }
                for r in all_tcas
            ]
        calc_steps.append(tca_step)
        step_idx += 1

        if rejected_tcas:
            calc_steps.append({
                "stepIndex": step_idx,
                "phase": "PHASE_09",
                "title": "Rejected TCA Solutions",
                "status": "completed",
                "equation": "",
                "substitutions": {
                    "rejected_count": len(rejected_tcas),
                    "rejected": [
                        {"tca_s": r.tca, "reason": r.validation_note}
                        for r in rejected_tcas
                    ],
                },
                "result": (
                    f"{len(rejected_tcas)} converged solution(s) discarded before reporting"
                ),
                "explanation": (
                    "These solves converged but failed validation — typically a local maximum "
                    "(objects receding then approaching) or a root outside the analysis "
                    "window. They are excluded from the reported conjunctions."
                ),
                "beginnerExplanation": (
                    "Some answers the solver found turned out to be the moment the objects "
                    "were furthest apart, not closest, so we threw them away."
                ),
            })
            step_idx += 1

        # --- Process each validated TCA ---
        conjunction_events: list[ConjunctionEvent] = []

        for tca in validated_tcas:

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

            # Deterministic collision geometry, evaluated at the validated TCA
            # miss distance -- never at a screening sample or a display time.
            collision = assess_collision_geometry(
                miss_distance_m=tca.miss_distance,
                geom_a=geometry_a,
                geom_b=geometry_b,
                grazing_tolerance_m=grazing_tolerance_m,
            )

            if collision.is_evaluated:
                calc_steps.append({
                    "stepIndex": step_idx,
                    "phase": "PHASE_09",
                    "title": "Collision Geometry",
                    "status": "completed",
                    "equation": "clearance = d(TCA) − (R_A + R_B)",
                    "substitutions": {
                        "miss_distance_m": collision.miss_distance_m,
                        "R_A_m": geometry_a.collision_radius_m,
                        "R_B_m": geometry_b.collision_radius_m,
                        "combined_hard_body_radius_m": collision.combined_hard_body_radius_m,
                        "clearance_m": collision.clearance_m,
                    },
                    "result": (
                        f"clearance = {collision.clearance_m:.3f} m — {collision.status.value}"
                    ),
                    "units": "m",
                    "explanation": (
                        "Deterministic test of whether the nominal bodies touched. "
                        "Negative clearance means the hard-body spheres overlap. "
                        "This is geometry only; the probability of collision under "
                        "trajectory uncertainty is a separate Phase 10 calculation."
                    ),
                    "beginnerExplanation": (
                        "We subtract the two spacecraft sizes from the gap between "
                        "their centres. If what's left is negative, they hit."
                    ),
                    "assumptions": [
                        "Spherical hard-body approximation for both objects",
                        "Evaluated on the nominal trajectories, with no uncertainty",
                    ],
                })
                step_idx += 1

            conjunction_events.append(ConjunctionEvent(
                tca_result=tca,
                encounter_angle_deg=angle_deg,
                encounter_type=enc_type,
                b_plane=b_plane,
                collision=collision,
                object_a_id=object_a_id,
                object_b_id=object_b_id,
            ))

        beyond_threshold = sum(
            1 for e in conjunction_events
            if e.tca_result.miss_distance > self.threshold
        )

        # Closing step.  Every analysis ends with an explicit statement of what
        # survived, including the case where candidates existed but nothing was
        # accepted -- previously the trace simply stopped after claiming that
        # TCAs had been "found".
        calc_steps.append({
            "stepIndex": step_idx,
            "phase": "PHASE_09",
            "title": (
                "Conjunctions Reported" if conjunction_events
                else "No Conjunction Accepted"
            ),
            "status": "completed",
            "equation": "",
            "substitutions": {
                "intervals_screened": screen_diag.intervals_screened,
                "intervals_accepted": screen_diag.intervals_accepted,
                "candidate_intervals": len(candidates),
                "tca_attempts": tca_attempts,
                "tca_converged": len(all_tcas),
                "tca_validated": len(validated_tcas),
                "accepted_conjunctions": len(conjunction_events),
                "accepted_beyond_screening_threshold": beyond_threshold,
            },
            "result": (
                f"{len(conjunction_events)} conjunction event(s) reported"
                if conjunction_events else
                f"0 conjunction events reported — {tca_attempts} refinement attempt(s) "
                f"produced no validated closest approach"
            ),
            "explanation": (
                "Final accounting. Each stage narrows the previous one: intervals screened "
                "→ candidate intervals → refinement attempts → converged solutions → "
                "validated minima → reported conjunctions. The counts in this step are the "
                "contents of the returned result, not an estimate."
                + (
                    f" {beyond_threshold} reported event(s) have a miss distance above the "
                    f"{self.threshold/1e3:.0f} km screening threshold: the screen admits "
                    f"intervals it cannot prove clear, so this is expected and is not a "
                    f"rejection criterion."
                    if beyond_threshold else ""
                )
            ),
            "beginnerExplanation": (
                f"In the end we are reporting {len(conjunction_events)} close approach(es)."
                if conjunction_events else
                "In the end, none of the periods we examined contained a genuine closest "
                "approach, so nothing is reported."
            ),
        })

        return ConjunctionResult(
            events=conjunction_events,
            screening_threshold_m=self.threshold,
            analysis_window=(t_start, t_end),
            candidate_intervals=candidates,
            model_metadata=self._model_metadata(),
            calculation_steps=calc_steps,
            accounting=ConjunctionAccounting(
                intervals_screened=screen_diag.intervals_screened,
                intervals_accepted=screen_diag.intervals_accepted,
                candidate_intervals=len(candidates),
                tca_attempts=tca_attempts,
                tca_converged=len(all_tcas),
                tca_validated=len(validated_tcas),
                tca_rejected_not_converged=tca_not_converged,
                tca_rejected_not_validated=len(rejected_tcas),
                accepted_conjunctions=len(conjunction_events),
                accepted_beyond_screening_threshold=beyond_threshold,
                rejection_reasons=rejection_reasons,
            ),
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
