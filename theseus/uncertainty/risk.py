"""
Risk classification and decision layer for conjunction assessment.

Provides deterministic risk categorization based on configurable
Probability of Collision (Pc) thresholds.

DISCLAIMER:
Universal operational collision probability thresholds do not exist.
Operational thresholds vary by space agency, satellite operator, asset value,
and maneuver capability (e.g. NASA CARA, ESA Space Debris Office, CNES, JAXA).
Thresholds in THESEUS are fully configurable and returned with all results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class RiskLevel(str, Enum):
    """
    Conjunction risk classification levels.

    INDETERMINATE is not a point on the risk scale.  It marks an analysis
    that could not be completed -- typically because no valid time of
    closest approach was found -- and therefore carries no probability and
    never requires action.  It exists so that "we could not evaluate this"
    can never be silently rendered as "we evaluated this and it is LOW".
    """
    LOW = "LOW"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    INDETERMINATE = "INDETERMINATE"


@dataclass
class RiskThresholds:
    """
    Configurable risk classification thresholds.

    Default values represent a common reference baseline:
    - Pc < 1e-7 : LOW
    - 1e-7 <= Pc < 1e-5 : ELEVATED
    - 1e-5 <= Pc < 1e-4 : HIGH
    - Pc >= 1e-4 : CRITICAL

    Attributes
    ----------
    low_threshold : float
        Threshold separating LOW from ELEVATED risk.
    elevated_threshold : float
        Threshold separating ELEVATED from HIGH risk.
    high_threshold : float
        Threshold separating HIGH from CRITICAL risk (maneuver planning threshold).
    name : str
        Configuration profile name.
    """
    low_threshold: float = 1e-7
    elevated_threshold: float = 1e-5
    high_threshold: float = 1e-4
    name: str = "Standard Reference Profile"

    def __post_init__(self) -> None:
        if not (0.0 <= self.low_threshold <= self.elevated_threshold <= self.high_threshold <= 1.0):
            raise ValueError(
                f"Invalid risk thresholds: must satisfy 0 <= low ({self.low_threshold}) <= "
                f"elevated ({self.elevated_threshold}) <= high ({self.high_threshold}) <= 1"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "low_threshold": float(self.low_threshold),
            "elevated_threshold": float(self.elevated_threshold),
            "high_threshold": float(self.high_threshold),
            "disclaimer": "Thresholds are configurable policy parameters, not physical constants.",
        }


# Reference threshold profiles
PROFILE_CONSERVATIVE = RiskThresholds(
    low_threshold=1e-8,
    elevated_threshold=1e-6,
    high_threshold=1e-5,
    name="Conservative Manned Mission Profile (e.g. ISS)",
)

PROFILE_STANDARD = RiskThresholds(
    low_threshold=1e-7,
    elevated_threshold=1e-5,
    high_threshold=1e-4,
    name="Standard Robotic LEO Satellite Profile (e.g. NASA CARA / ESA SDC)",
)

PROFILE_PERMISSIVE = RiskThresholds(
    low_threshold=1e-6,
    elevated_threshold=1e-4,
    high_threshold=1e-3,
    name="Permissive / Constellation Operations Profile",
)


@dataclass
class RiskAssessment:
    """
    Structured risk evaluation output.

    Attributes
    ----------
    level : RiskLevel
        Risk classification (LOW, ELEVATED, HIGH, CRITICAL, INDETERMINATE).
    probability : float | None
        Evaluated probability of collision Pc.  None when the analysis was
        indeterminate and no Pc was computed.
    thresholds : RiskThresholds
        Threshold configuration applied.
    action_required : bool
        Whether risk level warrants operational collision avoidance action.
        Always False for an INDETERMINATE assessment.
    recommendation : str
        Operational guidance note.
    reason : str
        For INDETERMINATE assessments, why the analysis could not be completed.
    """
    level: RiskLevel
    probability: Optional[float]
    thresholds: RiskThresholds
    action_required: bool
    recommendation: str
    reason: str = ""

    @property
    def is_determinate(self) -> bool:
        """True when a probability was actually evaluated and classified."""
        return self.level is not RiskLevel.INDETERMINATE

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "level": self.level.value,
            "probability": None if self.probability is None else float(self.probability),
            "action_required": bool(self.action_required),
            "recommendation": self.recommendation,
            "thresholds": self.thresholds.to_dict(),
            "determinate": self.is_determinate,
        }
        if self.reason:
            d["reason"] = self.reason
        return d


def classify_risk(
    probability: float,
    thresholds: Optional[RiskThresholds] = None,
) -> RiskAssessment:
    """
    Classify conjunction risk based on probability and configurable thresholds.

    Parameters
    ----------
    probability : float
        Calculated Pc.
    thresholds : RiskThresholds, optional
        Custom thresholds. If None, default standard profile is used.

    Returns
    -------
    RiskAssessment

    Raises
    ------
    ValueError
        If *probability* is None or non-finite.  A risk level must never be
        derived from a Pc that was not actually computed; callers that have
        no Pc must use :func:`indeterminate_risk` instead.
    """
    if probability is None:
        raise ValueError(
            "RISK CLASSIFICATION REFUSED: no collision probability was supplied. "
            "An analysis without a valid Pc must be reported via indeterminate_risk()."
        )
    pc_raw = float(probability)
    if not math.isfinite(pc_raw):
        raise ValueError(
            f"RISK CLASSIFICATION REFUSED: collision probability is not finite ({pc_raw})."
        )
    pc = max(0.0, min(1.0, pc_raw))
    th = thresholds or PROFILE_STANDARD

    if pc < th.low_threshold:
        level = RiskLevel.LOW
        action = False
        rec = "Routine monitoring. Collision probability is well below operational screening threshold."
    elif pc < th.elevated_threshold:
        level = RiskLevel.ELEVATED
        action = False
        rec = "Heightened monitoring recommended. Track orbit determination updates and covariance refinement."
    elif pc < th.high_threshold:
        level = RiskLevel.HIGH
        action = True
        rec = "High risk detected. Prepare collision avoidance maneuver (CAM) planning options."
    else:
        level = RiskLevel.CRITICAL
        action = True
        rec = "Critical risk. Execute collision avoidance maneuver unless updated tracking refines Pc below threshold."

    return RiskAssessment(
        level=level,
        probability=pc,
        thresholds=th,
        action_required=action,
        recommendation=rec,
    )


def indeterminate_risk(
    reason: str,
    thresholds: Optional[RiskThresholds] = None,
) -> RiskAssessment:
    """
    Build the risk assessment for an analysis that could not be completed.

    Use this whenever there is no valid probability of collision to classify --
    for example when no time of closest approach was found inside the analysis
    window.  The returned assessment carries no probability, is never
    actionable, and cannot be confused with a LOW-risk result.

    Parameters
    ----------
    reason : str
        Why the analysis is indeterminate.  Surfaced to the caller verbatim.
    thresholds : RiskThresholds, optional
        Threshold configuration, echoed back for traceability only.

    Returns
    -------
    RiskAssessment
    """
    return RiskAssessment(
        level=RiskLevel.INDETERMINATE,
        probability=None,
        thresholds=thresholds or PROFILE_STANDARD,
        action_required=False,
        recommendation=(
            "ANALYSIS INDETERMINATE — no collision probability was computed. "
            "This is not a low-risk finding: it means the encounter could not be "
            "evaluated. Re-run with screening parameters matched to the encounter "
            "geometry before drawing any conclusion."
        ),
        reason=reason,
    )
