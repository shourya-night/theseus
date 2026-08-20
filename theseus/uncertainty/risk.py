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

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class RiskLevel(str, Enum):
    """Conjunction risk classification levels."""
    LOW = "LOW"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


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
        Risk classification (LOW, ELEVATED, HIGH, CRITICAL).
    probability : float
        Evaluated probability of collision Pc.
    thresholds : RiskThresholds
        Threshold configuration applied.
    action_required : bool
        Whether risk level warrants operational collision avoidance action.
    recommendation : str
        Operational guidance note.
    """
    level: RiskLevel
    probability: float
    thresholds: RiskThresholds
    action_required: bool
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "probability": float(self.probability),
            "action_required": self.action_required,
            "recommendation": self.recommendation,
            "thresholds": self.thresholds.to_dict(),
        }


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
    """
    pc = max(0.0, min(1.0, float(probability)))
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
