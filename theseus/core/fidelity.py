"""
Model fidelity and assumption tracking system.

Every physics model in THESEUS must declare:
    - its fidelity level (simplified / moderate / high)
    - the assumptions it makes
    - its known limitations
    - its valid domain

This enables:
    - users to understand what is being computed
    - automated warnings when a model is used outside its valid domain
    - progressive fidelity upgrades without losing traceability
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FidelityLevel(Enum):
    """Qualitative fidelity tier for a physics model."""
    SIMPLIFIED = "simplified"
    MODERATE = "moderate"
    HIGH = "high"
    REFERENCE = "reference"  # validated against external reference data


@dataclass(frozen=True)
class Assumption:
    """
    A single documented assumption made by a model.

    Attributes
    ----------
    name : str
        Short identifier (e.g. 'spherical_earth').
    description : str
        Human-readable statement of the assumption.
    impact : str
        What happens if this assumption is violated.
    """
    name: str
    description: str
    impact: str = ""


@dataclass
class ModelFidelity:
    """
    Fidelity descriptor attached to a physics model or algorithm.

    Attributes
    ----------
    model_name : str
        Human-readable model name (e.g. 'US Standard Atmosphere 1976').
    level : FidelityLevel
        Qualitative fidelity tier.
    assumptions : list[Assumption]
        Documented assumptions.
    valid_domain : str
        Human-readable domain description (e.g. 'altitude 0–86 km').
    source : str
        Reference / citation for the model.
    limitations : str
        Known limitations.
    """
    model_name: str
    level: FidelityLevel
    assumptions: list[Assumption] = field(default_factory=list)
    valid_domain: str = ""
    source: str = ""
    limitations: str = ""

    def summary(self) -> dict:
        """Return a serialisable summary dictionary."""
        return {
            "model_name": self.model_name,
            "level": self.level.value,
            "assumptions": [
                {"name": a.name, "description": a.description, "impact": a.impact}
                for a in self.assumptions
            ],
            "valid_domain": self.valid_domain,
            "source": self.source,
            "limitations": self.limitations,
        }


class FidelityRegistry:
    """
    Global registry of all active model-fidelity descriptors.

    Models register themselves on construction; the registry can be
    queried to produce a complete fidelity/assumption audit.
    """

    _instance: Optional[FidelityRegistry] = None
    _models: dict[str, ModelFidelity]

    def __init__(self) -> None:
        self._models = {}

    @classmethod
    def get(cls) -> FidelityRegistry:
        """Return the singleton registry, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for testing)."""
        cls._instance = None

    def register(self, fidelity: ModelFidelity) -> None:
        """Register a model fidelity descriptor."""
        self._models[fidelity.model_name] = fidelity

    def get_model(self, name: str) -> Optional[ModelFidelity]:
        """Retrieve a registered model by name."""
        return self._models.get(name)

    def all_models(self) -> dict[str, ModelFidelity]:
        """Return all registered models."""
        return dict(self._models)

    def audit(self) -> list[dict]:
        """Return a serialisable audit of every active model."""
        return [m.summary() for m in self._models.values()]
