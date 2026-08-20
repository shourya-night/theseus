"""
Structured calculation-trace system.

Every major algorithm can optionally generate a CalculationTrace that
records: operation name, equation, inputs, intermediate steps, and result.

These traces are data-only — no UI dependency — and will later power
the step-by-step calculation interface in the frontend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CalculationTrace:
    """
    A structured record of a single calculation.

    Attributes
    ----------
    operation : str
        Name of the operation (e.g. 'specific_orbital_energy').
    equation : str
        Human-readable equation string (e.g. 'ε = v²/2 − μ/r').
    inputs : dict[str, Any]
        Named input values.
    steps : list[dict[str, Any]]
        Intermediate computation steps.
    result : Any
        Final result value.
    metadata : dict[str, Any]
        Additional metadata (e.g. convergence info).
    """
    operation: str
    equation: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    result: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(self, description: str, **values: Any) -> None:
        """Append an intermediate step."""
        self.steps.append({"description": description, **values})

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "operation": self.operation,
            "equation": self.equation,
            "inputs": self.inputs,
            "steps": self.steps,
            "result": self.result,
            "metadata": self.metadata,
        }


class TraceContext:
    """
    Context manager that collects :class:`CalculationTrace` objects
    produced during a block of computation.

    Usage::

        ctx = TraceContext()
        with ctx:
            # ... any code that calls ctx.record(trace) ...
        print(ctx.traces)
    """

    _active: Optional[TraceContext] = None

    def __init__(self) -> None:
        self.traces: list[CalculationTrace] = []

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> TraceContext:
        TraceContext._active = self
        return self

    def __exit__(self, *_: Any) -> None:
        TraceContext._active = None

    # -- recording -----------------------------------------------------------

    def record(self, trace: CalculationTrace) -> None:
        """Add a trace to this context."""
        self.traces.append(trace)

    @classmethod
    def current(cls) -> Optional[TraceContext]:
        """Return the currently active TraceContext, or None."""
        return cls._active

    @classmethod
    def emit(cls, trace: CalculationTrace) -> None:
        """Record a trace into the active context, if any.  Otherwise no-op."""
        ctx = cls._active
        if ctx is not None:
            ctx.record(trace)
