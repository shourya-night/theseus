"""
Simulation event system.

Every simulation produces a chronological stream of structured events
(INITIALIZATION, BURN_START, ORBIT_CHANGE, …).  These events are
pure data — no UI dependency — and are designed so a future frontend
can synchronise visualisation to the exact computation timeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    """Enumeration of simulation event types."""
    INITIALIZATION = auto()
    STATE_UPDATE = auto()
    BURN_START = auto()
    BURN_END = auto()
    ORBIT_CHANGE = auto()
    CLOSE_APPROACH = auto()
    ATMOSPHERIC_ENTRY = auto()
    PEAK_HEATING = auto()
    CORRECTION = auto()
    LANDING = auto()
    MISSION_SUCCESS = auto()
    MISSION_FAILURE = auto()
    WARNING = auto()
    SOI_TRANSITION = auto()


@dataclass
class SimulationEvent:
    """
    A single simulation event.

    Attributes
    ----------
    time : float
        Simulation time (seconds from epoch).
    event_type : EventType
        Category of the event.
    description : str
        Human-readable description.
    parameters : dict[str, Any]
        Event-specific data (e.g. Δv for a burn event).
    state : dict[str, Any] | None
        Snapshot of the simulation state at event time, if available.
    """
    time: float
    event_type: EventType
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "time": self.time,
            "event_type": self.event_type.name,
            "description": self.description,
            "parameters": self.parameters,
            "state": self.state,
        }


class EventLog:
    """Chronological collection of simulation events."""

    def __init__(self) -> None:
        self._events: list[SimulationEvent] = []

    def log(self, event: SimulationEvent) -> None:
        """Append an event to the log."""
        self._events.append(event)

    def emit(
        self,
        time: float,
        event_type: EventType,
        description: str = "",
        **parameters: Any,
    ) -> SimulationEvent:
        """Create, log, and return a new event."""
        evt = SimulationEvent(
            time=time,
            event_type=event_type,
            description=description,
            parameters=parameters,
        )
        self._events.append(evt)
        return evt

    @property
    def events(self) -> list[SimulationEvent]:
        """Return an ordered copy of the event list."""
        return list(self._events)

    def filter(self, event_type: EventType) -> list[SimulationEvent]:
        """Return events matching *event_type*."""
        return [e for e in self._events if e.event_type == event_type]

    def to_list(self) -> list[dict[str, Any]]:
        """Serialise all events to a list of dicts."""
        return [e.to_dict() for e in self._events]

    def __len__(self) -> int:
        return len(self._events)
