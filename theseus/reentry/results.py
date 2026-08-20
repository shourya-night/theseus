"""
Reentry results container.

Stores the complete output of a reentry simulation:
trajectory, atmospheric state, aerodynamic/thermal telemetry,
detected events, summary statistics, and model metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


class ReentryEventType(Enum):
    """Types of reentry events detected during propagation."""
    ENTRY_INTERFACE = auto()
    PEAK_DYNAMIC_PRESSURE = auto()
    PEAK_HEATING = auto()
    PEAK_DECELERATION = auto()
    SUBSONIC_TRANSITION = auto()
    GROUND_IMPACT = auto()
    SKIP_OUT = auto()
    MAX_TIME_EXCEEDED = auto()


@dataclass
class ReentryEvent:
    """
    A detected event during atmospheric entry.

    Attributes
    ----------
    event_type : ReentryEventType
    time : float
        Simulation time (s).
    altitude : float
        Altitude at event (m).
    velocity : float
        Speed at event (m/s).
    value : float
        The relevant physical quantity at the event
        (e.g. peak q for PEAK_DYNAMIC_PRESSURE).
    units : str
        Units of `value`.
    detection_method : str
        How the event was detected (e.g. 'sign change in dq/dt').
    state : dict[str, Any]
        Full state snapshot at event time.
    """
    event_type: ReentryEventType
    time: float
    altitude: float
    velocity: float
    value: float
    units: str
    detection_method: str
    state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.name,
            "time_s": self.time,
            "altitude_m": self.altitude,
            "altitude_km": self.altitude / 1e3,
            "velocity_m_s": self.velocity,
            "velocity_km_s": self.velocity / 1e3,
            "value": self.value,
            "units": self.units,
            "detection_method": self.detection_method,
            "state": self.state,
        }


@dataclass
class ReentryTelemetryPoint:
    """A single point in the reentry telemetry timeseries."""
    time: float              # s
    altitude: float          # m
    velocity: float          # m/s
    flight_path_angle: float # rad
    downrange: float         # m (arc length along surface)
    latitude: float          # rad (range angle)
    density: float           # kg/m³
    pressure: float          # Pa
    temperature: float       # K
    mach: float              # dimensionless
    dynamic_pressure: float  # Pa
    drag: float              # N
    lift: float              # N
    acceleration_mag: float  # m/s²
    g_load: float            # dimensionless (|a|/g₀)
    heating_rate: float      # W/m²
    cumulative_heat: float   # J/m²
    stagnation_temp: float   # K

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": self.time,
            "altitude_m": self.altitude,
            "altitude_km": self.altitude / 1e3,
            "velocity_m_s": self.velocity,
            "velocity_km_s": self.velocity / 1e3,
            "flight_path_angle_deg": self.flight_path_angle * 57.29577951308232,
            "downrange_km": self.downrange / 1e3,
            "density_kg_m3": self.density,
            "pressure_Pa": self.pressure,
            "temperature_K": self.temperature,
            "mach": self.mach,
            "dynamic_pressure_Pa": self.dynamic_pressure,
            "dynamic_pressure_kPa": self.dynamic_pressure / 1e3,
            "drag_N": self.drag,
            "lift_N": self.lift,
            "acceleration_m_s2": self.acceleration_mag,
            "g_load": self.g_load,
            "heating_rate_W_m2": self.heating_rate,
            "heating_rate_kW_m2": self.heating_rate / 1e3,
            "cumulative_heat_J_m2": self.cumulative_heat,
            "cumulative_heat_MJ_m2": self.cumulative_heat / 1e6,
            "stagnation_temp_K": self.stagnation_temp,
        }


@dataclass
class ReentryResult:
    """
    Complete result of a reentry simulation.

    Attributes
    ----------
    telemetry : list[ReentryTelemetryPoint]
        Full trajectory timeseries.
    events : list[ReentryEvent]
        Detected events.
    termination_reason : str
        Why propagation stopped ('ground_impact', 'skip_out', 'max_time').
    vehicle : dict[str, Any]
        Vehicle descriptor.
    model_metadata : dict[str, Any]
        Physical model and numerical method metadata.
    peak_statistics : dict[str, Any]
        Peak values of key quantities.
    impact_conditions : dict[str, Any] | None
        Conditions at ground impact (if applicable).
    calculation_steps : list[dict[str, Any]]
        Structured calculation trace for the frontend overlay.
    """
    telemetry: list[ReentryTelemetryPoint] = field(default_factory=list)
    events: list[ReentryEvent] = field(default_factory=list)
    termination_reason: str = ""
    vehicle: dict[str, Any] = field(default_factory=dict)
    model_metadata: dict[str, Any] = field(default_factory=dict)
    peak_statistics: dict[str, Any] = field(default_factory=dict)
    impact_conditions: Optional[dict[str, Any]] = None
    calculation_steps: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "telemetry": [t.to_dict() for t in self.telemetry],
            "events": [e.to_dict() for e in self.events],
            "termination_reason": self.termination_reason,
            "vehicle": self.vehicle,
            "model_metadata": self.model_metadata,
            "peak_statistics": self.peak_statistics,
            "impact_conditions": self.impact_conditions,
            "calculation_steps": self.calculation_steps,
            "summary": {
                "total_time_s": self.telemetry[-1].time if self.telemetry else 0.0,
                "total_points": len(self.telemetry),
                "events_detected": len(self.events),
                "event_types": [e.event_type.name for e in self.events],
            },
        }
