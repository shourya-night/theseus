"""
THESEUS Unified Multi-Object Simulation & Orbital Environment Engine.

Provides multi-spacecraft trajectory propagation, interplanetary Lambert transfer
solving targeting moving planetary positions, pairwise conjunction screening across
N(N-1)/2 pairs, TCA refinement, Phase 10 probabilistic risk assessment, physical
collision detection (miss <= HBR_comb), and physically-propagated 4-fragment debris generation.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from theseus.bodies.catalog import ALL_BODIES, get_body, EARTH, SUN
from theseus.constants.physical import G0_VAL
from theseus.core.state import SimulationState, StateHistory
from theseus.dynamics.force_model import CompositeForceModel
from theseus.dynamics.gravity import PointMassGravity, J2Perturbation
from theseus.dynamics.drag import DragModel
from theseus.dynamics.srp import SolarRadiationPressure
from theseus.atmosphere.models import US1976StandardAtmosphere
from theseus.propagation.numerical import NumericalPropagator
from theseus.orbital.conversions import elements_to_state, state_to_elements
from theseus.orbital.elements import OrbitalElements
from theseus.orbital.lambert import solve_lambert, LambertSolution

from theseus.propagation.interpolation import interpolator_from_state_history
from theseus.conjunction.screening import ConjunctionScreener, CandidateInterval
from theseus.conjunction.tca import find_all_tca, TCAResult
from theseus.conjunction.b_plane import compute_b_plane, BPlaneResult
from theseus.conjunction.analysis import classify_encounter
from theseus.conjunction.geometry import (
    CollisionGeometry, CollisionStatus, assess_collision_geometry,
)
from theseus.uncertainty.covariance import StateCovariance
from theseus.uncertainty.state_transition import propagate_stm
from theseus.uncertainty.propagation import propagate_covariance
from theseus.uncertainty.relative import compute_relative_covariance
from theseus.uncertainty.b_plane import project_covariance_to_b_plane
from theseus.uncertainty.hard_body import compute_hard_body_radius, HardBodyResult
from theseus.uncertainty.collision_probability import compute_collision_probability, CollisionProbabilityResult
from theseus.uncertainty.risk import classify_risk, RiskAssessment, RiskThresholds, PROFILE_STANDARD

# AU in meters
AU_METERS = 149597870700.0

# Authoritative Keplerian Elements for Planetary Motion
PLANET_KEPLERIAN_DATA: Dict[str, Dict[str, float]] = {
    "mercury": {
        "a_m": 0.38709893 * AU_METERS,
        "e": 0.20563069,
        "w_rad": math.radians(77.45645),
        "period_sec": 87.9691 * 86400.0,
        "m0_rad": math.radians(174.7947),
    },
    "venus": {
        "a_m": 0.72333199 * AU_METERS,
        "e": 0.00677323,
        "w_rad": math.radians(131.53298),
        "period_sec": 224.701 * 86400.0,
        "m0_rad": math.radians(50.115),
    },
    "earth": {
        "a_m": 1.00000011 * AU_METERS,
        "e": 0.01671022,
        "w_rad": math.radians(102.94719),
        "period_sec": 365.25636 * 86400.0,
        "m0_rad": math.radians(358.617),
    },
    "mars": {
        "a_m": 1.52366231 * AU_METERS,
        "e": 0.09341233,
        "w_rad": math.radians(336.04084),
        "period_sec": 686.971 * 86400.0,
        "m0_rad": math.radians(19.373),
    },
    "jupiter": {
        "a_m": 5.20336301 * AU_METERS,
        "e": 0.04839266,
        "w_rad": math.radians(14.75385),
        "period_sec": 4332.59 * 86400.0,
        "m0_rad": math.radians(20.020),
    },
    "saturn": {
        "a_m": 9.53707032 * AU_METERS,
        "e": 0.05415060,
        "w_rad": math.radians(92.43194),
        "period_sec": 10759.22 * 86400.0,
        "m0_rad": math.radians(317.020),
    },
    "uranus": {
        "a_m": 19.19126393 * AU_METERS,
        "e": 0.04716771,
        "w_rad": math.radians(170.96424),
        "period_sec": 30685.4 * 86400.0,
        "m0_rad": math.radians(142.2386),
    },
    "neptune": {
        "a_m": 30.06896348 * AU_METERS,
        "e": 0.00858587,
        "w_rad": math.radians(44.97135),
        "period_sec": 60189.0 * 86400.0,
        "m0_rad": math.radians(256.228),
    },
}


def solve_kepler_equation(M_rad: float, e: float) -> float:
    """Solve Kepler's equation M = E - e*sin(E) using Newton-Raphson."""
    M = M_rad % (2.0 * math.pi)
    if M < 0:
        M += 2.0 * math.pi
    E = M + e * math.sin(M)
    for _ in range(25):
        f = E - e * math.sin(E) - M
        f_prime = 1.0 - e * math.cos(E)
        dE = f / f_prime
        E -= dE
        if abs(dE) < 1e-12:
            break
    return E


def get_planet_state_at_time(planet_name: str, time_sec: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate accurate heliocentric position (m) and velocity (m/s) for any planet at elapsed seconds t.
    Uses real elliptical Keplerian elements with Sun at (0, 0) focus.
    """
    key = planet_name.strip().lower()
    if key not in PLANET_KEPLERIAN_DATA:
        # Default Earth orbit if unknown
        key = "earth"
    
    elem = PLANET_KEPLERIAN_DATA[key]
    a = elem["a_m"]
    e = elem["e"]
    w = elem["w_rad"]
    T = elem["period_sec"]
    m0 = elem["m0_rad"]
    mu_sun = SUN.mu

    n = (2.0 * math.pi) / T
    M = m0 + n * time_sec
    E = solve_kepler_equation(M, e)

    # True anomaly nu
    nu = 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(E / 2.0), math.sqrt(1.0 - e) * math.cos(E / 2.0))
    r_mag = a * (1.0 - e * math.cos(E))

    theta = nu + w
    pos = np.array([r_mag * math.cos(theta), r_mag * math.sin(theta), 0.0], dtype=np.float64)

    # Heliocentric orbital velocity components
    p = a * (1.0 - e * e)
    h = math.sqrt(mu_sun * p)
    vx = -(mu_sun / h) * (math.sin(theta) + e * math.sin(w))
    vy = (mu_sun / h) * (math.cos(theta) + e * math.cos(w))
    vel = np.array([vx, vy, 0.0], dtype=np.float64)

    return pos, vel


class SolarRadiationPressureUnavailable(NotImplementedError):
    """
    Raised when ``enable_srp=True`` is requested.

    See ``MultiObjectEnvironment._build_force_model``: the option is declared
    throughout the API surface but the force model has never been constructible,
    so it is refused explicitly rather than crashing with a TypeError.
    """


@dataclass
class SpacecraftDefinition:
    """
    Complete configuration for a single spacecraft in the multi-object environment.
    Supports both geocentric/orbital configurations and interplanetary transfers.
    """
    id: str
    name: str
    vehicle_type: str = "falcon9"
    color: str = "#ff9900"
    sprite_id: str = "falcon9"
    
    # Interplanetary / Mission specification
    origin: Optional[str] = None
    destination: Optional[str] = None
    payload_mass_kg: float = 0.0
    tof_days: Optional[float] = None
    departure_epoch_date: Optional[str] = None
    
    # Vehicle physical specifications
    dry_mass_kg: float = 1000.0
    fuel_mass_kg: float = 500.0
    cross_section_area_m2: float = 10.0
    drag_coefficient: float = 2.2
    reflectivity_coefficient: float = 1.5
    thrust_n: float = 0.0
    specific_impulse_s: float = 300.0
    
    # State specification (Cartesian or Keplerian)
    central_body: str = "Earth"
    initial_r_m: Optional[np.ndarray] = None
    initial_v_m_s: Optional[np.ndarray] = None
    
    # Keplerian elements (used if initial_r_m is None and not interplanetary)
    semi_major_axis_km: Optional[float] = 6778.137
    eccentricity: Optional[float] = 0.0
    inclination_deg: Optional[float] = 51.6
    raan_deg: Optional[float] = 0.0
    arg_periapsis_deg: Optional[float] = 0.0
    true_anomaly_deg: Optional[float] = 0.0
    
    # Collision geometry (Phase 9) and state uncertainty (Phase 10).
    # hard_body_radius_m is the radius of the sphere enclosing the object
    # including appendages; it is the single source of this object's collision
    # size and is adapted into a CollisionGeometry by get_collision_geometry().
    hard_body_radius_m: float = 5.0
    sigma_pos_m: Optional[List[float]] = field(default_factory=lambda: [100.0, 100.0, 100.0])
    sigma_vel_m_s: Optional[List[float]] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    covariance_matrix_si: Optional[np.ndarray] = None
    
    # Active state & Debris metadata
    is_active: bool = True
    is_debris: bool = False
    debris_type: Optional[str] = None  # 'solar_panel', 'truss', 'nozzle', 'body'
    parent_collision_id: Optional[str] = None

    def get_initial_state(self, body_mu: float) -> Tuple[np.ndarray, np.ndarray]:
        """Resolve initial Cartesian position (m) and velocity (m/s)."""
        if self.initial_r_m is not None and self.initial_v_m_s is not None:
            return np.asarray(self.initial_r_m, dtype=np.float64), np.asarray(self.initial_v_m_s, dtype=np.float64)
        
        # Convert Keplerian elements to state vector
        a_m = (self.semi_major_axis_km or 6778.137) * 1e3
        e = self.eccentricity or 0.0
        inc_rad = math.radians(self.inclination_deg or 0.0)
        raan_rad = math.radians(self.raan_deg or 0.0)
        argp_rad = math.radians(self.arg_periapsis_deg or 0.0)
        nu_rad = math.radians(self.true_anomaly_deg or 0.0)
        
        elems = OrbitalElements(
            a=a_m,
            e=e,
            i=inc_rad,
            raan=raan_rad,
            argp=argp_rad,
            nu=nu_rad,
            mu=body_mu,
        )
        r, v = elements_to_state(elems)
        return r, v

    def get_initial_covariance(self) -> StateCovariance:
        """Construct initial StateCovariance object."""
        if self.covariance_matrix_si is not None:
            return StateCovariance(
                matrix=np.asarray(self.covariance_matrix_si, dtype=np.float64),
                frame="ICRF",
                name=self.name,
            )
        sp = self.sigma_pos_m or [100.0, 100.0, 100.0]
        sv = self.sigma_vel_m_s or [0.1, 0.1, 0.1]
        return StateCovariance.from_diagonal(sigma_pos=sp, sigma_vel=sv, name=self.name)

    def get_collision_geometry(self) -> CollisionGeometry:
        """
        Adapt this definition's hard-body radius into a CollisionGeometry.

        No new dimension is introduced: ``hard_body_radius_m`` remains the one
        place this object's collision size is defined, and this method only
        presents it in the form the Phase 9 geometry model expects.
        """
        return CollisionGeometry.from_radius(
            radius_m=self.hard_body_radius_m,
            name=self.name,
            object_type="debris" if self.is_debris else "payload",
            source="SpacecraftDefinition.hard_body_radius_m",
        )


@dataclass
class MultiConjunctionEvent:
    """A detected close-approach conjunction between two spacecraft."""
    event_id: str
    spacecraft_a_id: str
    spacecraft_b_id: str
    spacecraft_a_name: str
    spacecraft_b_name: str
    tca_s: float
    miss_distance_m: float
    relative_velocity_m_s: float
    encounter_angle_deg: float
    encounter_type: str
    r_rel_m: List[float]
    v_rel_m_s: List[float]
    b_plane_b_t_m: Optional[float] = None
    b_plane_b_r_m: Optional[float] = None
    b_plane_sigma_major_m: Optional[float] = None
    b_plane_sigma_minor_m: Optional[float] = None
    b_plane_ellipse_angle_deg: Optional[float] = None
    b_plane_covariance_m2: Optional[List[List[float]]] = None
    hard_body_radius_m: float = 10.0
    collision_probability: Optional[float] = None
    clearance_m: Optional[float] = None
    collision_status: str = "UNKNOWN"
    risk_level: str = "LOW"
    action_required: bool = False
    is_physical_collision: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "spacecraft_a_id": self.spacecraft_a_id,
            "spacecraft_b_id": self.spacecraft_b_id,
            "spacecraft_a_name": self.spacecraft_a_name,
            "spacecraft_b_name": self.spacecraft_b_name,
            "tca_s": float(self.tca_s),
            "miss_distance_m": float(self.miss_distance_m),
            "miss_distance_km": float(self.miss_distance_m / 1e3),
            "relative_velocity_m_s": float(self.relative_velocity_m_s),
            "relative_velocity_km_s": float(self.relative_velocity_m_s / 1e3),
            "encounter_angle_deg": float(self.encounter_angle_deg),
            "encounter_type": self.encounter_type,
            "r_rel_m": self.r_rel_m,
            "v_rel_m_s": self.v_rel_m_s,
            "b_plane_b_t_m": self.b_plane_b_t_m,
            "b_plane_b_r_m": self.b_plane_b_r_m,
            "b_plane_b_t_km": (self.b_plane_b_t_m / 1e3) if self.b_plane_b_t_m is not None else None,
            "b_plane_b_r_km": (self.b_plane_b_r_m / 1e3) if self.b_plane_b_r_m is not None else None,
            "b_plane_sigma_major_m": self.b_plane_sigma_major_m,
            "b_plane_sigma_minor_m": self.b_plane_sigma_minor_m,
            "b_plane_sigma_major_km": (self.b_plane_sigma_major_m / 1e3) if self.b_plane_sigma_major_m is not None else None,
            "b_plane_sigma_minor_km": (self.b_plane_sigma_minor_m / 1e3) if self.b_plane_sigma_minor_m is not None else None,
            "b_plane_ellipse_angle_deg": self.b_plane_ellipse_angle_deg,
            "b_plane_covariance_m2": self.b_plane_covariance_m2,
            "hard_body_radius_m": float(self.hard_body_radius_m),
            "hard_body_radius_km": float(self.hard_body_radius_m / 1e3),
            "collision_probability": self.collision_probability,
            "collision_probability_scientific": f"{self.collision_probability:.6e}" if self.collision_probability is not None else "0.000000e+00",
            "risk_level": self.risk_level,
            "action_required": self.action_required,
            "is_physical_collision": self.is_physical_collision,
            "collision_status": self.collision_status,
            "clearance_m": (None if self.clearance_m is None else float(self.clearance_m)),
            "clearance_km": (None if self.clearance_m is None else float(self.clearance_m) / 1e3),
        }


@dataclass
class PhysicalCollisionEvent:
    """A verified physical collision event where miss distance <= combined HBR."""
    collision_id: str
    time_s: float
    spacecraft_a_id: str
    spacecraft_b_id: str
    spacecraft_a_name: str
    spacecraft_b_name: str
    collision_position_m: List[float]
    relative_velocity_m_s: float
    miss_distance_m: float
    combined_hbr_m: float
    debris_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "collision_id": self.collision_id,
            "time_s": float(self.time_s),
            "spacecraft_a_id": self.spacecraft_a_id,
            "spacecraft_b_id": self.spacecraft_b_id,
            "spacecraft_a_name": self.spacecraft_a_name,
            "spacecraft_b_name": self.spacecraft_b_name,
            "collision_position_m": self.collision_position_m,
            "collision_position_km": [x / 1e3 for x in self.collision_position_m],
            "relative_velocity_m_s": float(self.relative_velocity_m_s),
            "relative_velocity_km_s": float(self.relative_velocity_m_s / 1e3),
            "miss_distance_m": float(self.miss_distance_m),
            "combined_hbr_m": float(self.combined_hbr_m),
            "debris_ids": self.debris_ids,
        }


@dataclass
class SpacecraftTrackResult:
    """Tracked state history, trajectory, delta-V, propellant budget, and trace for one object."""
    definition: SpacecraftDefinition
    state_history: List[Dict[str, Any]]
    destroyed: bool = False
    destruction_time_s: Optional[float] = None
    destruction_reason: Optional[str] = None
    delta_v_budget: Optional[Dict[str, Any]] = None
    propellant_budget: Optional[Dict[str, Any]] = None
    calculation_trace: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.definition.id,
            "name": self.definition.name,
            "vehicle_type": self.definition.vehicle_type,
            "color": self.definition.color,
            "sprite_id": self.definition.sprite_id,
            "origin": self.definition.origin,
            "destination": self.definition.destination,
            "is_debris": self.definition.is_debris,
            "debris_type": self.definition.debris_type,
            "parent_collision_id": self.definition.parent_collision_id,
            "hard_body_radius_m": float(self.definition.hard_body_radius_m),
            "destroyed": self.destroyed,
            "destruction_time_s": self.destruction_time_s,
            "destruction_reason": self.destruction_reason,
            "delta_v_budget": self.delta_v_budget,
            "propellant_budget": self.propellant_budget,
            "calculation_trace": self.calculation_trace or [],
            "state_history": self.state_history,
        }


@dataclass
class MultiObjectSimulationResult:
    """Complete simulation response container for the multi-spacecraft environment."""
    objects: List[SpacecraftTrackResult]
    conjunctions: List[MultiConjunctionEvent]
    collisions: List[PhysicalCollisionEvent]
    time_span_s: Tuple[float, float]
    central_body: str
    calculation_steps: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objects": [obj.to_dict() for obj in self.objects],
            "conjunctions": [c.to_dict() for c in self.conjunctions],
            "collisions": [coll.to_dict() for coll in self.collisions],
            "time_span_s": list(self.time_span_s),
            "central_body": self.central_body,
            "calculation_steps": self.calculation_steps,
            "summary": self.summary if self.summary else {
                "total_spacecraft": len([o for o in self.objects if not o.definition.is_debris]),
                "total_debris": len([o for o in self.objects if o.definition.is_debris]),
                "total_conjunctions": len(self.conjunctions),
                "total_collisions": len(self.collisions),
                "active_spacecraft_count": len([o for o in self.objects if not o.destroyed and not o.definition.is_debris]),
                "destroyed_spacecraft_count": len([o for o in self.objects if o.destroyed]),
            },
        }


def solve_interplanetary_transfer(
    sc: SpacecraftDefinition,
    departure_t_sec: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float, List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    Solve Lambert boundary value transfer problem for an interplanetary spacecraft targeting the
    future moving position of the target planet at arrival epoch.

    Returns:
    (r0, v0, tof_sec, calculation_trace, delta_v_budget, propellant_budget)
    """
    orig_name = sc.origin or "Earth"
    dest_name = sc.destination or "Mars"
    mu_sun = SUN.mu

    # 1. Ephemeris state at departure t0
    r1, v_body1 = get_planet_state_at_time(orig_name, departure_t_sec)
    r1_mag = float(np.linalg.norm(r1))

    # 2. Time-of-flight estimation
    orig_elem = PLANET_KEPLERIAN_DATA.get(orig_name.lower(), PLANET_KEPLERIAN_DATA["earth"])
    dest_elem = PLANET_KEPLERIAN_DATA.get(dest_name.lower(), PLANET_KEPLERIAN_DATA["mars"])

    if sc.tof_days is not None and sc.tof_days > 0:
        tof_sec = sc.tof_days * 86400.0
    else:
        # Standard Hohmann transfer time of flight
        a_tx = (r1_mag + dest_elem["a_m"]) / 2.0
        tof_sec = math.pi * math.sqrt((a_tx ** 3) / mu_sun)

    # 3. Exact future moving position of target planet at arrival t_arrival = t0 + TOF
    t_arrival = departure_t_sec + tof_sec
    r2, v_body2 = get_planet_state_at_time(dest_name, t_arrival)
    r2_mag = float(np.linalg.norm(r2))

    # 4. Universal-Variable Lambert Solver
    lambert_sol: LambertSolution = solve_lambert(r1, r2, tof_sec, mu_sun, prograde=True)
    v1 = lambert_sol.v1
    v2 = lambert_sol.v2



    # 5. Delta-V & Propellant Budget
    dv1 = float(np.linalg.norm(v1 - v_body1))
    dv2 = float(np.linalg.norm(v_body2 - v2))
    total_dv = dv1 + dv2

    m_dry_tot = sc.dry_mass_kg + sc.payload_mass_kg
    m0 = m_dry_tot + sc.fuel_mass_kg
    ve = sc.specific_impulse_s * G0_VAL
    fuel_req = m0 * (1.0 - math.exp(-total_dv / ve)) if ve > 0 else 0.0
    fuel_margin = sc.fuel_mass_kg - fuel_req
    dv_avail = ve * math.log(m0 / m_dry_tot) if (m_dry_tot > 0 and ve > 0) else 0.0

    # 6. Detailed Step-by-Step Calculation Trace
    traces = [
        {
            "stepIndex": 1,
            "phase": "DEPARTURE_EPHEMERIS",
            "title": f"Departure Body ({orig_name}) Heliocentric State at t = {departure_t_sec:.0f}s",
            "status": "ACQUIRED",
            "equation": "r_dep = r(nu_dep),  v_dep = v(nu_dep)",
            "result": f"|r_dep| = {r1_mag / AU_METERS:.4f} AU, |v_dep| = {np.linalg.norm(v_body1) / 1e3:.2f} km/s",
            "explanation": f"Evaluated Keplerian orbital position of {orig_name} around the Sun at departure epoch.",
            "beginnerExplanation": f"Locating {orig_name}'s precise position in the solar system at launch.",
        },
        {
            "stepIndex": 2,
            "phase": "MOVING_TARGET_PREDICTION",
            "title": f"Target Body ({dest_name}) Predicted Future State at Arrival (t = {t_arrival / 86400:.1f} days)",
            "status": "CALCULATED",
            "equation": "t_arrival = t_dep + TOF;  r_target = r(t_arrival)",
            "result": f"|r_target| = {r2_mag / AU_METERS:.4f} AU, |v_target| = {np.linalg.norm(v_body2) / 1e3:.2f} km/s",
            "explanation": f"Calculated {dest_name}'s future position after {tof_sec / 86400:.1f} days of orbital motion.",
            "beginnerExplanation": f"Targeting where {dest_name} will be when the spacecraft arrives, not where it is today.",
        },
        {
            "stepIndex": 3,
            "phase": "LAMBERT_BOUNDARY_VALUE_SOLVER",
            "title": f"Solve Universal-Variable Lambert Problem ({lambert_sol.iterations} iterations)",
            "status": "CONVERGED",
            "equation": "F(z) = (y/C)^(3/2) S + A√y - √μ Δt = 0",
            "result": f"Transfer angle = {math.degrees(lambert_sol.transfer_angle):.2f}°, z = {lambert_sol.z_final:.6f}",
            "explanation": f"Converged to machine precision residual ({lambert_sol.residual:.2e} s) in {lambert_sol.iterations} iterations.",
            "beginnerExplanation": "Solving the exact curved space trajectory linking the two planetary positions.",
        },
        {
            "stepIndex": 4,
            "phase": "DELTA_V_EVALUATION",
            "title": "Evaluate Impulsive Velocity Changes (Δv₁ and Δv₂)",
            "status": "CALCULATED",
            "equation": "Δv₁ = |v₁ - v_body1|;  Δv₂ = |v_body2 - v₂|;  Δv_tot = Δv₁ + Δv₂",
            "result": f"Δv₁ = {dv1 / 1e3:.3f} km/s, Δv₂ = {dv2 / 1e3:.3f} km/s, Total Δv = {total_dv / 1e3:.3f} km/s",
            "explanation": f"Injection requires +{dv1 / 1e3:.2f} km/s boost and arrival capture requires +{dv2 / 1e3:.2f} km/s.",
            "beginnerExplanation": "Calculating the rocket engine burns needed at departure and destination arrival.",
        },
        {
            "stepIndex": 5,
            "phase": "TSIOLKOVSKY_PROPULSION",
            "title": "Evaluate Spacecraft Propellant Consumption",
            "status": "COMPLETE",
            "equation": "m_fuel = m₀ · (1 - e^(-Δv / (I_sp · g₀)))",
            "result": f"Fuel consumed = {fuel_req:.1f} kg (Margin: {fuel_margin:.1f} kg)",
            "explanation": f"Spacecraft consumes {fuel_req:.1f} kg out of {sc.fuel_mass_kg:.1f} kg propellant loaded.",
            "beginnerExplanation": "Verifying fuel tanks have sufficient propellant capacity with safety margin.",
        },
    ]

    delta_v_budget = {
        "delta_v1": float(dv1),
        "delta_v2": float(dv2),
        "total_delta_v": float(total_dv),
        "available_delta_v": float(dv_avail),
        "margin_delta_v": float(dv_avail - total_dv),
    }

    propellant_budget = {
        "initial_total_mass_kg": float(m0),
        "dry_mass_kg": float(m_dry_tot),
        "initial_fuel_kg": float(sc.fuel_mass_kg),
        "fuel_consumed_kg": float(fuel_req),
        "fuel_margin_kg": float(fuel_margin),
    }

    return r1, v1, tof_sec, traces, delta_v_budget, propellant_budget


class MultiObjectEnvironment:
    """
    Unified astrodynamics runtime for N spacecraft simultaneously.
    Supports geocentric orbits, heliocentric interplanetary transfers to moving planets,
    pairwise conjunction screening, covariance propagation, and 4-debris physical collision simulation.
    """

    def __init__(
        self,
        central_body: str = "Earth",
        screening_threshold_km: float = 100.0,
        coarse_dt_s: float = 15.0,
        tca_tolerance_s: float = 1e-5,
        enable_j2: bool = True,
        enable_drag: bool = True,
        enable_srp: bool = False,
    ) -> None:
        self.body_name = central_body
        self.body = get_body(central_body)
        self.screening_threshold_m = screening_threshold_km * 1e3
        self.coarse_dt = coarse_dt_s
        self.tca_tol = tca_tolerance_s
        self.enable_j2 = enable_j2
        self.enable_drag = enable_drag
        self.enable_srp = enable_srp

    @staticmethod
    def _analytic_jacobian_covers(force_model: CompositeForceModel) -> bool:
        """
        Is the STM's analytic Jacobian complete for this force model?

        The analytic Jacobian in ``theseus.uncertainty.state_transition`` has
        terms for point-mass gravity and J2 only, and asserts ∂a/∂v = 0.  Drag
        depends on both position (through density and the co-rotating
        atmosphere) and velocity; SRP and any third-body term depend on
        position.  Whenever one of those is active the analytic Jacobian
        describes a different dynamical system than the one being propagated,
        so the STM must be built from the numerical Jacobian of the actual
        acceleration.

        Decided from the assembled force model rather than from the
        ``enable_*`` flags, so a model composed by any other route is judged
        by what it actually contains -- and rather than from the trajectory,
        so an eccentric orbit whose drag is negligible at the reference epoch
        but significant at perigee is still handled correctly.
        """
        return all(
            isinstance(model, (PointMassGravity, J2Perturbation))
            for model in force_model.active_models
        )

    @staticmethod
    def _analytic_j2_for(force_model: CompositeForceModel,
                         body_j2: float) -> float:
        """
        The J2 coefficient the analytic Jacobian should be built with, which is
        the one this force model actually applies -- not the one the central
        body happens to have.

        The STM call site used to pass ``j2=self.body.J2`` unconditionally.
        For a two-body-only force model (``enable_j2=False``, which the
        ``/api/simulate/environment`` endpoint exposes) that describes a
        different dynamical system than the trajectory, so P10-06's guard
        rejected the analytic Jacobian and the STM fell back to the numerical
        one -- twelve extra acceleration evaluations per integration step for
        no reason.  Measured: gravity-only model with ``j2 = 1.08263e-3``
        passed is rejected; with ``j2 = 0.0`` it is accepted.

        The result was never wrong, because the guard caught the mismatch every
        time.  It was needlessly slow, and it made the guard fire on a case it
        was not written for, which obscures the cases it was.
        """
        has_j2 = any(isinstance(model, J2Perturbation)
                     for model in force_model.active_models)
        return float(body_j2) if has_j2 else 0.0

    def _build_force_model(self, sc_def: SpacecraftDefinition) -> CompositeForceModel:
        """Assemble active physical force models for a spacecraft."""
        fm = CompositeForceModel()
        
        # 1. Point mass gravity
        fm.add(PointMassGravity(self.body))
        
        # 2. J2 Oblateness (if body has non-zero J2)
        if self.enable_j2 and getattr(self.body, "J2", 0.0) != 0.0:
            fm.add(J2Perturbation(self.body))
            
        # 3. Atmospheric drag (if central body has atmosphere)
        if self.enable_drag and self.body.atmosphere and self.body.atmosphere.has_atmosphere:
            drag_area = sc_def.cross_section_area_m2
            drag_cd = sc_def.drag_coefficient
            atm_model = US1976StandardAtmosphere()
            rot_rate = getattr(self.body, "rotation_rate_rad_s", 7.2921159e-5)
            fm.add(DragModel(atmosphere=atm_model, cd=drag_cd, area=drag_area, body_radius=self.body.radius, body_rotation_rate=rot_rate))
            
        # 4. Solar Radiation Pressure
        if self.enable_srp:
            # SolarRadiationPressure requires an ephemeris provider -- it needs
            # to know where the Sun is -- and this call site never supplied one,
            # so `enable_srp=True` raised
            #
            #     TypeError: SolarRadiationPressure.__init__() missing 1
            #     required positional argument: 'ephemeris'
            #
            # every time.  Reached through /api/simulate/environment, which
            # documents `enable_srp` as a supported option, that surfaced as a
            # bare HTTP 500.
            #
            # Passing an ephemeris here would make the flag "work", but it
            # would also switch on a perturbation this project has never
            # validated end to end, silently changing every trajectory of
            # anyone who sets it.  A documented option that has never run is
            # not made trustworthy by making it run; it is made honest by
            # saying it is not available.  The wiring is recorded as a known
            # limitation instead.
            raise SolarRadiationPressureUnavailable(
                "Solar radiation pressure is not available: the force-model "
                "assembly has no ephemeris provider to locate the Sun, and the "
                "perturbation has not been validated in this configuration. "
                "Re-run with enable_srp=false. This is a known limitation, not "
                "a transient failure."
            )

        return fm

    def simulate(
        self,
        spacecraft_list: List[SpacecraftDefinition],
        t_start: float = 0.0,
        t_end: Optional[float] = None,
        output_dt: float = 30.0,
    ) -> MultiObjectSimulationResult:
        """
        Execute multi-object simulation across all spacecraft.
        Automatically resolves interplanetary Lambert transfers when destination bodies are specified.

        Note on ``output_dt``
        --------------------
        This is **not** a fixed output grid.  It is passed to the propagator as
        the *initial* integrator step; from there the RKF45 error controller
        chooses every subsequent step, so the actual node spacing is whatever
        the requested tolerances demand.  For a 400 km LEO trajectory at the
        tolerances used here that settles at roughly 69 s regardless of whether
        10 s, 30 s or 120 s was requested -- only the first step reflects the
        request.

        This is correct behaviour for an adaptive integrator and is not forced
        to a uniform grid.  It does not limit downstream accuracy: states
        between nodes are reconstructed by cubic Hermite interpolation using
        the stored velocities, which is sub-metre at this spacing, so any
        consumer may evaluate the trajectory at any instant it likes.
        """
        if not spacecraft_list:
            return MultiObjectSimulationResult(
                objects=[],
                conjunctions=[],
                collisions=[],
                time_span_s=(t_start, t_end or 7200.0),
                central_body=self.body_name,
            )

        calc_steps: List[Dict[str, Any]] = []
        step_idx = 1

        # Check if any spacecraft is interplanetary (e.g. Earth -> Mars) or heliocentric
        is_interplanetary_sim = any(
            (sc.destination and sc.destination.lower() not in ["earth", "orbit", "target"]) or (sc.central_body.lower() == "sun")
            for sc in spacecraft_list
        )

        if is_interplanetary_sim:
            self.body_name = "Sun"
            self.body = get_body("Sun")

        # -------------------------------------------------------------
        # STEP 1: Resolve Initial States & Interplanetary Transfers
        # -------------------------------------------------------------
        resolved_initial_states: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        sc_delta_v_budgets: Dict[str, Dict[str, Any]] = {}
        sc_propellant_budgets: Dict[str, Dict[str, Any]] = {}
        sc_traces: Dict[str, List[Dict[str, Any]]] = {}
        max_transfer_tof = 7200.0

        for sc in spacecraft_list:
            if is_interplanetary_sim and sc.destination and sc.destination.lower() not in ["orbit", "target"]:
                # Solve independent interplanetary transfer to moving target planet
                r0, v0, sc_tof, traces, dv_budget, prop_budget = solve_interplanetary_transfer(sc, departure_t_sec=t_start)
                resolved_initial_states[sc.id] = (r0, v0)
                sc_delta_v_budgets[sc.id] = dv_budget
                sc_propellant_budgets[sc.id] = prop_budget
                sc_traces[sc.id] = traces
                if sc_tof > max_transfer_tof:
                    max_transfer_tof = sc_tof
            else:
                r0, v0 = sc.get_initial_state(self.body.mu)
                resolved_initial_states[sc.id] = (r0, v0)

        # Determine overall simulation duration
        sim_t_end = t_end if t_end is not None else max_transfer_tof

        # Adapt output step size for large time spans
        effective_dt = output_dt
        if sim_t_end > 86400.0 * 5.0 and output_dt < 3600.0:
            effective_dt = max(output_dt, sim_t_end / 500.0)

        # -------------------------------------------------------------
        # STEP 2: Numerical Propagation of all Spacecraft Trajectories
        # -------------------------------------------------------------
        prop_histories: Dict[str, StateHistory] = {}
        interpolators: Dict[str, Tuple[Callable[[float], np.ndarray], Callable[[float], np.ndarray]]] = {}
        initial_covariances: Dict[str, StateCovariance] = {}
        
        is_sun = (self.body_name.lower() == "sun")
        atol_val = 10.0 if is_sun else 1e-4
        rtol_val = 1e-7 if is_sun else 1e-8
        
        for sc in spacecraft_list:
            r0, v0 = resolved_initial_states[sc.id]
            cov0 = sc.get_initial_covariance()
            initial_covariances[sc.id] = cov0
            
            fm = self._build_force_model(sc)
            mass_tot = sc.dry_mass_kg + sc.payload_mass_kg + sc.fuel_mass_kg
            
            propagator = NumericalPropagator(
                acceleration_fn=fm.compute_acceleration,
                integrator="rkf45",
                dt=effective_dt,
                atol=atol_val,
                rtol=rtol_val,
                mu=self.body.mu,
            )
            
            history, _, _ = propagator.propagate(
                r0=r0,
                v0=v0,
                t_span=(t_start, sim_t_end),
                mass=mass_tot,
                fuel_mass=sc.fuel_mass_kg,
            )
            prop_histories[sc.id] = history

            
            # Build the continuous trajectory interpolator.
            #
            # Cubic Hermite, using the velocities the propagator already stored
            # at every node.  Linear interpolation of position discards those
            # velocities and replaces each arc by its chord, which at LEO node
            # spacing costs kilometres -- far more than the miss distances the
            # conjunction pipeline downstream is trying to resolve.
            trajectory_interp = interpolator_from_state_history(history)
            interpolators[sc.id] = trajectory_interp.as_callables()

        calc_steps.append({
            "stepIndex": step_idx,
            "phase": "MULTI_FLEET_PROPAGATION",
            "title": f"Simultaneous Propagation of {len(spacecraft_list)} Spacecraft ({self.body_name} Gravity)",
            "status": "completed",
            "equation": "d²r/dt² = -μ/r³ r + a_pert",
            "result": f"Propagated {len(spacecraft_list)} spacecraft across [{t_start:.0f}s, {sim_t_end:.0f}s] window",
            "explanation": f"Numerical 7-DOF integration (RKF45) with central body gravity ({self.body_name}).",
            "beginnerExplanation": f"Simulating {len(spacecraft_list)} spacecraft traveling through space.",
        })
        step_idx += 1

        # -------------------------------------------------------------
        # STEP 3: Pairwise Conjunction Screening across N(N-1)/2 Pairs
        # -------------------------------------------------------------
        screener = ConjunctionScreener(
            threshold_m=self.screening_threshold_m,
            coarse_dt=min(self.coarse_dt, effective_dt, 86400.0 * 2.0) if is_sun else min(self.coarse_dt, effective_dt, 15.0),
        )


        all_conjunctions: List[MultiConjunctionEvent] = []
        all_collisions: List[PhysicalCollisionEvent] = []
        destroyed_flags: Dict[str, Tuple[float, str]] = {}  # id -> (destruction_time, reason)

        sc_ids = list(prop_histories.keys())
        sc_map = {sc.id: sc for sc in spacecraft_list}

        for i in range(len(sc_ids)):
            for j in range(i + 1, len(sc_ids)):
                id_a = sc_ids[i]
                id_b = sc_ids[j]
                sc_a = sc_map[id_a]
                sc_b = sc_map[id_b]
                
                pos_a_fn, vel_a_fn = interpolators[id_a]
                pos_b_fn, vel_b_fn = interpolators[id_b]
                
                candidates = screener.screen(
                    pos_a_fn, pos_b_fn, t_start, sim_t_end,
                    vel_fn_a=vel_a_fn, vel_fn_b=vel_b_fn,
                    # Name the actual spacecraft, so a non-finite trajectory
                    # is reported against the object it belongs to rather
                    # than against an anonymous "A"/"B" of some pair.
                    object_a_id=id_a, object_b_id=id_b,
                )
                if not candidates:
                    continue

                for ci in candidates:
                    tcas = find_all_tca(
                        pos_a_fn, vel_a_fn, pos_b_fn, vel_b_fn,
                        ci.t_start, ci.t_end,
                        tol=self.tca_tol,
                        object_a_id=id_a, object_b_id=id_b,
                    )
                    for tca in tcas:
                        if not tca.validated:
                            continue
                        
                        r_rel = tca.r_rel
                        v_rel = tca.v_rel
                        miss_d = float(tca.miss_distance)
                        rel_v = float(tca.relative_velocity)
                        enc_angle, enc_type = classify_encounter(tca.v_a, tca.v_b)
                        
                        # B-Plane Kizner Analysis
                        b_res = compute_b_plane(r_rel, v_rel)
                        
                        # Phase 10 Covariance Propagation to TCA
                        cov_a_0 = initial_covariances[id_a]
                        cov_b_0 = initial_covariances[id_b]
                        
                        fm_a = self._build_force_model(sc_a)
                        fm_b = self._build_force_model(sc_b)
                        m_a_tot = sc_a.dry_mass_kg + sc_a.payload_mass_kg + sc_a.fuel_mass_kg
                        m_b_tot = sc_b.dry_mass_kg + sc_b.payload_mass_kg + sc_b.fuel_mass_kg
                        
                        acc_fn_a = lambda t, r, v: fm_a.compute_acceleration(t, r, v, m_a_tot)
                        acc_fn_b = lambda t, r, v: fm_b.compute_acceleration(t, r, v, m_b_tot)
                        
                        # --- Epochs, named explicitly -------------------------
                        #
                        # covariance_epoch_s : where cov_*_0 is defined, i.e.
                        #     the propagation start.  This is also the STM's
                        #     reference epoch, because P(t) = Phi P0 Phi^T is
                        #     only meaningful when both share one epoch.
                        # tca_epoch_s : where the covariance is evaluated.
                        #
                        # propagate_stm establishes Phi = I at t_span[0] and
                        # integrates [r, v, vec(Phi)] forward together, so its
                        # (r0, v0) argument is the nominal state *at the
                        # reference epoch* -- the trajectory the variational
                        # equations are linearised about -- and not the state
                        # to evaluate at.  Passing tca.r_a / tca.v_a here made
                        # Phi the transition matrix of a trajectory that leaves
                        # the covariance epoch already holding the state the
                        # object only reaches at TCA: a trajectory that never
                        # existed.  Measured on an e = 0.6 orbit that put the
                        # matrix 67 % away from the true Phi and shrank the
                        # reported position uncertainty at TCA to 0.27x, with
                        # the B-plane ellipse misoriented even for e = 0.
                        covariance_epoch_s = t_start
                        tca_epoch_s = tca.tca

                        r_a_at_covariance_epoch, v_a_at_covariance_epoch = \
                            resolved_initial_states[id_a]
                        r_b_at_covariance_epoch, v_b_at_covariance_epoch = \
                            resolved_initial_states[id_b]

                        # The Jacobian must describe the same acceleration
                        # model that produced the nominal trajectory, so each
                        # object's force model decides which one is valid.
                        analytic_ok_a = self._analytic_jacobian_covers(fm_a)
                        analytic_ok_b = self._analytic_jacobian_covers(fm_b)
                        body_j2 = float(getattr(self.body, "J2", 0.0) or 0.0)
                        analytic_j2_a = self._analytic_j2_for(fm_a, body_j2)
                        analytic_j2_b = self._analytic_j2_for(fm_b, body_j2)

                        stm_dt = max(60.0, (tca_epoch_s - covariance_epoch_s) / 200.0)
                        stm_a = propagate_stm(
                            acc_fn_a,
                            r_a_at_covariance_epoch, v_a_at_covariance_epoch,
                            (covariance_epoch_s, tca_epoch_s),
                            mu=self.body.mu,
                            j2=analytic_j2_a,
                            radius=self.body.radius,
                            dt=stm_dt,
                            atol=atol_val,
                            rtol=rtol_val,
                            use_analytic_jacobian=analytic_ok_a,
                        )
                        stm_b = propagate_stm(
                            acc_fn_b,
                            r_b_at_covariance_epoch, v_b_at_covariance_epoch,
                            (covariance_epoch_s, tca_epoch_s),
                            mu=self.body.mu,
                            j2=analytic_j2_b,
                            radius=self.body.radius,
                            dt=stm_dt,
                            atol=atol_val,
                            rtol=rtol_val,
                            use_analytic_jacobian=analytic_ok_b,
                        )

                        cov_a_tca = propagate_covariance(cov_a_0, stm_a.stm, tca_epoch_s)
                        cov_b_tca = propagate_covariance(cov_b_0, stm_b.stm, tca_epoch_s)
                        
                        rel_cov = compute_relative_covariance(cov_a_tca, cov_b_tca)
                        
                        # B-Plane Uncertainty Projection
                        b_unc = project_covariance_to_b_plane(
                            rel_pos_cov=rel_cov.position_covariance,
                            r_rel=r_rel,
                            v_rel=v_rel,
                            b_plane_result=b_res,
                        )
                        
                        # Deterministic collision geometry.  The Phase 9 model
                        # owns this decision; nothing here re-derives it.
                        collision = assess_collision_geometry(
                            miss_distance_m=miss_d,
                            geom_a=sc_a.get_collision_geometry(),
                            geom_b=sc_b.get_collision_geometry(),
                        )
                        combined_hbr = collision.combined_hard_body_radius_m
                        is_collision = collision.is_physical_intersection

                        # Probability of collision is a separate, probabilistic
                        # question and stays in Phase 10.
                        pc_res = compute_collision_probability(b_unc, combined_hbr)
                        risk_res = classify_risk(pc_res.probability, PROFILE_STANDARD)

                        event_id = f"CONJ-{id_a}-{id_b}-{tca.tca:.1f}"
                        conj_event = MultiConjunctionEvent(
                            event_id=event_id,
                            spacecraft_a_id=id_a,
                            spacecraft_b_id=id_b,
                            spacecraft_a_name=sc_a.name,
                            spacecraft_b_name=sc_b.name,
                            tca_s=tca.tca,
                            miss_distance_m=miss_d,
                            relative_velocity_m_s=rel_v,
                            encounter_angle_deg=enc_angle,
                            encounter_type=enc_type,
                            r_rel_m=r_rel.tolist(),
                            v_rel_m_s=v_rel.tolist(),
                            b_plane_b_t_m=b_unc.b_dot_t if b_res.applicable else None,
                            b_plane_b_r_m=b_unc.b_dot_r if b_res.applicable else None,
                            b_plane_sigma_major_m=b_unc.sigma_major if b_res.applicable else None,
                            b_plane_sigma_minor_m=b_unc.sigma_minor if b_res.applicable else None,
                            b_plane_ellipse_angle_deg=b_unc.ellipse_angle_deg if b_res.applicable else None,
                            b_plane_covariance_m2=b_unc.b_plane_covariance.tolist() if b_res.applicable else None,
                            hard_body_radius_m=combined_hbr,
                            collision_probability=pc_res.probability,
                            risk_level=risk_res.level.value,
                            action_required=risk_res.action_required,
                            is_physical_collision=is_collision,
                            clearance_m=collision.clearance_m,
                            collision_status=collision.status.value,
                        )
                        all_conjunctions.append(conj_event)
                        
                        if is_collision:
                            coll_id = f"COLL-{id_a}-{id_b}-{tca.tca:.1f}"
                            r_coll = 0.5 * (tca.r_a + tca.r_b)
                            all_collisions.append(PhysicalCollisionEvent(
                                collision_id=coll_id,
                                time_s=tca.tca,
                                spacecraft_a_id=id_a,
                                spacecraft_b_id=id_b,
                                spacecraft_a_name=sc_a.name,
                                spacecraft_b_name=sc_b.name,
                                collision_position_m=r_coll.tolist(),
                                relative_velocity_m_s=rel_v,
                                miss_distance_m=miss_d,
                                combined_hbr_m=combined_hbr,
                            ))
                            
                            if id_a not in destroyed_flags or tca.tca < destroyed_flags[id_a][0]:
                                destroyed_flags[id_a] = (tca.tca, f"Destroyed in physical collision with {sc_b.name}")
                            if id_b not in destroyed_flags or tca.tca < destroyed_flags[id_b][0]:
                                destroyed_flags[id_b] = (tca.tca, f"Destroyed in physical collision with {sc_a.name}")

        all_conjunctions.sort(key=lambda c: c.tca_s)
        all_collisions.sort(key=lambda c: c.time_s)

        calc_steps.append({
            "stepIndex": step_idx,
            "phase": "CONJUNCTION_SCREENING",
            "title": f"Pairwise Conjunction Screening ({len(all_conjunctions)} Conjunctions, {len(all_collisions)} Collisions)",
            "status": "completed",
            "equation": "(r₁ - r₂) · (v₁ - v₂) = 0",
            "result": f"Screened {len(spacecraft_list)*(len(spacecraft_list)-1)//2} spacecraft pairs; identified {len(all_conjunctions)} close encounters",
            "explanation": "Evaluated pairwise relative geometry and refined Times of Closest Approach (TCA) via Brent's method root-solver.",
            "beginnerExplanation": f"Scanned all satellite paths for potential near misses and collisions.",
        })
        step_idx += 1


        # -------------------------------------------------------------
        # STEP 4: Physical 4-Debris Fragment Generation & Propagation
        # -------------------------------------------------------------
        debris_tracks: List[SpacecraftTrackResult] = []

        for coll in all_collisions:
            t_coll = coll.time_s
            r_coll = np.array(coll.collision_position_m, dtype=np.float64)
            
            sc_a = sc_map[coll.spacecraft_a_id]
            sc_b = sc_map[coll.spacecraft_b_id]
            pos_a_fn, vel_a_fn = interpolators[coll.spacecraft_a_id]
            pos_b_fn, vel_b_fn = interpolators[coll.spacecraft_b_id]
            
            v_a_coll = vel_a_fn(t_coll)
            v_b_coll = vel_b_fn(t_coll)
            
            m_a = sc_a.dry_mass_kg + sc_a.payload_mass_kg + sc_a.fuel_mass_kg
            m_b = sc_b.dry_mass_kg + sc_b.payload_mass_kg + sc_b.fuel_mass_kg
            m_tot = m_a + m_b
            
            # Momentum conservation center-of-mass frame
            v_cm = (m_a * v_a_coll + m_b * v_b_coll) / m_tot
            v_rel = v_a_coll - v_b_coll
            v_rel_mag = max(1.0, float(np.linalg.norm(v_rel)))
            
            e_v = v_rel / v_rel_mag
            r_coll_mag = max(1.0, float(np.linalg.norm(r_coll)))
            r_unit = r_coll / r_coll_mag
            
            e_n = np.cross(r_unit, e_v)
            e_n_mag = float(np.linalg.norm(e_n))
            if e_n_mag < 1e-6:
                e_n = np.cross(np.array([0.0, 0.0, 1.0]), e_v)
                e_n_mag = float(np.linalg.norm(e_n))
            e_n = e_n / e_n_mag
            e_b = np.cross(e_v, e_n)
            
            dv_base = min(120.0, max(15.0, 0.03 * v_rel_mag + 20.0))
            
            debris_specs = [
                ("A", "solar_panel", "#44bbff", 0.10 * m_tot, 4.0, 2.8, dv_base * 1.2 * (0.8 * e_v + 0.6 * e_n)),
                ("B", "truss", "#e6dfd5", 0.35 * m_tot, 2.5, 2.2, dv_base * 0.9 * (-0.7 * e_v + 0.7 * e_b)),
                ("C", "nozzle", "#ff9900", 0.35 * m_tot, 1.5, 1.8, dv_base * 0.8 * (-0.5 * e_n - 0.8 * e_b)),
                ("D", "body", "#a0988e", 0.20 * m_tot, 3.0, 2.4, dv_base * 1.0 * (0.6 * e_n - 0.8 * e_v)),
            ]
            
            coll_debris_ids = []

            for letter, d_type, d_color, d_mass, d_area, d_cd, dv_vec in debris_specs:
                d_id = f"DEBRIS-{coll.collision_id}-{letter}"
                coll_debris_ids.append(d_id)
                
                v_debris_0 = v_cm + dv_vec
                
                debris_def = SpacecraftDefinition(
                    id=d_id,
                    name=f"Debris-{letter} ({d_type.replace('_', ' ').title()})",
                    vehicle_type="debris",
                    color=d_color,
                    sprite_id=f"debris_{d_type}",
                    dry_mass_kg=d_mass,
                    fuel_mass_kg=0.0,
                    cross_section_area_m2=d_area,
                    drag_coefficient=d_cd,
                    hard_body_radius_m=1.0,
                    is_active=True,
                    is_debris=True,
                    debris_type=d_type,
                    parent_collision_id=coll.collision_id,
                )
                
                fm_debris = self._build_force_model(debris_def)
                debris_prop = NumericalPropagator(
                    acceleration_fn=fm_debris.compute_acceleration,
                    integrator="rkf45",
                    dt=effective_dt,
                    atol=atol_val,
                    rtol=rtol_val,
                    mu=self.body.mu,
                )

                
                if t_coll < sim_t_end - 1.0:
                    d_hist, _, _ = debris_prop.propagate(
                        r0=r_coll,
                        v0=v_debris_0,
                        t_span=(t_coll, sim_t_end),
                        mass=d_mass,
                        fuel_mass=0.0,
                    )
                else:
                    d_hist = StateHistory()
                    d_hist.append(SimulationState(
                        time=t_coll,
                        position=r_coll.copy(),
                        velocity=v_debris_0.copy(),
                        mass=d_mass,
                    ))

                full_history_dicts: List[Dict[str, Any]] = []
                base_times = prop_histories[sc_ids[0]].times
                debris_interp = interpolator_from_state_history(d_hist)


                for t in base_times:
                    if t < t_coll:
                        full_history_dicts.append({
                            "time_seconds": float(t),
                            "position": r_coll.tolist(),
                            "velocity": [0.0, 0.0, 0.0],
                            "altitude": float(np.linalg.norm(r_coll) - self.body.radius),
                            "speed": 0.0,
                            "mass": float(d_mass),
                            "fuel_mass": 0.0,
                            "thrust_active": False,
                            "active": False,
                        })
                    else:
                        # Same Hermite interpolation as the primary tracks, so
                        # debris output is resampled at the same fidelity.
                        debris_state = debris_interp.state_at(float(t))
                        pos_t = debris_state.position
                        vel_t = debris_state.velocity


                        full_history_dicts.append({
                            "time_seconds": float(t),
                            "position": pos_t.tolist(),
                            "velocity": vel_t.tolist(),
                            "altitude": float(np.linalg.norm(pos_t) - self.body.radius),
                            "speed": float(np.linalg.norm(vel_t)),
                            "mass": float(d_mass),
                            "fuel_mass": 0.0,
                            "thrust_active": False,
                            "active": True,
                        })
                        
                debris_tracks.append(SpacecraftTrackResult(
                    definition=debris_def,
                    state_history=full_history_dicts,
                    destroyed=False,
                ))
                
            coll.debris_ids = coll_debris_ids

        # -------------------------------------------------------------
        # STEP 5: Final Result Object Construction
        # -------------------------------------------------------------
        final_objects: List[SpacecraftTrackResult] = []

        for sc in spacecraft_list:
            hist = prop_histories[sc.id]
            is_dest = sc.id in destroyed_flags
            dest_t = destroyed_flags[sc.id][0] if is_dest else None
            dest_reason = destroyed_flags[sc.id][1] if is_dest else None
            
            state_list: List[Dict[str, Any]] = []
            for st in hist.states:
                state_list.append({
                    "time_seconds": float(st.time),
                    "position": st.position.tolist(),
                    "velocity": st.velocity.tolist(),
                    "altitude": float(st.r_mag - self.body.radius),
                    "speed": float(st.speed),
                    "mass": float(st.mass),
                    "fuel_mass": float(st.fuel_mass),
                    "thrust_active": False,
                    "active": (not is_dest) or (st.time <= (dest_t or float("inf"))),
                    "destroyed": is_dest and (st.time >= (dest_t or float("inf"))),
                })
                
            final_objects.append(SpacecraftTrackResult(
                definition=sc,
                state_history=state_list,
                destroyed=is_dest,
                destruction_time_s=dest_t,
                destruction_reason=dest_reason,
                delta_v_budget=sc_delta_v_budgets.get(sc.id),
                propellant_budget=sc_propellant_budgets.get(sc.id),
                calculation_trace=sc_traces.get(sc.id),
            ))

        final_objects.extend(debris_tracks)

        summary = {
            "total_spacecraft": len([o for o in final_objects if not o.definition.is_debris]),
            "total_debris": len([o for o in final_objects if o.definition.is_debris]),
            "total_conjunctions": len(all_conjunctions),
            "total_collisions": len(all_collisions),
            "active_spacecraft_count": len([o for o in final_objects if not o.destroyed and not o.definition.is_debris]),
            "destroyed_spacecraft_count": len([o for o in final_objects if o.destroyed]),
        }

        return MultiObjectSimulationResult(
            objects=final_objects,
            conjunctions=all_conjunctions,
            collisions=all_collisions,
            time_span_s=(t_start, sim_t_end),
            central_body=self.body_name,
            calculation_steps=calc_steps,
            summary=summary,
        )
