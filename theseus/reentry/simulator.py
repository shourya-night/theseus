"""
Reentry simulator — Phase 8 of the THESEUS astrodynamics engine.

Propagates a vehicle through a planetary atmosphere using 2D planar
entry equations over a spherical, rotating planet.

Equations of Motion (2-D Planar Entry)
--------------------------------------
State vector: [r, V, γ, λ]

    dr/dt = V sin(γ)

    dV/dt = −D/m − g sin(γ)
            + ω²r cos(λ)[sin(γ)cos(λ) − cos(γ)sin(λ)]

    dγ/dt = (1/V)[ L/m − (g − V²/r)cos(γ)
            + 2ωV cos(λ)
            + ω²r cos(λ)(cos(γ)cos(λ) + sin(γ)sin(λ)) ]

    dλ/dt = V cos(γ) / r

Where:
    r     = radial distance from body centre (m)
    V     = speed relative to atmosphere (m/s)
    γ     = flight-path angle (rad), negative = descending
    λ     = latitude / range angle (rad)
    D     = ½ρV²C_D A     aerodynamic drag (N)
    L     = ½ρV²C_L A     aerodynamic lift (N)
    g     = μ/r²          gravitational acceleration (m/s²)
    ω     = body rotation rate (rad/s)
    ρ     = atmospheric density (kg/m³)

Reference: Vinh, Busemann, Culp, "Hypersonic and Planetary Entry
Flight Mechanics", 1980.

Numerical method: RKF45 adaptive integrator (reused from Phase 2).

Physical model limitations:
- Spherical, non-rotating atmosphere (rotation enters only via
  Coriolis/centripetal terms on the vehicle, atmosphere co-rotates).
- Constant C_D, C_L (no Mach/Reynolds dependence).
- US Standard Atmosphere 1976 or exponential model.
- No winds, terrain, ablation, or weather.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from theseus.reentry.vehicle import ReentryVehicle
from theseus.reentry.heating import (
    sutton_graves_convective,
    stagnation_temperature,
    mach_number,
    dynamic_pressure,
    aerodynamic_force,
    speed_of_sound,
    heating_model_metadata,
    SUTTON_GRAVES_K_EARTH,
)
from theseus.reentry.results import (
    ReentryResult,
    ReentryEvent,
    ReentryEventType,
    ReentryTelemetryPoint,
)
from theseus.atmosphere.models import AtmosphereModel, US1976StandardAtmosphere
from theseus.propagation.integrators import RKF45Integrator
from theseus.constants.physical import G0_VAL


class ReentrySimulator:
    """
    Atmospheric reentry simulator.

    Parameters
    ----------
    vehicle : ReentryVehicle
    atmosphere : AtmosphereModel
        Atmospheric density model.
    body_mu : float
        Gravitational parameter of the central body (m³/s²).
    body_radius : float
        Mean/equatorial radius of the central body (m).
    body_rotation_rate : float
        Sidereal angular rotation rate (rad/s).
    entry_interface_alt : float
        Altitude defining the entry interface (m).  Default 120 km.
    atol : float
        Absolute integrator tolerance.
    rtol : float
        Relative integrator tolerance.
    max_time : float
        Maximum propagation time (s).
    """

    def __init__(
        self,
        vehicle: ReentryVehicle,
        atmosphere: Optional[AtmosphereModel] = None,
        body_mu: float = 3.986004418e14,      # Earth
        body_radius: float = 6_378_137.0,       # Earth equatorial
        body_rotation_rate: float = 7.2921159e-5,  # Earth
        entry_interface_alt: float = 120_000.0,  # 120 km
        atol: float = 1e-10,
        rtol: float = 1e-10,
        max_time: float = 7200.0,               # 2 hours
    ) -> None:
        self.vehicle = vehicle
        self.atmosphere = atmosphere or US1976StandardAtmosphere()
        self.mu = body_mu
        self.R = body_radius
        self.omega = body_rotation_rate
        self.entry_interface_alt = entry_interface_alt
        self.atol = atol
        self.rtol = rtol
        self.max_time = max_time

    def simulate(
        self,
        entry_altitude: float,
        entry_velocity: float,
        entry_fpa_deg: float,
        entry_latitude_deg: float = 0.0,
        output_every_n: int = 1,
    ) -> ReentryResult:
        """
        Run reentry simulation.

        Parameters
        ----------
        entry_altitude : float
            Entry altitude above surface (m).
        entry_velocity : float
            Entry speed relative to atmosphere (m/s).
        entry_fpa_deg : float
            Entry flight-path angle (deg).  Negative = descending.
        entry_latitude_deg : float
            Initial latitude (deg).  Default 0 (equatorial).
        output_every_n : int
            Record telemetry every N integration steps.

        Returns
        -------
        ReentryResult
        """
        # --- initial state ---
        r0 = self.R + entry_altitude
        V0 = entry_velocity
        gamma0 = math.radians(entry_fpa_deg)
        lam0 = math.radians(entry_latitude_deg)

        y0 = np.array([r0, V0, gamma0, lam0], dtype=np.float64)

        # --- model constants ---
        m = self.vehicle.mass
        Cd = self.vehicle.cd
        Cl = self.vehicle.cl
        A = self.vehicle.reference_area
        r_n = self.vehicle.nose_radius
        mu = self.mu
        R = self.R
        omega = self.omega

        # --- derivative function ---
        def deriv(t: float, y: np.ndarray) -> np.ndarray:
            r = y[0]
            V = max(y[1], 1.0)     # avoid division by zero
            gamma = y[2]
            lam = y[3]

            alt = r - R
            if alt < 0:
                alt = 0.0

            # Atmosphere
            rho = self.atmosphere.density(alt)

            # Gravity
            g = mu / (r * r)

            # Aerodynamic forces
            D = 0.5 * rho * V * V * Cd * A
            L = 0.5 * rho * V * V * Cl * A

            sin_g = math.sin(gamma)
            cos_g = math.cos(gamma)
            sin_l = math.sin(lam)
            cos_l = math.cos(lam)

            # dr/dt
            dr = V * sin_g

            # dV/dt
            dV = (-D / m
                  - g * sin_g
                  + omega * omega * r * cos_l * (sin_g * cos_l - cos_g * sin_l))

            # dγ/dt
            dgamma = (1.0 / V) * (
                L / m
                - (g - V * V / r) * cos_g
                + 2.0 * omega * V * cos_l
                + omega * omega * r * cos_l * (cos_g * cos_l + sin_g * sin_l)
            )

            # dλ/dt
            dlam = V * cos_g / r

            return np.array([dr, dV, dgamma, dlam])

        # --- integrate with RKF45 ---
        integrator = RKF45Integrator(
            atol=self.atol,
            rtol=self.rtol,
            dt_initial=0.1,
            dt_min=1e-6,
            dt_max=10.0,
            max_steps=5_000_000,
        )

        # Manual step-by-step integration for event detection
        t = 0.0
        y = y0.copy()
        h = 0.1

        telemetry: list[ReentryTelemetryPoint] = []
        events: list[ReentryEvent] = []
        calc_steps: list[dict] = []

        # Tracking variables for event detection
        prev_q_dot = 0.0       # rate of change of dynamic pressure
        prev_qdot_heat = 0.0   # rate of change of heating rate
        prev_a_dot = 0.0       # rate of change of acceleration magnitude
        prev_mach = 999.0      # previous Mach number
        prev_alt = entry_altitude
        prev_vel = entry_velocity
        prev_dq = 0.0
        prev_qheat = 0.0
        prev_acc = 0.0
        cumulative_heat = 0.0
        step_count = 0
        entered = False
        peak_q_found = False
        peak_heat_found = False
        peak_g_found = False
        subsonic_found = False
        termination = "max_time"
        entry_event_logged = False

        # Record initial state
        def make_telemetry(t_: float, y_: np.ndarray, cum_heat: float) -> ReentryTelemetryPoint:
            r_ = y_[0]
            V_ = max(y_[1], 0.0)
            gamma_ = y_[2]
            lam_ = y_[3]
            alt_ = max(r_ - R, 0.0)
            atm = self.atmosphere.get_properties(alt_)
            M = mach_number(V_, atm.temperature)
            q = dynamic_pressure(atm.density, V_)
            D_ = aerodynamic_force(atm.density, V_, Cd, A)
            L_ = aerodynamic_force(atm.density, V_, Cl, A)
            g_ = mu / (r_ * r_)
            acc_drag = D_ / m
            acc_lift = L_ / m
            # Total aero deceleration magnitude (drag dominates)
            acc_total = math.sqrt(acc_drag**2 + acc_lift**2 + (g_ * math.sin(gamma_))**2)
            # Approximate: use drag + gravity projection
            acc_felt = acc_drag  # primary sensed deceleration
            g_load = acc_total / G0_VAL
            q_heat = sutton_graves_convective(atm.density, V_, r_n)
            T_stag = stagnation_temperature(atm.temperature, M)
            downrange = lam_ * R

            return ReentryTelemetryPoint(
                time=t_,
                altitude=alt_,
                velocity=V_,
                flight_path_angle=gamma_,
                downrange=downrange,
                latitude=lam_,
                density=atm.density,
                pressure=atm.pressure,
                temperature=atm.temperature,
                mach=M,
                dynamic_pressure=q,
                drag=D_,
                lift=L_,
                acceleration_mag=acc_total,
                g_load=g_load,
                heating_rate=q_heat,
                cumulative_heat=cum_heat,
                stagnation_temp=T_stag,
            )

        def make_state_dict(tp: ReentryTelemetryPoint) -> dict:
            return tp.to_dict()

        # Entry interface event
        if entry_altitude >= self.entry_interface_alt:
            entry_event_logged = True

        # Generate initial calculation trace
        init_tp = make_telemetry(0.0, y0, 0.0)
        calc_steps.append({
            "stepIndex": 1,
            "phase": "PHASE_08",
            "title": "Acquire Atmospheric State",
            "status": "completed",
            "equation": "ρ(h), T(h), P(h)",
            "substitutions": {
                "altitude_km": init_tp.altitude / 1e3,
                "density_kg_m3": init_tp.density,
                "temperature_K": init_tp.temperature,
                "pressure_Pa": init_tp.pressure,
            },
            "result": f"ρ = {init_tp.density:.6e} kg/m³",
            "units": "kg/m³, K, Pa",
            "explanation": "Atmospheric state at entry altitude from US Standard Atmosphere 1976.",
            "beginnerExplanation": "We look up how dense, hot, and pressurised the air is at this height.",
            "modelName": "US Standard Atmosphere 1976",
            "source": "NOAA/NASA/USAF",
            "validityRange": "0–86 km geopotential altitude",
            "limitations": "No solar-activity variation; no weather; exponential extrapolation above 86 km",
        })
        calc_steps.append({
            "stepIndex": 2,
            "phase": "PHASE_08",
            "title": "Calculate Mach Number",
            "status": "completed",
            "equation": "M = V / a,  a = √(γRT/M_air)",
            "substitutions": {
                "V_m_s": init_tp.velocity,
                "a_m_s": speed_of_sound(init_tp.temperature),
                "T_K": init_tp.temperature,
            },
            "result": f"M = {init_tp.mach:.2f}",
            "units": "dimensionless",
            "explanation": "Ratio of vehicle speed to local speed of sound.",
            "beginnerExplanation": "How many times faster than sound the vehicle is moving.",
        })
        calc_steps.append({
            "stepIndex": 3,
            "phase": "PHASE_08",
            "title": "Calculate Dynamic Pressure",
            "status": "completed",
            "equation": "q = ½ρV²",
            "substitutions": {
                "rho_kg_m3": init_tp.density,
                "V_m_s": init_tp.velocity,
            },
            "result": f"q = {init_tp.dynamic_pressure:.2f} Pa",
            "units": "Pa",
            "explanation": "Dynamic pressure represents the aerodynamic loading from the vehicle's motion through the atmosphere.",
            "beginnerExplanation": "How strongly the air is 'hitting' the spacecraft.",
        })
        calc_steps.append({
            "stepIndex": 4,
            "phase": "PHASE_08",
            "title": "Calculate Drag Force",
            "status": "completed",
            "equation": "D = ½ρV²C_D A",
            "substitutions": {
                "rho_kg_m3": init_tp.density,
                "V_m_s": init_tp.velocity,
                "Cd": Cd,
                "A_m2": A,
            },
            "result": f"D = {init_tp.drag:.2f} N",
            "units": "N",
            "explanation": "Aerodynamic drag force opposing the vehicle's motion.",
            "beginnerExplanation": "The air resistance slowing the spacecraft down.",
        })
        calc_steps.append({
            "stepIndex": 5,
            "phase": "PHASE_08",
            "title": "Calculate Lift Force",
            "status": "completed",
            "equation": "L = ½ρV²C_L A",
            "substitutions": {
                "rho_kg_m3": init_tp.density,
                "V_m_s": init_tp.velocity,
                "Cl": Cl,
                "A_m2": A,
            },
            "result": f"L = {init_tp.lift:.2f} N",
            "units": "N",
            "explanation": "Aerodynamic lift force perpendicular to the velocity.",
            "beginnerExplanation": "The force pushing the spacecraft upward (like an airplane wing).",
        })
        calc_steps.append({
            "stepIndex": 6,
            "phase": "PHASE_08",
            "title": "Calculate Ballistic Coefficient",
            "status": "completed",
            "equation": "β = m / (C_D A)",
            "substitutions": {
                "m_kg": m,
                "Cd": Cd,
                "A_m2": A,
            },
            "result": f"β = {self.vehicle.ballistic_coefficient:.2f} kg/m²",
            "units": "kg/m²",
            "explanation": "A higher ballistic coefficient means the vehicle penetrates deeper into the atmosphere before decelerating.",
            "beginnerExplanation": "Tells us whether the spacecraft is 'heavy and sleek' (high β, slow to brake) or 'light and blunt' (low β, brakes quickly).",
        })
        calc_steps.append({
            "stepIndex": 7,
            "phase": "PHASE_08",
            "title": "Entry Equations of Motion",
            "status": "completed",
            "equation": "dr/dt = V sin(γ)\ndV/dt = −D/m − g sin(γ) + ω²r cos(λ)...\ndγ/dt = (1/V)[L/m − (g−V²/r)cos(γ) + 2ωV cos(λ)...]\ndλ/dt = V cos(γ)/r",
            "substitutions": {
                "entry_altitude_km": entry_altitude / 1e3,
                "entry_velocity_km_s": entry_velocity / 1e3,
                "entry_fpa_deg": entry_fpa_deg,
                "integrator": "RKF45",
                "atol": self.atol,
                "rtol": self.rtol,
            },
            "result": "Propagating...",
            "units": "m, m/s, rad, rad",
            "explanation": "2D planar entry equations over a spherical rotating planet. Integrating with RKF45 adaptive step control.",
            "beginnerExplanation": "We calculate the spacecraft's path through the atmosphere step-by-step, accounting for drag, lift, gravity, and Earth's rotation.",
            "scientificNotes": "Reference: Vinh, Busemann, Culp (1980). Spherical non-rotating atmosphere.",
        })
        calc_steps.append({
            "stepIndex": 8,
            "phase": "PHASE_08",
            "title": "Convective Heating (Sutton-Graves)",
            "status": "completed",
            "equation": "q̇ = k √(ρ/r_n) V³",
            "substitutions": {
                "k": SUTTON_GRAVES_K_EARTH,
                "k_units": "kg^0.5/m",
                "rho_kg_m3": init_tp.density,
                "r_n_m": r_n,
                "V_m_s": init_tp.velocity,
            },
            "result": f"q̇ = {init_tp.heating_rate:.2f} W/m²  ({init_tp.heating_rate/1e3:.2f} kW/m²)",
            "units": "W/m²",
            "explanation": "Stagnation-point convective heating rate. ENGINEERING ESTIMATE — NOT CFD.",
            "beginnerExplanation": "Estimates how much heat the nose of the spacecraft feels due to air friction. A blunter nose (larger r_n) spreads the heat over a wider area.",
            "modelName": "Sutton-Graves stagnation-point correlation",
            "source": "Sutton & Graves, NASA TR R-376 (1971)",
            "validityRange": "Continuum hypersonic flow (Kn << 1)",
            "limitations": "No catalytic-wall effects; no radiative heating; assumes equilibrium chemistry",
            "assumptions": ["constant Cd/Cl", "equilibrium boundary layer", "no ablation"],
        })
        calc_steps.append({
            "stepIndex": 9,
            "phase": "PHASE_08",
            "title": "Cumulative Heat Load",
            "status": "completed",
            "equation": "Q = ∫q̇ dt",
            "result": "Q = 0.00 J/m² (initial)",
            "units": "J/m²",
            "explanation": "Running integral of heat flux over time. Total thermal energy deposited per unit area.",
            "beginnerExplanation": "Total heat energy absorbed by the heat shield so far.",
        })
        calc_steps.append({
            "stepIndex": 10,
            "phase": "PHASE_08",
            "title": "G-Loading",
            "status": "completed",
            "equation": "n = |a| / g₀",
            "substitutions": {
                "a_m_s2": init_tp.acceleration_mag,
                "g0_m_s2": G0_VAL,
            },
            "result": f"n = {init_tp.g_load:.2f} g",
            "units": "g (multiples of standard gravity)",
            "explanation": "Deceleration experienced by the vehicle in units of Earth surface gravity.",
            "beginnerExplanation": "How many 'g's the astronauts feel. Normal standing on Earth = 1g. Apollo reentry peak was about 6-7g.",
        })
        calc_steps.append({
            "stepIndex": 11,
            "phase": "PHASE_08",
            "title": "Event Detection",
            "status": "in_progress",
            "equation": "dq/dt sign change → Peak Q\ndq̇/dt sign change → Peak Heating\nd|a|/dt sign change → Peak G\nh ≤ 0 → Impact\nM < 1 → Subsonic",
            "result": "Monitoring...",
            "units": "various",
            "explanation": "Events are detected from actual propagated state history by monitoring sign changes in rate-of-change quantities.",
            "beginnerExplanation": "We watch for critical moments: when air pressure peaks, when heating peaks, when deceleration peaks, and when the spacecraft hits the ground or bounces back to space.",
        })

        # === PROPAGATION LOOP ===
        while t < self.max_time:
            # Current state
            r_curr = y[0]
            V_curr = max(y[1], 0.0)
            gamma_curr = y[2]
            lam_curr = y[3]
            alt_curr = r_curr - R

            # Termination: ground impact
            if alt_curr <= 0 and t > 0:
                termination = "ground_impact"
                tp = make_telemetry(t, y, cumulative_heat)
                telemetry.append(tp)
                events.append(ReentryEvent(
                    event_type=ReentryEventType.GROUND_IMPACT,
                    time=t,
                    altitude=0.0,
                    velocity=V_curr,
                    value=0.0,
                    units="m",
                    detection_method="altitude ≤ 0",
                    state=make_state_dict(tp),
                ))
                break

            # Termination: skip-out
            if (t > 1.0
                    and alt_curr > self.entry_interface_alt
                    and gamma_curr > 0
                    and entered):
                termination = "skip_out"
                tp = make_telemetry(t, y, cumulative_heat)
                telemetry.append(tp)
                events.append(ReentryEvent(
                    event_type=ReentryEventType.SKIP_OUT,
                    time=t,
                    altitude=alt_curr,
                    velocity=V_curr,
                    value=alt_curr,
                    units="m",
                    detection_method="altitude > entry interface, γ > 0, after initial entry",
                    state=make_state_dict(tp),
                ))
                break

            # Compute current telemetry
            tp = make_telemetry(t, y, cumulative_heat)

            # Entry interface event
            if not entry_event_logged and alt_curr <= self.entry_interface_alt:
                entry_event_logged = True
                entered = True
                events.append(ReentryEvent(
                    event_type=ReentryEventType.ENTRY_INTERFACE,
                    time=t,
                    altitude=alt_curr,
                    velocity=V_curr,
                    value=alt_curr,
                    units="m",
                    detection_method=f"altitude crossed {self.entry_interface_alt/1e3:.0f} km threshold",
                    state=make_state_dict(tp),
                ))

            if alt_curr <= self.entry_interface_alt:
                entered = True

            # Event detection by sign-change
            curr_q = tp.dynamic_pressure
            curr_qheat = tp.heating_rate
            curr_acc = tp.acceleration_mag
            curr_mach = tp.mach

            if step_count > 1 and entered:
                # Peak dynamic pressure
                dq = curr_q - prev_dq
                if not peak_q_found and prev_q_dot > 0 and dq < 0 and curr_q > 100:
                    peak_q_found = True
                    events.append(ReentryEvent(
                        event_type=ReentryEventType.PEAK_DYNAMIC_PRESSURE,
                        time=t,
                        altitude=alt_curr,
                        velocity=V_curr,
                        value=curr_q,
                        units="Pa",
                        detection_method="sign change in dq/dt (increasing → decreasing)",
                        state=make_state_dict(tp),
                    ))
                prev_q_dot = dq

                # Peak heating
                dqheat = curr_qheat - prev_qheat
                if not peak_heat_found and prev_qdot_heat > 0 and dqheat < 0 and curr_qheat > 100:
                    peak_heat_found = True
                    events.append(ReentryEvent(
                        event_type=ReentryEventType.PEAK_HEATING,
                        time=t,
                        altitude=alt_curr,
                        velocity=V_curr,
                        value=curr_qheat,
                        units="W/m²",
                        detection_method="sign change in dq̇/dt (increasing → decreasing)",
                        state=make_state_dict(tp),
                    ))
                prev_qdot_heat = dqheat

                # Peak deceleration
                da = curr_acc - prev_acc
                if not peak_g_found and prev_a_dot > 0 and da < 0 and tp.g_load > 0.5:
                    peak_g_found = True
                    events.append(ReentryEvent(
                        event_type=ReentryEventType.PEAK_DECELERATION,
                        time=t,
                        altitude=alt_curr,
                        velocity=V_curr,
                        value=tp.g_load,
                        units="g",
                        detection_method="sign change in d|a|/dt (increasing → decreasing)",
                        state=make_state_dict(tp),
                    ))
                prev_a_dot = da

                # Subsonic transition
                if not subsonic_found and prev_mach > 1.0 and curr_mach <= 1.0:
                    subsonic_found = True
                    events.append(ReentryEvent(
                        event_type=ReentryEventType.SUBSONIC_TRANSITION,
                        time=t,
                        altitude=alt_curr,
                        velocity=V_curr,
                        value=curr_mach,
                        units="Mach",
                        detection_method="Mach crossed M = 1 (supersonic → subsonic)",
                        state=make_state_dict(tp),
                    ))

            prev_dq = curr_q
            prev_qheat = curr_qheat
            prev_acc = curr_acc
            prev_mach = curr_mach

            # Record telemetry
            if step_count % output_every_n == 0:
                telemetry.append(tp)

            # --- RKF45 step ---
            h_step = min(h, self.max_time - t)
            h_step = max(h_step, 1e-6)

            k1 = deriv(t, y)
            k2 = deriv(t + 0.25 * h_step, y + 0.25 * h_step * k1)
            k3 = deriv(t + 3.0/8.0 * h_step,
                       y + h_step * (3.0/32.0 * k1 + 9.0/32.0 * k2))
            k4 = deriv(t + 12.0/13.0 * h_step,
                       y + h_step * (1932.0/2197.0 * k1 - 7200.0/2197.0 * k2 + 7296.0/2197.0 * k3))
            k5 = deriv(t + h_step,
                       y + h_step * (439.0/216.0 * k1 - 8.0 * k2 + 3680.0/513.0 * k3 - 845.0/4104.0 * k4))
            k6 = deriv(t + 0.5 * h_step,
                       y + h_step * (-8.0/27.0 * k1 + 2.0 * k2 - 3544.0/2565.0 * k3
                                     + 1859.0/4104.0 * k4 - 11.0/40.0 * k5))

            # 4th and 5th order solutions
            y4 = y + h_step * (25.0/216.0 * k1 + 1408.0/2565.0 * k3
                               + 2197.0/4104.0 * k4 - 1.0/5.0 * k5)
            y5 = y + h_step * (16.0/135.0 * k1 + 6656.0/12825.0 * k3
                               + 28561.0/56430.0 * k4 - 9.0/50.0 * k5 + 2.0/55.0 * k6)

            # Error estimate
            err_vec = y5 - y4
            scale = self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y5))
            err = float(np.sqrt(np.mean((err_vec / scale) ** 2)))

            if err <= 1.0 or h_step <= 1e-6:
                # Accept step
                dt_step = h_step

                # Update cumulative heat
                new_tp = make_telemetry(t + dt_step, y5, cumulative_heat)
                cumulative_heat += 0.5 * (tp.heating_rate + new_tp.heating_rate) * dt_step

                t += dt_step
                y = y5
                step_count += 1

            # Step size control
            if err > 1e-30:
                h_new = h_step * 0.84 * (1.0 / err) ** 0.2
            else:
                h_new = h_step * 5.0
            h = max(1e-6, min(h_new, 10.0))

        # === POST-PROCESSING ===

        # Peak statistics
        peak_stats = {}
        if telemetry:
            max_q = max(telemetry, key=lambda tp: tp.dynamic_pressure)
            max_heat = max(telemetry, key=lambda tp: tp.heating_rate)
            max_g = max(telemetry, key=lambda tp: tp.g_load)
            peak_stats = {
                "peak_dynamic_pressure_Pa": max_q.dynamic_pressure,
                "peak_dynamic_pressure_kPa": max_q.dynamic_pressure / 1e3,
                "peak_q_altitude_km": max_q.altitude / 1e3,
                "peak_q_time_s": max_q.time,
                "peak_heating_rate_W_m2": max_heat.heating_rate,
                "peak_heating_rate_kW_m2": max_heat.heating_rate / 1e3,
                "peak_heating_altitude_km": max_heat.altitude / 1e3,
                "peak_heating_time_s": max_heat.time,
                "peak_g_load": max_g.g_load,
                "peak_g_altitude_km": max_g.altitude / 1e3,
                "peak_g_time_s": max_g.time,
                "total_heat_load_MJ_m2": cumulative_heat / 1e6,
            }

        # Impact conditions
        impact_cond = None
        if termination == "ground_impact" and telemetry:
            last = telemetry[-1]
            impact_cond = {
                "impact_velocity_m_s": last.velocity,
                "impact_velocity_km_s": last.velocity / 1e3,
                "impact_flight_path_angle_deg": last.flight_path_angle * 57.29577951308232,
                "impact_time_s": last.time,
                "downrange_km": last.downrange / 1e3,
            }

        # Update event-detection calc step
        event_names = [e.event_type.name for e in events]
        for cs in calc_steps:
            if cs.get("title") == "Event Detection":
                cs["status"] = "completed"
                cs["result"] = f"Detected {len(events)} events: {', '.join(event_names)}"

        # Update EOM step
        for cs in calc_steps:
            if cs.get("title") == "Entry Equations of Motion":
                cs["result"] = (
                    f"Propagated {step_count} steps over {t:.1f} s. "
                    f"Termination: {termination}"
                )

        # Model metadata
        model_meta = {
            "numerical": {
                "integrator": "RKF45 (Runge-Kutta-Fehlberg 4(5))",
                "atol": self.atol,
                "rtol": self.rtol,
                "dt_initial": 0.1,
                "dt_min": 1e-6,
                "dt_max": 10.0,
                "steps_taken": step_count,
                "note": "Numerical tolerance controls integration error, NOT real-world trajectory accuracy",
            },
            "physical": {
                "atmosphere": "US Standard Atmosphere 1976",
                "gravity": "Spherical (μ/r²)",
                "planet": "Spherical rotating Earth",
                "drag_model": "Constant Cd",
                "lift_model": "Constant Cl",
                "equations": "2D planar entry (Vinh, Busemann, Culp 1980)",
            },
            "heating": heating_model_metadata(),
            "limitations": [
                "No winds",
                "No terrain (spherical surface)",
                "No ablation or mass loss",
                "No atmospheric weather variation",
                "No Mach-dependent Cd/Cl",
                "No attitude dynamics",
                "Radiative heating not enabled",
                "2D planar (no cross-range)",
            ],
            "assumptions": [
                "Constant drag and lift coefficients",
                "Co-rotating atmosphere",
                "Spherical gravitational field (no J2)",
                "US Standard Atmosphere 1976 density profile",
            ],
        }

        return ReentryResult(
            telemetry=telemetry,
            events=events,
            termination_reason=termination,
            vehicle=self.vehicle.to_dict(),
            model_metadata=model_meta,
            peak_statistics=peak_stats,
            impact_conditions=impact_cond,
            calculation_steps=calc_steps,
        )
