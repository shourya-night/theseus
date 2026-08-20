export type ComplexityMode = "BASIC" | "ADVANCED" | "EXPERT";

export interface CelestialBodyInfo {
  name: string;
  mu: number; // m^3/s^2
  radius_km: number;
  mass_kg: number;
  j2?: number;
  j3?: number;
  parent: string | null;
  has_atmosphere: boolean;
  color: string;
  rotation_period_s?: number;
  axial_tilt_rad?: number;
  position_km?: [number, number, number];
  texture_style?: string;
  orbit_radius_km?: number;
  orbit_period_days?: number;
}

export interface RocketPreset {
  id: string;
  name: string;
  operator: string;
  category: string;
  stages: number;
  dry_mass_kg: number;
  propellant_mass_kg: number;
  payload_leo_kg?: number;
  payload_gto_kg?: number;
  max_thrust_n: number;
  specific_impulse_s: number;
  cross_section_area_m2: number;
  drag_coefficient: number;
  reflectivity_coefficient: number;
  confidence: string;
  citation: string;
  description: string;
  sprite_id: string;
}

export interface StateVector {
  time_seconds: number;
  position: [number, number, number]; // [x, y, z] in meters
  velocity: [number, number, number]; // [vx, vy, vz] in m/s
  altitude: number; // meters above mean surface
  speed: number; // m/s
  mass: number; // kg
  fuel_mass: number; // kg
  thrust_active: boolean;
  distance_to_target_m?: number;
}

export interface CalculationIteration {
  iteration: number;
  z: number;
  y?: number;
  tof_calculated_s: number;
  residual: number;
  status: string;
}

export type ScientificScalarOrStructured =
  | string
  | number
  | boolean
  | null
  | undefined
  | number[]
  | string[]
  | Record<string, any>;

export interface CalculationStep {
  stepIndex: number;
  phase: string;
  title: string;
  status: "ACQUIRED" | "CALCULATED" | "CONVERGED" | "COMPLETE" | string;
  equation: string;
  substitutions?: Record<string, ScientificScalarOrStructured>;
  intermediateCalculation?: (string | Record<string, any>)[];
  result: ScientificScalarOrStructured;
  units?: string;
  explanation?: string;
  beginnerExplanation?: string;
  scientificNotes?: string;
  iterations?: CalculationIteration[];
}

export interface MissionEvent {
  time: number;
  name: string;
  type: "MANEUVER_START" | "MANEUVER_END" | "WAYPOINT" | "STATE_CHANGE" | "MISSION_SUCCESS" | "MISSION_FAILURE" | "COLLISION" | "CONJUNCTION" | string;
  details: string;
}

export interface DeltaVBudget {
  delta_v1?: number;
  delta_v2?: number;
  total_delta_v: number;
  available_delta_v: number;
  margin_delta_v: number;
  relative_velocity_at_arrival?: number;
}

export interface PropellantBudget {
  initial_total_mass_kg: number;
  dry_mass_kg: number;
  initial_fuel_kg: number;
  fuel_consumed_kg: number;
  fuel_margin_kg: number;
}

export interface SimulationMetadata {
  name: string;
  origin?: string;
  destination?: string;
  central_body: string;
  transfer_type?: string;
  trajectory_type?: string;
  plane_change_deg?: number;
  duration_hours?: number;
  status: "SUCCESS" | "WARNING_INSUFFICIENT_FUEL" | "FAILED" | "SOLVER_FAILED";
}

export interface SimulationResult {
  mission_id: string;
  metadata: SimulationMetadata;
  delta_v_budget: DeltaVBudget;
  propellant_budget: PropellantBudget;
  state_history: StateVector[];
  chaser_state_history?: StateVector[];
  target_state_history?: StateVector[];
  calculation_trace: CalculationStep[];
  events: MissionEvent[];
  diagnostics: {
    solver: string;
    numerical_tolerance: string;
    endpoint_miss_distance_m?: number;
    iterations_count?: number;
    energy_drift_relative?: number;
    scientific_honesty_note: string;
  };
}

// ---------------------------------------------------------------------------
// Multi-Spacecraft & Collision Environment Types (Phases 9 & 10 Integration)
// ---------------------------------------------------------------------------

export interface SpacecraftConfig {
  id: string;
  name: string;
  vehicle_type: string;
  color: string;
  sprite_id: string;
  origin?: string;
  destination?: string;
  payload_mass_kg?: number;
  tof_days?: number;
  departure_epoch_date?: string;
  dry_mass_kg: number;
  fuel_mass_kg: number;
  cross_section_area_m2: number;
  drag_coefficient: number;
  reflectivity_coefficient: number;
  thrust_n: number;
  specific_impulse_s: number;
  central_body: string;
  initial_r_m?: [number, number, number];
  initial_v_m_s?: [number, number, number];
  semi_major_axis_km?: number;
  eccentricity?: number;
  inclination_deg?: number;
  raan_deg?: number;
  arg_periapsis_deg?: number;
  true_anomaly_deg?: number;
  hard_body_radius_m: number;
  sigma_pos_m?: [number, number, number];
  sigma_vel_m_s?: [number, number, number];
}

export interface SpacecraftTrackState {
  time_seconds: number;
  position: [number, number, number];
  velocity: [number, number, number];
  altitude: number;
  speed: number;
  mass: number;
  fuel_mass: number;
  thrust_active: boolean;
  active: boolean;
  destroyed?: boolean;
}

export interface SpacecraftTrack {
  id: string;
  name: string;
  vehicle_type: string;
  color: string;
  sprite_id: string;
  origin?: string;
  destination?: string;
  is_debris: boolean;
  debris_type?: string;
  parent_collision_id?: string;
  hard_body_radius_m: number;
  destroyed: boolean;
  destruction_time_s?: number;
  destruction_reason?: string;
  delta_v_budget?: DeltaVBudget;
  propellant_budget?: PropellantBudget;
  calculation_trace?: CalculationStep[];
  state_history: SpacecraftTrackState[];
}


export interface MultiConjunctionEvent {
  event_id: string;
  spacecraft_a_id: string;
  spacecraft_b_id: string;
  spacecraft_a_name: string;
  spacecraft_b_name: string;
  tca_s: number;
  miss_distance_m: number;
  miss_distance_km: number;
  relative_velocity_m_s: number;
  relative_velocity_km_s: number;
  encounter_angle_deg: number;
  encounter_type: string;
  r_rel_m: [number, number, number];
  v_rel_m_s: [number, number, number];
  b_plane_b_t_m?: number;
  b_plane_b_r_m?: number;
  b_plane_b_t_km?: number;
  b_plane_b_r_km?: number;
  b_plane_sigma_major_m?: number;
  b_plane_sigma_minor_m?: number;
  b_plane_sigma_major_km?: number;
  b_plane_sigma_minor_km?: number;
  b_plane_ellipse_angle_deg?: number;
  b_plane_covariance_m2?: number[][];
  hard_body_radius_m: number;
  hard_body_radius_km: number;
  collision_probability?: number;
  collision_probability_scientific?: string;
  risk_level: "LOW" | "ELEVATED" | "HIGH" | "CRITICAL" | string;
  action_required: boolean;
  is_physical_collision: boolean;
}

export interface PhysicalCollisionEvent {
  collision_id: string;
  time_s: number;
  spacecraft_a_id: string;
  spacecraft_b_id: string;
  spacecraft_a_name: string;
  spacecraft_b_name: string;
  collision_position_m: [number, number, number];
  collision_position_km: [number, number, number];
  relative_velocity_m_s: number;
  relative_velocity_km_s: number;
  miss_distance_m: number;
  combined_hbr_m: number;
  debris_ids: string[];
}

export interface MultiSimulationResult {
  objects: SpacecraftTrack[];
  conjunctions: MultiConjunctionEvent[];
  collisions: PhysicalCollisionEvent[];
  time_span_s: [number, number];
  central_body: string;
  calculation_steps: CalculationStep[];
  summary: {
    total_spacecraft: number;
    total_debris: number;
    total_conjunctions: number;
    total_collisions: number;
    active_spacecraft_count: number;
    destroyed_spacecraft_count: number;
  };
}

