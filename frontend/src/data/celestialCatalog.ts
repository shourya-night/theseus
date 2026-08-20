import { CelestialBodyInfo } from "../types/mission";

// Standard Astronomical Unit in kilometers and meters
export const AU_KM = 149597870.7;
export const AU_METERS = AU_KM * 1000.0;

export interface KeplerianElements {
  a_km: number;       // Semi-major axis in km
  e: number;          // Eccentricity
  inc_deg: number;    // Inclination in degrees
  w_deg: number;      // Longitude of perihelion (varpi = omega + Omega) in degrees
  period_days: number;// Orbital period in tropical days
  m0_deg: number;     // Mean anomaly at J2000 epoch in degrees
}

export const PLANETARY_ORBITAL_ELEMENTS: Record<string, KeplerianElements> = {
  mercury: {
    a_km: 0.38709893 * AU_KM,
    e: 0.20563069,
    inc_deg: 7.00487,
    w_deg: 77.45645,
    period_days: 87.9691,
    m0_deg: 174.7947,
  },
  venus: {
    a_km: 0.72333199 * AU_KM,
    e: 0.00677323,
    inc_deg: 3.39471,
    w_deg: 131.53298,
    period_days: 224.701,
    m0_deg: 50.115,
  },
  earth: {
    a_km: 1.00000011 * AU_KM,
    e: 0.01671022,
    inc_deg: 0.00005,
    w_deg: 102.94719,
    period_days: 365.25636,
    m0_deg: 358.617,
  },
  mars: {
    a_km: 1.52366231 * AU_KM,
    e: 0.09341233,
    inc_deg: 1.85061,
    w_deg: 336.04084,
    period_days: 686.971,
    m0_deg: 19.373,
  },
  jupiter: {
    a_km: 5.20336301 * AU_KM,
    e: 0.04839266,
    inc_deg: 1.30530,
    w_deg: 14.75385,
    period_days: 4332.59,
    m0_deg: 20.020,
  },
  saturn: {
    a_km: 9.53707032 * AU_KM,
    e: 0.05415060,
    inc_deg: 2.48446,
    w_deg: 92.43194,
    period_days: 10759.22,
    m0_deg: 317.020,
  },
  uranus: {
    a_km: 19.19126393 * AU_KM,
    e: 0.04716771,
    inc_deg: 0.76986,
    w_deg: 170.96424,
    period_days: 30685.4,
    m0_deg: 142.2386,
  },
  neptune: {
    a_km: 30.06896348 * AU_KM,
    e: 0.00858587,
    inc_deg: 1.76917,
    w_deg: 44.97135,
    period_days: 60189.0,
    m0_deg: 256.228,
  },
};

/**
 * Solve Kepler's Equation M = E - e*sin(E) using Newton-Raphson solver
 */
export function solveKeplerEquation(M_rad: number, e: number): number {
  // Normalize M into [-pi, pi]
  let M = M_rad % (2 * Math.PI);
  if (M < 0) M += 2 * Math.PI;

  let E = M + e * Math.sin(M); // Initial guess
  for (let iter = 0; iter < 15; iter++) {
    const dE = (E - e * Math.sin(E) - M) / (1.0 - e * Math.cos(E));
    E -= dE;
    if (Math.abs(dE) < 1e-11) break;
  }
  return E;
}

/**
 * Calculate accurate heliocentric position (x, y in SI meters) and true orbital path for any planet at elapsed seconds t
 */
export function getPlanetStateAtTime(
  planetKey: string,
  timeElapsedSec: number = 0
): {
  positionM: [number, number, number];
  orbitalPathM: [number, number][]; // Elliptical orbit points
  distanceAU: number;
  trueAnomalyDeg: number;
} {
  const elem = PLANETARY_ORBITAL_ELEMENTS[planetKey.toLowerCase()];
  if (!elem) {
    return {
      positionM: [0, 0, 0],
      orbitalPathM: [],
      distanceAU: 0,
      trueAnomalyDeg: 0,
    };
  }

  const a_m = elem.a_km * 1000.0;
  const e = elem.e;
  const w_rad = (elem.w_deg * Math.PI) / 180.0;
  const T_sec = elem.period_days * 86400.0;
  const m0_rad = (elem.m0_deg * Math.PI) / 180.0;

  // 1. Mean anomaly at time t
  const n = (2.0 * Math.PI) / T_sec; // Mean motion (rad/s)
  const M = m0_rad + n * timeElapsedSec;

  // 2. Eccentric Anomaly E
  const E = solveKeplerEquation(M, e);

  // 3. True Anomaly nu
  const nu = 2.0 * Math.atan2(Math.sqrt(1.0 + e) * Math.sin(E / 2.0), Math.sqrt(1.0 - e) * Math.cos(E / 2.0));

  // 4. Heliocentric radius r
  const r = a_m * (1.0 - e * Math.cos(E));

  // 5. Heliocentric 2D coordinates in ecliptic plane (rotated by longitude of perihelion)
  const theta = nu + w_rad;
  const x = r * Math.cos(theta);
  const y = r * Math.sin(theta);

  // 6. Pre-generate accurate elliptical orbital trajectory (Sun at (0, 0) focus)
  const orbitalPathM: [number, number][] = [];
  const nPoints = 120;
  for (let i = 0; i <= nPoints; i++) {
    const f_nu = (i / nPoints) * 2.0 * Math.PI;
    const f_r = (a_m * (1.0 - e * e)) / (1.0 + e * Math.cos(f_nu));
    const f_th = f_nu + w_rad;
    orbitalPathM.push([f_r * Math.cos(f_th), f_r * Math.sin(f_th)]);
  }

  return {
    positionM: [x, y, 0.0],
    orbitalPathM,
    distanceAU: r / AU_METERS,
    trueAnomalyDeg: ((nu * 180.0) / Math.PI + 360.0) % 360.0,
  };
}

export const CELESTIAL_BODIES: Record<string, CelestialBodyInfo> = {
  sun: {
    name: "Sun",
    mu: 1.32712440018e20,
    radius_km: 696340.0,
    mass_kg: 1.9885e30,
    parent: null,
    rotation_period_s: 2.1642e6,
    axial_tilt_rad: 0.1265,
    has_atmosphere: false,
    position_km: [0, 0, 0],
    orbit_radius_km: 0,
    color: "#ffcc00",
    texture_style: "star",
  },
  mercury: {
    name: "Mercury",
    mu: 2.2032e13,
    radius_km: 2439.7,
    mass_kg: 3.3011e23,
    j2: 5.03e-5,
    parent: "Sun",
    rotation_period_s: 5.0674e6,
    axial_tilt_rad: 0.0006,
    has_atmosphere: false,
    orbit_radius_km: PLANETARY_ORBITAL_ELEMENTS.mercury.a_km,
    orbit_period_days: PLANETARY_ORBITAL_ELEMENTS.mercury.period_days,
    color: "#a49b8f",
    texture_style: "cratered",
  },
  venus: {
    name: "Venus",
    mu: 3.24859e14,
    radius_km: 6051.8,
    mass_kg: 4.8675e24,
    j2: 4.458e-6,
    parent: "Sun",
    rotation_period_s: -2.0997e7,
    axial_tilt_rad: 3.096,
    has_atmosphere: true,
    orbit_radius_km: PLANETARY_ORBITAL_ELEMENTS.venus.a_km,
    orbit_period_days: PLANETARY_ORBITAL_ELEMENTS.venus.period_days,
    color: "#e3bb76",
    texture_style: "gas_giant",
  },
  earth: {
    name: "Earth",
    mu: 3.986004418e14,
    radius_km: 6378.137,
    mass_kg: 5.9722e24,
    j2: 1.08263e-3,
    j3: -2.5327e-6,
    parent: "Sun",
    rotation_period_s: 86164.0905,
    axial_tilt_rad: 0.4091,
    has_atmosphere: true,
    orbit_radius_km: PLANETARY_ORBITAL_ELEMENTS.earth.a_km,
    orbit_period_days: PLANETARY_ORBITAL_ELEMENTS.earth.period_days,
    color: "#3388ff",
    texture_style: "ocean_clouds",
  },
  moon: {
    name: "Moon",
    mu: 4.9028695e12,
    radius_km: 1737.4,
    mass_kg: 7.346e22,
    j2: 2.027e-4,
    parent: "Earth",
    rotation_period_s: 2.3606e6,
    axial_tilt_rad: 0.1167,
    has_atmosphere: false,
    orbit_radius_km: 384400.0,
    orbit_period_days: 27.32,
    color: "#c0c5cc",
    texture_style: "cratered",
  },
  mars: {
    name: "Mars",
    mu: 4.282837e13,
    radius_km: 3396.19,
    mass_kg: 6.4171e23,
    j2: 1.96045e-3,
    parent: "Sun",
    rotation_period_s: 88642.663,
    axial_tilt_rad: 0.4396,
    has_atmosphere: true,
    orbit_radius_km: PLANETARY_ORBITAL_ELEMENTS.mars.a_km,
    orbit_period_days: PLANETARY_ORBITAL_ELEMENTS.mars.period_days,
    color: "#e26638",
    texture_style: "rocky",
  },
  jupiter: {
    name: "Jupiter",
    mu: 1.26686534e17,
    radius_km: 71492.0,
    mass_kg: 1.8982e27,
    j2: 1.4736e-2,
    parent: "Sun",
    rotation_period_s: 35730.0,
    axial_tilt_rad: 0.0546,
    has_atmosphere: true,
    orbit_radius_km: PLANETARY_ORBITAL_ELEMENTS.jupiter.a_km,
    orbit_period_days: PLANETARY_ORBITAL_ELEMENTS.jupiter.period_days,
    color: "#d4a373",
    texture_style: "gas_giant",
  },
  saturn: {
    name: "Saturn",
    mu: 3.7931187e16,
    radius_km: 60268.0,
    mass_kg: 5.6834e26,
    j2: 1.6298e-2,
    parent: "Sun",
    rotation_period_s: 38360.0,
    axial_tilt_rad: 0.4665,
    has_atmosphere: true,
    orbit_radius_km: PLANETARY_ORBITAL_ELEMENTS.saturn.a_km,
    orbit_period_days: PLANETARY_ORBITAL_ELEMENTS.saturn.period_days,
    color: "#e9c46a",
    texture_style: "ringed",
  },
  uranus: {
    name: "Uranus",
    mu: 5.793939e15,
    radius_km: 25559.0,
    mass_kg: 8.6810e25,
    j2: 3.3434e-3,
    parent: "Sun",
    rotation_period_s: -62060.0,
    axial_tilt_rad: 1.708,
    has_atmosphere: true,
    orbit_radius_km: PLANETARY_ORBITAL_ELEMENTS.uranus.a_km,
    orbit_period_days: PLANETARY_ORBITAL_ELEMENTS.uranus.period_days,
    color: "#70d6ff",
    texture_style: "gas_giant",
  },
  neptune: {
    name: "Neptune",
    mu: 6.836529e15,
    radius_km: 24764.0,
    mass_kg: 1.02413e26,
    j2: 3.411e-3,
    parent: "Sun",
    rotation_period_s: 57996.0,
    axial_tilt_rad: 0.4943,
    has_atmosphere: true,
    orbit_radius_km: PLANETARY_ORBITAL_ELEMENTS.neptune.a_km,
    orbit_period_days: PLANETARY_ORBITAL_ELEMENTS.neptune.period_days,
    color: "#4361ee",
    texture_style: "gas_giant",
  },
};
