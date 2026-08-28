/**
 * THESEUS Astronomical Object Data Model
 * =======================================
 * Data-driven catalog of all celestial objects in the Solar System.
 * This replaces hard-coded planet rendering with a structured, extensible model.
 *
 * Data sources:
 *   - JPL Solar System Dynamics (ssd.jpl.nasa.gov)
 *   - IAU Working Group on Cartographic Coordinates
 *   - DE430/DE440 ephemerides
 *   - NASA Planetary Fact Sheets
 *
 * IMPORTANT: All orbital elements are J2000 epoch mean values.
 * Positions are NOT real-time. This is reference/catalog data.
 */

// ─── TYPE DEFINITIONS ───────────────────────────────────────────────

export type ObjectType = 
  | 'STAR' | 'PLANET' | 'DWARF_PLANET' | 'MOON' 
  | 'ASTEROID' | 'COMET' | 'NEO' | 'KBO' | 'SCATTERED'
  | 'SATELLITE' | 'STATION' | 'TELESCOPE' | 'PROBE' | 'SPACECRAFT';

export type ObjectCategory =
  | 'CELESTIAL' | 'SMALL_BODY' | 'ARTIFICIAL' | 'SIMULATED';

export type DataSource = 'REAL' | 'REFERENCE' | 'SIMULATED';

export type TextureType =
  | 'star' | 'rocky' | 'cratered' | 'gas_giant' | 'ice_giant'
  | 'earth_like' | 'venusian' | 'martian' | 'icy' | 'irregular';

export type NEOClass = 'APOLLO' | 'ATEN' | 'AMOR' | 'PHA' | 'EARTH_CROSSING';

export interface KeplerianElements {
  a_km: number;        // Semi-major axis (km)
  e: number;           // Eccentricity
  inc_deg: number;     // Inclination (degrees)
  raan_deg: number;    // Ω — Right ascension of the ascending node (degrees)

  /**
   * ω — Argument of periapsis (degrees), measured in the orbit plane from the
   * ascending node. Small-body elements from JPL SBDB are published this way.
   * Ignored when `varpi_deg` is present.
   */
  w_deg?: number;

  /**
   * ϖ — Longitude of perihelion (degrees), ϖ = Ω + ω. This is what the JPL
   * planetary element tables publish, so planetary entries store the value
   * verbatim here and the renderer derives ω = ϖ − Ω. Do not copy a ϖ value
   * into `w_deg`: that rotates the orbit in its own plane by Ω.
   */
  varpi_deg?: number;

  m0_deg: number;      // M₀ — Mean anomaly at epoch (degrees)
  period_days: number; // Orbital period (days)

  /**
   * Provenance of the orientation and phase angles (Ω, ω/ϖ, M₀).
   *
   * 'CATALOG'     — the angles are published values and may be displayed.
   * 'PLACEHOLDER' — only a, e, i and the period are catalog values; the
   *                 angles exist to space objects apart on screen and are
   *                 NOT measured. Never present them as orbital elements —
   *                 show DATA UNAVAILABLE instead.
   *
   * Absent is treated as 'CATALOG'.
   */
  elementProvenance?: 'CATALOG' | 'PLACEHOLDER';
}

export interface AtmosphereParams {
  color: [number, number, number];  // RGB 0-1
  density: number;                   // 0-1 visual density
  scaleHeight: number;               // Relative to body radius
  hasHaze: boolean;
  hasClouds: boolean;
}

export interface RingSystemParams {
  innerRadius: number;   // Multiple of body radius
  outerRadius: number;   // Multiple of body radius
  color: [number, number, number]; // RGB 0-1
  densityScale: number;  // 0-1
  tilt_deg: number;      // Ring plane tilt from equatorial
  complexity: 'HIGH' | 'MEDIUM' | 'LOW'; // Level of band/gap detail
}

export interface SurfaceParams {
  type: TextureType;
  baseColor: [number, number, number];
  secondaryColor?: [number, number, number];
  craterDensity?: number;    // 0-1
  roughness?: number;        // 0-1
  hasIceCaps?: boolean;
  hasOceans?: boolean;
  hasCanyons?: boolean;
  hasMountains?: boolean;
  hasVolcanoes?: boolean;
}

export interface GasGiantParams {
  bandColors: [number, number, number][];
  bandCount: number;
  stormIntensity: number;  // 0-1
  stormCenter?: [number, number]; // UV coords
}

export interface AstronomicalObject {
  id: string;
  name: string;
  type: ObjectType;
  category: ObjectCategory;
  parent: string | null;   // Parent object ID
  dataSource: DataSource;

  // Physical
  radius_km: number;
  /**
   * Mass in kg. Optional: omit it rather than inventing a value. An absent
   * mass means DATA UNAVAILABLE and must be rendered as such, never as 0.
   */
  mass_kg?: number;
  mu?: number;             // Gravitational parameter m³/s²
  j2?: number;
  rotation_period_s?: number;
  axial_tilt_deg?: number;

  // Orbital
  orbit?: KeplerianElements;

  // Visual
  color: [number, number, number]; // Primary display color RGB 0-1
  surface?: SurfaceParams;
  atmosphere?: AtmosphereParams;
  rings?: RingSystemParams;
  gasGiant?: GasGiantParams;

  // LOD
  minVisibleDistance_AU?: number; // Don't render beyond this
  labelScale?: number;           // Label size multiplier

  // Selection
  selectable: boolean;
  defaultVisible: boolean;

  // NEO-specific
  neoClass?: NEOClass;
  isPHA?: boolean;  // Potentially Hazardous Asteroid
}


// ─── CONSTANTS ──────────────────────────────────────────────────────

export const AU_KM = 149597870.7;
export const AU_M = AU_KM * 1000.0;


// ─── SOLAR SYSTEM CATALOG ───────────────────────────────────────────

export const SOLAR_SYSTEM_OBJECTS: AstronomicalObject[] = [
  // ═══════════════════════════════════════════════════════════════════
  // SUN
  // ═══════════════════════════════════════════════════════════════════
  {
    id: 'sun',
    name: 'Sun',
    type: 'STAR',
    category: 'CELESTIAL',
    parent: null,
    dataSource: 'REAL',
    radius_km: 696340,
    mass_kg: 1.9885e30,
    mu: 1.32712440018e20,
    rotation_period_s: 2.1642e6,
    axial_tilt_deg: 7.25,
    color: [1.0, 0.85, 0.4],
    surface: { type: 'star', baseColor: [1.0, 0.78, 0.3] },
    selectable: true,
    defaultVisible: true,
  },

  // ═══════════════════════════════════════════════════════════════════
  // MERCURY
  // ═══════════════════════════════════════════════════════════════════
  {
    id: 'mercury',
    name: 'Mercury',
    type: 'PLANET',
    category: 'CELESTIAL',
    parent: 'sun',
    dataSource: 'REAL',
    radius_km: 2439.7,
    mass_kg: 3.3011e23,
    mu: 2.2032e13,
    j2: 5.03e-5,
    rotation_period_s: 5.0674e6,
    axial_tilt_deg: 0.034,
    orbit: {
      a_km: 0.38709893 * AU_KM,
      e: 0.20563069,
      inc_deg: 7.00487,
      raan_deg: 48.33167,
      varpi_deg: 77.45645,
      m0_deg: 174.7947,
      period_days: 87.9691,
    },
    color: [0.54, 0.52, 0.48],
    surface: {
      type: 'cratered',
      baseColor: [0.54, 0.52, 0.48],
      secondaryColor: [0.36, 0.34, 0.31],
      craterDensity: 0.9,
      roughness: 0.85,
    },
    selectable: true,
    defaultVisible: true,
  },

  // ═══════════════════════════════════════════════════════════════════
  // VENUS
  // ═══════════════════════════════════════════════════════════════════
  {
    id: 'venus',
    name: 'Venus',
    type: 'PLANET',
    category: 'CELESTIAL',
    parent: 'sun',
    dataSource: 'REAL',
    radius_km: 6051.8,
    mass_kg: 4.8675e24,
    mu: 3.24859e14,
    j2: 4.458e-6,
    rotation_period_s: 2.0997e7,
    axial_tilt_deg: 177.36,
    orbit: {
      a_km: 0.72333199 * AU_KM,
      e: 0.00677323,
      inc_deg: 3.39471,
      raan_deg: 76.68069,
      varpi_deg: 131.53298,
      m0_deg: 50.115,
      period_days: 224.701,
    },
    color: [0.89, 0.73, 0.46],
    surface: { type: 'venusian', baseColor: [0.85, 0.72, 0.42] },
    atmosphere: {
      color: [0.92, 0.82, 0.55],
      density: 0.95,
      scaleHeight: 0.025,
      hasHaze: true,
      hasClouds: true,
    },
    selectable: true,
    defaultVisible: true,
  },

  // ═══════════════════════════════════════════════════════════════════
  // EARTH
  // ═══════════════════════════════════════════════════════════════════
  {
    id: 'earth',
    name: 'Earth',
    type: 'PLANET',
    category: 'CELESTIAL',
    parent: 'sun',
    dataSource: 'REAL',
    radius_km: 6378.137,
    mass_kg: 5.9722e24,
    mu: 3.986004418e14,
    j2: 1.08263e-3,
    rotation_period_s: 86164.0905,
    axial_tilt_deg: 23.4393,
    orbit: {
      a_km: 1.00000011 * AU_KM,
      e: 0.01671022,
      inc_deg: 0.00005,
      raan_deg: -11.26064,
      varpi_deg: 102.94719,
      m0_deg: 358.617,
      period_days: 365.25636,
    },
    color: [0.2, 0.53, 1.0],
    surface: {
      type: 'earth_like',
      baseColor: [0.02, 0.06, 0.18],
      hasIceCaps: true,
      hasOceans: true,
    },
    atmosphere: {
      color: [0.4, 0.65, 1.0],
      density: 0.5,
      scaleHeight: 0.013,
      hasHaze: false,
      hasClouds: true,
    },
    selectable: true,
    defaultVisible: true,
  },

  // ═══════════════════════════════════════════════════════════════════
  // MARS
  // ═══════════════════════════════════════════════════════════════════
  {
    id: 'mars',
    name: 'Mars',
    type: 'PLANET',
    category: 'CELESTIAL',
    parent: 'sun',
    dataSource: 'REAL',
    radius_km: 3396.19,
    mass_kg: 6.4171e23,
    mu: 4.282837e13,
    j2: 1.96045e-3,
    rotation_period_s: 88642.663,
    axial_tilt_deg: 25.19,
    orbit: {
      a_km: 1.52366231 * AU_KM,
      e: 0.09341233,
      inc_deg: 1.85061,
      raan_deg: 49.57854,
      varpi_deg: 336.04084,
      m0_deg: 19.373,
      period_days: 686.971,
    },
    color: [0.72, 0.32, 0.12],
    surface: {
      type: 'martian',
      baseColor: [0.72, 0.32, 0.12],
      hasIceCaps: true,
      hasCanyons: true,
      hasMountains: true,
    },
    atmosphere: {
      color: [0.85, 0.6, 0.35],
      density: 0.08,
      scaleHeight: 0.015,
      hasHaze: true,
      hasClouds: false,
    },
    selectable: true,
    defaultVisible: true,
  },

  // ═══════════════════════════════════════════════════════════════════
  // JUPITER
  // ═══════════════════════════════════════════════════════════════════
  {
    id: 'jupiter',
    name: 'Jupiter',
    type: 'PLANET',
    category: 'CELESTIAL',
    parent: 'sun',
    dataSource: 'REAL',
    radius_km: 71492,
    mass_kg: 1.8982e27,
    mu: 1.26686534e17,
    j2: 1.4736e-2,
    rotation_period_s: 35730,
    axial_tilt_deg: 3.13,
    orbit: {
      a_km: 5.20336301 * AU_KM,
      e: 0.04839266,
      inc_deg: 1.3053,
      raan_deg: 100.55615,
      varpi_deg: 14.75385,
      m0_deg: 20.020,
      period_days: 4332.59,
    },
    color: [0.83, 0.64, 0.45],
    gasGiant: {
      bandColors: [
        [0.83, 0.68, 0.48],  // Light band
        [0.65, 0.42, 0.22],  // Dark band
        [0.78, 0.58, 0.35],  // Medium band
        [0.72, 0.50, 0.28],  // Medium-dark
        [0.88, 0.75, 0.55],  // Bright zone
        [0.58, 0.35, 0.18],  // Dark belt
      ],
      bandCount: 6,
      stormIntensity: 0.85,
      stormCenter: [0.65, 0.38],  // GRS position
    },
    rings: {
      innerRadius: 1.72,
      outerRadius: 1.81,
      color: [0.5, 0.4, 0.3],
      densityScale: 0.15,
      tilt_deg: 3.13,
      complexity: 'LOW',
    },
    selectable: true,
    defaultVisible: true,
  },

  // ═══════════════════════════════════════════════════════════════════
  // SATURN
  // ═══════════════════════════════════════════════════════════════════
  {
    id: 'saturn',
    name: 'Saturn',
    type: 'PLANET',
    category: 'CELESTIAL',
    parent: 'sun',
    dataSource: 'REAL',
    radius_km: 60268,
    mass_kg: 5.6834e26,
    mu: 3.7931187e16,
    j2: 1.6298e-2,
    rotation_period_s: 38360,
    axial_tilt_deg: 26.73,
    orbit: {
      a_km: 9.53707032 * AU_KM,
      e: 0.05415060,
      inc_deg: 2.48446,
      raan_deg: 113.71504,
      varpi_deg: 92.43194,
      m0_deg: 317.020,
      period_days: 10759.22,
    },
    color: [0.88, 0.77, 0.55],
    gasGiant: {
      bandColors: [
        [0.90, 0.82, 0.62],
        [0.82, 0.70, 0.48],
        [0.88, 0.78, 0.56],
        [0.78, 0.65, 0.42],
        [0.92, 0.85, 0.68],
        [0.75, 0.60, 0.38],
      ],
      bandCount: 6,
      stormIntensity: 0.2,
    },
    rings: {
      innerRadius: 1.24,  // D ring inner edge ~66,900 km / 60,268 km
      outerRadius: 2.27,  // F ring outer ~136,780 km / 60,268 km
      color: [0.82, 0.73, 0.58],
      densityScale: 1.0,
      tilt_deg: 26.73,
      complexity: 'HIGH',
    },
    selectable: true,
    defaultVisible: true,
  },

  // ═══════════════════════════════════════════════════════════════════
  // URANUS
  // ═══════════════════════════════════════════════════════════════════
  {
    id: 'uranus',
    name: 'Uranus',
    type: 'PLANET',
    category: 'CELESTIAL',
    parent: 'sun',
    dataSource: 'REAL',
    radius_km: 25559,
    mass_kg: 8.681e25,
    mu: 5.793939e15,
    j2: 3.3434e-3,
    rotation_period_s: 62060,
    axial_tilt_deg: 97.77,
    orbit: {
      a_km: 19.19126393 * AU_KM,
      e: 0.04716771,
      inc_deg: 0.76986,
      raan_deg: 74.22988,
      varpi_deg: 170.96424,
      m0_deg: 142.2386,
      period_days: 30685.4,
    },
    color: [0.44, 0.84, 1.0],
    surface: {
      type: 'ice_giant',
      baseColor: [0.44, 0.78, 0.88],
      secondaryColor: [0.52, 0.86, 0.92],
    },
    atmosphere: {
      color: [0.5, 0.82, 0.95],
      density: 0.3,
      scaleHeight: 0.02,
      hasHaze: true,
      hasClouds: false,
    },
    rings: {
      innerRadius: 1.56,
      outerRadius: 1.97,
      color: [0.4, 0.4, 0.45],
      densityScale: 0.2,
      tilt_deg: 97.77,
      complexity: 'LOW',
    },
    selectable: true,
    defaultVisible: true,
  },

  // ═══════════════════════════════════════════════════════════════════
  // NEPTUNE
  // ═══════════════════════════════════════════════════════════════════
  {
    id: 'neptune',
    name: 'Neptune',
    type: 'PLANET',
    category: 'CELESTIAL',
    parent: 'sun',
    dataSource: 'REAL',
    radius_km: 24764,
    mass_kg: 1.02413e26,
    mu: 6.836529e15,
    j2: 3.411e-3,
    rotation_period_s: 57996,
    axial_tilt_deg: 28.32,
    orbit: {
      a_km: 30.06896348 * AU_KM,
      e: 0.00858587,
      inc_deg: 1.76917,
      raan_deg: 131.72169,
      varpi_deg: 44.97135,
      m0_deg: 256.228,
      period_days: 60189.0,
    },
    color: [0.22, 0.35, 0.88],
    surface: {
      type: 'ice_giant',
      baseColor: [0.18, 0.30, 0.75],
      secondaryColor: [0.25, 0.38, 0.82],
    },
    atmosphere: {
      color: [0.25, 0.4, 0.9],
      density: 0.35,
      scaleHeight: 0.02,
      hasHaze: false,
      hasClouds: true,
    },
    rings: {
      innerRadius: 1.69,
      outerRadius: 2.54,
      color: [0.35, 0.35, 0.4],
      densityScale: 0.1,
      tilt_deg: 28.32,
      complexity: 'LOW',
    },
    selectable: true,
    defaultVisible: true,
  },

  // ═══════════════════════════════════════════════════════════════════
  // DWARF PLANETS
  // ═══════════════════════════════════════════════════════════════════
  {
    id: 'pluto',
    name: 'Pluto',
    type: 'DWARF_PLANET',
    category: 'CELESTIAL',
    parent: 'sun',
    dataSource: 'REAL',
    radius_km: 1188.3,
    mass_kg: 1.303e22,
    mu: 8.71e11,
    rotation_period_s: 551856.672,
    axial_tilt_deg: 122.53,
    orbit: {
      a_km: 39.48168677 * AU_KM,
      e: 0.24880766,
      inc_deg: 17.14175,
      raan_deg: 110.30347,
      varpi_deg: 224.06676,
      m0_deg: 14.53,
      period_days: 90560,
    },
    color: [0.72, 0.62, 0.52],
    surface: {
      type: 'icy',
      baseColor: [0.72, 0.62, 0.52],
      secondaryColor: [0.55, 0.48, 0.40],
      craterDensity: 0.3,
      roughness: 0.4,
    },
    selectable: true,
    defaultVisible: true,
  },
  {
    id: 'ceres',
    name: 'Ceres',
    type: 'DWARF_PLANET',
    category: 'CELESTIAL',
    parent: 'sun',
    dataSource: 'REAL',
    radius_km: 473,
    mass_kg: 9.3835e20,
    orbit: {
      a_km: 2.7675 * AU_KM,
      e: 0.0758,
      inc_deg: 10.594,
      raan_deg: 80.394,
      w_deg: 73.597,
      m0_deg: 77.37,
      period_days: 1681.63,
    },
    color: [0.55, 0.52, 0.48],
    surface: {
      type: 'cratered',
      baseColor: [0.55, 0.52, 0.48],
      craterDensity: 0.7,
      roughness: 0.6,
    },
    selectable: true,
    defaultVisible: true,
  },
  {
    id: 'eris',
    name: 'Eris',
    type: 'DWARF_PLANET',
    category: 'CELESTIAL',
    parent: 'sun',
    dataSource: 'REAL',
    radius_km: 1163,
    mass_kg: 1.66e22,
    orbit: {
      a_km: 67.781 * AU_KM,
      e: 0.4407,
      inc_deg: 44.04,
      raan_deg: 35.87,
      w_deg: 151.43,
      m0_deg: 205.989,
      period_days: 203830,
    },
    color: [0.85, 0.83, 0.80],
    surface: { type: 'icy', baseColor: [0.85, 0.83, 0.80], roughness: 0.3 },
    selectable: true,
    defaultVisible: true,
  },
  {
    id: 'haumea',
    name: 'Haumea',
    type: 'DWARF_PLANET',
    category: 'CELESTIAL',
    parent: 'sun',
    dataSource: 'REAL',
    radius_km: 816,
    mass_kg: 4.006e21,
    orbit: {
      a_km: 43.218 * AU_KM,
      e: 0.1912,
      inc_deg: 28.19,
      raan_deg: 122.167,
      w_deg: 239.041,
      m0_deg: 218.205,
      period_days: 103774,
    },
    color: [0.75, 0.72, 0.68],
    surface: { type: 'icy', baseColor: [0.75, 0.72, 0.68], roughness: 0.5 },
    selectable: true,
    defaultVisible: true,
  },
  {
    id: 'makemake',
    name: 'Makemake',
    type: 'DWARF_PLANET',
    category: 'CELESTIAL',
    parent: 'sun',
    dataSource: 'REAL',
    radius_km: 715,
    mass_kg: 3.1e21,
    orbit: {
      a_km: 45.792 * AU_KM,
      e: 0.1559,
      inc_deg: 28.98,
      raan_deg: 79.382,
      w_deg: 296.534,
      m0_deg: 165.514,
      period_days: 113183,
    },
    color: [0.68, 0.55, 0.45],
    surface: { type: 'icy', baseColor: [0.68, 0.55, 0.45], roughness: 0.4 },
    selectable: true,
    defaultVisible: true,
  },
  {
    id: 'quaoar',
    name: 'Quaoar',
    type: 'DWARF_PLANET',
    category: 'CELESTIAL',
    parent: 'sun',
    dataSource: 'REAL',
    radius_km: 555,
    mass_kg: 1.4e21,
    orbit: {
      a_km: 43.694 * AU_KM,
      e: 0.0395,
      inc_deg: 7.99,
      raan_deg: 189.07,
      w_deg: 147.48,
      m0_deg: 302.5,
      period_days: 105495,
    },
    color: [0.55, 0.45, 0.40],
    surface: { type: 'icy', baseColor: [0.55, 0.45, 0.40], roughness: 0.5 },
    selectable: true,
    defaultVisible: true,
  },
  {
    id: 'sedna',
    name: 'Sedna',
    type: 'DWARF_PLANET',
    category: 'CELESTIAL',
    parent: 'sun',
    dataSource: 'REAL',
    radius_km: 498,
    mass_kg: 1.0e21,
    orbit: {
      a_km: 506.8 * AU_KM,
      e: 0.8496,
      inc_deg: 11.93,
      raan_deg: 144.246,
      w_deg: 311.46,
      m0_deg: 358.117,
      period_days: 4161900,
    },
    color: [0.72, 0.38, 0.25],
    surface: { type: 'icy', baseColor: [0.72, 0.38, 0.25], roughness: 0.6 },
    selectable: true,
    defaultVisible: true,
  },
];


// ─── LOOKUP HELPERS ─────────────────────────────────────────────────

const _objectMap = new Map<string, AstronomicalObject>();
SOLAR_SYSTEM_OBJECTS.forEach(obj => _objectMap.set(obj.id, obj));

/**
 * Deterministic pseudo-phase in degrees, derived from an object id.
 *
 * Used only where a catalog supplies a, e, i and the period but no mean
 * anomaly at epoch. It replaces Math.random() so that an object occupies the
 * same place on every reload — a random phase is both unreproducible and
 * indistinguishable from a real element. Any orbit using this MUST set
 * `elementProvenance: 'PLACEHOLDER'`.
 */
export function deterministicPhaseDeg(seed: string): number {
  let h = 2166136261;
  for (let i = 0; i < seed.length; i++) {
    h ^= seed.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 360000) / 1000;
}

export function getObject(id: string): AstronomicalObject | undefined {
  return _objectMap.get(id.toLowerCase());
}

export function getObjectsByType(type: ObjectType): AstronomicalObject[] {
  return SOLAR_SYSTEM_OBJECTS.filter(o => o.type === type);
}

export function getObjectsByParent(parentId: string): AstronomicalObject[] {
  return SOLAR_SYSTEM_OBJECTS.filter(o => o.parent === parentId);
}

export function getPlanets(): AstronomicalObject[] {
  return getObjectsByType('PLANET');
}

export function getDwarfPlanets(): AstronomicalObject[] {
  return getObjectsByType('DWARF_PLANET');
}

export function getMoons(parentId: string): AstronomicalObject[] {
  return SOLAR_SYSTEM_OBJECTS.filter(o => o.type === 'MOON' && o.parent === parentId);
}

/**
 * Register additional objects at runtime (e.g., moons, asteroids from separate modules).
 * This allows the catalog to be extended without circular imports.
 */
export function registerObjects(objects: AstronomicalObject[]): void {
  objects.forEach(obj => {
    if (!_objectMap.has(obj.id)) {
      SOLAR_SYSTEM_OBJECTS.push(obj);
      _objectMap.set(obj.id, obj);
    }
  });
}
