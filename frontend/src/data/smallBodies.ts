/**
 * THESEUS Small Bodies Catalog
 * ============================
 * Named asteroids, NEOs, comets, Kuiper Belt objects, and population parameters.
 *
 * Sources: JPL Small-Body Database, MPC, IAU Minor Planet Center
 *
 * IMPORTANT: Individual object positions are catalog/reference epoch values.
 * They do NOT represent current real-time positions.
 */

import {
  AstronomicalObject, NEOClass, AU_KM,
  registerObjects,
  deterministicPhaseDeg,
} from './astronomicalObjects';

export { AU_KM };


// ─── NAMED ASTEROIDS (Main Belt) ────────────────────────────────────

function asteroid(
  id: string, name: string, radius_km: number, mass_kg: number,
  a_AU: number, e: number, inc_deg: number, period_days: number,
  color: [number, number, number],
): AstronomicalObject {
  return {
    id, name, type: 'ASTEROID', category: 'SMALL_BODY', parent: 'sun',
    dataSource: 'REAL', radius_km, mass_kg,
    orbit: {
      a_km: a_AU * AU_KM, e, inc_deg, raan_deg: 0, w_deg: 0,
      m0_deg: deterministicPhaseDeg(id), period_days,
      elementProvenance: 'PLACEHOLDER',
    },
    color,
    surface: { type: 'irregular', baseColor: color, roughness: 0.9, craterDensity: 0.5 },
    selectable: true, defaultVisible: false,
  };
}

export const NAMED_ASTEROIDS: AstronomicalObject[] = [
  asteroid('vesta', 'Vesta', 262.7, 2.59e20, 2.3615, 0.0887, 7.14, 1325.75, [0.55, 0.52, 0.48]),
  asteroid('pallas', 'Pallas', 256, 2.11e20, 2.7716, 0.2313, 34.83, 1686.43, [0.48, 0.46, 0.43]),
  asteroid('hygiea', 'Hygiea', 217, 8.67e19, 3.1416, 0.1146, 3.84, 2035.04, [0.40, 0.38, 0.36]),
  asteroid('eros', 'Eros', 8.42, 6.687e15, 1.458, 0.2226, 10.83, 643.0, [0.58, 0.48, 0.38]),
  asteroid('bennu', 'Bennu', 0.245, 7.329e10, 1.126, 0.2037, 6.035, 436.65, [0.42, 0.38, 0.35]),
  asteroid('ryugu', 'Ryugu', 0.45, 4.50e11, 1.190, 0.1903, 5.884, 473.89, [0.35, 0.32, 0.30]),
  asteroid('apophis', 'Apophis', 0.185, 2.7e10, 0.9224, 0.1912, 3.339, 323.59, [0.52, 0.45, 0.38]),
  asteroid('ida', 'Ida', 15.7, 4.2e16, 2.862, 0.0452, 1.14, 1768.0, [0.50, 0.46, 0.42]),
  asteroid('gaspra', 'Gaspra', 6.1, 2.5e16, 2.210, 0.1735, 4.10, 1199.0, [0.55, 0.50, 0.45]),
];

// Mark certain asteroids as NEOs
const NEO_IDS = new Set(['eros', 'bennu', 'ryugu', 'apophis']);
NAMED_ASTEROIDS.forEach(a => {
  if (NEO_IDS.has(a.id)) {
    a.type = 'NEO';
  }
});

// Apophis is a PHA
const apophis = NAMED_ASTEROIDS.find(a => a.id === 'apophis');
if (apophis) {
  apophis.neoClass = 'ATEN';
  apophis.isPHA = true;
}
const bennu = NAMED_ASTEROIDS.find(a => a.id === 'bennu');
if (bennu) { bennu.neoClass = 'APOLLO'; bennu.isPHA = true; }
const eros = NAMED_ASTEROIDS.find(a => a.id === 'eros');
if (eros) { eros.neoClass = 'AMOR'; }
const ryugu = NAMED_ASTEROIDS.find(a => a.id === 'ryugu');
if (ryugu) { ryugu.neoClass = 'APOLLO'; }


// ─── COMETS ─────────────────────────────────────────────────────────

function comet(
  id: string, name: string, radius_km: number,
  a_AU: number, e: number, inc_deg: number, period_days: number,
): AstronomicalObject {
  return {
    id, name, type: 'COMET', category: 'SMALL_BODY', parent: 'sun',
    // Nucleus masses are not well constrained for most of these bodies and
    // are deliberately omitted rather than filled with a placeholder value.
    dataSource: 'REFERENCE', radius_km,
    orbit: {
      // a, e, i and the period are catalog values; Ω, ω and M₀ are not
      // available in this dataset, so the orbit is oriented for display only.
      a_km: a_AU * AU_KM, e, inc_deg, raan_deg: 0, w_deg: 0,
      m0_deg: deterministicPhaseDeg(id), period_days,
      elementProvenance: 'PLACEHOLDER',
    },
    color: [0.72, 0.70, 0.65],
    surface: { type: 'irregular', baseColor: [0.45, 0.42, 0.38], roughness: 0.95 },
    selectable: true, defaultVisible: false,
  };
}

export const NAMED_COMETS: AstronomicalObject[] = [
  comet('halley', "Halley's Comet (1P)", 5.5, 17.834, 0.9671, 162.26, 27507),
  comet('67p', '67P/Churyumov–Gerasimenko', 2.0, 3.4630, 0.6410, 7.04, 2354),
  comet('hale-bopp', 'Hale-Bopp (C/1995 O1)', 30, 186.0, 0.9951, 89.43, 927175),
  comet('hyakutake', 'Hyakutake (C/1996 B2)', 2.0, 1700.0, 0.9998, 124.92, 25583700),
  comet('encke', 'Encke (2P)', 2.4, 2.2152, 0.8483, 11.78, 1204),
];


// ─── ASTEROID BELT POPULATION PARAMETERS ────────────────────────────

export interface PopulationParams {
  innerRadius_AU: number;
  outerRadius_AU: number;
  count: number;
  sizeRange: [number, number]; // min, max in scene units
  inclinationRange_deg: [number, number];
  eccentricityRange: [number, number];
  color: [number, number, number];
  colorVariation: number; // 0-1

  /**
   * For resonance-locked populations (e.g. Jupiter Trojans, Hildas) that
   * cluster around specific orbital longitudes rather than being azimuthally
   * uniform.
   */
  resonancePhase?: {
    parentBody: string;      // e.g. 'jupiter'
    lagrangePoint?: 'L4' | 'L5';
    /** Half-width of the libration region in degrees of ecliptic longitude. */
    librationAmplitude_deg?: number;
  };
}

export const ASTEROID_BELT_PARAMS: PopulationParams = {
  innerRadius_AU: 2.06,
  outerRadius_AU: 3.27,
  count: 12000,
  sizeRange: [0.3, 2.5],
  inclinationRange_deg: [-20, 20],
  eccentricityRange: [0.0, 0.3],
  color: [0.52, 0.48, 0.42],
  colorVariation: 0.15,
};

export const KUIPER_BELT_PARAMS: PopulationParams = {
  innerRadius_AU: 30,
  outerRadius_AU: 55,
  count: 6000,
  sizeRange: [0.2, 1.5],
  inclinationRange_deg: [-30, 30],
  eccentricityRange: [0.0, 0.3],
  color: [0.45, 0.48, 0.55],
  colorVariation: 0.1,
};

export const SCATTERED_DISK_PARAMS: PopulationParams = {
  innerRadius_AU: 30,
  outerRadius_AU: 150,
  count: 2000,
  sizeRange: [0.2, 1.0],
  inclinationRange_deg: [-45, 45],
  eccentricityRange: [0.3, 0.8],
  color: [0.40, 0.42, 0.48],
  colorVariation: 0.12,
};

export const OORT_CLOUD_PARAMS: PopulationParams = {
  innerRadius_AU: 2000,
  outerRadius_AU: 50000,
  count: 3000,
  sizeRange: [0.1, 0.5],
  inclinationRange_deg: [-90, 90],
  eccentricityRange: [0.0, 0.99],
  color: [0.35, 0.38, 0.42],
  colorVariation: 0.05,
};


// ─── JUPITER TROJANS ────────────────────────────────────────────────
//
// Two swarms locked in 1:1 mean-motion resonance with Jupiter at the
// Sun–Jupiter L4 (Greek camp, 60° ahead) and L5 (Trojan camp, 60° behind)
// Lagrange points. Semi-major axes are near Jupiter's (~5.2 AU).

export const TROJAN_L4_PARAMS: PopulationParams = {
  innerRadius_AU: 4.7,
  outerRadius_AU: 5.5,
  count: 3000,
  sizeRange: [0.2, 1.8],
  inclinationRange_deg: [-35, 35],
  eccentricityRange: [0.0, 0.2],
  color: [0.58, 0.50, 0.38],
  colorVariation: 0.12,
  resonancePhase: {
    parentBody: 'jupiter',
    lagrangePoint: 'L4',
    librationAmplitude_deg: 25,
  },
};

export const TROJAN_L5_PARAMS: PopulationParams = {
  innerRadius_AU: 4.7,
  outerRadius_AU: 5.5,
  count: 2500,
  sizeRange: [0.2, 1.8],
  inclinationRange_deg: [-35, 35],
  eccentricityRange: [0.0, 0.2],
  color: [0.55, 0.48, 0.36],
  colorVariation: 0.12,
  resonancePhase: {
    parentBody: 'jupiter',
    lagrangePoint: 'L5',
    librationAmplitude_deg: 25,
  },
};


// ─── HILDAS ─────────────────────────────────────────────────────────
//
// Asteroids in the 3:2 mean-motion resonance with Jupiter. They form a
// characteristic dynamical triangle when viewed in the co-rotating frame,
// with semi-major axes near 3.97 AU.

export const HILDA_PARAMS: PopulationParams = {
  innerRadius_AU: 3.70,
  outerRadius_AU: 4.20,
  count: 2000,
  sizeRange: [0.3, 2.0],
  inclinationRange_deg: [-20, 20],
  eccentricityRange: [0.05, 0.30],
  color: [0.55, 0.42, 0.32],
  colorVariation: 0.10,
};


// ─── CYBELES ────────────────────────────────────────────────────────
//
// Asteroids near the 7:4 mean-motion resonance with Jupiter,
// between the main belt outer edge and the Hildas (a ≈ 3.28–3.70 AU).

export const CYBELE_PARAMS: PopulationParams = {
  innerRadius_AU: 3.28,
  outerRadius_AU: 3.70,
  count: 1000,
  sizeRange: [0.3, 2.2],
  inclinationRange_deg: [-25, 25],
  eccentricityRange: [0.0, 0.25],
  color: [0.50, 0.45, 0.38],
  colorVariation: 0.10,
};


// ─── CENTAURS ───────────────────────────────────────────────────────
//
// Small bodies orbiting between Jupiter and Neptune (roughly 5–30 AU).
// Dynamically unstable; higher eccentricities and inclinations than
// main-belt asteroids. Surfaces are icy.

export const CENTAUR_PARAMS: PopulationParams = {
  innerRadius_AU: 5.5,
  outerRadius_AU: 30.0,
  count: 500,
  sizeRange: [0.3, 2.0],
  inclinationRange_deg: [-35, 35],
  eccentricityRange: [0.10, 0.55],
  color: [0.48, 0.52, 0.58],
  colorVariation: 0.10,
};

// ─── NEO POPULATION CATEGORIES ──────────────────────────────────────

export interface NEOPopulationCategory {
  name: string;
  class: NEOClass;
  description: string;
  count: number;
  color: [number, number, number];
  perihelion_AU_range: [number, number];
  aphelion_AU_range: [number, number];
}

export const NEO_CATEGORIES: NEOPopulationCategory[] = [
  {
    name: 'Apollo',
    class: 'APOLLO',
    description: 'Semi-major axis > 1 AU, perihelion < 1.017 AU (Earth-crossing)',
    count: 400,
    color: [0.85, 0.55, 0.2],
    perihelion_AU_range: [0.3, 1.017],
    aphelion_AU_range: [1.0, 5.0],
  },
  {
    name: 'Aten',
    class: 'ATEN',
    description: 'Semi-major axis < 1 AU, aphelion > 0.983 AU',
    count: 150,
    color: [0.9, 0.4, 0.2],
    perihelion_AU_range: [0.2, 1.0],
    aphelion_AU_range: [0.983, 1.5],
  },
  {
    name: 'Amor',
    class: 'AMOR',
    description: 'Perihelion 1.017–1.3 AU (Earth-approaching)',
    count: 300,
    color: [0.7, 0.65, 0.3],
    perihelion_AU_range: [1.017, 1.3],
    aphelion_AU_range: [1.3, 5.0],
  },
  {
    name: 'Potentially Hazardous',
    class: 'PHA',
    description: 'MOID < 0.05 AU, absolute magnitude H < 22',
    count: 80,
    color: [0.95, 0.25, 0.15],
    perihelion_AU_range: [0.2, 1.05],
    aphelion_AU_range: [0.95, 5.0],
  },
];

// ─── METEOROID STREAMS ──────────────────────────────────────────────

export interface MeteorStream {
  name: string;
  id: string;
  radiantRA_deg: number;
  radiantDec_deg: number;
  speed_km_s: number;
  peakDate: string;
  parentBody: string | null;
  color: [number, number, number];
  particleCount: number;
  orbitalElements: {
    a_AU: number;
    e: number;
    inc_deg: number;
  };
}

export const METEOR_STREAMS: MeteorStream[] = [
  {
    name: 'Perseids', id: 'perseids',
    radiantRA_deg: 48, radiantDec_deg: 58,
    speed_km_s: 59, peakDate: 'Aug 12',
    parentBody: 'Swift–Tuttle (109P)',
    color: [0.6, 0.7, 0.9],
    particleCount: 800,
    orbitalElements: { a_AU: 24.3, e: 0.963, inc_deg: 113.0 },
  },
  {
    name: 'Leonids', id: 'leonids',
    radiantRA_deg: 152, radiantDec_deg: 22,
    speed_km_s: 71, peakDate: 'Nov 17',
    parentBody: 'Tempel–Tuttle (55P)',
    color: [0.7, 0.8, 0.5],
    particleCount: 500,
    orbitalElements: { a_AU: 10.3, e: 0.906, inc_deg: 162.5 },
  },
  {
    name: 'Geminids', id: 'geminids',
    radiantRA_deg: 112, radiantDec_deg: 33,
    speed_km_s: 35, peakDate: 'Dec 13',
    parentBody: 'Phaethon (3200)',
    color: [0.9, 0.8, 0.5],
    particleCount: 1000,
    orbitalElements: { a_AU: 1.271, e: 0.890, inc_deg: 23.4 },
  },
  {
    name: 'Orionids', id: 'orionids',
    radiantRA_deg: 95, radiantDec_deg: 16,
    speed_km_s: 66, peakDate: 'Oct 21',
    parentBody: "Halley's Comet (1P)",
    color: [0.5, 0.6, 0.8],
    particleCount: 400,
    orbitalElements: { a_AU: 17.8, e: 0.967, inc_deg: 162.3 },
  },
  {
    name: 'Quadrantids', id: 'quadrantids',
    radiantRA_deg: 230, radiantDec_deg: 49,
    speed_km_s: 41, peakDate: 'Jan 3',
    parentBody: 'Asteroid 2003 EH1',
    color: [0.4, 0.5, 0.7],
    particleCount: 600,
    orbitalElements: { a_AU: 3.14, e: 0.680, inc_deg: 71.0 },
  },
  {
    name: 'Taurids', id: 'taurids',
    radiantRA_deg: 51, radiantDec_deg: 14,
    speed_km_s: 28, peakDate: 'Nov 5',
    parentBody: 'Encke (2P)',
    color: [0.8, 0.6, 0.3],
    particleCount: 300,
    orbitalElements: { a_AU: 2.22, e: 0.848, inc_deg: 5.4 },
  },
];

// ─── NAMED CENTAURS ─────────────────────────────────────────────────
//
// Individually catalogued Centaurs with real JPL SBDB orbital elements.

export const NAMED_CENTAURS: AstronomicalObject[] = [
  {
    id: 'chiron', name: '2060 Chiron', type: 'ASTEROID', category: 'SMALL_BODY', parent: 'sun',
    dataSource: 'REAL', radius_km: 117, mass_kg: 2.4e18,
    orbit: {
      a_km: 13.708 * AU_KM, e: 0.3823, inc_deg: 6.935,
      raan_deg: 209.382, w_deg: 339.557, m0_deg: 135.67,
      period_days: 18530,
    },
    color: [0.50, 0.55, 0.60],
    surface: { type: 'icy', baseColor: [0.50, 0.55, 0.60], roughness: 0.7 },
    selectable: true, defaultVisible: false,
  },
  {
    id: 'chariklo', name: '10199 Chariklo', type: 'ASTEROID', category: 'SMALL_BODY', parent: 'sun',
    dataSource: 'REAL', radius_km: 151, mass_kg: 8.0e18,
    orbit: {
      a_km: 15.765 * AU_KM, e: 0.1719, inc_deg: 23.377,
      raan_deg: 300.45, w_deg: 241.42, m0_deg: 95.13,
      period_days: 22866,
    },
    color: [0.45, 0.50, 0.55],
    surface: { type: 'icy', baseColor: [0.45, 0.50, 0.55], roughness: 0.6 },
    selectable: true, defaultVisible: false,
  },
];


// ─── REGISTER ALL ───────────────────────────────────────────────────

registerObjects(NAMED_ASTEROIDS);
registerObjects(NAMED_COMETS);
registerObjects(NAMED_CENTAURS);
