/**
 * THESEUS Moon Systems Catalog
 * ============================
 * Hierarchical moon data for all planetary systems.
 * Orbital elements are approximate J2000 values.
 *
 * Sources: JPL SSD, IAU 2018, NASA Planetary Fact Sheets
 */

import { AstronomicalObject, registerObjects, deterministicPhaseDeg } from './astronomicalObjects';

// Helper to create a moon entry
function moon(
  id: string, name: string, parent: string,
  radius_km: number, mass_kg: number,
  a_km: number, e: number, inc_deg: number, period_days: number,
  color: [number, number, number],
  surfaceType: 'cratered' | 'icy' | 'rocky' | 'irregular' = 'cratered',
  opts: Partial<AstronomicalObject> = {}
): AstronomicalObject {
  return {
    id, name, type: 'MOON', category: 'CELESTIAL', parent,
    dataSource: 'REAL',
    radius_km, mass_kg,
    orbit: {
      a_km, e, inc_deg, raan_deg: 0, w_deg: 0, m0_deg: deterministicPhaseDeg(id),
      elementProvenance: 'PLACEHOLDER',
      period_days,
    },
    color,
    surface: {
      type: surfaceType,
      baseColor: color,
      craterDensity: surfaceType === 'cratered' ? 0.6 : 0.2,
      roughness: 0.5,
    },
    selectable: true,
    defaultVisible: true,
    ...opts,
  };
}

// ─── EARTH SYSTEM ───────────────────────────────────────────────────

const EARTH_MOONS: AstronomicalObject[] = [
  moon('moon', 'Moon', 'earth', 1737.4, 7.346e22, 384400, 0.0549, 5.145, 27.322,
    [0.69, 0.71, 0.74], 'cratered', {
      mu: 4.9028695e12,
      j2: 2.033e-4,
      surface: {
        type: 'cratered',
        baseColor: [0.69, 0.71, 0.74],
        secondaryColor: [0.34, 0.36, 0.39],
        craterDensity: 0.85,
        roughness: 0.7,
      },
    }),
];

// ─── MARS SYSTEM ────────────────────────────────────────────────────

const MARS_MOONS: AstronomicalObject[] = [
  moon('phobos', 'Phobos', 'mars', 11.267, 1.0659e16, 9376, 0.0151, 1.093, 0.319,
    [0.45, 0.40, 0.35], 'irregular'),
  moon('deimos', 'Deimos', 'mars', 6.2, 1.4762e15, 23463, 0.00033, 0.93, 1.263,
    [0.50, 0.45, 0.40], 'irregular'),
];

// ─── JUPITER SYSTEM ─────────────────────────────────────────────────

const JUPITER_MOONS: AstronomicalObject[] = [
  // Galilean moons
  moon('io', 'Io', 'jupiter', 1821.6, 8.9319e22, 421800, 0.0041, 0.05, 1.769,
    [0.85, 0.72, 0.30], 'rocky', {
      surface: {
        type: 'rocky',
        baseColor: [0.85, 0.72, 0.30],
        secondaryColor: [0.55, 0.20, 0.10],
        hasVolcanoes: true,
        roughness: 0.8,
      },
    }),
  moon('europa', 'Europa', 'jupiter', 1560.8, 4.7998e22, 671100, 0.009, 0.47, 3.551,
    [0.78, 0.72, 0.60], 'icy', {
      surface: {
        type: 'icy',
        baseColor: [0.78, 0.72, 0.60],
        secondaryColor: [0.55, 0.48, 0.38],
        roughness: 0.3,
      },
    }),
  moon('ganymede', 'Ganymede', 'jupiter', 2634.1, 1.4819e23, 1070400, 0.0013, 0.20, 7.155,
    [0.55, 0.50, 0.45], 'icy', {
      surface: {
        type: 'icy',
        baseColor: [0.55, 0.50, 0.45],
        secondaryColor: [0.68, 0.65, 0.60],
        craterDensity: 0.4,
        roughness: 0.5,
      },
    }),
  moon('callisto', 'Callisto', 'jupiter', 2410.3, 1.0759e23, 1882700, 0.0074, 0.19, 16.689,
    [0.40, 0.38, 0.35], 'cratered', {
      surface: {
        type: 'cratered',
        baseColor: [0.40, 0.38, 0.35],
        secondaryColor: [0.25, 0.23, 0.20],
        craterDensity: 0.9,
        roughness: 0.75,
      },
    }),
];

// ─── SATURN SYSTEM ──────────────────────────────────────────────────

const SATURN_MOONS: AstronomicalObject[] = [
  moon('titan', 'Titan', 'saturn', 2574.7, 1.3452e23, 1221870, 0.0288, 0.34, 15.945,
    [0.72, 0.58, 0.32], 'rocky', {
      atmosphere: {
        color: [0.8, 0.65, 0.35],
        density: 0.8,
        scaleHeight: 0.04,
        hasHaze: true,
        hasClouds: true,
      },
    }),
  moon('enceladus', 'Enceladus', 'saturn', 252.1, 1.0802e20, 238042, 0.0047, 0.02, 1.370,
    [0.92, 0.93, 0.95], 'icy'),
  moon('rhea', 'Rhea', 'saturn', 763.8, 2.3065e21, 527068, 0.0013, 0.35, 4.518,
    [0.70, 0.68, 0.65], 'cratered'),
  moon('iapetus', 'Iapetus', 'saturn', 734.5, 1.8056e21, 3560854, 0.0286, 15.47, 79.322,
    [0.55, 0.50, 0.45], 'icy'),
  moon('dione', 'Dione', 'saturn', 561.4, 1.0955e21, 377415, 0.0022, 0.02, 2.737,
    [0.72, 0.70, 0.68], 'icy'),
  moon('tethys', 'Tethys', 'saturn', 531.1, 6.1745e20, 294672, 0.0001, 1.12, 1.888,
    [0.78, 0.76, 0.74], 'icy'),
  moon('mimas', 'Mimas', 'saturn', 198.2, 3.7493e19, 185539, 0.0196, 1.53, 0.942,
    [0.72, 0.71, 0.70], 'cratered'),
  moon('hyperion', 'Hyperion', 'saturn', 135, 5.6199e18, 1500933, 0.1042, 0.43, 21.277,
    [0.60, 0.52, 0.42], 'irregular'),
];

// ─── URANUS SYSTEM ──────────────────────────────────────────────────

const URANUS_MOONS: AstronomicalObject[] = [
  moon('titania', 'Titania', 'uranus', 788.4, 3.527e21, 436300, 0.0011, 0.08, 8.706,
    [0.62, 0.58, 0.55], 'icy'),
  moon('oberon', 'Oberon', 'uranus', 761.4, 3.014e21, 583519, 0.0014, 0.07, 13.463,
    [0.55, 0.52, 0.50], 'cratered'),
  moon('ariel', 'Ariel', 'uranus', 578.9, 1.353e21, 190900, 0.0012, 0.04, 2.520,
    [0.68, 0.65, 0.62], 'icy'),
  moon('umbriel', 'Umbriel', 'uranus', 584.7, 1.172e21, 266000, 0.0039, 0.13, 4.144,
    [0.42, 0.40, 0.38], 'cratered'),
  moon('miranda', 'Miranda', 'uranus', 235.8, 6.59e19, 129900, 0.0013, 4.34, 1.413,
    [0.58, 0.55, 0.52], 'icy'),
];

// ─── NEPTUNE SYSTEM ─────────────────────────────────────────────────

const NEPTUNE_MOONS: AstronomicalObject[] = [
  moon('triton', 'Triton', 'neptune', 1353.4, 2.14e22, 354759, 0.000016, 156.885, 5.877,
    [0.62, 0.58, 0.52], 'icy', {
      atmosphere: {
        color: [0.5, 0.55, 0.65],
        density: 0.05,
        scaleHeight: 0.03,
        hasHaze: true,
        hasClouds: false,
      },
    }),
  moon('nereid', 'Nereid', 'neptune', 170, 3.1e19, 5513818, 0.7512, 7.23, 360.13,
    [0.55, 0.52, 0.48], 'irregular'),
  moon('proteus', 'Proteus', 'neptune', 210, 4.4e19, 117646, 0.00053, 0.075, 1.122,
    [0.48, 0.45, 0.42], 'irregular'),
];

// ─── PLUTO SYSTEM ───────────────────────────────────────────────────

const PLUTO_MOONS: AstronomicalObject[] = [
  moon('charon', 'Charon', 'pluto', 606, 1.586e21, 19591, 0.0002, 0.08, 6.387,
    [0.55, 0.52, 0.50], 'icy'),
];


// ─── REGISTER ALL MOONS ─────────────────────────────────────────────

export const ALL_MOONS: AstronomicalObject[] = [
  ...EARTH_MOONS,
  ...MARS_MOONS,
  ...JUPITER_MOONS,
  ...SATURN_MOONS,
  ...URANUS_MOONS,
  ...NEPTUNE_MOONS,
  ...PLUTO_MOONS,
];

// Auto-register all moons into the main catalog
registerObjects(ALL_MOONS);

export {
  EARTH_MOONS, MARS_MOONS, JUPITER_MOONS, SATURN_MOONS,
  URANUS_MOONS, NEPTUNE_MOONS, PLUTO_MOONS,
};
