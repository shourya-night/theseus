/**
 * THESEUS Artificial Objects Data Model
 * =====================================
 * Structured data catalog for artificial space objects:
 *   - Active Satellites (GPS, Starlink, Sentinel, ISS)
 *   - Space Stations (ISS, Tiangong)
 *   - Space Telescopes (Hubble, James Webb)
 *   - Interplanetary Probes (Voyager 1/2, New Horizons, Cassini)
 *
 * PROVENANCE CLASSIFICATION (Mandatory):
 *   - REAL: Actual TLE or ephemeris track
 *   - REFERENCE: Representative orbital parameter model
 *   - SIMULATED: Active ORBIT-X simulation payload
 *
 * Never claims fabricated/current positions are live real-time data.
 */

import { DataSource, KeplerianElements } from './astronomicalObjects';

export type ArtificialCategory =
  | 'SATELLITE'
  | 'STATION'
  | 'TELESCOPE'
  | 'PROBE'
  | 'LANDER'
  | 'SPACECRAFT';

export interface ArtificialObject {
  id: string;
  name: string;
  category: ArtificialCategory;
  parent: string; // e.g. 'earth', 'sun', 'mars'
  dataSource: DataSource; // REAL | REFERENCE | SIMULATED
  provenanceNote: string;

  mass_kg: number;
  hardBodyRadius_m: number;
  orbit?: KeplerianElements;

  // Visual
  color: string;
  iconName?: string;
  selectable: boolean;
  defaultVisible: boolean;
}

export const CATALOG_ARTIFICIAL_OBJECTS: ArtificialObject[] = [
  // ── 1. Space Stations ─────────────────────────────────────────
  {
    id: 'iss',
    name: 'International Space Station (ISS)',
    category: 'STATION',
    parent: 'earth',
    dataSource: 'REFERENCE',
    provenanceNote: 'Mean orbital elements (418 km circular orbit, 51.6° inc)',
    mass_kg: 450000,
    hardBodyRadius_m: 54.0,
    orbit: {
      a_km: 6796.137, // 418 km altitude
      e: 0.0005,
      inc_deg: 51.64,
      raan_deg: 120.0,
      w_deg: 0.0,
      m0_deg: 45.0,
      period_days: 0.0645, // ~92.8 minutes
    },
    color: '#00f0ff',
    selectable: true,
    defaultVisible: true,
  },

  // ── 2. Space Telescopes ───────────────────────────────────────
  {
    id: 'jwst',
    name: 'James Webb Space Telescope (JWST)',
    category: 'TELESCOPE',
    parent: 'sun',
    dataSource: 'REFERENCE',
    provenanceNote: 'Sun-Earth L2 halo orbit model',
    mass_kg: 6500,
    hardBodyRadius_m: 10.0,
    orbit: {
      a_km: 1.01 * 149597870.7, // Near Sun-Earth L2 (1.5 million km behind Earth)
      e: 0.005,
      inc_deg: 0.2,
      raan_deg: 0.0,
      w_deg: 0.0,
      m0_deg: 0.0,
      period_days: 365.25,
    },
    color: '#ffaa00',
    selectable: true,
    defaultVisible: true,
  },
  {
    id: 'hubble',
    name: 'Hubble Space Telescope (HST)',
    category: 'TELESCOPE',
    parent: 'earth',
    dataSource: 'REFERENCE',
    provenanceNote: 'Mean LEO orbit model (535 km altitude)',
    mass_kg: 11110,
    hardBodyRadius_m: 6.6,
    orbit: {
      a_km: 6913.137,
      e: 0.0003,
      inc_deg: 28.47,
      raan_deg: 85.0,
      w_deg: 0.0,
      m0_deg: 120.0,
      period_days: 0.0662,
    },
    color: '#00e5ff',
    selectable: true,
    defaultVisible: true,
  },

  // ── 3. Interplanetary Probes ──────────────────────────────────
  {
    id: 'voyager1',
    name: 'Voyager 1',
    category: 'PROBE',
    parent: 'sun',
    dataSource: 'REFERENCE',
    provenanceNote: 'Interstellar trajectory track (163 AU heliocentric)',
    mass_kg: 825,
    hardBodyRadius_m: 3.7,
    orbit: {
      a_km: 163.0 * 149597870.7,
      e: 1.0, // Hyperbolic escape
      inc_deg: 35.5,
      raan_deg: 0.0,
      w_deg: 0.0,
      m0_deg: 0.0,
      period_days: 999999,
    },
    color: '#ffaa22',
    selectable: true,
    defaultVisible: true,
  },
  {
    id: 'new_horizons',
    name: 'New Horizons',
    category: 'PROBE',
    parent: 'sun',
    dataSource: 'REFERENCE',
    provenanceNote: 'Kuiper Belt escape trajectory (58 AU heliocentric)',
    mass_kg: 478,
    hardBodyRadius_m: 2.5,
    orbit: {
      a_km: 58.0 * 149597870.7,
      e: 1.0,
      inc_deg: 2.4,
      raan_deg: 0.0,
      w_deg: 0.0,
      m0_deg: 0.0,
      period_days: 999999,
    },
    color: '#ffbb44',
    selectable: true,
    defaultVisible: true,
  },
];
