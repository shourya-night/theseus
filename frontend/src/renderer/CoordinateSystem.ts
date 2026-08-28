/**
 * THESEUS Canonical Coordinate & Orbit Transformation System
 * ==========================================================
 * THIS IS THE ONLY PLACE IN THE FRONTEND THAT MAY CONVERT ORBITAL ELEMENTS
 * OR ENGINE STATE INTO THREE.JS SCENE COORDINATES.
 *
 * Renderers must not implement their own Kepler propagation, their own axis
 * swaps, or their own km->unit scaling. If you find yourself writing
 * `y = x * Math.sin(inc)` in a renderer, you are reintroducing the bug this
 * module exists to remove.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * PIPELINE
 * ─────────────────────────────────────────────────────────────────────────
 *   ENGINE (ORBIT-X)                 FRONTEND                  THREE.JS
 *   ----------------                 --------                  --------
 *   units:  metres, m/s        →   * SCENE_SCALE (1e-7)   →   1 unit = 10,000 km
 *   axes:   X east, Y, Z north  →   axis remap             →   X right, Y up, Z toward viewer
 *   frame:  heliocentric ecliptic (unless a body-centred mission state is
 *           explicitly declared by the caller)
 *   origin: Sun barycentre                                 →   scene origin (0,0,0)
 *
 * AXIS REMAP (Z-north becomes Y-up, handedness PRESERVED):
 *   X_three =  X_engine * SCENE_SCALE
 *   Y_three =  Z_engine * SCENE_SCALE
 *   Z_three = -Y_engine * SCENE_SCALE
 *
 * The negation on Z is load-bearing and must not be "tidied away".
 *
 * Both frames are right-handed, so the map between them has to be a rotation.
 * The obvious-looking swap (x, y, z) -> (x, z, y) is a REFLECTION: its
 * determinant is -1. Under it the whole solar system renders mirrored — every
 * orbit runs backwards, prograde motion reads as retrograde, and a genuinely
 * retrograde body such as Halley reads as prograde relative to the planets.
 * Negating one of the swapped axes restores determinant +1.
 *
 * Sanity check, asserted in scripts/verifyOrbits.ts: for any prograde orbit,
 * r x v must point along +Y (ecliptic north) in scene coordinates.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * SCALE POLICY
 * ─────────────────────────────────────────────────────────────────────────
 * PHYSICAL POSITION and PHYSICAL SIZE are converted with the same factor and
 * are never distorted. Any VISUAL SIZE exaggeration required for readability
 * must be a named, exported constant applied to geometry only — never to a
 * position. There are no magic literals in this file.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * ANGLE CONVENTIONS
 * ─────────────────────────────────────────────────────────────────────────
 * Catalog element sets follow JPL/IAU usage:
 *   Ω  raan_deg   right ascension of the ascending node
 *   ω  w_deg      argument of periapsis
 *   ϖ  varpi_deg  longitude of perihelion, ϖ = Ω + ω
 *   M₀ m0_deg     mean anomaly at epoch
 *
 * JPL planetary tables publish ϖ, not ω. When `varpi_deg` is present it wins
 * and ω is derived as ϖ − Ω, so the catalog can keep the published numbers
 * verbatim. `w_deg` is used directly only when no `varpi_deg` is supplied.
 */

import * as THREE from 'three';

// ─── SCALE CONSTANTS ────────────────────────────────────────────────────

/** 1 Three.js scene unit = 10,000 km. */
export const KM_PER_SCENE_UNIT = 10000;

/** Metres -> scene units. Equivalent to 1 / (KM_PER_SCENE_UNIT * 1000). */
export const SCENE_SCALE = 1 / (KM_PER_SCENE_UNIT * 1000); // 1e-7

/** Astronomical Unit in km. */
export const AU_KM = 149597870.7;

/** Astronomical Unit in metres. */
export const AU_M = AU_KM * 1000;

/** One AU expressed in scene units (~14,959.787). */
export const AU_SCENE = AU_KM / KM_PER_SCENE_UNIT;

/** Gravitational parameter of the Sun (km^3 / s^2). */
export const MU_SUN_KM = 1.32712440018e11;

/** Gravitational parameter of the Earth (km^3 / s^2). */
export const MU_EARTH_KM = 398600.4418;

// ─── SCALAR CONVERSIONS ─────────────────────────────────────────────────

/** Kilometres -> scene units. Use for radii and distances alike. */
export function kmToScene(km: number): number {
  return km / KM_PER_SCENE_UNIT;
}

/** Scene units -> kilometres. */
export function sceneToKm(units: number): number {
  return units * KM_PER_SCENE_UNIT;
}

/** Astronomical units -> scene units. */
export function auToScene(au: number): number {
  return au * AU_SCENE;
}

/** Scene units -> astronomical units. */
export function sceneToAu(units: number): number {
  return units / AU_SCENE;
}

/**
 * Camera distance required to fit a sphere of the given scene radius in a
 * perspective frustum. Used so view presets are derived from catalog values
 * rather than hardcoded camera positions.
 */
export function framingDistanceForRadius(sceneRadius: number, fovDeg = 50): number {
  return sceneRadius / Math.tan((fovDeg * Math.PI) / 360);
}

// ─── ENGINE STATE CONVERSIONS ───────────────────────────────────────────

/** ORBIT-X position [x, y, z] in metres -> Three.js scene position. */
export function engineToThreePos(posMeters: [number, number, number] | number[]): THREE.Vector3 {
  return engineToThreePosInto(posMeters, new THREE.Vector3());
}

/** Allocation-free variant of {@link engineToThreePos}. */
export function engineToThreePosInto(
  posMeters: [number, number, number] | number[],
  out: THREE.Vector3
): THREE.Vector3 {
  return out.set(
     posMeters[0] * SCENE_SCALE,
     posMeters[2] * SCENE_SCALE, // engine Z (north) -> three Y (up)
    -posMeters[1] * SCENE_SCALE  // engine Y -> three -Z, preserving handedness
  );
}

/**
 * ORBIT-X velocity [vx, vy, vz] in m/s -> Three.js direction vector.
 * Axis remap only; magnitude stays in m/s so callers can read speed off it.
 */
export function engineToThreeVel(velMetersPerSec: [number, number, number] | number[]): THREE.Vector3 {
  return engineToThreeVelInto(velMetersPerSec, new THREE.Vector3());
}

/** Allocation-free variant of {@link engineToThreeVel}. */
export function engineToThreeVelInto(
  velMetersPerSec: [number, number, number] | number[],
  out: THREE.Vector3
): THREE.Vector3 {
  return out.set(velMetersPerSec[0], velMetersPerSec[2], -velMetersPerSec[1]);
}

/** Three.js scene position -> ORBIT-X position [x, y, z] in metres. */
export function threeToEnginePos(v: THREE.Vector3): [number, number, number] {
  const inv = 1 / SCENE_SCALE;
  return [v.x * inv, -v.z * inv, v.y * inv];
}

// ─── KEPLER SOLVER ──────────────────────────────────────────────────────

const TWO_PI = Math.PI * 2;
const DEG = Math.PI / 180;

function wrapTwoPi(x: number): number {
  const m = x % TWO_PI;
  return m < 0 ? m + TWO_PI : m;
}

/**
 * Solve Kepler's equation M = E − e·sin E for the eccentric anomaly E.
 *
 * Uses a Danby-style starter, which keeps Newton–Raphson convergent at the
 * high eccentricities in the comet catalog (e ≈ 0.97) where the naive
 * starter E₀ = M oscillates. Visualization-side only; this does not
 * participate in any ORBIT-X computation.
 */
export function solveKepler(M: number, e: number, tolerance = 1e-10, maxIter = 60): number {
  const Mn = wrapTwoPi(M);
  if (e < 1e-9) return Mn;

  // Danby starter: accurate enough that Newton converges in a few steps
  // across the whole elliptical range.
  let E = e < 0.8 ? Mn : Mn + 0.85 * e * Math.sign(Math.sin(Mn) || 1);

  for (let i = 0; i < maxIter; i++) {
    const f = E - e * Math.sin(E) - Mn;
    const fp = 1 - e * Math.cos(E);
    // Guard against a vanishing derivative near periapsis at e -> 1.
    const delta = f / (Math.abs(fp) < 1e-12 ? 1e-12 * Math.sign(fp || 1) : fp);
    E -= delta;
    if (Math.abs(delta) < tolerance) break;
  }
  return E;
}

// ─── ORBITAL ELEMENTS ───────────────────────────────────────────────────

/**
 * Element set accepted by this module. Structurally compatible with
 * `KeplerianElements` in the astronomical catalog.
 */
export interface OrbitElementsInput {
  a_km: number;
  e: number;
  inc_deg: number;
  raan_deg: number;
  /** Argument of periapsis ω. Ignored when `varpi_deg` is present. */
  w_deg?: number;
  /** Longitude of perihelion ϖ = Ω + ω, as published by JPL. Takes priority. */
  varpi_deg?: number;
  m0_deg: number;
  period_days: number;
}

/**
 * Can this element set be propagated by the elliptical Kepler solver?
 *
 * `prepareOrbit` and `orbitPositionInto` implement the ELLIPTICAL two-body
 * problem. At e >= 1 the conic is a parabola or hyperbola: r = a(1 - e cos E)
 * is meaningless, sqrt(1 - e) is imaginary, and the solver degenerates — an
 * e = 1 entry collapses to r = 0 at epoch, i.e. exactly on top of the focus.
 * That is how two escape-trajectory probes ended up rendering at the centre of
 * the Sun.
 *
 * Callers must check this before propagating, and must NOT substitute a
 * plausible-looking position for an object that fails it.
 */
export function isEllipticalOrbit(orbit: OrbitElementsInput): boolean {
  return (
    Number.isFinite(orbit.a_km) && orbit.a_km > 0 &&
    Number.isFinite(orbit.e) && orbit.e >= 0 && orbit.e < 1 &&
    Number.isFinite(orbit.period_days) && orbit.period_days > 0
  );
}

/**
 * Resolve the argument of periapsis ω in degrees from an element set,
 * deriving it from the longitude of perihelion when that is what the
 * catalog stores.
 */
export function argumentOfPeriapsisDeg(orbit: OrbitElementsInput): number {
  if (typeof orbit.varpi_deg === 'number') {
    const w = (orbit.varpi_deg - orbit.raan_deg) % 360;
    return w < 0 ? w + 360 : w;
  }
  return orbit.w_deg ?? 0;
}

/**
 * An element set with its perifocal->scene basis precomputed.
 *
 * Prepare once, evaluate many times. This is what makes it affordable for a
 * 12,000-instance population to use the same correct math as a planet.
 */
export interface PreparedOrbit {
  /** Semi-major axis in scene units. */
  a: number;
  e: number;
  /** Mean motion, rad/s. */
  n: number;
  /** Mean anomaly at epoch, rad. */
  m0: number;
  /** Perifocal P axis (toward periapsis), already in scene axes. */
  px: number; py: number; pz: number;
  /** Perifocal Q axis (90° ahead in the orbit plane), already in scene axes. */
  qx: number; qy: number; qz: number;
}

/**
 * Build the perifocal -> scene rotation basis for an element set.
 *
 * The classical 3-1-3 rotation R_z(Ω)·R_x(i)·R_z(ω) is expanded into the P
 * and Q column vectors, then remapped into Three.js axes (ecliptic Z-north
 * becomes scene Y-up) once, here — so no renderer has to know about the
 * axis convention.
 */
export function prepareOrbit(orbit: OrbitElementsInput): PreparedOrbit {
  const inc = orbit.inc_deg * DEG;
  const raan = orbit.raan_deg * DEG;
  const w = argumentOfPeriapsisDeg(orbit) * DEG;

  const cw = Math.cos(w), sw = Math.sin(w);
  const cr = Math.cos(raan), sr = Math.sin(raan);
  const ci = Math.cos(inc), si = Math.sin(inc);

  // Perifocal basis in ecliptic frame (X, Y in plane of ecliptic, Z north).
  const Px = cw * cr - sw * ci * sr;
  const Py = cw * sr + sw * ci * cr;
  const Pz = sw * si;

  const Qx = -sw * cr - cw * ci * sr;
  const Qy = -sw * sr + cw * ci * cr;
  const Qz = cw * si;

  const periodSec = Math.max(1e-6, orbit.period_days * 86400);

  return {
    a: kmToScene(orbit.a_km),
    e: orbit.e,
    n: TWO_PI / periodSec,
    m0: orbit.m0_deg * DEG,
    // Ecliptic (X, Y, Z-north) -> scene (X, Y-up, Z).
    // Z_ecl -> +Y_scene, Y_ecl -> -Z_scene. The negation keeps the map a
    // rotation rather than a reflection; without it every orbit is mirrored.
    px: Px, py: Pz, pz: -Py,
    qx: Qx, qy: Qz, qz: -Qy,
  };
}

/**
 * Write the scene-space position of a prepared orbit at `simTimeSec` into
 * `out`. Allocation-free — safe to call once per instance per frame.
 *
 * Positions are relative to the orbit's focus. For a heliocentric orbit the
 * focus is the scene origin; for a satellite orbit the caller adds the
 * parent's world position.
 */
export function orbitPositionInto(o: PreparedOrbit, simTimeSec: number, out: THREE.Vector3): THREE.Vector3 {
  const M = o.m0 + o.n * simTimeSec;
  const E = solveKepler(M, o.e);

  // True anomaly from eccentric anomaly, via the half-angle form (stable at
  // high eccentricity).
  const nu = 2 * Math.atan2(
    Math.sqrt(1 + o.e) * Math.sin(E / 2),
    Math.sqrt(1 - o.e) * Math.cos(E / 2)
  );
  const r = o.a * (1 - o.e * Math.cos(E));

  const xp = r * Math.cos(nu);
  const yp = r * Math.sin(nu);

  return out.set(
    xp * o.px + yp * o.qx,
    xp * o.py + yp * o.qy,
    xp * o.pz + yp * o.qz
  );
}

/** Allocating convenience wrapper around {@link orbitPositionInto}. */
export function orbitPosition(o: PreparedOrbit, simTimeSec: number): THREE.Vector3 {
  return orbitPositionInto(o, simTimeSec, new THREE.Vector3());
}

/**
 * Closed 3D ellipse for an orbit path line, in scene units relative to the
 * orbit's focus.
 *
 * Sampled uniformly in eccentric anomaly, which distributes points more
 * densely near periapsis where the curvature is highest — so eccentric
 * orbits stay smooth without raising the segment count.
 */
export function orbitPathPoints(o: PreparedOrbit, segments = 256): THREE.Vector3[] {
  const points: THREE.Vector3[] = new Array(segments + 1);
  const sqrt1pe = Math.sqrt(1 + o.e);
  const sqrt1me = Math.sqrt(1 - o.e);

  for (let i = 0; i <= segments; i++) {
    const E = (i / segments) * TWO_PI;
    const nu = 2 * Math.atan2(sqrt1pe * Math.sin(E / 2), sqrt1me * Math.cos(E / 2));
    const r = o.a * (1 - o.e * Math.cos(E));
    const xp = r * Math.cos(nu);
    const yp = r * Math.sin(nu);

    points[i] = new THREE.Vector3(
      xp * o.px + yp * o.qx,
      xp * o.py + yp * o.qy,
      xp * o.pz + yp * o.qz
    );
  }
  return points;
}

/** Periapsis and apoapsis distances in scene units. */
export function apsides(o: PreparedOrbit): { periapsis: number; apoapsis: number } {
  return { periapsis: o.a * (1 - o.e), apoapsis: o.a * (1 + o.e) };
}

// ─── CONVENIENCE WRAPPERS ───────────────────────────────────────────────

/**
 * One-shot position for an element set. Prefer {@link prepareOrbit} +
 * {@link orbitPositionInto} in a render loop; this exists for one-off calls
 * and for callers that hold no per-object state.
 */
export function orbitElementsToScenePosition(
  orbit: OrbitElementsInput,
  simTimeSec: number
): THREE.Vector3 {
  return orbitPosition(prepareOrbit(orbit), simTimeSec);
}

/** One-shot orbit path for an element set. */
export function orbitElementsToPathPoints(
  orbit: OrbitElementsInput,
  segments = 256
): THREE.Vector3[] {
  return orbitPathPoints(prepareOrbit(orbit), segments);
}
