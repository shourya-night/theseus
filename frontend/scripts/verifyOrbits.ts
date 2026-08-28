/**
 * THESEUS Orbit Pipeline Verification (headless)
 * ==============================================
 * Asserts the physical invariants of the rendering orbit pipeline without a
 * browser: Kepler solver accuracy across the catalog's eccentricity range,
 * every body lying on its own drawn orbit path, apsides matching a(1∓e),
 * orbit-plane inclinations matching the catalog, the ϖ -> ω derivation, and
 * period closure.
 *
 * This checks the FRONTEND orbit renderer only. It does not touch, import, or
 * validate ORBIT-X.
 *
 *   cd frontend
 *   npx tsx scripts/verifyOrbits.ts
 *
 * Exits non-zero on any failure, so it can gate a change to
 * renderer/CoordinateSystem.ts or to the catalog element sets.
 */
import * as THREE from 'three';
import {
  prepareOrbit, orbitPositionInto, orbitPathPoints,
  argumentOfPeriapsisDeg, solveKepler, sceneToAu, AU_SCENE,
  engineToThreePos, threeToEnginePos, isEllipticalOrbit,
} from '../src/renderer/CoordinateSystem';
import { CATALOG_ARTIFICIAL_OBJECTS } from '../src/data/artificialObjects';
import { SOLAR_SYSTEM_OBJECTS } from '../src/data/astronomicalObjects';
import { NAMED_COMETS, NAMED_ASTEROIDS } from '../src/data/smallBodies';
import {
  ViewContext, MIN_APPARENT_RADIUS_PX,
  apparentRadiusPixels, visualRadiusScene, visualScaleMultiplier,
  spacecraftCharacteristicRadiusM,
} from '../src/renderer/VisualScale';
import { kmToScene } from '../src/renderer/CoordinateSystem';

let fails = 0;
const rows: string[] = [];
function check(name: string, ok: boolean, detail: string) {
  if (!ok) fails++;
  rows.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(46)} ${detail}`);
}

// distance from point p to a polyline
function distToPath(p: THREE.Vector3, path: THREE.Vector3[]): number {
  let best = Infinity;
  const ab = new THREE.Vector3(), ap = new THREE.Vector3(), q = new THREE.Vector3();
  for (let i = 0; i < path.length - 1; i++) {
    ab.subVectors(path[i + 1], path[i]);
    ap.subVectors(p, path[i]);
    const t = Math.max(0, Math.min(1, ap.dot(ab) / Math.max(1e-12, ab.lengthSq())));
    q.copy(path[i]).addScaledVector(ab, t);
    best = Math.min(best, q.distanceTo(p));
  }
  return best;
}

const bodies = SOLAR_SYSTEM_OBJECTS.filter(o => o.orbit);
const scratch = new THREE.Vector3();

// ── 1. Kepler solver accuracy across the eccentricity range ──────────
for (const e of [0, 0.0167, 0.2056, 0.6, 0.9671, 0.9951, 0.9998]) {
  let worst = 0;
  for (let k = 0; k < 400; k++) {
    const M = (k / 400) * 2 * Math.PI;
    const E = solveKepler(M, e);
    const residual = Math.abs(((E - e * Math.sin(E) - M) + Math.PI) % (2 * Math.PI) - Math.PI);
    worst = Math.max(worst, residual);
  }
  check(`Kepler residual e=${e}`, worst < 1e-8, `max |E - e sinE - M| = ${worst.toExponential(2)}`);
}

// ── 2. Every body sits on its own drawn orbit (§49.1) ────────────────
for (const b of bodies) {
  const o = prepareOrbit(b.orbit!);
  const path = orbitPathPoints(o, 512);
  let worstRatio = 0;
  for (let k = 0; k < 24; k++) {
    const t = (k / 24) * b.orbit!.period_days * 86400;
    orbitPositionInto(o, t, scratch);
    worstRatio = Math.max(worstRatio, distToPath(scratch, path) / o.a);
  }
  check(`${b.name} lies on its own orbit path`, worstRatio < 1e-4,
        `max offset = ${(worstRatio * 100).toExponential(2)}% of a`);
}

for (const c of NAMED_COMETS) {
  const o = prepareOrbit(c.orbit!);
  const path = orbitPathPoints(o, 512);
  let worstRatio = 0;
  for (let k = 0; k < 24; k++) {
    orbitPositionInto(o, (k / 24) * c.orbit!.period_days * 86400, scratch);
    worstRatio = Math.max(worstRatio, distToPath(scratch, path) / o.a);
  }
  check(`${c.name.slice(0, 28)} on its own orbit`, worstRatio < 2e-3,
        `max offset = ${(worstRatio * 100).toExponential(2)}% of a  (e=${c.orbit!.e})`);
}

// ── 3. Eccentricity is actually represented (§49.3) ──────────────────
for (const id of ['earth', 'mercury', 'pluto']) {
  const b = bodies.find(x => x.id === id)!;
  const o = prepareOrbit(b.orbit!);
  let rmin = Infinity, rmax = 0;
  for (let k = 0; k < 2000; k++) {
    orbitPositionInto(o, (k / 2000) * b.orbit!.period_days * 86400, scratch);
    const r = scratch.length();
    rmin = Math.min(rmin, r); rmax = Math.max(rmax, r);
  }
  const expMin = o.a * (1 - o.e), expMax = o.a * (1 + o.e);
  const ok = Math.abs(rmin - expMin) / expMin < 2e-3 && Math.abs(rmax - expMax) / expMax < 2e-3;
  check(`${b.name} apsides match a(1∓e)`, ok,
        `r = ${sceneToAu(rmin).toFixed(4)}..${sceneToAu(rmax).toFixed(4)} AU (expect ${sceneToAu(expMin).toFixed(4)}..${sceneToAu(expMax).toFixed(4)})`);
}

// ── 4. Orbits are not parallel / coplanar-degenerate (§49.5, §49.6) ──
const normals = new Map<string, THREE.Vector3>();
for (const b of bodies) {
  const o = prepareOrbit(b.orbit!);
  const P = new THREE.Vector3(o.px, o.py, o.pz);
  const Q = new THREE.Vector3(o.qx, o.qy, o.qz);
  normals.set(b.id, new THREE.Vector3().crossVectors(P, Q).normalize());
}
const earthN = normals.get('earth')!;
for (const id of ['mercury', 'pluto']) {
  const ang = THREE.MathUtils.radToDeg(Math.acos(Math.min(1, Math.abs(earthN.dot(normals.get(id)!)))));
  const expected = SOLAR_SYSTEM_OBJECTS.find(o => o.id === id)!.orbit!.inc_deg;
  check(`${id} orbit plane tilt vs Earth`, Math.abs(ang - expected) < 0.5,
        `${ang.toFixed(3)}° (catalog inclination ${expected}°)`);
}

// ── 5. ω derived from ϖ, not used raw (§ catalog convention) ─────────
const earthOrb = SOLAR_SYSTEM_OBJECTS.find(o => o.id === 'earth')!.orbit!;
const wEarth = argumentOfPeriapsisDeg(earthOrb);
check('Earth ω derived as ϖ − Ω', Math.abs(wEarth - 114.20783) < 1e-3,
      `ω = ${wEarth.toFixed(5)}° from ϖ=${(earthOrb as any).varpi_deg}, Ω=${earthOrb.raan_deg}`);
const ceresOrb = SOLAR_SYSTEM_OBJECTS.find(o => o.id === 'ceres')!.orbit!;
check('Ceres ω taken directly (SBDB convention)', Math.abs(argumentOfPeriapsisDeg(ceresOrb) - 73.597) < 1e-6,
      `ω = ${argumentOfPeriapsisDeg(ceresOrb).toFixed(3)}°`);

// ── 6. Bodies actually move (the un-freezing fix) ────────────────────
const eo = prepareOrbit(earthOrb);
const p0 = new THREE.Vector3(), p90 = new THREE.Vector3(), p365 = new THREE.Vector3();
orbitPositionInto(eo, 0, p0);
orbitPositionInto(eo, 91.31 * 86400, p90);
orbitPositionInto(eo, earthOrb.period_days * 86400, p365);
check('Earth advances ~90° in a quarter period',
      Math.abs(THREE.MathUtils.radToDeg(p0.angleTo(p90)) - 90) < 2.5,
      `swept ${THREE.MathUtils.radToDeg(p0.angleTo(p90)).toFixed(2)}°`);
check('Earth returns to start after one period', p0.distanceTo(p365) / eo.a < 1e-6,
      `closure error = ${(p0.distanceTo(p365) / eo.a).toExponential(2)} of a`);

// ── 7. Scale sanity ──────────────────────────────────────────────────
check('1 AU = 14959.787 scene units', Math.abs(AU_SCENE - 14959.78707) < 1e-3, `${AU_SCENE.toFixed(5)}`);
orbitPositionInto(eo, 0, scratch);
check('Earth is ~1 AU from origin', Math.abs(sceneToAu(scratch.length()) - 1) < 0.02,
      `${sceneToAu(scratch.length()).toFixed(5)} AU`);

// ── 7b. HANDEDNESS ──────────────────────────────────────────────────
// The ecliptic -> scene axis map must be a ROTATION, not a reflection.
// A reflection mirrors the entire solar system: prograde reads as retrograde
// and every chiral relationship is inverted. This is the assertion that stops
// the negated Z in CoordinateSystem from being "tidied away".
{
  // Basis images under the transform, read straight from the converter.
  const ex = engineToThreePos([1, 0, 0]);
  const ey = engineToThreePos([0, 1, 0]);
  const ez = engineToThreePos([0, 0, 1]);
  const det = ex.dot(new THREE.Vector3().crossVectors(ey, ez));
  check('axis map is a rotation, not a reflection', det > 0,
        `determinant sign = ${Math.sign(det)} (${det > 0 ? 'rotation' : 'REFLECTION'})`);

  check('engine Z (north) maps to scene +Y',
        Math.abs(ez.y - 1e-7 * 1) < 1e-12 && Math.abs(ez.x) < 1e-30 && Math.abs(ez.z) < 1e-30,
        `e_z -> (${ez.x}, ${ez.y}, ${ez.z})`);

  // Round trip must be the identity.
  const p: [number, number, number] = [1.234e11, -5.678e10, 9.012e9];
  const back = threeToEnginePos(engineToThreePos(p));
  const rt = Math.max(...back.map((v, i) => Math.abs(v - p[i]) / Math.abs(p[i])));
  check('engine -> scene -> engine round trip', rt < 1e-12, `max relative error = ${rt.toExponential(2)}`);

  // Every prograde catalog orbit must have r x v pointing at ecliptic north.
  // Note: smallBodies.ts calls registerObjects() on import, so SOLAR_SYSTEM_OBJECTS
  // here also contains the named asteroids and comets — the check covers them too.
  // The tightest margin is Hale-Bopp at i = 89.43 deg, cos(i) = 0.0099.
  let worstProgradeY = 1;
  const offenders: string[] = [];
  for (const b of bodies) {
    const o = prepareOrbit(b.orbit!);
    const T = b.orbit!.period_days * 86400;
    const r0 = new THREE.Vector3(), r1 = new THREE.Vector3();
    orbitPositionInto(o, 0, r0);
    orbitPositionInto(o, T * 1e-4, r1);
    const h = new THREE.Vector3().crossVectors(r0, new THREE.Vector3().subVectors(r1, r0)).normalize();
    // Catalog inclination < 90 deg => prograde => h.y must be positive.
    if (b.orbit!.inc_deg < 90) {
      if (h.y <= 0) offenders.push(b.name);
      worstProgradeY = Math.min(worstProgradeY, h.y);
    }
  }
  check('all prograde bodies orbit prograde as drawn', offenders.length === 0,
        offenders.length ? `mirrored: ${offenders.join(', ')}` : `min h·ŷ = ${worstProgradeY.toFixed(4)}`);

  // Halley is genuinely retrograde and must read that way.
  const halley = NAMED_COMETS.find(c => c.id === 'halley');
  if (halley) {
    const o = prepareOrbit(halley.orbit!);
    const T = halley.orbit!.period_days * 86400;
    const r0 = new THREE.Vector3(), r1 = new THREE.Vector3();
    orbitPositionInto(o, 0, r0);
    orbitPositionInto(o, T * 1e-5, r1);
    const h = new THREE.Vector3().crossVectors(r0, new THREE.Vector3().subVectors(r1, r0)).normalize();
    check('Halley (i = 162 deg) reads as retrograde', h.y < 0, `h·ŷ = ${h.y.toFixed(4)}`);
  }
}

// ── 7c. Non-elliptical orbits must be rejected, not faked ───────────
{
  const bad = CATALOG_ARTIFICIAL_OBJECTS.filter((o: any) => o.orbit && !isEllipticalOrbit(o.orbit));
  check('escape trajectories are detected as non-elliptical', bad.length === 2,
        bad.map((o: any) => `${o.name} (e=${o.orbit.e})`).join('; ') || 'none found');
  for (const o of CATALOG_ARTIFICIAL_OBJECTS as any[]) {
    if (o.orbit && isEllipticalOrbit(o.orbit)) {
      const pos = new THREE.Vector3();
      orbitPositionInto(prepareOrbit(o.orbit), 0, pos);
      check(`${o.name.slice(0, 30)} is not at the focus`, pos.length() > 1e-6,
            `r = ${pos.length().toFixed(4)} scene units from its parent`);
    }
  }
}

// ── 8. Visual scale policy (step 5) ─────────────────────────────────
{
  const camera = new THREE.PerspectiveCamera(50, 16 / 9, 0.01, 1e9);
  const ctx: ViewContext = { camera, viewportHeightPx: 900 };
  const at = (d: number) => { camera.position.set(0, 0, 0); return new THREE.Vector3(0, 0, d); };

  // The multiplier may enlarge, never shrink.
  let minSeen = Infinity;
  for (const d of [1e-3, 1, 100, AU_SCENE, AU_SCENE * 30]) {
    for (const r of [1e-8, 1e-5, 0.01, 1, 100]) {
      minSeen = Math.min(minSeen, visualScaleMultiplier(r, at(d), 3, ctx));
    }
  }
  check('visual multiplier never shrinks an object', minSeen >= 1, `min multiplier = ${minSeen}`);

  // When the rule binds, the drawn size is EXACTLY the minimum, not more.
  let worstPx = 0;
  for (const d of [1, 1000, AU_SCENE, AU_SCENE * 30]) {
    const drawn = visualRadiusScene(1e-9, at(d), MIN_APPARENT_RADIUS_PX.SMALL_BODY, ctx);
    worstPx = Math.max(worstPx, Math.abs(apparentRadiusPixels(drawn, d, ctx) - MIN_APPARENT_RADIUS_PX.SMALL_BODY));
  }
  check('bound objects render at exactly the minimum px', worstPx < 1e-6,
        `max deviation = ${worstPx.toExponential(2)} px`);

  // A planet close up is already big enough: multiplier must collapse to 1.
  const earth = SOLAR_SYSTEM_OBJECTS.find(o => o.id === 'earth')!;
  const earthR = kmToScene(earth.radius_km);
  const mEarth = visualScaleMultiplier(earthR, at(earthR * 5), MIN_APPARENT_RADIUS_PX.SMALL_BODY, ctx);
  check('close-up planet renders at true size', mEarth === 1,
        `multiplier = ${mEarth} at 5 radii (r = ${earthR.toFixed(4)} units)`);

  // Bennu: unfindable at true size, legible under the rule, true size up close.
  const bennu = NAMED_ASTEROIDS.find(a => a.id === 'bennu');
  if (bennu) {
    const r = kmToScene(bennu.radius_km);
    const far = AU_SCENE;
    const rawPx = apparentRadiusPixels(r, far, ctx);
    const drawnPx = apparentRadiusPixels(
      visualRadiusScene(r, at(far), MIN_APPARENT_RADIUS_PX.SMALL_BODY, ctx), far, ctx);
    check('Bennu is sub-pixel at true size, 1 AU away', rawPx < 0.01, `${rawPx.toExponential(2)} px`);
    check('Bennu reaches the legibility minimum', Math.abs(drawnPx - MIN_APPARENT_RADIUS_PX.SMALL_BODY) < 1e-6,
          `${drawnPx.toFixed(3)} px`);
    const mClose = visualScaleMultiplier(r, at(r * 3), MIN_APPARENT_RADIUS_PX.SMALL_BODY, ctx);
    check('Bennu renders true-size when close', mClose === 1, `multiplier = ${mClose} at 3 radii`);
  }

  // Spacecraft characteristic radius is derived, not invented.
  const apolloA = 12.0; // Apollo CSM cross_section_area_m2, cited in the presets
  const rM = spacecraftCharacteristicRadiusM(apolloA)!;
  check('spacecraft radius derived from cross-section', Math.abs(rM - Math.sqrt(12 / Math.PI)) < 1e-9,
        `r = ${rM.toFixed(3)} m from A = ${apolloA} m²`);
  check('missing cross-section yields null, not a guess',
        spacecraftCharacteristicRadiusM(undefined) === null, 'null');

  // Degenerate inputs must not produce NaN scales.
  const bad = [visualScaleMultiplier(0, at(1), 3, ctx), visualScaleMultiplier(-1, at(1), 3, ctx),
               visualScaleMultiplier(1, at(0), 3, ctx)];
  check('degenerate inputs stay finite', bad.every(v => Number.isFinite(v) && v >= 1),
        `[${bad.join(', ')}]`);
}

console.log(rows.join('\n'));
console.log(`\n${rows.length - fails}/${rows.length} checks passed`);
process.exit(fails ? 1 : 0);
