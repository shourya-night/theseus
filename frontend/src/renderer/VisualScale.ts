/**
 * THESEUS Visual Scale Policy
 * ===========================
 * THE ONLY PLACE IN THE FRONTEND THAT MAY EXAGGERATE THE SIZE OF ANYTHING.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * THE THREE CONCEPTS, KEPT SEPARATE
 * ─────────────────────────────────────────────────────────────────────────
 *   PHYSICAL POSITION — where the object actually is. Comes from ORBIT-X or
 *                       from catalog elements via CoordinateSystem. NOTHING
 *                       in this module touches it, ever.
 *   PHYSICAL SIZE     — the object's true radius, from the catalog. Geometry
 *                       is built at this size.
 *   VISUAL SIZE       — what gets drawn. Equal to the physical size whenever
 *                       that is large enough to see, and otherwise inflated
 *                       just enough to reach a minimum apparent size on
 *                       screen.
 *
 * The inflation is a per-frame, camera-relative SCALE applied to a mesh. It
 * is never baked into a position, and never baked into geometry. Zoom in far
 * enough and the multiplier falls to exactly 1, at which point you are
 * looking at the object's true size.
 *
 * Why this has to be camera-relative: Bennu's radius is 245 m, which is
 * 2.45e-5 scene units. A fixed floor — the previous approach — made it a
 * 2,000 km sphere that stayed 2,000 km wide however close you got. A
 * camera-relative rule keeps it findable at a distance and truthful up close.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * ANYTHING REPORTING A SIZE TO THE USER MUST READ THE CATALOG, NOT THIS.
 * ─────────────────────────────────────────────────────────────────────────
 * Telemetry, the object inspector and any measurement overlay must use
 * `radius_km` from the catalog. The multipliers here exist for legibility
 * only and are not physical quantities.
 */

import * as THREE from 'three';

/** What the camera currently is, for apparent-size arithmetic. */
export interface ViewContext {
  camera: THREE.PerspectiveCamera;
  viewportHeightPx: number;
}

/**
 * Minimum apparent RADIUS, in pixels, per object class.
 *
 * These are legibility thresholds, not physics. Kept small on purpose: the
 * goal is that an object can be seen and clicked, not that it dominates the
 * frame. A few pixels is enough to find something; more than that starts to
 * misrepresent the scene.
 */
export const MIN_APPARENT_RADIUS_PX = {
  /** Mission vehicles. Slightly larger since they are the subject of the app. */
  SPACECRAFT: 6,
  /** Catalogued asteroids and NEOs. */
  SMALL_BODY: 3,
  /** Comet nuclei. The coma and tails carry the visual weight. */
  COMET_NUCLEUS: 3,
} as const;

/**
 * Static geometry floor for planets, dwarf planets and moons, in scene units.
 *
 * These bodies are built once at a fixed subdivision, so they use a floor
 * rather than a per-frame multiplier. At 0.02 scene units (200 km) it binds
 * only for small moons — every planet is already far larger, so for the
 * planets this constant has no effect at all.
 */
export const MIN_BODY_VISUAL_RADIUS_SCENE = 0.02;

/**
 * Upper bound on the visual multiplier, as a guard against a degenerate
 * camera state producing a non-finite scale. It is far above anything the
 * minimum-pixel rule produces in normal use and should never bind.
 */
export const MAX_VISUAL_MULTIPLIER = 1e9;

/**
 * Scene units per screen pixel at a given distance from the camera.
 * Small-angle form, which is exact enough well inside a degree — the regime
 * every object handled here occupies.
 */
export function sceneUnitsPerPixelAt(distance: number, ctx: ViewContext): number {
  const halfFov = (ctx.camera.fov * Math.PI) / 360;
  const halfHeightAtDistance = Math.max(1e-30, distance) * Math.tan(halfFov);
  return halfHeightAtDistance / Math.max(1, ctx.viewportHeightPx / 2);
}

/** Apparent radius in pixels of a sphere of `radiusScene` at `distance`. */
export function apparentRadiusPixels(radiusScene: number, distance: number, ctx: ViewContext): number {
  const perPixel = sceneUnitsPerPixelAt(distance, ctx);
  return perPixel > 0 ? radiusScene / perPixel : 0;
}

/** Scene-unit radius that projects to exactly `pixels` at `distance`. */
export function radiusForPixels(pixels: number, distance: number, ctx: ViewContext): number {
  return pixels * sceneUnitsPerPixelAt(distance, ctx);
}

/**
 * The radius an object should be DRAWN at: its true radius, or the radius
 * needed to reach `minPixels` on screen, whichever is larger.
 */
export function visualRadiusScene(
  physicalRadiusScene: number,
  worldPosition: THREE.Vector3,
  minPixels: number,
  ctx: ViewContext
): number {
  const distance = ctx.camera.position.distanceTo(worldPosition);
  if (!Number.isFinite(distance)) return physicalRadiusScene;
  return Math.max(physicalRadiusScene, radiusForPixels(minPixels, distance, ctx));
}

/**
 * Multiplier to apply to a mesh whose geometry was built at its TRUE physical
 * radius, so it reaches `minPixels` on screen. Always >= 1 — this rule can
 * enlarge an object, never shrink it below its real size.
 */
export function visualScaleMultiplier(
  physicalRadiusScene: number,
  worldPosition: THREE.Vector3,
  minPixels: number,
  ctx: ViewContext
): number {
  if (!(physicalRadiusScene > 0)) return 1;
  const target = visualRadiusScene(physicalRadiusScene, worldPosition, minPixels, ctx);
  const m = target / physicalRadiusScene;
  if (!Number.isFinite(m)) return 1;
  return Math.min(MAX_VISUAL_MULTIPLIER, Math.max(1, m));
}

/**
 * Characteristic radius of a spacecraft, in metres, derived from its
 * catalogued cross-sectional area: r = sqrt(A / π).
 *
 * The rocket presets carry cited `cross_section_area_m2` values but no
 * dimensions, so this is a DERIVED characteristic size, not a published one.
 * It is used only to decide when the minimum-pixel rule stops binding. In
 * practice it never does — a 3 m vehicle is 3e-7 scene units, so you would
 * have to be a few hundred metres away for its true size to exceed a few
 * pixels — but the code path exists so that physical and visual size remain
 * genuinely separate quantities rather than a notional distinction.
 *
 * Returns null when no area is catalogued; callers must then treat the drawn
 * size as purely symbolic.
 */
export function spacecraftCharacteristicRadiusM(crossSectionAreaM2?: number): number | null {
  if (typeof crossSectionAreaM2 !== 'number' || !(crossSectionAreaM2 > 0)) return null;
  return Math.sqrt(crossSectionAreaM2 / Math.PI);
}
