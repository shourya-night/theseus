/**
 * THESEUS Sun Renderer
 * ====================
 * Scientific representation of the Sun.
 * Features photospheric granulation, sunspots, limb darkening, and corona glow shell.
 * Uses procedural GLSL shaders (no external textures).
 * Scientifically restrained — no giant neon glow balls.
 */

import * as THREE from 'three';
import { GLSL_SUN_VERTEX, GLSL_SUN_FRAGMENT, GLSL_CORONA_FRAGMENT, GLSL_ATMOSPHERE_VERTEX } from '../ShaderLib';
import { kmToScene } from '../CoordinateSystem';

/**
 * Visual exaggeration applied to the solar photosphere radius.
 *
 * 1.0 = true physical scale (696,340 km -> 69.634 scene units). This constant
 * exists so that any future readability adjustment is a single declared,
 * reviewable number rather than a literal buried in a constructor. It is a
 * VISUAL SIZE factor: it scales geometry only and never moves anything.
 */
export const SUN_VISUAL_RADIUS_EXAGGERATION = 1.0;

/**
 * Outer edge of the rendered corona shell, in photosphere radii.
 *
 * The real K-corona is traceable to many solar radii and has no sharp
 * boundary; this shell is a bounded stand-in for it, deliberately kept close
 * so it reads as a limb glow rather than a glowing ball.
 */
export const CORONA_SHELL_RADII = 2.6;

/**
 * Photospheric radiance multiplier.
 *
 * The renderer uses ACES filmic tone mapping, which maps values above 1.0 into
 * the highlight rolloff. A star whose fragments never exceed display white
 * tone-maps to a flat mid-orange disc; pushing it above 1 is what makes it
 * read as a light source rather than a painted ball.
 */
export const PHOTOSPHERE_INTENSITY = 1.35;

export class SunRenderer {
  readonly group: THREE.Group;
  readonly mesh: THREE.Mesh;
  readonly coronaMesh: THREE.Mesh;
  readonly light: THREE.PointLight;

  private surfaceMaterial: THREE.ShaderMaterial;
  private coronaMaterial: THREE.ShaderMaterial;
  private radiusScene: number;

  /** True physical radius in scene units, before any visual exaggeration. */
  readonly physicalRadiusScene: number;

  constructor(radiusKm = 696340) {
    // Physical scale, through the canonical converter. The previous
    // log10(radiusKm) * 2.5 expression rendered the Sun at roughly a fifth of
    // its true size, which removed the only absolute size reference in frame.
    this.physicalRadiusScene = kmToScene(radiusKm);
    this.radiusScene = this.physicalRadiusScene * SUN_VISUAL_RADIUS_EXAGGERATION;

    this.group = new THREE.Group();
    this.group.name = 'SunGroup';

    // ── 1. Solar Surface Mesh ─────────────────────────────────────
    const geometry = new THREE.SphereGeometry(this.radiusScene, 64, 64);
    this.surfaceMaterial = new THREE.ShaderMaterial({
      vertexShader: GLSL_SUN_VERTEX,
      fragmentShader: GLSL_SUN_FRAGMENT,
      uniforms: {
        uTime: { value: 0 },
        uIntensity: { value: PHOTOSPHERE_INTENSITY },
      },
    });
    this.mesh = new THREE.Mesh(geometry, this.surfaceMaterial);
    this.mesh.name = 'SunSurface';
    this.group.add(this.mesh);

    // ── 2. Solar Corona Shell ──────────────────────────────────────
    const coronaRadius = this.radiusScene * CORONA_SHELL_RADII;
    const coronaGeo = new THREE.SphereGeometry(coronaRadius, 48, 48);
    this.coronaMaterial = new THREE.ShaderMaterial({
      vertexShader: GLSL_ATMOSPHERE_VERTEX,
      fragmentShader: GLSL_CORONA_FRAGMENT,
      uniforms: {
        uTime: { value: 0 },
        uSunColor: { value: new THREE.Color(1.0, 0.78, 0.42) },
        // The Sun sits at the scene origin; passed explicitly rather than
        // assumed, so the glow stays anchored if that ever changes.
        uSunCenter: { value: new THREE.Vector3(0, 0, 0) },
        uPhotosphereRadius: { value: this.radiusScene },
        uCoronaRadius: { value: coronaRadius },
      },
      transparent: true,
      blending: THREE.AdditiveBlending,
      // FRONT side. The glow is anchored to the photosphere's apparent edge by
      // the impact-parameter calculation in the shader, not to this shell's
      // silhouette, so the shell is only a volume to rasterise within.
      side: THREE.FrontSide,
      depthWrite: false,
    });

    this.coronaMesh = new THREE.Mesh(coronaGeo, this.coronaMaterial);
    this.coronaMesh.name = 'SunCorona';
    this.group.add(this.coronaMesh);

    // ── 3. Solar Point Light ──────────────────────────────────────
    this.light = new THREE.PointLight(0xfff4e6, 2.5, 0, 0);
    this.group.add(this.light);
  }

  /** Radius the photosphere is actually drawn at, in scene units. */
  get visualRadiusScene(): number {
    return this.radiusScene;
  }

  /** Radius of the rendered corona shell, in scene units. */
  get coronaRadiusScene(): number {
    return this.radiusScene * CORONA_SHELL_RADII;
  }

  update(timeSeconds: number): void {
    this.surfaceMaterial.uniforms.uTime.value = timeSeconds;
    this.coronaMaterial.uniforms.uTime.value = timeSeconds;

    // Sidereal rotation at the equator: 25.38 days. The corona shell is NOT
    // rotated — it is a line-of-sight effect, not a surface.
    this.mesh.rotation.y = (2 * Math.PI * timeSeconds) / (25.38 * 86400);
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.surfaceMaterial.dispose();
    this.coronaMesh.geometry.dispose();
    this.coronaMaterial.dispose();
  }
}
