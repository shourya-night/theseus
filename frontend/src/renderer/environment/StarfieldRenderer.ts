/**
 * THESEUS Dense Starfield Renderer
 * ================================
 * Astronomical-grade deep space starfield.
 *   - 12,000+ stars with a realistic stellar magnitude distribution
 *   - Restrained colour-temperature variation (O, B, A, F, G, K, M classes)
 *   - Non-repeating spherical distribution
 *   - Procedural Milky Way galactic plane band structure
 *   - Shader-based size attenuation and subtle scintillation
 *
 * ─────────────────────────────────────────────────────────────────────────
 * ANCHORING
 * ─────────────────────────────────────────────────────────────────────────
 * The star shell is CAMERA-ANCHORED, not world-anchored.
 *
 * Previously the points sat on a fixed shell 5,000–15,000 scene units from
 * the scene origin. That is 50–150 million km — between Mercury and Jupiter.
 * The "sky" was therefore an object inside the solar system: at the
 * full-system framing, or with the camera focused on Neptune at 450,000
 * units, the observer stood outside the shell and the entire sky collapsed
 * into one patch of the viewport, leaving the rest black.
 *
 * The fix is anchoring only. Every frame the points object is translated to
 * the camera's position and uniformly scaled so the shell sits comfortably
 * inside the current frustum. The generated distribution, magnitudes and
 * spectral colours are untouched, and because the vertex shader sizes points
 * in screen space rather than by distance, the uniform scale has no effect on
 * apparent star size or brightness — it exists purely to keep the shell
 * between the near and far planes as the camera ranges over eleven orders of
 * magnitude.
 *
 * DEPTH: depthWrite is off, so stars never occlude anything. depthTest is
 * deliberately left ON. Because the shell is always scaled beyond all scene
 * geometry, ordinary depth testing then gives the physically correct result
 * for free — planets occlude stars, stars never occlude planets — without
 * needing to disable depth or special-case the render order. renderOrder is
 * set far negative so the field is also drawn before other transparent
 * geometry such as orbit lines.
 */

import * as THREE from 'three';
import { GLSL_STAR_VERTEX, GLSL_STAR_FRAGMENT } from '../ShaderLib';

/** Inner radius of the generated star shell, in scene units before scaling. */
export const STAR_SHELL_INNER = 5000;

/** Outer radius of the generated star shell, in scene units before scaling. */
export const STAR_SHELL_OUTER = 15000;

/**
 * Fraction of the camera's far plane at which the OUTER edge of the shell is
 * placed each frame. Below 1 so the far plane never clips the field; high
 * enough that the shell is always beyond scene geometry, which is what makes
 * depth testing produce correct occlusion.
 */
export const STAR_SHELL_FAR_FRACTION = 0.75;

export class StarfieldRenderer {
  readonly points: THREE.Points;
  private count: number;

  constructor(count = 12000) {
    this.count = count;

    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const magnitudes = new Float32Array(count);

    // Spectral class colours (RGB 0-1)
    const spectralColors: [number, number, number][] = [
      [0.65, 0.78, 1.0],  // O/B - Blue-white
      [0.85, 0.90, 1.0],  // A   - White
      [1.0,  0.98, 0.90], // F   - Yellow-white
      [1.0,  0.90, 0.70], // G   - Yellow (Sun-like)
      [1.0,  0.75, 0.50], // K   - Orange
      [1.0,  0.55, 0.40], // M   - Pale red
    ];

    const eulerGalactic = new THREE.Euler(0.45, 1.05, 0.7, 'XYZ');

    for (let i = 0; i < count; i++) {
      const r = STAR_SHELL_INNER + Math.random() * (STAR_SHELL_OUTER - STAR_SHELL_INNER);
      const posVec = new THREE.Vector3();

      if (i < count * 0.4) {
        // Dense galactic equator band (±14°)
        const galLat = (Math.random() - 0.5) * 0.5;
        const galLon = Math.random() * Math.PI * 2;
        posVec.set(
          r * Math.cos(galLat) * Math.cos(galLon),
          r * Math.sin(galLat),
          r * Math.cos(galLat) * Math.sin(galLon)
        );
        posVec.applyEuler(eulerGalactic);
      } else {
        // Isotropic uniform sky sphere
        const u = Math.random() * 2 - 1;
        const theta = Math.random() * Math.PI * 2;
        const rXY = Math.sqrt(Math.max(0, 1 - u * u));
        posVec.set(
          r * rXY * Math.cos(theta),
          r * u,
          r * rXY * Math.sin(theta)
        );
      }

      positions[i * 3] = posVec.x;
      positions[i * 3 + 1] = posVec.y;
      positions[i * 3 + 2] = posVec.z;

      // Power-law magnitude distribution: many faint stars, few bright ones.
      magnitudes[i] = Math.pow(Math.random(), 2.5) * 6.5;

      const col = spectralColors[Math.floor(Math.random() * spectralColors.length)];
      colors[i * 3] = col[0];
      colors[i * 3 + 1] = col[1];
      colors[i * 3 + 2] = col[2];
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('aColor', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('aMagnitude', new THREE.BufferAttribute(magnitudes, 1));

    const material = new THREE.ShaderMaterial({
      vertexShader: GLSL_STAR_VERTEX,
      fragmentShader: GLSL_STAR_FRAGMENT,
      transparent: true,
      depthWrite: false,
      // depthTest stays on — see the ANCHORING note above.
      depthTest: true,
      blending: THREE.AdditiveBlending,
    });

    this.points = new THREE.Points(geometry, material);
    this.points.name = 'StarfieldMesh';

    // Drawn before any other transparent geometry.
    this.points.renderOrder = -1000;

    // The object is repositioned every frame immediately before the draw, so
    // the cull test would be evaluated against a stale bounding sphere.
    this.points.frustumCulled = false;

    this.points.matrixAutoUpdate = true;
  }

  /**
   * Anchor the field to the camera. Call once per frame, before rendering.
   *
   * Translation keeps the observer at the centre of the sky in every
   * direction. The uniform scale keeps the shell between the near and far
   * planes as the camera moves between a low orbit and the outer system.
   */
  update(camera: THREE.PerspectiveCamera): void {
    this.points.position.copy(camera.position);

    const targetOuter = camera.far * STAR_SHELL_FAR_FRACTION;
    const scale = targetOuter / STAR_SHELL_OUTER;
    this.points.scale.setScalar(scale);

    this.points.updateMatrix();
    this.points.updateMatrixWorld(true);
  }

  /** Number of stars generated. */
  get starCount(): number {
    return this.count;
  }

  setVisible(visible: boolean): void {
    this.points.visible = visible;
  }

  dispose(): void {
    this.points.geometry.dispose();
    (this.points.material as THREE.Material).dispose();
  }
}
