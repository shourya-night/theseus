/**
 * THESEUS Zodiacal Dust Renderer
 * ===============================
 * Faint particle disc in the ecliptic plane (0.1–3 AU), representing the
 * interplanetary dust cloud responsible for the zodiacal light.
 *
 * This is a CONCEPTUAL MODEL visualization. Particles are static (no Kepler
 * propagation — dust grains are sub-millimetre and have complex non-Keplerian
 * dynamics). The canonical scale pipeline is used for correct placement.
 *
 * Very low opacity and additive blending give a subtle luminous haze in the
 * inner solar system.
 */

import * as THREE from 'three';
import { auToScene } from '../CoordinateSystem';

/** Deterministic pseudo-random source (Mulberry32). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class ZodiacalDustRenderer {
  readonly points: THREE.Points;

  constructor(count = 4000, seed = 0xDEAD) {
    const rand = mulberry32(seed);

    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);

    const innerScene = auToScene(0.1);
    const outerScene = auToScene(3.0);

    for (let i = 0; i < count; i++) {
      // Radial distribution: density falls off as ~1/r, so use sqrt for
      // visual weighting (more particles near the Sun).
      const r = innerScene + Math.pow(rand(), 0.5) * (outerScene - innerScene);

      const theta = rand() * Math.PI * 2;

      // Thin disc: scatter above/below the ecliptic by a few degrees.
      // The zodiacal dust cloud has a half-thickness of ~10° at 1 AU,
      // narrowing inward.
      const heightFrac = (rand() - 0.5) * 0.12 * r / auToScene(1);

      // Scene axes: X right, Y up, Z toward viewer. The ecliptic plane
      // maps to the X–Z plane (Y is up = ecliptic north).
      positions[i * 3]     = r * Math.cos(theta);
      positions[i * 3 + 1] = heightFrac;
      positions[i * 3 + 2] = r * Math.sin(theta);
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const material = new THREE.PointsMaterial({
      color: 0xc8b898,
      size: 1.0,
      sizeAttenuation: false,
      transparent: true,
      opacity: 0.12,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    this.points = new THREE.Points(geometry, material);
    this.points.name = 'ZodiacalDustMesh';
    this.points.visible = false; // off by default
  }

  setVisible(visible: boolean): void {
    this.points.visible = visible;
  }

  dispose(): void {
    this.points.geometry.dispose();
    (this.points.material as THREE.Material).dispose();
  }
}
