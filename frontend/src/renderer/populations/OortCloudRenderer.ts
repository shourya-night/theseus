/**
 * THESEUS Oort Cloud Renderer
 * ===========================
 * Conceptual spherical cloud model representing the theoretical Oort Cloud.
 *
 * IMPORTANT: This is a STATISTICAL MODEL visualization, clearly labeled as
 * conceptual. Individual objects in the real Oort Cloud have never been
 * directly observed — this renders a representative spherical distribution.
 *
 * Unlike the belt populations, Oort Cloud members have orbital periods
 * measured in millions of years. Per-frame Kepler propagation is not
 * meaningful on human timescales, so positions are static. The canonical
 * scale pipeline (auToScene) is still used so that the cloud sits at the
 * correct distance relative to the rest of the solar system.
 *
 * Rendering extent is capped at 2,000–10,000 AU for visual accessibility.
 * The full theoretical cloud (50,000+ AU) extends far beyond practical
 * camera framing.
 */

import * as THREE from 'three';
import { OORT_CLOUD_PARAMS } from '../../data/smallBodies';
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

export class OortCloudRenderer {
  readonly points: THREE.Points;

  constructor(count = OORT_CLOUD_PARAMS.count, seed = 0xCAFE) {
    const rand = mulberry32(seed);

    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);

    // Use scene-consistent scale. Cap outer radius for visual accessibility.
    const innerScene = auToScene(OORT_CLOUD_PARAMS.innerRadius_AU);
    const outerScene = auToScene(Math.min(OORT_CLOUD_PARAMS.outerRadius_AU, 10000));

    for (let i = 0; i < count; i++) {
      // Logarithmic radial distribution for the outer cloud, ensuring
      // density falls off with distance as expected for a 1/r² distribution.
      const r = innerScene * Math.exp(rand() * Math.log(outerScene / innerScene));

      // Isotropic: uniform on the sphere via Archimedes' theorem.
      const u = rand() * 2 - 1;
      const phi = rand() * Math.PI * 2;
      const sinTheta = Math.sqrt(Math.max(0, 1 - u * u));

      // Scene axes: X right, Y up, Z toward viewer. The cloud is isotropic
      // so no axis remap is needed beyond using the correct scale.
      positions[i * 3]     = r * sinTheta * Math.cos(phi);
      positions[i * 3 + 1] = r * u;
      positions[i * 3 + 2] = r * sinTheta * Math.sin(phi);
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const material = new THREE.PointsMaterial({
      color: new THREE.Color(...OORT_CLOUD_PARAMS.color),
      size: 1.5,
      sizeAttenuation: false,
      transparent: true,
      opacity: 0.30,
    });

    this.points = new THREE.Points(geometry, material);
    this.points.name = 'OortCloudMesh';
    // Members span 2,000–10,000 AU. Default to invisible (toggle is off).
    this.points.visible = false;
  }

  setVisible(visible: boolean): void {
    this.points.visible = visible;
  }

  dispose(): void {
    this.points.geometry.dispose();
    (this.points.material as THREE.Material).dispose();
  }
}
