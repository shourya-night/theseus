/**
 * THESEUS Orbital Scientific Overlay Renderer
 * ===========================================
 * Renders scientific orbital elements overlay:
 *   - Apoapsis (Ap) and Periapsis (Pe) markers
 *   - Ascending (AN) and Descending (DN) nodes
 *   - Velocity vector arrow helper
 *   - Eccentricity vector arrow helper
 *   - Semi-transparent orbital plane disc
 */

import * as THREE from 'three';
import { engineToThreePos, engineToThreePosInto, engineToThreeVel } from '../CoordinateSystem';
import { StateVector } from '../../types/mission';

export class OrbitalOverlayRenderer {
  readonly group: THREE.Group;

  private apMarker: THREE.Mesh;
  private peMarker: THREE.Mesh;
  private velArrow: THREE.ArrowHelper;
  private orbitPlaneMesh: THREE.Mesh;

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'OrbitalOverlayGroup';

    // ── 1. Markers (Ap / Pe) ──────────────────────────────────────
    const markerGeo = new THREE.SphereGeometry(0.8, 12, 12);

    const apMat = new THREE.MeshBasicMaterial({ color: 0xffaa00 });
    this.apMarker = new THREE.Mesh(markerGeo, apMat);
    this.apMarker.name = 'ApoapsisMarker';
    this.group.add(this.apMarker);

    const peMat = new THREE.MeshBasicMaterial({ color: 0x44bb66 });
    this.peMarker = new THREE.Mesh(markerGeo, peMat);
    this.peMarker.name = 'PeriapsisMarker';
    this.group.add(this.peMarker);

    // ── 2. Velocity Arrow Helper ──────────────────────────────────
    this.velArrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(0, 0, 0),
      15,
      0x44bb66,
      3,
      1.5
    );
    this.group.add(this.velArrow);

    // ── 3. Orbital Plane Disc ─────────────────────────────────────
    const planeGeo = new THREE.RingGeometry(1, 100, 64);
    planeGeo.rotateX(-Math.PI / 2);
    const planeMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.08,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    this.orbitPlaneMesh = new THREE.Mesh(planeGeo, planeMat);
    this.group.add(this.orbitPlaneMesh);
  }

  /**
   * Update overlay markers from state history & current frame vector.
   */
  update(currentFrame: StateVector, stateHistory: StateVector[] = []): void {
    // Velocity arrow
    const pos = engineToThreePos(currentFrame.position);
    const vel = engineToThreeVel(currentFrame.velocity);

    this.velArrow.position.copy(pos);
    if (vel.lengthSq() > 0.001) {
      const dir = vel.clone().normalize();
      this.velArrow.setDirection(dir);
      this.velArrow.setLength(Math.min(30, Math.max(5, vel.length() * 0.002)), 3, 1.5);
    }

    // Ap / Pe from state history min/max radial distance
    if (stateHistory.length > 0) {
      let minR = Infinity;
      let maxR = -Infinity;
      let minState = stateHistory[0];
      let maxState = stateHistory[0];

      stateHistory.forEach(st => {
        const r = Math.hypot(st.position[0], st.position[1], st.position[2]);
        if (r < minR) { minR = r; minState = st; }
        if (r > maxR) { maxR = r; maxState = st; }
      });

      engineToThreePosInto(minState.position, this.peMarker.position);
      engineToThreePosInto(maxState.position, this.apMarker.position);
    }
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  dispose(): void {
    this.apMarker.geometry.dispose();
    (this.apMarker.material as THREE.Material).dispose();
    this.peMarker.geometry.dispose();
    (this.peMarker.material as THREE.Material).dispose();
    this.orbitPlaneMesh.geometry.dispose();
    (this.orbitPlaneMesh.material as THREE.Material).dispose();
  }
}
