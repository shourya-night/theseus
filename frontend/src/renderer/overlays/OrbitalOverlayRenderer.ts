/**
 * THESEUS Orbital Scientific Overlay Renderer
 * ===========================================
 * Renders scientific orbital elements overlay:
 *   - Velocity vector arrow helper
 *   - Eccentricity vector arrow helper
 *
 * This is telemetry only. It must never provide a second spacecraft visual.
 */

import * as THREE from 'three';
import { engineToThreePos, engineToThreeVel } from '../CoordinateSystem';
import { StateVector } from '../../types/mission';

export class OrbitalOverlayRenderer {
  readonly group: THREE.Group;

  private velArrow: THREE.ArrowHelper;

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'OrbitalOverlayGroup';

    // ── Velocity Arrow Helper (telemetry, never the spacecraft) ───
    this.velArrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(0, 0, 0),
      15,
      0x44bb66,
      3,
      1.5
    );
    this.group.add(this.velArrow);

  }

  /**
   * Update overlay markers from state history & current frame vector.
   */
  update(currentFrame: StateVector, _stateHistory: StateVector[] = []): void {
    // Velocity arrow
    const pos = engineToThreePos(currentFrame.position);
    const vel = engineToThreeVel(currentFrame.velocity);

    this.velArrow.position.copy(pos);
    if (vel.lengthSq() > 0.001) {
      const dir = vel.clone().normalize();
      this.velArrow.setDirection(dir);
      this.velArrow.setLength(Math.min(30, Math.max(5, vel.length() * 0.002)), 3, 1.5);
    }
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  dispose(): void {
  }
}
