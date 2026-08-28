/**
 * THESEUS Dynamics Force Vector Overlay Renderer
 * ==============================================
 * Visualizes active physical force vectors acting on the spacecraft:
 *   - Central body gravity vector (yellow-orange)
 *   - Propulsion thrust vector (flame orange)
 *   - Solar Radiation Pressure / Drag vectors (cyan)
 * Only displays forces actually active in the simulation.
 */

import * as THREE from 'three';
import { engineToThreePos, engineToThreeVel } from '../CoordinateSystem';
import { StateVector } from '../../types/mission';

export class DynamicsOverlayRenderer {
  readonly group: THREE.Group;

  private gravityArrow: THREE.ArrowHelper;
  private thrustArrow: THREE.ArrowHelper;

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'DynamicsOverlayGroup';

    // Gravity vector arrow (pointing toward central body / origin)
    this.gravityArrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 0, -1),
      new THREE.Vector3(0, 0, 0),
      12,
      0xffaa00,
      2.5,
      1.2
    );
    this.group.add(this.gravityArrow);

    // Thrust vector arrow (active burn direction)
    this.thrustArrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(0, 0, 0),
      18,
      0xff4400,
      3.0,
      1.5
    );
    this.group.add(this.thrustArrow);
  }

  update(currentFrame: StateVector, isThrustActive = false): void {
    const pos = engineToThreePos(currentFrame.position);

    // Gravity force direction: points toward central body (origin)
    const toOrigin = new THREE.Vector3(0, 0, 0).sub(pos);
    if (toOrigin.lengthSq() > 0.001) {
      const gDir = toOrigin.clone().normalize();
      this.gravityArrow.position.copy(pos);
      this.gravityArrow.setDirection(gDir);
      this.gravityArrow.setLength(15, 2.5, 1.2);
    }

    // Thrust force direction: aligned with velocity when burning
    const burning = isThrustActive || currentFrame.thrust_active;
    this.thrustArrow.visible = burning;
    if (burning) {
      const vel = new THREE.Vector3(
        currentFrame.velocity[0],
        currentFrame.velocity[2],
        currentFrame.velocity[1]
      );
      if (vel.lengthSq() > 0.001) {
        this.thrustArrow.position.copy(pos);
        this.thrustArrow.setDirection(vel.normalize());
        this.thrustArrow.setLength(20, 3.0, 1.5);
      }
    }
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  dispose(): void {
    // Arrow helpers dispose internally
  }
}
