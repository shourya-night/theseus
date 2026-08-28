/**
 * Explicit world frame for one ORBIT-X mission result.
 *
 * It converts the engine's SI ecliptic states through THESEUS's canonical
 * axis map, then applies only the declared central-body origin. It never
 * reads or writes a planet transform.
 */

import * as THREE from 'three';
import { engineToThreePosInto } from '../CoordinateSystem';

export class MissionRenderFrame {
  private readonly origin = new THREE.Vector3();

  setOrigin(origin: THREE.Vector3): void {
    this.origin.copy(origin);
  }

  positionInto(positionMeters: [number, number, number], out: THREE.Vector3): THREE.Vector3 {
    return engineToThreePosInto(positionMeters, out).add(this.origin);
  }
}
