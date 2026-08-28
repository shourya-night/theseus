/**
 * THESEUS Level-of-Detail (LOD) Manager
 * =====================================
 * Dynamically evaluates object visibility, geometric detail, shader complexity,
 * and label density based on camera position and distance.
 *
 * Prevents performance collapse when rendering complex celestial systems.
 */

import * as THREE from 'three';
import { AstronomicalObject } from '../data/astronomicalObjects';
import { kmToScene, sceneToAu } from './CoordinateSystem';
import { apparentRadiusPixels } from './VisualScale';

export type LODLevel = 'ULTRA' | 'HIGH' | 'MEDIUM' | 'LOW' | 'BILLBOARD' | 'CULLED';

export interface LODThresholds {
  ultraRadiusRatio: number;   // Distance < R * ratio => ULTRA
  highRadiusRatio: number;    // Distance < R * ratio => HIGH
  mediumRadiusRatio: number;  // Distance < R * ratio => MEDIUM
  lowRadiusRatio: number;     // Distance < R * ratio => LOW
  billboardDistanceAU: number; // Distance < threshold => BILLBOARD point/icon
}

export interface LODState {
  level: LODLevel;
  distance: number;
  distanceAU: number;
  apparentSizePixels: number;
  showLabels: boolean;
  show3DGeometry: boolean;
  showAtmosphere: boolean;
  showRings: boolean;
  geometrySubdivisions: number;
}

export const DEFAULT_LOD_THRESHOLDS: LODThresholds = {
  ultraRadiusRatio: 5,     // Close inspection (< 5 radii)
  highRadiusRatio: 50,    // Local system view (< 50 radii)
  mediumRadiusRatio: 500,  // Regional view (< 500 radii)
  lowRadiusRatio: 5000,   // System view (< 5000 radii)
  billboardDistanceAU: 80, // Beyond 80 AU -> billboard/point
};

export class LODManager {
  private camera: THREE.PerspectiveCamera;
  private thresholds: LODThresholds;

  constructor(camera: THREE.PerspectiveCamera, thresholds = DEFAULT_LOD_THRESHOLDS) {
    this.camera = camera;
    this.thresholds = thresholds;
  }

  /**
   * Evaluate the LOD state for a given astronomical object.
   */
  evaluateObject(
    obj: AstronomicalObject,
    worldPosition: THREE.Vector3,
    viewportHeightPixels: number
  ): LODState {
    const distance = this.camera.position.distanceTo(worldPosition);
    const distanceAU = sceneToAu(distance);

    const radiusScene = kmToScene(obj.radius_km);
    const radiusRatio = distance / Math.max(1e-4, radiusScene);

    // Apparent screen size, through the shared VisualScale formula so LOD and
    // the minimum-size rule can never disagree about how big something looks.
    const apparentSizePixels = apparentRadiusPixels(radiusScene, distance, {
      camera: this.camera,
      viewportHeightPx: viewportHeightPixels,
    });

    let level: LODLevel = 'CULLED';
    let geometrySubdivisions = 16;
    let showLabels = true;
    let show3DGeometry = true;
    let showAtmosphere = false;
    let showRings = false;

    if (apparentSizePixels < 0.5 && distanceAU > this.thresholds.billboardDistanceAU) {
      level = 'BILLBOARD';
      show3DGeometry = false;
      geometrySubdivisions = 8;
    } else if (radiusRatio < this.thresholds.ultraRadiusRatio) {
      level = 'ULTRA';
      geometrySubdivisions = 128;
      showAtmosphere = !!obj.atmosphere;
      showRings = !!obj.rings;
    } else if (radiusRatio < this.thresholds.highRadiusRatio) {
      level = 'HIGH';
      geometrySubdivisions = 64;
      showAtmosphere = !!obj.atmosphere;
      showRings = !!obj.rings;
    } else if (radiusRatio < this.thresholds.mediumRadiusRatio) {
      level = 'MEDIUM';
      geometrySubdivisions = 32;
      showAtmosphere = !!obj.atmosphere && apparentSizePixels > 10;
      showRings = !!obj.rings;
    } else if (radiusRatio < this.thresholds.lowRadiusRatio || apparentSizePixels > 1.5) {
      level = 'LOW';
      geometrySubdivisions = 16;
    } else {
      level = 'BILLBOARD';
      show3DGeometry = false;
    }

    // Label visibility threshold: only show labels when apparent size or prominence warrants it
    showLabels = apparentSizePixels > 3 || obj.type === 'PLANET' || obj.type === 'STAR';

    return {
      level,
      distance,
      distanceAU,
      apparentSizePixels,
      showLabels,
      show3DGeometry,
      showAtmosphere,
      showRings,
      geometrySubdivisions,
    };
  }

  /**
   * Check if a world point is within the camera frustum.
   */
  isInFrustum(point: THREE.Vector3, boundingRadius = 0): boolean {
    const frustum = new THREE.Frustum();
    const matrix = new THREE.Matrix4().multiplyMatrices(
      this.camera.projectionMatrix,
      this.camera.matrixWorldInverse
    );
    frustum.setFromProjectionMatrix(matrix);
    return frustum.intersectsSphere(new THREE.Sphere(point, boundingRadius));
  }
}
