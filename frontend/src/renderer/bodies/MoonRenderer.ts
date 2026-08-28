/**
 * THESEUS Moon System Renderer
 * ============================
 * Parent-child celestial hierarchy for natural satellites.
 *
 * A moon is a PlanetRenderer whose orbit focus is its parent body rather than
 * the scene origin. It therefore uses exactly the same Kepler propagation and
 * the same orbit-path generator as a planet — there is no separate, simplified
 * satellite orbit model in this file.
 */

import * as THREE from 'three';
import { AstronomicalObject } from '../../data/astronomicalObjects';
import { PlanetRenderer } from './PlanetRenderer';
import { LODLevel } from '../LODManager';
import { orbitPathPoints } from '../CoordinateSystem';

export class MoonRenderer {
  readonly planetRenderer: PlanetRenderer;
  readonly orbitLine: THREE.Line;
  readonly group: THREE.Group;

  constructor(objectData: AstronomicalObject) {
    this.planetRenderer = new PlanetRenderer(objectData);

    this.group = new THREE.Group();
    this.group.name = `MoonGroup_${objectData.id}`;
    this.group.add(this.planetRenderer.group);

    // Orbit path, expressed relative to the parent. The whole line is
    // translated to the parent's world position each frame, which keeps the
    // moon on its drawn path by construction.
    const prepared = this.planetRenderer.preparedOrbit;
    const points = prepared ? orbitPathPoints(prepared, 256) : [];

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
      color: new THREE.Color(...(objectData.color ?? [0.5, 0.5, 0.5])),
      transparent: true,
      opacity: 0.28,
      depthWrite: false,
    });

    this.orbitLine = new THREE.Line(geometry, material);
    this.orbitLine.name = `MoonOrbit_${objectData.id}`;
  }

  /**
   * Advance the moon along its orbit about `parentWorldPos`.
   * `sunWorldPos` drives illumination and is the star's world position.
   */
  updateOrbitPosition(
    simTimeSec: number,
    parentWorldPos: THREE.Vector3,
    sunWorldPos: THREE.Vector3
  ): void {
    this.planetRenderer.positionAtTime(simTimeSec, parentWorldPos);
    this.planetRenderer.update(simTimeSec, sunWorldPos);
    this.orbitLine.position.copy(parentWorldPos);
  }

  setLOD(lod: LODLevel): void {
    this.planetRenderer.setLOD(lod);
    this.orbitLine.visible = lod === 'ULTRA' || lod === 'HIGH';
  }

  setOrbitVisible(visible: boolean): void {
    this.orbitLine.visible = visible;
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible;
    this.orbitLine.visible = visible;
  }

  dispose(): void {
    this.planetRenderer.dispose();
    this.orbitLine.geometry.dispose();
    (this.orbitLine.material as THREE.Material).dispose();
  }
}
