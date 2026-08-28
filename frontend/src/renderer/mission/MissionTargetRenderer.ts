/**
 * ORBIT-X mission-frame destination marker.
 *
 * Placed at the exact physical center of the destination body at the arrival epoch
 * (t_terminal), using the same conversion as the trajectory and RocketGroup.
 */

import * as THREE from 'three';
import { MissionRenderFrame } from './MissionRenderFrame';
import { ViewContext, visualRadiusScene } from '../VisualScale';

export class MissionTargetRenderer {
  readonly group: THREE.Group;
  private readonly missionFrame = new MissionRenderFrame();
  private readonly marker: THREE.LineSegments;
  private readonly baseRadius = 0.08;

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'MissionTargetGroup';

    // Three orthogonal crosshair axes: a mission target, never a planet mesh
    // or a spacecraft placeholder.
    const r = this.baseRadius;
    const geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-r, 0, 0), new THREE.Vector3(r, 0, 0),
      new THREE.Vector3(0, -r, 0), new THREE.Vector3(0, r, 0),
      new THREE.Vector3(0, 0, -r), new THREE.Vector3(0, 0, r),
    ]);
    this.marker = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({
      color: 0xff8a2a,
      transparent: true,
      opacity: 0.95,
      depthTest: false,
    }));
    this.marker.name = 'MissionTargetReticle';
    this.marker.frustumCulled = false;
    this.group.add(this.marker);
  }

  setFrameOrigin(origin: THREE.Vector3): void {
    this.missionFrame.setOrigin(origin);
  }

  update(positionMeters: [number, number, number]): void {
    this.missionFrame.positionInto(positionMeters, this.group.position);
  }

  updateVisualScale(ctx: ViewContext): void {
    if (!ctx?.camera) return;
    const targetR = visualRadiusScene(this.baseRadius, this.group.position, 6, ctx);
    const s = targetR / this.baseRadius;
    this.marker.scale.setScalar(Math.max(1, s));
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  dispose(): void {
    this.marker.geometry.dispose();
    (this.marker.material as THREE.Material).dispose();
  }
}
