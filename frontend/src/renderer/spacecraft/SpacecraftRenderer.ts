/**
 * THESEUS High-Fidelity Spacecraft Renderer
 * =========================================
 * Renders a mission vehicle with procedural 3D geometry, engine plume,
 * velocity-vector alignment and a trajectory trail.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * POSITION IS PHYSICAL. SIZE IS NOT.
 * ─────────────────────────────────────────────────────────────────────────
 * `group.position` comes straight from the ORBIT-X state vector through the
 * canonical scale factor, with no adjustment of any kind. What the vehicle
 * looks like is decided separately: the geometry is built in unitless vehicle
 * units. Its inner group uses a camera-aware visual scale so it remains
 * visible in solar-system view and drops to true detailed proportions in
 * close-up spectator mode.
 */

import * as THREE from 'three';
import { SpacecraftGeometryBuilder } from './SpacecraftGeometry';
import { ThrustPlumeRenderer } from './ThrustPlumeRenderer';
import { MissionRenderFrame } from '../mission/MissionRenderFrame';
import { StateVector } from '../../types/mission';
import {
  engineToThreePos,
  engineToThreeVelInto,
} from '../CoordinateSystem';
import { ViewContext, visualRadiusScene, MIN_APPARENT_RADIUS_PX } from '../VisualScale';

/**
 * Deliberately symbolic base model radius in scene units.
 * Scaled per-frame by VisualScale to guarantee minimum apparent pixels on screen
 * while remaining in true detailed proportions during close-up spectator inspection.
 */
const ROCKET_MODEL_RADIUS_SCENE = 0.02;

export class SpacecraftRenderer {
  readonly group: THREE.Group;
  /** Scaled per frame. Holds the vehicle mesh and its plume. */
  readonly scaleGroup: THREE.Group;
  readonly vehicleMesh: THREE.Group;
  readonly plumeRenderer: ThrustPlumeRenderer;
  readonly trajectoryLine: THREE.Line;

  private currentQuat = new THREE.Quaternion();
  private targetQuat = new THREE.Quaternion();
  private _vel = new THREE.Vector3();
  private _tempWorldPos = new THREE.Vector3();
  private static readonly NOSE_AXIS = new THREE.Vector3(0, 0, 1);
  private lineGeometry: THREE.BufferGeometry;
  private linePoints: THREE.Vector3[] = [];

  /** Bounding radius of the builder's output, in vehicle units. */
  private unitRadius: number;

  /**
   * Scene position of the body the ORBIT-X state history is referenced to.
   * Zero for a heliocentric mission. Applied as a translation to both the
   * vehicle and its trail, so the two can never disagree.
   */
  private readonly missionFrame = new MissionRenderFrame();

  constructor(type = 'falcon9', color = '#c9a05a') {
    this.group = new THREE.Group();
    this.group.name = 'RocketGroup';

    this.scaleGroup = new THREE.Group();
    this.scaleGroup.name = 'RocketVisualScale';
    this.group.add(this.scaleGroup);

    // ── 1. Vehicle 3D mesh (unitless vehicle units) ───────────────
    this.vehicleMesh = SpacecraftGeometryBuilder.buildVehicleGroup({
      type,
      color,
      // A transfer vehicle is rendered as a launch vehicle, not as a generic
      // satellite with wings. Keep the silhouette legible at pixel scale.
      showSolarPanels: false,
      showDishAntenna: type.includes('voyager') || type.includes('probe'),
    });
    this.scaleGroup.add(this.vehicleMesh);

    // Measure the builder's output once so the scale factor below is derived
    // rather than guessed.
    const box = new THREE.Box3().setFromObject(this.vehicleMesh);
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    this.unitRadius = sphere.radius > 0 ? sphere.radius : 1;

    // ── 2. Thrust plume (same vehicle-unit space) ─────────────────
    this.plumeRenderer = new ThrustPlumeRenderer();
    this.scaleGroup.add(this.plumeRenderer.group);

    // ── 3. Trajectory line (world scale, never exaggerated) ───────
    this.lineGeometry = new THREE.BufferGeometry();
    const lineMat = new THREE.LineBasicMaterial({
      color: new THREE.Color(color || 0xffaa00),
      transparent: true,
      opacity: 0.95,
      depthTest: true,
      depthWrite: false,
    });
    this.trajectoryLine = new THREE.Line(this.lineGeometry, lineMat);
    this.trajectoryLine.name = `TrajectoryLine_${type}`;
    this.trajectoryLine.frustumCulled = false;
    this.trajectoryLine.renderOrder = 999;
  }

  /**
   * Set the reference frame the state history is expressed in.
   *
   * ORBIT-X reports state relative to `metadata.central_body`. Passing that
   * body's scene position here places the whole mission — vehicle and trail
   * together — in the right place, with a single translation rather than an
   * arithmetic offset applied separately in two places.
   */
  setFrameOrigin(originScene: THREE.Vector3): void {
    this.missionFrame.setOrigin(originScene);
    this.trajectoryLine.position.copy(originScene);
  }

  /**
   * Place the vehicle from an ORBIT-X state vector.
   *
   * `frame.position` is the RAW engine position in metres, relative to the
   * mission's central body. This method applies the canonical scale, the axis
   * remap, and the declared frame origin — nothing else. No nudges.
   */
  updateFrame(frame: StateVector, isThrustActive = false): void {
    this.missionFrame.positionInto(frame.position, this.group.position);

    // Orient along the velocity vector, through the same canonical remap.
    const velVec = engineToThreeVelInto(frame.velocity, this._vel);

    if (velVec.lengthSq() > 0.001) {
      velVec.normalize();
      // Builder emits the vehicle nose along +Z.
      this.targetQuat.setFromUnitVectors(SpacecraftRenderer.NOSE_AXIS, velVec);
      this.currentQuat.slerp(this.targetQuat, 0.25);
      this.vehicleMesh.quaternion.copy(this.currentQuat);
    }

    const burning = isThrustActive || frame.thrust_active;
    this.plumeRenderer.setThrust(burning, burning ? 1.0 : 0.0);
  }

  /**
   * Apply camera-aware spacecraft visual scale using VisualScale policy.
   * Keeps the rocket visible on screen across vast solar system distances
   * while dropping down to true symbolic model radius (0.02) during spectator
   * close-up inspection.
   */
  updateVisualScale(ctx: ViewContext): void {
    if (!ctx?.camera) {
      this.scaleGroup.scale.setScalar(ROCKET_MODEL_RADIUS_SCENE / this.unitRadius);
      return;
    }
    this.group.getWorldPosition(this._tempWorldPos);
    const targetRadius = visualRadiusScene(
      ROCKET_MODEL_RADIUS_SCENE,
      this._tempWorldPos,
      MIN_APPARENT_RADIUS_PX.SPACECRAFT,
      ctx
    );
    this.scaleGroup.scale.setScalar(targetRadius / this.unitRadius);
  }

  /**
   * Rebuild the trail from the full state history.
   *
   * Points are in the engine's own frame; the line object itself is
   * translated by `setFrameOrigin`, so this is O(n) once per mission rather
   * than per frame.
   */
  updateTrajectoryHistory(history: StateVector[]): void {
    if (!history || history.length === 0) return;
    this.linePoints = history.map(s => engineToThreePos(s.position));
    this.lineGeometry.setFromPoints(this.linePoints);
    this.lineGeometry.computeBoundingSphere();
    this.lineGeometry.computeBoundingBox();

    if (import.meta.env.DEV) {
      const posAttr = this.lineGeometry.attributes.position;
      const count = posAttr.count;
      const getV = (i: number) => [posAttr.getX(i), posAttr.getY(i), posAttr.getZ(i)];
      const idx25 = Math.floor(count * 0.25);
      const idx50 = Math.floor(count * 0.50);
      const idx75 = Math.floor(count * 0.75);
      const idxEnd = count - 1;

      console.info('[THESEUS TrajectoryLine Geometry Diagnostic]', {
        name: this.trajectoryLine.name,
        vertexCount: count,
        firstVertex: getV(0),
        v25: getV(idx25),
        v50: getV(idx50),
        v75: getV(idx75),
        finalVertex: getV(idxEnd),
        localPosition: this.trajectoryLine.position.toArray(),
        localRotation: [this.trajectoryLine.rotation.x, this.trajectoryLine.rotation.y, this.trajectoryLine.rotation.z],
        localScale: this.trajectoryLine.scale.toArray(),
        worldPosition: this.trajectoryLine.getWorldPosition(new THREE.Vector3()).toArray(),
        parentName: this.trajectoryLine.parent?.name || 'Scene',
        visible: this.trajectoryLine.visible,
      });
    }
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible;
    this.trajectoryLine.visible = visible;
  }

  setSpacecraftVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  setTrajectoryVisible(visible: boolean): void {
    this.trajectoryLine.visible = visible;
  }

  dispose(): void {
    this.vehicleMesh.traverse(o => {
      if (o instanceof THREE.Mesh) {
        o.geometry.dispose();
        (o.material as THREE.Material).dispose();
      }
    });
    this.plumeRenderer.dispose();
    this.lineGeometry.dispose();
    (this.trajectoryLine.material as THREE.Material).dispose();
  }
}
