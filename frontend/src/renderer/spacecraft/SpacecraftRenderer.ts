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
 * units, and an inner group is scaled each frame by VisualScale so the craft
 * reaches a minimum apparent size on screen.
 *
 * In practice a real vehicle is a few metres across, which is ~1e-7 scene
 * units, so the minimum-size rule always binds and the drawn size is
 * effectively symbolic. The physical radius is still threaded through rather
 * than assumed away, so the two remain genuinely separate quantities — and if
 * the camera ever gets within a few hundred metres the multiplier collapses
 * to 1 and the vehicle renders true-size.
 */

import * as THREE from 'three';
import { SpacecraftGeometryBuilder } from './SpacecraftGeometry';
import { ThrustPlumeRenderer } from './ThrustPlumeRenderer';
import { StateVector } from '../../types/mission';
import {
  SCENE_SCALE,
  engineToThreePos,
  engineToThreePosInto,
  engineToThreeVelInto,
} from '../CoordinateSystem';
import {
  ViewContext,
  MIN_APPARENT_RADIUS_PX,
  visualRadiusScene,
} from '../VisualScale';

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
  private frameOrigin = new THREE.Vector3();

  /**
   * True characteristic radius of the vehicle in scene units, when known.
   * Null means no catalogued size, so the drawn size is purely symbolic.
   */
  private physicalRadiusScene: number | null = null;

  constructor(type = 'falcon9', color = '#c9a05a') {
    this.group = new THREE.Group();
    this.group.name = `SpacecraftGroup_${type}`;

    this.scaleGroup = new THREE.Group();
    this.scaleGroup.name = 'SpacecraftScale';
    this.group.add(this.scaleGroup);

    // ── 1. Vehicle 3D mesh (unitless vehicle units) ───────────────
    this.vehicleMesh = SpacecraftGeometryBuilder.buildVehicleGroup({
      type,
      color,
      showSolarPanels: true,
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
      color: new THREE.Color(color),
      transparent: true,
      opacity: 0.85,
      depthWrite: false,
    });
    this.trajectoryLine = new THREE.Line(this.lineGeometry, lineMat);
    this.trajectoryLine.name = `TrajectoryLine_${type}`;
  }

  /**
   * Supply the vehicle's true characteristic radius in metres, from the
   * mission's rocket preset. Pass null when no size is catalogued.
   */
  setPhysicalRadiusMeters(radiusM: number | null): void {
    this.physicalRadiusScene = radiusM !== null && radiusM > 0 ? radiusM * SCENE_SCALE : null;
  }

  /**
   * Set the reference frame the state history is expressed in.
   *
   * ORBIT-X reports state relative to `metadata.central_body`. Passing that
   * body's scene position here places the whole mission — vehicle and trail
   * together — in the right place, with a single translation rather than an
   * arithmetic offset applied separately in two places.
   *
   * APPROXIMATION: the trail is translated by the central body's position at
   * the CURRENT time, not at each sample's own time. That is exact for a
   * heliocentric mission and for any body-centred mission short enough that
   * the central body barely moves. For a long body-centred trajectory the
   * trail will lag; fixing that properly needs the per-sample body ephemeris
   * that ORBIT-X can supply in `result.bodies`.
   */
  setFrameOrigin(originScene: THREE.Vector3): void {
    this.frameOrigin.copy(originScene);
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
    engineToThreePosInto(frame.position, this.group.position);
    this.group.position.add(this.frameOrigin);

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
   * Apply the camera-relative legibility scale. Call once per frame, after
   * updateFrame. Touches scale only — never position.
   */
  updateVisualScale(ctx: ViewContext): void {
    const physical = this.physicalRadiusScene ?? 0;
    const drawnRadius = visualRadiusScene(
      physical,
      this.group.position,
      MIN_APPARENT_RADIUS_PX.SPACECRAFT,
      ctx
    );
    this.scaleGroup.scale.setScalar(drawnRadius / this.unitRadius);
  }

  /**
   * Rebuild the trail from the full state history.
   *
   * Points are in the engine's own frame; the line object itself is
   * translated by `setFrameOrigin`, so this is O(n) once per mission rather
   * than per frame.
   */
  updateTrajectoryHistory(history: StateVector[]): void {
    this.linePoints = history.map(s => engineToThreePos(s.position));
    this.lineGeometry.setFromPoints(this.linePoints);
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible;
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
