/**
 * THESEUS High-Fidelity Planet Renderer
 * =====================================
 * Complete 3D representation of a planet, dwarf planet or moon.
 * Manages geometric LOD, procedural surface shaders, atmospheric scattering,
 * ring systems, axial tilt, rotation, and the body's own orbital path line.
 *
 * All orbit geometry and unit conversion is delegated to CoordinateSystem.
 * This file contains no Kepler math and no scale literals.
 */

import * as THREE from 'three';
import { AstronomicalObject } from '../../data/astronomicalObjects';
import { PlanetShaderFactory } from './PlanetShaders';
import { AtmosphereRenderer } from './AtmosphereRenderer';
import { RingRenderer } from './RingRenderer';
import { LODLevel } from '../LODManager';

import {
  PreparedOrbit,
  prepareOrbit,
  orbitPathPoints,
  orbitPositionInto,
  kmToScene,
} from '../CoordinateSystem';
import { MIN_BODY_VISUAL_RADIUS_SCENE } from '../VisualScale';

export class PlanetRenderer {
  readonly group: THREE.Group;
  readonly mesh: THREE.Mesh;
  readonly orbitLine: THREE.Line;
  readonly objectData: AstronomicalObject;

  /** Prepared orbital elements, or null for a body with no orbit (the Sun). */
  readonly preparedOrbit: PreparedOrbit | null;

  private material: THREE.ShaderMaterial;
  private atmosphereRenderer: AtmosphereRenderer | null = null;
  private ringRenderer: RingRenderer | null = null;

  private currentLOD: LODLevel = 'HIGH';

  /** True physical radius in scene units. */
  readonly physicalRadiusScene: number;
  /** Radius the mesh is actually built at (>= physical, for tiny bodies). */
  readonly visualRadiusScene: number;

  private _sunDir = new THREE.Vector3(0, 0, 1);

  constructor(objectData: AstronomicalObject) {
    this.objectData = objectData;

    this.physicalRadiusScene = kmToScene(objectData.radius_km);
    this.visualRadiusScene = Math.max(MIN_BODY_VISUAL_RADIUS_SCENE, this.physicalRadiusScene);

    this.group = new THREE.Group();
    this.group.name = `PlanetGroup_${objectData.id}`;

    // Axial tilt is a property of the body, applied to the whole group so
    // rings and atmosphere inherit it exactly once.
    if (objectData.axial_tilt_deg) {
      this.group.rotation.z = (objectData.axial_tilt_deg * Math.PI) / 180;
    }

    // ─── Surface Mesh ─────────────────────────────────────────────
    const geometry = new THREE.SphereGeometry(this.visualRadiusScene, 96, 96);
    this.material = PlanetShaderFactory.createMaterial(objectData);
    this.mesh = new THREE.Mesh(geometry, this.material);
    this.mesh.name = `PlanetMesh_${objectData.id}`;
    this.group.add(this.mesh);

    // ─── Orbital Path Line ────────────────────────────────────────
    // Built from the same prepared elements that drive the body's position,
    // so the body is guaranteed to lie on its own path.
    this.preparedOrbit = objectData.orbit ? prepareOrbit(objectData.orbit) : null;
    const orbitPoints = this.preparedOrbit ? orbitPathPoints(this.preparedOrbit, 512) : [];

    const orbitGeo = new THREE.BufferGeometry().setFromPoints(orbitPoints);
    const orbitMat = new THREE.LineBasicMaterial({
      color: new THREE.Color(...objectData.color),
      transparent: true,
      opacity: 0.4,
      depthTest: true,
      depthWrite: false,
    });
    this.orbitLine = new THREE.Line(orbitGeo, orbitMat);
    this.orbitLine.name = `PlanetOrbit_${objectData.id}`;

    // ─── Atmosphere ───────────────────────────────────────────────
    if (objectData.atmosphere) {
      this.atmosphereRenderer = new AtmosphereRenderer(this.visualRadiusScene, objectData.atmosphere);
      this.group.add(this.atmosphereRenderer.mesh);
    }

    // ─── Rings ────────────────────────────────────────────────────
    if (objectData.rings) {
      this.ringRenderer = new RingRenderer(this.visualRadiusScene, objectData.rings);
      this.group.add(this.ringRenderer.mesh);
    }
  }

  /**
   * Position this body on its own orbit at the given simulation time.
   * `focusWorldPos` is the world position of the orbit's focus — the scene
   * origin for a heliocentric body, the parent's position for a satellite.
   */
  positionAtTime(simTimeSec: number, focusWorldPos?: THREE.Vector3): THREE.Vector3 {
    if (!this.preparedOrbit) {
      if (focusWorldPos) this.group.position.copy(focusWorldPos);
      return this.group.position;
    }
    orbitPositionInto(this.preparedOrbit, simTimeSec, this.group.position);
    if (focusWorldPos) this.group.position.add(focusWorldPos);
    return this.group.position;
  }

  /** Set physical position in scene units directly. */
  setPosition(pos: THREE.Vector3): void {
    this.group.position.copy(pos);
  }

  /**
   * Update frame animation.
   *
   * `sunWorldPos` is the world position of the illuminating star. The sun
   * direction is derived per body from actual geometry — never a constant.
   */
  update(timeSeconds: number, sunWorldPos: THREE.Vector3): void {
    // Rotation on the polar axis.
    const period = this.objectData.rotation_period_s;
    if (period) {
      this.mesh.rotation.y = (2 * Math.PI * timeSeconds) / period;
    }

    // Direction from this body toward the star.
    this._sunDir.subVectors(sunWorldPos, this.group.position);
    if (this._sunDir.lengthSq() < 1e-12) this._sunDir.set(0, 0, 1);
    else this._sunDir.normalize();

    PlanetShaderFactory.updateUniforms(this.material, this._sunDir, timeSeconds);

    if (this.atmosphereRenderer) {
      this.atmosphereRenderer.updateSunDirection(this._sunDir);
    }
    if (this.ringRenderer) {
      this.ringRenderer.update(this._sunDir, this.group.position);
    }
  }

  /** Direction from this body toward the star, in world space. */
  get sunDirection(): THREE.Vector3 {
    return this._sunDir;
  }

  /** Adjust level of detail (component visibility). */
  setLOD(lod: LODLevel): void {
    if (this.currentLOD === lod) return;
    this.currentLOD = lod;

    this.group.visible = lod !== 'CULLED';

    if (this.atmosphereRenderer) {
      this.atmosphereRenderer.mesh.visible = lod === 'ULTRA' || lod === 'HIGH' || lod === 'MEDIUM';
    }
    if (this.ringRenderer) {
      this.ringRenderer.mesh.visible = lod !== 'CULLED' && lod !== 'BILLBOARD';
    }
  }

  setOrbitVisible(visible: boolean): void {
    this.orbitLine.visible = visible;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.material.dispose();
    this.orbitLine.geometry.dispose();
    (this.orbitLine.material as THREE.Material).dispose();
    if (this.atmosphereRenderer) this.atmosphereRenderer.dispose();
    if (this.ringRenderer) this.ringRenderer.dispose();
  }
}
