/**
 * THESEUS Kuiper Belt Renderer
 * ============================
 * GPU-instanced visualization of the classical Kuiper Belt (30–55 AU).
 *
 * Follows the same architecture as AsteroidBeltRenderer: every member has a
 * full Keplerian element set prepared once at construction. Per-frame, each
 * orbit is propagated through CoordinateSystem's canonical solver, so
 * positions, axis orientation, and scale are all consistent with the rest of
 * the scene.
 *
 * The belt is a STATISTICAL POPULATION — its members are drawn from
 * KUIPER_BELT_PARAMS and are deliberately unlabelled and unselectable.
 */

import * as THREE from 'three';
import { KUIPER_BELT_PARAMS } from '../../data/smallBodies';
import {
  PreparedOrbit,
  prepareOrbit,
  orbitPositionInto,
  auToScene,
  MU_SUN_KM,
  KM_PER_SCENE_UNIT,
} from '../CoordinateSystem';

/** Deterministic pseudo-random source (Mulberry32). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Keplerian period in days for a heliocentric semi-major axis in scene units. */
function periodDaysForSceneA(a_scene: number): number {
  const a_km = a_scene * KM_PER_SCENE_UNIT;
  const seconds = 2 * Math.PI * Math.sqrt((a_km * a_km * a_km) / MU_SUN_KM);
  return seconds / 86400;
}

export class KuiperBeltRenderer {
  readonly instancedMesh: THREE.InstancedMesh;
  private count: number;

  private orbits: PreparedOrbit[] = [];
  private scales: Float32Array;

  private dummy = new THREE.Object3D();
  private scratch = new THREE.Vector3();

  constructor(count = KUIPER_BELT_PARAMS.count, seed = 0xBEEF) {
    this.count = count;
    this.scales = new Float32Array(count);

    // Shared irregular low-poly body.
    const geometry = new THREE.IcosahedronGeometry(0.08, 1);
    const posAttr = geometry.attributes.position;
    for (let i = 0; i < posAttr.count; i++) {
      const vx = posAttr.getX(i), vy = posAttr.getY(i), vz = posAttr.getZ(i);
      const n = 1 + Math.sin(vx * 12 + vy * 18) * 0.12;
      posAttr.setXYZ(i, vx * n, vy * n, vz * n);
    }
    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(...KUIPER_BELT_PARAMS.color),
      roughness: 0.92,
      metalness: 0.05,
    });

    this.instancedMesh = new THREE.InstancedMesh(geometry, material, this.count);
    this.instancedMesh.name = 'KuiperBeltMesh';
    this.instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.instancedMesh.frustumCulled = false;

    this.initPopulation(seed);
  }

  private initPopulation(seed: number): void {
    const params = KUIPER_BELT_PARAMS;
    const rand = mulberry32(seed);

    const innerScene = auToScene(params.innerRadius_AU);
    const outerScene = auToScene(params.outerRadius_AU);
    const color = new THREE.Color();

    for (let i = 0; i < this.count; i++) {
      const a_scene = innerScene + Math.sqrt(rand()) * (outerScene - innerScene);

      const e = params.eccentricityRange[0]
        + rand() * (params.eccentricityRange[1] - params.eccentricityRange[0]);
      const inc_deg = params.inclinationRange_deg[0]
        + rand() * (params.inclinationRange_deg[1] - params.inclinationRange_deg[0]);

      const period_days = periodDaysForSceneA(a_scene);

      this.orbits.push(prepareOrbit({
        a_km: a_scene * KM_PER_SCENE_UNIT,
        e,
        inc_deg,
        raan_deg: rand() * 360,
        w_deg: rand() * 360,
        m0_deg: rand() * 360,
        period_days,
      }));

      this.scales[i] = params.sizeRange[0] + rand() * (params.sizeRange[1] - params.sizeRange[0]);

      const colVar = (rand() - 0.5) * params.colorVariation;
      color.setRGB(
        Math.min(1, Math.max(0, params.color[0] + colVar)),
        Math.min(1, Math.max(0, params.color[1] + colVar * 0.8)),
        Math.min(1, Math.max(0, params.color[2] + colVar * 1.4)),
      );
      this.instancedMesh.setColorAt(i, color);
    }

    if (this.instancedMesh.instanceColor) {
      this.instancedMesh.instanceColor.needsUpdate = true;
    }

    this.update(0);
  }

  update(simTimeSec: number): void {
    if (!this.instancedMesh.visible) return;

    for (let i = 0; i < this.count; i++) {
      orbitPositionInto(this.orbits[i], simTimeSec, this.scratch);
      this.dummy.position.copy(this.scratch);
      this.dummy.scale.setScalar(this.scales[i]);
      this.dummy.rotation.set(this.scratch.x * 0.05, this.scratch.y * 0.05, this.scratch.z * 0.05);
      this.dummy.updateMatrix();
      this.instancedMesh.setMatrixAt(i, this.dummy.matrix);
    }

    this.instancedMesh.instanceMatrix.needsUpdate = true;
  }

  setVisible(visible: boolean): void {
    this.instancedMesh.visible = visible;
  }

  dispose(): void {
    this.instancedMesh.geometry.dispose();
    (this.instancedMesh.material as THREE.Material).dispose();
  }
}
