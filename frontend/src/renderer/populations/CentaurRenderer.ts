/**
 * THESEUS Centaur Renderer
 * ========================
 * GPU-instanced visualization of Centaurs — small icy bodies orbiting between
 * Jupiter and Neptune (roughly 5–30 AU).
 *
 * Centaurs are dynamically unstable and have higher eccentricities and
 * inclinations than main-belt asteroids. Their surfaces are icy, giving them
 * a cooler color palette.
 *
 * Same architecture as AsteroidBeltRenderer: InstancedMesh, deterministic
 * PRNG, per-frame Kepler propagation through CoordinateSystem.
 */

import * as THREE from 'three';
import { CENTAUR_PARAMS } from '../../data/smallBodies';
import {
  PreparedOrbit,
  prepareOrbit,
  orbitPositionInto,
  auToScene,
  MU_SUN_KM,
  KM_PER_SCENE_UNIT,
} from '../CoordinateSystem';

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function periodDaysForSceneA(a_scene: number): number {
  const a_km = a_scene * KM_PER_SCENE_UNIT;
  const seconds = 2 * Math.PI * Math.sqrt((a_km * a_km * a_km) / MU_SUN_KM);
  return seconds / 86400;
}

export class CentaurRenderer {
  readonly instancedMesh: THREE.InstancedMesh;
  private count: number;

  private orbits: PreparedOrbit[] = [];
  private scales: Float32Array;

  private dummy = new THREE.Object3D();
  private scratch = new THREE.Vector3();

  constructor(count = CENTAUR_PARAMS.count, seed = 0xC3C3) {
    this.count = count;
    this.scales = new Float32Array(count);

    const geometry = new THREE.IcosahedronGeometry(0.09, 1);
    const posAttr = geometry.attributes.position;
    for (let i = 0; i < posAttr.count; i++) {
      const vx = posAttr.getX(i), vy = posAttr.getY(i), vz = posAttr.getZ(i);
      const n = 1 + Math.sin(vx * 11 + vy * 15) * 0.12;
      posAttr.setXYZ(i, vx * n, vy * n, vz * n);
    }
    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(...CENTAUR_PARAMS.color),
      roughness: 0.88,
      metalness: 0.08,
    });

    this.instancedMesh = new THREE.InstancedMesh(geometry, material, this.count);
    this.instancedMesh.name = 'CentaurMesh';
    this.instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.instancedMesh.frustumCulled = false;

    this.initPopulation(seed);
  }

  private initPopulation(seed: number): void {
    const params = CENTAUR_PARAMS;
    const rand = mulberry32(seed);

    const innerScene = auToScene(params.innerRadius_AU);
    const outerScene = auToScene(params.outerRadius_AU);
    const color = new THREE.Color();

    for (let i = 0; i < this.count; i++) {
      // Centaurs: semi-major axis skewed toward smaller values (more
      // concentrated near Jupiter than Neptune).
      const a_scene = innerScene + Math.pow(rand(), 1.3) * (outerScene - innerScene);

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
        Math.min(1, Math.max(0, params.color[2] + colVar * 1.2)),
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
      this.dummy.rotation.set(this.scratch.x * 0.04, this.scratch.y * 0.04, this.scratch.z * 0.04);
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
