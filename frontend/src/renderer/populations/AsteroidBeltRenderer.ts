/**
 * THESEUS Asteroid Belt Renderer
 * ==============================
 * GPU-instanced renderer for the Main Asteroid Belt between Mars and Jupiter.
 * A single InstancedMesh carries the whole population in one draw call.
 *
 * Every member is propagated by the same Kepler solver the planets use — the
 * elements are prepared once at construction, so the per-frame cost is the
 * anomaly solve and a basis multiply, with no allocation.
 *
 * The belt is a STATISTICAL POPULATION, not a catalog. Its members are drawn
 * from the distribution in ASTEROID_BELT_PARAMS and are deliberately
 * unlabelled and unselectable: they represent the belt, they do not claim to
 * be particular asteroids. Named, catalogued asteroids are rendered by
 * NEORenderer from real element sets.
 */

import * as THREE from 'three';
import { ASTEROID_BELT_PARAMS } from '../../data/smallBodies';
import {
  PreparedOrbit,
  prepareOrbit,
  orbitPositionInto,
  auToScene,
  MU_SUN_KM,
  KM_PER_SCENE_UNIT,
} from '../CoordinateSystem';

/**
 * Deterministic pseudo-random source.
 *
 * The population must be identical on every reload — a belt that reshuffles
 * itself when the page refreshes is not a reproducible visualization.
 */
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

export class AsteroidBeltRenderer {
  readonly instancedMesh: THREE.InstancedMesh;
  private count: number;

  private orbits: PreparedOrbit[] = [];
  private scales: Float32Array;

  private dummy = new THREE.Object3D();
  private scratch = new THREE.Vector3();

  constructor(count = ASTEROID_BELT_PARAMS.count, seed = 0x7e51) {
    this.count = count;
    this.scales = new Float32Array(count);

    // Irregular low-poly body, shared across every instance.
    const geometry = new THREE.IcosahedronGeometry(0.12, 1);
    const posAttr = geometry.attributes.position;
    for (let i = 0; i < posAttr.count; i++) {
      const vx = posAttr.getX(i), vy = posAttr.getY(i), vz = posAttr.getZ(i);
      const n = 1 + Math.sin(vx * 15 + vy * 20) * 0.15;
      posAttr.setXYZ(i, vx * n, vy * n, vz * n);
    }
    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(...ASTEROID_BELT_PARAMS.color),
      roughness: 0.9,
      metalness: 0.1,
    });

    this.instancedMesh = new THREE.InstancedMesh(geometry, material, this.count);
    this.instancedMesh.name = 'AsteroidBeltMesh';
    this.instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    // Members are spread over the whole belt; a per-instance frustum test
    // against the shared bounding sphere would cull the entire mesh.
    this.instancedMesh.frustumCulled = false;

    this.initPopulation(seed);
  }

  private initPopulation(seed: number): void {
    const params = ASTEROID_BELT_PARAMS;
    const rand = mulberry32(seed);

    const innerScene = auToScene(params.innerRadius_AU);
    const outerScene = auToScene(params.outerRadius_AU);
    const color = new THREE.Color();

    for (let i = 0; i < this.count; i++) {
      const a_scene = innerScene + Math.pow(rand(), 0.8) * (outerScene - innerScene);

      const e = params.eccentricityRange[0]
        + rand() * (params.eccentricityRange[1] - params.eccentricityRange[0]);
      const inc_deg = params.inclinationRange_deg[0]
        + rand() * (params.inclinationRange_deg[1] - params.inclinationRange_deg[0]);

      // Period from Kepler's third law rather than a fitted ratio.
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
        Math.min(1, Math.max(0, params.color[1] + colVar)),
        Math.min(1, Math.max(0, params.color[2] + colVar))
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
      this.dummy.rotation.set(this.scratch.x * 0.1, this.scratch.y * 0.1, this.scratch.z * 0.1);
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
