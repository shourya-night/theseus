/**
 * THESEUS Jupiter Trojan Renderer
 * ===============================
 * GPU-instanced visualization of Jupiter's Trojan asteroids at the L4 (Greek
 * camp, 60° ahead) and L5 (Trojan camp, 60° behind) Lagrange points.
 *
 * Key difference from belt populations: members have semi-major axes near
 * Jupiter's (~5.2 AU) and their mean longitudes librate around ±60° from
 * Jupiter's current ecliptic longitude. This renderer takes Jupiter's world
 * position each frame and clusters its population accordingly.
 *
 * Two independently toggleable InstancedMesh groups (L4 and L5).
 */

import * as THREE from 'three';
import {
  TROJAN_L4_PARAMS,
  TROJAN_L5_PARAMS,
  PopulationParams,
} from '../../data/smallBodies';
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

function periodDaysForSceneA(a_scene: number): number {
  const a_km = a_scene * KM_PER_SCENE_UNIT;
  const seconds = 2 * Math.PI * Math.sqrt((a_km * a_km * a_km) / MU_SUN_KM);
  return seconds / 86400;
}

/**
 * One swarm of Trojans around a single Lagrange point.
 *
 * Trojan orbits have the same semi-major axis as Jupiter but their mean
 * longitudes are offset by ±60° (plus a libration width). This is modeled by
 * giving each member a mean anomaly at epoch offset by ±60° relative to
 * Jupiter's, with scatter from the libration amplitude.
 */
class TrojanSwarm {
  readonly instancedMesh: THREE.InstancedMesh;
  private count: number;
  private orbits: PreparedOrbit[] = [];
  private scales: Float32Array;
  private dummy = new THREE.Object3D();
  private scratch = new THREE.Vector3();

  constructor(params: PopulationParams, seed: number) {
    this.count = params.count;
    this.scales = new Float32Array(this.count);

    const geometry = new THREE.IcosahedronGeometry(0.1, 1);
    const posAttr = geometry.attributes.position;
    const rand0 = mulberry32(seed - 1);
    for (let i = 0; i < posAttr.count; i++) {
      const vx = posAttr.getX(i), vy = posAttr.getY(i), vz = posAttr.getZ(i);
      const n = 1 + Math.sin(vx * 14 + vy * 19) * 0.15;
      posAttr.setXYZ(i, vx * n, vy * n, vz * n);
    }
    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(...params.color),
      roughness: 0.9,
      metalness: 0.08,
    });

    this.instancedMesh = new THREE.InstancedMesh(geometry, material, this.count);
    this.instancedMesh.name = `TrojanSwarm_${params.resonancePhase?.lagrangePoint ?? 'X'}`;
    this.instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.instancedMesh.frustumCulled = false;

    this.initPopulation(params, seed);
  }

  private initPopulation(params: PopulationParams, seed: number): void {
    const rand = mulberry32(seed);
    const innerScene = auToScene(params.innerRadius_AU);
    const outerScene = auToScene(params.outerRadius_AU);
    const color = new THREE.Color();

    // Trojans librate around the Lagrange point. The L4/L5 offset is ±60°
    // of ecliptic longitude, modeled as a mean-anomaly offset from the parent.
    const lagrangeOffset_deg = params.resonancePhase?.lagrangePoint === 'L5' ? -60 : 60;
    const libWidth = params.resonancePhase?.librationAmplitude_deg ?? 25;

    for (let i = 0; i < this.count; i++) {
      const a_scene = innerScene + Math.sqrt(rand()) * (outerScene - innerScene);

      const e = params.eccentricityRange[0]
        + rand() * (params.eccentricityRange[1] - params.eccentricityRange[0]);
      const inc_deg = params.inclinationRange_deg[0]
        + rand() * (params.inclinationRange_deg[1] - params.inclinationRange_deg[0]);

      const period_days = periodDaysForSceneA(a_scene);

      // Mean anomaly clustered around Jupiter's M₀ + lagrangeOffset.
      // Jupiter's M₀ at J2000 ≈ 20.020°. Libration scatters members around
      // the nominal ±60° point.
      const jupiterM0 = 20.020;
      const m0_deg = jupiterM0 + lagrangeOffset_deg + (rand() - 0.5) * 2 * libWidth;

      this.orbits.push(prepareOrbit({
        a_km: a_scene * KM_PER_SCENE_UNIT,
        e,
        inc_deg,
        raan_deg: rand() * 360,
        w_deg: rand() * 360,
        m0_deg,
        period_days,
      }));

      this.scales[i] = params.sizeRange[0] + rand() * (params.sizeRange[1] - params.sizeRange[0]);

      const colVar = (rand() - 0.5) * params.colorVariation;
      color.setRGB(
        Math.min(1, Math.max(0, params.color[0] + colVar)),
        Math.min(1, Math.max(0, params.color[1] + colVar)),
        Math.min(1, Math.max(0, params.color[2] + colVar)),
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
      this.dummy.rotation.set(this.scratch.x * 0.06, this.scratch.y * 0.06, this.scratch.z * 0.06);
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

export class TrojanRenderer {
  readonly group: THREE.Group;
  private l4: TrojanSwarm;
  private l5: TrojanSwarm;

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'TrojanGroup';

    this.l4 = new TrojanSwarm(TROJAN_L4_PARAMS, 0xAAAA);
    this.l5 = new TrojanSwarm(TROJAN_L5_PARAMS, 0xBBBB);

    this.group.add(this.l4.instancedMesh);
    this.group.add(this.l5.instancedMesh);
  }

  update(simTimeSec: number): void {
    if (!this.group.visible) return;
    this.l4.update(simTimeSec);
    this.l5.update(simTimeSec);
  }

  setL4Visible(visible: boolean): void {
    this.l4.setVisible(visible);
  }

  setL5Visible(visible: boolean): void {
    this.l5.setVisible(visible);
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  dispose(): void {
    this.l4.dispose();
    this.l5.dispose();
  }
}
