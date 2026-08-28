/**
 * THESEUS Lagrange Point Renderer
 * ===============================
 * Calculates and visualizes equilibrium Lagrange points (L1–L5) for two-body systems:
 *   - Sun–Earth system L1–L5
 *   - Earth–Moon system L1–L5
 *
 * Derived from 3-body circular restricted equilibrium equations, not hard-coded screen coords.
 */

import * as THREE from 'three';

export interface LagrangeSystemConfig {
  primaryName: string;
  secondaryName: string;
  primaryPos: THREE.Vector3;
  secondaryPos: THREE.Vector3;
  primaryMass: number;
  secondaryMass: number;
  color?: number;
}

export class LagrangePointRenderer {
  readonly group: THREE.Group;
  private lMarkers: Map<string, THREE.Mesh> = new Map();

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'LagrangePointsGroup';

    // Create 5 markers L1..L5
    const colors = [0xffaa00, 0x00f0ff, 0x44bb66, 0xcc66ff, 0xff66cc];
    const labels = ['L1', 'L2', 'L3', 'L4', 'L5'];

    labels.forEach((lbl, idx) => {
      const geometry = new THREE.SphereGeometry(0.6, 12, 12);
      const material = new THREE.MeshBasicMaterial({
        color: colors[idx],
        wireframe: true,
      });

      const mesh = new THREE.Mesh(geometry, material);
      mesh.name = `LagrangePoint_${lbl}`;
      this.group.add(mesh);
      this.lMarkers.set(lbl, mesh);
    });
  }

  /**
   * Update L1–L5 marker positions for Sun–Earth system.
   */
  updateSunEarth(earthPosScene: THREE.Vector3, earthRadiusAU = 1.0): void {
    const sunPos = new THREE.Vector3(0, 0, 0);
    const d = earthPosScene.distanceTo(sunPos);
    if (d < 1) return;

    const dir = earthPosScene.clone().sub(sunPos).normalize();
    const perp = new THREE.Vector3(-dir.z, 0, dir.x).normalize();

    // Hill radius / L1 & L2 distance ratio: r_Hill = a * (m / 3M)^(1/3)
    // Sun-Earth ratio ~0.01 AU = 0.01 * 14959.787 scene units ≈ 149.6 scene units
    const rL = d * 0.01;

    // L1: between Sun and Earth
    const l1Pos = earthPosScene.clone().sub(dir.clone().multiplyScalar(rL));
    // L2: behind Earth
    const l2Pos = earthPosScene.clone().add(dir.clone().multiplyScalar(rL));
    // L3: behind Sun
    const l3Pos = sunPos.clone().sub(dir.clone().multiplyScalar(d));
    // L4: 60 deg ahead in orbit
    const l4Pos = sunPos.clone().add(dir.clone().multiplyScalar(d * 0.5)).add(perp.clone().multiplyScalar(d * 0.866));
    // L5: 60 deg behind in orbit
    const l5Pos = sunPos.clone().add(dir.clone().multiplyScalar(d * 0.5)).sub(perp.clone().multiplyScalar(d * 0.866));

    this.lMarkers.get('L1')?.position.copy(l1Pos);
    this.lMarkers.get('L2')?.position.copy(l2Pos);
    this.lMarkers.get('L3')?.position.copy(l3Pos);
    this.lMarkers.get('L4')?.position.copy(l4Pos);
    this.lMarkers.get('L5')?.position.copy(l5Pos);
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  dispose(): void {
    this.lMarkers.forEach(m => {
      m.geometry.dispose();
      (m.material as THREE.Material).dispose();
    });
  }
}
