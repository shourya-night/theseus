/**
 * THESEUS Sphere of Influence (SOI) / Hill Sphere Renderer
 * =================================────────────────========
 * Renders gravitational Sphere of Influence (SOI) and Hill sphere boundary shells
 * around celestial bodies.
 *
 * Formulae:
 *   Laplace SOI:  r_SOI = a * (m / M)^(2/5)
 *   Hill Sphere:  r_Hill = a * (m / 3M)^(1/3)
 */

import * as THREE from 'three';
import { AstronomicalObject } from '../../data/astronomicalObjects';

export class SOIRenderer {
  readonly group: THREE.Group;
  private soiShells: Map<string, THREE.Mesh> = new Map();

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'SOIGroup';
  }

  /**
   * Add a Sphere of Influence shell around a celestial body.
   */
  addBodySOI(obj: AstronomicalObject, parentMassKg: number): void {
    if (!obj.orbit || !obj.mass_kg || parentMassKg <= 0) return;

    // Laplace SOI radius in km: r_SOI = a * (m / M)^(2/5)
    const r_soi_km = obj.orbit.a_km * Math.pow(obj.mass_kg / parentMassKg, 0.4);
    const r_soi_scene = r_soi_km / 10000.0;

    const geometry = new THREE.SphereGeometry(r_soi_scene, 32, 32);
    const material = new THREE.MeshBasicMaterial({
      color: new THREE.Color(...(obj.color ?? [0.0, 0.9, 1.0])),
      transparent: true,
      opacity: 0.07,
      wireframe: true,
      depthWrite: false,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = `SOIShell_${obj.id}`;
    this.group.add(mesh);
    this.soiShells.set(obj.id, mesh);
  }

  updatePosition(objId: string, worldPos: THREE.Vector3): void {
    const shell = this.soiShells.get(objId);
    if (shell) shell.position.copy(worldPos);
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  dispose(): void {
    this.soiShells.forEach(m => {
      m.geometry.dispose();
      (m.material as THREE.Material).dispose();
    });
  }
}
