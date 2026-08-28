/**
 * THESEUS Thrust Plume Renderer
 * =============================
 * Physically motivated rocket engine exhaust plume visualization.
 * Dynamic gradient cone with inner hot core, expanding outer shock cone,
 * and particle emission scaling with engine thrust level.
 */

import * as THREE from 'three';

export class ThrustPlumeRenderer {
  readonly group: THREE.Group;
  private coreMesh: THREE.Mesh;
  private outerMesh: THREE.Mesh;
  private coreMaterial: THREE.MeshBasicMaterial;
  private outerMaterial: THREE.MeshBasicMaterial;

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'ThrustPlumeGroup';

    // ── 1. Inner Hot Core Cone ────────────────────────────────────
    const coreGeo = new THREE.ConeGeometry(0.35, 2.2, 16);
    coreGeo.rotateX(-Math.PI / 2);
    coreGeo.translate(0, 0, -1.1);

    this.coreMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff, // White-hot core
      transparent: true,
      opacity: 0.95,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.coreMesh = new THREE.Mesh(coreGeo, this.coreMaterial);
    this.group.add(this.coreMesh);

    // ── 2. Outer Expanding Exhaust Cone ───────────────────────────
    const outerGeo = new THREE.ConeGeometry(0.7, 3.8, 16);
    outerGeo.rotateX(-Math.PI / 2);
    outerGeo.translate(0, 0, -1.9);

    this.outerMaterial = new THREE.MeshBasicMaterial({
      color: 0xff6600, // Bright orange flame
      transparent: true,
      opacity: 0.7,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.outerMesh = new THREE.Mesh(outerGeo, this.outerMaterial);
    this.group.add(this.outerMesh);

    this.group.visible = false;
  }

  /**
   * Set thrust state (active/inactive) and scale factor (0-1).
   */
  setThrust(active: boolean, level = 1.0): void {
    this.group.visible = active && level > 0.01;
    if (!this.group.visible) return;

    // Pulse/flicker effect
    const flicker = 0.85 + Math.random() * 0.3;
    const s = level * flicker;

    this.coreMesh.scale.set(s * 0.9, s * 0.9, s * (1 + Math.random() * 0.2));
    this.outerMesh.scale.set(s * 1.1, s * 1.1, s * (1 + Math.random() * 0.3));

    this.coreMaterial.opacity = 0.9 * s;
    this.outerMaterial.opacity = 0.65 * s;
  }

  dispose(): void {
    this.coreMesh.geometry.dispose();
    this.coreMaterial.dispose();
    this.outerMesh.geometry.dispose();
    this.outerMaterial.dispose();
  }
}
