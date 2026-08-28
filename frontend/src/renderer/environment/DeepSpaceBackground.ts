/**
 * THESEUS Deep Space Background Renderer
 * =====================================
 * Subtle background structure for deep space.
 * Provides faint, non-distracting cosmic depth without cartoonish nebulae.
 */

import * as THREE from 'three';

export class DeepSpaceBackground {
  readonly mesh: THREE.Mesh;

  constructor() {
    const geometry = new THREE.SphereGeometry(15000, 32, 32);

    const material = new THREE.MeshBasicMaterial({
      color: 0x010307,
      side: THREE.BackSide,
      depthWrite: false,
    });

    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.name = 'DeepSpaceBackgroundMesh';
  }

  setVisible(visible: boolean): void {
    this.mesh.visible = visible;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
  }
}
