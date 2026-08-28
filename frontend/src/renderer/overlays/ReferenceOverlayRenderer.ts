/**
 * THESEUS Reference Frame Scientific Overlay Renderer
 * =================================─────────────────
 * Renders reference coordinate frames:
 *   - Ecliptic reference plane grid (J2000)
 *   - Coordinate axes (X = Vernal Equinox, Y, Z = North Celestial Pole)
 *   - Body-fixed reference frames
 *
 * Restrained scientific aesthetic — thin lines, grayscale/amber palette.
 */

import * as THREE from 'three';

export class ReferenceOverlayRenderer {
  readonly group: THREE.Group;
  readonly eclipticGrid: THREE.GridHelper;
  readonly axesHelper: THREE.AxesHelper;

  constructor(gridSize = 200, gridDivisions = 40) {
    this.group = new THREE.Group();
    this.group.name = 'ReferenceOverlayGroup';

    // ── 1. Ecliptic Reference Grid ─────────────────────────────────
    this.eclipticGrid = new THREE.GridHelper(gridSize, gridDivisions, 0x333344, 0x151522);
    this.eclipticGrid.position.y = -0.05; // Slightly below zero plane to prevent z-fighting
    this.group.add(this.eclipticGrid);

    // ── 2. J2000 Coordinate Axes ──────────────────────────────────
    // X (Vernal Equinox) = Red, Y = Green, Z (North Pole) = Blue
    this.axesHelper = new THREE.AxesHelper(30);
    this.group.add(this.axesHelper);
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  setGridVisible(visible: boolean): void {
    this.eclipticGrid.visible = visible;
  }

  setAxesVisible(visible: boolean): void {
    this.axesHelper.visible = visible;
  }

  dispose(): void {
    this.eclipticGrid.geometry.dispose();
    (this.eclipticGrid.material as THREE.Material).dispose();
    this.axesHelper.geometry.dispose();
    (this.axesHelper.material as THREE.Material).dispose();
  }
}
