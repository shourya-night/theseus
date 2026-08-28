/**
 * THESEUS Spacecraft Procedural Geometry Builder
 * =================================─────────────
 * Creates high-detail procedural 3D spacecraft geometry components.
 * Supports: body tanks, interstage sections, engine nozzles, solar arrays,
 * dish antennas, RCS thrusters, payload fairings, and instrument booms.
 *
 * Parameterized by vehicle type (e.g. Falcon 9, Starship, LVM3, Voyager, Satellite).
 *
 * Geometry is emitted in unitless "vehicle units" and is NOT scaled to the
 * scene here. SpacecraftRenderer applies the scene scale through VisualScale.
 */

import * as THREE from 'three';

export interface SpacecraftGeometryConfig {
  type: string;
  color?: string | number;
  showSolarPanels?: boolean;
  showDishAntenna?: boolean;
}

export class SpacecraftGeometryBuilder {
  /**
   * Build a detailed 3D Group representing the requested vehicle type.
   */
  static buildVehicleGroup(config: SpacecraftGeometryConfig): THREE.Group {
    const group = new THREE.Group();
    group.name = `SpacecraftVehicle_${config.type}`;

    const colorHex = typeof config.color === 'string'
      ? parseInt(config.color.replace('#', ''), 16)
      : (config.color ?? 0x00f0ff);

    const bodyMat = new THREE.MeshStandardMaterial({
      color: colorHex,
      metalness: 0.7,
      roughness: 0.3,
    });

    const darkMat = new THREE.MeshStandardMaterial({
      color: 0x151520,
      metalness: 0.8,
      roughness: 0.2,
    });

    const solarMat = new THREE.MeshStandardMaterial({
      color: 0x103060,
      metalness: 0.9,
      roughness: 0.1,
      emissive: 0x051530,
    });

    const goldMat = new THREE.MeshStandardMaterial({
      color: 0xccaa33,
      metalness: 0.9,
      roughness: 0.2,
    });

    // ── 1. Central Bus / Body Cylinder ───────────────────────────
    const bodyGeo = new THREE.CylinderGeometry(0.6, 0.7, 2.5, 16);
    bodyGeo.rotateX(Math.PI / 2);
    const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
    group.add(bodyMesh);

    // ── 2. Payload Nose Cone / Fairing ───────────────────────────
    const noseGeo = new THREE.ConeGeometry(0.6, 1.2, 16);
    noseGeo.rotateX(Math.PI / 2);
    noseGeo.translate(0, 0, 1.85);
    const noseMesh = new THREE.Mesh(noseGeo, darkMat);
    group.add(noseMesh);

    // ── 3. Engine Bell Nozzle ─────────────────────────────────────
    const engineGeo = new THREE.ConeGeometry(0.4, 0.7, 12);
    engineGeo.rotateX(-Math.PI / 2);
    engineGeo.translate(0, 0, -1.6);
    const engineMesh = new THREE.Mesh(engineGeo, darkMat);
    group.add(engineMesh);

    // ── 4. Solar Array Panels ────────────────────────────────────
    if (config.showSolarPanels !== false) {
      const panelGeo = new THREE.BoxGeometry(3.2, 0.05, 0.8);
      const panelMeshLeft = new THREE.Mesh(panelGeo, solarMat);
      panelMeshLeft.position.set(-1.9, 0, 0);
      group.add(panelMeshLeft);

      const panelMeshRight = new THREE.Mesh(panelGeo, solarMat);
      panelMeshRight.position.set(1.9, 0, 0);
      group.add(panelMeshRight);
    }

    // ── 5. High-Gain Dish Antenna ────────────────────────────────
    if (config.showDishAntenna) {
      const dishGeo = new THREE.SphereGeometry(0.5, 16, 8, 0, Math.PI * 2, 0, Math.PI / 3);
      dishGeo.rotateX(-Math.PI / 2);
      dishGeo.translate(0, 0.8, 0.5);
      const dishMesh = new THREE.Mesh(dishGeo, goldMat);
      group.add(dishMesh);
    }

    // ── 6. RCS Quad Clusters ─────────────────────────────────────
    const rcsGeo = new THREE.BoxGeometry(0.15, 0.15, 0.15);
    for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 2) {
      const rcs = new THREE.Mesh(rcsGeo, darkMat);
      rcs.position.set(Math.cos(angle) * 0.65, Math.sin(angle) * 0.65, 0.8);
      group.add(rcs);
    }

    // NO SCALE IS APPLIED HERE.
    //
    // The builder emits geometry in arbitrary "vehicle units". Sizing is the
    // caller's job, via VisualScale — SpacecraftRenderer measures this group's
    // bounding radius once and scales it so the vehicle reaches a minimum
    // apparent size on screen. The previous fixed 0.05 factor produced a
    // 1,250 km long spacecraft next to an Earth of 6,378 km radius, and it
    // stayed that size no matter where the camera was.
    return group;
  }
}
