/**
 * THESEUS Near-Earth Object Renderer
 * ==================================
 * Renders catalogued near-Earth asteroids (Apollo, Aten, Amor, PHA classes)
 * from real element sets, colour-coded by class.
 *
 * Positions and orbit lines both come from CoordinateSystem, so an object is
 * always drawn on its own path. Meshes are built at each body's TRUE radius;
 * legibility comes from VisualScale's camera-relative minimum apparent size,
 * applied as a per-frame mesh scale. No position is ever exaggerated.
 */

import * as THREE from 'three';
import { NAMED_ASTEROIDS } from '../../data/smallBodies';
import { AstronomicalObject } from '../../data/astronomicalObjects';
import {
  PreparedOrbit,
  prepareOrbit,
  orbitPathPoints,
  orbitPositionInto,
  kmToScene,
} from '../CoordinateSystem';
import {
  ViewContext,
  MIN_APPARENT_RADIUS_PX,
  visualScaleMultiplier,
} from '../VisualScale';

interface NEOEntry {
  object: AstronomicalObject;
  orbit: PreparedOrbit;
  mesh: THREE.Mesh;
  orbitLine: THREE.Line | null;
  /** True radius in scene units. Geometry is built at exactly this size. */
  physicalRadiusScene: number;
}

export class NEORenderer {
  readonly group: THREE.Group;

  /** Near-Earth objects and PHAs. */
  readonly neoGroup: THREE.Group;
  /** Catalogued main-belt asteroids (Vesta, Pallas, Hygiea, Ida, Gaspra). */
  readonly namedAsteroidGroup: THREE.Group;

  private entries: NEOEntry[] = [];
  private scratch = new THREE.Vector3();

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'SmallBodyGroup';

    this.neoGroup = new THREE.Group();
    this.neoGroup.name = 'NEOGroup';
    this.group.add(this.neoGroup);

    this.namedAsteroidGroup = new THREE.Group();
    this.namedAsteroidGroup.name = 'NamedAsteroidGroup';
    this.group.add(this.namedAsteroidGroup);

    this.initCatalog();
  }

  private static colorFor(obj: AstronomicalObject): number {
    if (obj.isPHA) return 0xd94a3d;         // potentially hazardous
    if (obj.neoClass === 'ATEN') return 0xe08a2a;
    if (obj.neoClass === 'APOLLO') return 0xd6a83c;
    if (obj.neoClass === 'AMOR') return 0xb9b45a;
    if (obj.type === 'NEO') return 0xc9a05a;
    return 0x9a8f7a;                        // main-belt, deliberately duller
  }

  private static isNEO(obj: AstronomicalObject): boolean {
    return obj.type === 'NEO' || !!obj.isPHA;
  }

  private initCatalog(): void {
    // Every catalogued named asteroid, not just the near-Earth ones. The
    // main-belt entries (Vesta, Pallas, Hygiea, Ida, Gaspra) were previously
    // filtered out entirely and never appeared in the scene at all.
    NAMED_ASTEROIDS.forEach(obj => {
      if (!obj.orbit) return;
      const isNeo = NEORenderer.isNEO(obj);
      const parent = isNeo ? this.neoGroup : this.namedAsteroidGroup;

      const colorHex = NEORenderer.colorFor(obj);

      // Geometry is built at the body's TRUE radius. Legibility is handled
      // per frame by a camera-relative scale, so zooming in eventually shows
      // the object at its real size.
      const physicalRadiusScene = kmToScene(obj.radius_km);

      const geometry = new THREE.IcosahedronGeometry(physicalRadiusScene, 2);
      const material = new THREE.MeshStandardMaterial({
        color: colorHex,
        roughness: 0.85,
        metalness: 0.05,
      });

      const mesh = new THREE.Mesh(geometry, material);
      mesh.name = `SmallBody_${obj.id}`;
      parent.add(mesh);

      const prepared = prepareOrbit(obj.orbit);

      const points = orbitPathPoints(prepared, 256);
      const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
      const lineMat = new THREE.LineBasicMaterial({
        color: colorHex,
        transparent: true,
        opacity: 0.35,
        depthWrite: false,
      });
      const orbitLine = new THREE.Line(lineGeo, lineMat);
      orbitLine.name = `SmallBodyOrbit_${obj.id}`;
      parent.add(orbitLine);

      this.entries.push({ object: obj, orbit: prepared, mesh, orbitLine, physicalRadiusScene });
    });
  }

  /**
   * @param ctx Camera and viewport, for the minimum-apparent-size rule.
   *            Positions are computed from the orbit regardless; `ctx` only
   *            affects how large each body is drawn.
   */
  update(simTimeSec: number, ctx?: ViewContext): void {
    if (!this.group.visible) return;
    for (const entry of this.entries) {
      if (!entry.mesh.parent?.visible) continue;
      orbitPositionInto(entry.orbit, simTimeSec, this.scratch);
      entry.mesh.position.copy(this.scratch);

      if (ctx) {
        entry.mesh.scale.setScalar(visualScaleMultiplier(
          entry.physicalRadiusScene,
          entry.mesh.position,
          MIN_APPARENT_RADIUS_PX.SMALL_BODY,
          ctx
        ));
      }
    }
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  /** Near-Earth objects and PHAs. */
  setNEOsVisible(visible: boolean): void {
    this.neoGroup.visible = visible;
  }

  /** Catalogued main-belt asteroids. */
  setNamedAsteroidsVisible(visible: boolean): void {
    this.namedAsteroidGroup.visible = visible;
  }

  setOrbitsVisible(visible: boolean): void {
    for (const entry of this.entries) {
      if (entry.orbitLine) entry.orbitLine.visible = visible;
    }
  }

  /** World position of a rendered NEO, for camera focus and selection. */
  getPosition(id: string): THREE.Vector3 | null {
    const entry = this.entries.find(e => e.object.id === id);
    return entry ? entry.mesh.position : null;
  }

  dispose(): void {
    for (const entry of this.entries) {
      entry.mesh.geometry.dispose();
      (entry.mesh.material as THREE.Material).dispose();
      if (entry.orbitLine) {
        entry.orbitLine.geometry.dispose();
        (entry.orbitLine.material as THREE.Material).dispose();
      }
    }
    this.entries = [];
  }
}
