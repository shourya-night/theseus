/**
 * THESEUS Artificial Objects Renderer
 * ===================================
 * Visualizes artificial catalog space objects (ISS, JWST, Hubble, Voyager 1).
 * Features distinct geometries per category (stations, telescopes, probes),
 * orbit paths, and provenance badges.
 */

import * as THREE from 'three';
import { CATALOG_ARTIFICIAL_OBJECTS, ArtificialObject } from '../../data/artificialObjects';
import {
  PreparedOrbit,
  prepareOrbit,
  orbitPathPoints,
  orbitPositionInto,
  isEllipticalOrbit,
  kmToScene,
} from '../CoordinateSystem';
import {
  ViewContext,
  MIN_APPARENT_RADIUS_PX,
  visualScaleMultiplier,
} from '../VisualScale';

export class ArtificialObjectRenderer {
  readonly group: THREE.Group;
  private objectMeshes: Map<string, THREE.Mesh> = new Map();
  private orbitLines: Map<string, THREE.Line> = new Map();
  private orbits: Map<string, PreparedOrbit> = new Map();
  private physicalRadii: Map<string, number> = new Map();
  private scratch = new THREE.Vector3();

  private warnedParents = new Set<string>();

  /** Objects the elliptical propagator cannot place, and why. */
  readonly unplaceable: Array<{ id: string; name: string; reason: string }> = [];

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'ArtificialObjectsGroup';

    this.initObjects();
  }

  private initObjects(): void {
    CATALOG_ARTIFICIAL_OBJECTS.forEach(obj => {
      const colorHex = parseInt(obj.color.replace('#', ''), 16);

      // Voyager 1 and New Horizons are catalogued with e = 1.0 (escape
      // trajectories). The elliptical solver cannot represent them: r =
      // a(1 - e cos E) collapses to 0 at epoch, which placed both of them
      // exactly at the focus — the centre of the Sun. Rather than substitute a
      // plausible-looking position, they are left out of the scene entirely and
      // recorded here so the UI can report DATA UNAVAILABLE.
      if (!obj.orbit || !isEllipticalOrbit(obj.orbit)) {
        this.unplaceable.push({
          id: obj.id,
          name: obj.name,
          reason: obj.orbit
            ? `e = ${obj.orbit.e} is not an elliptical orbit; an escape trajectory needs a state vector, not Keplerian elements`
            : 'no orbital elements in catalog',
        });
        return;
      }

      // Geometry at true hard-body size; legibility comes from VisualScale.
      const sizeScene = kmToScene(obj.hardBodyRadius_m / 1000);

      // Geometry based on category
      let geometry: THREE.BufferGeometry;
      if (obj.category === 'STATION') {
        geometry = new THREE.BoxGeometry(sizeScene * 2, sizeScene * 0.5, sizeScene * 1.5);
      } else if (obj.category === 'TELESCOPE') {
        geometry = new THREE.CylinderGeometry(sizeScene * 0.6, sizeScene * 0.6, sizeScene * 1.8, 12);
      } else {
        geometry = new THREE.OctahedronGeometry(sizeScene, 0);
      }

      const material = new THREE.MeshStandardMaterial({
        color: colorHex,
        emissive: colorHex,
        emissiveIntensity: 0.3,
        roughness: 0.3,
        metalness: 0.8,
      });

      const mesh = new THREE.Mesh(geometry, material);
      mesh.name = `Artificial_${obj.id}`;
      this.group.add(mesh);
      this.objectMeshes.set(obj.id, mesh);

      // Prepared elements drive both the mesh position and the path line,
      // so the object always sits on the orbit that is drawn for it.
      this.orbits.set(obj.id, prepareOrbit(obj.orbit));
      this.physicalRadii.set(obj.id, sizeScene);

      // Orbit Line
      if (obj.orbit.period_days < 1000) {
        const line = this.createOrbitLine(obj, colorHex);
        this.group.add(line);
        this.orbitLines.set(obj.id, line);
      }
    });
  }

  private createOrbitLine(obj: ArtificialObject, colorHex: number): THREE.Line {
    const prepared = this.orbits.get(obj.id)!;
    const geometry = new THREE.BufferGeometry().setFromPoints(orbitPathPoints(prepared, 192));
    const material = new THREE.LineBasicMaterial({
      color: colorHex,
      transparent: true,
      opacity: 0.4,
      depthWrite: false,
    });

    return new THREE.Line(geometry, material);
  }

  update(
    timeSeconds: number,
    bodyPositions: Map<string, THREE.Vector3>,
    ctx?: ViewContext
  ): void {
    if (!this.group.visible) return;

    CATALOG_ARTIFICIAL_OBJECTS.forEach(obj => {
      const mesh = this.objectMeshes.get(obj.id);
      const prepared = this.orbits.get(obj.id);
      if (!mesh || !prepared) return;

      // Element sets are referenced to the parent body, so the parent's world
      // position is the orbit focus. An unresolvable parent must hide the
      // object rather than silently dropping it on the scene origin, which is
      // the Sun.
      const parentPos = bodyPositions.get(obj.parent.trim().toLowerCase());
      if (!parentPos) {
        if (!this.warnedParents.has(obj.parent)) {
          this.warnedParents.add(obj.parent);
          console.warn(
            `[THESEUS] Artificial object "${obj.name}" declares parent "${obj.parent}", ` +
            `which is not in the astronomical catalog. It cannot be placed and is hidden.`
          );
        }
        mesh.visible = false;
        const missingLine = this.orbitLines.get(obj.id);
        if (missingLine) missingLine.visible = false;
        return;
      }
      mesh.visible = true;

      orbitPositionInto(prepared, timeSeconds, this.scratch);
      mesh.position.addVectors(parentPos, this.scratch);
      mesh.rotation.y = timeSeconds * 0.1;

      if (ctx) {
        const r = this.physicalRadii.get(obj.id) ?? 0;
        mesh.scale.setScalar(visualScaleMultiplier(r, mesh.position, MIN_APPARENT_RADIUS_PX.SPACECRAFT, ctx));
      }

      const line = this.orbitLines.get(obj.id);
      if (line) line.position.copy(parentPos);
    });
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  dispose(): void {
    this.objectMeshes.forEach(m => {
      m.geometry.dispose();
      (m.material as THREE.Material).dispose();
    });
    this.orbitLines.forEach(l => {
      l.geometry.dispose();
      (l.material as THREE.Material).dispose();
    });
  }
}
