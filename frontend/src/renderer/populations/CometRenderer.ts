/**
 * THESEUS Comet Renderer
 * ======================
 * Nucleus, coma, dust tail and ion tail for catalogued comets.
 *
 * Orbits come from CoordinateSystem. This matters more here than anywhere
 * else in the renderer: at e ≈ 0.97 the previous shortcut of substituting the
 * mean anomaly for the true anomaly put a comet on the far side of the Sun
 * from its own drawn orbit. Kepler's equation is now solved properly.
 *
 * Tail direction is derived from the comet-Sun geometry every frame. Nothing
 * about the tails is anchored to the screen or to a fixed axis.
 */

import * as THREE from 'three';
import { NAMED_COMETS } from '../../data/smallBodies';
import { AstronomicalObject } from '../../data/astronomicalObjects';
import {
  PreparedOrbit,
  prepareOrbit,
  orbitPathPoints,
  orbitPositionInto,
  kmToScene,
  auToScene,
} from '../CoordinateSystem';
import {
  ViewContext,
  MIN_APPARENT_RADIUS_PX,
  visualScaleMultiplier,
} from '../VisualScale';

/**
 * Tail length in scene units at peak activity.
 *
 * Real dust tails reach ~1e7–1e8 km (roughly 0.07–0.7 AU). Drawing them at
 * true length would fill the inner system, so tails are drawn at a fraction
 * of their physical extent. This is a declared visual compression, applied to
 * the tail mesh only.
 */
export const DUST_TAIL_LENGTH_SCENE = auToScene(0.09);
export const ION_TAIL_LENGTH_SCENE = auToScene(0.14);

/**
 * Tail widths, as a fraction of tail length.
 *
 * Previously these were multiples of the nucleus radius, which broke once the
 * nucleus was built at its true size: a 5 km nucleus would have produced a
 * tail a few kilometres wide and millions of kilometres long. Keying the
 * width to the length is also closer to the real geometry, since both tails
 * flare with distance from the nucleus.
 */
export const DUST_TAIL_HALF_ANGLE = 0.075;
export const ION_TAIL_HALF_ANGLE = 0.012;

/** Heliocentric distance, in AU, inside which a comet is treated as active. */
export const COMA_ACTIVITY_ONSET_AU = 3.5;

interface CometEntry {
  object: AstronomicalObject;
  orbit: PreparedOrbit;
  group: THREE.Group;
  /** Holds the nucleus and coma. Scaled per frame; tails are NOT inside it. */
  nucleusGroup: THREE.Group;
  coma: THREE.Mesh;
  dustTail: THREE.Mesh;
  ionTail: THREE.Mesh;
  orbitLine: THREE.Line;
  /** True nucleus radius in scene units. */
  physicalRadiusScene: number;
}

export class CometRenderer {
  readonly group: THREE.Group;
  private entries: CometEntry[] = [];

  private pos = new THREE.Vector3();
  private awayFromSun = new THREE.Vector3();
  private dustDir = new THREE.Vector3();
  private quat = new THREE.Quaternion();
  private static readonly TAIL_AXIS = new THREE.Vector3(0, 0, 1);

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'CometsGroup';
    this.initComets();
  }

  private initComets(): void {
    NAMED_COMETS.forEach(obj => {
      if (!obj.orbit) return;

      const cometGroup = new THREE.Group();
      cometGroup.name = `Comet_${obj.id}`;

      // Nucleus and coma are built at TRUE size and share one group, which is
      // scaled per frame by the camera-relative legibility rule. The tails sit
      // outside that group: their length is a physical extent in its own right
      // and must not inherit the nucleus multiplier.
      const nucleusGroup = new THREE.Group();
      nucleusGroup.name = `CometNucleus_${obj.id}`;
      cometGroup.add(nucleusGroup);

      const physicalRadiusScene = kmToScene(obj.radius_km);

      // ── Nucleus: irregular, dark, non-spherical ──────────────────
      const nucleusGeo = new THREE.IcosahedronGeometry(physicalRadiusScene, 2);
      const posAttr = nucleusGeo.attributes.position;
      for (let i = 0; i < posAttr.count; i++) {
        const x = posAttr.getX(i), y = posAttr.getY(i), z = posAttr.getZ(i);
        const d = 1 + Math.sin(x * 10 + y * 8) * 0.22 + Math.cos(z * 13 - x * 6) * 0.12;
        posAttr.setXYZ(i, x * d, y * d, z * d);
      }
      nucleusGeo.computeVertexNormals();
      nucleusGroup.add(new THREE.Mesh(nucleusGeo, new THREE.MeshStandardMaterial({
        color: 0x3a3733,
        roughness: 0.96,
        metalness: 0.0,
      })));

      // ── Coma: scales with the nucleus, so it stays a halo around it ──
      const coma = new THREE.Mesh(
        new THREE.SphereGeometry(physicalRadiusScene * 5, 20, 20),
        new THREE.MeshBasicMaterial({
          color: 0x9fc4cf,
          transparent: true,
          opacity: 0.0,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
        })
      );
      nucleusGroup.add(coma);

      // ── Dust tail: broad, slightly lagging the anti-solar line ───
      const dustGeo = new THREE.ConeGeometry(
        DUST_TAIL_LENGTH_SCENE * DUST_TAIL_HALF_ANGLE, DUST_TAIL_LENGTH_SCENE, 20, 1, true);
      dustGeo.rotateX(Math.PI / 2);
      dustGeo.translate(0, 0, DUST_TAIL_LENGTH_SCENE / 2);
      const dustTail = new THREE.Mesh(dustGeo, new THREE.MeshBasicMaterial({
        color: 0xd8c8a8,
        transparent: true,
        opacity: 0.0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        side: THREE.DoubleSide,
      }));
      cometGroup.add(dustTail);

      // ── Ion tail: narrow, straight, anti-solar ───────────────────
      const ionGeo = new THREE.ConeGeometry(
        ION_TAIL_LENGTH_SCENE * ION_TAIL_HALF_ANGLE, ION_TAIL_LENGTH_SCENE, 14, 1, true);
      ionGeo.rotateX(Math.PI / 2);
      ionGeo.translate(0, 0, ION_TAIL_LENGTH_SCENE / 2);
      const ionTail = new THREE.Mesh(ionGeo, new THREE.MeshBasicMaterial({
        color: 0x6f9fd8,
        transparent: true,
        opacity: 0.0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        side: THREE.DoubleSide,
      }));
      cometGroup.add(ionTail);

      this.group.add(cometGroup);

      const prepared = prepareOrbit(obj.orbit);

      // Highly eccentric orbits need dense sampling near periapsis; the path
      // generator samples in eccentric anomaly, which supplies exactly that.
      const points = orbitPathPoints(prepared, 512);
      const orbitLine = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(points),
        new THREE.LineBasicMaterial({
          color: 0x7fa8bd,
          transparent: true,
          opacity: 0.28,
          depthWrite: false,
        })
      );
      orbitLine.name = `CometOrbit_${obj.id}`;
      this.group.add(orbitLine);

      this.entries.push({
        object: obj, orbit: prepared, group: cometGroup, nucleusGroup,
        coma, dustTail, ionTail, orbitLine, physicalRadiusScene,
      });
    });
  }

  /**
   * @param sunWorldPos World position of the Sun — the scene origin in a
   *                    heliocentric scene, passed explicitly so tail direction
   *                    is never assumed.
   * @param ctx         Camera and viewport, for the nucleus minimum apparent
   *                    size. Affects only how large the nucleus and coma are
   *                    drawn; the comet's position is unaffected.
   */
  update(
    simTimeSec: number,
    sunWorldPos: THREE.Vector3 = new THREE.Vector3(0, 0, 0),
    ctx?: ViewContext
  ): void {
    if (!this.group.visible) return;

    for (const entry of this.entries) {
      orbitPositionInto(entry.orbit, simTimeSec, this.pos);
      entry.group.position.copy(this.pos);

      // Nucleus and coma only. The tails keep their own physical extent.
      if (ctx) {
        entry.nucleusGroup.scale.setScalar(visualScaleMultiplier(
          entry.physicalRadiusScene,
          entry.group.position,
          MIN_APPARENT_RADIUS_PX.COMET_NUCLEUS,
          ctx
        ));
      }

      // Anti-solar direction from actual geometry.
      this.awayFromSun.subVectors(this.pos, sunWorldPos);
      const distanceScene = this.awayFromSun.length();
      if (distanceScene < 1e-6) continue;
      this.awayFromSun.divideScalar(distanceScene);

      // Ion tail: straight down the anti-solar line.
      this.quat.setFromUnitVectors(CometRenderer.TAIL_AXIS, this.awayFromSun);
      entry.ionTail.quaternion.copy(this.quat);

      // Dust tail: the same line, swung back toward the orbital trailing
      // side, which is what produces the characteristic curve.
      this.dustDir.copy(this.awayFromSun);
      this.dustDir.x -= this.pos.z * 1e-6;
      this.dustDir.z += this.pos.x * 1e-6;
      this.dustDir.normalize();
      entry.dustTail.quaternion.setFromUnitVectors(CometRenderer.TAIL_AXIS, this.dustDir);

      // Activity rises steeply as the comet approaches the Sun. Sublimation
      // is negligible beyond the onset distance, so the tails fade out.
      const distanceAU = distanceScene / auToScene(1);
      const activity = Math.min(1, Math.max(0, (COMA_ACTIVITY_ONSET_AU - distanceAU) / COMA_ACTIVITY_ONSET_AU));
      const a2 = activity * activity;

      (entry.coma.material as THREE.MeshBasicMaterial).opacity = 0.35 * a2;
      (entry.dustTail.material as THREE.MeshBasicMaterial).opacity = 0.22 * a2;
      (entry.ionTail.material as THREE.MeshBasicMaterial).opacity = 0.30 * a2;

      const visible = a2 > 0.001;
      entry.coma.visible = visible;
      entry.dustTail.visible = visible;
      entry.ionTail.visible = visible;
    }
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  setOrbitsVisible(visible: boolean): void {
    for (const entry of this.entries) entry.orbitLine.visible = visible;
  }

  getPosition(id: string): THREE.Vector3 | null {
    const entry = this.entries.find(e => e.object.id === id);
    return entry ? entry.group.position : null;
  }

  dispose(): void {
    for (const entry of this.entries) {
      entry.group.traverse(o => {
        if (o instanceof THREE.Mesh) {
          o.geometry.dispose();
          (o.material as THREE.Material).dispose();
        }
      });
      entry.orbitLine.geometry.dispose();
      (entry.orbitLine.material as THREE.Material).dispose();
    }
    this.entries = [];
  }
}
