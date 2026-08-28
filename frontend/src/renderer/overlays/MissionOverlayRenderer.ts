/**
 * THESEUS Mission Overlay Renderer
 * ================================
 * Visualizes mission-specific events and outputs:
 *   - Impulsive burn markers (ΔV1, ΔV2)
 *   - Waypoint indicators
 *   - Conjunction closest approach distance indicators
 *   - Collision warning zones & 3D covariance uncertainty ellipsoids
 */

import * as THREE from 'three';
import { engineToThreePos, engineToThreePosInto } from '../CoordinateSystem';
import { MissionEvent, MultiConjunctionEvent, PhysicalCollisionEvent } from '../../types/mission';

export class MissionOverlayRenderer {
  readonly group: THREE.Group;
  private markers: THREE.Mesh[] = [];
  private warningLines: THREE.Line[] = [];

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'MissionOverlayGroup';
  }

  /**
   * Render burn and waypoint markers from mission events timeline.
   */
  updateEvents(events: MissionEvent[], stateHistoryPositions: [number, number, number][] = []): void {
    // Clear previous markers
    this.markers.forEach(m => {
      this.group.remove(m);
      m.geometry.dispose();
      (m.material as THREE.Material).dispose();
    });
    this.markers = [];

    events.forEach(evt => {
      if (evt.type === 'MANEUVER_START' || evt.type === 'MANEUVER_END' || evt.type === 'WAYPOINT') {
        const color = evt.type.includes('MANEUVER') ? 0xff9900 : 0x44bb66;
        const geometry = new THREE.OctahedronGeometry(1.2, 0);
        const material = new THREE.MeshBasicMaterial({
          color,
          wireframe: true,
        });

        const marker = new THREE.Mesh(geometry, material);
        marker.name = `EventMarker_${evt.name}`;

        // Find position from state history matching event time
        if (stateHistoryPositions.length > 0) {
          const idx = Math.min(stateHistoryPositions.length - 1, Math.floor((evt.time / 86400) * 10));
          const pos = stateHistoryPositions[idx] ?? stateHistoryPositions[0];
          engineToThreePosInto(pos, marker.position);
        }

        this.group.add(marker);
        this.markers.push(marker);
      }
    });
  }

  /**
   * Render conjunction miss distance indicators and collision warnings.
   */
  updateConjunctions(conjunctions: MultiConjunctionEvent[], collisions: PhysicalCollisionEvent[]): void {
    // Clear previous warning lines
    this.warningLines.forEach(l => {
      this.group.remove(l);
      l.geometry.dispose();
      (l.material as THREE.Material).dispose();
    });
    this.warningLines = [];

    conjunctions.forEach(conj => {
      if (conj.action_required || conj.is_physical_collision) {
        const p1 = engineToThreePos(conj.r_rel_m);
        const p2 = new THREE.Vector3(0, 0, 0); // Relative miss vector

        const lineGeo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
        const lineMat = new THREE.LineDashedMaterial({
          color: conj.is_physical_collision ? 0xcc3333 : 0xffaa00,
          dashSize: 1,
          gapSize: 0.5,
        });

        const line = new THREE.Line(lineGeo, lineMat);
        line.computeLineDistances();
        this.group.add(line);
        this.warningLines.push(line);
      }
    });
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  dispose(): void {
    this.markers.forEach(m => {
      m.geometry.dispose();
      (m.material as THREE.Material).dispose();
    });
    this.warningLines.forEach(l => {
      l.geometry.dispose();
      (l.material as THREE.Material).dispose();
    });
  }
}
