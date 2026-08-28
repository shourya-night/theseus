/**
 * THESEUS Meteoroid Stream Renderer
 * =================================
 * Statistical particle-based visualization for major meteoroid streams
 * (Perseids, Leonids, Geminids, Orionids, Quadrantids, Taurids).
 *
 * Particles are distributed along the parent body's orbital path with
 * radial and cross-track dispersion. Each particle has its own Keplerian
 * element set (slightly perturbed from the stream's reference orbit), so
 * positions evolve correctly over time through CoordinateSystem.
 *
 * Uses Points per stream (particles are too small / too numerous for
 * individual meshes). Additive blending gives a luminous appearance.
 */

import * as THREE from 'three';
import { METEOR_STREAMS, MeteorStream } from '../../data/smallBodies';
import {
  PreparedOrbit,
  prepareOrbit,
  orbitPositionInto,
  AU_KM,
} from '../CoordinateSystem';

/** Deterministic pseudo-random source (Mulberry32). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface StreamEntry {
  stream: MeteorStream;
  orbits: PreparedOrbit[];
  points: THREE.Points;
}

export class MeteorStreamRenderer {
  readonly group: THREE.Group;
  private streams: StreamEntry[] = [];
  private scratch = new THREE.Vector3();

  constructor(seed = 0xD00D) {
    this.group = new THREE.Group();
    this.group.name = 'MeteorStreamsGroup';

    let seedOffset = seed;
    METEOR_STREAMS.forEach(stream => {
      const entry = this.createStream(stream, seedOffset);
      this.group.add(entry.points);
      this.streams.push(entry);
      seedOffset += 0x1234;
    });

    this.update(0);
  }

  private createStream(stream: MeteorStream, seed: number): StreamEntry {
    const rand = mulberry32(seed);
    const count = stream.particleCount;
    const elem = stream.orbitalElements;

    // Kepler period from a and mu_sun.
    const a_km = elem.a_AU * AU_KM;
    const periodSec = 2 * Math.PI * Math.sqrt((a_km * a_km * a_km) / 1.32712440018e11);
    const period_days = periodSec / 86400;

    // Each particle gets its own slightly perturbed orbit. This spreads them
    // along the stream tube rather than clumping at one true anomaly.
    const orbits: PreparedOrbit[] = [];
    for (let i = 0; i < count; i++) {
      orbits.push(prepareOrbit({
        a_km: a_km * (1 + (rand() - 0.5) * 0.04),          // ±2% semi-major axis spread
        e: Math.min(0.999, Math.max(0, elem.e + (rand() - 0.5) * 0.02)), // ±1% eccentricity spread
        inc_deg: elem.inc_deg + (rand() - 0.5) * 3,          // ±1.5° inclination spread
        raan_deg: rand() * 360,                               // spread around the node line
        w_deg: rand() * 360,                                  // spread in argument of periapsis
        m0_deg: (i / count) * 360 + (rand() - 0.5) * 20,     // distribute along the orbit
        period_days,
      }));
    }

    // Geometry: Points with position buffer updated per frame.
    const positions = new Float32Array(count * 3);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setDrawRange(0, count);

    const material = new THREE.PointsMaterial({
      color: new THREE.Color(...stream.color),
      size: 1.5,
      sizeAttenuation: false,
      transparent: true,
      opacity: 0.45,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    const points = new THREE.Points(geometry, material);
    points.name = `MeteorStream_${stream.id}`;
    points.frustumCulled = false;

    return { stream, orbits, points };
  }

  update(simTimeSec: number): void {
    if (!this.group.visible) return;

    for (const entry of this.streams) {
      if (!entry.points.visible) continue;

      const posArray = (entry.points.geometry.attributes.position as THREE.BufferAttribute).array as Float32Array;
      const count = entry.orbits.length;

      for (let i = 0; i < count; i++) {
        orbitPositionInto(entry.orbits[i], simTimeSec, this.scratch);
        posArray[i * 3]     = this.scratch.x;
        posArray[i * 3 + 1] = this.scratch.y;
        posArray[i * 3 + 2] = this.scratch.z;
      }

      (entry.points.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    }
  }

  setStreamVisible(streamId: string, visible: boolean): void {
    const entry = this.streams.find(s => s.stream.id === streamId);
    if (entry) entry.points.visible = visible;
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  dispose(): void {
    for (const entry of this.streams) {
      entry.points.geometry.dispose();
      (entry.points.material as THREE.Material).dispose();
    }
    this.streams = [];
  }
}
