/**
 * THESEUS Ring System Renderer
 * ============================
 * Scientifically detailed planetary ring system renderer.
 * Features radial density profiles, divisions/gaps (Cassini, Encke),
 * planet-cast shadows onto rings, ring shadows cast onto planets,
 * and correct orbital plane tilt.
 */

import * as THREE from 'three';
import { RingSystemParams } from '../../data/astronomicalObjects';
import { GLSL_RING_VERTEX, GLSL_RING_FRAGMENT } from '../ShaderLib';

export class RingRenderer {
  readonly mesh: THREE.Mesh;
  private material: THREE.ShaderMaterial;

  constructor(planetRadiusScene: number, params: RingSystemParams) {
    const innerRadius = planetRadiusScene * params.innerRadius;
    const outerRadius = planetRadiusScene * params.outerRadius;

    // High angular segment count for smooth circles; several radial segments
    // so the shader's radial profile is sampled evenly across the span.
    const geometry = new THREE.RingGeometry(innerRadius, outerRadius, 256, 12);

    // Re-orient geometry to lie in XZ plane (equatorial plane)
    geometry.rotateX(-Math.PI / 2);

    const ringColor = new THREE.Color(...params.color);

    this.material = new THREE.ShaderMaterial({
      vertexShader: GLSL_RING_VERTEX,
      fragmentShader: GLSL_RING_FRAGMENT,
      uniforms: {
        uSunDirection: { value: new THREE.Vector3(1, 0, 0) },
        uPlanetPosition: { value: new THREE.Vector3(0, 0, 0) },
        uPlanetRadius: { value: planetRadiusScene },
        uInnerRadius: { value: innerRadius },
        uOuterRadius: { value: outerRadius },
        uRingColor: { value: ringColor },
        uRingDensityScale: { value: params.densityScale },
      },
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false,
      depthTest: true,
    });

    this.mesh = new THREE.Mesh(geometry, this.material);
    this.mesh.name = 'RingMesh';
    this.mesh.renderOrder = 1;

    // NO TILT IS APPLIED HERE.
    //
    // The ring plane is the planet's equatorial plane, and the equatorial
    // plane is already established by the axial tilt that PlanetRenderer puts
    // on the parent group. Applying params.tilt_deg again here rotated the
    // rings a second time — for Saturn, 26.73 deg of axial tilt plus another
    // 26.73 from this line, leaving the rings at roughly 53 deg to the
    // ecliptic instead of sitting square on the planet's equator.
    //
    // params.tilt_deg is retained in the catalog for any ring system genuinely
    // inclined to its planet's equator, but it must then be applied as a
    // difference from the equatorial plane, not as an absolute rotation.
  }

  update(sunDir: THREE.Vector3, planetWorldPos: THREE.Vector3): void {
    this.material.uniforms.uSunDirection.value.copy(sunDir);
    this.material.uniforms.uPlanetPosition.value.copy(planetWorldPos);
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.material.dispose();
  }
}
