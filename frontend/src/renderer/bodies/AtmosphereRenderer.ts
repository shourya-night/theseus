/**
 * THESEUS Atmosphere Renderer
 * ===========================
 * Creates physically motivated atmospheric scattering shells around planets.
 * Renders a transparent sphere slightly larger than the planet surface with
 * Rayleigh/Mie scattering approximation, limb brightening, and day/night transitions.
 */

import * as THREE from 'three';
import { AtmosphereParams } from '../../data/astronomicalObjects';
import { GLSL_ATMOSPHERE_VERTEX, GLSL_ATMOSPHERE_FRAGMENT } from '../ShaderLib';

export class AtmosphereRenderer {
  readonly mesh: THREE.Mesh;
  private material: THREE.ShaderMaterial;

  constructor(planetRadiusScene: number, params: AtmosphereParams) {
    const scaleHeight = params.scaleHeight ?? 0.015;
    const atmosphereRadius = planetRadiusScene * (1.0 + scaleHeight * 4.0);

    const geometry = new THREE.SphereGeometry(atmosphereRadius, 48, 48);

    const atmosphereColor = new THREE.Color(...params.color);

    this.material = new THREE.ShaderMaterial({
      vertexShader: GLSL_ATMOSPHERE_VERTEX,
      fragmentShader: GLSL_ATMOSPHERE_FRAGMENT,
      uniforms: {
        uSunDirection: { value: new THREE.Vector3(1, 0, 0) },
        uAtmosphereColor: { value: atmosphereColor },
        uAtmosphereDensity: { value: params.density },
        uAtmosphereRadius: { value: atmosphereRadius },
        uPlanetRadius: { value: planetRadiusScene },
      },
      transparent: true,
      // FRONT side, deliberately.
      //
      // The rim term is `1 - max(dot(viewDir, normal), 0)`. On the FRONT
      // hemisphere that is 0 facing the camera and 1 at the silhouette, which
      // is a limb glow: transparent over the body, brightest at the edge.
      //
      // On BackSide — what this used to be — only the FAR hemisphere is drawn,
      // where the outward normal points away from the camera, so dot < 0, the
      // max() clamps to 0, and rim is 1 across the entire disc. The shell then
      // renders as a flat uniformly-bright ball of its own, larger than and
      // visually detached from the body it belongs to.
      side: THREE.FrontSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    this.mesh = new THREE.Mesh(geometry, this.material);
    this.mesh.name = 'AtmosphereShell';
  }

  updateSunDirection(sunDir: THREE.Vector3): void {
    this.material.uniforms.uSunDirection.value.copy(sunDir);
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.material.dispose();
  }
}
