/**
 * THESEUS Planet Shaders
 * ======================
 * Constructs customized Three.js ShaderMaterial instances for celestial bodies.
 * Uses procedural GLSL shaders from ShaderLib.ts to generate realistic surface detail
 * without requiring external image downloads.
 */

import * as THREE from 'three';
import { AstronomicalObject } from '../../data/astronomicalObjects';
import {
  GLSL_PLANET_VERTEX,
  GLSL_ROCKY_FRAGMENT,
  GLSL_EARTH_FRAGMENT,
  GLSL_GAS_GIANT_FRAGMENT,
  GLSL_ICE_GIANT_FRAGMENT,
  GLSL_MARS_FRAGMENT,
  GLSL_VENUS_FRAGMENT,
} from '../ShaderLib';

export interface PlanetMaterialOptions {
  sunDirection?: THREE.Vector3;
  time?: number;
}

export class PlanetShaderFactory {
  /**
   * Create a procedural ShaderMaterial tailored to the specific astronomical object.
   */
  static createMaterial(
    obj: AstronomicalObject,
    opts: PlanetMaterialOptions = {}
  ): THREE.ShaderMaterial {
    const sunDir = opts.sunDirection ?? new THREE.Vector3(1, 0.2, 0.5).normalize();
    const time = opts.time ?? 0;
    const surface = obj.surface;

    // ── 1. Earth ──────────────────────────────────────────────────
    if (obj.id === 'earth') {
      return new THREE.ShaderMaterial({
        vertexShader: GLSL_PLANET_VERTEX,
        fragmentShader: GLSL_EARTH_FRAGMENT,
        uniforms: {
          uSunDirection: { value: sunDir },
          uTime: { value: time },
          uCloudCover: { value: 0.55 },
        },
      });
    }

    // ── 2. Mars ───────────────────────────────────────────────────
    if (obj.id === 'mars') {
      return new THREE.ShaderMaterial({
        vertexShader: GLSL_PLANET_VERTEX,
        fragmentShader: GLSL_MARS_FRAGMENT,
        uniforms: {
          uSunDirection: { value: sunDir },
          uTime: { value: time },
        },
      });
    }

    // ── 3. Venus ──────────────────────────────────────────────────
    if (obj.id === 'venus') {
      return new THREE.ShaderMaterial({
        vertexShader: GLSL_PLANET_VERTEX,
        fragmentShader: GLSL_VENUS_FRAGMENT,
        uniforms: {
          uSunDirection: { value: sunDir },
          uTime: { value: time },
        },
      });
    }

    // ── 4. Gas Giants (Jupiter, Saturn) ───────────────────────────
    if (obj.gasGiant) {
      const gg = obj.gasGiant;
      const bandColors = gg.bandColors.map(c => new THREE.Color(c[0], c[1], c[2]));
      // Pad to 6 colors if needed
      while (bandColors.length < 6) {
        bandColors.push(bandColors[bandColors.length - 1] ?? new THREE.Color(0.8, 0.7, 0.5));
      }

      return new THREE.ShaderMaterial({
        vertexShader: GLSL_PLANET_VERTEX,
        fragmentShader: GLSL_GAS_GIANT_FRAGMENT,
        uniforms: {
          uSunDirection: { value: sunDir },
          uBandColors: { value: bandColors },
          uBandCount: { value: gg.bandCount },
          uStormIntensity: { value: gg.stormIntensity },
          uStormCenter: { value: new THREE.Vector2(gg.stormCenter?.[0] ?? 0.65, gg.stormCenter?.[1] ?? 0.38) },
          uTime: { value: time },
        },
      });
    }

    // ── 5. Ice Giants (Uranus, Neptune) ───────────────────────────
    if (surface?.type === 'ice_giant' || obj.id === 'uranus' || obj.id === 'neptune') {
      const baseCol = new THREE.Color(...(obj.color ?? [0.3, 0.7, 0.9]));
      const bandCol = surface?.secondaryColor
        ? new THREE.Color(...surface.secondaryColor)
        : baseCol.clone().multiplyScalar(0.85);

      return new THREE.ShaderMaterial({
        vertexShader: GLSL_PLANET_VERTEX,
        fragmentShader: GLSL_ICE_GIANT_FRAGMENT,
        uniforms: {
          uSunDirection: { value: sunDir },
          uBaseColor: { value: baseCol },
          uBandColor: { value: bandCol },
          uBandFrequency: { value: obj.id === 'neptune' ? 4.0 : 2.5 },
          uCloudIntensity: { value: obj.id === 'neptune' ? 0.8 : 0.2 },
          uTime: { value: time },
        },
      });
    }

    // ── 6. Generic Rocky / Cratered / Icy Bodies ──────────────────
    const baseColor = new THREE.Color(...(surface?.baseColor ?? obj.color ?? [0.6, 0.6, 0.6]));
    const craterColor = surface?.secondaryColor
      ? new THREE.Color(...surface.secondaryColor)
      : baseColor.clone().multiplyScalar(0.65);

    return new THREE.ShaderMaterial({
      vertexShader: GLSL_PLANET_VERTEX,
      fragmentShader: GLSL_ROCKY_FRAGMENT,
      uniforms: {
        uSunDirection: { value: sunDir },
        uBaseColor: { value: baseColor },
        uCraterColor: { value: craterColor },
        uCraterDensity: { value: surface?.craterDensity ?? 0.6 },
        uRoughness: { value: surface?.roughness ?? 0.5 },
      },
    });
  }

  /**
   * Update shader uniforms (e.g. sun direction and animation time).
   */
  static updateUniforms(material: THREE.ShaderMaterial, sunDirection: THREE.Vector3, time: number): void {
    if (material.uniforms.uSunDirection) {
      material.uniforms.uSunDirection.value.copy(sunDirection);
    }
    if (material.uniforms.uTime) {
      material.uniforms.uTime.value = time;
    }
  }
}
