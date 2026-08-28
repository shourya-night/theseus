/**
 * THESEUS Shader Preview Harness (headless)
 * =========================================
 * Renders the procedural body shaders to a PNG in a headless browser, so
 * shader work can be seen and judged without running the full app.
 *
 * Compilation success proves almost nothing about a shader — every problem
 * fixed with this harness (a corona detached from the photosphere, a blown-out
 * photosphere with no visible granulation, gold speckle aliasing around
 * Earth's terminator, ring bands drawn as parallel stripes) type-checked
 * perfectly beforehand.
 *
 *   npm i -D playwright esbuild        # not app dependencies
 *   npx esbuild scripts/shaderPreview.ts --bundle --outfile=/tmp/preview.js --format=iife
 *   node scripts/renderShaderPreview.mjs
 *
 * Writes /tmp/shaders.png and prints any console or shader-compile errors.
 */
import * as THREE from 'three';
import { GLSL_SUN_VERTEX, GLSL_SUN_FRAGMENT, GLSL_CORONA_FRAGMENT,
         GLSL_ATMOSPHERE_VERTEX, GLSL_PLANET_VERTEX, GLSL_EARTH_FRAGMENT,
         GLSL_RING_VERTEX, GLSL_RING_FRAGMENT } from '../src/renderer/ShaderLib';

const W = 420, H = 420;
declare const document: any, window: any;

function mk(id: string, build: (scene: THREE.Scene, cam: THREE.PerspectiveCamera) => void) {
  const canvas = document.getElementById(id);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setSize(W, H);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);
  const cam = new THREE.PerspectiveCamera(50, 1, 0.01, 1e7);
  build(scene, cam);
  renderer.render(scene, cam);
  const info = (renderer.getContext() as any).getShaderPrecisionFormat ? 'ok' : 'ok';
  window.__errors = window.__errors || [];
  return info;
}

// ── 1. Sun: photosphere + corona ──
mk('sun', (scene, cam) => {
  const R = 69.63;
  const surf = new THREE.Mesh(new THREE.SphereGeometry(R, 96, 96),
    new THREE.ShaderMaterial({ vertexShader: GLSL_SUN_VERTEX, fragmentShader: GLSL_SUN_FRAGMENT,
      uniforms: { uTime: { value: 120 }, uIntensity: { value: 1.35 } } }));
  scene.add(surf);
  const CR = R * 2.6;
  scene.add(new THREE.Mesh(new THREE.SphereGeometry(CR, 64, 64),
    new THREE.ShaderMaterial({ vertexShader: GLSL_ATMOSPHERE_VERTEX, fragmentShader: GLSL_CORONA_FRAGMENT,
      uniforms: { uTime: { value: 120 }, uSunColor: { value: new THREE.Color(1.0, 0.78, 0.42) },
        uSunCenter: { value: new THREE.Vector3(0,0,0) },
        uPhotosphereRadius: { value: R }, uCoronaRadius: { value: CR } },
      transparent: true, blending: THREE.AdditiveBlending, side: THREE.FrontSide, depthWrite: false })));
  cam.position.set(0, 60, 260); cam.lookAt(0,0,0);
});

// ── 2. Earth ──
mk('earth', (scene, cam) => {
  const R = 0.6378;
  scene.add(new THREE.Mesh(new THREE.SphereGeometry(R, 96, 96),
    new THREE.ShaderMaterial({ vertexShader: GLSL_PLANET_VERTEX, fragmentShader: GLSL_EARTH_FRAGMENT,
      uniforms: { uSunDirection: { value: new THREE.Vector3(0.6, 0.25, 0.76).normalize() },
                  uTime: { value: 0 }, uCloudCover: { value: 0.5 } } })));
  cam.position.set(0, 0.5, 1.9); cam.lookAt(0,0,0);
});

// ── 2b. Earth from a different camera, SAME sun direction ──
// The terminator must stay in the same place relative to the Sun. If it
// rotates with the camera, the shader is using a view-space normal.
mk('earth2', (scene, cam) => {
  const R = 0.6378;
  scene.add(new THREE.Mesh(new THREE.SphereGeometry(R, 96, 96),
    new THREE.ShaderMaterial({ vertexShader: GLSL_PLANET_VERTEX, fragmentShader: GLSL_EARTH_FRAGMENT,
      uniforms: { uSunDirection: { value: new THREE.Vector3(0.6, 0.25, 0.76).normalize() },
                  uTime: { value: 0 }, uCloudCover: { value: 0.5 } } })));
  cam.position.set(1.75, 0.5, 0.55); cam.lookAt(0,0,0);
});

// ── 3. Saturn + rings ──
mk('rings', (scene, cam) => {
  const R = 6.0268;
  const tilt = 26.73 * Math.PI / 180;
  const g = new THREE.Group(); g.rotation.z = tilt; scene.add(g);
  g.add(new THREE.Mesh(new THREE.SphereGeometry(R, 64, 64),
    new THREE.MeshBasicMaterial({ color: 0xd9c9a0 })));
  const inner = R * 1.24, outer = R * 2.27;
  const geo = new THREE.RingGeometry(inner, outer, 256, 12);
  geo.rotateX(-Math.PI / 2);
  g.add(new THREE.Mesh(geo, new THREE.ShaderMaterial({
    vertexShader: GLSL_RING_VERTEX, fragmentShader: GLSL_RING_FRAGMENT,
    uniforms: { uSunDirection: { value: new THREE.Vector3(0.5, 0.6, 0.6).normalize() },
      uPlanetPosition: { value: new THREE.Vector3(0,0,0) }, uPlanetRadius: { value: R },
      uInnerRadius: { value: inner }, uOuterRadius: { value: outer },
      uRingColor: { value: new THREE.Color(0.82, 0.76, 0.64) }, uRingDensityScale: { value: 1.0 } },
    transparent: true, side: THREE.DoubleSide, depthWrite: false })));
  cam.position.set(0, 12, 34); cam.lookAt(0,0,0);
});

window.__done = true;
