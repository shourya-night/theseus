/**
 * THESEUS Scene Manager
 * =====================
 * Core Three.js rendering infrastructure. Manages the WebGL renderer,
 * scene graph, lighting, post-processing, and render loop.
 *
 * This replaces the Canvas 2D pixel-art renderer with a full 3D scene.
 */

import * as THREE from 'three';

export interface SceneManagerConfig {
  container: HTMLElement;
  antialias?: boolean;
  pixelRatio?: number;
  shadowMap?: boolean;
}

export interface RenderStats {
  fps: number;
  drawCalls: number;
  triangles: number;
  frameTime: number;
}

export class SceneManager {
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;
  readonly sunLight: THREE.DirectionalLight;
  readonly ambientLight: THREE.AmbientLight;
  readonly sunPointLight: THREE.PointLight;

  private container: HTMLElement;
  private animationId: number = 0;
  private clock: THREE.Clock;
  private resizeObserver: ResizeObserver;
  private renderCallbacks: Array<(dt: number, elapsed: number) => void> = [];
  private _disposed = false;

  // Performance stats
  private frameCount = 0;
  private lastFpsTime = 0;
  private lastFrameTime = 0;
  stats: RenderStats = { fps: 0, drawCalls: 0, triangles: 0, frameTime: 0 };

  constructor(config: SceneManagerConfig) {
    this.container = config.container;
    this.clock = new THREE.Clock();

    // ─── Scene ───────────────────────────────────────────────────
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x000000); // Pure black void

    // ─── Camera ──────────────────────────────────────────────────
    const aspect = config.container.clientWidth / Math.max(1, config.container.clientHeight);
    this.camera = new THREE.PerspectiveCamera(50, aspect, 0.001, 1e12);
    this.camera.position.set(0, 50, 150);
    this.camera.lookAt(0, 0, 0);

    // Use logarithmic depth buffer for extreme scale range
    // (planets at 1e-2 to Oort cloud at 1e12)

    // ─── Renderer ────────────────────────────────────────────────
    this.renderer = new THREE.WebGLRenderer({
      antialias: config.antialias !== false,
      alpha: false,
      powerPreference: 'high-performance',
      logarithmicDepthBuffer: true,
    });
    this.renderer.setSize(config.container.clientWidth, config.container.clientHeight);
    this.renderer.setPixelRatio(Math.min(config.pixelRatio ?? window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;

    if (config.shadowMap) {
      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    }

    config.container.appendChild(this.renderer.domElement);
    this.renderer.domElement.style.display = 'block';

    // ─── Lighting ────────────────────────────────────────────────
    // Ambient: very dim, simulates indirect scattered light
    this.ambientLight = new THREE.AmbientLight(0x0a0a14, 0.15);
    this.scene.add(this.ambientLight);

    // Sun point light at origin
    this.sunPointLight = new THREE.PointLight(0xffffff, 3.0, 0, 0);
    this.sunPointLight.position.set(0, 0, 0);
    this.scene.add(this.sunPointLight);

    // Directional light for shadow casting (optional)
    this.sunLight = new THREE.DirectionalLight(0xffffff, 1.5);
    this.sunLight.position.set(0, 0, 0);
    if (config.shadowMap) {
      this.sunLight.castShadow = true;
      this.sunLight.shadow.mapSize.set(2048, 2048);
      this.sunLight.shadow.camera.near = 0.5;
      this.sunLight.shadow.camera.far = 1e6;
    }
    this.scene.add(this.sunLight);

    // ─── Resize Observer ─────────────────────────────────────────
    this.resizeObserver = new ResizeObserver(() => this.handleResize());
    this.resizeObserver.observe(config.container);
  }

  // ─── Render Loop ─────────────────────────────────────────────────

  start(): void {
    if (this._disposed) return;
    this.clock.start();
    this.lastFpsTime = performance.now();
    this.animate();
  }

  stop(): void {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = 0;
    }
  }

  private animate = (): void => {
    if (this._disposed) return;
    this.animationId = requestAnimationFrame(this.animate);

    const dt = this.clock.getDelta();
    const elapsed = this.clock.getElapsedTime();
    const frameStart = performance.now();

    // Execute all registered render callbacks
    for (const cb of this.renderCallbacks) {
      cb(dt, elapsed);
    }

    // Render
    this.renderer.render(this.scene, this.camera);

    // Stats
    this.lastFrameTime = performance.now() - frameStart;
    this.frameCount++;
    const now = performance.now();
    if (now - this.lastFpsTime >= 1000) {
      this.stats = {
        fps: Math.round(this.frameCount * 1000 / (now - this.lastFpsTime)),
        drawCalls: this.renderer.info.render.calls,
        triangles: this.renderer.info.render.triangles,
        frameTime: this.lastFrameTime,
      };
      this.frameCount = 0;
      this.lastFpsTime = now;
    }
  };

  // ─── Callback Registration ───────────────────────────────────────

  onRender(callback: (dt: number, elapsed: number) => void): () => void {
    this.renderCallbacks.push(callback);
    return () => {
      const idx = this.renderCallbacks.indexOf(callback);
      if (idx >= 0) this.renderCallbacks.splice(idx, 1);
    };
  }

  // ─── Scene Graph Helpers ─────────────────────────────────────────

  add(...objects: THREE.Object3D[]): void {
    objects.forEach(o => this.scene.add(o));
  }

  remove(...objects: THREE.Object3D[]): void {
    objects.forEach(o => this.scene.remove(o));
  }

  // ─── Resize ──────────────────────────────────────────────────────

  private handleResize(): void {
    if (this._disposed) return;
    const w = this.container.clientWidth;
    const h = Math.max(1, this.container.clientHeight);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  // ─── Coordinate Helpers ──────────────────────────────────────────
  //
  // Deliberately absent. Unit conversion and axis remapping live in
  // renderer/CoordinateSystem.ts and nowhere else — a second copy here is how
  // the scene ended up with several disagreeing conversions. Import
  // SCENE_SCALE / kmToScene / engineToThreePos from there instead.

  // ─── Utility ─────────────────────────────────────────────────────

  getCanvas(): HTMLCanvasElement {
    return this.renderer.domElement;
  }

  /**
   * Update the sun directional light to point from origin toward a target.
   * Used to ensure shadows are cast correctly relative to the sun.
   */
  updateSunLightTarget(targetPosition: THREE.Vector3): void {
    this.sunLight.target.position.copy(targetPosition);
    this.sunLight.target.updateMatrixWorld();
  }

  // ─── Disposal ────────────────────────────────────────────────────

  dispose(): void {
    this._disposed = true;
    this.stop();
    this.resizeObserver.disconnect();
    this.renderCallbacks.length = 0;

    // Dispose all scene objects
    this.scene.traverse((obj) => {
      if (obj instanceof THREE.Mesh) {
        obj.geometry?.dispose();
        if (Array.isArray(obj.material)) {
          obj.material.forEach(m => m.dispose());
        } else {
          obj.material?.dispose();
        }
      }
    });

    this.renderer.dispose();
    if (this.renderer.domElement.parentNode) {
      this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
    }
  }
}
