/**
 * THESEUS Camera Controller
 * =========================
 * Multi-scale orbital camera for navigating from local orbit to outer solar system.
 * Supports: orbital rotation, logarithmic zoom, focus-on-object, animated transitions,
 * keyboard/mouse input, and view presets.
 */

import * as THREE from 'three';
import { auToScene, framingDistanceForRadius } from './CoordinateSystem';

/**
 * View presets, expressed as the heliocentric radius each one must contain,
 * in AU. Camera distance is then derived from the frustum — there are no
 * hardcoded camera positions here, and no per-object special cases.
 *
 *   INNER  — Mercury through Mars, with margin
 *   FULL   — all eight planets (Neptune at 30.07 AU)
 *   OUTER  — the classical Kuiper belt edge
 *   DEEP   — scattered disk / detached objects
 */
export const VIEW_PRESET_RADIUS_AU: Record<string, number> = {
  INNER_SYSTEM: 1.75,
  FULL_SYSTEM: 31.5,
  OUTER_SYSTEM: 58,
  DEEP_SYSTEM: 950,
};

/**
 * Camera framing distance for a preset, in scene units.
 * `fovDeg` must match the camera's vertical field of view.
 */
export function presetRadius(preset: keyof typeof VIEW_PRESET_RADIUS_AU, fovDeg: number): number {
  return framingDistanceForRadius(auToScene(VIEW_PRESET_RADIUS_AU[preset]), fovDeg);
}

/**
 * Camera distance used when focusing an object, expressed in multiples of
 * that object's own radius. Derived from the target, never a literal per
 * planet.
 */
export const FOCUS_DISTANCE_IN_RADII = 4.5;

export type CameraViewPreset =
  | 'FREE'
  | 'INNER_SYSTEM'
  | 'FULL_SYSTEM'
  | 'OUTER_SYSTEM'
  | 'DEEP_SYSTEM'
  | 'FOLLOW_SPACECRAFT'
  | 'FOLLOW_BODY';

interface CameraState {
  theta: number;     // Azimuthal angle (radians)
  phi: number;       // Polar angle (radians, 0 = top)
  radius: number;    // Distance from target (scene units)
  target: THREE.Vector3;
}

export interface CameraControllerConfig {
  camera: THREE.PerspectiveCamera;
  domElement: HTMLElement;
  minRadius?: number;
  maxRadius?: number;
  zoomSpeed?: number;
  rotateSpeed?: number;
  panSpeed?: number;
  dampingFactor?: number;
}

export class CameraController {
  private camera: THREE.PerspectiveCamera;
  private domElement: HTMLElement;

  private state: CameraState;
  private targetState: CameraState;

  private minRadius: number;
  private maxRadius: number;
  private zoomSpeed: number;
  private rotateSpeed: number;
  private panSpeed: number;
  private dampingFactor: number;

  // Input state
  private isDragging = false;
  private isRightDragging = false;
  private prevMouse = { x: 0, y: 0 };
  private touchDistance = 0;

  // Animation
  private isAnimating = false;
  private animationProgress = 0;
  private animationFrom: CameraState | null = null;
  private animationDuration = 1.5; // seconds

  // Current preset
  currentPreset: CameraViewPreset = 'FREE';

  // Follow target
  private followTarget: THREE.Object3D | null = null;
  private followOffset = new THREE.Vector3(0, 30, 50);

  // Bound handlers for cleanup
  private _onMouseDown: (e: MouseEvent) => void;
  private _onMouseMove: (e: MouseEvent) => void;
  private _onMouseUp: (e: MouseEvent) => void;
  private _onWheel: (e: WheelEvent) => void;
  private _onContextMenu: (e: Event) => void;
  private _onTouchStart: (e: TouchEvent) => void;
  private _onTouchMove: (e: TouchEvent) => void;
  private _onTouchEnd: (e: TouchEvent) => void;

  constructor(config: CameraControllerConfig) {
    this.camera = config.camera;
    this.domElement = config.domElement;
    this.minRadius = config.minRadius ?? 0.5;
    this.maxRadius = config.maxRadius ?? 1e8;
    this.zoomSpeed = config.zoomSpeed ?? 0.08;
    this.rotateSpeed = config.rotateSpeed ?? 0.005;
    this.panSpeed = config.panSpeed ?? 0.5;
    this.dampingFactor = config.dampingFactor ?? 0.08;

    // Open on a framing that actually contains the inner planetary orbits.
    // The previous 200-unit start put the camera 2 million km from the Sun —
    // inside every orbit in the scene — so each orbit ring filled the frame as
    // a single arc and read as a straight line.
    this.state = {
      theta: 0.3,
      phi: 1.0,
      radius: presetRadius('INNER_SYSTEM', this.camera.fov),
      target: new THREE.Vector3(0, 0, 0),
    };
    this.currentPreset = 'INNER_SYSTEM';

    this.targetState = {
      theta: this.state.theta,
      phi: this.state.phi,
      radius: this.state.radius,
      target: this.state.target.clone(),
    };

    // Bind event handlers
    this._onMouseDown = this.onMouseDown.bind(this);
    this._onMouseMove = this.onMouseMove.bind(this);
    this._onMouseUp = this.onMouseUp.bind(this);
    this._onWheel = this.onWheel.bind(this);
    this._onContextMenu = (e: Event) => e.preventDefault();
    this._onTouchStart = this.onTouchStart.bind(this);
    this._onTouchMove = this.onTouchMove.bind(this);
    this._onTouchEnd = this.onTouchEnd.bind(this);

    this.attachListeners();
  }

  // ─── Event Listeners ─────────────────────────────────────────────

  private attachListeners(): void {
    this.domElement.addEventListener('mousedown', this._onMouseDown);
    window.addEventListener('mousemove', this._onMouseMove);
    window.addEventListener('mouseup', this._onMouseUp);
    this.domElement.addEventListener('wheel', this._onWheel, { passive: false });
    this.domElement.addEventListener('contextmenu', this._onContextMenu);
    this.domElement.addEventListener('touchstart', this._onTouchStart, { passive: false });
    this.domElement.addEventListener('touchmove', this._onTouchMove, { passive: false });
    this.domElement.addEventListener('touchend', this._onTouchEnd);
  }

  private onMouseDown(e: MouseEvent): void {
    if (e.button === 0) {
      this.isDragging = true;
    } else if (e.button === 2) {
      this.isRightDragging = true;
    }
    this.prevMouse = { x: e.clientX, y: e.clientY };
    this.isAnimating = false;
  }

  private onMouseMove(e: MouseEvent): void {
    const dx = e.clientX - this.prevMouse.x;
    const dy = e.clientY - this.prevMouse.y;
    this.prevMouse = { x: e.clientX, y: e.clientY };

    if (this.isDragging) {
      // Orbit rotation
      this.targetState.theta -= dx * this.rotateSpeed;
      this.targetState.phi = Math.max(
        0.05,
        Math.min(Math.PI - 0.05, this.targetState.phi - dy * this.rotateSpeed)
      );
    } else if (this.isRightDragging) {
      // Pan
      const right = new THREE.Vector3();
      const up = new THREE.Vector3();
      this.camera.getWorldDirection(up);
      right.crossVectors(this.camera.up, up).normalize();
      up.crossVectors(right, up).normalize();

      const panScale = this.targetState.radius * this.panSpeed * 0.001;
      this.targetState.target.addScaledVector(right, -dx * panScale);
      this.targetState.target.addScaledVector(up, dy * panScale);
    }
  }

  private onMouseUp(_e: MouseEvent): void {
    this.isDragging = false;
    this.isRightDragging = false;
  }

  private onWheel(e: WheelEvent): void {
    e.preventDefault();
    this.isAnimating = false;

    // Logarithmic zoom for multi-scale navigation
    const factor = 1 + Math.sign(e.deltaY) * this.zoomSpeed;
    this.targetState.radius = Math.max(
      this.minRadius,
      Math.min(this.maxRadius, this.targetState.radius * factor)
    );
  }

  private onTouchStart(e: TouchEvent): void {
    e.preventDefault();
    this.isAnimating = false;
    if (e.touches.length === 1) {
      this.isDragging = true;
      this.prevMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    } else if (e.touches.length === 2) {
      this.isDragging = false;
      this.touchDistance = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      );
    }
  }

  private onTouchMove(e: TouchEvent): void {
    e.preventDefault();
    if (e.touches.length === 1 && this.isDragging) {
      const dx = e.touches[0].clientX - this.prevMouse.x;
      const dy = e.touches[0].clientY - this.prevMouse.y;
      this.prevMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      this.targetState.theta -= dx * this.rotateSpeed;
      this.targetState.phi = Math.max(
        0.05,
        Math.min(Math.PI - 0.05, this.targetState.phi - dy * this.rotateSpeed)
      );
    } else if (e.touches.length === 2) {
      const newDist = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      );
      const factor = this.touchDistance / Math.max(1, newDist);
      this.targetState.radius = Math.max(
        this.minRadius,
        Math.min(this.maxRadius, this.targetState.radius * factor)
      );
      this.touchDistance = newDist;
    }
  }

  private onTouchEnd(_e: TouchEvent): void {
    this.isDragging = false;
  }

  // ─── View Presets ────────────────────────────────────────────────

  setPreset(preset: CameraViewPreset, duration = 1.5): void {
    this.currentPreset = preset;

    const origin = new THREE.Vector3(0, 0, 0);
    const fov = this.camera.fov;

    switch (preset) {
      case 'INNER_SYSTEM':
        this.animateTo({ theta: 0.3, phi: 0.8, radius: presetRadius('INNER_SYSTEM', fov), target: origin }, duration);
        break;
      case 'FULL_SYSTEM':
        this.animateTo({ theta: 0.2, phi: 0.7, radius: presetRadius('FULL_SYSTEM', fov), target: origin }, duration);
        break;
      case 'OUTER_SYSTEM':
        this.animateTo({ theta: 0.5, phi: 0.6, radius: presetRadius('OUTER_SYSTEM', fov), target: origin }, duration);
        break;
      case 'DEEP_SYSTEM':
        this.animateTo({ theta: 0.1, phi: 0.5, radius: presetRadius('DEEP_SYSTEM', fov), target: origin }, duration);
        break;
      default:
        break;
    }
  }

  /**
   * Focus camera on a specific 3D position with animated transition.
   */
  focusOnPosition(position: THREE.Vector3, viewDistance?: number, duration = 1.5): void {
    const dist = Math.max(this.minRadius, viewDistance ?? Math.max(5, position.length() * 0.3));
    this.animateTo({
      theta: this.targetState.theta,
      phi: this.targetState.phi,
      radius: dist,
      target: position.clone(),
    }, duration);
    this.currentPreset = 'FREE';
  }

  /**
   * Set a follow target. Camera will track this object each frame.
   */
  /**
   * Frame an object using its own radius, so no planet needs a bespoke camera
   * distance. `objectRadiusScene` is the body's visual radius in scene units.
   */
  focusOnObject(position: THREE.Vector3, objectRadiusScene: number, duration = 1.5): void {
    this.focusOnPosition(
      position,
      Math.max(this.minRadius, objectRadiusScene * FOCUS_DISTANCE_IN_RADII),
      duration
    );
  }

  setFollowTarget(obj: THREE.Object3D | null, offset?: THREE.Vector3): void {
    this.followTarget = obj;
    if (offset) this.followOffset.copy(offset);
    this.currentPreset = obj ? 'FOLLOW_SPACECRAFT' : 'FREE';
  }

  // ─── Animation ───────────────────────────────────────────────────

  private animateTo(target: CameraState, duration: number): void {
    this.animationFrom = {
      theta: this.state.theta,
      phi: this.state.phi,
      radius: this.state.radius,
      target: this.state.target.clone(),
    };
    this.targetState = {
      ...target,
      target: target.target.clone(),
    };
    this.isAnimating = true;
    this.animationProgress = 0;
    this.animationDuration = duration;
  }

  // ─── Update (call each frame) ────────────────────────────────────

  update(dt: number): void {
    // Follow target
    if (this.followTarget) {
      const worldPos = new THREE.Vector3();
      this.followTarget.getWorldPosition(worldPos);
      this.targetState.target.copy(worldPos);
    }

    if (this.isAnimating && this.animationFrom) {
      this.animationProgress += dt / this.animationDuration;
      if (this.animationProgress >= 1.0) {
        this.animationProgress = 1.0;
        this.isAnimating = false;
      }

      // Smooth ease-in-out
      const t = this.animationProgress;
      const ease = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

      this.state.theta = this.animationFrom.theta + (this.targetState.theta - this.animationFrom.theta) * ease;
      this.state.phi = this.animationFrom.phi + (this.targetState.phi - this.animationFrom.phi) * ease;
      // Logarithmic radius interpolation for smooth zoom across scales
      this.state.radius = Math.exp(
        Math.log(this.animationFrom.radius) + (Math.log(this.targetState.radius) - Math.log(this.animationFrom.radius)) * ease
      );
      this.state.target.lerpVectors(this.animationFrom.target, this.targetState.target, ease);
    } else {
      // Smooth damping
      const d = this.dampingFactor;
      this.state.theta += (this.targetState.theta - this.state.theta) * d;
      this.state.phi += (this.targetState.phi - this.state.phi) * d;
      this.state.radius += (this.targetState.radius - this.state.radius) * d;
      this.state.target.lerp(this.targetState.target, d);
    }

    // Apply spherical coordinates to camera position
    const { theta, phi, radius, target } = this.state;
    this.camera.position.set(
      target.x + radius * Math.sin(phi) * Math.sin(theta),
      target.y + radius * Math.cos(phi),
      target.z + radius * Math.sin(phi) * Math.cos(theta)
    );
    this.camera.lookAt(target);

    // Dynamic near/far plane based on radius (prevents z-fighting)
    // Near/far track the viewing distance. The log depth buffer carries the
    // dynamic range; these bounds only need to bracket it generously.
    this.camera.near = Math.max(0.0001, radius * 0.00005);
    this.camera.far = Math.max(1e7, radius * 1000);
    this.camera.updateProjectionMatrix();
  }

  // ─── Accessors ───────────────────────────────────────────────────

  getTarget(): THREE.Vector3 {
    return this.state.target.clone();
  }

  getRadius(): number {
    return this.state.radius;
  }

  setRadius(r: number): void {
    this.targetState.radius = Math.max(this.minRadius, Math.min(this.maxRadius, r));
  }

  setTarget(t: THREE.Vector3): void {
    this.targetState.target.copy(t);
  }

  // ─── Cleanup ─────────────────────────────────────────────────────

  dispose(): void {
    this.domElement.removeEventListener('mousedown', this._onMouseDown);
    window.removeEventListener('mousemove', this._onMouseMove);
    window.removeEventListener('mouseup', this._onMouseUp);
    this.domElement.removeEventListener('wheel', this._onWheel);
    this.domElement.removeEventListener('contextmenu', this._onContextMenu);
    this.domElement.removeEventListener('touchstart', this._onTouchStart);
    this.domElement.removeEventListener('touchmove', this._onTouchMove);
    this.domElement.removeEventListener('touchend', this._onTouchEnd);
  }
}
