import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { StateVector, CelestialBodyInfo } from "../../types/mission";
import { CELESTIAL_BODIES } from "../../data/celestialCatalog";
import { 
  Camera, 
  Layers, 
  Maximize2, 
  Minimize2, 
  Eye, 
  Compass, 
  Crosshair, 
  Sparkles,
  RefreshCw
} from "lucide-react";

interface Viewport3DProps {
  stateHistory: StateVector[];
  targetStateHistory?: StateVector[];
  currentFrameIdx: number;
  originBodyName?: string;
  destinationBodyName?: string;
  isThrustActive?: boolean;
}

type CameraMode = "free" | "follow_spacecraft" | "follow_target" | "origin" | "destination" | "system";
type ScaleMode = "enhanced" | "physical";

export const Viewport3D: React.FC<Viewport3DProps> = ({
  stateHistory,
  targetStateHistory,
  currentFrameIdx,
  originBodyName = "earth",
  destinationBodyName = "mars",
  isThrustActive = false,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [cameraMode, setCameraMode] = useState<CameraMode>("free");
  const [scaleMode, setScaleMode] = useState<ScaleMode>("enhanced");
  const [showGrids, setShowGrids] = useState<boolean>(true);
  const [showAxes, setShowAxes] = useState<boolean>(true);

  // Three.js Scene References
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const spacecraftMeshRef = useRef<THREE.Group | null>(null);
  const targetMeshRef = useRef<THREE.Mesh | null>(null);
  const plumeMeshRef = useRef<THREE.Mesh | null>(null);
  const trajectoryLineRef = useRef<THREE.Line | null>(null);
  const targetLineRef = useRef<THREE.Line | null>(null);
  const planetMeshesRef = useRef<Record<string, THREE.Mesh>>({});

  // Mouse interaction state
  const isDraggingRef = useRef(false);
  const prevMousePosRef = useRef({ x: 0, y: 0 });
  const cameraAnglesRef = useRef({ theta: 0.6, phi: 0.8, radius: 120 });

  const currentFrame = stateHistory[currentFrameIdx] || stateHistory[0];

  // Scale converter from meters to 3D scene units
  const getScaleFactor = () => {
    // 1 scene unit = 10,000 km for enhanced mode
    return scaleMode === "enhanced" ? 1 / 1e7 : 1 / 1e8;
  };

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // 1. Initialize Scene, Camera & Renderer
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    scene.background = new THREE.Color(0x03070d);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 50000);
    cameraRef.current = camera;
    camera.position.set(0, 80, 120);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    rendererRef.current = renderer;
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    // 2. Lighting
    const ambientLight = new THREE.AmbientLight(0x223344, 1.2);
    scene.add(ambientLight);

    const sunLight = new THREE.PointLight(0xffffff, 2.5, 0, 0);
    sunLight.position.set(0, 0, 0);
    scene.add(sunLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
    dirLight.position.set(50, 100, 50);
    scene.add(dirLight);

    // 3. Starfield Particles (Instanced)
    const starCount = 1500;
    const starGeo = new THREE.BufferGeometry();
    const starPos = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount * 3; i += 3) {
      const r = 2000 + Math.random() * 8000;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(Math.random() * 2 - 1);
      starPos[i] = r * Math.sin(phi) * Math.cos(theta);
      starPos[i + 1] = r * Math.sin(phi) * Math.sin(theta);
      starPos[i + 2] = r * Math.cos(phi);
    }
    starGeo.setAttribute("position", new THREE.BufferAttribute(starPos, 3));
    const starMat = new THREE.PointsMaterial({ color: 0x738ca6, size: 2, sizeAttenuation: false });
    const stars = new THREE.Points(starGeo, starMat);
    scene.add(stars);

    // 4. Reference Grid & Coordinate Axes
    const grid = new THREE.GridHelper(200, 40, 0x162238, 0x0c1424);
    grid.position.y = -0.1;
    scene.add(grid);

    const axes = new THREE.AxesHelper(30);
    scene.add(axes);

    // 5. Celestial Bodies (Sun, Earth, Mars, Moon, Jupiter, Saturn)
    const planetMeshes: Record<string, THREE.Mesh> = {};
    Object.entries(CELESTIAL_BODIES).forEach(([key, body]) => {
      let rVis = Math.max(1.5, Math.log10(body.radius_km) * 1.2);
      if (key === "sun") rVis = 8.0;
      if (key === "earth") rVis = 3.5;
      if (key === "mars") rVis = 2.4;
      if (key === "moon") rVis = 1.0;

      const geo = new THREE.SphereGeometry(rVis, 32, 32);
      let mat: THREE.Material;

      if (key === "sun") {
        mat = new THREE.MeshBasicMaterial({ color: new THREE.Color(body.color) });
      } else {
        mat = new THREE.MeshStandardMaterial({
          color: new THREE.Color(body.color),
          roughness: 0.8,
          metalness: 0.1,
        });
      }

      const mesh = new THREE.Mesh(geo, mat);
      mesh.name = body.name;
      scene.add(mesh);
      planetMeshes[key] = mesh;

      // Saturn Rings
      if (key === "saturn") {
        const ringGeo = new THREE.RingGeometry(rVis * 1.4, rVis * 2.3, 64);
        const ringMat = new THREE.MeshBasicMaterial({
          color: 0xd4a373,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.6,
        });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.rotation.x = Math.PI / 2.2;
        mesh.add(ring);
      }
    });
    planetMeshesRef.current = planetMeshes;

    // 6. Spacecraft 3D Entity with Dynamic Thrust Plume
    const scGroup = new THREE.Group();
    const scGeo = new THREE.ConeGeometry(0.8, 2.0, 8);
    scGeo.rotateX(Math.PI / 2);
    const scMat = new THREE.MeshStandardMaterial({ color: 0x00f0ff, roughness: 0.3, metalness: 0.8 });
    const scMesh = new THREE.Mesh(scGeo, scMat);
    scGroup.add(scMesh);

    // Thrust Plume
    const plumeGeo = new THREE.ConeGeometry(0.5, 2.5, 8);
    plumeGeo.rotateX(-Math.PI / 2);
    const plumeMat = new THREE.MeshBasicMaterial({
      color: 0xff6600,
      transparent: true,
      opacity: 0.85,
    });
    const plume = new THREE.Mesh(plumeGeo, plumeMat);
    plume.position.z = -1.8;
    plume.visible = false;
    scGroup.add(plume);
    plumeMeshRef.current = plume;

    scene.add(scGroup);
    spacecraftMeshRef.current = scGroup;

    // Target Spacecraft (for rendezvous)
    const tgtGeo = new THREE.BoxGeometry(1.2, 1.2, 1.2);
    const tgtMat = new THREE.MeshStandardMaterial({ color: 0xffb000, wireframe: false });
    const tgtMesh = new THREE.Mesh(tgtGeo, tgtMat);
    tgtMesh.visible = false;
    scene.add(tgtMesh);
    targetMeshRef.current = tgtMesh;

    // 7. Mouse Orbit Controls Event Handlers
    const onMouseDown = (e: MouseEvent) => {
      isDraggingRef.current = true;
      prevMousePosRef.current = { x: e.clientX, y: e.clientY };
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!isDraggingRef.current) return;
      const dx = e.clientX - prevMousePosRef.current.x;
      const dy = e.clientY - prevMousePosRef.current.y;
      prevMousePosRef.current = { x: e.clientX, y: e.clientY };

      cameraAnglesRef.current.theta -= dx * 0.008;
      cameraAnglesRef.current.phi = Math.max(0.1, Math.min(Math.PI - 0.1, cameraAnglesRef.current.phi - dy * 0.008));
    };

    const onMouseUp = () => {
      isDraggingRef.current = false;
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      cameraAnglesRef.current.radius = Math.max(10, Math.min(1000, cameraAnglesRef.current.radius + e.deltaY * 0.15));
    };

    container.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    container.addEventListener("wheel", onWheel, { passive: false });

    // 8. Render Animation Loop
    let animId: number;
    const animate = () => {
      animId = requestAnimationFrame(animate);

      // Camera position update based on spherical angles
      const { theta, phi, radius } = cameraAnglesRef.current;
      const targetPos = new THREE.Vector3(0, 0, 0);

      if (cameraMode === "follow_spacecraft" && spacecraftMeshRef.current) {
        targetPos.copy(spacecraftMeshRef.current.position);
      } else if (cameraMode === "follow_target" && targetMeshRef.current) {
        targetPos.copy(targetMeshRef.current.position);
      }

      camera.position.x = targetPos.x + radius * Math.sin(phi) * Math.sin(theta);
      camera.position.y = targetPos.y + radius * Math.cos(phi);
      camera.position.z = targetPos.z + radius * Math.sin(phi) * Math.cos(theta);
      camera.lookAt(targetPos);

      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!containerRef.current) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(animId);
      container.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      container.removeEventListener("wheel", onWheel);
      window.removeEventListener("resize", handleResize);
      renderer.dispose();
    };
  }, []);

  // Update Trajectory Lines & Spacecraft Positions from State History
  useEffect(() => {
    if (!sceneRef.current || !stateHistory || stateHistory.length === 0) return;
    const scene = sceneRef.current;
    const scale = getScaleFactor();

    // 1. Build Spacecraft Trajectory Line
    if (trajectoryLineRef.current) {
      scene.remove(trajectoryLineRef.current);
    }
    const points: THREE.Vector3[] = stateHistory.map(
      (s) => new THREE.Vector3(s.position[0] * scale, s.position[2] * scale, s.position[1] * scale)
    );
    const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
    const lineMat = new THREE.LineBasicMaterial({
      color: 0x00f0ff,
      linewidth: 2,
      transparent: true,
      opacity: 0.85,
    });
    const line = new THREE.Line(lineGeo, lineMat);
    scene.add(line);
    trajectoryLineRef.current = line;

    // 2. Build Target Trajectory Line (if rendezvous)
    if (targetStateHistory && targetStateHistory.length > 0) {
      if (targetLineRef.current) scene.remove(targetLineRef.current);
      const tgtPoints = targetStateHistory.map(
        (s) => new THREE.Vector3(s.position[0] * scale, s.position[2] * scale, s.position[1] * scale)
      );
      const tgtLineGeo = new THREE.BufferGeometry().setFromPoints(tgtPoints);
      const tgtLineMat = new THREE.LineBasicMaterial({ color: 0xffb000, linewidth: 2 });
      const tgtLine = new THREE.Line(tgtLineGeo, tgtLineMat);
      scene.add(tgtLine);
      targetLineRef.current = tgtLine;
    }
  }, [stateHistory, targetStateHistory, scaleMode]);

  // Update Active Frame Spacecraft Position, Velocity Alignment, and Plume
  useEffect(() => {
    if (!currentFrame || !spacecraftMeshRef.current) return;
    const scale = getScaleFactor();
    const sc = spacecraftMeshRef.current;

    const pos = new THREE.Vector3(
      currentFrame.position[0] * scale,
      currentFrame.position[2] * scale,
      currentFrame.position[1] * scale
    );
    sc.position.copy(pos);

    // Orient along velocity vector
    const vel = new THREE.Vector3(
      currentFrame.velocity[0],
      currentFrame.velocity[2],
      currentFrame.velocity[1]
    );
    if (vel.lengthSq() > 0.001) {
      vel.normalize();
      const targetQuat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), vel);
      sc.quaternion.slerp(targetQuat, 0.4);
    }

    // Toggle plume animation
    if (plumeMeshRef.current) {
      const active = isThrustActive || currentFrame.thrust_active;
      plumeMeshRef.current.visible = Boolean(active);
      if (active) {
        plumeMeshRef.current.scale.set(
          0.8 + Math.random() * 0.4,
          0.8 + Math.random() * 0.4,
          1.0 + Math.random() * 0.5
        );
      }
    }

    // Target Spacecraft position
    if (targetMeshRef.current && targetStateHistory && targetStateHistory[currentFrameIdx]) {
      const tgtFrame = targetStateHistory[currentFrameIdx];
      targetMeshRef.current.visible = true;
      targetMeshRef.current.position.set(
        tgtFrame.position[0] * scale,
        tgtFrame.position[2] * scale,
        tgtFrame.position[1] * scale
      );
    }
  }, [currentFrame, currentFrameIdx, isThrustActive, targetStateHistory, scaleMode]);

  return (
    <div className="relative w-full h-full bg-[#03070d] overflow-hidden select-none">
      
      {/* Three.js Canvas Container */}
      <div ref={containerRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Top Banner: Scale & Ephemeris Mode */}
      <div className="absolute top-3 left-4 flex flex-col space-y-1 text-xs font-mono z-10 pointer-events-none">
        <div className="flex items-center space-x-2 bg-[#070d18]/90 border border-[#162238] px-3 py-1.5 rounded backdrop-blur">
          <span className="w-2 h-2 rounded-full bg-[#00f0ff] animate-ping" />
          <span className="text-[#00f0ff] font-bold">
            {scaleMode === "enhanced"
              ? "DISPLAY SCALE ENHANCED"
              : "TRUE PHYSICAL SCALE (1:1 RATIO)"}
          </span>
          <span className="text-[#455a73]">|</span>
          <span className="text-[#738ca6]">
            TRAJECTORIES: <span className="text-[#00ff66]">EXACT NUMERICAL STATE</span>
          </span>
        </div>
        <div className="text-[10px] text-[#455a73] px-1">
          {scaleMode === "enhanced"
            ? "Body radii visually scaled for orbital overview. Orbital lines and vectors are exact."
            : "Metric scale: 1 unit = 100,000 km."}
        </div>
      </div>

      {/* Floating Viewport Controls Overlay */}
      <div className="absolute top-3 right-4 flex items-center space-x-2 z-10">
        
        {/* Camera Selector */}
        <div className="flex items-center space-x-1 bg-[#070d18]/90 border border-[#162238] p-1 rounded backdrop-blur text-xs font-mono">
          <Camera className="w-3.5 h-3.5 text-[#00f0ff] ml-1" />
          {(["free", "follow_spacecraft", "system"] as CameraMode[]).map((mode) => (
            <button
              key={mode}
              onClick={() => setCameraMode(mode)}
              className={`px-2 py-1 rounded transition-colors text-[10px] ${
                cameraMode === mode
                  ? "bg-[#00f0ff]/20 text-[#00f0ff] font-bold"
                  : "text-[#738ca6] hover:text-[#e2ecf8]"
              }`}
            >
              {mode.replace("_", " ").toUpperCase()}
            </button>
          ))}
        </div>

        {/* Scale Toggle */}
        <button
          onClick={() => setScaleMode(scaleMode === "enhanced" ? "physical" : "enhanced")}
          className="bg-[#070d18]/90 border border-[#162238] hover:border-[#00f0ff] px-2.5 py-1.5 rounded text-xs font-mono text-[#00f0ff] transition-all flex items-center space-x-1.5"
        >
          <Layers className="w-3.5 h-3.5" />
          <span className="text-[10px] font-bold">{scaleMode === "enhanced" ? "ENHANCED" : "PHYSICAL"}</span>
        </button>
      </div>

      {/* Bottom Center: Coordinates & Velocity Readout */}
      {currentFrame && (
        <div className="absolute bottom-4 left-4 bg-[#070d18]/90 border border-[#162238] p-2.5 rounded backdrop-blur text-[11px] font-mono text-[#738ca6] space-y-1 z-10 pointer-events-none">
          <div className="flex justify-between space-x-4 text-[#e2ecf8]">
            <span className="text-[#00f0ff] font-bold">STATE VECTOR (ECI):</span>
            <span className="text-[#00ff66]">|v| = {(currentFrame.speed / 1000).toFixed(3)} km/s</span>
          </div>
          <div className="grid grid-cols-3 gap-2 text-[10px]">
            <div>X: {(currentFrame.position[0] / 1e6).toFixed(3)}M m</div>
            <div>Y: {(currentFrame.position[1] / 1e6).toFixed(3)}M m</div>
            <div>Z: {(currentFrame.position[2] / 1e6).toFixed(3)}M m</div>
          </div>
        </div>
      )}
    </div>
  );
};
