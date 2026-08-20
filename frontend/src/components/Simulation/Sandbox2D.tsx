import React, { useRef, useEffect, useState, useCallback } from "react";
import { 
  StateVector, 
  BodyStateHistory,
  MultiSimulationResult, 
  SpacecraftTrack, 
  MultiConjunctionEvent, 
  PhysicalCollisionEvent,
  ActiveRocket,
  ActiveExplosion
} from "../../types/mission";
import { 
  CELESTIAL_BODIES, 
  AU_KM, 
  AU_METERS,
  PLANETARY_ORBITAL_ELEMENTS,
  getPlanetStateAtTime
} from "../../data/celestialCatalog";
import { getRocketStateAtTime } from "../../lib/simulationClock";
import { 
  renderStarField, 
  drawPixelPlanet, 
  drawPixelSpacecraft,
  drawPixelDebris,
  drawPixelImpactExplosion,
} from "./PixelArtSprites";
import { 
  Maximize2, 
  Compass, 
  ZoomIn, 
  ZoomOut, 
  Globe2, 
  Sun, 
  Orbit,
  Code2,
  Crosshair,
  Target,
  AlertTriangle,
  Flame,
  Layers,
  Sparkles,
} from "lucide-react";

export type ViewContext = "SYSTEM" | "MISSION" | "LOCAL" | "OBJECT";
export type ScaleMode = "DISPLAY" | "TRUE";

interface Sandbox2DProps {
  activeRockets?: ActiveRocket[];
  activeExplosions?: ActiveExplosion[];
  simTimeSec?: number;
  stateHistory?: StateVector[];
  targetStateHistory?: StateVector[];
  bodyHistories?: BodyStateHistory[];
  multiSimResult?: MultiSimulationResult | null;
  currentFrameIdx: number;
  originBodyName?: string;
  destinationBodyName?: string;
  spacecraftPresetId?: string;
  isThrustActive?: boolean;
  selectedObjectId?: string | null;
  onSelectObject?: (id: string | null) => void;
  onSelectConjunction?: (event: MultiConjunctionEvent | null) => void;
}

export const Sandbox2D: React.FC<Sandbox2DProps> = ({
  activeRockets = [],
  activeExplosions = [],
  simTimeSec: propSimTimeSec = 0,
  stateHistory = [],
  targetStateHistory,
  bodyHistories = [],
  multiSimResult,
  currentFrameIdx,
  originBodyName = "earth",
  destinationBodyName = "mars",
  spacecraftPresetId = "isro-lvm3",
  isThrustActive = false,
  selectedObjectId = null,
  onSelectObject,
  onSelectConjunction,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Camera State (world coordinates in SI meters, zoom in px/meter)
  const [camera, setCamera] = useState<{ x: number; y: number; zoom: number }>({
    x: 0,
    y: 0,
    zoom: 1e-9,
  });

  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [viewContext, setViewContext] = useState<ViewContext>("MISSION");
  const [scaleMode, setScaleMode] = useState<ScaleMode>("DISPLAY");
  const [showDebug, setShowDebug] = useState<boolean>(false);

  // Determine current simulation time (seconds)
  let simTimeSec = propSimTimeSec || 0;
  let totalDurationSec = 7200;

  if (activeRockets.length > 0 && activeRockets[0].result.state_history.length > 0) {
    const primaryHist = activeRockets[0].result.state_history;
    const clampedIdx = Math.min(currentFrameIdx, primaryHist.length - 1);
    simTimeSec = primaryHist[clampedIdx] ? primaryHist[clampedIdx].time_seconds : 0;
    totalDurationSec = primaryHist[primaryHist.length - 1].time_seconds;
  } else if (multiSimResult && multiSimResult.objects.length > 0 && multiSimResult.objects[0].state_history.length > 0) {
    const primaryHist = multiSimResult.objects[0].state_history;
    const clampedIdx = Math.min(currentFrameIdx, primaryHist.length - 1);
    simTimeSec = primaryHist[clampedIdx] ? primaryHist[clampedIdx].time_seconds : 0;
    totalDurationSec = primaryHist[primaryHist.length - 1].time_seconds;
  } else if (stateHistory && stateHistory.length > 0) {
    const clampedIdx = Math.min(currentFrameIdx, stateHistory.length - 1);
    const currentFrame = stateHistory[clampedIdx] || stateHistory[0];
    simTimeSec = currentFrame ? currentFrame.time_seconds : 0;
    totalDurationSec = stateHistory[stateHistory.length - 1].time_seconds;
  }

  // Helper: World (SI meters) to Screen Coordinate
  const worldToScreen = useCallback((wx: number, wy: number, width: number, height: number) => {
    const sx = (wx - camera.x) * camera.zoom + width / 2;
    const sy = -(wy - camera.y) * camera.zoom + height / 2;
    return { x: sx, y: sy };
  }, [camera]);

  // Helper: Screen Coordinate to World (SI meters)
  const screenToWorld = useCallback((sx: number, sy: number, width: number, height: number) => {
    const wx = (sx - width / 2) / camera.zoom + camera.x;
    const wy = -(sy - height / 2) / camera.zoom + camera.y;
    return { x: wx, y: wy };
  }, [camera]);

  const origKey = originBodyName.toLowerCase();
  const destKey = destinationBodyName.toLowerCase();
  const origBody = CELESTIAL_BODIES[origKey] || CELESTIAL_BODIES["earth"];
  const destBody = CELESTIAL_BODIES[destKey] || CELESTIAL_BODIES["mars"];
  
  // Authoritative Frame Determination
  const isHeliocentric = true; // Solar System is centered on Sun (0, 0)
  const centralBodyKey = "sun";

  // 1. Fit Full Solar System View (Frames Sun to Neptune, ~65 AU)
  const fitSystem = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    const maxSpan = 65.0 * AU_METERS;
    const newZoom = Math.min(width, height) / maxSpan;
    setCamera({
      x: 0,
      y: 0,
      zoom: Math.max(1e-16, newZoom),
    });
    setViewContext("SYSTEM");
  }, []);

  // 2. Fit Mission View (Bounds around Sun, Earth, Mars, Jupiter, Uranus, and active rocket trajectories)
  const fitMission = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

    const includePoint = (x: number, y: number) => {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    };

    includePoint(0, 0); // Include Sun at origin

    // Include active rockets trajectories
    if (activeRockets.length > 0) {
      activeRockets.forEach((r) => {
        if (r.result.state_history) {
          r.result.state_history.forEach((pt) => includePoint(pt.position[0], pt.position[1]));
        }
      });
    } else if (stateHistory && stateHistory.length > 0) {
      stateHistory.forEach((pt) => includePoint(pt.position[0], pt.position[1]));
    } else {
      // Default to Earth and Mars bounds
      const eState = getPlanetStateAtTime("earth", simTimeSec);
      const mState = getPlanetStateAtTime("mars", simTimeSec);
      includePoint(eState.positionM[0], eState.positionM[1]);
      includePoint(mState.positionM[0], mState.positionM[1]);
    }

    if (!isFinite(minX)) {
      minX = -1e11; maxX = 1e11; minY = -1e11; maxY = 1e11;
    }

    const spanX = Math.max(1000, maxX - minX);
    const spanY = Math.max(1000, maxY - minY);
    const maxSpan = Math.max(spanX, spanY) * 1.35;

    const newZoom = Math.min(width, height) / maxSpan;
    setCamera({
      x: (minX + maxX) / 2,
      y: (minY + maxY) / 2,
      zoom: Math.max(1e-16, Math.min(1e2, newZoom)),
    });
    setViewContext("MISSION");
  }, [activeRockets, stateHistory]);

  // Auto-fit on initial render or activeRockets change
  useEffect(() => {
    fitMission();
  }, [activeRockets, multiSimResult, stateHistory]);

  // Mouse pan & zoom handlers
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging) return;
    const dx = e.clientX - dragStart.x;
    const dy = e.clientY - dragStart.y;

    setCamera((prev) => ({
      ...prev,
      x: prev.x - dx / prev.zoom,
      y: prev.y + dy / prev.zoom,
    }));

    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    if (activeRockets.length > 0 && onSelectObject) {
      let closestRocket: ActiveRocket | null = null;
      let minScreenDist = 30.0;

      activeRockets.forEach((r) => {
        const idx = Math.min(currentFrameIdx, r.result.state_history.length - 1);
        const st = r.result.state_history[idx];
        if (!st) return;
        const screenPt = worldToScreen(st.position[0], st.position[1], rect.width, rect.height);
        const dist = Math.hypot(clickX - screenPt.x, clickY - screenPt.y);
        if (dist < minScreenDist) {
          minScreenDist = dist;
          closestRocket = r;
        }
      });

      if (closestRocket) {
        onSelectObject((closestRocket as ActiveRocket).id);
      }
    }
  };

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const worldX = (mouseX - rect.width / 2) / camera.zoom + camera.x;
    const worldY = -(mouseY - rect.height / 2) / camera.zoom + camera.y;

    const zoomFactor = e.deltaY < 0 ? 1.25 : 0.8;
    const nextZoom = Math.max(1e-16, Math.min(1e4, camera.zoom * zoomFactor));

    const nextCamX = worldX - (mouseX - rect.width / 2) / nextZoom;
    const nextCamY = worldY + (mouseY - rect.height / 2) / nextZoom;

    setCamera({
      x: nextCamX,
      y: nextCamY,
      zoom: nextZoom,
    });
  };

  // Main Canvas Render Pipeline
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    // 1. PURE BLACK VOID BACKGROUND
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, width, height);

    // 2. SPARSE SEEDED WHITE PIXEL STARFIELD
    renderStarField(ctx, camera.x, camera.y, width, height);

    ctx.imageSmoothingEnabled = false;

    // -------------------------------------------------------------
    // 3. SOLAR SYSTEM LAYER: SUN + 8 PLANETS ON KEPLERIAN ORBITS
    // -------------------------------------------------------------
    const PLANET_KEYS = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"];

    // Collect active departure and target planet keys for highlighting
    const activeDepartures = new Set<string>();
    const activeTargets = new Set<string>();

    if (activeRockets.length > 0) {
      activeRockets.forEach((r) => {
        activeDepartures.add(r.origin.toLowerCase());
        activeTargets.add(r.destination.toLowerCase());
      });
    } else {
      activeDepartures.add(origKey);
      activeTargets.add(destKey);
    }

    // A. Draw Sun at (0, 0)
    const sunScreen = worldToScreen(0, 0, width, height);
    const sunRad = scaleMode === "DISPLAY" ? 14 : Math.max(4, 696340000 * camera.zoom);
    ctx.save();
    ctx.translate(sunScreen.x, sunScreen.y);
    drawPixelPlanet(ctx, "Sun", sunRad, "Sun");
    ctx.restore();

    // B. Draw All 8 Planets on Keplerian Elliptical Orbits
    PLANET_KEYS.forEach((pKey) => {
      const pData = getPlanetStateAtTime(pKey, simTimeSec);
      const bodyInfo = CELESTIAL_BODIES[pKey];

      // Draw Keplerian Elliptical Orbit
      ctx.strokeStyle = pKey === "earth" 
        ? "rgba(50, 160, 255, 0.35)" 
        : (pKey === "mars" ? "rgba(255, 100, 50, 0.35)" : "rgba(140, 140, 140, 0.25)");
      ctx.setLineDash([3, 4]);
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      pData.orbitalPathM.forEach((pt, idx) => {
        const s = worldToScreen(pt[0], pt[1], width, height);
        if (idx === 0) ctx.moveTo(s.x, s.y); else ctx.lineTo(s.x, s.y);
      });
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw Planet Sprite at current position
      const ps = worldToScreen(pData.positionM[0], pData.positionM[1], width, height);
      const pr = scaleMode === "DISPLAY" 
        ? (["jupiter", "saturn"].includes(pKey) ? 8 : (["uranus", "neptune"].includes(pKey) ? 7 : (pKey === "earth" ? 7 : 5)))
        : Math.max(3, (bodyInfo?.radius_km || 3000) * 1000.0 * camera.zoom);

      // Highlighting for active departure / target planets
      const isDeparture = activeDepartures.has(pKey);
      const isTarget = activeTargets.has(pKey);

      if (isDeparture || isTarget) {
        ctx.strokeStyle = isDeparture ? "#ffcc00" : "#00ffcc";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(ps.x, ps.y, pr + 6, 0, Math.PI * 2);
        ctx.stroke();

        ctx.font = "8px 'JetBrains Mono', monospace";
        ctx.fillStyle = isDeparture ? "#ffcc00" : "#00ffcc";
        ctx.textAlign = "center";
        ctx.fillText(isDeparture ? "DEPARTURE" : "TARGET", ps.x, ps.y - pr - 8);
      }

      ctx.save();
      ctx.translate(ps.x, ps.y);
      drawPixelPlanet(ctx, bodyInfo?.name || pKey, pr, bodyInfo?.name || pKey, true);
      ctx.restore();
    });

    // -------------------------------------------------------------
    // 4. MULTI-ROCKET TRAJECTORIES & SPACECRAFT LAYER
    // -------------------------------------------------------------
    if (activeRockets.length > 0) {
      activeRockets.forEach((r) => {
        if (!r.result.state_history || r.result.state_history.length < 2) return;

        const isSelected = selectedObjectId === r.id;
        ctx.strokeStyle = r.color || "#ff9900";
        ctx.lineWidth = isSelected ? 2.5 : 1.8;
        ctx.setLineDash([]);

        // Draw Trajectory Line
        ctx.beginPath();
        r.result.state_history.forEach((pt, idx) => {
          const s = worldToScreen(pt.position[0], pt.position[1], width, height);
          if (idx === 0) ctx.moveTo(s.x, s.y); else ctx.lineTo(s.x, s.y);
        });
        ctx.stroke();

        // Check if rocket is destroyed by Sun or collided at current simTimeSec
        const curSt = getRocketStateAtTime(r, simTimeSec);
        const distSun = curSt ? Math.hypot(curSt.position[0], curSt.position[1], curSt.position[2]) : Infinity;
        const isSunDestroyed = distSun < 2.0e9 || r.collisionState === "DESTROYED_BY_SUN";
        const isCollided = r.collisionState === "COLLIDED" && simTimeSec >= (r.collisionTimeSec || 0);

        // Deterministic Priority: Sun Destruction -> Rocket Collision
        if (isSunDestroyed) {
          const sunScreen = worldToScreen(0, 0, width, height);
          ctx.save();
          ctx.translate(sunScreen.x, sunScreen.y);
          drawPixelImpactExplosion(ctx, 0.5);

          ctx.font = "9px 'JetBrains Mono', monospace";
          ctx.fillStyle = "#ff5500";
          ctx.textAlign = "center";
          ctx.fillText("DESTROYED BY SUN", 0, -30);
          ctx.fillText(`${r.id} SOLAR IMPACT`, 0, -18);
          ctx.restore();
        } else if (isCollided && r.collisionPosM) {
          // Render Historical Impact Location Label
          const expScreen = worldToScreen(r.collisionPosM[0], r.collisionPosM[1], width, height);
          ctx.save();
          ctx.translate(expScreen.x, expScreen.y);
          ctx.font = "9px 'JetBrains Mono', monospace";
          ctx.fillStyle = "#ff3333";
          ctx.textAlign = "center";
          ctx.fillText("COLLISION IMPACT SITE", 0, -12);
          ctx.fillText(`${r.id} DESTROYED`, 0, -2);
          ctx.restore();

        } else if (curSt) {
          // Render Active Spacecraft Sprite at simTimeSec
          const s = worldToScreen(curSt.position[0], curSt.position[1], width, height);
          ctx.save();
          ctx.translate(s.x, s.y);
          drawPixelSpacecraft(ctx, r.presetId, curSt.velocity[0], curSt.velocity[1], curSt.thrust_active);

          if (isSelected) {
            ctx.strokeStyle = "#ffcc00";
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 2]);
            ctx.strokeRect(-16, -16, 32, 32);
            ctx.setLineDash([]);
          }

          ctx.font = "9px 'JetBrains Mono', monospace";
          ctx.fillStyle = r.color;
          ctx.textAlign = "center";
          ctx.fillText(r.name.toUpperCase(), 0, 18);

          ctx.font = "8px 'JetBrains Mono', monospace";
          ctx.fillStyle = "rgba(220, 220, 220, 0.85)";
          ctx.fillText(`${(curSt.speed / 1e3).toFixed(2)} km/s`, 0, 28);
          ctx.restore();
        }
      });
    }

    // Render Multi-Phase Active Retro Pixel Explosions
    if (activeExplosions.length > 0) {
      activeExplosions.forEach((exp) => {
        const elapsed = simTimeSec - exp.startTimeSec;
        if (elapsed < 0 || elapsed > exp.durationSec) return;

        const progress = Math.min(1.0, Math.max(0.0, elapsed / exp.durationSec));
        const expScreen = worldToScreen(exp.positionM[0], exp.positionM[1], width, height);

        ctx.save();
        ctx.translate(expScreen.x, expScreen.y);
        drawPixelImpactExplosion(ctx, progress);
        ctx.restore();
      });
    } else if (stateHistory && stateHistory.length > 1) {
      // Fallback single-trajectory rendering
      ctx.strokeStyle = "#ffcc00";
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      stateHistory.forEach((pt, idx) => {
        const s = worldToScreen(pt.position[0], pt.position[1], width, height);
        if (idx === 0) ctx.moveTo(s.x, s.y); else ctx.lineTo(s.x, s.y);
      });
      ctx.stroke();

      const curSt = stateHistory[Math.min(currentFrameIdx, stateHistory.length - 1)];
      if (curSt) {
        const s = worldToScreen(curSt.position[0], curSt.position[1], width, height);
        ctx.save();
        ctx.translate(s.x, s.y);
        drawPixelSpacecraft(ctx, spacecraftPresetId, curSt.velocity[0], curSt.velocity[1], isThrustActive);

        ctx.font = "9px 'JetBrains Mono', monospace";
        ctx.fillStyle = "#ffcc00";
        ctx.textAlign = "center";
        ctx.fillText("SPACECRAFT (SC-01)", 0, 18);
        ctx.font = "8px 'JetBrains Mono', monospace";
        ctx.fillStyle = "rgba(220, 220, 220, 0.85)";
        ctx.fillText(`${(curSt.speed / 1e3).toFixed(2)} km/s`, 0, 28);
        ctx.restore();
      }
    }

  }, [
    camera,
    viewContext,
    scaleMode,
    currentFrameIdx,
    simTimeSec,
    activeRockets,
    multiSimResult,
    stateHistory,
    selectedObjectId,
    originBodyName,
    destinationBodyName,
    spacecraftPresetId,
    isThrustActive,
    worldToScreen,
  ]);

  return (
    <div className="relative w-full h-full bg-black overflow-hidden select-none font-mono">
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-crosshair block"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onClick={handleClick}
        onWheel={handleWheel}
      />

      {/* Top Left: Viewport Controls & Camera Framing */}
      <div className="absolute top-3 left-3 flex flex-wrap gap-1.5 z-10">
        <button
          onClick={fitSystem}
          className={`px-2.5 py-1 text-xs border uppercase tracking-wider transition-colors ${
            viewContext === "SYSTEM"
              ? "bg-amber-500/20 text-amber-400 border-amber-500/60"
              : "bg-black/80 text-neutral-400 border-neutral-800 hover:border-neutral-600 hover:text-white"
          }`}
        >
          <Globe2 className="w-3 h-3 inline mr-1" />
          Full System
        </button>

        <button
          onClick={fitMission}
          className={`px-2.5 py-1 text-xs border uppercase tracking-wider transition-colors ${
            viewContext === "MISSION"
              ? "bg-amber-500/20 text-amber-400 border-amber-500/60"
              : "bg-black/80 text-neutral-400 border-neutral-800 hover:border-neutral-600 hover:text-white"
          }`}
        >
          <Orbit className="w-3 h-3 inline mr-1" />
          Mission View
        </button>

        <button
          onClick={() => setScaleMode((m) => (m === "DISPLAY" ? "TRUE" : "DISPLAY"))}
          className={`px-2.5 py-1 text-xs border uppercase tracking-wider transition-colors ${
            scaleMode === "TRUE"
              ? "bg-neutral-800 text-white border-neutral-600"
              : "bg-black/80 text-neutral-400 border-neutral-800 hover:text-white"
          }`}
        >
          Scale: {scaleMode}
        </button>
      </div>

      {/* Top Right: Zoom & Framing Utilities */}
      <div className="absolute top-3 right-3 flex items-center gap-1.5 z-10">
        <button
          onClick={() => setCamera((c) => ({ ...c, zoom: c.zoom * 1.35 }))}
          className="p-1.5 bg-black/80 text-neutral-400 border border-neutral-800 hover:border-neutral-600 hover:text-white"
          title="Zoom In"
        >
          <ZoomIn className="w-3.5 h-3.5" />
        </button>

        <button
          onClick={() => setCamera((c) => ({ ...c, zoom: c.zoom * 0.75 }))}
          className="p-1.5 bg-black/80 text-neutral-400 border border-neutral-800 hover:border-neutral-600 hover:text-white"
          title="Zoom Out"
        >
          <ZoomOut className="w-3.5 h-3.5" />
        </button>

        <button
          onClick={fitMission}
          className="p-1.5 bg-black/80 text-neutral-400 border border-neutral-800 hover:border-neutral-600 hover:text-white"
          title="Reset Camera Framing"
        >
          <Maximize2 className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Active Rockets Status HUD Overlay */}
      {activeRockets.length > 0 && (
        <div className="absolute top-14 left-3 w-72 bg-black/90 border border-neutral-800 p-2.5 text-[11px] text-neutral-300 font-mono z-10 shadow-xl space-y-1.5">
          <div className="flex justify-between items-center border-b border-neutral-800 pb-1 font-bold text-white">
            <span>ACTIVE ROCKET FLEET ({activeRockets.length})</span>
            <span className="text-[9px] bg-amber-950 text-amber-400 px-1 border border-amber-800">SIMULATING</span>
          </div>

          <div className="space-y-1 max-h-36 overflow-y-auto pr-1">
            {activeRockets.map((r) => (
              <div key={r.id} className="flex items-center justify-between bg-neutral-950 p-1.5 border border-neutral-800 text-[10px]">
                <div className="flex items-center space-x-1.5">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: r.color }} />
                  <span className="font-bold text-white">{r.name}</span>
                </div>
                <span className={`font-bold ${
                  r.collisionState === "COLLIDED"
                    ? "text-red-400 animate-pulse"
                    : (r.collisionState === "TARGETING" ? "text-amber-400" : "text-emerald-400")
                }`}>
                  {r.collisionState === "COLLIDED" ? "COLLIDED" : (r.collisionState === "TARGETING" ? "TARGETING" : "ACTIVE")}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Bottom Left: Simulation Timestamp & Object Counter */}
      <div className="absolute bottom-3 left-3 bg-black/85 border border-neutral-800 px-3 py-2 text-xs text-neutral-300 z-10">
        <div className="flex items-center gap-2">
          <span className="text-amber-400 font-semibold">T+ {formatTimeSeconds(simTimeSec)}</span>
          <span className="text-neutral-500">|</span>
          <span>{activeRockets.length > 0 ? `${activeRockets.length} ROCKETS ACTIVE` : "1 ROCKET TRACKED"}</span>
        </div>
      </div>
    </div>
  );
};

function formatTimeSeconds(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return `${h.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}
