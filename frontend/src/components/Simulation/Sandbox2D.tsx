import React, { useRef, useEffect, useState, useCallback } from "react";
import { 
  StateVector, 
  MultiSimulationResult, 
  SpacecraftTrack, 
  MultiConjunctionEvent, 
  PhysicalCollisionEvent 
} from "../../types/mission";
import { 
  CELESTIAL_BODIES, 
  AU_KM, 
  AU_METERS, 
  getPlanetStateAtTime, 
  PLANETARY_ORBITAL_ELEMENTS 
} from "../../data/celestialCatalog";
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
  stateHistory?: StateVector[];
  targetStateHistory?: StateVector[];
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
  stateHistory = [],
  targetStateHistory,
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
  let simTimeSec = 0;
  let totalDurationSec = 7200;

  if (multiSimResult && multiSimResult.objects.length > 0 && multiSimResult.objects[0].state_history.length > 0) {
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
  const isHeliocentric = (origBody.parent === "Sun" && destBody.parent === "Sun") || origKey === "sun";

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

  // 2. Fit Mission View (Tight bounds around active spacecraft trajectories)
  const fitMission = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

    if (multiSimResult && multiSimResult.objects.length > 0) {
      multiSimResult.objects.forEach((obj) => {
        obj.state_history.forEach((pt) => {
          if (pt.position[0] < minX) minX = pt.position[0];
          if (pt.position[0] > maxX) maxX = pt.position[0];
          if (pt.position[1] < minY) minY = pt.position[1];
          if (pt.position[1] > maxY) maxY = pt.position[1];
        });
      });
    } else if (stateHistory && stateHistory.length > 0) {
      stateHistory.forEach((pt) => {
        if (pt.position[0] < minX) minX = pt.position[0];
        if (pt.position[0] > maxX) maxX = pt.position[0];
        if (pt.position[1] < minY) minY = pt.position[1];
        if (pt.position[1] > maxY) maxY = pt.position[1];
      });
    }

    if (!isFinite(minX)) {
      minX = -1e7; maxX = 1e7; minY = -1e7; maxY = 1e7;
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
  }, [multiSimResult, stateHistory]);

  // 3. Fit Local / Selected Object View
  const fitObject = useCallback((objId?: string | null) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    const targetId = objId || selectedObjectId;
    let targetPos: [number, number, number] | null = null;

    if (multiSimResult && targetId) {
      const obj = multiSimResult.objects.find((o) => o.id === targetId);
      if (obj && obj.state_history.length > 0) {
        const idx = Math.min(currentFrameIdx, obj.state_history.length - 1);
        targetPos = obj.state_history[idx].position;
      }
    } else if (stateHistory && stateHistory.length > 0) {
      const idx = Math.min(currentFrameIdx, stateHistory.length - 1);
      targetPos = stateHistory[idx].position;
    }

    if (targetPos) {
      const span = 200000.0; // 200 km local framing
      const newZoom = Math.min(width, height) / span;
      setCamera({
        x: targetPos[0],
        y: targetPos[1],
        zoom: Math.max(1e-16, newZoom),
      });
      setViewContext("OBJECT");
    }
  }, [multiSimResult, selectedObjectId, stateHistory, currentFrameIdx]);

  // Auto-fit on initial render or simulation change
  useEffect(() => {
    fitMission();
  }, [multiSimResult, fitMission]);

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
    const worldPt = screenToWorld(clickX, clickY, rect.width, rect.height);

    // Check collision / conjunction click or spacecraft selection
    if (multiSimResult && multiSimResult.objects.length > 0) {
      let closestObj: SpacecraftTrack | null = null;
      let minScreenDist = 25.0; // 25 px radius

      multiSimResult.objects.forEach((obj) => {
        const idx = Math.min(currentFrameIdx, obj.state_history.length - 1);
        const st = obj.state_history[idx];
        if (!st || !st.active) return;

        const screenPt = worldToScreen(st.position[0], st.position[1], rect.width, rect.height);
        const dist = Math.hypot(clickX - screenPt.x, clickY - screenPt.y);
        if (dist < minScreenDist) {
          minScreenDist = dist;
          closestObj = obj;
        }
      });

      if (closestObj && onSelectObject) {
        onSelectObject((closestObj as SpacecraftTrack).id);
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

    // Disable image smoothing for crisp pixel rendering
    ctx.imageSmoothingEnabled = false;

    // -------------------------------------------------------------
    // CONTEXT A: SYSTEM VIEW (Full Heliocentric Solar System)
    // -------------------------------------------------------------
    if (viewContext === "SYSTEM") {
      // 1. Draw Sun at (0, 0)
      const sunScreen = worldToScreen(0, 0, width, height);
      const sunRad = scaleMode === "DISPLAY" ? 10 : Math.max(3, 696340000 * camera.zoom);
      ctx.save();
      ctx.translate(sunScreen.x, sunScreen.y);
      drawPixelPlanet(ctx, "Sun", sunRad, "Sun");
      ctx.restore();

      // 2. Draw all 8 planetary orbits (Keplerian Ellipses) + bodies
      const allPlanets = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"];
      
      allPlanets.forEach((pKey) => {
        const pState = getPlanetStateAtTime(pKey, simTimeSec);
        if (!pState || pState.orbitalPathM.length === 0) return;

        // Draw true elliptical Keplerian orbit path with Sun at primary focus
        ctx.strokeStyle = "rgba(120, 120, 120, 0.35)";
        ctx.setLineDash([2, 4]);
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        pState.orbitalPathM.forEach(([ox, oy], idx) => {
          const s = worldToScreen(ox, oy, width, height);
          if (idx === 0) ctx.moveTo(s.x, s.y);
          else ctx.lineTo(s.x, s.y);
        });
        ctx.closePath();
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw Planet at its physical dynamic position
        const ps = worldToScreen(pState.positionM[0], pState.positionM[1], width, height);

        let pr = 3;
        if (["jupiter", "saturn"].includes(pKey)) pr = 6;
        else if (["uranus", "neptune"].includes(pKey)) pr = 5;
        else if (pKey === "mercury") pr = 2;

        const body = CELESTIAL_BODIES[pKey];
        ctx.save();
        ctx.translate(ps.x, ps.y);
        drawPixelPlanet(ctx, body ? body.name : pKey, pr, body ? body.name : pKey, true);
        ctx.restore();
      });

      // Spacecraft in System View
      if (multiSimResult && multiSimResult.objects.length > 0) {
        multiSimResult.objects.forEach((obj) => {
          const idx = Math.min(currentFrameIdx, obj.state_history.length - 1);
          const st = obj.state_history[idx];
          if (!st || !st.active) return;

          const scPos = worldToScreen(st.position[0], st.position[1], width, height);
          ctx.fillStyle = obj.color || "#ffffff";
          ctx.fillRect(scPos.x - 2, scPos.y - 2, 4, 4);
        });
      }
    } 

    // -------------------------------------------------------------
    // CONTEXT B & C: MISSION & OBJECT VIEW
    // -------------------------------------------------------------
    else {
      // 1. Draw Primary Central Body
      const centralBodyKey = (multiSimResult ? multiSimResult.central_body : originBodyName).toLowerCase();
      const centralBody = CELESTIAL_BODIES[centralBodyKey] || CELESTIAL_BODIES["earth"];
      
      const bodyScreen = worldToScreen(0, 0, width, height);
      ctx.save();
      ctx.translate(bodyScreen.x, bodyScreen.y);
      const bRad = scaleMode === "DISPLAY" ? 18 : Math.max(4, centralBody.radius_km * 1000.0 * camera.zoom);
      drawPixelPlanet(ctx, centralBody.name, bRad, centralBody.name);
      ctx.restore();

      // If Sun is central body, draw the planetary Keplerian paths and dynamic planets
      if (centralBodyKey === "sun") {
        const planetsToDraw = ["mercury", "venus", "earth", "mars", "jupiter", "saturn"];
        planetsToDraw.forEach((pKey) => {
          const pState = getPlanetStateAtTime(pKey, simTimeSec);
          if (!pState.orbitalPathM.length) return;

          // Draw Elliptical Orbit Path
          ctx.strokeStyle = "rgba(140, 140, 140, 0.35)";
          ctx.setLineDash([2, 4]);
          ctx.lineWidth = 0.9;
          ctx.beginPath();
          pState.orbitalPathM.forEach(([ox, oy], idx) => {
            const s = worldToScreen(ox, oy, width, height);
            if (idx === 0) ctx.moveTo(s.x, s.y);
            else ctx.lineTo(s.x, s.y);
          });
          ctx.closePath();
          ctx.stroke();
          ctx.setLineDash([]);

          // Draw Planet at its physical dynamic position
          const ps = worldToScreen(pState.positionM[0], pState.positionM[1], width, height);
          const pr = scaleMode === "DISPLAY" ? (["jupiter", "saturn"].includes(pKey) ? 7 : 5) : Math.max(3, (CELESTIAL_BODIES[pKey]?.radius_km || 6000) * 1000.0 * camera.zoom);
          const body = CELESTIAL_BODIES[pKey];
          ctx.save();
          ctx.translate(ps.x, ps.y);
          drawPixelPlanet(ctx, body ? body.name : pKey, pr, body ? body.name : pKey, true);
          ctx.restore();
        });
      }

      // 2. Draw Multi-Spacecraft Trajectories & Sprites

      const allObjects: SpacecraftTrack[] = multiSimResult ? multiSimResult.objects : [];

      if (allObjects.length > 0) {
        // Draw Trajectory Trails
        allObjects.forEach((obj) => {
          if (obj.state_history.length < 2) return;

          const isSelected = selectedObjectId === obj.id;
          ctx.strokeStyle = obj.is_debris ? "rgba(160, 160, 160, 0.45)" : (obj.color || "#ff9900");
          ctx.lineWidth = isSelected ? 2.2 : (obj.is_debris ? 0.9 : 1.4);

          if (obj.is_debris) {
            ctx.setLineDash([2, 3]);
          } else {
            ctx.setLineDash([]);
          }

          ctx.beginPath();
          let started = false;
          obj.state_history.forEach((pt) => {
            if (obj.is_debris && !pt.active) return;
            const s = worldToScreen(pt.position[0], pt.position[1], width, height);
            if (!started) {
              ctx.moveTo(s.x, s.y);
              started = true;
            } else {
              ctx.lineTo(s.x, s.y);
            }
          });
          ctx.stroke();
          ctx.setLineDash([]);
        });

        // 3. Draw Conjunction Encounter Link Lines
        if (multiSimResult && multiSimResult.conjunctions.length > 0) {
          multiSimResult.conjunctions.forEach((conj) => {
            // Draw link line if close in time to TCA (|t - TCA| < 120s)
            const dtTCA = Math.abs(simTimeSec - conj.tca_s);
            if (dtTCA < 180.0) {
              const scA = allObjects.find((o) => o.id === conj.spacecraft_a_id);
              const scB = allObjects.find((o) => o.id === conj.spacecraft_b_id);

              if (scA && scB) {
                const idxA = Math.min(currentFrameIdx, scA.state_history.length - 1);
                const idxB = Math.min(currentFrameIdx, scB.state_history.length - 1);
                const stA = scA.state_history[idxA];
                const stB = scB.state_history[idxB];

                if (stA && stB && stA.active && stB.active) {
                  const sA = worldToScreen(stA.position[0], stA.position[1], width, height);
                  const sB = worldToScreen(stB.position[0], stB.position[1], width, height);

                  // Dashed conjunction connection
                  ctx.strokeStyle = conj.risk_level === "CRITICAL" ? "#ff3333" : (conj.risk_level === "HIGH" ? "#ff9900" : "rgba(230, 223, 213, 0.7)");
                  ctx.lineWidth = 1.0;
                  ctx.setLineDash([3, 3]);
                  ctx.beginPath();
                  ctx.moveTo(sA.x, sA.y);
                  ctx.lineTo(sB.x, sB.y);
                  ctx.stroke();
                  ctx.setLineDash([]);

                  // Midpoint Callout Tag
                  const midX = (sA.x + sB.x) / 2;
                  const midY = (sA.y + sB.y) / 2;
                  ctx.fillStyle = "rgba(0, 0, 0, 0.75)";
                  ctx.fillRect(midX - 32, midY - 14, 64, 13);
                  ctx.strokeStyle = ctx.strokeStyle;
                  ctx.strokeRect(midX - 32, midY - 14, 64, 13);

                  ctx.font = "8px 'JetBrains Mono', monospace";
                  ctx.fillStyle = "#ffffff";
                  ctx.textAlign = "center";
                  ctx.fillText(`MISS: ${(conj.miss_distance_m).toFixed(0)}m`, midX, midY - 4);
                }
              }
            }
          });
        }

        // 4. Draw Spacecraft & Debris Sprites at Current Frame
        allObjects.forEach((obj) => {
          const idx = Math.min(currentFrameIdx, obj.state_history.length - 1);
          const st = obj.state_history[idx];
          if (!st) return;

          // If destroyed before this timestamp, don't draw active sprite
          if (st.destroyed) {
            return;
          }

          if (obj.is_debris && !st.active) {
            return; // Pre-collision debris inactive
          }

          const s = worldToScreen(st.position[0], st.position[1], width, height);
          const isSelected = selectedObjectId === obj.id;

          ctx.save();
          ctx.translate(s.x, s.y);

          if (obj.is_debris) {
            // Draw Debris Fragment with physical rotation
            const rot = (simTimeSec * 0.8 + parseInt(obj.id.slice(-1) || "1")) % (Math.PI * 2);
            drawPixelDebris(ctx, obj.debris_type || "solar_panel", rot);

            // Debris label
            ctx.font = "8px 'JetBrains Mono', monospace";
            ctx.fillStyle = isSelected ? "#ffcc00" : "#a0988e";
            ctx.textAlign = "center";
            ctx.fillText(obj.name.toUpperCase(), 0, 14);

          } else {
            // Draw Spacecraft Sprite
            drawPixelSpacecraft(
              ctx,
              obj.sprite_id || "falcon9",
              st.velocity[0],
              st.velocity[1],
              isThrustActive && isSelected
            );

            // Selection Reticle
            if (isSelected) {
              ctx.strokeStyle = "#ffcc00";
              ctx.lineWidth = 1;
              ctx.setLineDash([2, 2]);
              ctx.strokeRect(-16, -16, 32, 32);
              ctx.setLineDash([]);
            }

            // Spacecraft Name & Telemetry Label
            ctx.font = "8.5px 'JetBrains Mono', monospace";
            ctx.fillStyle = isSelected ? "#ffcc00" : (obj.color || "#ffffff");
            ctx.textAlign = "center";
            ctx.fillText(obj.name.toUpperCase(), 0, 16);

            // Alt / Speed subtext
            ctx.font = "7.5px 'JetBrains Mono', monospace";
            ctx.fillStyle = "rgba(200, 200, 200, 0.75)";
            ctx.fillText(`${(st.altitude / 1e3).toFixed(1)}km | ${(st.speed / 1e3).toFixed(2)}km/s`, 0, 25);
          }

          ctx.restore();
        });

        // 5. Draw Physical Collision Explosions
        if (multiSimResult && multiSimResult.collisions.length > 0) {
          multiSimResult.collisions.forEach((coll) => {
            const dtColl = simTimeSec - coll.time_s;
            // Active explosion animation for 60 seconds after impact
            if (dtColl >= 0 && dtColl <= 60.0) {
              const progress = dtColl / 60.0;
              const cs = worldToScreen(coll.collision_position_m[0], coll.collision_position_m[1], width, height);

              ctx.save();
              ctx.translate(cs.x, cs.y);
              drawPixelImpactExplosion(ctx, progress);

              ctx.font = "9px 'JetBrains Mono', monospace";
              ctx.fillStyle = "#ff3333";
              ctx.textAlign = "center";
              ctx.fillText("COLLISION IMPACT", 0, -28);
              ctx.restore();
            }
          });
        }
      } else if (stateHistory && stateHistory.length > 1) {
        // Fallback Single-Trajectory Rendering
        ctx.strokeStyle = "#ff9900";
        ctx.lineWidth = 2.0;
        ctx.beginPath();
        stateHistory.forEach((pt, idx) => {
          const s = worldToScreen(pt.position[0], pt.position[1], width, height);
          if (idx === 0) ctx.moveTo(s.x, s.y);
          else ctx.lineTo(s.x, s.y);
        });
        ctx.stroke();

        const curSt = stateHistory[Math.min(currentFrameIdx, stateHistory.length - 1)];
        if (curSt) {
          const s = worldToScreen(curSt.position[0], curSt.position[1], width, height);
          ctx.save();
          ctx.translate(s.x, s.y);
          drawPixelSpacecraft(ctx, spacecraftPresetId, curSt.velocity[0], curSt.velocity[1], isThrustActive);
          ctx.restore();
        }
      }
    }

  }, [
    camera,
    viewContext,
    scaleMode,
    currentFrameIdx,
    simTimeSec,
    multiSimResult,
    stateHistory,
    selectedObjectId,
    originBodyName,
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
          System
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
          Mission
        </button>

        <button
          onClick={() => fitObject(selectedObjectId)}
          className={`px-2.5 py-1 text-xs border uppercase tracking-wider transition-colors ${
            viewContext === "OBJECT"
              ? "bg-amber-500/20 text-amber-400 border-amber-500/60"
              : "bg-black/80 text-neutral-400 border-neutral-800 hover:border-neutral-600 hover:text-white"
          }`}
        >
          <Crosshair className="w-3 h-3 inline mr-1" />
          Focus Object
        </button>

        <button
          onClick={() => setScaleMode((m) => (m === "DISPLAY" ? "TRUE" : "DISPLAY"))}
          className={`px-2.5 py-1 text-xs border uppercase tracking-wider transition-colors ${
            scaleMode === "TRUE"
              ? "bg-neutral-800 text-white border-neutral-600"
              : "bg-black/80 text-neutral-400 border-neutral-800 hover:text-white"
          }`}
          title="Toggle True Physical Scale vs Enhanced Display Mode"
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

      {/* Bottom Left: Simulation Timestamp & Object Counter */}
      <div className="absolute bottom-3 left-3 bg-black/85 border border-neutral-800 px-3 py-2 text-xs text-neutral-300 z-10">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-amber-400 font-semibold">T+ {formatTimeSeconds(simTimeSec)}</span>
          <span className="text-neutral-500">|</span>
          <span>{multiSimResult ? `${multiSimResult.objects.length} OBJECTS TRACKED` : "1 OBJECT TRACKED"}</span>
        </div>
        {multiSimResult && multiSimResult.collisions.length > 0 && (
          <div className="text-red-400 flex items-center gap-1.5 text-[11px] mt-0.5">
            <Flame className="w-3 h-3" />
            <span>{multiSimResult.collisions.length} PHYSICAL COLLISION(S) DETECTED</span>
          </div>
        )}
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
