import React, { useRef, useEffect, useState, useCallback } from "react";
import { 
  StateVector, 
  BodyStateHistory,
  MultiSimulationResult, 
  SpacecraftTrack, 
  MultiConjunctionEvent, 
  PhysicalCollisionEvent 
} from "../../types/mission";
import { 
  CELESTIAL_BODIES, 
  AU_KM, 
  AU_METERS, 
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
  
  // Authoritative Frame Determination
  const hasSunBody = bodyHistories.some((b) => b.id.toLowerCase() === "sun" || b.name.toLowerCase() === "sun");
  const isHeliocentric = (
    hasSunBody ||
    (multiSimResult && multiSimResult.central_body?.toLowerCase() === "sun") ||
    (origBody.parent === "Sun" && destBody.parent === "Sun") ||
    origKey === "sun"
  );
  const centralBodyKey = isHeliocentric ? "sun" : (multiSimResult ? multiSimResult.central_body : originBodyName).toLowerCase();

  const getAuthoritativeBodyState = useCallback((bodyKey: string) => {
    const body = bodyHistories.find((b) => b.id.toLowerCase() === bodyKey.toLowerCase() || b.name.toLowerCase() === bodyKey.toLowerCase());
    if (!body || body.state_history.length === 0) return null;
    let best = body.state_history[0];
    let bestDt = Math.abs(best.time_seconds - simTimeSec);
    for (const state of body.state_history) {
      const dt = Math.abs(state.time_seconds - simTimeSec);
      if (dt < bestDt) { best = state; bestDt = dt; }
    }
    return { body, state: best };
  }, [bodyHistories, simTimeSec]);

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

  // 2. Fit Mission View (Bounds around Sun, Earth, Mars, and active trajectories)
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

    if (isHeliocentric) {
      includePoint(0, 0); // Include Sun at origin
    }

    bodyHistories.forEach((b) => {
      b.state_history.forEach((st) => {
        includePoint(st.position[0], st.position[1]);
      });
    });

    if (multiSimResult && multiSimResult.objects.length > 0) {
      multiSimResult.objects.forEach((obj) => {
        obj.state_history.forEach((pt) => {
          includePoint(pt.position[0], pt.position[1]);
        });
      });
    } else if (stateHistory && stateHistory.length > 0) {
      stateHistory.forEach((pt) => {
        includePoint(pt.position[0], pt.position[1]);
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
  }, [multiSimResult, stateHistory, bodyHistories, isHeliocentric]);

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
  }, [multiSimResult, stateHistory, fitMission]);

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

      // 2. Draw backend-provided planetary histories and current bodies.
      bodyHistories.forEach((history) => {
        if (history.id === "sun" || history.state_history.length === 0) return;
        ctx.strokeStyle = history.id === "earth" ? "rgba(50, 150, 255, 0.35)" : (history.id === "mars" ? "rgba(255, 100, 50, 0.35)" : "rgba(140, 140, 140, 0.35)");
        ctx.setLineDash([2, 4]);
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        history.state_history.forEach((sample, idx) => {
          const s = worldToScreen(sample.position[0], sample.position[1], width, height);
          if (idx === 0) ctx.moveTo(s.x, s.y); else ctx.lineTo(s.x, s.y);
        });
        ctx.stroke();
        ctx.setLineDash([]);
        const current = getAuthoritativeBodyState(history.id)?.state || history.state_history[0];
        const ps = worldToScreen(current.position[0], current.position[1], width, height);
        const pr = scaleMode === "DISPLAY" ? (history.id === "jupiter" ? 7 : (history.id === "earth" ? 6 : 5)) : Math.max(3, history.radius_m * camera.zoom);
        ctx.save();
        ctx.translate(ps.x, ps.y);
        drawPixelPlanet(ctx, history.name, pr, history.name, true);
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
      if (isHeliocentric) {
        // 1. Draw Sun at (0, 0)
        const sunScreen = worldToScreen(0, 0, width, height);
        const sunRad = scaleMode === "DISPLAY" ? 14 : Math.max(4, 696340000 * camera.zoom);
        ctx.save();
        ctx.translate(sunScreen.x, sunScreen.y);
        drawPixelPlanet(ctx, "Sun", sunRad, "Sun");
        ctx.restore();

        // 2. Draw backend-provided planetary histories (Earth, Mars, etc.)
        bodyHistories.forEach((history) => {
          if (history.id === "sun" || history.state_history.length === 0) return;
          ctx.strokeStyle = history.id === "earth" ? "rgba(50, 160, 255, 0.4)" : (history.id === "mars" ? "rgba(255, 100, 50, 0.4)" : "rgba(140, 140, 140, 0.35)");
          ctx.setLineDash([3, 4]);
          ctx.lineWidth = 1.1;
          ctx.beginPath();
          history.state_history.forEach((sample, idx) => {
            const pt = worldToScreen(sample.position[0], sample.position[1], width, height);
            if (idx === 0) ctx.moveTo(pt.x, pt.y); else ctx.lineTo(pt.x, pt.y);
          });
          ctx.stroke();
          ctx.setLineDash([]);

          // Draw planet at current simTimeSec
          const current = getAuthoritativeBodyState(history.id)?.state || history.state_history[0];
          const ps = worldToScreen(current.position[0], current.position[1], width, height);
          const pr = scaleMode === "DISPLAY" ? (history.id === "jupiter" ? 8 : (history.id === "earth" ? 7 : 6)) : Math.max(3, history.radius_m * camera.zoom);
          ctx.save();
          ctx.translate(ps.x, ps.y);
          drawPixelPlanet(ctx, history.name, pr, history.name, true);
          ctx.restore();
        });
      } else {
        // Draw Primary Central Body at (0, 0) (e.g. Earth for geocentric)
        const centralBody = CELESTIAL_BODIES[centralBodyKey] || CELESTIAL_BODIES["earth"];
        const bodyScreen = worldToScreen(0, 0, width, height);
        ctx.save();
        ctx.translate(bodyScreen.x, bodyScreen.y);
        const bRad = scaleMode === "DISPLAY" ? 18 : Math.max(4, centralBody.radius_km * 1000.0 * camera.zoom);
        drawPixelPlanet(ctx, centralBody.name, bRad, centralBody.name);
        ctx.restore();
      }

      // 3. Draw Trajectories & Spacecraft Sprites
      const allObjects: SpacecraftTrack[] = multiSimResult ? multiSimResult.objects : [];

      if (allObjects.length > 0) {
        // Multi-Spacecraft Fleet Trajectories
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

        // Conjunction Encounter Link Lines
        if (multiSimResult && multiSimResult.conjunctions.length > 0) {
          multiSimResult.conjunctions.forEach((conj) => {
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

                  ctx.strokeStyle = conj.risk_level === "CRITICAL" ? "#ff3333" : (conj.risk_level === "HIGH" ? "#ff9900" : "rgba(230, 223, 213, 0.7)");
                  ctx.lineWidth = 1.0;
                  ctx.setLineDash([3, 3]);
                  ctx.beginPath();
                  ctx.moveTo(sA.x, sA.y);
                  ctx.lineTo(sB.x, sB.y);
                  ctx.stroke();
                  ctx.setLineDash([]);

                  const midX = (sA.x + sB.x) / 2;
                  const midY = (sA.y + sB.y) / 2;
                  ctx.fillStyle = "rgba(0, 0, 0, 0.75)";
                  ctx.fillRect(midX - 32, midY - 14, 64, 13);
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

        // Draw Spacecraft & Debris Sprites
        allObjects.forEach((obj) => {
          const idx = Math.min(currentFrameIdx, obj.state_history.length - 1);
          const st = obj.state_history[idx];
          if (!st || st.destroyed) return;
          if (obj.is_debris && !st.active) return;

          const s = worldToScreen(st.position[0], st.position[1], width, height);
          const isSelected = selectedObjectId === obj.id;

          ctx.save();
          ctx.translate(s.x, s.y);

          if (obj.is_debris) {
            const rot = (simTimeSec * 0.8 + parseInt(obj.id.slice(-1) || "1")) % (Math.PI * 2);
            drawPixelDebris(ctx, obj.debris_type || "solar_panel", rot);
            ctx.font = "8px 'JetBrains Mono', monospace";
            ctx.fillStyle = isSelected ? "#ffcc00" : "#a0988e";
            ctx.textAlign = "center";
            ctx.fillText(obj.name.toUpperCase(), 0, 14);
          } else {
            drawPixelSpacecraft(
              ctx,
              obj.sprite_id || "falcon9",
              st.velocity[0],
              st.velocity[1],
              isThrustActive && isSelected
            );

            if (isSelected) {
              ctx.strokeStyle = "#ffcc00";
              ctx.lineWidth = 1;
              ctx.setLineDash([2, 2]);
              ctx.strokeRect(-16, -16, 32, 32);
              ctx.setLineDash([]);
            }

            ctx.font = "8.5px 'JetBrains Mono', monospace";
            ctx.fillStyle = isSelected ? "#ffcc00" : (obj.color || "#ffffff");
            ctx.textAlign = "center";
            ctx.fillText(obj.name.toUpperCase(), 0, 16);

            ctx.font = "7.5px 'JetBrains Mono', monospace";
            ctx.fillStyle = "rgba(200, 200, 200, 0.75)";
            ctx.fillText(`${(st.altitude / 1e3).toFixed(1)}km | ${(st.speed / 1e3).toFixed(2)}km/s`, 0, 25);
          }

          ctx.restore();
        });

        // Draw Collision Explosions
        if (multiSimResult && multiSimResult.collisions.length > 0) {
          multiSimResult.collisions.forEach((coll) => {
            const dtColl = simTimeSec - coll.time_s;
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
        // Single-Trajectory Rendering (Authoritative Backend Trajectory)
        ctx.strokeStyle = "#ffcc00";
        ctx.lineWidth = 2.2;
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
    }

  }, [
    camera,
    viewContext,
    scaleMode,
    currentFrameIdx,
    simTimeSec,
    multiSimResult,
    stateHistory,
    bodyHistories,
    getAuthoritativeBodyState,
    selectedObjectId,
    originBodyName,
    spacecraftPresetId,
    isThrustActive,
    isHeliocentric,
    centralBodyKey,
    worldToScreen,
  ]);

  // Phase 15: Compute Diagnostic HUD telemetry
  const curSCState = stateHistory[Math.min(currentFrameIdx, stateHistory.length - 1)] || null;
  const earthAuthState = getAuthoritativeBodyState("earth")?.state || null;
  const marsAuthState = getAuthoritativeBodyState("mars")?.state || null;

  const distToEarthKm = (curSCState && earthAuthState)
    ? (Math.hypot(curSCState.position[0] - earthAuthState.position[0], curSCState.position[1] - earthAuthState.position[1]) / 1e3)
    : 0.0;

  const distToMarsKm = (curSCState && marsAuthState)
    ? (Math.hypot(curSCState.position[0] - marsAuthState.position[0], curSCState.position[1] - marsAuthState.position[1]) / 1e3)
    : 0.0;

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

        <button
          onClick={() => setShowDebug(!showDebug)}
          className={`px-2.5 py-1 text-xs border uppercase tracking-wider transition-colors ${
            showDebug
              ? "bg-emerald-950 text-emerald-400 border-emerald-600"
              : "bg-black/80 text-neutral-400 border-neutral-800 hover:text-white"
          }`}
          title="Toggle Development Diagnostic Overlay (Phase 15)"
        >
          <Code2 className="w-3 h-3 inline mr-1" />
          Debug HUD
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

      {/* Phase 15: Development Diagnostic Overlay Panel */}
      {showDebug && (
        <div className="absolute bottom-14 right-3 w-80 bg-black/90 border border-emerald-500/60 p-3 text-[11px] text-emerald-300 font-mono z-20 shadow-2xl space-y-1.5">
          <div className="flex justify-between items-center border-b border-emerald-800/80 pb-1 font-bold text-white">
            <span>DIAGNOSTIC OVERLAY (PHASE 15)</span>
            <span className="text-[9px] bg-emerald-900/60 text-emerald-300 px-1 border border-emerald-600">LIVE</span>
          </div>

          <div className="flex justify-between">
            <span className="text-neutral-400">FRAME:</span>
            <span className="text-amber-300 font-bold">{isHeliocentric ? "HELIOCENTRIC (ICRF)" : "GEOCENTRIC (ECI)"}</span>
          </div>

          <div className="flex justify-between">
            <span className="text-neutral-400">POSITION UNITS:</span>
            <span className="text-white font-bold">METERS (SI)</span>
          </div>

          <div className="flex justify-between">
            <span className="text-neutral-400">SIM TIME:</span>
            <span className="text-white font-bold">{simTimeSec.toFixed(0)} s ({(simTimeSec / 86400).toFixed(1)} days)</span>
          </div>

          <div className="border-t border-neutral-800 pt-1">
            <div className="text-amber-400 font-bold mb-0.5">SUN:</div>
            <div className="text-[10px] text-neutral-300">x: 0.00 km, y: 0.00 km</div>
          </div>

          {earthAuthState && (
            <div className="border-t border-neutral-800 pt-1">
              <div className="text-cyan-400 font-bold mb-0.5 flex justify-between">
                <span>EARTH:</span>
                <span className="text-white font-normal">dist: {distToEarthKm.toLocaleString(undefined, { maximumFractionDigits: 0 })} km</span>
              </div>
              <div className="text-[10px] text-neutral-300">
                x: {(earthAuthState.position[0] / 1e3).toLocaleString(undefined, { maximumFractionDigits: 0 })} km, y: {(earthAuthState.position[1] / 1e3).toLocaleString(undefined, { maximumFractionDigits: 0 })} km
              </div>
            </div>
          )}

          {marsAuthState && (
            <div className="border-t border-neutral-800 pt-1">
              <div className="text-red-400 font-bold mb-0.5 flex justify-between">
                <span>MARS:</span>
                <span className="text-white font-normal">dist: {distToMarsKm.toLocaleString(undefined, { maximumFractionDigits: 0 })} km</span>
              </div>
              <div className="text-[10px] text-neutral-300">
                x: {(marsAuthState.position[0] / 1e3).toLocaleString(undefined, { maximumFractionDigits: 0 })} km, y: {(marsAuthState.position[1] / 1e3).toLocaleString(undefined, { maximumFractionDigits: 0 })} km
              </div>
            </div>
          )}

          {curSCState && (
            <div className="border-t border-neutral-800 pt-1">
              <div className="text-amber-300 font-bold mb-0.5">SPACECRAFT:</div>
              <div className="text-[10px] text-neutral-300">
                x: {(curSCState.position[0] / 1e3).toLocaleString(undefined, { maximumFractionDigits: 0 })} km, y: {(curSCState.position[1] / 1e3).toLocaleString(undefined, { maximumFractionDigits: 0 })} km
              </div>
              <div className="text-[10px] text-emerald-400">speed: {(curSCState.speed / 1e3).toFixed(2)} km/s</div>
            </div>
          )}

          <div className="border-t border-neutral-800 pt-1 flex justify-between text-[10px]">
            <span className="text-neutral-400">TRAJECTORY SAMPLES:</span>
            <span className="text-white font-bold">{stateHistory.length} states</span>
          </div>
        </div>
      )}

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
