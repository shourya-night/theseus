import React, { useState, useEffect } from "react";
import { Sandbox2D } from "./components/Simulation/Sandbox2D";
import { CalculationAnalysisOverlay } from "./components/Calculations/CalculationAnalysisOverlay";
import { TimelineScrubber } from "./components/Telemetry/TimelineScrubber";
import { MissionSetupModal } from "./components/Mission/MissionSetupModal";
import { MultiSpacecraftSetupModal } from "./components/Mission/MultiSpacecraftSetupModal";
import { ConjunctionsPanel } from "./components/Analysis/ConjunctionsPanel";
import { BPlaneRiskOverlay } from "./components/Analysis/BPlaneRiskOverlay";
import { ErrorBoundary } from "./components/Common/ErrorBoundary";

import { SimulatorLayout } from "./components/Layout/SimulatorLayout";
import { Viewport3DContainer } from "./components/Simulation/Viewport3DContainer";
import { MissionBuildOverlay } from "./components/Mission/MissionBuildOverlay";
import { LayerState, DEFAULT_LAYER_STATE } from "./components/Visualization/VisualizationLayersPanel";

import { 
  SimulationResult, 
  MultiSimulationResult, 
  SpacecraftConfig, 
  SpacecraftTrack, 
  MultiConjunctionEvent, 
  PhysicalCollisionEvent,
  RocketPreset,
  ActiveRocket,
  ActiveExplosion
} from "./types/mission";
import { getRocketStateAtTime, hasFlyingRockets, getRocketLifecycleState } from "./lib/simulationClock";
import { CELESTIAL_BODIES } from "./data/celestialCatalog";
import { ROCKET_PRESETS } from "./data/rocketPresets";
import { 
  checkBackendHealth, 
  fetchHealthDetails,
  BackendHealth,
  simulateHohmann, 
  simulateLambert, 
  simulateRendezvous, 
  simulateMultiEnvironment,
  simulateIntercept,
  fetchDemoMission 
} from "./api/client";
import { formatSpeed, formatDistance, formatMass } from "./lib/formatter";
import { 
  Rocket, 
  Play, 
  Activity, 
  AlertTriangle,
  Globe2,
  Sliders,
  Settings2,
  Target,
  Edit3,
  CheckCircle2,
  HelpCircle,
  Terminal,
  BookOpen,
  Layers,
  Flame,
  Crosshair,
  Sparkles,
  ChevronRight,
  ChevronDown,
  ShieldCheck,
  Cpu
} from "lucide-react";

export function App() {
  const [backendOnline, setBackendOnline] = useState<boolean>(false);
  const [backendHealth, setBackendHealth] = useState<BackendHealth | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [simulationError, setSimulationError] = useState<string | null>(null);

  // Modals Visibility
  const [isSetupOpen, setIsSetupOpen] = useState<boolean>(false);
  const [isMultiSetupOpen, setIsMultiSetupOpen] = useState<boolean>(false);
  const [isCalculationsOpen, setIsCalculationsOpen] = useState<boolean>(false);
  const [isBuildOverlayOpen, setIsBuildOverlayOpen] = useState<boolean>(false);
  const [selectedConjunction, setSelectedConjunction] = useState<MultiConjunctionEvent | null>(null);
  const [showConjunctionsSidebar, setShowConjunctionsSidebar] = useState<boolean>(true);

  // Visualization Layers & Focus State
  const [layers, setLayers] = useState<LayerState>(DEFAULT_LAYER_STATE);
  const [focusedObjectId, setFocusedObjectId] = useState<string | null>(null);

  const handleToggleLayer = (key: keyof LayerState) => {
    setLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  // Active Simulation Mode: Default to 'SINGLE' (Phase 1-7 Core Mission)
  const [simMode, setSimMode] = useState<"MULTI" | "SINGLE">("SINGLE");

  // Single Mission Configuration State
  const [origin, setOrigin] = useState<string>("earth");
  const [destination, setDestination] = useState<string>("mars");
  const [selectedPresetId, setSelectedPresetId] = useState<string>("isro-lvm3");
  const [payloadKg, setPayloadKg] = useState<number>(2500);
  const [epochDate, setEpochDate] = useState<string>("2026-08-18");

  // Multi-Rocket Active Fleet State
  const [activeRockets, setActiveRockets] = useState<ActiveRocket[]>([]);
  const [activeExplosions, setActiveExplosions] = useState<ActiveExplosion[]>([]);

  // Multi-Spacecraft Fleet State
  const [fleetList, setFleetList] = useState<SpacecraftConfig[]>([
    {
      id: "SC-01",
      name: "Explorer-01",
      vehicle_type: "falcon9",
      color: "#ff9900",
      sprite_id: "falcon9",
      dry_mass_kg: 2000.0,
      fuel_mass_kg: 1000.0,
      cross_section_area_m2: 12.0,
      drag_coefficient: 2.2,
      reflectivity_coefficient: 1.5,
      thrust_n: 0.0,
      specific_impulse_s: 300.0,
      central_body: "Earth",
      semi_major_axis_km: 6778.137, // 400 km LEO
      eccentricity: 0.0,
      inclination_deg: 51.6,
      raan_deg: 0.0,
      arg_periapsis_deg: 0.0,
      true_anomaly_deg: 0.0,
      hard_body_radius_m: 8.0,
      sigma_pos_m: [80.0, 80.0, 80.0],
      sigma_vel_m_s: [0.08, 0.08, 0.08],
    },
    {
      id: "SC-02",
      name: "Relay-Sat-02",
      vehicle_type: "isro-lvm3",
      color: "#3388ff",
      sprite_id: "lvm3",
      dry_mass_kg: 3500.0,
      fuel_mass_kg: 1200.0,
      cross_section_area_m2: 15.0,
      drag_coefficient: 2.2,
      reflectivity_coefficient: 1.5,
      thrust_n: 0.0,
      specific_impulse_s: 300.0,
      central_body: "Earth",
      semi_major_axis_km: 6778.137,
      eccentricity: 0.0,
      inclination_deg: -51.6,
      raan_deg: 0.0,
      arg_periapsis_deg: 0.0,
      true_anomaly_deg: 0.0005, // Intersects exactly at node with miss < 8m (physical collision!)
      hard_body_radius_m: 10.0,
      sigma_pos_m: [80.0, 80.0, 80.0],
      sigma_vel_m_s: [0.08, 0.08, 0.08],
    },
  ]);

  // Active Simulation Results
  const [multiSimResult, setMultiSimResult] = useState<MultiSimulationResult | null>(null);
  const [simResult, setSimResult] = useState<SimulationResult | null>(null);
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>("SC-01");

  // Master Physical Simulation Clock State (Authoritative time in SI seconds)
  const [simTimeSec, setSimTimeSec] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);

  // Spacecraft Specs for single mode
  const selectedPreset = ROCKET_PRESETS.find((p) => p.id === selectedPresetId) || ROCKET_PRESETS[0];

  // Periodic Backend Health & Subsystem Status Check
  useEffect(() => {
    const check = async () => {
      const health = await fetchHealthDetails();
      if (health) {
        setBackendOnline(true);
        setBackendHealth(health);
      } else {
        setBackendOnline(false);
        setBackendHealth(null);
      }
    };
    check();
    const interval = setInterval(check, 8000);
    return () => clearInterval(interval);
  }, []);

  // Initial Auto-Run: Core Single Spacecraft Mission (Earth -> Mars)
  useEffect(() => {
    handleRunSingleSimulation();
  }, []);

  // Keyboard shortcut: Escape to close modals
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setIsSetupOpen(false);
        setIsMultiSetupOpen(false);
        setIsCalculationsOpen(false);
        setSelectedConjunction(null);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Master Physical Simulation Clock Loop (Advances simTimeSec directly)
  useEffect(() => {
    if (!isPlaying) return;

    let totalDurationSec = 7200;
    if (activeRockets.length > 0) {
      totalDurationSec = Math.max(...activeRockets.map((r) => r.result?.state_history ? r.result.state_history[r.result.state_history.length - 1]?.time_seconds || 0 : 0));
    } else if (simMode === "MULTI" && multiSimResult?.objects?.length) {
      totalDurationSec = multiSimResult.objects[0].state_history[multiSimResult.objects[0].state_history.length - 1]?.time_seconds || 7200;
    } else if (simResult?.state_history?.length) {
      totalDurationSec = simResult.state_history[simResult.state_history.length - 1]?.time_seconds || 864000;
    }

    const dtSimSec = Math.max(3600, Math.round((totalDurationSec / 200) * playbackSpeed));
    const intervalMs = 30; // ~33 FPS smooth tick

    const timer = setInterval(() => {
      setSimTimeSec((prev) => {
        const nextTime = prev + dtSimSec;

        // Check if multi-rocket fleet has any FLYING rockets
        if (activeRockets.length > 0 && !hasFlyingRockets(activeRockets, nextTime)) {
          console.log(`[SIMULATION COMPLETE] Time: ${(nextTime / 86400).toFixed(1)} days | Flying rockets: 0`);
          setIsPlaying(false);
          return nextTime;
        }

        if (activeRockets.length === 0 && nextTime >= totalDurationSec) {
          setIsPlaying(false);
          return totalDurationSec;
        }

        return nextTime;
      });
    }, intervalMs);

    return () => clearInterval(timer);
  }, [isPlaying, playbackSpeed, activeRockets, simMode, multiSimResult, simResult]);

  // Execute Multi-Spacecraft Environment Simulation
  const handleRunMultiSimulation = async (spacecraftToRun: SpacecraftConfig[] = fleetList) => {
    setIsLoading(true);
    setSimulationError(null);
    setSimMode("MULTI");

    try {
      const res = await simulateMultiEnvironment({
        spacecraft: spacecraftToRun,
        central_body: "Earth",
        duration_hours: 2.0,
        dt_sec: 30.0,
        screening_threshold_km: 100.0,
        enable_j2: true,
        enable_drag: true,
      });

      if (!res || !res.objects || res.objects.length === 0) {
        throw new Error("Multi-spacecraft simulation returned no active object trajectories.");
      }

      setMultiSimResult(res);
      setSelectedObjectId(res.objects[0].id);
      setSimTimeSec(0);
      setIsPlaying(true);
    } catch (err: any) {
      console.error("Multi-simulation error:", err);
      setSimulationError(err?.message || "Failed to execute multi-spacecraft simulation.");
    } finally {
      setIsLoading(false);
    }
  };

  // Execute Single-Spacecraft Interplanetary Mission Simulation (Appends to activeRockets)
  const handleRunSingleSimulation = async (
    targetOrigin: string = origin,
    targetDest: string = destination,
    presetId: string = selectedPresetId,
    payloadMass: number = payloadKg,
    collisionEnabled: boolean = false,
    collisionTargetId: string | null = null
  ) => {
    setIsLoading(true);
    setSimulationError(null);
    setSimMode("SINGLE");

    const preset = ROCKET_PRESETS.find((p) => p.id === presetId) || selectedPreset;

    try {
      const origKey = targetOrigin.toLowerCase();
      const destKey = targetDest.toLowerCase();
      const origBody = CELESTIAL_BODIES[origKey];
      const destBody = CELESTIAL_BODIES[destKey];

      if (!origBody || !destBody) {
        throw new Error(`MODEL NOT AVAILABLE: Celestial body '${!origBody ? origKey : destKey}' not in catalog.`);
      }

      let result: SimulationResult;

      if (collisionEnabled && collisionTargetId) {
        const targetRocket = activeRockets.find((r) => r.id === collisionTargetId);
        if (!targetRocket || !targetRocket.result.state_history || targetRocket.result.state_history.length === 0) {
          throw new Error("NO VALID INTERCEPT FOUND: Selected collision target rocket has no state history.");
        }

        result = await simulateIntercept({
          origin_body: origBody.name,
          target_state_history: targetRocket.result.state_history,
          central_body: "Sun",
          dry_mass_kg: preset.dry_mass_kg + payloadMass,
          fuel_mass_kg: preset.propellant_mass_kg,
          specific_impulse_s: preset.specific_impulse_s,
          thrust_n: preset.max_thrust_n,
          min_future_time_s: 86400.0,
        });

      } else if (origKey === "earth" && destKey === "moon") {
        result = await simulateHohmann({
          r1_km: 6678.137,
          r2_km: 384400.0,
          origin_body: "earth",
          plane_change_deg: 0.0,
          dry_mass_kg: preset.dry_mass_kg + payloadMass,
          fuel_mass_kg: preset.propellant_mass_kg,
          specific_impulse_s: preset.specific_impulse_s,
          thrust_n: preset.max_thrust_n,
        });
      } else if (origKey === "earth" && destKey === "earth") {
        result = await simulateHohmann({
          r1_km: 6678.137,
          r2_km: 42164.0,
          origin_body: "earth",
          plane_change_deg: 28.5,
          dry_mass_kg: preset.dry_mass_kg + payloadMass,
          fuel_mass_kg: preset.propellant_mass_kg,
          specific_impulse_s: preset.specific_impulse_s,
          thrust_n: preset.max_thrust_n,
        });
      } else {
        const isInterplanetary = (origBody.parent === "Sun" && destBody.parent === "Sun") || origKey === "sun";
        const central_body = isInterplanetary ? "sun" : origKey;

        const r1_km_val = origBody.orbit_radius_km || 149597870.7;
        const r2_km_val = destBody.orbit_radius_km || 227939200.0;

        const a_tx_m = ((r1_km_val + r2_km_val) / 2.0) * 1000.0;
        const mu_sun = 1.32712440018e20;
        const est_tof_s = Math.PI * Math.sqrt(Math.pow(a_tx_m, 3) / mu_sun);
        const est_tof_hours = Number((est_tof_s / 3600.0).toFixed(1));

        const r1_vector_km: [number, number, number] = [r1_km_val, 0, 0];
        const r2_vector_km: [number, number, number] = [0, r2_km_val, 0];

        result = await simulateLambert({
          r1_km: r1_vector_km,
          r2_km: r2_vector_km,
          tof_hours: est_tof_hours,
          central_body: central_body,
          prograde: true,
          dry_mass_kg: preset.dry_mass_kg + payloadMass,
          fuel_mass_kg: preset.propellant_mass_kg,
          specific_impulse_s: preset.specific_impulse_s,
          thrust_n: preset.max_thrust_n,
          origin_body: origBody.name,
          destination_body: destBody.name,
        });
      }

      result.metadata.origin = origKey;
      result.metadata.destination = destKey;
      result.metadata.name = collisionEnabled ? `${origBody.name} → Target Intercept` : `${origBody.name} → ${destBody.name} Transfer`;

      setSimResult(result);

      // Create Active Rocket representation
      const COLORS = ["#ff9900", "#3388ff", "#00ffcc", "#ff33aa", "#ffcc00", "#aa55ff", "#00ffaa", "#ff5555"];
      const nextIdx = activeRockets.length + 1;
      const rocketId = `Rocket-${nextIdx}`;
      const rocketName = `Rocket ${nextIdx} — ${origBody.name} → ${collisionEnabled ? "Target Intercept" : destBody.name}`;
      const color = COLORS[(nextIdx - 1) % COLORS.length];

      const newRocket: ActiveRocket = {
        id: rocketId,
        name: rocketName,
        origin: origKey,
        destination: destKey,
        presetId: preset.id,
        presetName: preset.name,
        color: color,
        result: result,
        collisionEnabled: collisionEnabled,
        collisionTargetId: collisionTargetId,
        collisionState: collisionEnabled && collisionTargetId ? "TARGETING" : "NONE",
      };

      setActiveRockets((prev) => [...prev, newRocket]);
      setSimTimeSec(0);
      setIsPlaying(false);
      setIsBuildOverlayOpen(true);
    } catch (err: any) {
      console.error("Simulation error:", err);
      setSimulationError(err?.message || "Failed to solve mission trajectory.");
    } finally {
      setIsLoading(false);
    }
  };

  // Derive telemetry and UI scrubber variables
  let totalDurationSec = 7200;
  let totalFrames = 200;
  let currentTimeSec = simTimeSec;
  let activeSelectedTrack: SpacecraftTrack | null = null;

  if (activeRockets.length > 0) {
    totalDurationSec = Math.max(...activeRockets.map((r) => r.result?.state_history ? r.result.state_history[r.result.state_history.length - 1]?.time_seconds || 0 : 0));
    totalFrames = Math.max(200, Math.round((totalDurationSec / 86400) * 2));
  } else if (simMode === "MULTI" && multiSimResult && multiSimResult.objects.length > 0) {
    totalFrames = multiSimResult.objects[0].state_history.length;
    totalDurationSec = multiSimResult.objects[0].state_history[totalFrames - 1]?.time_seconds || 7200;
    activeSelectedTrack = multiSimResult.objects.find((o) => o.id === selectedObjectId) || multiSimResult.objects[0];
  } else if (simResult && simResult.state_history && simResult.state_history.length > 0) {
    totalFrames = simResult.state_history.length;
    totalDurationSec = simResult.state_history[totalFrames - 1]?.time_seconds || 0;
  }

  // Derive currentFrameIdx for UI timeline scrubber
  const currentFrameIdx = Math.min(
    totalFrames - 1,
    Math.max(0, Math.round((simTimeSec / Math.max(1, totalDurationSec)) * (totalFrames - 1)))
  );

  // Strict Fleet Lifecycle & Pairwise Collision Monitor Loop (Evaluates positions and fleet status at currentTimeSec)
  useEffect(() => {
    if (activeRockets.length === 0) return;

    activeRockets.forEach((rA) => {
      const stA = getRocketStateAtTime(rA, currentTimeSec);
      if (!stA) return;

      // 1. Sun Collision Check (|R_rocket(t)| < 2.0e9 m)
      const distSun = Math.hypot(stA.position[0], stA.position[1], stA.position[2]);
      if (distSun < 2.0e9 && rA.collisionState !== "DESTROYED_BY_SUN") {
        console.log(`[ROCKET LIFECYCLE] ${rA.name} (${rA.id}): FLYING → DESTROYED_BY_SUN at T = ${(currentTimeSec / 86400).toFixed(1)} days`);
        setActiveRockets((prev) =>
          prev.map((r) => (r.id === rA.id ? { ...r, collisionState: "DESTROYED_BY_SUN" } : r))
        );
        return; // Deterministic priority: Sun destruction takes precedence over rocket-rocket collision
      }

      // 2. Rocket-Rocket Collision Check (t >= 86400s)
      if (!rA.collisionEnabled || !rA.collisionTargetId || rA.collisionState === "COLLIDED" || rA.collisionState === "DESTROYED_BY_SUN") return;

      const rB = activeRockets.find((r) => r.id === rA.collisionTargetId);
      if (!rB || rB.collisionState === "DESTROYED_BY_SUN") return;

      const stB = getRocketStateAtTime(rB, currentTimeSec);
      if (!stB) return;

      // Enforce minimum future time (> 86400s / 1 day) to prevent initial launch overlap false collision
      if (currentTimeSec < 86400.0) return;

      const distM = Math.hypot(
        stA.position[0] - stB.position[0],
        stA.position[1] - stB.position[1],
        stA.position[2] - stB.position[2]
      );

      // Separate Solver Tolerance (5e9 m) from Visual Collision Radius (5e7 m = 50,000 km)
      const COLLISION_RADIUS_M = 5e7;

      if (distM <= COLLISION_RADIUS_M) {
        console.log(`[ROCKET LIFECYCLE] ${rA.name} & ${rB.name}: FLYING → COLLIDED at T = ${(currentTimeSec / 86400).toFixed(1)} days`);
        console.log(`[INTERCEPT PLAYBACK] Sim Time: ${currentTimeSec.toFixed(1)}s | Target Pos: [${stB.position.map(n => n.toFixed(0)).join(", ")}] | Interceptor Pos: [${stA.position.map(n => n.toFixed(0)).join(", ")}] | Runtime Separation: ${distM.toFixed(0)} m`);

        // Mark collision state exactly once
        setActiveRockets((prev) =>
          prev.map((r) => {
            if (r.id === rA.id || r.id === rB.id) {
              return {
                ...r,
                collisionState: "COLLIDED",
                collisionTimeSec: currentTimeSec,
                collisionPosM: stA.position,
              };
            }
            return r;
          })
        );

        // Spawn active retro pixel explosion
        const newExplosion: ActiveExplosion = {
          id: `exp-${rA.id}-${rB.id}-${currentTimeSec}`,
          positionM: stA.position,
          startTimeSec: currentTimeSec,
          durationSec: 1.2,
        };
        setActiveExplosions((prev) => [...prev, newExplosion]);
      }
    });

    // Check Authoritative Fleet Lifecycle Rule: Stop simulation ONLY when NO rockets are FLYING
    if (isPlaying && activeRockets.length > 0 && !hasFlyingRockets(activeRockets, currentTimeSec)) {
      console.log(`[SIMULATION COMPLETE] Time: ${(currentTimeSec / 86400).toFixed(1)} days | Flying rockets: 0`);
      setIsPlaying(false);
    }
  }, [currentTimeSec, activeRockets, isPlaying]);

  // Jump to specific time (e.g. TCA)
  const handleJumpToTime = (targetTimeSec: number) => {
    setSimTimeSec(targetTimeSec);
    setIsPlaying(false);
  };

  const origBody = CELESTIAL_BODIES[origin.toLowerCase()] || CELESTIAL_BODIES["earth"];
  const destBody = CELESTIAL_BODIES[destination.toLowerCase()] || CELESTIAL_BODIES["mars"];

  return (
    <ErrorBoundary>
      <div className="w-full h-full relative overflow-hidden bg-black">
        <SimulatorLayout
          onOpenMissionSetup={() => setIsSetupOpen(true)}
          onOpenAnalysisOverlay={() => setIsCalculationsOpen(true)}
          activeRockets={activeRockets}
          selectedObjectId={selectedObjectId}
          onSelectObjectId={setSelectedObjectId}
          onFocusObjectId={setFocusedObjectId}
          simTimeSec={simTimeSec}
          maxSimTimeSec={totalDurationSec}
          isPlaying={isPlaying}
          onTogglePlay={() => setIsPlaying(!isPlaying)}
          onSeekTime={(t) => setSimTimeSec(t)}
          simSpeed={playbackSpeed}
          onChangeSimSpeed={(s) => setPlaybackSpeed(s)}
          layers={layers}
          onToggleLayer={handleToggleLayer}
          viewportContent={
            <Viewport3DContainer
              layers={layers}
              activeRockets={activeRockets}
              simTimeSec={simTimeSec}
              selectedObjectId={selectedObjectId}
              focusedObjectId={focusedObjectId}
            />
          }
        />

        {/* ORBIT-X Mission Construction Animation Overlay */}
        <MissionBuildOverlay
          isOpen={isBuildOverlayOpen}
          onClose={() => setIsBuildOverlayOpen(false)}
          simResult={simResult}
          originName={origin}
          destinationName={destination}
          vehicleName={selectedPreset.name}
          onComplete={() => {
            setSimTimeSec(0);
            setIsPlaying(true);
          }}
        />

        {/* Calculation Analysis Overlay */}
        <CalculationAnalysisOverlay
          isOpen={isCalculationsOpen}
          onClose={() => setIsCalculationsOpen(false)}
          simResult={simResult}
          originName={origin}
          destinationName={destination}
          vehicleName={selectedPreset.name}
          payloadKg={payloadKg}
          epochDate={epochDate}
        />

        {/* Single Mission Setup Modal */}
        <MissionSetupModal
          isOpen={isSetupOpen}
          onClose={() => setIsSetupOpen(false)}
          currentOrigin={origin}
          currentDestination={destination}
          currentPresetId={selectedPresetId}
          currentPayloadKg={payloadKg}
          onInitializeMission={(config) => {
            setOrigin(config.origin);
            setDestination(config.destination);
            setSelectedPresetId(config.presetId);
            setPayloadKg(config.payloadKg);
            setEpochDate(config.epochDate);
            handleRunSingleSimulation(
              config.origin,
              config.destination,
              config.presetId,
              config.payloadKg,
              config.collisionEnabled,
              config.collisionTargetId
            );
          }}
          activeRockets={activeRockets}
        />

        {/* Multi-Spacecraft Setup Modal */}
        <MultiSpacecraftSetupModal
          isOpen={isMultiSetupOpen}
          onClose={() => setIsMultiSetupOpen(false)}
          spacecraftList={fleetList}
          onUpdateSpacecraftList={setFleetList}
          onLaunchAll={(list) => handleRunMultiSimulation(list)}
          isLoading={isLoading}
        />

        {/* B-Plane Risk Detail Overlay Modal */}
        {selectedConjunction && (
          <BPlaneRiskOverlay
            conjunction={selectedConjunction}
            onClose={() => setSelectedConjunction(null)}
          />
        )}
      </div>
    </ErrorBoundary>
  );
}

export default App;
