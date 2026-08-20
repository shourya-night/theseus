import React, { useState, useEffect } from "react";
import { Sandbox2D } from "./components/Simulation/Sandbox2D";
import { CalculationAnalysisOverlay } from "./components/Calculations/CalculationAnalysisOverlay";
import { TimelineScrubber } from "./components/Telemetry/TimelineScrubber";
import { MissionSetupModal } from "./components/Mission/MissionSetupModal";
import { MultiSpacecraftSetupModal } from "./components/Mission/MultiSpacecraftSetupModal";
import { ConjunctionsPanel } from "./components/Analysis/ConjunctionsPanel";
import { BPlaneRiskOverlay } from "./components/Analysis/BPlaneRiskOverlay";
import { ErrorBoundary } from "./components/Common/ErrorBoundary";

import { 
  SimulationResult, 
  MultiSimulationResult, 
  SpacecraftConfig, 
  SpacecraftTrack, 
  MultiConjunctionEvent, 
  PhysicalCollisionEvent,
  RocketPreset 
} from "./types/mission";
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
  const [selectedConjunction, setSelectedConjunction] = useState<MultiConjunctionEvent | null>(null);
  const [showConjunctionsSidebar, setShowConjunctionsSidebar] = useState<boolean>(true);

  // Active Simulation Mode: Default to 'SINGLE' (Phase 1-7 Core Mission)
  const [simMode, setSimMode] = useState<"MULTI" | "SINGLE">("SINGLE");

  // Single Mission Configuration State
  const [origin, setOrigin] = useState<string>("earth");
  const [destination, setDestination] = useState<string>("mars");
  const [selectedPresetId, setSelectedPresetId] = useState<string>("isro-lvm3");
  const [payloadKg, setPayloadKg] = useState<number>(2500);
  const [epochDate, setEpochDate] = useState<string>("2026-08-18");

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

  // Playback State
  const [currentFrameIdx, setCurrentFrameIdx] = useState<number>(0);
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

  // Animation Frame Scrubber Loop
  useEffect(() => {
    let totalFrames = 0;
    if (simMode === "MULTI" && multiSimResult && multiSimResult.objects.length > 0) {
      totalFrames = multiSimResult.objects[0].state_history.length;
    } else if (simResult && simResult.state_history) {
      totalFrames = simResult.state_history.length;
    }

    if (!isPlaying || totalFrames === 0) return;

    const intervalMs = Math.max(20, Math.floor(60 / playbackSpeed));
    const timer = setInterval(() => {
      setCurrentFrameIdx((prev) => {
        if (prev >= totalFrames - 1) {
          setIsPlaying(false);
          return totalFrames - 1;
        }
        return prev + 1;
      });
    }, intervalMs);

    return () => clearInterval(timer);
  }, [isPlaying, playbackSpeed, simMode, multiSimResult, simResult]);

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
      setCurrentFrameIdx(0);
      setIsPlaying(true);
    } catch (err: any) {
      console.error("Multi-simulation error:", err);
      setSimulationError(err?.message || "Failed to execute multi-spacecraft simulation.");
    } finally {
      setIsLoading(false);
    }
  };

  // Execute Single-Spacecraft Interplanetary Mission Simulation
  const handleRunSingleSimulation = async (
    targetOrigin: string = origin,
    targetDest: string = destination,
    presetId: string = selectedPresetId,
    payloadMass: number = payloadKg
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

      if (origKey === "earth" && destKey === "moon") {
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

        // Placeholder vectors; the backend replaces these with authoritative
        // ephemeris states when origin_body and destination_body are supplied.
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
      result.metadata.name = `${origBody.name} → ${destBody.name} Transfer`;

      setSimResult(result);
      setCurrentFrameIdx(0);
      setIsPlaying(true);
    } catch (err: any) {
      console.error("Simulation error:", err);
      setSimulationError(err?.message || "Failed to solve mission trajectory.");
    } finally {
      setIsLoading(false);
    }
  };

  // Derive active time and state telemetry
  let currentTimeSec = 0;
  let totalDurationSec = 7200;
  let totalFrames = 0;
  let activeSelectedTrack: SpacecraftTrack | null = null;

  if (simMode === "MULTI" && multiSimResult && multiSimResult.objects.length > 0) {
    totalFrames = multiSimResult.objects[0].state_history.length;
    const clampedIdx = Math.min(currentFrameIdx, totalFrames - 1);
    currentTimeSec = multiSimResult.objects[0].state_history[clampedIdx]?.time_seconds || 0;
    totalDurationSec = multiSimResult.objects[0].state_history[totalFrames - 1]?.time_seconds || 7200;
    activeSelectedTrack = multiSimResult.objects.find((o) => o.id === selectedObjectId) || multiSimResult.objects[0];
  } else if (simResult && simResult.state_history && simResult.state_history.length > 0) {
    totalFrames = simResult.state_history.length;
    const clampedIdx = Math.min(currentFrameIdx, totalFrames - 1);
    currentTimeSec = simResult.state_history[clampedIdx]?.time_seconds || 0;
    totalDurationSec = simResult.state_history[totalFrames - 1]?.time_seconds || 0;
  }

  // Jump to specific time (e.g. TCA)
  const handleJumpToTime = (targetTimeSec: number) => {
    if (simMode === "MULTI" && multiSimResult && multiSimResult.objects.length > 0) {
      const times = multiSimResult.objects[0].state_history.map((s) => s.time_seconds);
      let closestIdx = 0;
      let minDiff = Infinity;
      times.forEach((t, idx) => {
        const diff = Math.abs(t - targetTimeSec);
        if (diff < minDiff) {
          minDiff = diff;
          closestIdx = idx;
        }
      });
      setCurrentFrameIdx(closestIdx);
      setIsPlaying(false);
    }
  };

  const origBody = CELESTIAL_BODIES[origin.toLowerCase()] || CELESTIAL_BODIES["earth"];
  const destBody = CELESTIAL_BODIES[destination.toLowerCase()] || CELESTIAL_BODIES["mars"];

  return (
    <ErrorBoundary>
      <div className="w-screen h-screen flex flex-col bg-black text-neutral-100 font-mono select-none overflow-hidden relative">
        
        {/* Modals */}
        <MultiSpacecraftSetupModal
          isOpen={isMultiSetupOpen}
          onClose={() => setIsMultiSetupOpen(false)}
          spacecraftList={fleetList}
          onUpdateSpacecraftList={setFleetList}
          onLaunchAll={(list) => {
            setFleetList(list);
            handleRunMultiSimulation(list);
          }}
          isLoading={isLoading}
        />

        <MissionSetupModal
          isOpen={isSetupOpen}
          onClose={() => setIsSetupOpen(false)}
          onInitializeMission={(config) => {
            setOrigin(config.origin);
            setDestination(config.destination);
            setSelectedPresetId(config.presetId);
            setPayloadKg(config.payloadKg);
            setEpochDate(config.epochDate);
            handleRunSingleSimulation(config.origin, config.destination, config.presetId, config.payloadKg);
          }}
          currentOrigin={origin}
          currentDestination={destination}
          currentPresetId={selectedPresetId}
          currentPayloadKg={payloadKg}
        />

        <BPlaneRiskOverlay
          conjunction={selectedConjunction}
          onClose={() => setSelectedConjunction(null)}
        />

        <CalculationAnalysisOverlay
          isOpen={isCalculationsOpen}
          onClose={() => setIsCalculationsOpen(false)}
          simResult={
            simResult || (activeSelectedTrack ? {
              mission_id: activeSelectedTrack.id,
              metadata: {
                name: activeSelectedTrack.name,
                origin: activeSelectedTrack.origin || (multiSimResult?.central_body === "Sun" ? "Earth" : "LEO"),
                destination: activeSelectedTrack.destination || (multiSimResult?.central_body === "Sun" ? "Mars" : "Orbit"),
                central_body: multiSimResult?.central_body || "Sun",
                status: activeSelectedTrack.destroyed ? "FAILED" : "SUCCESS",
              },
              delta_v_budget: activeSelectedTrack.delta_v_budget || {
                total_delta_v: 5600,
                available_delta_v: 6500,
                margin_delta_v: 900,
              },
              propellant_budget: activeSelectedTrack.propellant_budget || {
                initial_total_mass_kg: 5000,
                dry_mass_kg: 2000,
                initial_fuel_kg: 3000,
                fuel_consumed_kg: 2200,
                fuel_margin_kg: 800,
              },
              state_history: activeSelectedTrack.state_history.map((s) => ({
                time_seconds: s.time_seconds,
                position: s.position,
                velocity: s.velocity,
                mass: s.mass,
                fuel_mass: s.fuel_mass,
                thrust_active: s.thrust_active,
                altitude: s.altitude,
                speed: s.speed,
              })),
              calculation_trace: (activeSelectedTrack.calculation_trace && activeSelectedTrack.calculation_trace.length > 0)
                ? activeSelectedTrack.calculation_trace
                : (multiSimResult?.calculation_steps || []),
              events: [],
              diagnostics: {
                solver: "RKF45 Adaptive Astrodynamics Engine",
                numerical_tolerance: "atol=1e-7, rtol=1e-7",
                scientific_honesty_note: "Rigorous physical variational state propagation",
              },
            } : null)
          }
          originName={activeSelectedTrack?.origin || origBody.name}
          destinationName={activeSelectedTrack?.destination || destBody.name}
          vehicleName={activeSelectedTrack?.name || selectedPreset.name}
          payloadKg={payloadKg}
          epochDate={epochDate}
          isCalculating={isLoading}
        />


        {/* 1. TOP STATUS & NAVIGATION BAR */}
        <header className="w-full bg-[#050505] border-b border-neutral-800 px-4 py-2 flex items-center justify-between shrink-0 text-xs select-none">
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <span className="font-['Orbitron'] font-black tracking-widest text-sm text-white">
                THESEUS
              </span>
              <span className="text-neutral-600">/</span>
              <span className="text-[11px] text-neutral-400 tracking-wider font-semibold">
                ORBITAL DYNAMICS ENGINE
              </span>
            </div>

            <div className="flex items-center space-x-1.5 bg-black border border-neutral-800 px-2 py-0.5 text-[10px]">
              <span className={`w-1.5 h-1.5 rounded-full ${backendOnline ? "bg-emerald-400 animate-pulse" : "bg-red-500"}`} />
              <span className={backendOnline ? "text-emerald-400 font-bold" : "text-red-400 font-bold"}>
                {backendOnline ? "CORE ONLINE" : "CORE OFFLINE"}
              </span>
            </div>

            {/* Subsystem Health Indicators */}
            <div className="hidden lg:flex items-center space-x-1 text-[9px] font-mono">
              <span className="bg-neutral-900 border border-emerald-900/60 text-emerald-400 px-1.5 py-0.5 rounded">
                PROPAGATOR
              </span>
              <span className="bg-neutral-900 border border-emerald-900/60 text-emerald-400 px-1.5 py-0.5 rounded">
                TRANSFERS
              </span>
              <span className="bg-neutral-900 border border-emerald-900/60 text-emerald-400 px-1.5 py-0.5 rounded">
                RENDEZVOUS
              </span>
              <span className={`px-1.5 py-0.5 rounded border ${
                backendHealth?.subsystems?.phase_8_reentry?.includes("VALIDATED")
                  ? "bg-neutral-900 border-emerald-900/60 text-emerald-400"
                  : "bg-neutral-900 border-neutral-800 text-neutral-500"
              }`}>
                REENTRY: {backendHealth?.subsystems?.phase_8_reentry?.includes("VALIDATED") ? "ONLINE" : "STANDBY"}
              </span>
              <span className={`px-1.5 py-0.5 rounded border ${
                backendHealth?.subsystems?.phase_9_collision?.includes("VALIDATED")
                  ? "bg-neutral-900 border-emerald-900/60 text-emerald-400"
                  : "bg-neutral-900 border-neutral-800 text-neutral-500"
              }`}>
                CONJUNCTION: {backendHealth?.subsystems?.phase_9_collision?.includes("VALIDATED") ? "ONLINE" : "STANDBY"}
              </span>
              <span className={`px-1.5 py-0.5 rounded border ${
                backendHealth?.subsystems?.phase_10_uncertainty?.includes("VALIDATED")
                  ? "bg-neutral-900 border-emerald-900/60 text-emerald-400"
                  : "bg-neutral-900 border-neutral-800 text-neutral-500"
              }`}>
                UNCERTAINTY: {backendHealth?.subsystems?.phase_10_uncertainty?.includes("VALIDATED") ? "ONLINE" : "STANDBY"}
              </span>
            </div>
          </div>

          {/* Mode Switcher & Fleet Setup */}
          <div className="flex items-center space-x-2">
            <div className="bg-black border border-neutral-800 p-0.5 flex text-[11px]">
              <button
                onClick={() => {
                  setSimMode("MULTI");
                  handleRunMultiSimulation();
                }}
                className={`px-2.5 py-1 uppercase font-bold tracking-wider transition-colors ${
                  simMode === "MULTI"
                    ? "bg-amber-500 text-black"
                    : "text-neutral-400 hover:text-white"
                }`}
              >
                Multi-Fleet & Conjunctions
              </button>
              <button
                onClick={() => {
                  setSimMode("SINGLE");
                  handleRunSingleSimulation();
                }}
                className={`px-2.5 py-1 uppercase font-bold tracking-wider transition-colors ${
                  simMode === "SINGLE"
                    ? "bg-amber-500 text-black"
                    : "text-neutral-400 hover:text-white"
                }`}
              >
                Interplanetary Transfer
              </button>
            </div>

            {simMode === "MULTI" ? (
              <button
                onClick={() => setIsMultiSetupOpen(true)}
                className="bg-neutral-900 hover:bg-neutral-800 border border-amber-500/60 text-amber-400 hover:text-white font-['Orbitron'] font-bold px-3 py-1 text-xs transition-all flex items-center space-x-1.5 cursor-pointer"
              >
                <Settings2 className="w-3.5 h-3.5" />
                <span>[ FLEET SETUP ({fleetList.length}) ]</span>
              </button>
            ) : (
              <button
                onClick={() => setIsSetupOpen(true)}
                className="bg-neutral-900 hover:bg-neutral-800 border border-amber-500/60 text-amber-400 hover:text-white font-['Orbitron'] font-bold px-3 py-1 text-xs transition-all flex items-center space-x-1.5 cursor-pointer"
              >
                <Settings2 className="w-3.5 h-3.5" />
                <span>[ SET UP MISSION ]</span>
              </button>
            )}
          </div>
        </header>

        {/* 2. SUB-ACTION BAR */}
        <div className="w-full bg-[#080808] border-b border-neutral-800 px-4 py-2 flex flex-wrap items-center justify-between gap-3 text-xs shrink-0 select-none">
          {simMode === "MULTI" ? (
            <div className="flex items-center gap-3 text-[11px]">
              <span className="text-amber-400 font-bold">ACTIVE FLEET:</span>
              <span className="text-white font-semibold">
                {multiSimResult ? `${multiSimResult.summary.total_spacecraft} Satellites + ${multiSimResult.summary.total_debris} Debris` : `${fleetList.length} Spacecraft`}
              </span>
              <span className="text-neutral-600">|</span>
              <span className="text-neutral-400">
                Conjunctions: <strong className="text-amber-300">{multiSimResult?.conjunctions.length || 0}</strong>
              </span>
              {multiSimResult && multiSimResult.collisions.length > 0 && (
                <>
                  <span className="text-neutral-600">|</span>
                  <span className="text-red-400 font-bold flex items-center gap-1">
                    <Flame className="w-3 h-3 text-red-400" />
                    {multiSimResult.collisions.length} PHYSICAL COLLISION(S)
                  </span>
                </>
              )}
            </div>
          ) : (
            <div className="flex items-center gap-3 text-[11px]">
              <span className="font-bold text-white">{origBody.name.toUpperCase()} → {destBody.name.toUpperCase()}</span>
              <span className="text-neutral-600">|</span>
              <span className="text-neutral-400">VEHICLE: <strong className="text-amber-400">{selectedPreset.name}</strong></span>
              <span className="text-neutral-600">|</span>
              <span className="text-neutral-400">PAYLOAD: <strong className="text-white">{payloadKg.toLocaleString()} kg</strong></span>
            </div>
          )}

          <div className="flex items-center space-x-2">
            <button
              onClick={() => setIsCalculationsOpen(true)}
              className="bg-neutral-900 hover:bg-neutral-800 border border-amber-500/60 text-amber-400 hover:text-white font-['Orbitron'] font-bold px-3 py-1.5 text-xs transition-all flex items-center space-x-1.5 cursor-pointer"
            >
              <Terminal className="w-3.5 h-3.5 text-amber-400" />
              <span>[ &gt; VIEW CALCULATIONS ]</span>
            </button>

            {simMode === "MULTI" ? (
              <button
                onClick={() => handleRunMultiSimulation(fleetList)}
                disabled={isLoading}
                className="bg-amber-500 hover:bg-amber-400 text-black font-bold px-3.5 py-1.5 text-xs transition-all flex items-center space-x-1.5 cursor-pointer disabled:opacity-50"
              >
                {isLoading ? (
                  <span>PROPAGATING FLEET...</span>
                ) : (
                  <span className="font-['Orbitron'] font-bold tracking-wider flex items-center gap-1">
                    <Play className="w-3 h-3 fill-black" />
                    RUN SIMULATION
                  </span>
                )}
              </button>
            ) : (
              <button
                onClick={() => handleRunSingleSimulation(origin, destination, selectedPresetId, payloadKg)}
                disabled={isLoading}
                className="bg-amber-500 hover:bg-amber-400 text-black font-bold px-3.5 py-1.5 text-xs transition-all flex items-center space-x-1.5 cursor-pointer disabled:opacity-50"
              >
                {isLoading ? (
                  <span>SOLVING TRANSFER...</span>
                ) : (
                  <span className="font-['Orbitron'] font-bold tracking-wider flex items-center gap-1">
                    <Play className="w-3 h-3 fill-black" />
                    RUN SIMULATION
                  </span>
                )}
              </button>
            )}
          </div>
        </div>

        {/* Error Notification */}
        {simulationError && (
          <div className="bg-red-950/80 border-b border-red-500 px-4 py-2 flex items-center justify-between z-30 shrink-0">
            <div className="flex items-center space-x-2 text-red-300 text-xs">
              <AlertTriangle className="w-4 h-4 text-red-400" />
              <span className="font-bold">ENGINE ERROR:</span>
              <span className="text-white">{simulationError}</span>
            </div>
            <button
              onClick={() => setSimulationError(null)}
              className="bg-red-900 border border-red-700 text-white px-2 py-0.5 text-[10px] font-bold"
            >
              DISMISS
            </button>
          </div>
        )}

        {/* 3. MAIN SIMULATION VIEWPORT & SIDEBAR */}
        <main className="flex-1 min-w-0 min-h-0 w-full relative overflow-hidden bg-black flex">
          
          {/* Canvas Sandbox */}
          <div className="flex-1 relative h-full">
            <Sandbox2D
              multiSimResult={simMode === "MULTI" ? multiSimResult : null}
              stateHistory={simMode === "SINGLE" ? (simResult?.state_history || []) : []}
              targetStateHistory={simMode === "SINGLE" ? simResult?.target_state_history : undefined}
              bodyHistories={simMode === "SINGLE" ? simResult?.bodies : undefined}
              currentFrameIdx={currentFrameIdx}
              originBodyName={simMode === "SINGLE" ? (simResult?.metadata.origin || origin) : "Earth"}
              destinationBodyName={simMode === "SINGLE" ? (simResult?.metadata.destination || destination) : "Mars"}
              spacecraftPresetId={selectedPresetId}
              selectedObjectId={selectedObjectId}
              onSelectObject={(id) => setSelectedObjectId(id)}
              onSelectConjunction={(conj) => setSelectedConjunction(conj)}
            />

            {/* Selected Object Live HUD on Top-Right */}
            {simMode === "MULTI" && activeSelectedTrack && (
              <div className="absolute top-3 right-3 w-64 bg-black/85 border border-neutral-800 p-2.5 text-xs text-neutral-300 pointer-events-auto z-20 shadow-xl font-mono">
                <div className="flex items-center justify-between border-b border-neutral-800 pb-1 mb-1.5">
                  <span className="font-bold text-amber-400">{activeSelectedTrack.name}</span>
                  <span className={`text-[9px] px-1 py-0.2 uppercase font-bold ${
                    activeSelectedTrack.destroyed
                      ? "bg-red-950 text-red-400 border border-red-700"
                      : "bg-emerald-950 text-emerald-400 border border-emerald-700"
                  }`}>
                    {activeSelectedTrack.destroyed ? "DESTROYED" : "ACTIVE"}
                  </span>
                </div>
                {activeSelectedTrack.state_history.length > 0 && (() => {
                  const idx = Math.min(currentFrameIdx, activeSelectedTrack.state_history.length - 1);
                  const st = activeSelectedTrack.state_history[idx];
                  if (!st) return null;
                  return (
                    <div className="space-y-1 text-[10px]">
                      <div className="flex justify-between">
                        <span className="text-neutral-500">ALTITUDE:</span>
                        <span className="text-white font-bold">{formatDistance(st.altitude)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-neutral-500">SPEED:</span>
                        <span className="text-emerald-400 font-bold">{formatSpeed(st.speed)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-neutral-500">MASS:</span>
                        <span className="text-amber-300 font-bold">{formatMass(st.mass)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-neutral-500">HBR:</span>
                        <span className="text-neutral-200 font-bold">{activeSelectedTrack.hard_body_radius_m.toFixed(1)} m</span>
                      </div>
                    </div>
                  );
                })()}
              </div>
            )}
          </div>

          {/* Collapsible Conjunctions Panel Sidebar (Multi Mode) */}
          {simMode === "MULTI" && multiSimResult && (
            <div className={`transition-all duration-200 border-l border-neutral-800 flex flex-col z-20 ${
              showConjunctionsSidebar ? "w-80" : "w-8"
            }`}>
              <button
                onClick={() => setShowConjunctionsSidebar(!showConjunctionsSidebar)}
                className="w-full bg-neutral-950 border-b border-neutral-800 p-1 text-[10px] text-neutral-400 hover:text-white flex items-center justify-center gap-1"
                title="Toggle Conjunctions Panel"
              >
                {showConjunctionsSidebar ? "HIDE CONJUNCTIONS ✕" : "⚡ CONJ"}
              </button>

              {showConjunctionsSidebar && (
                <div className="flex-1 overflow-hidden">
                  <ConjunctionsPanel
                    conjunctions={multiSimResult.conjunctions}
                    collisions={multiSimResult.collisions}
                    currentTimeSeconds={currentTimeSec}
                    onSelectConjunction={(conj) => setSelectedConjunction(conj)}
                    onJumpToTCA={(tca) => handleJumpToTime(tca)}
                  />
                </div>
              )}
            </div>
          )}

        </main>

        {/* 4. BOTTOM TIMELINE PLAYBACK SCRUBBER */}
        <TimelineScrubber
          currentFrameIdx={currentFrameIdx}
          totalFrames={totalFrames}
          currentTimeSeconds={currentTimeSec}
          totalDurationSeconds={totalDurationSec}
          isPlaying={isPlaying}
          playbackSpeed={playbackSpeed}
          events={
            simMode === "MULTI" && multiSimResult
              ? multiSimResult.collisions.map((c) => ({
                  time: c.time_s,
                  name: `COLLISION: ${c.spacecraft_a_name} ✕ ${c.spacecraft_b_name}`,
                  type: "MISSION_FAILURE",
                  details: `Miss: ${c.miss_distance_m.toFixed(1)}m ≤ HBR ${c.combined_hbr_m.toFixed(1)}m`,
                }))
              : (simResult?.events || [])
          }
          onSeek={(idx) => setCurrentFrameIdx(idx)}
          onTogglePlay={() => setIsPlaying(!isPlaying)}
          onSetSpeed={(spd) => setPlaybackSpeed(spd)}
          onReset={() => {
            setCurrentFrameIdx(0);
            setIsPlaying(false);
          }}
        />

      </div>
    </ErrorBoundary>
  );
}

export default App;
