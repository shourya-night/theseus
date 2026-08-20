import React, { useState } from "react";
import { RocketPreset, ComplexityMode } from "../../types/mission";
import { ROCKET_PRESETS } from "../../data/rocketPresets";
import { CELESTIAL_BODIES } from "../../data/celestialCatalog";
import { 
  Rocket, 
  Globe2, 
  Sliders, 
  Flame, 
  ArrowRight, 
  Play, 
  Gauge,
  Compass
} from "lucide-react";

interface MissionBuilderProps {
  onRunSimulation: (config: any) => void;
  isLoading: boolean;
}

export const MissionBuilder: React.FC<MissionBuilderProps> = ({
  onRunSimulation,
  isLoading,
}) => {
  const [complexity, setComplexity] = useState<ComplexityMode>("BASIC");
  
  // Mission Configuration State
  const [origin, setOrigin] = useState<string>("earth");
  const [destination, setDestination] = useState<string>("mars");
  const [missionType, setMissionType] = useState<string>("lambert");
  const [selectedPresetId, setSelectedPresetId] = useState<string>("isro-chandrayaan-pm");

  // Spacecraft specs
  const selectedPreset = ROCKET_PRESETS.find((p) => p.id === selectedPresetId) || ROCKET_PRESETS[0];
  const [dryMassKg, setDryMassKg] = useState<number>(selectedPreset.dry_mass_kg);
  const [fuelMassKg, setFuelMassKg] = useState<number>(selectedPreset.propellant_mass_kg);
  const [payloadKg, setPayloadKg] = useState<number>(500);
  const [ispSec, setIspSec] = useState<number>(selectedPreset.specific_impulse_s);
  const [thrustN, setThrustN] = useState<number>(selectedPreset.max_thrust_n);

  // Advanced Orbital State
  const [r1Km, setR1Km] = useState<number>(149597870.7); // 1 AU
  const [r2Km, setR2Km] = useState<number>(227939200.0);  // 1.524 AU Mars
  const [planeChangeDeg, setPlaneChangeDeg] = useState<number>(1.85);
  const [tofHours, setTofHours] = useState<number>(6240); // 260 days

  // Expert Perturbations
  const [enableJ2, setEnableJ2] = useState<boolean>(true);
  const [enableDrag, setEnableDrag] = useState<boolean>(false);
  const [enableSrp, setEnableSrp] = useState<boolean>(true);
  const [dragCd] = useState<number>(2.2);
  const [crossAreaM2, setCrossAreaM2] = useState<number>(selectedPreset.cross_section_area_m2);

  const handlePresetSelect = (preset: RocketPreset) => {
    setSelectedPresetId(preset.id);
    setDryMassKg(preset.dry_mass_kg);
    setFuelMassKg(preset.propellant_mass_kg);
    setIspSec(preset.specific_impulse_s);
    setThrustN(preset.max_thrust_n);
    setCrossAreaM2(preset.cross_section_area_m2);
  };

  const handleOriginChange = (orig: string) => {
    setOrigin(orig);
    updateTrajectoryDefaults(orig, destination);
  };

  const handleDestinationChange = (dest: string) => {
    setDestination(dest);
    updateTrajectoryDefaults(origin, dest);
  };

  const updateTrajectoryDefaults = (orig: string, dest: string) => {
    const origBody = CELESTIAL_BODIES[orig.toLowerCase()];
    const destBody = CELESTIAL_BODIES[dest.toLowerCase()];
    if (!origBody || !destBody) return;

    if (orig.toLowerCase() === "earth" && dest.toLowerCase() === "moon") {
      // Earth-Moon Translunar Injection
      setR1Km(6678.137);
      setR2Km(384400.0);
      setPlaneChangeDeg(0.0);
      setTofHours(119.5);
      setMissionType("hohmann");
    } else if (orig.toLowerCase() === "earth" && dest.toLowerCase() === "earth") {
      // LEO to GEO Transfer
      setR1Km(6678.137);
      setR2Km(42164.0);
      setPlaneChangeDeg(28.5);
      setTofHours(5.27);
      setMissionType("hohmann");
    } else if (dest.toLowerCase() === "target" || missionType === "rendezvous") {
      // LEO Rendezvous
      setR1Km(6778.137);
      setR2Km(6798.137);
      setPlaneChangeDeg(0.0);
      setTofHours(1.0);
      setMissionType("rendezvous");
    } else {
      // Interplanetary Transfer (e.g. Earth -> Mars, Earth -> Venus, Earth -> Jupiter)
      const r1 = origBody.position_km ? Math.sqrt(origBody.position_km[0] ** 2 + origBody.position_km[1] ** 2) : 149597870.7;
      const r2 = destBody.position_km ? Math.sqrt(destBody.position_km[0] ** 2 + destBody.position_km[1] ** 2) : 227939200.0;
      setR1Km(r1);
      setR2Km(r2);

      // Estimate transfer flight time
      const a_tx_m = ((r1 + r2) / 2.0) * 1000.0;
      const mu_sun = 1.32712440018e20;
      const est_tof_s = Math.PI * Math.sqrt(Math.pow(a_tx_m, 3) / mu_sun);
      setTofHours(Number((est_tof_s / 3600.0).toFixed(1)));
      setPlaneChangeDeg(1.85);
      setMissionType("lambert");
    }
  };

  // Tsiolkovsky real-time Delta-V estimation
  const totalMass = dryMassKg + payloadKg + fuelMassKg;
  const emptyMass = dryMassKg + payloadKg;
  const exhaustVel = ispSec * 9.80665;
  const availableDeltaV = emptyMass > 0 && totalMass > emptyMass 
    ? exhaustVel * Math.log(totalMass / emptyMass) 
    : 0;

  const handleLaunch = () => {
    const config = {
      missionType,
      origin,
      destination,
      preset: selectedPreset,
      dry_mass_kg: dryMassKg + payloadKg,
      fuel_mass_kg: fuelMassKg,
      specific_impulse_s: ispSec,
      thrust_n: thrustN,
      r1_km: r1Km,
      r2_km: r2Km,
      plane_change_deg: planeChangeDeg,
      tof_hours: tofHours,
      enable_j2: enableJ2,
      enable_drag: enableDrag,
      enable_srp: enableSrp,
      cd: dragCd,
      area_m2: crossAreaM2,
    };
    onRunSimulation(config);
  };

  return (
    <div className="w-full h-full flex flex-col bg-[#04060a] text-[#e6dfd5] font-mono overflow-y-auto p-4 lg:p-6 space-y-5">
      
      {/* Title & Complexity Selector */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-[#221d17] pb-3">
        <div>
          <div className="flex items-center space-x-2 text-[#ff9900] text-xs font-semibold tracking-wider">
            <Compass className="w-4 h-4" />
            <span>FLIGHT DYNAMICS PARAMETERS</span>
          </div>
          <h1 className="text-base md:text-lg font-bold text-[#e6dfd5] mt-0.5">
            ASTRODYNAMICS MISSION BUILDER
          </h1>
        </div>

        {/* Complexity Mode Selector */}
        <div className="flex items-center space-x-1 bg-[#070d18] border border-[#221d17] p-1 rounded">
          <span className="text-[10px] text-[#8c8275] px-2 font-bold">MODE:</span>
          {(["BASIC", "ADVANCED", "EXPERT"] as ComplexityMode[]).map((mode) => (
            <button
              key={mode}
              onClick={() => setComplexity(mode)}
              className={`px-2.5 py-0.5 text-xs rounded transition-all ${
                complexity === mode
                  ? "bg-[#ff9900]/20 text-[#ff9900] border border-[#ff9900]/50 font-bold"
                  : "text-[#8c8275] hover:text-[#e6dfd5]"
              }`}
            >
              {mode}
            </button>
          ))}
        </div>
      </div>

      {/* Main Configuration Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">

        {/* Panel 1: Celestial Boundaries & Mission Archetype */}
        <div className="space-y-4">
          <div className="technical-panel">
            <div className="panel-header">
              <span className="flex items-center space-x-2 text-[#ff9900]">
                <Globe2 className="w-3.5 h-3.5" />
                <span>01. CELESTIAL BOUNDARIES</span>
              </span>
            </div>

            <div className="p-3.5 space-y-3.5 text-xs">
              {/* Origin */}
              <div>
                <label className="block text-[11px] text-[#8c8275] mb-1 font-semibold">ORIGIN BODY</label>
                <select
                  value={origin}
                  onChange={(e) => handleOriginChange(e.target.value)}
                >
                  {Object.keys(CELESTIAL_BODIES).map((k) => (
                    <option key={k} value={k}>
                      {CELESTIAL_BODIES[k].name} (Radius: {CELESTIAL_BODIES[k].radius_km.toLocaleString()} km)
                    </option>
                  ))}
                </select>
              </div>

              {/* Destination */}
              <div>
                <label className="block text-[11px] text-[#8c8275] mb-1 font-semibold">DESTINATION / TARGET</label>
                <select
                  value={destination}
                  onChange={(e) => handleDestinationChange(e.target.value)}
                >
                  {Object.keys(CELESTIAL_BODIES).map((k) => (
                    <option key={k} value={k}>
                      {CELESTIAL_BODIES[k].name} (μ: {CELESTIAL_BODIES[k].mu.toExponential(2)} m³/s²)
                    </option>
                  ))}
                </select>
              </div>

              {/* Mission Archetype */}
              <div>
                <label className="block text-[11px] text-[#8c8275] mb-1 font-semibold">MISSION ARCHETYPE</label>
                <select
                  value={missionType}
                  onChange={(e) => setMissionType(e.target.value)}
                >
                  <option value="lambert">Interplanetary Transfer (Universal Variable Lambert)</option>
                  <option value="hohmann">Planetary / Orbital Transfer (Hohmann / Plane Change)</option>
                  <option value="rendezvous">Orbital Rendezvous & Target Interception</option>
                </select>
              </div>

              {/* Route Summary */}
              <div className="bg-[#05080f] border border-[#221d17] p-2.5 rounded text-xs space-y-1">
                <div className="flex justify-between items-center text-[#e6dfd5] font-bold">
                  <span className="text-[#ff9900]">{CELESTIAL_BODIES[origin]?.name || origin}</span>
                  <ArrowRight className="w-4 h-4 text-[#8c8275]" />
                  <span className="text-[#44bb66]">{CELESTIAL_BODIES[destination]?.name || destination}</span>
                </div>
                <div className="text-[10px] text-[#8c8275]">
                  Ephemeris coordinates verified via Astropy JPL DE440.
                </div>
              </div>
            </div>
          </div>

          {/* Vehicle Presets Card */}
          <div className="technical-panel">
            <div className="panel-header">
              <span className="flex items-center space-x-2 text-[#ff9900]">
                <Rocket className="w-3.5 h-3.5" />
                <span>02. SPACECRAFT / VEHICLE PRESET</span>
              </span>
            </div>

            <div className="p-3 space-y-2">
              <div className="grid grid-cols-1 gap-1.5 max-h-48 overflow-y-auto pr-1">
                {ROCKET_PRESETS.map((preset) => (
                  <button
                    key={preset.id}
                    onClick={() => handlePresetSelect(preset)}
                    className={`p-2 rounded text-left border transition-all text-xs flex justify-between items-center ${
                      selectedPresetId === preset.id
                        ? "bg-[#ff9900]/15 border-[#ff9900] text-[#e6dfd5]"
                        : "bg-[#0b1424] border-[#221d17] text-[#c8c0b5] hover:border-[#332b22]"
                    }`}
                  >
                    <div>
                      <div className="font-bold text-[11px] text-[#ff9900] truncate">{preset.name}</div>
                      <div className="text-[9px] text-[#8c8275]">{preset.category} • {preset.operator}</div>
                    </div>
                    <span className="text-[10px] text-[#44bb66] font-bold">Isp {preset.specific_impulse_s}s</span>
                  </button>
                ))}
              </div>

              {/* Selected Preset Details */}
              <div className="bg-[#05080f] border border-[#221d17] p-2.5 rounded text-[10px] text-[#8c8275] space-y-1">
                <div className="flex justify-between items-center text-[#ff9900] font-bold">
                  <span>{selectedPreset.name}</span>
                  <span className="text-[8px] bg-[#ff9900]/10 px-1 py-0.5 rounded border border-[#ff9900]/30">
                    {selectedPreset.confidence}
                  </span>
                </div>
                <p className="text-[10px] text-[#c8c0b5]">{selectedPreset.description}</p>
                <p className="text-[9px] text-[#8c8275] italic truncate">Ref: {selectedPreset.citation}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Panel 2: Propulsion & Mass Sliders */}
        <div className="space-y-4">
          <div className="technical-panel">
            <div className="panel-header">
              <span className="flex items-center space-x-2 text-[#ff9900]">
                <Flame className="w-3.5 h-3.5" />
                <span>03. PROPULSION & MASS PROPERTIES</span>
              </span>
            </div>

            <div className="p-3.5 space-y-4">
              
              {/* Payload Mass Slider */}
              <div className="space-y-1">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-[#8c8275] font-semibold">PAYLOAD MASS</span>
                  <span className="bg-[#070d18] border border-[#ff9900]/40 text-[#ff9900] px-2 py-0.5 rounded font-bold">
                    {payloadKg.toLocaleString()} kg
                  </span>
                </div>
                <input
                  type="range"
                  min="50"
                  max="25000"
                  step="50"
                  value={payloadKg}
                  onChange={(e) => setPayloadKg(Number(e.target.value))}
                />
                <div className="flex justify-between text-[9px] text-[#8c8275]">
                  <span>50 kg</span>
                  <span>Scientific instrument payload</span>
                  <span>25,000 kg</span>
                </div>
              </div>

              {/* Spacecraft Dry Mass */}
              <div className="space-y-1">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-[#8c8275] font-semibold">SPACECRAFT DRY MASS</span>
                  <span className="bg-[#070d18] border border-[#221d17] text-[#e6dfd5] px-2 py-0.5 rounded font-bold">
                    {dryMassKg.toLocaleString()} kg
                  </span>
                </div>
                <input
                  type="range"
                  min="200"
                  max="150000"
                  step="100"
                  value={dryMassKg}
                  onChange={(e) => setDryMassKg(Number(e.target.value))}
                />
                <div className="flex justify-between text-[9px] text-[#8c8275]">
                  <span>200 kg</span>
                  <span>Structure & propulsion stage</span>
                  <span>150,000 kg</span>
                </div>
              </div>

              {/* Propellant Load */}
              <div className="space-y-1">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-[#8c8275] font-semibold">PROPELLANT LOAD</span>
                  <span className="bg-[#070d18] border border-[#ff9900]/40 text-[#ff9900] px-2 py-0.5 rounded font-bold">
                    {fuelMassKg.toLocaleString()} kg
                  </span>
                </div>
                <input
                  type="range"
                  min="100"
                  max="500000"
                  step="100"
                  value={fuelMassKg}
                  onChange={(e) => setFuelMassKg(Number(e.target.value))}
                />
                <div className="flex justify-between text-[9px] text-[#8c8275]">
                  <span>100 kg</span>
                  <span>Usable fuel & oxidizer</span>
                  <span>500,000 kg</span>
                </div>
              </div>

              {/* Specific Impulse */}
              <div className="space-y-1">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-[#8c8275] font-semibold">SPECIFIC IMPULSE (Isp)</span>
                  <span className="bg-[#070d18] border border-[#44bb66]/40 text-[#44bb66] px-2 py-0.5 rounded font-bold">
                    {ispSec.toFixed(1)} s
                  </span>
                </div>
                <input
                  type="range"
                  min="200"
                  max="3200"
                  step="5"
                  value={ispSec}
                  onChange={(e) => setIspSec(Number(e.target.value))}
                />
                <div className="flex justify-between text-[9px] text-[#8c8275]">
                  <span>230 s (Hydrazine)</span>
                  <span>453 s (Cryogenic)</span>
                  <span>3100 s (Ion)</span>
                </div>
              </div>

              {/* Live Tsiolkovsky Delta-V Capacity */}
              <div className="bg-[#05080f] border border-[#44bb66]/40 p-3 rounded space-y-1.5">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-[#8c8275] flex items-center space-x-1">
                    <Gauge className="w-3.5 h-3.5 text-[#44bb66]" />
                    <span>AVAILABLE ΔV CAPACITY:</span>
                  </span>
                  <span className="text-sm font-bold text-[#44bb66]">
                    {(availableDeltaV / 1000).toFixed(3)} km/s
                  </span>
                </div>
                <div className="w-full bg-[#070d18] h-2 rounded overflow-hidden border border-[#221d17]">
                  <div 
                    className="bg-[#44bb66] h-full transition-all duration-300"
                    style={{ width: `${Math.min(100, (availableDeltaV / 12000) * 100)}%` }}
                  />
                </div>
                <div className="flex justify-between text-[9px] text-[#8c8275]">
                  <span>Total Mass: {(totalMass / 1000).toFixed(2)} t</span>
                  <span>Tsiolkovsky Law: Δv = Isp·g₀·ln(m₀/m_f)</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Panel 3: Orbital Parameters & Trigger */}
        <div className="space-y-4">
          <div className="technical-panel">
            <div className="panel-header">
              <span className="flex items-center space-x-2 text-[#ff9900]">
                <Sliders className="w-3.5 h-3.5" />
                <span>04. ORBITAL BOUNDARIES</span>
              </span>
            </div>

            <div className="p-3.5 space-y-4">
              
              {/* Departure Radius */}
              <div className="space-y-1">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-[#8c8275] font-semibold">DEPARTURE RADIUS</span>
                  <span className="bg-[#070d18] border border-[#221d17] text-[#ff9900] px-2 py-0.5 rounded font-bold">
                    {r1Km > 1000000 ? `${(r1Km / 149597870.7).toFixed(3)} AU` : `${r1Km.toLocaleString()} km`}
                  </span>
                </div>
                <div className="text-[10px] text-[#8c8275]">
                  {r1Km > 1000000 ? `Heliocentric Radius (${(r1Km / 149597870.7).toFixed(3)} AU)` : `${r1Km.toLocaleString()} km`}
                </div>
              </div>

              {/* Target Radius */}
              <div className="space-y-1">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-[#8c8275] font-semibold">TARGET RADIUS</span>
                  <span className="bg-[#070d18] border border-[#221d17] text-[#ff9900] px-2 py-0.5 rounded font-bold">
                    {r2Km > 1000000 ? `${(r2Km / 149597870.7).toFixed(3)} AU` : `${r2Km.toLocaleString()} km`}
                  </span>
                </div>
                <div className="text-[10px] text-[#8c8275]">
                  {r2Km > 1000000 ? `Heliocentric Radius (${(r2Km / 149597870.7).toFixed(3)} AU)` : `${r2Km.toLocaleString()} km`}
                </div>
              </div>

              {/* Plane Change */}
              <div className="space-y-1">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-[#8c8275] font-semibold">PLANE CHANGE (Δi)</span>
                  <span className="bg-[#070d18] border border-[#221d17] text-[#ff9900] px-2 py-0.5 rounded font-bold">
                    {planeChangeDeg.toFixed(1)}°
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="90"
                  step="0.5"
                  value={planeChangeDeg}
                  onChange={(e) => setPlaneChangeDeg(Number(e.target.value))}
                />
              </div>

              {/* Time of Flight */}
              <div className="space-y-1">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-[#8c8275] font-semibold">TIME OF FLIGHT</span>
                  <span className="bg-[#070d18] border border-[#221d17] text-[#44bb66] px-2 py-0.5 rounded font-bold">
                    {tofHours >= 48 ? `${(tofHours / 24).toFixed(1)} days (${tofHours.toFixed(0)} hrs)` : `${tofHours.toFixed(2)} hrs`}
                  </span>
                </div>
                <input
                  type="range"
                  min="1"
                  max={r2Km > 1000000 ? "30000" : "240"}
                  step="1"
                  value={tofHours}
                  onChange={(e) => setTofHours(Number(e.target.value))}
                />
              </div>

              {/* Expert Perturbations */}
              {complexity === "EXPERT" && (
                <div className="border-t border-[#221d17] pt-2.5 space-y-2 text-xs">
                  <div className="text-[10px] text-[#ff9900] font-bold">FORCE MODELS:</div>
                  <div className="grid grid-cols-3 gap-2">
                    <label className="flex items-center space-x-1.5 cursor-pointer text-[10px]">
                      <input
                        type="checkbox"
                        checked={enableJ2}
                        onChange={(e) => setEnableJ2(e.target.checked)}
                        className="accent-[#ff9900]"
                      />
                      <span>J2 Bulge</span>
                    </label>
                    <label className="flex items-center space-x-1.5 cursor-pointer text-[10px]">
                      <input
                        type="checkbox"
                        checked={enableDrag}
                        onChange={(e) => setEnableDrag(e.target.checked)}
                        className="accent-[#ff9900]"
                      />
                      <span>US76 Drag</span>
                    </label>
                    <label className="flex items-center space-x-1.5 cursor-pointer text-[10px]">
                      <input
                        type="checkbox"
                        checked={enableSrp}
                        onChange={(e) => setEnableSrp(e.target.checked)}
                        className="accent-[#ff9900]"
                      />
                      <span>SRP Flux</span>
                    </label>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* RUN SIMULATION BUTTON */}
          <button
            onClick={handleLaunch}
            disabled={isLoading}
            className="w-full bg-[#ff9900] hover:bg-[#ffaa22] text-[#04060a] font-bold font-['Orbitron'] py-3 px-4 rounded transition-all flex items-center justify-center space-x-2 shadow-[0_0_15px_rgba(255,153,0,0.3)] disabled:opacity-50 text-xs tracking-wider cursor-pointer"
          >
            {isLoading ? (
              <div className="flex items-center space-x-2">
                <span className="w-3.5 h-3.5 border-2 border-[#04060a] border-t-transparent rounded-full animate-spin" />
                <span>SOLVING FLIGHT TRAJECTORY...</span>
              </div>
            ) : (
              <div className="flex items-center space-x-2">
                <Play className="w-3.5 h-3.5 fill-current" />
                <span className="tracking-widest font-extrabold">RUN SIMULATION</span>
              </div>
            )}
          </button>
        </div>

      </div>
    </div>
  );
};
