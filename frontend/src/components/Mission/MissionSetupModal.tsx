import React, { useState } from "react";
import { ActiveRocket, RocketPreset } from "../../types/mission";
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
  Compass,
  X,
  CheckCircle2,
  Calendar,
  Layers,
  Settings2,
  Target
} from "lucide-react";

interface MissionSetupModalProps {
  isOpen: boolean;
  onClose: () => void;
  onInitializeMission: (config: {
    origin: string;
    destination: string;
    presetId: string;
    payloadKg: number;
    missionType: string;
    epochDate: string;
    enableJ2: boolean;
    enableDrag: boolean;
    enableSrp: boolean;
    collisionEnabled: boolean;
    collisionTargetId: string | null;
  }) => void;
  currentOrigin: string;
  currentDestination: string;
  currentPresetId: string;
  currentPayloadKg: number;
  activeRockets?: ActiveRocket[];
}

export const MissionSetupModal: React.FC<MissionSetupModalProps> = ({
  isOpen,
  onClose,
  onInitializeMission,
  currentOrigin,
  currentDestination,
  currentPresetId,
  currentPayloadKg,
  activeRockets = [],
}) => {
  const [origin, setOrigin] = useState<string>(currentOrigin || "earth");
  const [destination, setDestination] = useState<string>(currentDestination || "mars");
  const [selectedPresetId, setSelectedPresetId] = useState<string>(currentPresetId || "isro-lvm3");
  const [payloadKg, setPayloadKg] = useState<number>(currentPayloadKg || 2500);
  const [missionType, setMissionType] = useState<string>("lambert");
  const [epochDate, setEpochDate] = useState<string>("2026-08-18");
  
  // Advanced physics flags
  const [enableJ2, setEnableJ2] = useState<boolean>(true);
  const [enableDrag, setEnableDrag] = useState<boolean>(false);
  const [enableSrp, setEnableSrp] = useState<boolean>(true);

  // Collision Setup state
  const [collisionEnabled, setCollisionEnabled] = useState<boolean>(false);
  const [collisionTargetId, setCollisionTargetId] = useState<string | null>(null);

  if (!isOpen) return null;

  const selectedPreset = ROCKET_PRESETS.find((p) => p.id === selectedPresetId) || ROCKET_PRESETS[0];

  // Real-time Delta-V estimation
  const m0 = selectedPreset.dry_mass_kg + payloadKg + selectedPreset.propellant_mass_kg;
  const mf = selectedPreset.dry_mass_kg + payloadKg;
  const ve = selectedPreset.specific_impulse_s * 9.80665;
  const dvAvail = mf > 0 && m0 > mf ? ve * Math.log(m0 / mf) : 0;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onInitializeMission({
      origin,
      destination,
      presetId: selectedPresetId,
      payloadKg,
      missionType,
      epochDate,
      enableJ2,
      enableDrag,
      enableSrp,
      collisionEnabled,
      collisionTargetId: collisionEnabled ? collisionTargetId : null,
    });
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 select-none font-mono">
      <div className="w-full max-w-3xl bg-[#080808] border border-[#262626] rounded shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        
        {/* Modal Header */}
        <div className="bg-[#0f0f0f] border-b border-[#222222] px-5 py-3 flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <Compass className="w-4 h-4 text-[#ff9900]" />
            <span className="font-['Orbitron'] font-extrabold tracking-wider text-sm text-[#ffffff]">
              MISSION SETUP & FLIGHT DYNAMICS CONFIGURATION
            </span>
          </div>
          <button
            onClick={onClose}
            className="text-[#777777] hover:text-[#ffffff] p-1 rounded hover:bg-[#1a1a1a]"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Body */}
        <form onSubmit={handleSubmit} className="flex-1 overflow-y-auto p-5 space-y-5 text-xs text-[#f0eee9]">
          
          {/* Section 1: Celestial Boundaries & Mission Type */}
          <div className="space-y-2">
            <div className="text-[10.5px] text-[#ff9900] font-bold tracking-wider uppercase flex items-center space-x-1.5">
              <Globe2 className="w-3.5 h-3.5" />
              <span>01. CELESTIAL BOUNDARIES & ARCHETYPE</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 bg-[#050505] p-3 rounded border border-[#1c1c1c]">
              
              {/* Origin */}
              <div>
                <label className="block text-[#888888] text-[10px] mb-1 font-semibold">ORIGIN BODY</label>
                <select
                  value={origin}
                  onChange={(e) => setOrigin(e.target.value)}
                  className="w-full bg-[#0d0d0d] text-[#ffffff] border border-[#282828] p-1.5 rounded focus:border-[#ff9900] outline-none text-xs"
                >
                  {Object.keys(CELESTIAL_BODIES).map((k) => (
                    <option key={k} value={k}>
                      {CELESTIAL_BODIES[k].name} ({CELESTIAL_BODIES[k].radius_km.toLocaleString()} km)
                    </option>
                  ))}
                </select>
              </div>

              {/* Destination */}
              <div>
                <label className="block text-[#888888] text-[10px] mb-1 font-semibold">DESTINATION / TARGET</label>
                <select
                  value={destination}
                  onChange={(e) => setDestination(e.target.value)}
                  className="w-full bg-[#0d0d0d] text-[#ffffff] border border-[#282828] p-1.5 rounded focus:border-[#ff9900] outline-none text-xs"
                >
                  {Object.keys(CELESTIAL_BODIES).map((k) => (
                    <option key={k} value={k}>
                      {CELESTIAL_BODIES[k].name} ({CELESTIAL_BODIES[k].parent ? `around ${CELESTIAL_BODIES[k].parent}` : "Heliocentric"})
                    </option>
                  ))}
                </select>
              </div>

              {/* Mission Archetype */}
              <div>
                <label className="block text-[#888888] text-[10px] mb-1 font-semibold">MISSION ARCHETYPE</label>
                <select
                  value={missionType}
                  onChange={(e) => setMissionType(e.target.value)}
                  className="w-full bg-[#0d0d0d] text-[#ffffff] border border-[#282828] p-1.5 rounded focus:border-[#ff9900] outline-none text-xs"
                >
                  <option value="lambert">Interplanetary Transfer (Universal Variable Lambert)</option>
                  <option value="hohmann">Planetary / Orbital Transfer (Hohmann / Plane Change)</option>
                  <option value="rendezvous">Orbital Rendezvous & Target Interception</option>
                </select>
              </div>

            </div>
          </div>

          {/* Section 2: Launch Vehicle vs Spacecraft Propulsion Stage */}
          <div className="space-y-2">
            <div className="text-[10.5px] text-[#ff9900] font-bold tracking-wider uppercase flex items-center space-x-1.5">
              <Rocket className="w-3.5 h-3.5" />
              <span>02. VEHICLE ARCHITECTURE & PROPULSION SPECIFICATION</span>
            </div>

            <div className="bg-[#050505] p-3 rounded border border-[#1c1c1c] space-y-3">
              
              <div>
                <label className="block text-[#888888] text-[10px] mb-1.5 font-semibold tracking-wider uppercase">
                  SELECT VEHICLE ARCHITECTURE / PROPULSION STAGE
                </label>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-48 overflow-y-auto pr-1">
                  {ROCKET_PRESETS.map((p) => (
                    <button
                      type="button"
                      key={p.id}
                      onClick={() => setSelectedPresetId(p.id)}
                      className={`p-2.5 rounded text-left border text-xs transition-all flex items-center justify-between cursor-pointer ${
                        selectedPresetId === p.id
                          ? "bg-[#ff9900]/15 border-[#ff9900] text-[#ffffff] shadow-md"
                          : "bg-[#0d0d0d] border-[#222222] text-[#aaaaaa] hover:border-[#444444]"
                      }`}
                    >
                      <div className="flex items-center space-x-2.5">
                        <div className={`w-7 h-7 rounded flex items-center justify-center font-bold text-[10px] ${
                          selectedPresetId === p.id ? "bg-[#ff9900] text-black" : "bg-neutral-800 text-neutral-300"
                        }`}>
                          <Rocket className="w-4 h-4" />
                        </div>
                        <div>
                          <div className="font-bold text-[11px] text-[#ffffff] flex items-center gap-1.5">
                            <span>{p.name}</span>
                            <span className="text-[9px] px-1 py-0.2 bg-neutral-900 border border-neutral-700 text-amber-400 font-semibold rounded">
                              {p.operator || "REAL"}
                            </span>
                          </div>
                          <div className="text-[9.5px] text-[#888888]">{p.category}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-[10px] text-[#44bb66] font-bold">Isp {p.specific_impulse_s}s</div>
                        <div className="text-[9px] text-[#666666]">{(p.max_thrust_n / 1000).toFixed(0)} kN</div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Selected Vehicle Technical Details & Provenance */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[10px] bg-[#0d0d0d] p-3 rounded border border-[#1f1f1f]">
                <div>
                  <span className="text-[#777777] block text-[9px]">VEHICLE DRY MASS:</span>
                  <div className="text-[#f0eee9] font-bold text-xs">{selectedPreset.dry_mass_kg.toLocaleString()} kg</div>
                </div>
                <div>
                  <span className="text-[#777777] block text-[9px]">MAX PROPELLANT LOAD:</span>
                  <div className="text-[#ff9900] font-bold text-xs">{selectedPreset.propellant_mass_kg.toLocaleString()} kg</div>
                </div>
                <div>
                  <span className="text-[#777777] block text-[9px]">VACUUM SPECIFIC IMPULSE:</span>
                  <div className="text-[#44bb66] font-bold text-xs">{selectedPreset.specific_impulse_s} s</div>
                </div>
                <div>
                  <span className="text-[#777777] block text-[9px]">MAXIMUM THRUST:</span>
                  <div className="text-[#cccccc] font-bold text-xs">{(selectedPreset.max_thrust_n / 1000).toFixed(1)} kN</div>
                </div>
              </div>

            </div>
          </div>

          {/* Section 3: Payload & Mission Epoch */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            
            {/* Payload Mass */}
            <div className="space-y-1.5 bg-[#050505] p-3 rounded border border-[#1c1c1c]">
              <div className="flex justify-between items-center">
                <span className="text-[10.5px] text-[#ff9900] font-bold uppercase">PAYLOAD MASS</span>
                <span className="text-xs font-bold text-[#ffffff] bg-[#0f0f0f] border border-[#282828] px-2 py-0.5 rounded">
                  {payloadKg.toLocaleString()} kg
                </span>
              </div>
              <input
                type="range"
                min="100"
                max="25000"
                step="100"
                value={payloadKg}
                onChange={(e) => setPayloadKg(Number(e.target.value))}
                className="w-full accent-[#ff9900]"
              />
              <div className="flex justify-between text-[9px] text-[#666666]">
                <span>100 kg (Cubesat)</span>
                <span>2,500 kg (Standard Probe)</span>
                <span>25,000 kg (Crew)</span>
              </div>
            </div>

            {/* Mission Epoch */}
            <div className="space-y-1.5 bg-[#050505] p-3 rounded border border-[#1c1c1c]">
              <div className="flex justify-between items-center">
                <span className="text-[10.5px] text-[#ff9900] font-bold uppercase flex items-center space-x-1">
                  <Calendar className="w-3 h-3" />
                  <span>MISSION DEPARTURE EPOCH</span>
                </span>
              </div>
              <input
                type="date"
                value={epochDate}
                onChange={(e) => setEpochDate(e.target.value)}
                className="w-full bg-[#0d0d0d] text-[#ffffff] border border-[#282828] p-1.5 rounded focus:border-[#ff9900] outline-none text-xs"
              />
              <div className="text-[9px] text-[#666666]">
                Ephemeris coordinates determined at UTC departure epoch.
              </div>
            </div>

          </div>

          {/* Section 4: Physics & Numerical Integrator */}
          <div className="space-y-2">
            <div className="text-[10.5px] text-[#ff9900] font-bold tracking-wider uppercase flex items-center space-x-1.5">
              <Settings2 className="w-3.5 h-3.5" />
              <span>03. FORCE MODELS & NUMERICAL INTEGRATOR</span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 bg-[#050505] p-3 rounded border border-[#1c1c1c] text-[10px]">
              <label className="flex items-center space-x-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={true}
                  disabled
                  className="accent-[#ff9900]"
                />
                <span className="text-[#ffffff]">Point-Mass 2-Body</span>
              </label>

              <label className="flex items-center space-x-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={enableJ2}
                  onChange={(e) => setEnableJ2(e.target.checked)}
                  className="accent-[#ff9900]"
                />
                <span className="text-[#aaaaaa]">J2 Oblateness</span>
              </label>

              <label className="flex items-center space-x-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={enableDrag}
                  onChange={(e) => setEnableDrag(e.target.checked)}
                  className="accent-[#ff9900]"
                />
                <span className="text-[#aaaaaa]">US76 Drag</span>
              </label>

              <label className="flex items-center space-x-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={enableSrp}
                  onChange={(e) => setEnableSrp(e.target.checked)}
                  className="accent-[#ff9900]"
                />
                <span className="text-[#aaaaaa]">Solar Radiation (SRP)</span>
              </label>
            </div>
          </div>

          {/* Section 5: Target Rocket Collision Configuration */}
          <div className="space-y-2">
            <div className="text-[10.5px] text-[#ff9900] font-bold tracking-wider uppercase flex items-center space-x-1.5">
              <Target className="w-3.5 h-3.5 text-[#ff3333]" />
              <span>04. TARGETED ROCKET COLLISION SETUP</span>
            </div>

            <div className="bg-[#050505] p-3 rounded border border-[#1c1c1c] space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-xs text-[#ffffff] font-semibold flex items-center space-x-2">
                  <Flame className="w-3.5 h-3.5 text-[#ff3333]" />
                  <span>ENABLE ROCKET COLLISION MODE</span>
                </label>
                <button
                  type="button"
                  onClick={() => {
                    const next = !collisionEnabled;
                    setCollisionEnabled(next);
                    if (next && activeRockets.length > 0) {
                      setCollisionTargetId(activeRockets[0].id);
                    } else {
                      setCollisionTargetId(null);
                    }
                  }}
                  className={`px-3 py-1 text-xs font-bold rounded border tracking-wider transition-colors cursor-pointer ${
                    collisionEnabled
                      ? "bg-[#ff3333]/20 text-[#ff4444] border-[#ff3333]"
                      : "bg-[#0d0d0d] text-[#888888] border-[#222222] hover:text-[#ffffff]"
                  }`}
                >
                  COLLISION: {collisionEnabled ? "ON" : "OFF"}
                </button>
              </div>

              {collisionEnabled && (
                <div>
                  <label className="block text-[#888888] text-[10px] mb-1 font-semibold">
                    SELECT COLLISION TARGET ROCKET
                  </label>
                  {activeRockets.length > 0 ? (
                    <select
                      value={collisionTargetId || ""}
                      onChange={(e) => setCollisionTargetId(e.target.value || null)}
                      className="w-full bg-[#0d0d0d] text-[#ffffff] border border-[#ff3333]/60 p-2 rounded focus:border-[#ff3333] outline-none text-xs font-mono"
                    >
                      {activeRockets.map((r) => (
                        <option key={r.id} value={r.id}>
                          {r.name} ({r.origin.toUpperCase()} → {r.destination.toUpperCase()})
                        </option>
                      ))}
                    </select>
                  ) : (
                    <div className="text-[11px] text-[#ff9900] bg-[#1a1000] p-2 rounded border border-[#664400]">
                      NO ACTIVE ROCKETS EXIST YET. LAUNCH ROCKET 1 FIRST, THEN LAUNCH ROCKET 2 WITH COLLISION ON TARGETING ROCKET 1.
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Delta-V Capability Preview */}
          <div className="bg-[#0d140d] border border-[#44bb66]/40 p-3 rounded flex items-center justify-between">
            <div className="space-y-0.5">
              <div className="text-[10px] text-[#888888] font-bold">TOTAL THEORETICAL VEHICLE CAPACITY</div>
              <div className="text-sm font-bold text-[#44bb66]">
                {(dvAvail / 1000).toFixed(3)} km/s Δv (Total Wet Mass: {(m0 / 1000).toFixed(2)} t)
              </div>
            </div>
            <div className="text-right text-[9px] text-[#777777]">
              Tsiolkovsky Equation: Δv = Isp·g₀·ln(m₀/m_f)
            </div>
          </div>

          {/* Modal Footer CTA */}
          <div className="pt-2 flex justify-end space-x-3 border-t border-[#1c1c1c]">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 bg-[#141414] hover:bg-[#1f1f1f] text-[#aaaaaa] hover:text-[#ffffff] rounded text-xs"
            >
              CANCEL
            </button>
            <button
              type="submit"
              className="px-5 py-2 bg-[#ff9900] hover:bg-[#ffaa22] text-[#000000] font-['Orbitron'] font-extrabold rounded text-xs tracking-wider flex items-center space-x-2 shadow-lg cursor-pointer"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>INITIALIZE & SOLVE MISSION</span>
            </button>
          </div>

        </form>

      </div>
    </div>
  );
};
