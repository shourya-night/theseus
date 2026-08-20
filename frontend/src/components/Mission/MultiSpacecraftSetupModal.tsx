import React, { useState } from "react";
import { SpacecraftConfig } from "../../types/mission";
import { ROCKET_PRESETS } from "../../data/rocketPresets";
import { CELESTIAL_BODIES } from "../../data/celestialCatalog";
import { 
  X, 
  Plus, 
  Trash2, 
  Rocket, 
  Layers, 
  AlertTriangle, 
  Settings2, 
  ChevronDown, 
  ChevronUp,
  Sparkles,
  Flame,
  Play,
  Globe,
  Compass
} from "lucide-react";

interface MultiSpacecraftSetupModalProps {
  isOpen: boolean;
  onClose: () => void;
  spacecraftList: SpacecraftConfig[];
  onUpdateSpacecraftList: (list: SpacecraftConfig[]) => void;
  onLaunchAll: (list: SpacecraftConfig[]) => void;
  isLoading: boolean;
}

const DEFAULT_PRESETS_SCENARIOS: { name: string; description: string; type: "interplanetary" | "orbital"; list: SpacecraftConfig[] }[] = [
  {
    name: "Interplanetary Armada (4 Probes: Mars, Jupiter, Venus)",
    description: "Four independent spacecraft departing Earth for Mars, Jupiter, and Venus simultaneously with moving planetary target solutions.",
    type: "interplanetary",
    list: [
      {
        id: "SC-01",
        name: "Explorer-01",
        vehicle_type: "falcon-heavy",
        color: "#ff9900",
        sprite_id: "falcon_heavy",
        origin: "Earth",
        destination: "Mars",
        payload_mass_kg: 2500.0,
        tof_days: 259.0,
        dry_mass_kg: 3000.0,
        fuel_mass_kg: 6000.0,
        cross_section_area_m2: 15.0,
        drag_coefficient: 2.2,
        reflectivity_coefficient: 1.5,
        thrust_n: 800.0,
        specific_impulse_s: 340.0,
        central_body: "Sun",
        hard_body_radius_m: 10.0,
        sigma_pos_m: [100.0, 100.0, 100.0],
        sigma_vel_m_s: [0.1, 0.1, 0.1],
      },
      {
        id: "SC-02",
        name: "Explorer-02",
        vehicle_type: "starship",
        color: "#e6dfd5",
        sprite_id: "starship",
        origin: "Earth",
        destination: "Mars",
        payload_mass_kg: 1200.0,
        tof_days: 259.0,
        dry_mass_kg: 5000.0,
        fuel_mass_kg: 10000.0,
        cross_section_area_m2: 30.0,
        drag_coefficient: 2.2,
        reflectivity_coefficient: 1.5,
        thrust_n: 1500.0,
        specific_impulse_s: 380.0,
        central_body: "Sun",
        hard_body_radius_m: 15.0,
        sigma_pos_m: [100.0, 100.0, 100.0],
        sigma_vel_m_s: [0.1, 0.1, 0.1],
      },
      {
        id: "SC-03",
        name: "Explorer-03 (Jupiter Deep)",
        vehicle_type: "saturn-v",
        color: "#3388ff",
        sprite_id: "saturn5",
        origin: "Earth",
        destination: "Jupiter",
        payload_mass_kg: 500.0,
        tof_days: 998.0,
        dry_mass_kg: 4000.0,
        fuel_mass_kg: 8000.0,
        cross_section_area_m2: 20.0,
        drag_coefficient: 2.2,
        reflectivity_coefficient: 1.5,
        thrust_n: 1000.0,
        specific_impulse_s: 420.0,
        central_body: "Sun",
        hard_body_radius_m: 12.0,
        sigma_pos_m: [100.0, 100.0, 100.0],
        sigma_vel_m_s: [0.1, 0.1, 0.1],
      },
      {
        id: "SC-04",
        name: "Explorer-04 (Venus Probe)",
        vehicle_type: "atlas-v",
        color: "#00ffcc",
        sprite_id: "atlas",
        origin: "Earth",
        destination: "Venus",
        payload_mass_kg: 800.0,
        tof_days: 146.0,
        dry_mass_kg: 2000.0,
        fuel_mass_kg: 3500.0,
        cross_section_area_m2: 10.0,
        drag_coefficient: 2.2,
        reflectivity_coefficient: 1.5,
        thrust_n: 600.0,
        specific_impulse_s: 320.0,
        central_body: "Sun",
        hard_body_radius_m: 8.0,
        sigma_pos_m: [100.0, 100.0, 100.0],
        sigma_vel_m_s: [0.1, 0.1, 0.1],
      },
    ],
  },
  {
    name: "LEO Conjunction & Physical Collision",
    description: "Two satellites in 400km LEO on intersecting inclination planes that physically collide at TCA.",
    type: "orbital",
    list: [
      {
        id: "SC-01",
        name: "Explorer-01",
        vehicle_type: "falcon9",
        color: "#ff9900",
        sprite_id: "falcon9",
        origin: "Earth",
        destination: "Orbit",
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
        origin: "Earth",
        destination: "Orbit",
        dry_mass_kg: 3500.0,
        fuel_mass_kg: 1200.0,
        cross_section_area_m2: 15.0,
        drag_coefficient: 2.2,
        reflectivity_coefficient: 1.5,
        thrust_n: 0.0,
        specific_impulse_s: 300.0,
        central_body: "Earth",
        semi_major_axis_km: 6778.137, // Same altitude intersecting plane
        eccentricity: 0.0,
        inclination_deg: -51.6,
        raan_deg: 0.0,
        arg_periapsis_deg: 0.0,
        true_anomaly_deg: 0.0005, // Intersects exactly at node with miss < 8m
        hard_body_radius_m: 10.0,
        sigma_pos_m: [80.0, 80.0, 80.0],
        sigma_vel_m_s: [0.08, 0.08, 0.08],
      },
    ],
  },
  {
    name: "5-Satellite Constellation Environment",
    description: "Five diverse orbital assets in independent orbital slots.",
    type: "orbital",
    list: [
      {
        id: "SC-01",
        name: "Falcon-9 Primary",
        vehicle_type: "falcon9",
        color: "#ff9900",
        sprite_id: "falcon9",
        dry_mass_kg: 2500.0,
        fuel_mass_kg: 1000.0,
        cross_section_area_m2: 12.0,
        drag_coefficient: 2.2,
        reflectivity_coefficient: 1.5,
        thrust_n: 0.0,
        specific_impulse_s: 300.0,
        central_body: "Earth",
        semi_major_axis_km: 6878.137,
        eccentricity: 0.002,
        inclination_deg: 28.5,
        raan_deg: 0.0,
        arg_periapsis_deg: 0.0,
        true_anomaly_deg: 0.0,
        hard_body_radius_m: 6.0,
        sigma_pos_m: [100.0, 100.0, 100.0],
        sigma_vel_m_s: [0.1, 0.1, 0.1],
      },
      {
        id: "SC-02",
        name: "Starship Cargo",
        vehicle_type: "starship",
        color: "#e6dfd5",
        sprite_id: "starship",
        dry_mass_kg: 120000.0,
        fuel_mass_kg: 50000.0,
        cross_section_area_m2: 65.0,
        drag_coefficient: 2.4,
        reflectivity_coefficient: 1.8,
        thrust_n: 0.0,
        specific_impulse_s: 380.0,
        central_body: "Earth",
        semi_major_axis_km: 6928.137,
        eccentricity: 0.001,
        inclination_deg: 45.0,
        raan_deg: 60.0,
        arg_periapsis_deg: 30.0,
        true_anomaly_deg: 90.0,
        hard_body_radius_m: 15.0,
        sigma_pos_m: [150.0, 150.0, 150.0],
        sigma_vel_m_s: [0.15, 0.15, 0.15],
      },
      {
        id: "SC-03",
        name: "Polar Observer",
        vehicle_type: "electron",
        color: "#00ffcc",
        sprite_id: "electron",
        dry_mass_kg: 350.0,
        fuel_mass_kg: 150.0,
        cross_section_area_m2: 2.5,
        drag_coefficient: 2.2,
        reflectivity_coefficient: 1.5,
        thrust_n: 0.0,
        specific_impulse_s: 310.0,
        central_body: "Earth",
        semi_major_axis_km: 7078.137,
        eccentricity: 0.005,
        inclination_deg: 98.2, // Sun-synchronous
        raan_deg: 120.0,
        arg_periapsis_deg: 90.0,
        true_anomaly_deg: 180.0,
        hard_body_radius_m: 3.0,
        sigma_pos_m: [50.0, 50.0, 50.0],
        sigma_vel_m_s: [0.05, 0.05, 0.05],
      },
    ],
  },
];

export const MultiSpacecraftSetupModal: React.FC<MultiSpacecraftSetupModalProps> = ({
  isOpen,
  onClose,
  spacecraftList,
  onUpdateSpacecraftList,
  onLaunchAll,
  isLoading,
}) => {
  if (!isOpen) return null;

  const [fleet, setFleet] = useState<SpacecraftConfig[]>(
    spacecraftList.length > 0 ? spacecraftList : DEFAULT_PRESETS_SCENARIOS[0].list
  );
  const [expandedId, setExpandedId] = useState<string | null>(fleet[0]?.id || null);
  const [activeTab, setActiveTab] = useState<"fleet" | "presets">("fleet");

  const handleAddSpacecraft = () => {
    const newIdx = fleet.length + 1;
    const newSc: SpacecraftConfig = {
      id: `SC-${String(newIdx).padStart(2, "0")}`,
      name: `Explorer-${String(newIdx).padStart(2, "0")}`,
      vehicle_type: "falcon9",
      color: ["#ff9900", "#00ffcc", "#3388ff", "#e6dfd5", "#ff5533", "#ffcc00"][newIdx % 6],
      sprite_id: "falcon9",
      origin: "Earth",
      destination: "Mars",
      payload_mass_kg: 1000.0,
      tof_days: 259.0,
      dry_mass_kg: 2000.0,
      fuel_mass_kg: 1000.0,
      cross_section_area_m2: 10.0,
      drag_coefficient: 2.2,
      reflectivity_coefficient: 1.5,
      thrust_n: 0.0,
      specific_impulse_s: 300.0,
      central_body: "Earth",
      semi_major_axis_km: 6778.137 + (newIdx - 1) * 100.0,
      eccentricity: 0.0,
      inclination_deg: 28.5 + (newIdx - 1) * 10.0,
      raan_deg: (newIdx - 1) * 45.0,
      arg_periapsis_deg: 0.0,
      true_anomaly_deg: (newIdx - 1) * 60.0,
      hard_body_radius_m: 6.0,
      sigma_pos_m: [100.0, 100.0, 100.0],
      sigma_vel_m_s: [0.1, 0.1, 0.1],
    };
    const updated = [...fleet, newSc];
    setFleet(updated);
    setExpandedId(newSc.id);
  };

  const handleRemoveSpacecraft = (id: string) => {
    if (fleet.length <= 1) return;
    const updated = fleet.filter((sc) => sc.id !== id);
    setFleet(updated);
    if (expandedId === id) {
      setExpandedId(updated[0]?.id || null);
    }
  };

  const handleUpdateSc = (id: string, updates: Partial<SpacecraftConfig>) => {
    const updated = fleet.map((sc) => (sc.id === id ? { ...sc, ...updates } : sc));
    setFleet(updated);
  };

  const handleApplyPreset = (presetList: SpacecraftConfig[]) => {
    setFleet(presetList);
    setExpandedId(presetList[0]?.id || null);
    setActiveTab("fleet");
  };

  const handlePresetRocketSelect = (id: string, presetKey: string) => {
    const preset = ROCKET_PRESETS.find((p) => p.id === presetKey);
    if (!preset) return;
    handleUpdateSc(id, {
      vehicle_type: preset.id,
      sprite_id: preset.sprite_id || preset.id,
      dry_mass_kg: preset.dry_mass_kg,
      fuel_mass_kg: preset.propellant_mass_kg,
      thrust_n: preset.max_thrust_n,
      specific_impulse_s: preset.specific_impulse_s,
      cross_section_area_m2: preset.cross_section_area_m2 || 10.0,
      drag_coefficient: preset.drag_coefficient || 2.2,
    });

  };

  const handleLaunch = () => {
    onUpdateSpacecraftList(fleet);
    onLaunchAll(fleet);
    onClose();
  };

  const isInterplanetaryFleet = fleet.some(
    (sc) => sc.destination && !["earth", "orbit", "target"].includes(sc.destination.toLowerCase())
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 overflow-y-auto font-mono text-xs">
      <div className="relative w-full max-w-4xl bg-[#0a0a0c] border border-neutral-800 shadow-2xl rounded-none flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-neutral-800 bg-neutral-950">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400">
              <Layers className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-neutral-100 font-bold tracking-wider text-sm">SET UP MISSION</span>
                <span className="px-1.5 py-0.5 bg-neutral-800 text-amber-400 text-[10px] uppercase font-bold">
                  {fleet.length} SPACECRAFT CONFIGURED
                </span>
                {isInterplanetaryFleet && (
                  <span className="px-1.5 py-0.5 bg-blue-900/40 text-blue-300 border border-blue-800/60 text-[10px] uppercase font-bold">
                    INTERPLANETARY MODE
                  </span>
                )}
              </div>
              <p className="text-neutral-500 text-[11px]">
                Configure fleet transfer parameters, destination ephemerides, payload budgets, and uncertainty matrices.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-neutral-500 hover:text-neutral-300 p-1.5 hover:bg-neutral-900 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Tab Selector */}
        <div className="flex border-b border-neutral-800 bg-neutral-950 px-6 pt-2">
          <button
            onClick={() => setActiveTab("fleet")}
            className={`px-4 py-2 text-xs font-bold uppercase tracking-wider border-b-2 transition-all ${
              activeTab === "fleet"
                ? "border-amber-400 text-amber-400 bg-amber-400/5"
                : "border-transparent text-neutral-400 hover:text-neutral-200"
            }`}
          >
            SPACECRAFT FLEET ({fleet.length})
          </button>
          <button
            onClick={() => setActiveTab("presets")}
            className={`px-4 py-2 text-xs font-bold uppercase tracking-wider border-b-2 transition-all flex items-center gap-1.5 ${
              activeTab === "presets"
                ? "border-amber-400 text-amber-400 bg-amber-400/5"
                : "border-transparent text-neutral-400 hover:text-neutral-200"
            }`}
          >
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            MISSION PRESETS
          </button>
        </div>

        {/* Body Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {activeTab === "presets" ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {DEFAULT_PRESETS_SCENARIOS.map((preset, idx) => (
                <div
                  key={idx}
                  className="p-4 bg-neutral-950 border border-neutral-800 hover:border-amber-500/50 transition-all flex flex-col justify-between"
                >
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-neutral-200 text-sm tracking-wide">{preset.name}</span>
                      <span className="px-1.5 py-0.5 bg-neutral-900 border border-neutral-700 text-neutral-300 text-[10px] uppercase font-bold">
                        {preset.list.length} CRAFT
                      </span>
                    </div>
                    <p className="text-neutral-400 text-xs leading-relaxed">{preset.description}</p>
                    <div className="flex flex-wrap gap-1.5 pt-1">
                      {preset.list.map((sc) => (
                        <span
                          key={sc.id}
                          className="px-1.5 py-0.5 bg-black border text-[10px] uppercase"
                          style={{ borderColor: sc.color, color: sc.color }}
                        >
                          {sc.name} ({sc.destination || "LEO"})
                        </span>
                      ))}
                    </div>
                  </div>
                  <button
                    onClick={() => handleApplyPreset(preset.list)}
                    className="mt-4 w-full py-2 bg-amber-500/10 hover:bg-amber-500 text-amber-400 hover:text-black border border-amber-500/40 font-bold uppercase tracking-wider transition-all"
                  >
                    LOAD THIS MISSION
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div className="space-y-3">
              {fleet.map((sc, idx) => {
                const isExpanded = expandedId === sc.id;
                const isInterplanetary = sc.destination && !["earth", "orbit", "target"].includes(sc.destination.toLowerCase());

                return (
                  <div
                    key={sc.id}
                    className="border border-neutral-800 bg-neutral-950 transition-all"
                    style={{ borderLeftColor: sc.color, borderLeftWidth: "4px" }}
                  >
                    {/* Spacecraft Header Bar */}
                    <div
                      className="p-3.5 flex items-center justify-between cursor-pointer hover:bg-neutral-900/50 transition-colors"
                      onClick={() => setExpandedId(isExpanded ? null : sc.id)}
                    >
                      <div className="flex items-center gap-3">
                        <div
                          className="w-3 h-3 rounded-none"
                          style={{ backgroundColor: sc.color }}
                        />
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-neutral-100 text-xs tracking-wider uppercase">
                              {sc.name}
                            </span>
                            <span className="text-neutral-500 text-[10px]">[{sc.id}]</span>
                            {isInterplanetary ? (
                              <span className="px-1.5 py-0.2 bg-blue-950/60 text-blue-300 border border-blue-800/40 text-[9px] uppercase font-bold">
                                {sc.origin || "Earth"} → {sc.destination}
                              </span>
                            ) : (
                              <span className="px-1.5 py-0.2 bg-amber-950/60 text-amber-300 border border-amber-800/40 text-[9px] uppercase font-bold">
                                {sc.semi_major_axis_km} km ({sc.inclination_deg}°)
                              </span>
                            )}
                          </div>
                          <div className="text-neutral-500 text-[10px] flex gap-3 pt-0.5">
                            <span>VEHICLE: {sc.vehicle_type.toUpperCase()}</span>
                            <span>PAYLOAD: {sc.payload_mass_kg || 0} kg</span>
                            <span>MASS: {(sc.dry_mass_kg + (sc.payload_mass_kg || 0) + sc.fuel_mass_kg).toLocaleString()} kg</span>
                            <span>HBR: {sc.hard_body_radius_m}m</span>
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        {fleet.length > 1 && (
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleRemoveSpacecraft(sc.id);
                            }}
                            className="p-1.5 text-neutral-500 hover:text-red-400 hover:bg-neutral-900 transition-colors"
                            title="Remove Spacecraft"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        )}
                        {isExpanded ? (
                          <ChevronUp className="w-4 h-4 text-neutral-400" />
                        ) : (
                          <ChevronDown className="w-4 h-4 text-neutral-400" />
                        )}
                      </div>
                    </div>

                    {/* Detailed Spacecraft Editor */}
                    {isExpanded && (
                      <div className="p-4 border-t border-neutral-800 bg-[#0d0d10] space-y-4">
                        {/* Basic Identity & Presets */}
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                          <div>
                            <label className="text-[10px] uppercase text-neutral-400 font-bold block mb-1">
                              SPACECRAFT NAME
                            </label>
                            <input
                              type="text"
                              value={sc.name}
                              onChange={(e) => handleUpdateSc(sc.id, { name: e.target.value })}
                              className="w-full bg-black border border-neutral-700 text-neutral-100 px-2.5 py-1.5 text-xs focus:border-amber-400 outline-none"
                            />
                          </div>

                          <div>
                            <label className="text-[10px] uppercase text-neutral-400 font-bold block mb-1">
                              VEHICLE PRESET
                            </label>
                            <select
                              value={sc.vehicle_type}
                              onChange={(e) => handlePresetRocketSelect(sc.id, e.target.value)}
                              className="w-full bg-black border border-neutral-700 text-neutral-100 px-2 py-1.5 text-xs focus:border-amber-400 outline-none"
                            >
                              {ROCKET_PRESETS.map((p) => (
                                <option key={p.id} value={p.id}>
                                  {p.name}
                                </option>
                              ))}
                            </select>
                          </div>

                          <div>
                            <label className="text-[10px] uppercase text-neutral-400 font-bold block mb-1">
                              TRACK COLOR
                            </label>
                            <div className="flex gap-1.5 items-center">
                              <input
                                type="color"
                                value={sc.color}
                                onChange={(e) => handleUpdateSc(sc.id, { color: e.target.value })}
                                className="w-7 h-7 bg-black border border-neutral-700 p-0.5 cursor-pointer"
                              />
                              <span className="text-neutral-400 text-xs uppercase">{sc.color}</span>
                            </div>
                          </div>

                          <div>
                            <label className="text-[10px] uppercase text-neutral-400 font-bold block mb-1">
                              HARD-BODY RADIUS (m)
                            </label>
                            <input
                              type="number"
                              step="0.5"
                              min="0.5"
                              value={sc.hard_body_radius_m}
                              onChange={(e) =>
                                handleUpdateSc(sc.id, { hard_body_radius_m: parseFloat(e.target.value) || 5.0 })
                              }
                              className="w-full bg-black border border-neutral-700 text-neutral-100 px-2.5 py-1.5 text-xs focus:border-amber-400 outline-none"
                            />
                          </div>
                        </div>

                        {/* Interplanetary Transfer Routing */}
                        <div className="p-3 bg-black/60 border border-neutral-800 space-y-3">
                          <div className="flex items-center gap-2 text-amber-400 font-bold text-[11px] uppercase tracking-wider">
                            <Globe className="w-3.5 h-3.5" />
                            INTERPLANETARY ROUTING & DESTINATION
                          </div>

                          <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                            <div>
                              <label className="text-[10px] uppercase text-neutral-400 font-bold block mb-1">
                                ORIGIN BODY
                              </label>
                              <select
                                value={sc.origin || "Earth"}
                                onChange={(e) => handleUpdateSc(sc.id, { origin: e.target.value })}
                                className="w-full bg-black border border-neutral-700 text-neutral-100 px-2 py-1.5 text-xs focus:border-amber-400 outline-none uppercase"
                              >
                                <option value="Earth">Earth</option>
                                <option value="Mars">Mars</option>
                                <option value="Venus">Venus</option>
                                <option value="Moon">Moon</option>
                              </select>
                            </div>

                            <div>
                              <label className="text-[10px] uppercase text-neutral-400 font-bold block mb-1">
                                DESTINATION BODY
                              </label>
                              <select
                                value={sc.destination || "Mars"}
                                onChange={(e) => handleUpdateSc(sc.id, { destination: e.target.value })}
                                className="w-full bg-black border border-neutral-700 text-neutral-100 px-2 py-1.5 text-xs focus:border-amber-400 outline-none uppercase"
                              >
                                <option value="Mars">Mars</option>
                                <option value="Jupiter">Jupiter</option>
                                <option value="Venus">Venus</option>
                                <option value="Saturn">Saturn</option>
                                <option value="Mercury">Mercury</option>
                                <option value="Orbit">Earth Orbit (LEO)</option>
                              </select>
                            </div>

                            <div>
                              <label className="text-[10px] uppercase text-neutral-400 font-bold block mb-1">
                                PAYLOAD MASS (kg)
                              </label>
                              <input
                                type="number"
                                step="100"
                                min="0"
                                value={sc.payload_mass_kg || 0}
                                onChange={(e) =>
                                  handleUpdateSc(sc.id, { payload_mass_kg: parseFloat(e.target.value) || 0 })
                                }
                                className="w-full bg-black border border-neutral-700 text-neutral-100 px-2.5 py-1.5 text-xs focus:border-amber-400 outline-none"
                              />
                            </div>

                            <div>
                              <label className="text-[10px] uppercase text-neutral-400 font-bold block mb-1">
                                TOF DAYS (OPTIONAL)
                              </label>
                              <input
                                type="number"
                                step="1"
                                placeholder="Auto Hohmann"
                                value={sc.tof_days || ""}
                                onChange={(e) =>
                                  handleUpdateSc(sc.id, {
                                    tof_days: e.target.value ? parseFloat(e.target.value) : undefined,
                                  })
                                }
                                className="w-full bg-black border border-neutral-700 text-neutral-100 px-2.5 py-1.5 text-xs focus:border-amber-400 outline-none"
                              />
                            </div>
                          </div>
                        </div>

                        {/* Keplerian Elements (for orbital modes) */}
                        {!isInterplanetary && (
                          <div className="p-3 bg-black/60 border border-neutral-800 space-y-3">
                            <div className="flex items-center gap-2 text-neutral-300 font-bold text-[11px] uppercase tracking-wider">
                              <Compass className="w-3.5 h-3.5 text-amber-400" />
                              KEPLERIAN ORBITAL ELEMENTS
                            </div>

                            <div className="grid grid-cols-2 md:grid-cols-6 gap-2">
                              <div>
                                <label className="text-[9px] uppercase text-neutral-500 block mb-0.5">
                                  SEMI-MAJOR AXIS (km)
                                </label>
                                <input
                                  type="number"
                                  step="10"
                                  value={sc.semi_major_axis_km || 6778.137}
                                  onChange={(e) =>
                                    handleUpdateSc(sc.id, { semi_major_axis_km: parseFloat(e.target.value) || 6778.137 })
                                  }
                                  className="w-full bg-black border border-neutral-700 px-2 py-1 text-xs text-neutral-200 outline-none"
                                />
                              </div>

                              <div>
                                <label className="text-[9px] uppercase text-neutral-500 block mb-0.5">
                                  ECCENTRICITY (e)
                                </label>
                                <input
                                  type="number"
                                  step="0.001"
                                  min="0"
                                  max="0.99"
                                  value={sc.eccentricity || 0.0}
                                  onChange={(e) =>
                                    handleUpdateSc(sc.id, { eccentricity: parseFloat(e.target.value) || 0.0 })
                                  }
                                  className="w-full bg-black border border-neutral-700 px-2 py-1 text-xs text-neutral-200 outline-none"
                                />
                              </div>

                              <div>
                                <label className="text-[9px] uppercase text-neutral-500 block mb-0.5">
                                  INCLINATION (deg)
                                </label>
                                <input
                                  type="number"
                                  step="0.5"
                                  value={sc.inclination_deg || 0.0}
                                  onChange={(e) =>
                                    handleUpdateSc(sc.id, { inclination_deg: parseFloat(e.target.value) || 0.0 })
                                  }
                                  className="w-full bg-black border border-neutral-700 px-2 py-1 text-xs text-neutral-200 outline-none"
                                />
                              </div>

                              <div>
                                <label className="text-[9px] uppercase text-neutral-500 block mb-0.5">
                                  RAAN Ω (deg)
                                </label>
                                <input
                                  type="number"
                                  step="5"
                                  value={sc.raan_deg || 0.0}
                                  onChange={(e) =>
                                    handleUpdateSc(sc.id, { raan_deg: parseFloat(e.target.value) || 0.0 })
                                  }
                                  className="w-full bg-black border border-neutral-700 px-2 py-1 text-xs text-neutral-200 outline-none"
                                />
                              </div>

                              <div>
                                <label className="text-[9px] uppercase text-neutral-500 block mb-0.5">
                                  ARGP ω (deg)
                                </label>
                                <input
                                  type="number"
                                  step="5"
                                  value={sc.arg_periapsis_deg || 0.0}
                                  onChange={(e) =>
                                    handleUpdateSc(sc.id, { arg_periapsis_deg: parseFloat(e.target.value) || 0.0 })
                                  }
                                  className="w-full bg-black border border-neutral-700 px-2 py-1 text-xs text-neutral-200 outline-none"
                                />
                              </div>

                              <div>
                                <label className="text-[9px] uppercase text-neutral-500 block mb-0.5">
                                  TRUE ANOMALY ν (deg)
                                </label>
                                <input
                                  type="number"
                                  step="5"
                                  value={sc.true_anomaly_deg || 0.0}
                                  onChange={(e) =>
                                    handleUpdateSc(sc.id, { true_anomaly_deg: parseFloat(e.target.value) || 0.0 })
                                  }
                                  className="w-full bg-black border border-neutral-700 px-2 py-1 text-xs text-neutral-200 outline-none"
                                />
                              </div>
                            </div>
                          </div>
                        )}

                        {/* Mass & Propulsion Specs */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-neutral-400">
                          <div>
                            <span className="text-[9px] uppercase text-neutral-500 block mb-0.5">DRY MASS (kg)</span>
                            <input
                              type="number"
                              value={sc.dry_mass_kg}
                              onChange={(e) => handleUpdateSc(sc.id, { dry_mass_kg: parseFloat(e.target.value) || 1000 })}
                              className="w-full bg-black border border-neutral-700 px-2 py-1 text-xs text-neutral-200 outline-none"
                            />
                          </div>

                          <div>
                            <span className="text-[9px] uppercase text-neutral-500 block mb-0.5">FUEL MASS (kg)</span>
                            <input
                              type="number"
                              value={sc.fuel_mass_kg}
                              onChange={(e) => handleUpdateSc(sc.id, { fuel_mass_kg: parseFloat(e.target.value) || 500 })}
                              className="w-full bg-black border border-neutral-700 px-2 py-1 text-xs text-neutral-200 outline-none"
                            />
                          </div>

                          <div>
                            <span className="text-[9px] uppercase text-neutral-500 block mb-0.5">SPECIFIC IMPULSE (s)</span>
                            <input
                              type="number"
                              value={sc.specific_impulse_s}
                              onChange={(e) => handleUpdateSc(sc.id, { specific_impulse_s: parseFloat(e.target.value) || 300 })}
                              className="w-full bg-black border border-neutral-700 px-2 py-1 text-xs text-neutral-200 outline-none"
                            />
                          </div>

                          <div>
                            <span className="text-[9px] uppercase text-neutral-500 block mb-0.5">POSITION 1-SIGMA (m)</span>
                            <input
                              type="number"
                              value={sc.sigma_pos_m ? sc.sigma_pos_m[0] : 100.0}
                              onChange={(e) => {
                                const val = parseFloat(e.target.value) || 100.0;
                                handleUpdateSc(sc.id, { sigma_pos_m: [val, val, val] });
                              }}
                              className="w-full bg-black border border-neutral-700 px-2 py-1 text-xs text-neutral-200 outline-none"
                            />
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}

              <button
                type="button"
                onClick={handleAddSpacecraft}
                className="w-full py-2.5 bg-neutral-950 hover:bg-neutral-900 border border-dashed border-neutral-700 hover:border-amber-400/60 text-neutral-300 hover:text-amber-400 font-bold uppercase tracking-wider flex items-center justify-center gap-2 transition-all"
              >
                <Plus className="w-4 h-4" />
                ADD SPACECRAFT TO MISSION
              </button>
            </div>
          )}
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-neutral-800 bg-neutral-950">
          <div className="flex items-center gap-2 text-neutral-500 text-[11px]">
            <span>TOTAL: {fleet.length} OBJECTS</span>
            <span>•</span>
            <span>CENTRAL: {isInterplanetaryFleet ? "SUN (HELIOCENTRIC)" : "EARTH"}</span>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 border border-neutral-800 hover:bg-neutral-900 text-neutral-300 font-bold uppercase tracking-wider transition-colors"
            >
              CANCEL
            </button>

            <button
              onClick={handleLaunch}
              disabled={isLoading || fleet.length === 0}
              className="px-6 py-2 bg-amber-400 hover:bg-amber-300 disabled:opacity-50 text-black font-bold uppercase tracking-wider flex items-center gap-2 shadow-lg shadow-amber-400/10 transition-all"
            >
              {isLoading ? (
                <>
                  <div className="w-3.5 h-3.5 border-2 border-black border-t-transparent rounded-full animate-spin" />
                  INTEGRATING...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-black" />
                  LAUNCH FLEET ({fleet.length})
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
