import React from "react";
import { 
  StateVector, 
  DeltaVBudget, 
  PropellantBudget, 
  SpacecraftTrack,
  MultiConjunctionEvent,
  PhysicalCollisionEvent
} from "../../types/mission";
import { formatDistance, formatSpeed, formatMass } from "../../lib/formatter";
import { 
  Activity, 
  Gauge, 
  Fuel, 
  Weight, 
  Flame, 
  Navigation, 
  Compass,
  AlertTriangle,
  Layers,
  Crosshair,
  Sparkles
} from "lucide-react";

interface TelemetryHUDProps {
  currentFrame?: StateVector;
  deltaVBudget?: DeltaVBudget;
  propellantBudget?: PropellantBudget;
  isCompact?: boolean;
  selectedObject?: SpacecraftTrack | null;
  allObjects?: SpacecraftTrack[];
  onSelectObjectId?: (id: string) => void;
  activeConjunctions?: MultiConjunctionEvent[];
  collisions?: PhysicalCollisionEvent[];
}

export const TelemetryHUD: React.FC<TelemetryHUDProps> = ({
  currentFrame,
  deltaVBudget,
  propellantBudget,
  isCompact = false,
  selectedObject = null,
  allObjects = [],
  onSelectObjectId,
  activeConjunctions = [],
  collisions = [],
}) => {
  // If an object from multi-simulation is selected, derive its current state
  let pos: [number, number, number] = [0, 0, 0];
  let vel: [number, number, number] = [0, 0, 0];
  let alt = 0;
  let speed = 0;
  let mass = 5000;
  let fuel = 2000;
  let isThrust = false;
  let isDestroyed = false;
  let objName = "ACTIVE VEHICLE";
  let objId = "SC-01";
  let isDebris = false;
  let hbr = 10.0;

  if (selectedObject) {
    objId = selectedObject.id;
    objName = selectedObject.name;
    isDebris = selectedObject.is_debris;
    hbr = selectedObject.hard_body_radius_m;
    isDestroyed = selectedObject.destroyed;

    if (selectedObject.state_history.length > 0) {
      const st = selectedObject.state_history[selectedObject.state_history.length - 1];
      if (st) {
        pos = st.position;
        vel = st.velocity;
        alt = st.altitude;
        speed = st.speed;
        mass = st.mass;
        fuel = st.fuel_mass;
        isThrust = st.thrust_active;
        isDestroyed = !!st.destroyed;
      }
    }
  } else if (currentFrame) {
    pos = currentFrame.position;
    vel = currentFrame.velocity;
    alt = currentFrame.altitude;
    speed = currentFrame.speed;
    mass = currentFrame.mass || 5000;
    fuel = currentFrame.fuel_mass || 2000;
    isThrust = currentFrame.thrust_active;
  } else {
    return (
      <div className="w-full h-full flex items-center justify-center p-4 text-neutral-600 font-mono text-xs italic">
        NO ACTIVE TELEMETRY FEED
      </div>
    );
  }

  if (isCompact) {
    return (
      <div className="w-full bg-neutral-950 border border-neutral-800 p-3 font-mono text-xs space-y-2 shadow-lg">
        <div className="flex items-center justify-between border-b border-neutral-800 pb-1.5">
          <div className="flex items-center space-x-1.5 text-amber-400">
            <Activity className="w-3.5 h-3.5 text-amber-400" />
            <span className="font-bold text-[11px] uppercase tracking-wider">{objName}</span>
          </div>
          <span
            className={`text-[9px] px-1.5 py-0.5 font-bold uppercase ${
              isDestroyed
                ? "bg-red-950/80 text-red-400 border border-red-500/60"
                : isThrust
                ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                : "bg-neutral-900 text-neutral-400 border border-neutral-800"
            }`}
          >
            {isDestroyed ? "DESTROYED" : isThrust ? "BURNING" : "ACTIVE"}
          </span>
        </div>

        <div className="space-y-1 text-[11px]">
          <div className="flex justify-between items-center">
            <span className="text-neutral-500">ALTITUDE:</span>
            <span className="text-white font-bold">{formatDistance(alt)}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-neutral-500">VELOCITY:</span>
            <span className="text-emerald-400 font-bold">{formatSpeed(speed)}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-neutral-500">MASS:</span>
            <span className="text-amber-300 font-bold">{formatMass(mass)}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-neutral-500">HBR:</span>
            <span className="text-neutral-200 font-bold">{hbr.toFixed(1)} m</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full flex flex-col bg-black text-neutral-200 font-mono overflow-y-auto p-4 md:p-6 space-y-4">
      {/* Header & Quick Object Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-neutral-800 pb-3 gap-2">
        <div className="flex items-center space-x-2 text-amber-400">
          <Activity className="w-4 h-4" />
          <span className="text-xs font-bold tracking-wider uppercase">
            FLIGHT TELEMETRY COCKPIT — {objName} ({objId})
          </span>
        </div>

        {/* Fleet Object Switcher */}
        {allObjects.length > 1 && onSelectObjectId && (
          <div className="flex flex-wrap items-center gap-1 text-[10px]">
            <span className="text-neutral-500">OBJECT:</span>
            {allObjects.map((obj) => (
              <button
                key={obj.id}
                onClick={() => onSelectObjectId(obj.id)}
                className={`px-2 py-0.5 border ${
                  selectedObject?.id === obj.id
                    ? "bg-amber-500/20 text-amber-300 border-amber-500/60 font-bold"
                    : "bg-neutral-950 text-neutral-400 border-neutral-800 hover:border-neutral-700 hover:text-neutral-200"
                }`}
              >
                {obj.id}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Primary Readout Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {/* Altitude */}
        <div className="bg-neutral-950 border border-neutral-800 p-3 space-y-1">
          <div className="text-[10px] text-neutral-500 flex items-center justify-between">
            <span>ALTITUDE</span>
            <Navigation className="w-3.5 h-3.5 text-amber-400" />
          </div>
          <div className="text-base font-bold text-white truncate">
            {formatDistance(alt)}
          </div>
          <div className="text-[9px] text-neutral-500">Above mean surface radius</div>
        </div>

        {/* Speed */}
        <div className="bg-neutral-950 border border-neutral-800 p-3 space-y-1">
          <div className="text-[10px] text-neutral-500 flex items-center justify-between">
            <span>INERTIAL SPEED</span>
            <Gauge className="w-3.5 h-3.5 text-emerald-400" />
          </div>
          <div className="text-base font-bold text-emerald-400 truncate">
            {formatSpeed(speed)}
          </div>
          <div className="text-[9px] text-neutral-500">ECI frame magnitude |v|</div>
        </div>

        {/* Total Mass */}
        <div className="bg-neutral-950 border border-neutral-800 p-3 space-y-1">
          <div className="text-[10px] text-neutral-500 flex items-center justify-between">
            <span>TOTAL MASS</span>
            <Weight className="w-3.5 h-3.5 text-amber-300" />
          </div>
          <div className="text-base font-bold text-amber-300 truncate">
            {formatMass(mass)}
          </div>
          <div className="text-[9px] text-neutral-500">
            {isDebris ? "Fragment Mass" : "Dry mass + Propellant"}
          </div>
        </div>

        {/* Status / Fuel */}
        <div className="bg-neutral-950 border border-neutral-800 p-3 space-y-1">
          <div className="text-[10px] text-neutral-500 flex items-center justify-between">
            <span>STATUS / HBR</span>
            <Fuel className="w-3.5 h-3.5 text-amber-400" />
          </div>
          <div className={`text-base font-bold truncate ${isDestroyed ? "text-red-400" : "text-white"}`}>
            {isDestroyed ? "DESTROYED" : `HBR: ${hbr.toFixed(1)}m`}
          </div>
          <div className="text-[9px] text-neutral-500">
            {isDestroyed ? "Collision Impact" : isDebris ? "Propagated Debris" : `Fuel: ${fuel.toFixed(0)} kg`}
          </div>
        </div>
      </div>

      {/* State Vector (Cartesian ECI) */}
      <div className="bg-neutral-950 border border-neutral-800 p-3.5 space-y-2.5">
        <div className="text-xs text-amber-400 font-bold uppercase tracking-wider flex items-center justify-between">
          <span>Instantaneous Cartesian State Vector (ECI J2000)</span>
          <span className="text-[10px] text-neutral-500 font-normal">SI Units (m, m/s)</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs">
          <div className="bg-black p-2.5 border border-neutral-850">
            <div className="text-[10px] text-neutral-500">POSITION X:</div>
            <div className="text-xs text-white font-bold">{pos[0].toLocaleString(undefined, { maximumFractionDigits: 1 })} m</div>
          </div>
          <div className="bg-black p-2.5 border border-neutral-850">
            <div className="text-[10px] text-neutral-500">POSITION Y:</div>
            <div className="text-xs text-white font-bold">{pos[1].toLocaleString(undefined, { maximumFractionDigits: 1 })} m</div>
          </div>
          <div className="bg-black p-2.5 border border-neutral-850">
            <div className="text-[10px] text-neutral-500">POSITION Z:</div>
            <div className="text-xs text-white font-bold">{pos[2].toLocaleString(undefined, { maximumFractionDigits: 1 })} m</div>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs">
          <div className="bg-black p-2.5 border border-neutral-850">
            <div className="text-[10px] text-neutral-500">VELOCITY VX:</div>
            <div className="text-xs text-emerald-400 font-bold">{vel[0].toFixed(3)} m/s</div>
          </div>
          <div className="bg-black p-2.5 border border-neutral-850">
            <div className="text-[10px] text-neutral-500">VELOCITY VY:</div>
            <div className="text-xs text-emerald-400 font-bold">{vel[1].toFixed(3)} m/s</div>
          </div>
          <div className="bg-black p-2.5 border border-neutral-850">
            <div className="text-[10px] text-neutral-500">VELOCITY VZ:</div>
            <div className="text-xs text-emerald-400 font-bold">{vel[2].toFixed(3)} m/s</div>
          </div>
        </div>
      </div>
    </div>
  );
};
