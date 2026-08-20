import React, { useState } from "react";
import { MultiConjunctionEvent, PhysicalCollisionEvent } from "../../types/mission";
import { 
  AlertTriangle, 
  Flame, 
  Crosshair, 
  SlidersHorizontal, 
  ChevronRight, 
  ArrowUpDown,
  Layers,
  Sparkles
} from "lucide-react";

interface ConjunctionsPanelProps {
  conjunctions: MultiConjunctionEvent[];
  collisions: PhysicalCollisionEvent[];
  currentTimeSeconds: number;
  onSelectConjunction: (conj: MultiConjunctionEvent) => void;
  onJumpToTCA: (tca: number) => void;
}

type SortBy = "RISK" | "MISS_DISTANCE" | "TCA";

export const ConjunctionsPanel: React.FC<ConjunctionsPanelProps> = ({
  conjunctions,
  collisions,
  currentTimeSeconds,
  onSelectConjunction,
  onJumpToTCA,
}) => {
  const [sortBy, setSortBy] = useState<SortBy>("RISK");
  const [filterPhysicalOnly, setFilterPhysicalOnly] = useState<boolean>(false);

  const sortedConjunctions = [...conjunctions].sort((a, b) => {
    if (sortBy === "RISK") {
      const pcA = a.collision_probability || 0;
      const pcB = b.collision_probability || 0;
      return pcB - pcA;
    } else if (sortBy === "MISS_DISTANCE") {
      return a.miss_distance_m - b.miss_distance_m;
    } else {
      return a.tca_s - b.tca_s;
    }
  }).filter((c) => !filterPhysicalOnly || c.is_physical_collision);

  const getRiskBadge = (level: string, isCollision: boolean) => {
    if (isCollision) {
      return (
        <span className="px-1.5 py-0.5 text-[10px] bg-red-950/80 border border-red-500 text-red-400 font-bold uppercase tracking-wider flex items-center gap-1">
          <Flame className="w-2.5 h-2.5 text-red-400" />
          COLLISION
        </span>
      );
    }
    switch (level) {
      case "CRITICAL":
        return <span className="px-1.5 py-0.5 text-[10px] bg-red-950/60 border border-red-600/60 text-red-400 font-bold">CRITICAL</span>;
      case "HIGH":
        return <span className="px-1.5 py-0.5 text-[10px] bg-amber-950/60 border border-amber-500/60 text-amber-400 font-semibold">HIGH</span>;
      case "ELEVATED":
        return <span className="px-1.5 py-0.5 text-[10px] bg-yellow-950/40 border border-yellow-500/50 text-yellow-300">ELEVATED</span>;
      default:
        return <span className="px-1.5 py-0.5 text-[10px] bg-neutral-900 border border-neutral-700 text-neutral-400">LOW</span>;
    }
  };

  return (
    <div className="flex flex-col h-full bg-black/90 border border-neutral-800 text-neutral-200 font-mono text-xs select-none">
      {/* Header */}
      <div className="p-2.5 border-b border-neutral-800 flex items-center justify-between bg-neutral-950">
        <div className="flex items-center gap-2">
          <Layers className="w-3.5 h-3.5 text-amber-400" />
          <span className="font-semibold tracking-wider text-neutral-100 text-[11px] uppercase">
            Conjunctions ({conjunctions.length})
          </span>
          {collisions.length > 0 && (
            <span className="text-[10px] text-red-400 bg-red-950/60 border border-red-800 px-1 py-0.2 rounded-none flex items-center gap-1">
              <Flame className="w-2.5 h-2.5" />
              {collisions.length} IMPACT
            </span>
          )}
        </div>

        {/* Sort Controls */}
        <div className="flex items-center gap-1 text-[10px]">
          <span className="text-neutral-500">SORT:</span>
          <button
            onClick={() => setSortBy("RISK")}
            className={`px-1.5 py-0.5 border ${
              sortBy === "RISK"
                ? "bg-amber-500/20 text-amber-300 border-amber-500/50"
                : "bg-black text-neutral-500 border-neutral-800 hover:text-neutral-300"
            }`}
          >
            Pc
          </button>
          <button
            onClick={() => setSortBy("MISS_DISTANCE")}
            className={`px-1.5 py-0.5 border ${
              sortBy === "MISS_DISTANCE"
                ? "bg-amber-500/20 text-amber-300 border-amber-500/50"
                : "bg-black text-neutral-500 border-neutral-800 hover:text-neutral-300"
            }`}
          >
            MISS
          </button>
          <button
            onClick={() => setSortBy("TCA")}
            className={`px-1.5 py-0.5 border ${
              sortBy === "TCA"
                ? "bg-amber-500/20 text-amber-300 border-amber-500/50"
                : "bg-black text-neutral-500 border-neutral-800 hover:text-neutral-300"
            }`}
          >
            TCA
          </button>
        </div>
      </div>

      {/* Conjunction List */}
      <div className="flex-1 overflow-y-auto divide-y divide-neutral-900">
        {sortedConjunctions.length === 0 ? (
          <div className="p-4 text-center text-neutral-600 text-xs italic">
            No close conjunction events detected in the current analysis window.
          </div>
        ) : (
          sortedConjunctions.map((conj) => {
            const isNearTCA = Math.abs(currentTimeSeconds - conj.tca_s) < 60.0;

            return (
              <div
                key={conj.event_id}
                className={`p-2.5 transition-colors hover:bg-neutral-900/60 cursor-pointer ${
                  conj.is_physical_collision ? "bg-red-950/20" : ""
                } ${isNearTCA ? "border-l-2 border-l-amber-400 bg-neutral-900/40" : ""}`}
                onClick={() => onSelectConjunction(conj)}
              >
                {/* Top Row: Spacecraft Pair + Risk Badge */}
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-1.5 font-bold text-neutral-100 text-xs">
                    <span className="text-amber-400">{conj.spacecraft_a_name}</span>
                    <span className="text-neutral-600">↔</span>
                    <span className="text-amber-400">{conj.spacecraft_b_name}</span>
                  </div>
                  {getRiskBadge(conj.risk_level, conj.is_physical_collision)}
                </div>

                {/* Metrics Grid */}
                <div className="grid grid-cols-3 gap-1 text-[11px] text-neutral-400 mb-2">
                  <div>
                    <span className="text-neutral-600 block text-[9px]">TCA</span>
                    <span className="font-semibold text-neutral-200">
                      T+ {formatTime(conj.tca_s)}
                    </span>
                  </div>
                  <div>
                    <span className="text-neutral-600 block text-[9px]">MISS DIST</span>
                    <span className={conj.is_physical_collision ? "text-red-400 font-bold" : "text-neutral-200"}>
                      {conj.miss_distance_m < 1000
                        ? `${conj.miss_distance_m.toFixed(1)} m`
                        : `${conj.miss_distance_km.toFixed(2)} km`}
                    </span>
                  </div>
                  <div>
                    <span className="text-neutral-600 block text-[9px]">REL VEL</span>
                    <span className="text-neutral-200">
                      {conj.relative_velocity_km_s.toFixed(2)} km/s
                    </span>
                  </div>
                </div>

                {/* Probabilistic metrics if calculated */}
                {conj.collision_probability !== undefined && (
                  <div className="flex items-center justify-between text-[10px] text-neutral-400 border-t border-neutral-900 pt-1.5 mt-1">
                    <span>
                      HBR: <strong className="text-neutral-300">{conj.hard_body_radius_m.toFixed(1)} m</strong>
                    </span>
                    <span>
                      Pc: <strong className="text-amber-300">{conj.collision_probability_scientific || conj.collision_probability.toExponential(3)}</strong>
                    </span>
                    <div className="flex items-center gap-1">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onJumpToTCA(conj.tca_s);
                        }}
                        className="px-1.5 py-0.5 bg-neutral-900 text-neutral-300 border border-neutral-800 hover:border-neutral-600 text-[9px]"
                        title="Jump simulation time to TCA"
                      >
                        TCA
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectConjunction(conj);
                        }}
                        className="px-1.5 py-0.5 bg-amber-500/20 text-amber-400 border border-amber-500/50 hover:bg-amber-500/30 text-[9px] flex items-center gap-0.5"
                      >
                        B-PLANE
                        <ChevronRight className="w-2.5 h-2.5" />
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

function formatTime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return `${h.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}
