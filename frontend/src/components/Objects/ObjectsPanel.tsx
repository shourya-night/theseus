/**
 * THESEUS Mission / Objects Explorer Panel
 * ========================================
 * Left-hand collapsible panel managing entity selection, visibility, and camera focus:
 *   - Mission Spacecraft fleet
 *   - Celestial bodies tree (Planets, Moons, Dwarf Planets)
 *   - Artificial catalog objects (ISS, JWST, Voyager)
 */

import React, { useState } from 'react';
import { Globe2, Rocket, Radio, ChevronRight, Eye, Crosshair } from 'lucide-react';
import { SOLAR_SYSTEM_OBJECTS, getMoons } from '../../data/astronomicalObjects';
import { CATALOG_ARTIFICIAL_OBJECTS } from '../../data/artificialObjects';
import { ActiveRocket } from '../../types/mission';

interface ObjectsPanelProps {
  activeRockets: ActiveRocket[];
  selectedObjectId: string | null;
  onSelectObject: (id: string, position?: [number, number, number]) => void;
  onFocusObject: (id: string) => void;
}

export const ObjectsPanel: React.FC<ObjectsPanelProps> = ({
  activeRockets,
  selectedObjectId,
  onSelectObject,
  onFocusObject,
}) => {
  const [activeTab, setActiveTab] = useState<'FLEET' | 'CELESTIAL' | 'ARTIFICIAL'>('FLEET');
  const [expandedBody, setExpandedBody] = useState<string | null>('earth');

  const planets = SOLAR_SYSTEM_OBJECTS.filter(o => o.type === 'PLANET');
  const dwarfPlanets = SOLAR_SYSTEM_OBJECTS.filter(o => o.type === 'DWARF_PLANET');

  return (
    <div className="flex flex-col h-full font-mono text-xs space-y-3">
      {/* Category Tabs */}
      <div className="flex items-center space-x-1 bg-[#05080f] p-1 border border-[#221d17] rounded select-none">
        {(['FLEET', 'CELESTIAL', 'ARTIFICIAL'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`flex-1 py-1 rounded text-[10px] font-bold transition-all ${
              activeTab === tab
                ? 'bg-[#ff9900]/20 text-[#ff9900] border border-[#ff9900]/50'
                : 'text-[#8c8275] hover:text-[#e6dfd5]'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* ── FLEET TAB ─────────────────────────────────────────────── */}
      {activeTab === 'FLEET' && (
        <div className="space-y-2 overflow-y-auto flex-1">
          {activeRockets.length === 0 ? (
            <div className="p-4 text-center text-[#8c8275] italic text-[11px]">
              NO ACTIVE MISSION VEHICLES LOADED
            </div>
          ) : (
            activeRockets.map(r => {
              const isSelected = selectedObjectId === r.id;
              return (
                <div
                  key={r.id}
                  onClick={() => onSelectObject(r.id)}
                  className={`p-2.5 rounded border transition-all cursor-pointer space-y-1.5 ${
                    isSelected
                      ? 'bg-[#ff9900]/15 border-[#ff9900] text-[#ffffff]'
                      : 'bg-[#070d18] border-[#162238] text-[#c8c0b5] hover:border-[#ff9900]/50'
                  }`}
                >
                  <div className="flex items-center justify-between font-bold text-[11px]">
                    <div className="flex items-center space-x-1.5 text-[#ff9900]">
                      <Rocket className="w-3.5 h-3.5" />
                      <span>{r.name}</span>
                    </div>
                    <button
                      onClick={e => {
                        e.stopPropagation();
                        onFocusObject(r.id);
                      }}
                      className="p-1 hover:bg-[#152238] rounded text-[#44bb66]"
                      title="Focus Camera"
                    >
                      <Crosshair className="w-3 h-3" />
                    </button>
                  </div>

                  <div className="text-[10px] text-[#8c8275] flex justify-between">
                    <span>PATH: {r.origin.toUpperCase()} → {r.destination.toUpperCase()}</span>
                    <span className="text-[#44bb66] font-bold">{r.collisionState}</span>
                  </div>
                </div>
              );
            })
          )}
        </div>
      )}

      {/* ── CELESTIAL TAB ─────────────────────────────────────────── */}
      {activeTab === 'CELESTIAL' && (
        <div className="space-y-1.5 overflow-y-auto flex-1">
          {/* Sun */}
          <div
            onClick={() => onFocusObject('sun')}
            className="flex items-center justify-between p-2 rounded bg-[#070d18] border border-[#162238] text-[#ffaa00] font-bold text-[11px] cursor-pointer hover:border-[#ffaa00]"
          >
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-[#ffaa00]" />
              <span>SUN (Star)</span>
            </div>
            <Crosshair className="w-3 h-3 text-[#ffaa00]" />
          </div>

          {/* Planets */}
          {planets.map(planet => {
            const moons = getMoons(planet.id);
            const isExpanded = expandedBody === planet.id;

            return (
              <div key={planet.id} className="space-y-1">
                <div
                  onClick={() => onFocusObject(planet.id)}
                  className="flex items-center justify-between p-2 rounded bg-[#070d18] border border-[#162238] text-[#e6dfd5] font-semibold text-[11px] cursor-pointer hover:border-[#ff9900]"
                >
                  <div className="flex items-center space-x-2">
                    {moons.length > 0 && (
                      <button
                        onClick={e => {
                          e.stopPropagation();
                          setExpandedBody(isExpanded ? null : planet.id);
                        }}
                        className="text-[#8c8275] hover:text-[#ff9900]"
                      >
                        <ChevronRight className={`w-3 h-3 transform transition-transform ${isExpanded ? 'rotate-90' : ''}`} />
                      </button>
                    )}
                    <span>{planet.name}</span>
                  </div>
                  <Crosshair className="w-3 h-3 text-[#44bb66]" />
                </div>

                {/* Sub-moons */}
                {isExpanded && moons.length > 0 && (
                  <div className="pl-4 space-y-1">
                    {moons.map(m => (
                      <div
                        key={m.id}
                        onClick={() => onFocusObject(m.id)}
                        className="flex items-center justify-between p-1.5 rounded bg-[#05080f] border border-[#141d2e] text-[#c8c0b5] text-[10px] cursor-pointer hover:border-[#ff9900]"
                      >
                        <span>{m.name}</span>
                        <Crosshair className="w-2.5 h-2.5 text-[#8c8275]" />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}

          {/* Dwarf Planets */}
          <div className="pt-2 border-t border-[#221d17]">
            <div className="text-[10px] font-bold text-[#8c8275] uppercase tracking-wider mb-1 px-1">
              DWARF PLANETS
            </div>
            <div className="grid grid-cols-2 gap-1">
              {dwarfPlanets.map(dp => (
                <button
                  key={dp.id}
                  onClick={() => onFocusObject(dp.id)}
                  className="p-1.5 rounded bg-[#070d18] border border-[#162238] text-[#c8c0b5] text-[10px] text-left truncate hover:border-[#ff9900]"
                >
                  {dp.name}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── ARTIFICIAL TAB ────────────────────────────────────────── */}
      {activeTab === 'ARTIFICIAL' && (
        <div className="space-y-1.5 overflow-y-auto flex-1">
          {CATALOG_ARTIFICIAL_OBJECTS.map(art => (
            <div
              key={art.id}
              onClick={() => onFocusObject(art.id)}
              className="p-2 rounded bg-[#070d18] border border-[#162238] text-[#c8c0b5] space-y-1 cursor-pointer hover:border-[#ff9900]"
            >
              <div className="flex items-center justify-between font-bold text-[11px]">
                <div className="flex items-center space-x-1.5 text-[#00f0ff]">
                  <Radio className="w-3 h-3" />
                  <span className="truncate">{art.name}</span>
                </div>
                <Crosshair className="w-3 h-3 text-[#44bb66]" />
              </div>
              <div className="text-[9px] text-[#8c8275] flex justify-between">
                <span>{art.category}</span>
                <span className="text-[#ff9900]">{art.dataSource}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
