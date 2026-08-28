/**
 * THESEUS Visualization Layers Control Panel
 * ==========================================
 * Dedicated collapsible control panel providing checkbox toggles for all
 * visualization categories (Celestial, Small Bodies, Artificial, Scientific).
 */

import React from 'react';
import { Layers, Eye, Globe2, Sparkles, Navigation, Activity } from 'lucide-react';

export interface LayerState {
  // Celestial
  planets: boolean;
  moons: boolean;
  dwarfPlanets: boolean;
  rings: boolean;

  // Small Bodies
  asteroidBelt: boolean;
  namedAsteroids: boolean;
  neos: boolean;
  comets: boolean;
  trojansL4: boolean;
  trojansL5: boolean;
  hildas: boolean;
  cybeles: boolean;
  centaurs: boolean;
  meteorStreams: boolean;
  kuiperBelt: boolean;
  scatteredDisk: boolean;
  oortCloud: boolean;
  zodiacalDust: boolean;

  // Artificial
  satellites: boolean;
  stations: boolean;
  telescopes: boolean;
  probes: boolean;
  missionSpacecraft: boolean;

  // Scientific
  orbits: boolean;
  trajectories: boolean;
  lagrangePoints: boolean;
  soi: boolean;
  hillSpheres: boolean;
  referencePlanes: boolean;
  forceVectors: boolean;
  uncertainty: boolean;
}

export const DEFAULT_LAYER_STATE: LayerState = {
  planets: true,
  moons: true,
  dwarfPlanets: true,
  rings: true,

  asteroidBelt: true,
  namedAsteroids: true,
  neos: true,
  comets: true,
  trojansL4: false,
  trojansL5: false,
  hildas: false,
  cybeles: false,
  centaurs: false,
  meteorStreams: false,
  kuiperBelt: false,
  scatteredDisk: false,
  oortCloud: false,
  zodiacalDust: false,

  satellites: true,
  stations: true,
  telescopes: true,
  probes: true,
  missionSpacecraft: true,

  orbits: true,
  trajectories: true,
  lagrangePoints: true,
  soi: false,
  hillSpheres: false,
  // Off by default. This layer is a THREE.GridHelper plus an AxesHelper at the
  // scene origin — developer gizmos, and only 200 units (2 million km) across,
  // so at any planetary framing it reads as a stray grid sitting on the Sun.
  // Brief §34 also asks that reference frames not all be shown by default.
  referencePlanes: false,
  forceVectors: true,
  uncertainty: true,
};

interface VisualizationLayersPanelProps {
  layers: LayerState;
  onToggleLayer: (key: keyof LayerState) => void;
}

export const VisualizationLayersPanel: React.FC<VisualizationLayersPanelProps> = ({
  layers,
  onToggleLayer,
}) => {
  const renderSection = (
    title: string,
    icon: React.ReactNode,
    items: Array<{ key: keyof LayerState; label: string; tag?: string }>
  ) => (
    <div className="space-y-1.5 pb-2.5 border-b border-[#221d17] last:border-b-0">
      <div className="text-[10px] font-bold text-[#ff9900] uppercase tracking-wider flex items-center space-x-1.5 px-1">
        {icon}
        <span>{title}</span>
      </div>
      <div className="grid grid-cols-2 gap-1 text-[11px]">
        {items.map(item => (
          <label
            key={item.key}
            className="flex items-center space-x-2 px-2 py-1 rounded bg-[#070d18] hover:bg-[#0b1424] border border-[#162238] cursor-pointer select-none transition-colors"
          >
            <input
              type="checkbox"
              checked={layers[item.key]}
              onChange={() => onToggleLayer(item.key)}
              className="accent-[#ff9900] w-3 h-3 rounded cursor-pointer"
            />
            <span className={layers[item.key] ? 'text-[#e6dfd5] font-semibold' : 'text-[#8c8275]'}>
              {item.label}
            </span>
            {item.tag && (
              <span className="text-[8px] px-1 bg-[#162238] text-[#ff9900] rounded font-bold">
                {item.tag}
              </span>
            )}
          </label>
        ))}
      </div>
    </div>
  );

  return (
    <div className="space-y-3 font-mono">
      {renderSection('CELESTIAL BODIES', <Globe2 className="w-3.5 h-3.5 text-[#ff9900]" />, [
        { key: 'planets', label: 'Planets' },
        { key: 'moons', label: 'Moons' },
        { key: 'dwarfPlanets', label: 'Dwarf Planets' },
        { key: 'rings', label: 'Rings' },
      ])}

      {renderSection('SMALL BODIES', <Sparkles className="w-3.5 h-3.5 text-[#ff9900]" />, [
        { key: 'asteroidBelt', label: 'Main Belt' },
        { key: 'namedAsteroids', label: 'Named Asteroids' },
        { key: 'neos', label: 'NEOs / PHAs' },
        { key: 'comets', label: 'Comets' },
        { key: 'trojansL4', label: 'Trojans (L4)', tag: 'JUP' },
        { key: 'trojansL5', label: 'Trojans (L5)', tag: 'JUP' },
        { key: 'hildas', label: 'Hildas', tag: '3:2' },
        { key: 'cybeles', label: 'Cybeles', tag: '7:4' },
        { key: 'centaurs', label: 'Centaurs' },
        { key: 'meteorStreams', label: 'Meteor Streams' },
        { key: 'kuiperBelt', label: 'Kuiper Belt' },
        { key: 'scatteredDisk', label: 'Scattered Disk' },
        { key: 'oortCloud', label: 'Oort Cloud', tag: 'MODEL' },
        { key: 'zodiacalDust', label: 'Zodiacal Dust', tag: 'MODEL' },
      ])}

      {renderSection('ARTIFICIAL OBJECTS', <Navigation className="w-3.5 h-3.5 text-[#ff9900]" />, [
        { key: 'satellites', label: 'Satellites' },
        { key: 'stations', label: 'Space Stations' },
        { key: 'telescopes', label: 'Telescopes' },
        { key: 'probes', label: 'Interplanetary Probes' },
        { key: 'missionSpacecraft', label: 'Mission Vehicles' },
      ])}

      {renderSection('SCIENTIFIC OVERLAYS', <Activity className="w-3.5 h-3.5 text-[#ff9900]" />, [
        { key: 'orbits', label: 'Orbital Paths' },
        { key: 'trajectories', label: 'Trajectories' },
        { key: 'lagrangePoints', label: 'Lagrange Points' },
        { key: 'soi', label: 'Sphere of Influence' },
        { key: 'hillSpheres', label: 'Hill Spheres' },
        { key: 'referencePlanes', label: 'Ecliptic / Axes' },
        { key: 'forceVectors', label: 'Force Vectors' },
        { key: 'uncertainty', label: 'Uncertainty Clouds' },
      ])}
    </div>
  );
};
