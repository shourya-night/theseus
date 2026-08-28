/**
 * THESEUS Master Simulator Layout Manager
 * =======================================
 * Top-level layout manager coordinating:
 *   - Header Bar (status, backend health, mission setup button)
 *   - Left Panel (Objects Explorer & Visualization Layers)
 *   - Center Viewport (Three.js 3D Solar System Renderer)
 *   - Right Panel (Flight Telemetry HUD)
 *   - Bottom Bar (Simulation Timeline Scrubber)
 *
 * ALL panels are collapsible, allowing the 3D viewport to dynamically expand.
 */

import React, { useState } from 'react';
import { CollapsiblePanel } from './CollapsiblePanel';
import { VisualizationLayersPanel, LayerState, DEFAULT_LAYER_STATE } from '../Visualization/VisualizationLayersPanel';
import { ObjectsPanel } from '../Objects/ObjectsPanel';
import { TelemetryHUD } from '../Telemetry/TelemetryHUD';
import { TimelineScrubber } from '../Telemetry/TimelineScrubber';
import { Layers, Globe2, Activity, Clock, Terminal, Play, Pause, RotateCcw } from 'lucide-react';
import { ActiveRocket, StateVector, SimulationResult } from '../../types/mission';

interface SimulatorLayoutProps {
  // Navigation & Actions
  onOpenMissionSetup: () => void;
  onOpenAnalysisOverlay: () => void;

  // Fleet & Objects State
  activeRockets: ActiveRocket[];
  selectedObjectId: string | null;
  onSelectObjectId: (id: string) => void;
  onFocusObjectId: (id: string) => void;

  // Simulation Clock
  simTimeSec: number;
  maxSimTimeSec: number;
  isPlaying: boolean;
  onTogglePlay: () => void;
  onSeekTime: (t: number) => void;
  simSpeed: number;
  onChangeSimSpeed: (s: number) => void;

  // Layers
  layers: LayerState;
  onToggleLayer: (key: keyof LayerState) => void;

  // Viewport Container
  viewportContent: React.ReactNode;
}

export const SimulatorLayout: React.FC<SimulatorLayoutProps> = ({
  onOpenMissionSetup,
  onOpenAnalysisOverlay,
  activeRockets,
  selectedObjectId,
  onSelectObjectId,
  onFocusObjectId,
  simTimeSec,
  maxSimTimeSec,
  isPlaying,
  onTogglePlay,
  onSeekTime,
  simSpeed,
  onChangeSimSpeed,
  layers,
  onToggleLayer,
  viewportContent,
}) => {
  // Panel Collapse States
  const [isLeftCollapsed, setIsLeftCollapsed] = useState(false);
  const [isRightCollapsed, setIsRightCollapsed] = useState(false);
  const [isBottomCollapsed, setIsBottomCollapsed] = useState(false);

  // Left panel mode: OBJECTS vs LAYERS
  const [leftPanelSubTab, setLeftPanelSubTab] = useState<'OBJECTS' | 'LAYERS'>('OBJECTS');

  const activeRocket = activeRockets[0];
  const simResult = activeRocket?.result;

  return (
    <div className="w-full h-full flex flex-col bg-[#04060a] text-[#e6dfd5] font-mono select-none overflow-hidden">
      {/* ── 1. HEADER BAR ─────────────────────────────────────────── */}
      <header className="h-11 bg-[#05080f] border-b border-[#221d17] px-4 flex items-center justify-between shrink-0 z-30">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <Terminal className="w-4 h-4 text-[#ff9900]" />
            <span className="font-['Orbitron'] font-black tracking-widest text-sm text-[#ffffff]">
              THESEUS <span className="text-[#ff9900] text-xs">// ASTRODYNAMICS</span>
            </span>
          </div>
          <span className="text-[#333333]">|</span>
          <span className="text-[10px] text-[#8c8275] hidden sm:inline">
            ORBIT-X ENGINE VISUALIZATION SIMULATOR
          </span>
        </div>

        {/* Header Action Buttons */}
        <div className="flex items-center space-x-2">
          <button
            onClick={onOpenMissionSetup}
            className="px-3 py-1 bg-[#ff9900] hover:bg-[#ffaa22] text-[#04060a] font-bold rounded text-xs transition-colors cursor-pointer"
          >
            [+ BUILD MISSION]
          </button>
          {simResult && (
            <button
              onClick={onOpenAnalysisOverlay}
              className="px-3 py-1 bg-[#0b1424] hover:bg-[#152238] border border-[#ff9900]/60 text-[#ff9900] font-bold rounded text-xs transition-colors cursor-pointer"
            >
              [ANALYSIS TRACE]
            </button>
          )}
        </div>
      </header>

      {/* ── 2. MAIN SPLIT AREA (PANELS + VIEWPORT) ────────────────── */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Left Collapsible Panel */}
        <CollapsiblePanel
          id="left-panel"
          title={leftPanelSubTab === 'OBJECTS' ? 'MISSION / OBJECTS' : 'VISUALIZATION LAYERS'}
          icon={leftPanelSubTab === 'OBJECTS' ? <Globe2 className="w-3.5 h-3.5 text-[#ff9900]" /> : <Layers className="w-3.5 h-3.5 text-[#ff9900]" />}
          side="left"
          isCollapsed={isLeftCollapsed}
          onToggle={() => setIsLeftCollapsed(!isLeftCollapsed)}
          width="w-80"
          headerControls={
            <div className="flex items-center space-x-1 bg-[#070d18] p-0.5 border border-[#162238] rounded">
              <button
                onClick={() => setLeftPanelSubTab('OBJECTS')}
                className={`px-1.5 py-0.5 rounded text-[9px] font-bold ${
                  leftPanelSubTab === 'OBJECTS' ? 'bg-[#ff9900]/20 text-[#ff9900]' : 'text-[#8c8275]'
                }`}
              >
                OBJECTS
              </button>
              <button
                onClick={() => setLeftPanelSubTab('LAYERS')}
                className={`px-1.5 py-0.5 rounded text-[9px] font-bold ${
                  leftPanelSubTab === 'LAYERS' ? 'bg-[#ff9900]/20 text-[#ff9900]' : 'text-[#8c8275]'
                }`}
              >
                LAYERS
              </button>
            </div>
          }
        >
          {leftPanelSubTab === 'OBJECTS' ? (
            <ObjectsPanel
              activeRockets={activeRockets}
              selectedObjectId={selectedObjectId}
              onSelectObject={onSelectObjectId}
              onFocusObject={onFocusObjectId}
            />
          ) : (
            <VisualizationLayersPanel layers={layers} onToggleLayer={onToggleLayer} />
          )}
        </CollapsiblePanel>

        {/* Center Dynamic 3D Viewport */}
        <main className="flex-1 h-full relative overflow-hidden bg-[#000000]">
          {viewportContent}
        </main>

        {/* Right Collapsible Panel (Telemetry HUD) */}
        <CollapsiblePanel
          id="right-panel"
          title="FLIGHT TELEMETRY"
          icon={<Activity className="w-3.5 h-3.5 text-[#ff9900]" />}
          side="right"
          isCollapsed={isRightCollapsed}
          onToggle={() => setIsRightCollapsed(!isRightCollapsed)}
          width="w-80"
        >
          <TelemetryHUD
            deltaVBudget={simResult?.delta_v_budget}
            propellantBudget={simResult?.propellant_budget}
            allObjects={[]}
          />
        </CollapsiblePanel>
      </div>

      {/* ── 3. BOTTOM COLLAPSIBLE TIMELINE ────────────────────────── */}
      <CollapsiblePanel
        id="bottom-timeline"
        title="SIMULATION TIMELINE"
        icon={<Clock className="w-3.5 h-3.5 text-[#ff9900]" />}
        side="bottom"
        isCollapsed={isBottomCollapsed}
        onToggle={() => setIsBottomCollapsed(!isBottomCollapsed)}
        height="h-28"
        headerControls={
          <div className="flex items-center space-x-2">
            <button
              onClick={onTogglePlay}
              className="p-1 rounded bg-[#ff9900] text-[#04060a] font-bold hover:bg-[#ffaa22] transition-colors"
            >
              {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            </button>
            <span className="text-[10px] text-[#ff9900] font-bold">
              {simSpeed}x
            </span>
          </div>
        }
      >
        <TimelineScrubber
          currentFrameIdx={Math.min(maxSimTimeSec > 0 ? Math.floor((simTimeSec / maxSimTimeSec) * 199) : 0, 199)}
          totalFrames={200}
          currentTimeSeconds={simTimeSec}
          totalDurationSeconds={maxSimTimeSec}
          isPlaying={isPlaying}
          playbackSpeed={simSpeed}
          events={simResult?.events || []}
          onSeek={(idx) => onSeekTime((idx / 199) * maxSimTimeSec)}
          onTogglePlay={onTogglePlay}
          onSetSpeed={onChangeSimSpeed}
          onReset={() => onSeekTime(0)}
        />
      </CollapsiblePanel>
    </div>
  );
};
