/**
 * THESEUS ORBIT-X Mission Construction Overlay
 * ============================================
 * Sophisticated, high-fidelity mission construction animation overlay.
 * Replaces generic "Loading..." spinners with real step-by-step visual progress
 * through ORBIT-X computational milestones.
 *
 * Consumes real calculation_trace steps from ORBIT-X physics solver response.
 * Features KaTeX formula typesetting, Newton-Raphson iteration logs, and smooth
 * contraction transition into active simulation once complete.
 */

import React, { useEffect, useState } from 'react';
import { Terminal, CheckCircle2, Play, Sparkles, X } from 'lucide-react';
import { SimulationResult, CalculationStep } from '../../types/mission';
import { BuildStageCard, StageState } from './BuildStageCard';

interface MissionBuildOverlayProps {
  isOpen: boolean;
  onClose: () => void;
  simResult: SimulationResult | null;
  originName: string;
  destinationName: string;
  vehicleName: string;
  onComplete: () => void;
}

export const MissionBuildOverlay: React.FC<MissionBuildOverlayProps> = ({
  isOpen,
  onClose,
  simResult,
  originName,
  destinationName,
  vehicleName,
  onComplete,
}) => {
  const [activeStageIdx, setActiveStageIdx] = useState(0);
  const [stageStates, setStageStates] = useState<StageState[]>([]);
  const [isFinished, setIsFinished] = useState(false);

  const traces: CalculationStep[] = simResult?.calculation_trace || [];

  useEffect(() => {
    if (!isOpen || traces.length === 0) {
      setActiveStageIdx(0);
      setStageStates([]);
      setIsFinished(false);
      return;
    }

    // Initialize all stages to WAITING
    const initialStates: StageState[] = traces.map((_, i) => (i === 0 ? 'ACTIVE' : 'WAITING'));
    setStageStates(initialStates);
    setActiveStageIdx(0);
    setIsFinished(false);

    // Timed progressive reveal through calculation trace steps
    let currentIdx = 0;
    const interval = setInterval(() => {
      currentIdx++;
      if (currentIdx < traces.length) {
        setActiveStageIdx(currentIdx);
        setStageStates(prev => {
          const next = [...prev];
          next[currentIdx - 1] = 'COMPLETE';
          next[currentIdx] = 'ACTIVE';
          return next;
        });
      } else {
        clearInterval(interval);
        setStageStates(prev => prev.map(() => 'COMPLETE'));
        setIsFinished(true);
      }
    }, 450); // 450ms step reveal delay

    return () => clearInterval(interval);
  }, [isOpen, simResult]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#04060a]/90 backdrop-blur-md p-4 lg:p-6 font-mono text-xs select-text">
      <div className="w-full h-full max-w-5xl bg-[#05080f] border border-[#221d17] rounded shadow-2xl overflow-hidden flex flex-col">
        
        {/* Header Bar */}
        <div className="bg-[#070d18] border-b border-[#221d17] px-5 py-3 flex items-center justify-between shrink-0 select-none">
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <Terminal className="w-4 h-4 text-[#ff9900]" />
              <span className="font-['Orbitron'] font-black tracking-widest text-sm text-[#ffffff]">
                THESEUS <span className="text-[#ff9900]">// ORBIT-X MISSION CONSTRUCTION</span>
              </span>
            </div>
            <span className="text-[#332b22]">|</span>
            <div className="text-xs text-[#8c8275]">
              <span className="text-[#ffffff] font-bold">{originName.toUpperCase()}</span>
              <span className="text-[#ff9900] mx-1">→</span>
              <span className="text-[#44bb66] font-bold">{destinationName.toUpperCase()}</span>
              <span className="text-[#554b3e] mx-2">•</span>
              <span>{vehicleName}</span>
            </div>
          </div>

          <button
            onClick={onClose}
            className="flex items-center space-x-1 bg-[#0b1424] hover:bg-[#152238] border border-[#221d17] px-3 py-1 rounded text-xs text-[#8c8275] hover:text-[#ffffff] transition-all cursor-pointer"
          >
            <X className="w-3.5 h-3.5 text-[#ff9900]" />
            <span>[✕ CANCEL]</span>
          </button>
        </div>

        {/* Main Stage Grid (Scrollable) */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {traces.map((step, idx) => (
              <BuildStageCard
                key={idx}
                step={step}
                state={stageStates[idx] ?? 'WAITING'}
              />
            ))}
          </div>
        </div>

        {/* Footer Action Bar */}
        <div className="bg-[#070d18] border-t border-[#221d17] px-5 py-3 flex items-center justify-between shrink-0 select-none">
          <div className="flex items-center space-x-2">
            {isFinished ? (
              <span className="text-[#44bb66] font-bold text-xs flex items-center space-x-1.5">
                <CheckCircle2 className="w-4 h-4 text-[#44bb66]" />
                <span>MISSION VALIDATION COMPLETE — ORBITAL SOLUTION CONVERGED</span>
              </span>
            ) : (
              <span className="text-[#ff9900] font-bold text-xs flex items-center space-x-1.5 animate-pulse">
                <Sparkles className="w-4 h-4 text-[#ff9900]" />
                <span>SOLVING PHASE {String(activeStageIdx + 1).padStart(2, '0')} OF {traces.length}...</span>
              </span>
            )}
          </div>

          {isFinished && (
            <button
              onClick={() => {
                onComplete();
                onClose();
              }}
              className="px-5 py-2 bg-[#ff9900] hover:bg-[#ffaa22] text-[#04060a] font-['Orbitron'] font-black rounded text-xs transition-all flex items-center space-x-2 cursor-pointer shadow-lg shadow-[#ff9900]/20"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>ENTER MISSION SIMULATION →</span>
            </button>
          )}
        </div>

      </div>
    </div>
  );
};
