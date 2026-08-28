/**
 * THESEUS Build Stage Card Component
 * ==================================
 * Renders an individual computational milestone stage during ORBIT-X mission construction.
 * Displays real calculation trace values, KaTeX equations, Newton-Raphson iteration logs,
 * and scientific explanation. No fake numbers!
 */

import React from 'react';
import { CheckCircle2, AlertTriangle, XCircle, Clock, ChevronRight } from 'lucide-react';
import { CalculationStep } from '../../types/mission';
import { MathRenderer } from './MathRenderer';

export type StageState = 'WAITING' | 'ACTIVE' | 'COMPLETE' | 'WARNING' | 'FAILED';

interface BuildStageCardProps {
  step: CalculationStep;
  state: StageState;
}

export const BuildStageCard: React.FC<BuildStageCardProps> = ({ step, state }) => {
  const getBadge = () => {
    switch (state) {
      case 'COMPLETE':
        return (
          <span className="px-2 py-0.5 rounded bg-[#44bb66]/15 border border-[#44bb66]/50 text-[#44bb66] font-bold text-[10px] flex items-center space-x-1">
            <CheckCircle2 className="w-3 h-3" />
            <span>COMPLETE</span>
          </span>
        );
      case 'ACTIVE':
        return (
          <span className="px-2 py-0.5 rounded bg-[#ff9900]/20 border border-[#ff9900]/60 text-[#ff9900] font-bold text-[10px] flex items-center space-x-1 animate-pulse">
            <Clock className="w-3 h-3 animate-spin" />
            <span>COMPUTING...</span>
          </span>
        );
      case 'WARNING':
        return (
          <span className="px-2 py-0.5 rounded bg-[#ffaa00]/15 border border-[#ffaa00]/50 text-[#ffaa00] font-bold text-[10px] flex items-center space-x-1">
            <AlertTriangle className="w-3 h-3" />
            <span>WARNING</span>
          </span>
        );
      case 'FAILED':
        return (
          <span className="px-2 py-0.5 rounded bg-[#cc3333]/15 border border-[#cc3333]/50 text-[#cc3333] font-bold text-[10px] flex items-center space-x-1">
            <XCircle className="w-3 h-3" />
            <span>FAILED</span>
          </span>
        );
      default:
        return (
          <span className="px-2 py-0.5 rounded bg-[#162238] border border-[#221d17] text-[#8c8275] font-bold text-[10px]">
            WAITING
          </span>
        );
    }
  };

  return (
    <div
      className={`p-3.5 rounded border transition-all duration-300 font-mono text-xs space-y-2.5 ${
        state === 'ACTIVE'
          ? 'bg-[#0b1424] border-[#ff9900] shadow-lg shadow-[#ff9900]/10 scale-[1.01]'
          : state === 'COMPLETE'
          ? 'bg-[#070d18] border-[#44bb66]/40 text-[#e6dfd5]'
          : 'bg-[#05080f] border-[#162238] opacity-60'
      }`}
    >
      {/* Header Bar */}
      <div className="flex items-center justify-between border-b border-[#162238] pb-2">
        <div className="flex items-center space-x-2">
          <span className="text-[#ff9900] font-bold text-[11px]">
            {String(step.stepIndex).padStart(2, '0')}
          </span>
          <span className="font-['Orbitron'] font-bold text-[11px] text-[#ffffff] tracking-wider">
            {step.phase}
          </span>
        </div>
        {getBadge()}
      </div>

      {/* Title & Beginner Explanation */}
      <div>
        <div className="font-bold text-[#e6dfd5] text-xs mb-0.5">{step.title}</div>
        {(step.beginnerExplanation || step.explanation) && (
          <p className="text-[10.5px] text-[#8c8275] leading-relaxed">
            {step.beginnerExplanation || step.explanation}
          </p>
        )}
      </div>

      {/* Governing Equation (rendered via KaTeX) */}
      {step.equation && state !== 'WAITING' && (
        <div className="p-2 rounded bg-[#04060a] border-l-2 border-[#ff9900] text-xs">
          <div className="text-[9px] text-[#8c8275] uppercase font-bold mb-1">
            GOVERNING EQUATION / PHYSICAL LAW:
          </div>
          <MathRenderer equation={step.equation} />
        </div>
      )}

      {/* Substitutions & Intermediate Evaluation */}
      {step.substitutions && Object.keys(step.substitutions).length > 0 && state !== 'WAITING' && (
        <div className="grid grid-cols-2 gap-1.5 p-2 rounded bg-[#04060a] border border-[#162238] text-[10px]">
          {Object.entries(step.substitutions).map(([param, val]) => (
            <div key={param} className="flex justify-between border-b border-[#0b1424] pb-0.5">
              <span className="text-[#8c8275]">{param}:</span>
              <span className="text-[#e6dfd5] font-bold truncate ml-1">{String(val)}</span>
            </div>
          ))}
        </div>
      )}

      {/* Intermediate Arithmetic Lines */}
      {step.intermediateCalculation && step.intermediateCalculation.length > 0 && state !== 'WAITING' && (
        <div className="space-y-0.5 p-2 rounded bg-[#04060a] border border-[#162238] text-[10px]">
          {step.intermediateCalculation.map((line, lIdx) => (
            <div key={lIdx} className="flex items-center space-x-1 text-[#c8c0b5]">
              <ChevronRight className="w-2.5 h-2.5 text-[#ff9900] shrink-0" />
              <span>{typeof line === 'string' ? line : JSON.stringify(line)}</span>
            </div>
          ))}
        </div>
      )}

      {/* Newton-Raphson Iteration Log (for Lambert) */}
      {step.iterations && step.iterations.length > 0 && state !== 'WAITING' && (
        <div className="p-2 rounded bg-[#04060a] border border-[#ff9900]/30 text-[9.5px] space-y-1">
          <div className="text-[#ff9900] font-bold uppercase tracking-wider text-[9px]">
            NEWTON-RAPHSON ROOT CONVERGENCE LOG (UNIVERSAL VARIABLE z):
          </div>
          <div className="space-y-0.5">
            {step.iterations.map(it => (
              <div key={it.iteration} className="flex justify-between text-[#8c8275]">
                <span>Iter #{String(it.iteration).padStart(2, '0')}</span>
                <span>z = {it.z.toFixed(6)}</span>
                <span className="text-[#e6dfd5]">{it.tof_calculated_s.toFixed(0)} s</span>
                <span className={it.status === 'CONVERGED' ? 'text-[#44bb66] font-bold' : 'text-[#ffaa00]'}>
                  {it.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Result Badge */}
      {step.result && state === 'COMPLETE' && (
        <div className="p-2 rounded bg-[#07170c] border border-[#44bb66]/50 text-[#44bb66] font-bold text-[11px] flex items-center justify-between">
          <span className="text-[9.5px] text-[#8c8275] uppercase">CALCULATED RESULT:</span>
          <span className="text-[#ffffff] font-mono">{String(step.result)}</span>
        </div>
      )}
    </div>
  );
};
