import React, { useState, useRef } from "react";
import { SimulationResult, CalculationStep, MissionEvent } from "../../types/mission";
import { ScientificValue, formatDistance, formatSpeed, formatMass } from "../../lib/formatter";
import { 
  Terminal, 
  X, 
  Check, 
  HelpCircle, 
  BookOpen, 
  Calculator, 
  Layers, 
  Compass, 
  ArrowDown, 
  CheckCircle2, 
  Sparkles,
  Activity,
  Flame,
  Globe2,
  Calendar,
  ChevronRight
} from "lucide-react";

interface CalculationAnalysisOverlayProps {
  isOpen: boolean;
  onClose: () => void;
  simResult: SimulationResult | null;
  originName: string;
  destinationName: string;
  vehicleName: string;
  payloadKg: number;
  epochDate?: string;
  isCalculating?: boolean;
}

export const CalculationAnalysisOverlay: React.FC<CalculationAnalysisOverlayProps> = ({
  isOpen,
  onClose,
  simResult,
  originName,
  destinationName,
  vehicleName,
  payloadKg,
  epochDate = "2026-08-18",
  isCalculating = false,
}) => {
  const [activeSectionId, setActiveSectionId] = useState<string>("phase-0");
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);

  if (!isOpen) return null;

  const traces: CalculationStep[] = simResult?.calculation_trace || [];
  const dvBudget = simResult?.delta_v_budget;
  const propBudget = simResult?.propellant_budget;
  const metadata = simResult?.metadata;
  const events = simResult?.events || [];

  const scrollToPhase = (id: string) => {
    setActiveSectionId(id);
    const element = document.getElementById(id);
    if (element && scrollContainerRef.current) {
      element.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-md p-4 lg:p-6 select-text font-mono">
      <div className="w-full h-full max-w-6xl bg-[#050505] border border-[#262626] rounded shadow-2xl overflow-hidden flex flex-col">
        
        {/* 1. OVERLAY HEADER */}
        <div className="bg-[#0a0a0a] border-b border-[#202020] px-5 py-3 flex items-center justify-between shrink-0 select-none">
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <Terminal className="w-4 h-4 text-[#ff9900]" />
              <span className="font-['Orbitron'] font-extrabold tracking-wider text-sm text-[#ffffff]">
                THESEUS // FLIGHT DYNAMICS ENGINEERING ANALYSIS
              </span>
            </div>
            <span className="text-[#444444]">|</span>
            <div className="text-xs text-[#aaaaaa]">
              <span className="text-[#ffffff] font-bold">{originName.toUpperCase()}</span>
              <span className="text-[#ff9900] mx-1">→</span>
              <span className="text-[#44bb66] font-bold">{destinationName.toUpperCase()}</span>
              <span className="text-[#666666] mx-2">•</span>
              <span>{vehicleName} ({payloadKg.toLocaleString()} kg)</span>
            </div>
          </div>

          <button
            onClick={onClose}
            className="flex items-center space-x-1.5 bg-[#141414] hover:bg-[#222222] border border-[#333333] px-3 py-1 rounded text-xs text-[#cccccc] hover:text-[#ffffff] transition-all cursor-pointer"
          >
            <X className="w-3.5 h-3.5 text-[#ff9900]" />
            <span>[✕ CLOSE ANALYSIS]</span>
          </button>
        </div>

        {/* 2. MAIN SPLIT CONTENT: TABLE OF CONTENTS + DETAILED CALCULATION TRACE */}
        <div className="flex-1 flex overflow-hidden min-h-0">
          
          {/* Left Navigation: Scientific Calculation Index */}
          <aside className="w-64 bg-[#080808] border-r border-[#1c1c1c] p-3 overflow-y-auto hidden md:flex flex-col space-y-1 text-xs shrink-0 select-none">
            <div className="text-[10px] text-[#777777] font-bold uppercase tracking-wider mb-2 px-2 flex items-center space-x-1.5">
              <Layers className="w-3 h-3 text-[#ff9900]" />
              <span>CALCULATION INDEX</span>
            </div>

            {traces.map((step, idx) => {
              const phaseId = `phase-${idx}`;
              const isActive = activeSectionId === phaseId;
              return (
                <button
                  key={idx}
                  onClick={() => scrollToPhase(phaseId)}
                  className={`w-full text-left px-2.5 py-1.5 rounded transition-all flex items-center justify-between text-[11px] ${
                    isActive
                      ? "bg-[#ff9900]/15 border border-[#ff9900]/60 text-[#ffffff] font-bold"
                      : "text-[#888888] hover:text-[#f0eee9] hover:bg-[#121212] border border-transparent"
                  }`}
                >
                  <div className="truncate">
                    <span className="text-[#ff9900] mr-1.5 font-mono">
                      {String(idx).padStart(2, "0")}
                    </span>
                    <span className="truncate">{step.phase.replace(/PHASE \d+ — /, "")}</span>
                  </div>
                  {step.status && (
                    <span className="text-[9px] text-[#44bb66] font-semibold shrink-0 ml-1">
                      ✓
                    </span>
                  )}
                </button>
              );
            })}

            <div className="pt-2 border-t border-[#1c1c1c] mt-2">
              <button
                onClick={() => scrollToPhase("mission-verdict-section")}
                className="w-full text-left px-2.5 py-1.5 rounded text-[11px] text-[#44bb66] hover:bg-[#44bb66]/10 font-bold flex items-center justify-between"
              >
                <span>MISSION VERDICT & RECAP</span>
                <CheckCircle2 className="w-3.5 h-3.5" />
              </button>
            </div>
          </aside>

          {/* Right Scrollable Area: Deep Mathematical Calculation Trace */}
          <main ref={scrollContainerRef} className="flex-1 overflow-y-auto p-5 lg:p-7 space-y-8 text-xs">
            
            {/* Top Mission Context Card */}
            <div className="bg-[#0a0a0a] border border-[#222222] p-4 rounded space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#1c1c1c] pb-2">
                <div className="flex items-center space-x-2">
                  <Globe2 className="w-4 h-4 text-[#ff9900]" />
                  <span className="font-bold text-[#ffffff] text-sm tracking-wide">
                    MISSION PROFILE: {metadata?.name || `${originName.toUpperCase()} → ${destinationName.toUpperCase()}`}
                  </span>
                </div>
                <div className="flex items-center space-x-2 text-[11px]">
                  <span className="text-[#888888]">EPOCH:</span>
                  <span className="text-[#cccccc] font-bold">{epochDate} UTC</span>
                  <span className="text-[#444444]">|</span>
                  <span className="text-[#888888]">STATUS:</span>
                  <span className="text-[#44bb66] font-bold">
                    {metadata?.status || "SOLVER CONVERGED"}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-[11px]">
                <div>
                  <span className="text-[#777777]">ORIGIN:</span>
                  <div className="font-bold text-[#ffffff] uppercase">{originName}</div>
                </div>
                <div>
                  <span className="text-[#777777]">DESTINATION:</span>
                  <div className="font-bold text-[#44bb66] uppercase">{destinationName}</div>
                </div>
                <div>
                  <span className="text-[#777777]">PROPULSION VEHICLE:</span>
                  <div className="font-bold text-[#ff9900]">{vehicleName}</div>
                </div>
                <div>
                  <span className="text-[#777777]">TOTAL ΔV BUDGET:</span>
                  <div className="font-bold text-[#ffffff]">
                    {dvBudget?.total_delta_v ? formatSpeed(dvBudget.total_delta_v) : "—"}
                  </div>
                </div>
              </div>
            </div>

            {/* If no calculation steps available */}
            {traces.length === 0 && (
              <div className="p-12 text-center text-[#777777] space-y-2">
                <Calculator className="w-8 h-8 mx-auto text-[#ff9900] animate-pulse" />
                <div className="text-sm font-bold text-[#cccccc]">NO ACTIVE SIMULATION TRACE LOADED</div>
                <div>Click [RUN SIMULATION] on the main workstation to execute the flight dynamics solver.</div>
              </div>
            )}

            {/* Chronological Calculation Phases */}
            {traces.map((step, idx) => {
              const phaseId = `phase-${idx}`;
              return (
                <section
                  key={idx}
                  id={phaseId}
                  className="bg-[#080808] border border-[#1f1f1f] rounded p-5 space-y-4 shadow-md scroll-mt-4"
                >
                  
                  {/* Phase Header */}
                  <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#1c1c1c] pb-2.5">
                    <div className="flex items-center space-x-2">
                      <span className="px-2 py-0.5 rounded bg-[#ff9900]/15 text-[#ff9900] font-bold text-[11px] border border-[#ff9900]/40">
                        {step.phase}
                      </span>
                      <h3 className="font-bold text-sm text-[#ffffff] tracking-wide">
                        {step.title}
                      </h3>
                    </div>

                    <div className="flex items-center space-x-2">
                      <span className="text-[10px] text-[#44bb66] bg-[#44bb66]/10 px-2 py-0.5 rounded border border-[#44bb66]/30 font-bold flex items-center space-x-1">
                        <Check className="w-3 h-3 text-[#44bb66]" />
                        <span>{step.status}</span>
                      </span>
                    </div>
                  </div>

                  {/* Context: WHAT ARE WE CALCULATING? & WHY DOES THIS MATTER? */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-[11px]">
                    <div className="bg-[#0d0d0d] border border-[#1a1a1a] p-3 rounded space-y-1">
                      <div className="text-[10px] text-[#ff9900] font-bold uppercase tracking-wider flex items-center space-x-1">
                        <HelpCircle className="w-3.5 h-3.5" />
                        <span>WHAT ARE WE CALCULATING?</span>
                      </div>
                      <p className="text-[#cccccc] leading-relaxed">
                        <ScientificValue value={step.beginnerExplanation || step.explanation} />
                      </p>
                    </div>

                    <div className="bg-[#0d0d0d] border border-[#1a1a1a] p-3 rounded space-y-1">
                      <div className="text-[10px] text-[#aaaaaa] font-bold uppercase tracking-wider flex items-center space-x-1">
                        <BookOpen className="w-3.5 h-3.5 text-[#ff9900]" />
                        <span>WHY DOES THIS MATTER?</span>
                      </div>
                      <p className="text-[#888888] leading-relaxed">
                        <ScientificValue value={step.scientificNotes || step.explanation} />
                      </p>
                    </div>
                  </div>

                  {/* Physical Equation */}
                  {step.equation && (
                    <div className="bg-[#0a0a0a] border-l-2 border-[#ff9900] p-3 rounded text-xs space-y-1">
                      <div className="text-[9.5px] text-[#777777] uppercase font-bold tracking-wider">
                        GOVERNING EQUATION / PHYSICAL LAW:
                      </div>
                      <div className="font-mono text-sm text-[#ffaa00] font-bold tracking-wide">
                        <ScientificValue value={step.equation} />
                      </div>
                    </div>
                  )}

                  {/* Actual Parameter Values & Physical Substitutions */}
                  {step.substitutions && Object.keys(step.substitutions).length > 0 && (
                    <div className="bg-[#0a0a0a] border border-[#1c1c1c] p-3.5 rounded space-y-2">
                      <div className="text-[10px] text-[#888888] font-bold uppercase tracking-wider">
                        ACTUAL PHYSICAL VALUES & SUBSTITUTIONS:
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-[11px]">
                        {Object.entries(step.substitutions).map(([param, val]) => (
                          <div key={param} className="flex justify-between border-b border-[#141414] pb-1 items-center">
                            <span className="text-[#888888] font-medium">{param}:</span>
                            <span className="text-[#f0eee9] font-bold font-mono">
                              <ScientificValue value={val} />
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Intermediate Step Arithmetic */}
                  {step.intermediateCalculation && step.intermediateCalculation.length > 0 && (
                    <div className="bg-[#070707] border border-[#1c1c1c] p-3.5 rounded space-y-1.5">
                      <div className="text-[9.5px] text-[#777777] font-bold uppercase tracking-wider">
                        INTERMEDIATE ARITHMETIC EVALUATION:
                      </div>
                      <div className="space-y-1">
                        {step.intermediateCalculation.map((line, lIdx) => (
                          <div key={lIdx} className="text-[#cccccc] font-mono text-[11px] leading-relaxed flex items-center space-x-2">
                            <ChevronRight className="w-3 h-3 text-[#ff9900] shrink-0" />
                            <span><ScientificValue value={line} /></span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Newton-Raphson Solver Iteration Log (for Lambert) */}
                  {step.iterations && step.iterations.length > 0 && (
                    <div className="bg-[#070707] border border-[#1c1c1c] p-3.5 rounded space-y-2">
                      <div className="text-[10px] text-[#ff9900] font-bold uppercase tracking-wider">
                        NEWTON-RAPHSON ROOT CONVERGENCE LOG (UNIVERSAL VARIABLE z):
                      </div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-[10px] text-left border-collapse">
                          <thead>
                            <tr className="border-b border-[#222222] text-[#777777]">
                              <th className="py-1">ITER</th>
                              <th className="py-1">VARIABLE z</th>
                              <th className="py-1">CALCULATED TOF</th>
                              <th className="py-1">RESIDUAL |F(z)|</th>
                              <th className="py-1 text-right">STATUS</th>
                            </tr>
                          </thead>
                          <tbody>
                            {step.iterations.map((it) => (
                              <tr key={it.iteration} className="border-b border-[#111111] hover:bg-[#111111]/50">
                                <td className="py-1 font-bold text-[#ff9900]">#{String(it.iteration).padStart(2, "0")}</td>
                                <td className="py-1 font-mono">{it.z.toFixed(6)}</td>
                                <td className="py-1 font-mono text-[#ffffff]">
                                  {it.tof_calculated_s.toLocaleString(undefined, { maximumFractionDigits: 1 })} s ({(it.tof_calculated_s / 3600).toFixed(2)} h)
                                </td>
                                <td className="py-1 font-mono text-[#888888]">{it.residual.toExponential(4)}</td>
                                <td className={`py-1 text-right font-bold ${it.status === "CONVERGED" ? "text-[#44bb66]" : "text-[#ffaa00]"}`}>
                                  {it.status}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Final Step Result */}
                  <div className="flex items-center space-x-2 bg-[#0c160c] border border-[#44bb66]/50 p-3 rounded text-[#44bb66] font-bold text-xs">
                    <CheckCircle2 className="w-4 h-4 text-[#44bb66] shrink-0" />
                    <div className="flex items-center space-x-2 flex-wrap">
                      <span className="text-[#888888] uppercase text-[10px]">CALCULATED RESULT:</span>
                      <span className="text-[#ffffff]"><ScientificValue value={step.result} /></span>
                    </div>
                  </div>

                </section>
              );
            })}

            {/* 3. MISSION RESULT & EXECUTIVE VERDICT RECAP */}
            <section
              id="mission-verdict-section"
              className="bg-[#090f09] border border-[#44bb66]/60 rounded p-6 space-y-4 shadow-xl scroll-mt-4"
            >
              <div className="flex items-center space-x-2 border-b border-[#44bb66]/30 pb-2">
                <CheckCircle2 className="w-5 h-5 text-[#44bb66]" />
                <h3 className="font-['Orbitron'] font-extrabold text-base text-[#ffffff] tracking-wider">
                  MISSION RESULT & VERIFICATION RECAP
                </h3>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                <div>
                  <span className="text-[#777777]">TRANSFER PATH:</span>
                  <div className="font-bold text-[#ffffff] uppercase">{originName} → {destinationName}</div>
                </div>
                <div>
                  <span className="text-[#777777]">TIME OF FLIGHT:</span>
                  <div className="font-bold text-[#ff9900]">
                    {metadata?.duration_hours ? `${metadata.duration_hours.toFixed(1)} hrs (${(metadata.duration_hours / 24).toFixed(1)} days)` : "—"}
                  </div>
                </div>
                <div>
                  <span className="text-[#777777]">TOTAL ΔV REQUIRED:</span>
                  <div className="font-bold text-[#44bb66]">
                    {dvBudget?.total_delta_v ? formatSpeed(dvBudget.total_delta_v) : "—"}
                  </div>
                </div>
                <div>
                  <span className="text-[#777777]">FUEL CONSUMPTION:</span>
                  <div className="font-bold text-[#ffffff]">
                    {propBudget?.fuel_consumed_kg ? formatMass(propBudget.fuel_consumed_kg) : "—"}
                  </div>
                </div>
              </div>

              <div className="bg-[#050505] border border-[#222222] p-3.5 rounded space-y-1.5 text-xs">
                <div className="text-[10px] text-[#ff9900] font-bold uppercase tracking-wider">
                  WHAT DID THESEUS ACTUALLY DO?
                </div>
                <p className="text-[#cccccc] leading-relaxed">
                  THESEUS calculated a high-precision heliocentric transfer from {originName.toUpperCase()}'s departure state at epoch {epochDate} to {destinationName.toUpperCase()}'s predicted moving arrival state.
                  A universal-variable Lambert boundary solver determined the required departure and arrival impulse velocity vectors.
                  The trajectory was then numerically propagated with adaptive Runge-Kutta equations of motion, verifying target orbital interception and positive propellant mass margins.
                </p>
              </div>
            </section>

          </main>

        </div>

      </div>
    </div>
  );
};
