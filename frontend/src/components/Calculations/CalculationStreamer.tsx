import React, { useState, useEffect } from "react";
import { CalculationStep } from "../../types/mission";
import { ScientificValue } from "../../lib/formatter";
import { 
  Terminal, 
  Play, 
  Pause, 
  FastForward, 
  CheckCircle2, 
  Check, 
  HelpCircle,
  BookOpen
} from "lucide-react";

interface CalculationStreamerProps {
  traces: CalculationStep[];
  isCalculating: boolean;
  onCalculationComplete?: () => void;
}

export const CalculationStreamer: React.FC<CalculationStreamerProps> = ({
  traces,
  isCalculating,
  onCalculationComplete,
}) => {
  // Reveal all steps by default or rapid sequential reveal so user immediately sees mathematical rigor
  const [revealedStepCount, setRevealedStepCount] = useState<number>(traces ? traces.length : 0);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [streamSpeedMs, setStreamSpeedMs] = useState<number>(80);

  useEffect(() => {
    if (!traces || traces.length === 0) {
      setRevealedStepCount(0);
      return;
    }
    // Automatically reveal all steps so user never gets stuck at Step 1/1
    setRevealedStepCount(traces.length);
    setIsStreaming(false);
  }, [traces]);

  const handleSkipToEnd = () => {
    if (!traces) return;
    setRevealedStepCount(traces.length);
    setIsStreaming(false);
  };

  if (!traces || traces.length === 0) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center p-4 bg-[#050505] text-[#777777] font-mono text-xs space-y-1.5 border-t border-[#1c1c1c]">
        <Terminal className="w-5 h-5 text-[#ff9900] animate-pulse" />
        <span>WAITING FOR MISSION TRAJECTORY COMMAND...</span>
      </div>
    );
  }

  const activeTraces = traces.slice(0, revealedStepCount);

  return (
    <div className="w-full h-full flex flex-col bg-[#050505] text-[#f0eee9] font-mono select-text overflow-hidden border-t border-[#1c1c1c]">
      
      {/* Terminal Header Strip */}
      <div className="w-full bg-[#0a0a0a] border-b border-[#1c1c1c] px-3 py-1.5 flex flex-wrap items-center justify-between gap-2 text-xs shrink-0 select-none">
        
        {/* Terminal Title */}
        <div className="flex items-center space-x-2">
          <Terminal className="w-3.5 h-3.5 text-[#ff9900]" />
          <span className="font-bold tracking-wider text-[#ff9900] text-[11px]">
            NUMERICAL CALCULATION STREAM
          </span>
          <span className="text-[9.5px] bg-[#141414] border border-[#262626] text-[#aaaaaa] px-1.5 py-0.2 rounded">
            {traces.length} STEPS SOLVED
          </span>
          {isCalculating && (
            <span className="text-[9.5px] text-[#44bb66] animate-pulse font-bold flex items-center space-x-1">
              <span className="w-1.5 h-1.5 rounded-full bg-[#44bb66]" />
              <span>COMPUTING...</span>
            </span>
          )}
        </div>

        {/* Stream Playback Controls */}
        <div className="flex items-center space-x-1.5">
          <button
            onClick={() => {
              if (revealedStepCount >= traces.length) {
                setRevealedStepCount(1);
                setIsStreaming(true);
              } else {
                setIsStreaming(!isStreaming);
              }
            }}
            className="px-2 py-0.5 bg-[#141414] hover:bg-[#202020] border border-[#262626] rounded text-[10px] text-[#aaaaaa] flex items-center space-x-1"
          >
            {isStreaming ? <Pause className="w-2.5 h-2.5" /> : <Play className="w-2.5 h-2.5 fill-current" />}
            <span>{isStreaming ? "PAUSE" : "REPLAY"}</span>
          </button>

          <button
            onClick={handleSkipToEnd}
            className="px-2 py-0.5 bg-[#ff9900]/15 hover:bg-[#ff9900]/25 border border-[#ff9900]/40 rounded text-[10px] text-[#ff9900] font-bold"
          >
            SHOW ALL
          </button>
        </div>

      </div>

      {/* Terminal Output Log Area */}
      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        
        {activeTraces.map((step, idx) => (
          <div key={idx} className="space-y-2 border-b border-[#181818] pb-3">
            
            {/* Phase Banner */}
            <div className="text-[#ff9900] text-[10px] font-bold tracking-widest uppercase flex items-center space-x-2">
              <span className="text-[#333333]">----------------------------------------</span>
              <span>PHASE {String(step.stepIndex).padStart(2, "0")} — {step.phase}</span>
              <span className="text-[#333333]">----------------------------------------</span>
            </div>

            {/* Step Header */}
            <div className="flex items-center justify-between text-xs">
              <div className="font-bold text-[#ffffff]">
                STEP {String(step.stepIndex).padStart(2, "0")} : {step.title}
              </div>
              <span className="text-[9.5px] text-[#44bb66] bg-[#44bb66]/10 px-1.5 py-0.2 rounded border border-[#44bb66]/30 font-bold flex items-center space-x-1">
                <Check className="w-2.5 h-2.5 text-[#44bb66]" />
                <span>{step.status}</span>
              </span>
            </div>

            {/* Formula Block */}
            <div className="bg-[#0a0a0a] border-l-2 border-[#ff9900] p-2 rounded text-xs space-y-0.5">
              <div className="text-[9px] text-[#777777] uppercase">EQUATION / LAW:</div>
              <div className="text-xs text-[#ffaa00] font-bold tracking-wide">
                <ScientificValue value={step.equation} />
              </div>
            </div>

            {/* Parameter Substitutions */}
            {step.substitutions && Object.keys(step.substitutions).length > 0 && (
              <div className="bg-[#0a0a0a] border border-[#1c1c1c] p-2 rounded text-xs space-y-1">
                <div className="text-[9.5px] text-[#777777] font-bold">SUBSTITUTE PHYSICAL VALUES:</div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-1.5 text-[10.5px]">
                  {Object.entries(step.substitutions).map(([param, val]) => (
                    <div key={param} className="flex justify-between border-b border-[#141414] pb-0.5 items-center">
                      <span className="text-[#888888] font-medium">{param}:</span>
                      <span className="text-[#f0eee9] font-bold">
                        <ScientificValue value={val} />
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Intermediate Calculation Lines */}
            {step.intermediateCalculation && step.intermediateCalculation.length > 0 && (
              <div className="bg-[#080808] border border-[#1c1c1c] p-2 rounded text-xs space-y-0.5">
                <div className="text-[9px] text-[#777777] font-bold uppercase">INTERMEDIATE EVALUATION:</div>
                {step.intermediateCalculation.map((line, lIdx) => (
                  <div key={lIdx} className="text-[#cccccc] text-[10.5px] leading-relaxed">
                    <ScientificValue value={line} />
                  </div>
                ))}
              </div>
            )}

            {/* Newton-Raphson Iteration Log (for Lambert solver) */}
            {step.iterations && step.iterations.length > 0 && (
              <div className="bg-[#080808] border border-[#1c1c1c] p-2 rounded text-xs space-y-1.5">
                <div className="text-[9.5px] text-[#ff9900] font-bold uppercase">
                  NEWTON-RAPHSON CONVERGENCE LOG:
                </div>
                <div className="space-y-1 text-[9.5px]">
                  {step.iterations.map((it) => (
                    <div key={it.iteration} className="flex justify-between border-b border-[#141414] pb-0.5">
                      <span className="text-[#ff9900] font-bold">ITER {String(it.iteration).padStart(2, "0")}</span>
                      <span className="text-[#888888]">z = {it.z?.toFixed(6)}</span>
                      <span className="text-[#cccccc]">TOF = {it.tof_calculated_s?.toLocaleString(undefined, { maximumFractionDigits: 1 })} s</span>
                      <span className="text-[#888888]">residual = {it.residual?.toExponential(4)}</span>
                      <span className={it.status === "CONVERGED" ? "text-[#44bb66] font-bold" : "text-[#ffaa00]"}>
                        {it.status}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Final Step Result */}
            <div className="flex items-center space-x-1.5 text-xs bg-[#0c140c] border border-[#44bb66]/40 p-2 rounded text-[#44bb66] font-bold">
              <CheckCircle2 className="w-3.5 h-3.5 text-[#44bb66] shrink-0" />
              <div className="flex items-center space-x-1.5 flex-wrap">
                <span>✓ RESULT:</span>
                <ScientificValue value={step.result} />
              </div>
            </div>

            {/* Plain English "What Just Happened?" & Aerospace Rigor */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 pt-1 text-[10px]">
              <div className="bg-[#0a0a0a] border border-[#1c1c1c] p-2 rounded space-y-0.5">
                <div className="flex items-center space-x-1 text-[#ff9900] font-bold text-[9.5px]">
                  <HelpCircle className="w-3 h-3" />
                  <span>INTUITION</span>
                </div>
                <p className="text-[#aaaaaa] leading-relaxed">
                  <ScientificValue value={step.beginnerExplanation || step.explanation} />
                </p>
              </div>

              <div className="bg-[#0a0a0a] border border-[#1c1c1c] p-2 rounded space-y-0.5">
                <div className="flex items-center space-x-1 text-[#cccccc] font-bold text-[9.5px]">
                  <BookOpen className="w-3 h-3" />
                  <span>PHYSICAL PRINCIPLE</span>
                </div>
                <p className="text-[#777777] leading-relaxed">
                  <ScientificValue value={step.scientificNotes || step.explanation} />
                </p>
              </div>
            </div>

          </div>
        ))}

      </div>
    </div>
  );
};
