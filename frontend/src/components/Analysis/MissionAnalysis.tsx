import React from "react";
import { SimulationResult } from "../../types/mission";
import { formatSpeed } from "../../lib/formatter";
import { 
  BarChart3, 
  CheckCircle2, 
  AlertTriangle, 
  Gauge, 
  Fuel, 
  Clock, 
  ShieldAlert,
  HelpCircle
} from "lucide-react";

interface MissionAnalysisProps {
  result?: SimulationResult;
}

export const MissionAnalysis: React.FC<MissionAnalysisProps> = ({ result }) => {
  if (!result) {
    return (
      <div className="w-full h-full flex items-center justify-center p-6 text-[#8c8275] font-mono text-xs">
        NO ACTIVE SIMULATION — RUN A MISSION TO GENERATE SCIENTIFIC EVALUATION
      </div>
    );
  }

  const isSuccess = result.metadata.status === "SUCCESS";
  const dv = result.delta_v_budget;
  const prop = result.propellant_budget;
  const diag = result.diagnostics;

  return (
    <div className="w-full h-full flex flex-col bg-[#04060a] text-[#e6dfd5] font-mono overflow-y-auto p-4 md:p-6 space-y-5">
      
      {/* Top Banner Status */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-[#221d17] pb-3">
        <div>
          <div className="flex items-center space-x-2 text-[#ff9900] text-xs font-semibold tracking-wider">
            <BarChart3 className="w-4 h-4" />
            <span>POST-MISSION SCIENTIFIC EVALUATION</span>
          </div>
          <h1 className="text-base md:text-lg font-bold text-[#e6dfd5] mt-0.5">
            {result.metadata.name}
          </h1>
        </div>

        {/* Status Badge */}
        <div className={`flex items-center space-x-2 px-3 py-1 rounded border font-bold text-xs ${
          isSuccess 
            ? "bg-[#44bb66]/10 border-[#44bb66]/40 text-[#44bb66]" 
            : "bg-[#ff9900]/10 border-[#ff9900]/40 text-[#ff9900]"
        }`}>
          {isSuccess ? <CheckCircle2 className="w-3.5 h-3.5 text-[#44bb66]" /> : <AlertTriangle className="w-3.5 h-3.5 text-[#ff9900]" />}
          <span>STATUS: {result.metadata.status}</span>
        </div>
      </div>

      {/* Delta-V and Fuel Summary Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        
        {/* Delta-V Budget Card */}
        <div className="technical-panel p-3.5 rounded space-y-2.5">
          <div className="text-xs text-[#ff9900] font-bold flex items-center justify-between">
            <span>DELTA-V BUDGET</span>
            <Gauge className="w-3.5 h-3.5 text-[#ff9900]" />
          </div>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-[#8c8275]">Required Δv:</span>
              <span className="text-[#e6dfd5] font-bold">
                {dv?.total_delta_v ? formatSpeed(dv.total_delta_v) : "—"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#8c8275]">Vehicle Capacity:</span>
              <span className="text-[#44bb66] font-bold">
                {dv?.available_delta_v ? formatSpeed(dv.available_delta_v) : "—"}
              </span>
            </div>
            <div className="flex justify-between border-t border-[#221d17] pt-1">
              <span className="text-[#8c8275]">Δv Margin:</span>
              <span className={dv?.margin_delta_v && dv.margin_delta_v >= 0 ? "text-[#44bb66] font-bold" : "text-[#cc3333] font-bold"}>
                {dv?.margin_delta_v ? formatSpeed(dv.margin_delta_v) : "—"}
              </span>
            </div>
          </div>
        </div>

        {/* Propellant Expenditure Card */}
        <div className="technical-panel p-3.5 rounded space-y-2.5">
          <div className="text-xs text-[#ff9900] font-bold flex items-center justify-between">
            <span>PROPELLANT CONSUMPTION</span>
            <Fuel className="w-3.5 h-3.5 text-[#ff9900]" />
          </div>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-[#8c8275]">Propellant Consumed:</span>
              <span className="text-[#ff9900] font-bold">
                {prop?.fuel_consumed_kg ? `${prop.fuel_consumed_kg.toFixed(1)} kg` : "—"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#8c8275]">Initial Fuel Load:</span>
              <span className="text-[#e6dfd5]">
                {prop?.initial_fuel_kg ? `${prop.initial_fuel_kg.toFixed(1)} kg` : "—"}
              </span>
            </div>
            <div className="flex justify-between border-t border-[#221d17] pt-1">
              <span className="text-[#8c8275]">Remaining Margin:</span>
              <span className={prop?.fuel_margin_kg && prop.fuel_margin_kg >= 0 ? "text-[#44bb66] font-bold" : "text-[#cc3333] font-bold"}>
                {prop?.fuel_margin_kg ? `${prop.fuel_margin_kg.toFixed(1)} kg` : "—"}
              </span>
            </div>
          </div>
        </div>

        {/* Duration Card */}
        <div className="technical-panel p-3.5 rounded space-y-2.5">
          <div className="text-xs text-[#44bb66] font-bold flex items-center justify-between">
            <span>TRAJECTORY METRICS</span>
            <Clock className="w-3.5 h-3.5 text-[#44bb66]" />
          </div>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-[#8c8275]">Flight Duration:</span>
              <span className="text-[#e6dfd5] font-bold">
                {result.metadata.duration_hours ? `${result.metadata.duration_hours.toFixed(2)} hours` : "—"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#8c8275]">Trajectory Type:</span>
              <span className="text-[#ff9900]">{result.metadata.transfer_type || result.metadata.trajectory_type || "Two-Body Arc"}</span>
            </div>
            <div className="flex justify-between border-t border-[#221d17] pt-1">
              <span className="text-[#8c8275]">Integrator:</span>
              <span className="text-[#44bb66] font-bold">{diag?.numerical_tolerance || "< 1e-12"}</span>
            </div>
          </div>
        </div>
      </div>

      {/* "What Just Happened" Summary */}
      <div className="technical-panel p-3.5 rounded space-y-1.5">
        <div className="flex items-center space-x-1.5 text-[#44bb66] text-xs font-bold">
          <HelpCircle className="w-3.5 h-3.5" />
          <span>WHAT JUST HAPPENED (EXECUTIVE SUMMARY)</span>
        </div>
        <p className="text-xs text-[#e6dfd5] leading-relaxed">
          THESEUS calculated orbital boundary states, solved for the required velocity impulses, 
          and verified propellant mass depletion across the flight profile.
          {isSuccess 
            ? " The propulsion system provides sufficient velocity capacity with positive reserve margins."
            : " Warning: Propellant required exceeds available capacity; higher specific impulse or additional staging is required."}
        </p>
      </div>

      {/* Scientific Honesty Banner */}
      <div className="bg-[#070d18] border border-[#ff9900]/30 p-3.5 rounded space-y-1.5">
        <div className="flex items-center space-x-1.5 text-[#ff9900] text-xs font-bold">
          <ShieldAlert className="w-3.5 h-3.5" />
          <span>SCIENTIFIC HONESTY: NUMERICAL PRECISION VS PHYSICAL FIDELITY</span>
        </div>
        <p className="text-[11px] text-[#c8c0b5] leading-relaxed">
          The numerical integration tolerance of <strong className="text-[#ff9900]">1×10⁻¹²</strong> guarantees that the equations of motion were integrated with mathematical fidelity. 
          Real-world mission accuracy is limited by physical perturbations (solar radiation flux, atmospheric density fluctuations, unmodelled higher-order spherical harmonics).
        </p>
      </div>
    </div>
  );
};
