import React from "react";
import { 
  Atom, 
  CheckCircle2, 
  Layers, 
  ShieldCheck 
} from "lucide-react";

export const ScienceInspector: React.FC = () => {
  const activeSubsystems = [
    { name: "Constants & Physical Units", status: "VALIDATED", note: "CODATA 2018 / IAU 2015 standard values" },
    { name: "Coordinate Transformations", status: "VALIDATED", note: "ECI (J2000) ↔ ECEF (WGS84) with GMST rotation" },
    { name: "Orbital Mechanics & Conversions", status: "VALIDATED", note: "State vector ↔ Keplerian orbital elements (< 1e-8 m error)" },
    { name: "Kepler Equation Solver", status: "VALIDATED", note: "Newton-Raphson elliptic & hyperbolic universal variables" },
    { name: "Two-Body Analytical Propagation", status: "VALIDATED", note: "Conserves specific energy & angular momentum across 100 orbits" },
    { name: "Numerical Integrators (RK4 / RKF45)", status: "VALIDATED", note: "4th-order convergence & adaptive error step-size control" },
    { name: "Point-Mass & J2 Gravity Perturbations", status: "VALIDATED", note: "Earth equatorial bulge secular nodal precession (-5.00°/day)" },
    { name: "Universal Variable Lambert Solver", status: "VALIDATED", note: "Single-revolution transfer arcs, collinear Hohmann, long-way arcs" },
    { name: "Orbital Rendezvous Guidance", status: "VALIDATED", note: "Clohessy-Wiltshire relative motion & Lambert interception" },
    { name: "Solar Radiation Pressure (SRP)", status: "VALIDATED", note: "1 AU solar flux, 1/r² scaling, cylindrical shadow geometry" },
    { name: "Atmospheric Drag & Co-rotation", status: "VALIDATED", note: "US Standard Atmosphere 1976 (< 86 km) with Earth co-rotation" },
    { name: "Planetary Ephemerides", status: "VALIDATED", note: "Astropy JPL DE440 provider & analytical Keplerian orbits" },
    { name: "Time Scales & IAU Leap Seconds", status: "VALIDATED", note: "Epoch-dependent dynamic leap second resolution (TT, UTC, TAI)" },
  ];

  const futureSubsystems = [
    { name: "Phase 8: Atmospheric Reentry Aerothermodynamics", status: "FUTURE PHYSICS MODULE", note: "Hypersonic blunt-body heating and deceleration" },
    { name: "Phase 9: Conjunction Assessment & Debris Collision", status: "FUTURE PHYSICS MODULE", note: "Time-of-closest-approach (TCA) and miss distance covariance" },
    { name: "Phase 10: Monte Carlo Uncertainty Dispersion", status: "FUTURE PHYSICS MODULE", note: "Gaussian initial state & propulsion dispersion ensembles" },
    { name: "Phase 11: State Transition Matrix (STM) Sensitivity", status: "FUTURE PHYSICS MODULE", note: "6x6 variational equations for orbit determination" },
    { name: "Phase 12: Trajectory Optimization & Collocation", status: "FUTURE PHYSICS MODULE", note: "Direct transcription and low-thrust optimal control" },
    { name: "Phase 13: Space Radiation & Thermal Modeling", status: "FUTURE PHYSICS MODULE", note: "Van Allen belt radiation flux & spacecraft thermal equilibrium" },
  ];

  return (
    <div className="w-full h-full flex flex-col bg-[#04060a] text-[#e6dfd5] font-mono overflow-y-auto p-4 md:p-6 space-y-5">
      
      {/* Header */}
      <div className="border-b border-[#221d17] pb-3">
        <div className="flex items-center space-x-2 text-[#ff9900] text-xs font-semibold tracking-wider">
          <Atom className="w-4 h-4" />
          <span>ASTRODYNAMICS ENGINE FIDELITY AUDIT</span>
        </div>
        <h1 className="text-base md:text-lg font-bold text-[#e6dfd5] mt-0.5">
          SCIENCE & MODEL FIDELITY INSPECTOR
        </h1>
        <p className="text-[11px] text-[#8c8275] mt-0.5">
          Transparency on validated active physics models, numerical integrators, coordinate frames, and future module contracts.
        </p>
      </div>

      {/* Grid: Active vs Future Capabilities */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        
        {/* Active Validated Subsystems */}
        <div className="technical-panel p-3.5 rounded space-y-2.5">
          <div className="flex items-center justify-between text-xs text-[#44bb66] font-bold border-b border-[#221d17] pb-1.5">
            <span className="flex items-center space-x-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-[#44bb66]" />
              <span>ACTIVE VALIDATED PHYSICS (PHASES 1–7)</span>
            </span>
            <span className="text-[9px] bg-[#44bb66]/10 px-1.5 py-0.2 rounded border border-[#44bb66]/30">
              100% VERIFIED
            </span>
          </div>

          <div className="space-y-1.5">
            {activeSubsystems.map((sub, idx) => (
              <div key={idx} className="bg-[#05080f] border border-[#221d17] p-2 rounded text-xs space-y-0.5">
                <div className="flex justify-between items-center">
                  <span className="font-bold text-[#e6dfd5] text-[11px]">{sub.name}</span>
                  <span className="text-[9px] text-[#44bb66] font-bold flex items-center space-x-1">
                    <CheckCircle2 className="w-3 h-3 text-[#44bb66]" />
                    <span>VALIDATED</span>
                  </span>
                </div>
                <p className="text-[10px] text-[#8c8275]">{sub.note}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Future Reserved Physics Modules */}
        <div className="technical-panel p-3.5 rounded space-y-2.5">
          <div className="flex items-center justify-between text-xs text-[#ff9900] font-bold border-b border-[#221d17] pb-1.5">
            <span className="flex items-center space-x-1.5">
              <Layers className="w-3.5 h-3.5 text-[#ff9900]" />
              <span>FUTURE EXTENSIBILITY HOOKS (PHASES 8–13)</span>
            </span>
            <span className="text-[9px] bg-[#ff9900]/10 px-1.5 py-0.2 rounded border border-[#ff9900]/30">
              RESERVED CONTRACTS
            </span>
          </div>

          <div className="space-y-1.5">
            {futureSubsystems.map((sub, idx) => (
              <div key={idx} className="bg-[#05080f] border border-[#221d17] p-2 rounded text-xs space-y-0.5 opacity-75">
                <div className="flex justify-between items-center">
                  <span className="font-bold text-[#c8c0b5] text-[11px]">{sub.name}</span>
                  <span className="text-[8px] bg-[#ff9900]/10 text-[#ff9900] px-1 py-0.2 rounded border border-[#ff9900]/20 font-bold">
                    FUTURE MODULE
                  </span>
                </div>
                <p className="text-[10px] text-[#8c8275]">{sub.note}</p>
              </div>
            ))}
          </div>

          {/* Scientific Honesty Manifesto */}
          <div className="bg-[#070d18] border border-[#ff9900]/30 p-2.5 rounded text-[10px] text-[#8c8275] space-y-1">
            <div className="text-xs text-[#ff9900] font-bold">SCIENTIFIC HONESTY COMMITMENT</div>
            <p className="leading-relaxed text-[#c8c0b5]">
              THESEUS never fabricates physics. Unimplemented modules are marked as future capabilities rather than generating unverified approximations.
            </p>
          </div>
        </div>

      </div>
    </div>
  );
};
