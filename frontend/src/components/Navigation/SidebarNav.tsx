import React from "react";
import { soundEngine } from "../../lib/audio";
import { 
  Compass, 
  Orbit, 
  Binary, 
  Activity, 
  BarChart3, 
  Atom, 
  Cpu 
} from "lucide-react";

interface SidebarNavProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  audioEnabled: boolean;
}

export const SidebarNav: React.FC<SidebarNavProps> = ({
  activeTab,
  setActiveTab,
  audioEnabled,
}) => {
  const navItems = [
    { id: "mission", label: "MISSION CONFIG", icon: Compass, number: "01" },
    { id: "simulate", label: "2D SANDBOX", icon: Orbit, number: "02" },
    { id: "math", label: "SOLVE STREAM", icon: Binary, number: "03" },
    { id: "telemetry", label: "TELEMETRY HUD", icon: Activity, number: "04" },
    { id: "analysis", label: "EVALUATION", icon: BarChart3, number: "05" },
    { id: "science", label: "SCIENCE AUDIT", icon: Atom, number: "06" },
    { id: "system", label: "DEMO MISSIONS", icon: Cpu, number: "07" },
  ];

  return (
    <aside className="w-52 lg:w-56 bg-[#070d18] border-r border-[#221d17] flex flex-col justify-between p-3 select-none shrink-0 z-30">
      
      {/* Navigation Links */}
      <div className="space-y-1">
        <div className="text-[10px] font-mono text-[#8c8275] px-2 py-1 tracking-wider uppercase">
          MISSION CONSOLE
        </div>

        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => {
                setActiveTab(item.id);
                if (audioEnabled) soundEngine.playTerminalBeep(750, 0.04);
              }}
              className={`w-full flex items-center justify-between px-3 py-2 rounded text-xs font-mono tracking-wide transition-all border text-left ${
                isActive
                  ? "bg-[#ff9900]/15 border-[#ff9900] text-[#ff9900] font-bold"
                  : "bg-[#0b1424] border-[#221d17] text-[#c8c0b5] hover:text-[#e6dfd5] hover:border-[#332b22]"
              }`}
            >
              <div className="flex items-center space-x-2.5">
                <span className={`text-[10px] ${isActive ? "text-[#ff9900]" : "text-[#8c8275]"}`}>
                  {item.number}
                </span>
                <Icon className={`w-3.5 h-3.5 shrink-0 ${isActive ? "text-[#ff9900]" : "text-[#8c8275]"}`} />
                <span className="truncate">{item.label}</span>
              </div>
              {isActive && <div className="w-1.5 h-1.5 rounded-full bg-[#ff9900]" />}
            </button>
          );
        })}
      </div>

      {/* Subsystem Summary Box */}
      <div className="bg-[#05080f] border border-[#221d17] p-2.5 rounded text-[10px] font-mono text-[#8c8275] space-y-1">
        <div className="flex justify-between items-center text-[#8c8275]">
          <span>INTEGRATOR:</span>
          <span className="text-[#44bb66] font-bold">RKF45 7-DOF</span>
        </div>
        <div className="flex justify-between items-center text-[#8c8275]">
          <span>TOLERANCE:</span>
          <span className="text-[#e6dfd5]">1×10⁻¹²</span>
        </div>
        <div className="flex justify-between items-center text-[#8c8275]">
          <span>EPHEMERIS:</span>
          <span className="text-[#ff9900]">JPL DE440</span>
        </div>
      </div>
    </aside>
  );
};
