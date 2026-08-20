import React, { useState, useEffect } from "react";
import { formatMET } from "../../lib/formatter";
import { soundEngine } from "../../lib/audio";
import { 
  Tv, 
  Volume2, 
  VolumeX, 
  CheckCircle2, 
  AlertCircle 
} from "lucide-react";

interface HeaderProps {
  simulationTime: number;
  backendOnline: boolean;
  crtEnabled: boolean;
  setCrtEnabled: (enabled: boolean) => void;
  audioEnabled: boolean;
  setAudioEnabled: (enabled: boolean) => void;
  activeTabTitle: string;
}

export const Header: React.FC<HeaderProps> = ({
  simulationTime,
  backendOnline,
  crtEnabled,
  setCrtEnabled,
  audioEnabled,
  setAudioEnabled,
  activeTabTitle,
}) => {
  const [utcTime, setUtcTime] = useState<string>("");

  useEffect(() => {
    const updateClock = () => {
      const now = new Date();
      setUtcTime(now.toISOString().replace("T", " ").slice(0, 19) + " UTC");
    };
    updateClock();
    const interval = setInterval(updateClock, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleAudioToggle = () => {
    const next = !audioEnabled;
    setAudioEnabled(next);
    soundEngine.enabled = next;
    if (next) soundEngine.playTerminalBeep(980, 0.08);
  };

  return (
    <header className="w-full h-12 bg-[#05080f] border-b border-[#221d17] flex items-center justify-between px-4 select-none shrink-0 z-40">
      
      {/* Left: Brand & Status */}
      <div className="flex items-center space-x-3 text-xs font-mono">
        <div className="flex items-center space-x-2">
          <span className="w-2 h-2 rounded-full bg-[#ff9900]" />
          <span className="font-['Orbitron'] font-bold tracking-widest text-[#ff9900] text-sm">
            THESEUS
          </span>
          <span className="text-[10px] text-[#8c8275] bg-[#070d18] px-1.5 py-0.5 rounded border border-[#221d17]">
            v0.1.0
          </span>
        </div>

        <span className="text-[#221d17]">|</span>

        <div className="hidden sm:flex items-center space-x-1 text-[#c8c0b5] text-[11px]">
          <span className="text-[#8c8275]">SUBSYSTEM:</span>
          <span className="font-semibold text-[#e6dfd5]">FLIGHT DYNAMICS CORE</span>
        </div>

        <span className="hidden sm:inline text-[#221d17]">|</span>

        {/* Backend Online Status */}
        <div className="flex items-center space-x-1 text-[11px]">
          {backendOnline ? (
            <span className="flex items-center space-x-1 text-[#44bb66] font-bold">
              <CheckCircle2 className="w-3.5 h-3.5 text-[#44bb66]" />
              <span>API ONLINE</span>
            </span>
          ) : (
            <span className="flex items-center space-x-1 text-[#ff9900]">
              <AlertCircle className="w-3.5 h-3.5 text-[#ff9900]" />
              <span>CLIENT ENGINE</span>
            </span>
          )}
        </div>
      </div>

      {/* Center: Active View Title */}
      <div className="hidden md:flex items-center space-x-2 text-xs font-mono">
        <span className="text-[#8c8275]">MODE:</span>
        <span className="text-[#ff9900] font-bold tracking-wider">
          {activeTabTitle}
        </span>
      </div>

      {/* Right: Clocks & Toggles */}
      <div className="flex items-center space-x-3 text-xs font-mono">
        
        {/* MET Clock */}
        <div className="flex items-center space-x-1.5 bg-[#070d18] border border-[#221d17] px-2.5 py-0.5 rounded">
          <span className="text-[#8c8275] text-[10px] font-bold">MET:</span>
          <span className="font-bold text-[#ff9900] tracking-wider text-xs">
            {formatMET(simulationTime)}
          </span>
        </div>

        {/* UTC Clock */}
        <div className="hidden lg:flex items-center space-x-1.5 text-[#c8c0b5] text-[11px]">
          <span className="text-[#8c8275]">UTC:</span>
          <span className="text-[#e6dfd5]">{utcTime}</span>
        </div>

        {/* CRT & Audio Toggles */}
        <div className="flex items-center space-x-1 pl-2 border-l border-[#221d17]">
          <button
            onClick={() => setCrtEnabled(!crtEnabled)}
            title="Toggle CRT Filter"
            className={`p-1.5 rounded transition-all text-xs flex items-center space-x-1 border ${
              crtEnabled
                ? "bg-[#ff9900]/15 border-[#ff9900]/50 text-[#ff9900]"
                : "bg-[#070d18] border-[#221d17] text-[#8c8275] hover:text-[#e6dfd5]"
            }`}
          >
            <Tv className="w-3.5 h-3.5" />
            <span className="text-[10px] font-bold hidden sm:inline">CRT</span>
          </button>

          <button
            onClick={handleAudioToggle}
            title="Toggle Audio"
            className={`p-1.5 rounded transition-all text-xs flex items-center space-x-1 border ${
              audioEnabled
                ? "bg-[#44bb66]/15 border-[#44bb66]/50 text-[#44bb66]"
                : "bg-[#070d18] border-[#221d17] text-[#8c8275] hover:text-[#e6dfd5]"
            }`}
          >
            {audioEnabled ? <Volume2 className="w-3.5 h-3.5" /> : <VolumeX className="w-3.5 h-3.5" />}
            <span className="text-[10px] font-bold hidden sm:inline">AUDIO</span>
          </button>
        </div>

      </div>
    </header>
  );
};
