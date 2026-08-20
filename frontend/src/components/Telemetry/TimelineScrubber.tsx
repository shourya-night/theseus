import React from "react";
import { MissionEvent } from "../../types/mission";
import { formatMET } from "../../lib/formatter";
import { 
  Play, 
  Pause, 
  RotateCcw, 
  ChevronRight, 
  ChevronLeft, 
  FastForward 
} from "lucide-react";

interface TimelineScrubberProps {
  currentFrameIdx: number;
  totalFrames: number;
  currentTimeSeconds: number;
  totalDurationSeconds: number;
  isPlaying: boolean;
  playbackSpeed: number;
  events: MissionEvent[];
  onSeek: (frameIdx: number) => void;
  onTogglePlay: () => void;
  onSetSpeed: (speed: number) => void;
  onReset: () => void;
}

export const TimelineScrubber: React.FC<TimelineScrubberProps> = ({
  currentFrameIdx,
  totalFrames,
  currentTimeSeconds,
  totalDurationSeconds,
  isPlaying,
  playbackSpeed,
  events,
  onSeek,
  onTogglePlay,
  onSetSpeed,
  onReset,
}) => {
  const speeds = [0.1, 1, 10, 100, 1000];

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onSeek(Number(e.target.value));
  };

  const handleEventClick = (eventTime: number) => {
    if (totalDurationSeconds <= 0 || totalFrames <= 0) return;
    const targetFraction = Math.min(1.0, Math.max(0.0, eventTime / totalDurationSeconds));
    const targetIdx = Math.floor(targetFraction * (totalFrames - 1));
    onSeek(targetIdx);
  };

  return (
    <div className="w-full bg-[#05080f] border-t border-[#221d17] p-2.5 text-xs font-mono select-none space-y-1.5 shrink-0 z-30">
      
      {/* Event Timeline Track */}
      <div className="relative w-full h-3.5 flex items-center">
        {events && events.map((ev, idx) => {
          const frac = totalDurationSeconds > 0 ? (ev.time / totalDurationSeconds) : 0;
          const leftPercent = Math.min(99, Math.max(1, frac * 100));
          return (
            <button
              key={idx}
              onClick={() => handleEventClick(ev.time)}
              title={`${ev.name} (${formatMET(ev.time)}) - ${ev.details}`}
              className="absolute -top-0.5 transform -translate-x-1/2 flex flex-col items-center group z-20"
              style={{ left: `${leftPercent}%` }}
            >
              <div className="w-2 h-2 rounded-none bg-[#ff9900] border border-[#04060a] group-hover:scale-150 transition-all" />
              <span className="opacity-0 group-hover:opacity-100 absolute bottom-3 bg-[#070d18] border border-[#ff9900] text-[9px] text-[#ff9900] px-1.5 py-0.5 rounded whitespace-nowrap z-30 pointer-events-none transition-opacity">
                {ev.name}
              </span>
            </button>
          );
        })}

        {/* Scrub Slider Input */}
        <input
          type="range"
          min="0"
          max={Math.max(1, totalFrames - 1)}
          value={currentFrameIdx}
          onChange={handleSliderChange}
          className="w-full h-1 bg-[#0b1424] rounded-none cursor-pointer accent-[#ff9900] z-10"
        />
      </div>

      {/* Control Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        
        {/* Left: MET & Time Elapsed */}
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1">
            <span className="text-[#8c8275]">MET:</span>
            <span className="text-[#ff9900] font-bold tracking-wider">
              {formatMET(currentTimeSeconds)}
            </span>
          </div>
          <span className="text-[#332b22]">/</span>
          <div className="text-[#8c8275]">
            {formatMET(totalDurationSeconds)}
          </div>
        </div>

        {/* Center: Play, Pause & Step Buttons */}
        <div className="flex items-center space-x-1.5 bg-[#070d18] border border-[#221d17] p-0.5 rounded">
          <button
            onClick={onReset}
            title="Reset"
            className="p-1 hover:bg-[#0e1726] rounded text-[#8c8275] hover:text-[#e6dfd5]"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => onSeek(Math.max(0, currentFrameIdx - 1))}
            title="Step Back"
            className="p-1 hover:bg-[#0e1726] rounded text-[#8c8275] hover:text-[#e6dfd5]"
          >
            <ChevronLeft className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={onTogglePlay}
            className="px-3 py-0.5 bg-[#ff9900]/20 hover:bg-[#ff9900]/30 text-[#ff9900] font-bold rounded flex items-center space-x-1 border border-[#ff9900]/50"
          >
            {isPlaying ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3 fill-current" />}
            <span className="text-[11px]">{isPlaying ? "PAUSE" : "PLAY"}</span>
          </button>
          <button
            onClick={() => onSeek(Math.min(totalFrames - 1, currentFrameIdx + 1))}
            title="Step Forward"
            className="p-1 hover:bg-[#0e1726] rounded text-[#8c8275] hover:text-[#e6dfd5]"
          >
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Right: Playback Speed Selector */}
        <div className="flex items-center space-x-1 bg-[#070d18] border border-[#221d17] p-0.5 rounded">
          <FastForward className="w-3 h-3 text-[#8c8275] ml-1 mr-0.5" />
          {speeds.map((spd) => (
            <button
              key={spd}
              onClick={() => onSetSpeed(spd)}
              className={`px-1.5 py-0.2 rounded text-[10px] ${
                playbackSpeed === spd
                  ? "bg-[#ff9900]/20 text-[#ff9900] font-bold"
                  : "text-[#8c8275] hover:text-[#e6dfd5]"
              }`}
            >
              {spd}x
            </button>
          ))}
        </div>

      </div>
    </div>
  );
};
