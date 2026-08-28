/**
 * THESEUS Collapsible Panel Component
 * ===================================
 * Reusable panel shell for Mission/Objects, Telemetry, Visualization Layers, and Timeline.
 * Features:
 *   - Edge collapse handle with Chevron icons
 *   - Smooth CSS transitions
 *   - Information-dense header bar
 *   - Allows viewport to dynamically expand into freed space
 */

import React from 'react';
import { ChevronLeft, ChevronRight, ChevronUp, ChevronDown } from 'lucide-react';

export interface CollapsiblePanelProps {
  id: string;
  title: string;
  icon?: React.ReactNode;
  side: 'left' | 'right' | 'bottom';
  isCollapsed: boolean;
  onToggle: () => void;
  headerControls?: React.ReactNode;
  children: React.ReactNode;
  width?: string; // e.g. "w-80" or "w-96"
  height?: string; // e.g. "h-32" for bottom
}

export const CollapsiblePanel: React.FC<CollapsiblePanelProps> = ({
  id,
  title,
  icon,
  side,
  isCollapsed,
  onToggle,
  headerControls,
  children,
  width = 'w-80',
  height = 'h-36',
}) => {
  if (side === 'bottom') {
    return (
      <div
        className={`relative transition-all duration-300 ease-in-out border-t border-[#221d17] bg-[#070d18]/95 backdrop-blur-md flex flex-col font-mono text-xs shadow-2xl z-20 ${
          isCollapsed ? 'h-9' : height
        }`}
      >
        {/* Toggle Bar */}
        <div className="flex items-center justify-between px-3 py-1.5 bg-[#05080f] border-b border-[#221d17] select-none shrink-0">
          <div className="flex items-center space-x-2">
            {icon}
            <span className="font-['Orbitron'] font-bold text-[11px] tracking-wider text-[#e6dfd5] uppercase">
              {title}
            </span>
          </div>

          <div className="flex items-center space-x-2">
            {headerControls}
            <button
              onClick={onToggle}
              className="p-1 rounded bg-[#0b1424] hover:bg-[#152238] border border-[#221d17] text-[#ff9900] transition-colors cursor-pointer"
              title={isCollapsed ? 'Expand Panel' : 'Collapse Panel'}
            >
              {isCollapsed ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>

        {/* Panel Content */}
        {!isCollapsed && <div className="flex-1 overflow-hidden">{children}</div>}
      </div>
    );
  }

  return (
    <div
      className={`relative transition-all duration-300 ease-in-out border-${
        side === 'left' ? 'r' : 'l'
      } border-[#221d17] bg-[#070d18]/95 backdrop-blur-md flex flex-col font-mono text-xs shadow-2xl z-20 ${
        isCollapsed ? 'w-9' : width
      }`}
    >
      {/* Header Bar */}
      <div className="flex items-center justify-between px-3 py-2 bg-[#05080f] border-b border-[#221d17] select-none shrink-0 overflow-hidden">
        {!isCollapsed && (
          <div className="flex items-center space-x-2 truncate">
            {icon}
            <span className="font-['Orbitron'] font-bold text-[11px] tracking-wider text-[#e6dfd5] uppercase truncate">
              {title}
            </span>
          </div>
        )}

        <div className="flex items-center space-x-1.5 ml-auto">
          {!isCollapsed && headerControls}
          <button
            onClick={onToggle}
            className="p-1 rounded bg-[#0b1424] hover:bg-[#152238] border border-[#221d17] text-[#ff9900] transition-colors cursor-pointer"
            title={isCollapsed ? 'Expand Panel' : 'Collapse Panel'}
          >
            {side === 'left' ? (
              isCollapsed ? <ChevronRight className="w-3.5 h-3.5" /> : <ChevronLeft className="w-3.5 h-3.5" />
            ) : (
              isCollapsed ? <ChevronLeft className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />
            )}
          </button>
        </div>
      </div>

      {/* Panel Content */}
      {!isCollapsed && <div className="flex-1 overflow-y-auto p-3">{children}</div>}

      {/* Vertical Title when collapsed */}
      {isCollapsed && (
        <div className="flex-1 flex items-center justify-center py-4 select-none">
          <span
            className="font-['Orbitron'] text-[10px] font-bold text-[#8c8275] tracking-widest uppercase cursor-pointer"
            style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}
            onClick={onToggle}
          >
            {title}
          </span>
        </div>
      )}
    </div>
  );
};
