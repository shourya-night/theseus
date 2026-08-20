import React from "react";
import { 
  Sparkles, 
  Play 
} from "lucide-react";

interface DemoLauncherProps {
  onLaunchDemo: (demoId: string) => void;
  isLoading: boolean;
}

export const DemoLauncher: React.FC<DemoLauncherProps> = ({
  onLaunchDemo,
  isLoading,
}) => {
  const demos = [
    {
      id: "earth-moon",
      title: "Earth → Moon (Translunar Injection)",
      origin: "Earth (300 km LEO)",
      destination: "Moon (384,400 km orbit)",
      vehicle: "ISRO Chandrayaan-3 Propulsion Module",
      delta_v: "3.13 km/s",
      duration: "119.5 hours (~5.0 days)",
      description: "Translunar transfer orbit. Injects spacecraft from parking orbit onto an elliptical intercept arc tangent to lunar distance.",
      badge: "SHOWCASE 01",
    },
    {
      id: "leo-geo",
      title: "LEO → GEO Transfer (Combined Plane Change)",
      origin: "LEO (300 km, 28.5° inc)",
      destination: "GEO (35,786 km, 0.0° inc)",
      vehicle: "SpaceX Dragon / Upper Stage",
      delta_v: "4.26 km/s",
      duration: "5.27 hours",
      description: "Combines apogee circularization with a 28.5° orbital inclination reduction into a single optimal maneuver vector.",
      badge: "SHOWCASE 02",
    },
    {
      id: "leo-rendezvous",
      title: "LEO Orbital Rendezvous & Interception",
      origin: "Chaser Orbit (400 km)",
      destination: "Target Station (420 km, 60° lead)",
      vehicle: "Autonomous Crew & Cargo Vehicle",
      delta_v: "0.24 km/s",
      duration: "1.0 hour",
      description: "Chaser executes a two-impulse Lambert rendezvous burn, closing distance and nulling relative velocity at docking.",
      badge: "SHOWCASE 03",
    },
    {
      id: "earth-mars",
      title: "Earth → Mars Interplanetary Transfer",
      origin: "Earth (1.000 AU)",
      destination: "Mars (1.524 AU)",
      vehicle: "Deep Space Chemical Explorer",
      delta_v: "5.76 km/s",
      duration: "260 days",
      description: "Universal-variable Lambert boundary solve across heliocentric orbital geometry, solving departure hyperbola and Mars capture.",
      badge: "SHOWCASE 04",
    },
  ];

  return (
    <div className="w-full h-full flex flex-col bg-[#04060a] text-[#e6dfd5] font-mono overflow-y-auto p-4 md:p-6 space-y-5">
      
      {/* Header */}
      <div className="border-b border-[#221d17] pb-3">
        <div className="flex items-center space-x-2 text-[#ff9900] text-xs font-semibold tracking-wider">
          <Sparkles className="w-4 h-4" />
          <span>1-CLICK GUIDED DEMONSTRATIONS</span>
        </div>
        <h1 className="text-base md:text-lg font-bold text-[#e6dfd5] mt-0.5">
          SHOWCASE DEMO FLIGHT PROFILES
        </h1>
        <p className="text-[11px] text-[#8c8275] mt-0.5">
          Curated, fully validated astrodynamic flight profiles ready for astronaut presentations, classroom lectures, and public review.
        </p>
      </div>

      {/* Demo Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {demos.map((demo) => (
          <div
            key={demo.id}
            className="technical-panel p-4 rounded space-y-3 hover:border-[#ff9900] transition-all flex flex-col justify-between"
          >
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-[9px] font-bold px-2 py-0.5 rounded bg-[#05080f] border border-[#221d17] text-[#8c8275]">
                  {demo.badge}
                </span>
                <span className="text-xs font-bold text-[#44bb66]">Δv: {demo.delta_v}</span>
              </div>

              <h2 className="text-sm font-bold text-[#e6dfd5]">
                {demo.title}
              </h2>

              <p className="text-xs text-[#c8c0b5] leading-relaxed">
                {demo.description}
              </p>

              <div className="bg-[#05080f] border border-[#221d17] p-2.5 rounded text-[10px] space-y-1">
                <div className="flex justify-between">
                  <span className="text-[#8c8275]">Origin:</span>
                  <span className="text-[#e6dfd5] font-bold">{demo.origin}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#8c8275]">Target:</span>
                  <span className="text-[#ff9900] font-bold">{demo.destination}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#8c8275]">Vehicle:</span>
                  <span className="text-[#c8c0b5]">{demo.vehicle}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#8c8275]">Duration:</span>
                  <span className="text-[#44bb66]">{demo.duration}</span>
                </div>
              </div>
            </div>

            <button
              onClick={() => onLaunchDemo(demo.id)}
              disabled={isLoading}
              className="w-full bg-[#0e1726] hover:bg-[#ff9900] text-[#ff9900] hover:text-[#04060a] border border-[#ff9900]/40 font-bold py-2 px-3 rounded transition-all flex items-center justify-center space-x-2 text-xs cursor-pointer"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>LAUNCH GUIDED MISSION</span>
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};
