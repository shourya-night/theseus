import React, { useRef, useEffect } from "react";
import { MultiConjunctionEvent } from "../../types/mission";
import { 
  X, 
  Layers, 
  AlertTriangle, 
  Flame, 
  Target, 
  HelpCircle, 
  Compass, 
  Scale,
  Maximize2
} from "lucide-react";

interface BPlaneRiskOverlayProps {
  conjunction: MultiConjunctionEvent | null;
  onClose: () => void;
}

export const BPlaneRiskOverlay: React.FC<BPlaneRiskOverlayProps> = ({
  conjunction,
  onClose,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  useEffect(() => {
    if (!conjunction) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    // Background
    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, width, height);

    // Grid properties
    const centerX = width / 2;
    const centerY = height / 2;

    const bt = conjunction.b_plane_b_t_m || 0;
    const br = conjunction.b_plane_b_r_m || 0;
    const missDist = conjunction.miss_distance_m || Math.hypot(bt, br);
    const sigmaMaj = conjunction.b_plane_sigma_major_m || 150.0;
    const sigmaMin = conjunction.b_plane_sigma_minor_m || 80.0;
    const angleRad = ((conjunction.b_plane_ellipse_angle_deg || 0.0) * Math.PI) / 180.0;
    const hbr = conjunction.hard_body_radius_m || 10.0;

    // Determine scale: ensure 3-sigma ellipse, HBR, and miss vector fit on canvas
    const maxDimension = Math.max(missDist * 1.6, sigmaMaj * 3.5, hbr * 4.0, 50.0);
    const scale = Math.min(width, height) / (2.2 * maxDimension); // px per meter

    // 1. Draw Grid Lines
    ctx.strokeStyle = "rgba(60, 60, 60, 0.4)";
    ctx.lineWidth = 0.8;
    ctx.setLineDash([2, 4]);

    const gridStepM = Math.pow(10, Math.floor(Math.log10(maxDimension / 2)));
    const gridStepPx = gridStepM * scale;

    for (let x = centerX % gridStepPx; x < width; x += gridStepPx) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = centerY % gridStepPx; y < height; y += gridStepPx) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // 2. Draw Principal Kizner B-Plane Axes: T̂ (Horizontal) and R̂ (Vertical)
    ctx.strokeStyle = "rgba(180, 180, 180, 0.6)";
    ctx.lineWidth = 1.2;

    // T-Axis (Horizontal)
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(width, centerY);
    ctx.stroke();

    // R-Axis (Vertical)
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, height);
    ctx.stroke();

    // Axis Labels
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#888888";
    ctx.fillText("T̂ (Transverse)", width - 90, centerY - 6);
    ctx.fillText("R̂ (Radial)", centerX + 6, 16);

    // 3. Draw Origin Marker (Secondary Object Center)
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(centerX - 2, centerY - 2, 4, 4);
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#a0988e";
    ctx.fillText(`ORIGIN: ${conjunction.spacecraft_b_name}`, centerX + 8, centerY + 14);

    // 4. Draw 1-sigma, 2-sigma, 3-sigma Uncertainty Ellipses
    const sigmas = [
      { mult: 3, stroke: "rgba(255, 153, 0, 0.2)", dash: [4, 4], label: "3σ" },
      { mult: 2, stroke: "rgba(255, 153, 0, 0.4)", dash: [3, 3], label: "2σ" },
      { mult: 1, stroke: "rgba(255, 153, 0, 0.9)", dash: [], label: "1σ" },
    ];

    sigmas.forEach(({ mult, stroke, dash, label }) => {
      ctx.save();
      ctx.translate(centerX, centerY);
      ctx.rotate(-angleRad); // Canvas y is inverted

      ctx.strokeStyle = stroke;
      ctx.lineWidth = mult === 1 ? 1.5 : 1.0;
      ctx.setLineDash(dash);

      ctx.beginPath();
      ctx.ellipse(0, 0, sigmaMaj * mult * scale, sigmaMin * mult * scale, 0, 0, Math.PI * 2);
      ctx.stroke();

      if (mult === 1) {
        // Semi-major & Semi-minor axis indicators
        ctx.strokeStyle = "rgba(255, 153, 0, 0.4)";
        ctx.setLineDash([1, 3]);
        ctx.beginPath();
        ctx.moveTo(-sigmaMaj * scale, 0);
        ctx.lineTo(sigmaMaj * scale, 0);
        ctx.moveTo(0, -sigmaMin * scale);
        ctx.lineTo(0, sigmaMin * scale);
        ctx.stroke();
      }

      ctx.restore();
    });

    // 5. Draw Nominal Miss Vector b₀ = [B·T, B·R]
    const missScreenX = centerX + bt * scale;
    const missScreenY = centerY - br * scale;

    ctx.strokeStyle = "#44bbff";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(missScreenX, missScreenY);
    ctx.stroke();

    // 6. Draw Combined Hard-Body Collision Disk (Radius = HBR) around Miss Vector
    const hbrRadiusPx = hbr * scale;
    ctx.fillStyle = conjunction.is_physical_collision ? "rgba(255, 51, 51, 0.45)" : "rgba(68, 187, 255, 0.25)";
    ctx.strokeStyle = conjunction.is_physical_collision ? "#ff3333" : "#44bbff";
    ctx.lineWidth = 1.5;

    ctx.beginPath();
    ctx.arc(missScreenX, missScreenY, Math.max(3, hbrRadiusPx), 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Center point of primary object
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(missScreenX - 2, missScreenY - 2, 4, 4);

    // Callout Label for Primary Object
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#44bbff";
    ctx.fillText(`${conjunction.spacecraft_a_name} (HBR=${hbr.toFixed(1)}m)`, missScreenX + 8, missScreenY - 6);

    // Scale Bar in bottom-left
    const scaleBarM = gridStepM;
    const scaleBarPx = scaleBarM * scale;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(20, height - 25, scaleBarPx, 3);
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#a0988e";
    ctx.fillText(`SCALE: ${scaleBarM >= 1000 ? `${scaleBarM / 1000} km` : `${scaleBarM} m`}`, 20, height - 32);

  }, [conjunction]);

  if (!conjunction) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-sm p-4 font-mono select-none">
      <div className="bg-black border border-neutral-800 w-full max-w-5xl h-[85vh] flex flex-col shadow-2xl overflow-hidden">
        
        {/* Header */}
        <div className="p-3 bg-neutral-950 border-b border-neutral-800 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <Layers className="w-4 h-4 text-amber-400" />
            <h2 className="text-sm font-semibold tracking-wider text-neutral-100 uppercase">
              Phase 10 — 2D B-Plane Encounter Geometry & Collision Risk
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors"
            title="Close (Esc)"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content Body: Canvas on Left, Telemetry & Math on Right */}
        <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
          
          {/* Left: 2D B-Plane Canvas */}
          <div className="flex-1 relative bg-[#0a0a0a] border-r border-neutral-800 h-full min-h-[350px]">
            <canvas ref={canvasRef} className="w-full h-full block" />
            <div className="absolute top-3 left-3 bg-black/75 border border-neutral-800 p-2 text-[10px] text-neutral-400">
              <span className="text-amber-400 font-bold block mb-0.5">KIZNER B-PLANE PROJECTION</span>
              <span>Encounter plane orthogonal to relative velocity Ŝ</span>
            </div>
          </div>

          {/* Right: Scientific Quantities & Probability Analysis */}
          <div className="w-full md:w-80 lg:w-96 bg-neutral-950 p-4 flex flex-col gap-3.5 overflow-y-auto text-xs text-neutral-300">
            
            {/* Spacecraft Pair */}
            <div className="p-2.5 bg-black border border-neutral-800">
              <div className="text-[10px] text-neutral-500 uppercase tracking-wider mb-1">CONJUNCTION PAIR</div>
              <div className="flex items-center justify-between font-bold text-sm text-neutral-100">
                <span className="text-amber-400">{conjunction.spacecraft_a_name}</span>
                <span className="text-neutral-600">↔</span>
                <span className="text-amber-400">{conjunction.spacecraft_b_name}</span>
              </div>
            </div>

            {/* Collision Risk Banner */}
            <div className={`p-3 border ${
              conjunction.is_physical_collision 
                ? "bg-red-950/40 border-red-500 text-red-300"
                : (conjunction.risk_level === "CRITICAL"
                    ? "bg-red-950/30 border-red-600 text-red-400"
                    : (conjunction.risk_level === "HIGH"
                        ? "bg-amber-950/30 border-amber-500 text-amber-400"
                        : "bg-black border-neutral-800 text-neutral-300"))
            }`}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] uppercase font-bold tracking-wider">COLLISION PROBABILITY (Pc)</span>
                <span className="text-xs font-bold uppercase">{conjunction.risk_level} RISK</span>
              </div>
              <div className="text-2xl font-black text-white font-mono my-1">
                {conjunction.collision_probability_scientific || (conjunction.collision_probability !== undefined ? conjunction.collision_probability.toExponential(4) : "0.0000e+00")}
              </div>
              <div className="text-[11px] text-neutral-400">
                {conjunction.is_physical_collision
                  ? "PHYSICAL COLLISION CONFIRMED: Miss distance is within combined hard-body collision radius."
                  : (conjunction.action_required
                      ? "High collision risk exceeds maneuver threshold. Collision avoidance maneuver recommended."
                      : "Risk is within standard operational screening tolerance.")}
              </div>
            </div>

            {/* Geometric Quantities */}
            <div className="space-y-1.5 bg-black p-3 border border-neutral-800 text-[11px]">
              <div className="text-[10px] text-neutral-500 uppercase tracking-wider mb-2">ENCOUNTER TELEMETRY</div>
              
              <div className="flex justify-between py-0.5 border-b border-neutral-900">
                <span className="text-neutral-400">Time of Closest Approach (TCA)</span>
                <span className="font-semibold text-white">T+ {conjunction.tca_s.toFixed(2)} s</span>
              </div>

              <div className="flex justify-between py-0.5 border-b border-neutral-900">
                <span className="text-neutral-400">Miss Distance |b₀|</span>
                <span className={`font-semibold ${conjunction.is_physical_collision ? "text-red-400 font-bold" : "text-white"}`}>
                  {conjunction.miss_distance_m.toFixed(2)} m ({conjunction.miss_distance_km.toFixed(4)} km)
                </span>
              </div>

              <div className="flex justify-between py-0.5 border-b border-neutral-900">
                <span className="text-neutral-400">Relative Velocity |v_rel|</span>
                <span className="font-semibold text-white">{conjunction.relative_velocity_km_s.toFixed(3)} km/s</span>
              </div>

              <div className="flex justify-between py-0.5 border-b border-neutral-900">
                <span className="text-neutral-400">Combined Hard-Body Radius (HBR)</span>
                <span className="font-semibold text-amber-300">{conjunction.hard_body_radius_m.toFixed(1)} m</span>
              </div>

              <div className="flex justify-between py-0.5 border-b border-neutral-900">
                <span className="text-neutral-400">B-Plane Component B·T</span>
                <span className="font-semibold text-white">{(conjunction.b_plane_b_t_m || 0).toFixed(1)} m</span>
              </div>

              <div className="flex justify-between py-0.5">
                <span className="text-neutral-400">B-Plane Component B·R</span>
                <span className="font-semibold text-white">{(conjunction.b_plane_b_r_m || 0).toFixed(1)} m</span>
              </div>
            </div>

            {/* Uncertainty Ellipse Quantities */}
            <div className="space-y-1.5 bg-black p-3 border border-neutral-800 text-[11px]">
              <div className="text-[10px] text-neutral-500 uppercase tracking-wider mb-2">B-PLANE UNCERTAINTY ELLIPSE (1σ)</div>
              
              <div className="flex justify-between py-0.5 border-b border-neutral-900">
                <span className="text-neutral-400">Semi-Major Axis (σ_major)</span>
                <span className="font-semibold text-amber-400">{(conjunction.b_plane_sigma_major_m || 0).toFixed(1)} m</span>
              </div>

              <div className="flex justify-between py-0.5 border-b border-neutral-900">
                <span className="text-neutral-400">Semi-Minor Axis (σ_minor)</span>
                <span className="font-semibold text-amber-400">{(conjunction.b_plane_sigma_minor_m || 0).toFixed(1)} m</span>
              </div>

              <div className="flex justify-between py-0.5">
                <span className="text-neutral-400">Ellipse Orientation Angle (θ)</span>
                <span className="font-semibold text-white">{(conjunction.b_plane_ellipse_angle_deg || 0).toFixed(1)}°</span>
              </div>
            </div>

            {/* Mathematical Model Footnote */}
            <div className="text-[10px] text-neutral-500 italic p-2 border border-neutral-900 bg-black">
              Integrated via 2D Gaussian Encounter Plane Polar Quadrature over circular disk |z - b₀| ≤ HBR.
            </div>

          </div>
        </div>

        {/* Footer */}
        <div className="p-2.5 bg-neutral-950 border-t border-neutral-800 flex items-center justify-between text-xs text-neutral-400">
          <span>THESEUS Astrodynamics Engine — Phase 10 Covariance Analysis</span>
          <button
            onClick={onClose}
            className="px-3 py-1 bg-neutral-800 hover:bg-neutral-700 text-white transition-colors"
          >
            Close
          </button>
        </div>

      </div>
    </div>
  );
};
