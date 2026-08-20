import React from "react";

/**
 * THESEUS Scientific & Telemetry Formatter Utilities
 */

export function formatDistance(meters: number): string {
  if (meters === undefined || meters === null || isNaN(meters)) return "—";
  const km = meters / 1000.0;
  if (Math.abs(km) >= 1e6) {
    return `${(km / 1e6).toFixed(3)} M km`;
  } else if (Math.abs(km) >= 1000) {
    return `${km.toLocaleString(undefined, { maximumFractionDigits: 1 })} km`;
  } else if (Math.abs(meters) >= 1.0) {
    return `${meters.toFixed(2)} m`;
  } else {
    return `${(meters * 1000.0).toFixed(2)} mm`;
  }
}

export function formatSpeed(metersPerSec: number): string {
  if (metersPerSec === undefined || metersPerSec === null || isNaN(metersPerSec)) return "—";
  const kmPerSec = metersPerSec / 1000.0;
  if (Math.abs(kmPerSec) >= 1.0) {
    return `${kmPerSec.toFixed(3)} km/s`;
  } else {
    return `${metersPerSec.toFixed(2)} m/s`;
  }
}

export function formatMass(kg: number): string {
  if (kg === undefined || kg === null || isNaN(kg)) return "—";
  if (kg >= 1000.0) {
    return `${(kg / 1000.0).toFixed(2)} t (${kg.toLocaleString(undefined, { maximumFractionDigits: 0 })} kg)`;
  }
  return `${kg.toFixed(1)} kg`;
}

export function formatMET(seconds: number): string {
  if (seconds === undefined || seconds === null || isNaN(seconds)) return "T+00:00:00.0";
  const sign = seconds < 0 ? "-" : "+";
  const absSec = Math.abs(seconds);
  const hrs = Math.floor(absSec / 3600);
  const mins = Math.floor((absSec % 3600) / 60);
  const secs = Math.floor(absSec % 60);
  const frac = Math.floor((absSec % 1) * 10);

  const hStr = String(hrs).padStart(2, "0");
  const mStr = String(mins).padStart(2, "0");
  const sStr = String(secs).padStart(2, "0");

  return `T${sign}${hStr}:${mStr}:${sStr}.${frac}`;
}

export function formatScientificNumber(val: number, precision: number = 4): string {
  if (val === undefined || val === null || isNaN(val)) return "—";
  if (Math.abs(val) === 0) return "0.0";
  if (Math.abs(val) >= 0.01 && Math.abs(val) < 1e6) {
    return val.toLocaleString(undefined, { maximumFractionDigits: precision });
  }
  return val.toExponential(precision);
}

function formatKeyLabel(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/**
 * Robust Scientific Value Renderer
 * Safely renders strings, numbers, vectors, booleans, and structured objects
 * (e.g. { v_c1: ..., v_c2: ... }) into React children without throwing.
 */
export const ScientificValue: React.FC<{ value: any; depth?: number }> = ({ value, depth = 0 }) => {
  if (value === null || value === undefined) {
    return <span className="text-[#8c8275] italic">—</span>;
  }

  if (typeof value === "string") {
    return <span>{value}</span>;
  }

  if (typeof value === "number") {
    return <span>{formatScientificNumber(value)}</span>;
  }

  if (typeof value === "boolean") {
    return (
      <span className={value ? "text-[#44bb66] font-bold" : "text-[#cc3333] font-bold"}>
        {value ? "TRUE" : "FALSE"}
      </span>
    );
  }

  // Handle Array (e.g. 3D Vector or list of numbers/strings)
  if (Array.isArray(value)) {
    if (value.length === 3 && value.every((v) => typeof v === "number")) {
      return (
        <span className="font-mono text-[#e6dfd5]">
          [ {value.map((v, i) => `${["X", "Y", "Z"][i]}: ${formatScientificNumber(v)}`).join(", ")} ]
        </span>
      );
    }
    return (
      <span className="font-mono text-[#e6dfd5]">
        [ {value.map((item, idx) => (
          <React.Fragment key={idx}>
            {idx > 0 && ", "}
            <ScientificValue value={item} depth={depth + 1} />
          </React.Fragment>
        ))} ]
      </span>
    );
  }

  // Handle Structured Scientific Object (e.g. { v_c1: ..., v_c2: ... })
  if (typeof value === "object") {
    const entries = Object.entries(value);
    if (entries.length === 0) {
      return <span className="text-[#8c8275] italic">{"{}"}</span>;
    }

    if (depth > 2) {
      return (
        <span className="text-[#c8c0b5] font-mono text-[10px]">
          {entries.map(([k, v]) => `${k}: ${typeof v === "object" ? "..." : v}`).join(", ")}
        </span>
      );
    }

    return (
      <div className="inline-flex flex-wrap gap-1.5 items-center my-0.5">
        {entries.map(([k, v]) => (
          <span
            key={k}
            className="inline-flex items-center space-x-1.5 bg-[#05080f] border border-[#221d17] px-2 py-0.5 rounded text-[10px] shadow-sm"
          >
            <span className="text-[#ff9900] font-bold">{k}:</span>
            <span className="text-[#e6dfd5] font-semibold">
              <ScientificValue value={v} depth={depth + 1} />
            </span>
          </span>
        ))}
      </div>
    );
  }

  return <span>{String(value)}</span>;
};

/**
 * String conversion fallback
 */
export function formatScientificValueToString(value: any): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "string") return value;
  if (typeof value === "number") return formatScientificNumber(value);
  if (typeof value === "boolean") return value ? "TRUE" : "FALSE";
  if (Array.isArray(value)) {
    if (value.length === 3 && value.every((v) => typeof v === "number")) {
      return `[X: ${formatScientificNumber(value[0])}, Y: ${formatScientificNumber(value[1])}, Z: ${formatScientificNumber(value[2])}]`;
    }
    return `[${value.map(formatScientificValueToString).join(", ")}]`;
  }
  if (typeof value === "object") {
    return Object.entries(value)
      .map(([k, v]) => `${k} = ${formatScientificValueToString(v)}`)
      .join(", ");
  }
  return String(value);
}
