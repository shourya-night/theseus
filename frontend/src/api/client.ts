/**
 * THESEUS API Client Bridge
 * Direct authoritative bridge to the THESEUS FastAPI Physics Backend.
 * Strict scientific data flow: User Configuration -> Physics Engine -> Visualization.
 * No fake trajectories, no synthetic math fallbacks.
 */

import { 
  SimulationResult, 
  RocketPreset, 
  SpacecraftConfig, 
  MultiSimulationResult,
} from "../types/mission";
import { ROCKET_PRESETS } from "../data/rocketPresets";
import { CELESTIAL_BODIES } from "../data/celestialCatalog";

const API_BASE_URL = typeof window !== "undefined" && window.location.hostname === "localhost"
  ? "http://localhost:8000/api"
  : "http://127.0.0.1:8000/api";

export interface BackendHealth {
  status: string;
  engine: string;
  version: string;
  timestamp: string;
  subsystems: Record<string, string>;
}

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { signal: AbortSignal.timeout(2500) });
    return res.ok;
  } catch {
    try {
      const altUrl = API_BASE_URL.includes("localhost") ? "http://127.0.0.1:8000/api" : "http://localhost:8000/api";
      const res2 = await fetch(`${altUrl}/health`, { signal: AbortSignal.timeout(2500) });
      return res2.ok;
    } catch {
      return false;
    }
  }
}

export async function fetchHealthDetails(): Promise<BackendHealth | null> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      return await res.json();
    }
  } catch {
    try {
      const altUrl = API_BASE_URL.includes("localhost") ? "http://127.0.0.1:8000/api" : "http://localhost:8000/api";
      const res2 = await fetch(`${altUrl}/health`, { signal: AbortSignal.timeout(3000) });
      if (res2.ok) return await res2.json();
    } catch {
      return null;
    }
  }
  return null;
}

export async function fetchBodies(): Promise<Record<string, any>> {
  try {
    const res = await fetch(`${API_BASE_URL}/bodies`);
    if (res.ok) {
      const data = await res.json();
      return data.bodies;
    }
  } catch {
    // Return static catalog when backend is starting
  }
  return CELESTIAL_BODIES;
}

export async function fetchPresets(): Promise<RocketPreset[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/presets`);
    if (res.ok) {
      const data = await res.json();
      return data.presets;
    }
  } catch {
    // Return static catalog when backend is starting
  }
  return ROCKET_PRESETS;
}

export async function simulateHohmann(payload: {
  r1_km: number;
  r2_km: number;
  origin_body: string;
  plane_change_deg: number;
  dry_mass_kg: number;
  fuel_mass_kg: number;
  specific_impulse_s: number;
  thrust_n: number;
}): Promise<SimulationResult> {
  const res = await fetch(`${API_BASE_URL}/simulate/hohmann`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errText = await res.text().catch(() => "");
    throw new Error(`Hohmann Simulation Failed (HTTP ${res.status}): ${errText || res.statusText}`);
  }
  return await res.json();
}

export async function simulateLambert(payload: {
  r1_km: [number, number, number];
  r2_km: [number, number, number];
  tof_hours: number;
  central_body: string;
  prograde: boolean;
  dry_mass_kg: number;
  fuel_mass_kg: number;
  specific_impulse_s: number;
  thrust_n: number;
}): Promise<SimulationResult> {
  const res = await fetch(`${API_BASE_URL}/simulate/lambert`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errText = await res.text().catch(() => "");
    throw new Error(`Lambert Simulation Failed (HTTP ${res.status}): ${errText || res.statusText}`);
  }
  return await res.json();
}

export async function simulateRendezvous(payload: {
  chaser_alt_km: number;
  target_alt_km: number;
  target_lead_deg: number;
  tof_hours: number;
  central_body: string;
  dry_mass_kg: number;
  fuel_mass_kg: number;
  specific_impulse_s: number;
  thrust_n: number;
}): Promise<SimulationResult> {
  const res = await fetch(`${API_BASE_URL}/simulate/rendezvous`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errText = await res.text().catch(() => "");
    throw new Error(`Rendezvous Simulation Failed (HTTP ${res.status}): ${errText || res.statusText}`);
  }
  return await res.json();
}

export async function simulateMultiEnvironment(payload: {
  spacecraft: SpacecraftConfig[];
  central_body?: string;
  duration_hours?: number;
  dt_sec?: number;
  screening_threshold_km?: number;
  enable_j2?: boolean;
  enable_drag?: boolean;
  enable_srp?: boolean;
}): Promise<MultiSimulationResult> {
  const res = await fetch(`${API_BASE_URL}/simulate/environment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errText = await res.text().catch(() => "");
    throw new Error(`Multi-Object Simulation Failed (HTTP ${res.status}): ${errText || res.statusText}`);
  }
  return await res.json();
}

export async function fetchDemoMission(demoId: string): Promise<SimulationResult> {
  const res = await fetch(`${API_BASE_URL}/demo/${demoId}`);
  if (!res.ok) {
    const errText = await res.text().catch(() => "");
    throw new Error(`Failed to load demo mission '${demoId}' (HTTP ${res.status}): ${errText || res.statusText}`);
  }
  return await res.json();
}
