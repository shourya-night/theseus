import { ActiveRocket } from "../types/mission";

export interface InterpolatedRocketState {
  time_seconds: number;
  position: [number, number, number];
  velocity: [number, number, number];
  speed: number;
  altitude: number;
  /** Total vehicle mass in kg, carried through from the ORBIT-X state. */
  mass: number;
  fuel_mass: number;
  thrust_active: boolean;
}

/**
 * Universal Simulation Clock Position & Velocity Interpolation Helper
 * Evaluates 3D physical world state for any rocket at exact simulation timestamp simTimeSec.
 * Eliminates timing lags caused by frame index mismatch.
 */
export function getRocketStateAtTime(
  rocket: ActiveRocket,
  simTimeSec: number
): InterpolatedRocketState | null {
  const history = rocket.result?.state_history;
  if (!history || history.length === 0) return null;

  if (simTimeSec <= history[0].time_seconds) {
    const st = history[0];
    return {
      time_seconds: st.time_seconds,
      position: [st.position[0], st.position[1], st.position[2]],
      velocity: [st.velocity[0], st.velocity[1], st.velocity[2]],
      speed: st.speed || 0,
      altitude: st.altitude || 0,
      mass: st.mass || 0,
      fuel_mass: st.fuel_mass || 0,
      thrust_active: st.thrust_active || false,
    };
  }

  const lastSt = history[history.length - 1];
  if (simTimeSec >= lastSt.time_seconds) {
    return {
      time_seconds: lastSt.time_seconds,
      position: [lastSt.position[0], lastSt.position[1], lastSt.position[2]],
      velocity: [lastSt.velocity[0], lastSt.velocity[1], lastSt.velocity[2]],
      speed: lastSt.speed || 0,
      altitude: lastSt.altitude || 0,
      mass: lastSt.mass || 0,
      fuel_mass: lastSt.fuel_mass || 0,
      thrust_active: false,
    };
  }

  let low = 0;
  let high = history.length - 1;
  while (low <= high) {
    const mid = (low + high) >> 1;
    if (history[mid].time_seconds < simTimeSec) {
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }

  const idx1 = Math.max(0, high);
  const idx2 = Math.min(history.length - 1, low);

  if (idx1 === idx2) {
    const st = history[idx1];
    return {
      time_seconds: st.time_seconds,
      position: [st.position[0], st.position[1], st.position[2]],
      velocity: [st.velocity[0], st.velocity[1], st.velocity[2]],
      speed: st.speed || 0,
      altitude: st.altitude || 0,
      mass: st.mass || 0,
      fuel_mass: st.fuel_mass || 0,
      thrust_active: st.thrust_active || false,
    };
  }

  const st1 = history[idx1];
  const st2 = history[idx2];
  const dt = st2.time_seconds - st1.time_seconds;

  if (dt <= 1e-6) {
    return {
      time_seconds: st1.time_seconds,
      position: [st1.position[0], st1.position[1], st1.position[2]],
      velocity: [st1.velocity[0], st1.velocity[1], st1.velocity[2]],
      speed: st1.speed || 0,
      altitude: st1.altitude || 0,
      mass: st1.mass || 0,
      fuel_mass: st1.fuel_mass || 0,
      thrust_active: st1.thrust_active || false,
    };
  }

  const frac = (simTimeSec - st1.time_seconds) / dt;

  const posX = st1.position[0] + frac * (st2.position[0] - st1.position[0]);
  const posY = st1.position[1] + frac * (st2.position[1] - st1.position[1]);
  const posZ = st1.position[2] + frac * (st2.position[2] - st1.position[2]);

  const velX = st1.velocity[0] + frac * (st2.velocity[0] - st1.velocity[0]);
  const velY = st1.velocity[1] + frac * (st2.velocity[1] - st1.velocity[1]);
  const velZ = st1.velocity[2] + frac * (st2.velocity[2] - st1.velocity[2]);

  return {
    time_seconds: simTimeSec,
    position: [posX, posY, posZ],
    velocity: [velX, velY, velZ],
    speed: Math.hypot(velX, velY, velZ),
    altitude: st1.altitude + frac * (st2.altitude - st1.altitude),
    mass: st1.mass + frac * (st2.mass - st1.mass),
    fuel_mass: st1.fuel_mass + frac * (st2.fuel_mass - st1.fuel_mass),
    thrust_active: frac < 0.5 ? st1.thrust_active : st2.thrust_active,
  };
}

export type RocketLifecycleState = "FLYING" | "ARRIVED" | "COLLIDED" | "DESTROYED_BY_SUN";

/**
 * Returns the exact lifecycle state of a rocket at physical simulation time simTimeSec.
 */
export function getRocketLifecycleState(
  rocket: ActiveRocket,
  simTimeSec: number
): RocketLifecycleState {
  if (rocket.collisionState === "DESTROYED_BY_SUN") return "DESTROYED_BY_SUN";
  if (rocket.collisionState === "COLLIDED") return "COLLIDED";

  const history = rocket.result?.state_history;
  if (!history || history.length === 0) return "ARRIVED";

  const finalT = history[history.length - 1].time_seconds;
  if (simTimeSec >= finalT) return "ARRIVED";

  return "FLYING";
}

/**
 * Single Authoritative Fleet Lifecycle Rule:
 * Returns true if AT LEAST ONE rocket in activeRockets is currently FLYING at simTimeSec.
 */
export function hasFlyingRockets(activeRockets: ActiveRocket[], simTimeSec: number): boolean {
  if (!activeRockets || activeRockets.length === 0) return false;
  return activeRockets.some((r) => getRocketLifecycleState(r, simTimeSec) === "FLYING");
}
