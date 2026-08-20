/**
 * THESEUS Pixel-Art Astronomical & Vehicle Engine
 * Hard-edged deterministic pixel sprites for celestial bodies, solar system,
 * distinct spacecraft silhouettes, burn plumes, and seeded starfields.
 */

// Seeded PRNG for deterministic star distribution
function pseudoRandom(seed: number) {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

export interface Star {
  x: number; // World space offset
  y: number;
  size: number; // 1 or 2 pixels
  brightness: string; // Off-white variations
  parallax: number; // 0.05 to 0.15
}

// Generate static sparse starfield (240 stars)
export function generateStarfield(count: number = 240): Star[] {
  const stars: Star[] = [];
  const spread = 20000; // Large coordinate space
  const shades = ["#ffffff", "#e6dfd5", "#c8c0b5", "#a0988e"];

  for (let i = 0; i < count; i++) {
    const x = (pseudoRandom(i * 1.3 + 1) - 0.5) * spread;
    const y = (pseudoRandom(i * 2.7 + 2) - 0.5) * spread;
    const size = pseudoRandom(i * 4.1 + 3) > 0.88 ? 2 : 1;
    const colorIdx = Math.floor(pseudoRandom(i * 5.9 + 4) * shades.length);
    const parallax = 0.03 + pseudoRandom(i * 7.3 + 5) * 0.08;

    stars.push({
      x,
      y,
      size,
      brightness: shades[colorIdx],
      parallax,
    });
  }
  return stars;
}

export const STARFIELD_CACHE: Star[] = generateStarfield(240);

export function renderStarField(
  ctx: CanvasRenderingContext2D,
  cameraX: number,
  cameraY: number,
  viewportWidth: number,
  viewportHeight: number
) {
  ctx.save();
  ctx.imageSmoothingEnabled = false;

  for (let i = 0; i < STARFIELD_CACHE.length; i++) {
    const star = STARFIELD_CACHE[i];
    // Screen position with subtle parallax
    const sx = Math.floor(((star.x - cameraX * star.parallax) % viewportWidth + viewportWidth) % viewportWidth);
    const sy = Math.floor(((star.y - cameraY * star.parallax) % viewportHeight + viewportHeight) % viewportHeight);

    ctx.fillStyle = star.brightness;
    ctx.fillRect(sx, sy, star.size, star.size);
  }

  ctx.restore();
}

/**
 * Draw hard-edged pixel-art celestial bodies
 */
export function drawPixelPlanet(
  ctx: CanvasRenderingContext2D,
  name: string,
  radiusPx: number,
  label: string,
  showLabel: boolean = true
) {
  ctx.save();
  ctx.imageSmoothingEnabled = false;

  const lname = name.toLowerCase();
  const r = Math.max(2, Math.round(radiusPx));

  if (r <= 3) {
    // Micro pixel point for distant planets in Solar System view
    let pColor = "#c8c0b5";
    if (lname.includes("sun")) pColor = "#ffcc00";
    else if (lname.includes("mercury")) pColor = "#a49b8f";
    else if (lname.includes("venus")) pColor = "#e3bb76";
    else if (lname.includes("earth")) pColor = "#3388ff";
    else if (lname.includes("mars")) pColor = "#e26638";
    else if (lname.includes("jupiter")) pColor = "#d4a373";
    else if (lname.includes("saturn")) pColor = "#e9c46a";
    else if (lname.includes("uranus")) pColor = "#70d6ff";
    else if (lname.includes("neptune")) pColor = "#4361ee";

    ctx.fillStyle = pColor;
    ctx.fillRect(-r, -r, r * 2, r * 2);

    if (showLabel) {
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.fillStyle = "#c8c0b5";
      ctx.textAlign = "center";
      ctx.fillText(label.toUpperCase(), 0, r + 11);
    }
    ctx.restore();
    return;
  }

  if (lname.includes("sun")) {
    // SUN: Pixel golden core with flare corona
    ctx.fillStyle = "#ff6600";
    ctx.fillRect(-r * 1.3, -r * 0.3, r * 2.6, r * 0.6);
    ctx.fillRect(-r * 0.3, -r * 1.3, r * 0.6, r * 2.6);

    ctx.fillStyle = "#ff9900";
    ctx.fillRect(-r * 1.1, -r * 1.1, r * 2.2, r * 2.2);

    ctx.fillStyle = "#ffcc00";
    ctx.fillRect(-r, -r, r * 2, r * 2);

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(-r * 0.4, -r * 0.4, r * 0.8, r * 0.8);
  } else if (lname.includes("mercury")) {
    // MERCURY: Slate-grey cratered pixel disc
    ctx.fillStyle = "#8a857e";
    ctx.fillRect(-r, -r, r * 2, r * 2);
    ctx.fillStyle = "#5c5750";
    ctx.fillRect(-r * 0.5, -r * 0.5, r * 0.5, r * 0.5);
    ctx.fillRect(r * 0.1, r * 0.1, r * 0.4, r * 0.4);
  } else if (lname.includes("venus")) {
    // VENUS: Ochre-cream thick atmosphere
    ctx.fillStyle = "#e3bb76";
    ctx.fillRect(-r, -r, r * 2, r * 2);
    ctx.fillStyle = "#cfa156";
    ctx.fillRect(-r, -r * 0.3, r * 2, r * 0.6);
    ctx.fillStyle = "#fff0c2";
    ctx.fillRect(-r * 0.4, -r * 0.7, r * 0.8, r * 0.3);
  } else if (lname.includes("earth")) {
    // EARTH: Pixel globe with oceans, continents, and ice caps
    ctx.fillStyle = "#163866";
    ctx.fillRect(-r - 1, -r - 1, (r + 1) * 2, (r + 1) * 2);

    ctx.fillStyle = "#1b4d89";
    ctx.fillRect(-r, -r, r * 2, r * 2);

    ctx.fillStyle = "#338844";
    ctx.fillRect(-r * 0.7, -r * 0.5, r * 0.7, r * 0.8);
    ctx.fillRect(0, r * 0.1, r * 0.6, r * 0.6);
    ctx.fillRect(-r * 0.2, -r * 0.1, r * 0.4, r * 0.4);

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(-r * 0.6, -r, r * 1.2, Math.max(1, r * 0.25));
    ctx.fillRect(-r * 0.6, r - Math.max(1, r * 0.25), r * 1.2, Math.max(1, r * 0.25));
  } else if (lname.includes("moon")) {
    // MOON: Grey cratered disc
    ctx.fillStyle = "#b0b5bc";
    ctx.fillRect(-r, -r, r * 2, r * 2);
    ctx.fillStyle = "#555b63";
    ctx.fillRect(-r * 0.5, -r * 0.3, r * 0.4, r * 0.4);
    ctx.fillRect(r * 0.1, r * 0.2, r * 0.4, r * 0.4);
  } else if (lname.includes("mars")) {
    // MARS: Rust-red/ochre disc with polar ice cap and dark canyon
    ctx.fillStyle = "#c1440e";
    ctx.fillRect(-r, -r, r * 2, r * 2);
    ctx.fillStyle = "#7a2200";
    ctx.fillRect(-r * 0.6, 0, r * 1.2, r * 0.35);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(-r * 0.4, -r, r * 0.8, Math.max(1, r * 0.25));
  } else if (lname.includes("jupiter")) {
    // JUPITER: Banded gas giant with Great Red Spot
    ctx.fillStyle = "#d4a373";
    ctx.fillRect(-r, -r, r * 2, r * 2);
    ctx.fillStyle = "#8a4f2d";
    ctx.fillRect(-r, -r * 0.5, r * 2, r * 0.25);
    ctx.fillRect(-r, r * 0.1, r * 2, r * 0.25);
    ctx.fillStyle = "#b83311";
    ctx.fillRect(r * 0.2, r * 0.2, r * 0.4, r * 0.3);
  } else if (lname.includes("saturn")) {
    // SATURN: Gas giant with vector ring band
    ctx.fillStyle = "#e0c9a6";
    ctx.fillRect(-r, -r, r * 2, r * 2);
    ctx.strokeStyle = "#c4ab84";
    ctx.lineWidth = Math.max(1, r * 0.25);
    ctx.beginPath();
    ctx.ellipse(0, 0, r * 2.1, r * 0.55, Math.PI / 8, 0, Math.PI * 2);
    ctx.stroke();
  } else if (lname.includes("uranus")) {
    // URANUS: Cyan-aquamarine gas disc
    ctx.fillStyle = "#70d6ff";
    ctx.fillRect(-r, -r, r * 2, r * 2);
    ctx.fillStyle = "#52b6de";
    ctx.fillRect(-r, -r * 0.2, r * 2, r * 0.4);
  } else if (lname.includes("neptune")) {
    // NEPTUNE: Deep azure-blue gas disc
    ctx.fillStyle = "#3a56d4";
    ctx.fillRect(-r, -r, r * 2, r * 2);
    ctx.fillStyle = "#1e37a3";
    ctx.fillRect(-r, -r * 0.2, r * 2, r * 0.4);
  } else {
    ctx.fillStyle = "#8c8275";
    ctx.fillRect(-r, -r, r * 2, r * 2);
  }

  // Label
  if (showLabel) {
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#e6dfd5";
    ctx.textAlign = "center";
    ctx.fillText(label.toUpperCase(), 0, r + 12);
  }

  ctx.restore();
}

/**
 * Draw distinct spacecraft silhouettes
 */
export function drawPixelSpacecraft(
  ctx: CanvasRenderingContext2D,
  spriteId: string,
  vx: number,
  vy: number,
  isThrustActive: boolean
) {
  ctx.save();
  ctx.imageSmoothingEnabled = false;

  // Heading rotation along velocity vector
  const heading = Math.atan2(vy, vx);
  ctx.rotate(heading);

  const sId = (spriteId || "chandrayaan").toLowerCase();

  if (sId.includes("lvm3")) {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(-10, -9, 18, 4);
    ctx.fillRect(-10, 5, 18, 4);
    ctx.fillStyle = "#333333";
    ctx.fillRect(-12, -8, 2, 2);
    ctx.fillRect(-12, 6, 2, 2);
    ctx.fillStyle = "#d8d0c5";
    ctx.fillRect(-12, -4, 24, 8);
    ctx.fillStyle = "#ff9900";
    ctx.fillRect(12, -4, 6, 8);
    ctx.fillRect(18, -2, 3, 4);

  } else if (sId.includes("pslv")) {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(-14, -3, 26, 6);
    ctx.fillStyle = "#cc3333";
    ctx.fillRect(-6, -3, 3, 6);
    ctx.fillRect(4, -3, 3, 6);
    ctx.fillStyle = "#8c8275";
    ctx.fillRect(-12, -5, 10, 2);
    ctx.fillRect(-12, 3, 10, 2);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(12, -2, 4, 4);

  } else if (sId.includes("falconheavy")) {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(-12, -8, 22, 4);
    ctx.fillRect(-14, -3, 26, 6);
    ctx.fillRect(-12, 4, 22, 4);
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(4, -3, 3, 6);
    ctx.fillRect(-14, -8, 2, 16);

  } else if (sId.includes("falcon9") || sId.includes("falcon")) {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(-14, -3.5, 26, 7);
    ctx.fillStyle = "#111111";
    ctx.fillRect(3, -3.5, 3, 7);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(12, -2.5, 4, 5);

  } else if (sId.includes("starship")) {
    ctx.fillStyle = "#c8d0d8";
    ctx.fillRect(-12, -5, 24, 10);
    ctx.fillStyle = "#333333";
    ctx.fillRect(8, -8, 3, 3);
    ctx.fillRect(8, 5, 3, 3);
    ctx.fillRect(-10, -9, 5, 4);
    ctx.fillRect(-10, 5, 5, 4);
    ctx.fillStyle = "#e0e8f0";
    ctx.fillRect(12, -3, 5, 6);

  } else if (sId.includes("saturn5") || sId.includes("saturn")) {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(-16, -6, 30, 12);
    ctx.fillStyle = "#111111";
    ctx.fillRect(-8, -6, 5, 12);
    ctx.fillRect(4, -6, 4, 12);
    ctx.fillStyle = "#ff3333";
    ctx.fillRect(14, -1, 8, 2);

  } else if (sId.includes("chandrayaan") || sId.includes("pm")) {
    ctx.fillStyle = "#1b4d89";
    ctx.fillRect(-2, -14, 4, 8);
    ctx.fillRect(-2, 6, 4, 8);
    ctx.strokeStyle = "#44bbff";
    ctx.lineWidth = 1;
    ctx.strokeRect(-2, -14, 4, 8);
    ctx.strokeRect(-2, 6, 4, 8);
    ctx.fillStyle = "#cc9900";
    ctx.fillRect(-5, -5, 10, 10);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(-2, -3, 4, 6);
    ctx.fillStyle = "#333333";
    ctx.fillRect(-8, -2, 3, 4);

  } else if (sId.includes("apollo") || sId.includes("csm")) {
    ctx.fillStyle = "#d8e0e8";
    ctx.fillRect(-7, -5, 12, 10);
    ctx.fillStyle = "#a4b8cc";
    ctx.beginPath();
    ctx.moveTo(5, -5);
    ctx.lineTo(11, 0);
    ctx.lineTo(5, 5);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = "#222222";
    ctx.fillRect(-11, -3, 4, 6);

  } else if (sId.includes("dragon")) {
    ctx.fillStyle = "#1b4d89";
    ctx.fillRect(-6, -5, 6, 10);
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.moveTo(0, -5);
    ctx.lineTo(8, 0);
    ctx.lineTo(0, 5);
    ctx.closePath();
    ctx.fill();

  } else if (sId.includes("voyager")) {
    ctx.fillStyle = "#ffcc00";
    ctx.beginPath();
    ctx.ellipse(3, 0, 3, 10, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#8c8275";
    ctx.fillRect(-6, -4, 8, 8);
    ctx.strokeStyle = "#8c8275";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(-6, 0);
    ctx.lineTo(-15, 0);
    ctx.stroke();

  } else if (sId.includes("ion")) {
    ctx.fillStyle = "#8c8275";
    ctx.fillRect(-5, -4, 10, 8);
    ctx.fillStyle = "#1b4d89";
    ctx.fillRect(-2, -12, 4, 6);
    ctx.fillRect(-2, 6, 4, 6);
    ctx.fillStyle = "#00f0ff";
    ctx.fillRect(-8, -2, 3, 4);

  } else {
    ctx.fillStyle = "#d8d0c0";
    ctx.fillRect(-5, -4, 10, 8);
    ctx.fillStyle = "#1b4d89";
    ctx.fillRect(-2, -10, 4, 5);
    ctx.fillRect(-2, 5, 4, 5);
  }

  // Active Burn Exhaust Plume
  if (isThrustActive) {
    drawPixelBurnPlume(ctx);
  }

  ctx.restore();
}

function drawPixelBurnPlume(ctx: CanvasRenderingContext2D) {
  const flicker = (Math.random() - 0.5) * 4;
  const length = 12 + Math.random() * 8;

  ctx.fillStyle = "#cc2200";
  ctx.fillRect(-8 - length - 4, Math.floor(flicker - 2), Math.floor(length + 4), 4);

  ctx.fillStyle = "#ff6600";
  ctx.fillRect(-8 - length, Math.floor(flicker - 1), Math.floor(length), 3);

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(-8 - Math.floor(length * 0.5), 0, Math.floor(length * 0.5), 2);

  if (Math.random() > 0.4) {
    ctx.fillStyle = "#ffaa00";
    ctx.fillRect(-8 - length - 6, Math.floor(flicker), 2, 2);
  }
}

/**
 * Draw four distinct pixel-art debris fragments
 */
export function drawPixelDebris(
  ctx: CanvasRenderingContext2D,
  debrisType: string = "solar_panel",
  rotation: number = 0
) {
  ctx.save();
  ctx.imageSmoothingEnabled = false;
  ctx.rotate(rotation);

  const dtype = (debrisType || "solar_panel").toLowerCase();

  if (dtype.includes("solar") || dtype.includes("panel")) {
    // DEBRIS-A: Broken Solar Panel Fragment
    // Jagged frame with fractured blue photovoltaic cells
    ctx.fillStyle = "#1b4d89"; // Solar cell blue
    ctx.fillRect(-6, -4, 12, 7);
    ctx.fillStyle = "#44bbff"; // High-reflectivity cell line
    ctx.fillRect(-5, -3, 4, 5);
    ctx.fillRect(1, -3, 4, 3);
    ctx.fillStyle = "#ffcc00"; // Gold busbar
    ctx.fillRect(-6, -1, 12, 1);
    ctx.fillStyle = "#555555"; // Broken bracket
    ctx.fillRect(-7, -5, 2, 9);
    ctx.fillStyle = "#ff3300"; // Exposed wire tip
    ctx.fillRect(6, 1, 3, 1);

  } else if (dtype.includes("truss") || dtype.includes("struct")) {
    // DEBRIS-B: Structural Truss / Lattice Fragment
    // Triangular metallic struts with jagged break points
    ctx.fillStyle = "#d8d0c5"; // Titanium strut
    ctx.fillRect(-6, -6, 12, 2);
    ctx.fillRect(-6, 4, 10, 2);
    ctx.fillRect(-4, -4, 2, 8);
    ctx.fillRect(2, -4, 2, 8);
    // Diagonal brace
    ctx.fillStyle = "#9c9489";
    ctx.fillRect(-2, -2, 4, 4);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(-7, -7, 2, 2); // Fractured lug

  } else if (dtype.includes("nozzle") || dtype.includes("engine")) {
    // DEBRIS-C: Engine Nozzle / Bell Fragment
    // Dark charred bell with copper cooling manifold tubes
    ctx.fillStyle = "#333333"; // Charred bell
    ctx.beginPath();
    ctx.moveTo(-5, -6);
    ctx.lineTo(6, -3);
    ctx.lineTo(6, 3);
    ctx.lineTo(-5, 6);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = "#ff9900"; // Hot throat rim
    ctx.fillRect(-6, -3, 2, 6);
    ctx.fillStyle = "#cc6600"; // Copper manifold lines
    ctx.fillRect(-2, -4, 2, 8);
    ctx.fillRect(2, -3, 2, 6);

  } else {
    // DEBRIS-D: Spacecraft Body / Avionics Core Fragment
    // Gold Kapton MLI foil wrapped block with protruding antenna/electronics
    ctx.fillStyle = "#c8960c"; // Gold Kapton foil
    ctx.fillRect(-5, -5, 10, 10);
    ctx.fillStyle = "#ffe066"; // Foil glint
    ctx.fillRect(-4, -4, 3, 3);
    ctx.fillRect(1, 1, 3, 3);
    ctx.fillStyle = "#222222"; // Dark cavity
    ctx.fillRect(-2, -1, 4, 3);
    ctx.fillStyle = "#00f0ff"; // Micro-thruster nozzle / wire
    ctx.fillRect(-6, 2, 2, 2);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(4, -6, 1, 3);
  }

  ctx.restore();
}

/**
 * Draw retro pixel impact explosion & shockwave
 */
export function drawPixelImpactExplosion(
  ctx: CanvasRenderingContext2D,
  progress: number // 0.0 to 1.0 (flash -> expansion -> dissipation)
) {
  if (progress < 0.0 || progress > 1.0) return;

  ctx.save();
  ctx.imageSmoothingEnabled = false;

  const maxR = 24;
  const curR = Math.max(2, Math.floor(progress * maxR));

  if (progress < 0.25) {
    // Phase 1: High-intensity white/amber contact flash
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(-curR * 2, -curR * 0.5, curR * 4, curR);
    ctx.fillRect(-curR * 0.5, -curR * 2, curR, curR * 4);
    ctx.fillStyle = "#ffcc00";
    ctx.fillRect(-curR, -curR, curR * 2, curR * 2);

  } else if (progress < 0.7) {
    // Phase 2: Expanding pixel fireball & shockwave spikes
    const alpha = (1.0 - (progress - 0.25) / 0.45);
    ctx.fillStyle = `rgba(255, 102, 0, ${alpha})`;
    ctx.fillRect(-curR, -curR, curR * 2, curR * 2);

    ctx.fillStyle = `rgba(255, 204, 0, ${alpha})`;
    ctx.fillRect(-curR * 0.7, -curR * 0.7, curR * 1.4, curR * 1.4);

    ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
    ctx.fillRect(-curR * 0.3, -curR * 0.3, curR * 0.6, curR * 0.6);

    // Radiating pixel shockwave spikes
    const spikeDist = curR * 1.5;
    ctx.fillStyle = `rgba(255, 51, 0, ${alpha})`;
    const spikes = [
      [spikeDist, 0], [-spikeDist, 0], [0, spikeDist], [0, -spikeDist],
      [spikeDist * 0.7, spikeDist * 0.7], [-spikeDist * 0.7, spikeDist * 0.7],
      [spikeDist * 0.7, -spikeDist * 0.7], [-spikeDist * 0.7, -spikeDist * 0.7],
    ];
    spikes.forEach(([sx, sy]) => {
      ctx.fillRect(sx - 1.5, sy - 1.5, 3, 3);
    });

  } else {
    // Phase 3: Dissipating incandescent sparks
    const alpha = (1.0 - progress) / 0.3;
    ctx.fillStyle = `rgba(255, 153, 0, ${alpha})`;
    const sparkOffsets = [
      [curR * 1.2, curR * 0.4], [-curR * 1.1, -curR * 0.6],
      [-curR * 0.5, curR * 1.3], [curR * 0.7, -curR * 1.1],
      [curR * 1.4, -curR * 0.3], [-curR * 1.3, curR * 0.8],
    ];
    sparkOffsets.forEach(([sx, sy]) => {
      ctx.fillRect(sx - 1, sy - 1, 2, 2);
    });
  }

  ctx.restore();
}

