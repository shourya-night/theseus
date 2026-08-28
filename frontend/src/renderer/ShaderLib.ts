/**
 * THESEUS Shader Library
 * ======================
 * Shared GLSL code fragments for procedural planet surfaces,
 * atmospheres, rings, stars, and other visual effects.
 *
 * All noise functions produce deterministic results from position inputs.
 * No external textures required.
 */

// ─── COMMON GLSL INCLUDES ───────────────────────────────────────────

export const GLSL_NOISE = /* glsl */ `
// Simplex 3D noise (Ashima Arts)
vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x * 34.0) + 10.0) * x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
  const vec2 C = vec2(1.0/6.0, 1.0/3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

  vec3 i = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);

  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);

  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;

  i = mod289(i);
  vec4 p = permute(permute(permute(
    i.z + vec4(0.0, i1.z, i2.z, 1.0))
    + i.y + vec4(0.0, i1.y, i2.y, 1.0))
    + i.x + vec4(0.0, i1.x, i2.x, 1.0));

  float n_ = 0.142857142857;
  vec3 ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);

  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);

  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);

  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;

  vec4 m = max(0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 105.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}

// Fractal Brownian Motion
float fbm(vec3 p, int octaves) {
  float value = 0.0;
  float amplitude = 0.5;
  float frequency = 1.0;
  for (int i = 0; i < 8; i++) {
    if (i >= octaves) break;
    value += amplitude * snoise(p * frequency);
    frequency *= 2.0;
    amplitude *= 0.5;
  }
  return value;
}

// Domain-warped FBM for more organic-looking surfaces
float warpedFbm(vec3 p, int octaves) {
  vec3 q = vec3(
    fbm(p + vec3(0.0, 0.0, 0.0), octaves),
    fbm(p + vec3(5.2, 1.3, 2.8), octaves),
    fbm(p + vec3(1.7, 9.2, 3.4), octaves)
  );
  return fbm(p + 4.0 * q, octaves);
}

// Crater function: returns depth at point p relative to crater center c
float crater(vec3 p, vec3 c, float radius, float depth) {
  float d = length(p - c);
  float rim = smoothstep(radius * 0.85, radius, d) * depth * 0.3;
  float bowl = (1.0 - smoothstep(0.0, radius * 0.85, d)) * depth;
  return rim - bowl;
}
`;

// ─── ATMOSPHERE SHADER ──────────────────────────────────────────────

export const GLSL_ATMOSPHERE_VERTEX = /* glsl */ `
varying vec3 vNormal;
varying vec3 vWorldPosition;

void main() {
  vNormal = normalize(mat3(modelMatrix) * normal);
  vec4 worldPos = modelMatrix * vec4(position, 1.0);
  vWorldPosition = worldPos.xyz;
  gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

export const GLSL_ATMOSPHERE_FRAGMENT = /* glsl */ `
uniform vec3 uSunDirection;
uniform vec3 uAtmosphereColor;
uniform float uAtmosphereDensity;
uniform float uAtmosphereRadius;
uniform float uPlanetRadius;

varying vec3 vNormal;
varying vec3 vWorldPosition;

void main() {
  vec3 viewDir = normalize(cameraPosition - vWorldPosition);
  float rim = 1.0 - max(dot(viewDir, vNormal), 0.0);
  float sunDot = max(dot(vNormal, uSunDirection), 0.0);

  // Rayleigh-like scattering approximation
  float scatter = pow(rim, 2.5) * uAtmosphereDensity;
  float daylight = 0.3 + 0.7 * sunDot;

  vec3 color = uAtmosphereColor * scatter * daylight;
  float alpha = scatter * 0.85;

  // Limb brightening
  float limb = pow(rim, 4.0) * 0.3;
  color += uAtmosphereColor * limb;
  alpha += limb * 0.5;

  gl_FragColor = vec4(color, clamp(alpha, 0.0, 0.9));
}
`;

// ─── RING SHADER ────────────────────────────────────────────────────

export const GLSL_RING_VERTEX = /* glsl */ `
varying vec2 vUv;
varying vec3 vWorldPosition;
varying vec3 vNormal;
// Position in the ring's own object space. The radial coordinate has to come
// from here — see the note in the fragment shader.
varying vec3 vLocalPosition;

void main() {
  vUv = uv;
  vLocalPosition = position;
  vNormal = normalize(mat3(modelMatrix) * normal);
  vec4 worldPos = modelMatrix * vec4(position, 1.0);
  vWorldPosition = worldPos.xyz;
  gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

export const GLSL_RING_FRAGMENT = /* glsl */ `
${GLSL_NOISE}

uniform vec3 uSunDirection;
uniform vec3 uPlanetPosition;
uniform float uPlanetRadius;
uniform float uInnerRadius;
uniform float uOuterRadius;
uniform vec3 uRingColor;
uniform float uRingDensityScale;

varying vec2 vUv;
varying vec3 vWorldPosition;
varying vec3 vNormal;
varying vec3 vLocalPosition;

void main() {
  // RADIAL COORDINATE FROM GEOMETRY, NOT FROM UV.
  //
  // THREE.RingGeometry assigns uv as a PLANAR projection of the vertex over
  // the disc's bounding square — uv.y is a linear function of the local y
  // coordinate, not of the distance from the centre. Keying the ring
  // structure off vUv.y therefore drew the A/B/C bands and the Cassini
  // Division as straight parallel stripes across the disc instead of as
  // concentric annuli. The geometry is rotated into the XZ plane at build
  // time, so the true radius is the length of the local XZ vector.
  float radius = length(vLocalPosition.xz);
  float t = clamp((radius - uInnerRadius) / max(uOuterRadius - uInnerRadius, 1e-6), 0.0, 1.0);

  // ── Radial density profile ──────────────────────────────────────
  // Fractions of the way across the ring span, following Saturn's structure.
  float density = 0.0;

  // C Ring: faint, innermost
  density += smoothstep(0.00, 0.06, t) * smoothstep(0.24, 0.18, t) * 0.30;
  // B Ring: brightest and densest
  density += smoothstep(0.24, 0.30, t) * smoothstep(0.60, 0.54, t) * 0.95;
  // A Ring: outside the Cassini Division
  density += smoothstep(0.66, 0.71, t) * smoothstep(0.94, 0.90, t) * 0.68;
  // F Ring: narrow and detached
  density += smoothstep(0.965, 0.975, t) * smoothstep(0.995, 0.985, t) * 0.45;

  // Cassini Division: the real gap between B and A
  density *= 1.0 - smoothstep(0.60, 0.63, t) * smoothstep(0.69, 0.66, t) * 0.94;
  // Encke Gap: narrow, in the outer A Ring
  density *= 1.0 - smoothstep(0.855, 0.862, t) * smoothstep(0.878, 0.871, t) * 0.85;

  // ── Fine radial structure (ringlets) ────────────────────────────
  float ringlets = snoise(vec3(t * 340.0, 0.0, 0.0)) * 0.06
                 + snoise(vec3(t * 1100.0, 7.3, 0.0)) * 0.03;
  density = clamp((density + ringlets) * uRingDensityScale, 0.0, 1.0);

  if (density < 0.012) discard;

  // ── Illumination ────────────────────────────────────────────────
  // Ring particles are lit from whichever face the Sun is on, so the
  // response uses |n . s| rather than a signed dot: the unlit face is dimmer
  // but not black, which is what forward scattering through the ring plane
  // actually looks like.
  vec3 ringNormal = normalize(vNormal);
  float incidence = abs(dot(ringNormal, normalize(uSunDirection)));
  // Ring particles are strongly forward-scattering and the unlit face still
  // picks up plenty of light through the plane, so the floor is generous.
  float lit = 0.38 + 0.62 * incidence;

  // ── Planet shadow cast onto the rings ───────────────────────────
  // Cylinder test: how far this point sits from the planet-Sun axis, on the
  // far side of the planet.
  vec3 toSun = normalize(uSunDirection);
  vec3 fromPlanet = vWorldPosition - uPlanetPosition;
  float alongSun = dot(fromPlanet, toSun);
  if (alongSun < 0.0) {
    float perpDist = length(fromPlanet - toSun * alongSun);
    // Soft edge across one planetary radius approximates the penumbra.
    lit *= mix(0.10, 1.0, smoothstep(uPlanetRadius * 0.92, uPlanetRadius * 1.12, perpDist));
  }

  // Slight colour gradient: the B Ring is the brightest and least red.
  vec3 color = mix(uRingColor * 0.86, uRingColor * 1.22, smoothstep(0.2, 0.6, t));
  color *= lit;

  gl_FragColor = vec4(color, density * 0.9);
}
`;

// ─── STAR POINT SHADER ──────────────────────────────────────────────

export const GLSL_STAR_VERTEX = /* glsl */ `
attribute float aMagnitude;
attribute vec3 aColor;

varying vec3 vColor;
varying float vMagnitude;

void main() {
  vColor = aColor;
  vMagnitude = aMagnitude;

  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  float size = max(1.0, 4.0 - aMagnitude * 0.5);
  gl_PointSize = size;
  gl_Position = projectionMatrix * mvPosition;
}
`;

export const GLSL_STAR_FRAGMENT = /* glsl */ `
varying vec3 vColor;
varying float vMagnitude;

void main() {
  float d = length(gl_PointCoord - vec2(0.5));
  if (d > 0.5) discard;
  float alpha = smoothstep(0.5, 0.1, d);
  float brightness = max(0.3, 1.0 - vMagnitude * 0.12);
  gl_FragColor = vec4(vColor * brightness, alpha);
}
`;

// ─── SUN SHADER ─────────────────────────────────────────────────────

export const GLSL_SUN_VERTEX = /* glsl */ `
varying vec3 vNormal;
varying vec3 vPosition;
varying vec3 vWorldPosition;
varying vec2 vUv;

void main() {
  vNormal = normalize(mat3(modelMatrix) * normal);
  vPosition = position;
  vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

export const GLSL_SUN_FRAGMENT = /* glsl */ `
${GLSL_NOISE}

uniform float uTime;
uniform float uIntensity;

varying vec3 vNormal;
varying vec3 vPosition;
varying vec3 vWorldPosition;

/**
 * Convective cell field.
 *
 * abs() of a simplex field puts thin dark lanes along the zero crossings with
 * bright interiors between them — which is what granulation actually looks
 * like. The exponent brightens the interiors without widening the lanes.
 */
float granule(vec3 p) {
  return pow(abs(snoise(p)), 0.45);
}

void main() {
  vec3 n = normalize(vPosition);

  // ── Convection at three scales ──────────────────────────────────
  // Supergranulation (~30,000 km), granulation (~1,000 km), and a fine
  // scale that keeps the surface from looking smooth when zoomed in.
  // Weights are deliberately biased toward the small scales: the previous
  // shader put most of its variance in a low-frequency term, which is what
  // made the disc read as blotchy rather than granular.
  float superGran = granule(n *  9.0 + uTime * 0.004);
  float gran      = granule(n * 34.0 + uTime * 0.020);
  float fine      = granule(n * 95.0 + uTime * 0.050);

  float convection = superGran * 0.26 + gran * 0.46 + fine * 0.28;
  // Stretch the mid range so the cells actually separate from the lanes.
  // Without this the field clusters near its mean, the disc saturates to flat
  // white under tone mapping, and the granulation is only visible at the limb.
  convection = smoothstep(0.18, 0.80, convection);

  // ── Faculae: bright magnetic network on supergranule boundaries ──
  float network = 1.0 - smoothstep(0.0, 0.22, abs(snoise(n * 9.0)));
  float faculae = network * 0.16;

  // ── Sunspots ────────────────────────────────────────────────────
  // Rare, small, umbra inside penumbra, and confined to the active
  // latitudes (roughly +/- 35 deg) rather than scattered over the poles.
  float latMask  = smoothstep(0.60, 0.40, abs(n.y));
  float spotField = snoise(n * 2.6 + vec3(11.3, 4.7, 2.9));
  // Wide, soft penumbra with a much smaller umbra inside it, so a spot reads
  // as a structure rather than as a flat dark dot.
  float penumbra = smoothstep(0.60, 0.80, spotField) * latMask;
  float umbra    = smoothstep(0.82, 0.90, spotField) * latMask;

  // ── Photospheric colour ─────────────────────────────────────────
  vec3 coreColor     = vec3(1.00, 0.97, 0.90);
  vec3 midColor      = vec3(0.99, 0.72, 0.34);
  vec3 limbColor     = vec3(0.99, 0.62, 0.26);
  vec3 penumbraColor = vec3(0.70, 0.38, 0.13);
  vec3 umbraColor    = vec3(0.26, 0.12, 0.05);

  vec3 color = mix(midColor, coreColor, convection);
  color += coreColor * faculae;
  color = mix(color, penumbraColor, penumbra * 0.75);
  color = mix(color, umbraColor, umbra * 0.90);

  // ── Limb darkening ──────────────────────────────────────────────
  // Quadratic law I(mu)/I(1) = 1 - u1(1-mu) - u2(1-mu)^2, with coefficients
  // for the visible band. mu is the cosine of the angle between the surface
  // normal and the line of sight, so it goes 1 at disc centre to 0 at the limb.
  vec3 viewDir = normalize(cameraPosition - vWorldPosition);
  float mu = clamp(dot(normalize(vNormal), viewDir), 0.0, 1.0);
  float oneMinusMu = 1.0 - mu;
  float limbDarkening = 1.0 - 0.47 * oneMinusMu - 0.23 * oneMinusMu * oneMinusMu;

  // Cooler, higher layers dominate near the limb, so it also reddens.
  color = mix(limbColor, color, smoothstep(0.0, 0.55, mu));
  color *= limbDarkening;

  // Output above 1.0 on purpose: the renderer uses ACES tone mapping, and a
  // star that never exceeds display white reads as a flat orange ball.
  gl_FragColor = vec4(color * uIntensity, 1.0);
}
`;

// ─── CORONA SHADER ──────────────────────────────────────────────────

/**
 * Corona / limb glow.
 *
 * Anchored to the PHOTOSPHERE EDGE rather than to the shell it is drawn on.
 *
 * The previous version keyed its brightness off the shell's own surface
 * normal, so the glow appeared at the shell's silhouette — a ring at 1.45
 * solar radii, visibly detached from the disc, which read as a second
 * mismatched sphere. Here each fragment instead computes the IMPACT
 * PARAMETER of its view ray: the perpendicular distance from the Sun's centre
 * to the line of sight through that fragment. That is the apparent radius at
 * which the fragment sits on screen, independent of the shell's geometry, so
 * the glow always begins exactly at the visible edge of the photosphere.
 */
export const GLSL_CORONA_FRAGMENT = /* glsl */ `
${GLSL_NOISE}

uniform float uTime;
uniform vec3  uSunColor;
uniform vec3  uSunCenter;
uniform float uPhotosphereRadius;
uniform float uCoronaRadius;

varying vec3 vNormal;
varying vec3 vWorldPosition;

void main() {
  vec3 toFragment = vWorldPosition - uSunCenter;
  vec3 viewRay = normalize(vWorldPosition - cameraPosition);

  // Perpendicular component of the fragment offset about the view ray.
  vec3 perpendicular = toFragment - viewRay * dot(toFragment, viewRay);
  float apparentRadius = length(perpendicular);

  // Normalised so 1.0 is exactly the photosphere limb.
  float t = apparentRadius / max(uPhotosphereRadius, 1e-6);

  // Nothing over the disc itself; the photosphere owns that area.
  float inner = smoothstep(0.93, 1.02, t);

  // Two components. A tight bright band hugging the limb, plus a much
  // broader, fainter halo. A single exponential gives either a hard ring or a
  // washed-out ball; the sum is what reads as a corona.
  float extent = max(uCoronaRadius / max(uPhotosphereRadius, 1e-6) - 1.0, 1e-3);
  float d = max(t - 1.0, 0.0);
  float nearGlow = exp(-d * (9.0 / extent)) * 0.85;
  float farGlow  = exp(-d * (2.2 / extent)) * 0.30;

  // Streamers: broad and shallow. Keyed to direction around the limb rather
  // than to world position, so they stay put as the camera moves. Kept mild
  // on purpose — high-contrast angular noise here reads as a cartoon
  // starburst rather than as coronal structure.
  vec3 streamerDir = normalize(perpendicular + vec3(1e-6));
  float streamers = 0.88 + 0.20 * snoise(streamerDir * 2.1 + vec3(0.0, uTime * 0.006, 0.0));
  // Streamers modulate only the extended halo; the limb band stays smooth.
  float intensity = inner * (nearGlow + farGlow * streamers);

  gl_FragColor = vec4(uSunColor * intensity * 1.35, clamp(intensity, 0.0, 1.0));
}
`;

// ─── PLANET SURFACE SHADERS ─────────────────────────────────────────

/** Generic rocky/cratered planet (Mercury, Moon) */
export const GLSL_ROCKY_FRAGMENT = /* glsl */ `
${GLSL_NOISE}

uniform vec3 uSunDirection;
uniform vec3 uBaseColor;
uniform vec3 uCraterColor;
uniform float uCraterDensity;
uniform float uRoughness;

varying vec3 vNormal;
varying vec3 vPosition;
varying vec2 vUv;

void main() {
  vec3 pos = normalize(vPosition) * 5.0;

  // Surface relief via FBM
  float terrain = fbm(pos * 3.0, 6) * 0.5 + 0.5;

  // Crater features
  float craters = 0.0;
  craters += smoothstep(0.55, 0.65, snoise(pos * 8.0)) * 0.3;
  craters += smoothstep(0.6, 0.7, snoise(pos * 15.0)) * 0.2;
  craters += smoothstep(0.65, 0.75, snoise(pos * 25.0)) * 0.15;
  craters *= uCraterDensity;

  vec3 color = mix(uBaseColor, uCraterColor, craters);
  color *= 0.7 + 0.3 * terrain;

  // Normal perturbation for bump effect
  float eps = 0.01;
  float hx = fbm((pos + vec3(eps, 0.0, 0.0)) * 3.0, 4);
  float hy = fbm((pos + vec3(0.0, eps, 0.0)) * 3.0, 4);
  float hz = fbm((pos + vec3(0.0, 0.0, eps)) * 3.0, 4);
  float h0 = fbm(pos * 3.0, 4);
  vec3 bumpNormal = normalize(vNormal + vec3(hx - h0, hy - h0, hz - h0) * uRoughness * 8.0);

  // Lighting
  float diff = max(dot(bumpNormal, uSunDirection), 0.0);
  float ambient = 0.06;
  color *= ambient + diff * 0.94;

  gl_FragColor = vec4(color, 1.0);
}
`;

/** Earth-like planet with continents, oceans, clouds */
export const GLSL_EARTH_FRAGMENT = /* glsl */ `
${GLSL_NOISE}

uniform vec3 uSunDirection;
uniform float uTime;
uniform float uCloudCover;

varying vec3 vNormal;
varying vec3 vPosition;
varying vec3 vWorldPosition;
varying vec2 vUv;

void main() {
  vec3 pos = normalize(vPosition);
  vec3 sPos = pos * 2.5;

  // ── Continents ──────────────────────────────────────────────────
  // Low-frequency structure carries the landmasses; the warped term adds
  // coastline detail without breaking them up.
  float baseContinent = fbm(sPos * 0.7, 4);
  float detailTerrain = warpedFbm(sPos * 1.8, 4) * 0.25;
  float landValue = baseContinent + detailTerrain;
  float isLand = smoothstep(-0.02, 0.08, landValue);

  // ── Ocean ───────────────────────────────────────────────────────
  vec3 deepOcean = vec3(0.012, 0.055, 0.20);
  vec3 shallowOcean = vec3(0.05, 0.26, 0.45);
  float oceanDepth = clamp(-landValue * 2.5, 0.0, 1.0);
  vec3 oceanColor = mix(shallowOcean, deepOcean, oceanDepth);

  // ── Land biomes ─────────────────────────────────────────────────
  vec3 lowlandGreen = vec3(0.12, 0.38, 0.09);
  vec3 highlandBrown = vec3(0.42, 0.32, 0.16);
  vec3 mountainPeak = vec3(0.48, 0.45, 0.42);
  vec3 desertOchre = vec3(0.65, 0.52, 0.28);

  float elev = clamp(landValue * 2.0, 0.0, 1.0);
  float moisture = fbm(sPos * 1.5 + vec3(5.2, 1.3, 0.0), 3) * 0.5 + 0.5;
  vec3 landColor = mix(lowlandGreen, highlandBrown, elev);
  landColor = mix(landColor, mountainPeak, smoothstep(0.5, 0.85, elev));
  landColor = mix(landColor, desertOchre, smoothstep(0.25, 0.65, 1.0 - moisture) * (1.0 - elev));

  // ── Polar ice ───────────────────────────────────────────────────
  float latitude = abs(pos.y);
  float iceNoise = fbm(sPos * 3.0, 3) * 0.06;
  // Permanent ice sits poleward of roughly 66 deg, i.e. |sin(lat)| > 0.91.
  // The previous 0.68 threshold is 43 deg, which iced over a quarter of the
  // globe and buried the northern continents.
  float iceCap = smoothstep(0.88 + iceNoise, 0.955 + iceNoise, latitude);
  vec3 iceColor = vec3(0.92, 0.94, 0.97);

  vec3 surfaceColor = mix(oceanColor, landColor, isLand);
  surfaceColor = mix(surfaceColor, iceColor, iceCap);

  // ── Clouds ──────────────────────────────────────────────────────
  // Previously: smoothstep(0.45 - cover*0.15, 0.75, fbm) mixed at 0.65 over
  // the whole sphere, which put partial cloud on almost every pixel and
  // washed the surface out to a uniform mottled grey. Two changes fix that.
  //
  // First, the coverage mask is thresholded much higher, so most of the
  // sphere is genuinely clear and the clouds that remain are discrete
  // systems rather than a haze.
  //
  // Second, banding: real cloud cover is concentrated in the ITCZ and the
  // mid-latitude storm tracks, and sparse over the subtropical highs. That
  // structure is what makes an image read as Earth rather than as noise.
  float cloudNoise = fbm(sPos * 2.4 + vec3(uTime * 0.003, 0.0, uTime * 0.0015), 5) * 0.5 + 0.5;
  float wisps = fbm(sPos * 6.5 + vec3(uTime * 0.006, 0.0, 0.0), 3) * 0.5 + 0.5;

  float lat = pos.y;
  float itcz = exp(-pow(lat / 0.14, 2.0));                       // equatorial band
  float stormTracks = exp(-pow((abs(lat) - 0.72) / 0.20, 2.0));  // mid-latitudes
  float subtropicalHigh = 1.0 - exp(-pow((abs(lat) - 0.42) / 0.16, 2.0)) * 0.75;
  float latBias = clamp((0.35 + 0.65 * max(itcz, stormTracks)) * subtropicalHigh, 0.0, 1.0);

  float cloudField = cloudNoise * 0.72 + wisps * 0.28;
  float clouds = smoothstep(0.62 - uCloudCover * 0.10, 0.86, cloudField * latBias + 0.18 * latBias);
  clouds = clamp(clouds, 0.0, 1.0);

  vec3 cloudColor = vec3(0.96, 0.97, 0.99);
  surfaceColor = mix(surfaceColor, cloudColor, clouds * 0.88);

  // ── Day / night ─────────────────────────────────────────────────
  float diff = dot(normalize(vNormal), normalize(uSunDirection));
  float daylight = smoothstep(-0.12, 0.18, diff);

  // City lights on the dark side, on land, away from the ice caps and
  // dimmed under cloud.
  // smoothstep, not step: a hard threshold on a high-frequency field aliases
  // badly where the surface is nearly edge-on, which scattered gold speckle
  // right around the limb.
  float cityNoise = snoise(sPos * 16.0) * 0.5 + 0.5;
  // Fade the lights out where the surface is nearly edge-on. A dense noise
  // field sampled at grazing incidence aliases into scattered bright dots
  // around the limb, which is what produced the ring of gold speckle.
  float facing = clamp(dot(normalize(vNormal), normalize(cameraPosition - vWorldPosition)), 0.0, 1.0);
  float cities = isLand * smoothstep(0.60, 0.74, cityNoise)
               * (1.0 - iceCap) * (1.0 - clouds)
               * smoothstep(0.18, 0.48, facing) * 0.45;
  vec3 nightColor = vec3(1.0, 0.85, 0.42) * cities;

  vec3 litColor = surfaceColor * (0.04 + daylight * 0.96);
  litColor += nightColor * (1.0 - daylight);

  gl_FragColor = vec4(litColor, 1.0);
}
`;

/** Gas giant (Jupiter, Saturn) */
export const GLSL_GAS_GIANT_FRAGMENT = /* glsl */ `
${GLSL_NOISE}

uniform vec3 uSunDirection;
uniform vec3 uBandColors[6];
uniform int uBandCount;
uniform float uStormIntensity;
uniform vec2 uStormCenter;
uniform float uTime;

varying vec3 vNormal;
varying vec3 vPosition;
varying vec2 vUv;

void main() {
  vec3 pos = normalize(vPosition);
  float lat = asin(pos.y) / 3.14159;

  // Atmospheric band structure
  float bandFreq = float(uBandCount) * 2.0;
  float bandBase = sin(lat * bandFreq * 3.14159) * 0.5 + 0.5;

  // Turbulence in bands
  float turb = snoise(vec3(pos.x * 8.0 + uTime * 0.002, lat * 20.0, pos.z * 8.0)) * 0.15;
  bandBase += turb;
  bandBase = clamp(bandBase, 0.0, 1.0);

  // Zonal wind shear distortion
  float shear = snoise(vec3(lat * 30.0, pos.x * 5.0 + uTime * 0.01, pos.z * 5.0)) * 0.1;
  bandBase += shear;

  // Map to band colors
  int idx = int(bandBase * float(uBandCount - 1));
  int idx2 = min(idx + 1, uBandCount - 1);
  float frac = fract(bandBase * float(uBandCount - 1));
  vec3 color = mix(uBandColors[idx], uBandColors[idx2], frac);

  // Storm feature (e.g. Great Red Spot)
  if (uStormIntensity > 0.01) {
    vec2 stormUV = vec2(
      atan(pos.z, pos.x) / 6.28318 + 0.5,
      lat + 0.5
    );
    float stormDist = length(stormUV - uStormCenter);
    float storm = smoothstep(0.06, 0.02, stormDist);
    // Spiral structure
    float angle = atan(stormUV.y - uStormCenter.y, stormUV.x - uStormCenter.x);
    float spiral = sin(angle * 3.0 + stormDist * 40.0 - uTime * 0.05) * 0.5 + 0.5;
    vec3 stormColor = mix(vec3(0.72, 0.28, 0.12), vec3(0.85, 0.5, 0.25), spiral);
    color = mix(color, stormColor, storm * uStormIntensity);
  }

  // Lighting
  float diff = max(dot(vNormal, uSunDirection), 0.0);
  color *= 0.12 + diff * 0.88;

  gl_FragColor = vec4(color, 1.0);
}
`;

/** Ice giant (Uranus, Neptune) */
export const GLSL_ICE_GIANT_FRAGMENT = /* glsl */ `
${GLSL_NOISE}

uniform vec3 uSunDirection;
uniform vec3 uBaseColor;
uniform vec3 uBandColor;
uniform float uBandFrequency;
uniform float uCloudIntensity;
uniform float uTime;

varying vec3 vNormal;
varying vec3 vPosition;

void main() {
  vec3 pos = normalize(vPosition);
  float lat = asin(pos.y) / 3.14159;

  // Subtle atmospheric bands
  float bands = sin(lat * uBandFrequency * 3.14159) * 0.5 + 0.5;
  float turb = snoise(vec3(pos.x * 6.0, lat * 15.0, pos.z * 6.0 + uTime * 0.001)) * 0.1;
  bands = clamp(bands + turb, 0.0, 1.0);

  vec3 color = mix(uBaseColor, uBandColor, bands * 0.3);

  // Cloud features
  float clouds = snoise(vec3(pos.x * 10.0 + uTime * 0.003, lat * 8.0, pos.z * 10.0)) * 0.5 + 0.5;
  clouds = smoothstep(0.55, 0.75, clouds) * uCloudIntensity;
  color = mix(color, vec3(0.8, 0.85, 0.9), clouds * 0.3);

  // Lighting
  float diff = max(dot(vNormal, uSunDirection), 0.0);
  color *= 0.1 + diff * 0.9;

  gl_FragColor = vec4(color, 1.0);
}
`;

/** Mars surface */
export const GLSL_MARS_FRAGMENT = /* glsl */ `
${GLSL_NOISE}

uniform vec3 uSunDirection;
uniform float uTime;

varying vec3 vNormal;
varying vec3 vPosition;

void main() {
  vec3 pos = normalize(vPosition);
  vec3 sPos = pos * 5.0;

  // Mars terrain
  float terrain = fbm(sPos * 2.0, 6) * 0.5 + 0.5;
  float detail = fbm(sPos * 8.0, 4) * 0.2;

  // Color palette
  vec3 rust = vec3(0.72, 0.32, 0.12);
  vec3 ochre = vec3(0.78, 0.52, 0.22);
  vec3 darkRock = vec3(0.35, 0.18, 0.08);
  vec3 brightDust = vec3(0.85, 0.6, 0.35);

  vec3 color = mix(rust, ochre, terrain);
  color = mix(color, darkRock, smoothstep(0.3, 0.5, detail + terrain * 0.3));
  color = mix(color, brightDust, smoothstep(0.65, 0.8, terrain));

  // Valles Marineris feature (equatorial canyon)
  float eqDist = abs(pos.y);
  float canyon = smoothstep(0.08, 0.02, eqDist) *
    smoothstep(0.3, 0.5, snoise(vec3(pos.x * 3.0, 0.0, pos.z * 3.0)));
  color = mix(color, darkRock * 0.7, canyon * 0.6);

  // Olympus Mons (large feature near equator)
  vec3 olympusCenter = normalize(vec3(0.8, 0.15, 0.3));
  float olympusDist = acos(clamp(dot(pos, olympusCenter), -1.0, 1.0));
  float olympus = smoothstep(0.15, 0.05, olympusDist) * 0.4;
  color = mix(color, brightDust, olympus);

  // Polar ice caps
  float latitude = abs(pos.y);
  float iceCap = smoothstep(0.78, 0.88, latitude + fbm(sPos, 3) * 0.05);
  vec3 iceColor = vec3(0.88, 0.90, 0.93);
  color = mix(color, iceColor, iceCap);

  // Bump normal
  float eps = 0.01;
  float h0 = fbm(sPos * 2.0, 4);
  float hx = fbm((sPos + vec3(eps, 0.0, 0.0)) * 2.0, 4);
  float hy = fbm((sPos + vec3(0.0, eps, 0.0)) * 2.0, 4);
  vec3 bumpNormal = normalize(vNormal + vec3(hx - h0, hy - h0, 0.0) * 5.0);

  // Lighting
  float diff = max(dot(bumpNormal, uSunDirection), 0.0);
  color *= 0.08 + diff * 0.92;

  gl_FragColor = vec4(color, 1.0);
}
`;

/** Venus atmosphere */
export const GLSL_VENUS_FRAGMENT = /* glsl */ `
${GLSL_NOISE}

uniform vec3 uSunDirection;
uniform float uTime;

varying vec3 vNormal;
varying vec3 vPosition;
varying vec3 vWorldPosition;

void main() {
  vec3 pos = normalize(vPosition) * 3.0;

  // Dense cloud layers
  float cloud1 = fbm(pos * 2.0 + vec3(uTime * 0.001, 0.0, 0.0), 5) * 0.5 + 0.5;
  float cloud2 = fbm(pos * 4.0 + vec3(0.0, uTime * 0.002, 0.0), 4) * 0.5 + 0.5;

  // Venus color palette (cream/ochre atmosphere)
  vec3 baseColor = vec3(0.85, 0.72, 0.42);
  vec3 darkBand = vec3(0.72, 0.58, 0.32);
  vec3 brightBand = vec3(0.92, 0.82, 0.55);

  vec3 color = mix(baseColor, darkBand, cloud1 * 0.4);
  color = mix(color, brightBand, cloud2 * 0.3);

  // Atmospheric haze
  vec3 viewDir = normalize(cameraPosition - vWorldPosition);
  float haze = 1.0 - max(dot(viewDir, vNormal), 0.0);
  haze = pow(haze, 2.0) * 0.4;
  color = mix(color, vec3(0.95, 0.88, 0.65), haze);

  // Lighting
  float diff = max(dot(vNormal, uSunDirection), 0.0);
  color *= 0.15 + diff * 0.85;

  gl_FragColor = vec4(color, 1.0);
}
`;

// ─── GENERIC PLANET VERTEX (shared) ─────────────────────────────────

export const GLSL_PLANET_VERTEX = /* glsl */ `
// NORMALS ARE IN WORLD SPACE.
//
// normalMatrix produces a VIEW-space normal, but every light direction in
// this renderer (uSunDirection, and the view vector built from
// cameraPosition minus vWorldPosition) is in world space. Mixing the two makes
// illumination follow the camera instead of the Sun: orbit around a planet and
// the terminator rotates with you. mat3(modelMatrix) keeps the normal in world
// space. Uniform scaling only, which is all this renderer applies to bodies.
varying vec3 vNormal;
varying vec3 vPosition;
varying vec3 vWorldPosition;
varying vec2 vUv;

void main() {
  vNormal = normalize(mat3(modelMatrix) * normal);
  vPosition = position;
  vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;
