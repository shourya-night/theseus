/* ===========================================================================
 * THESEUS — Orbit Alignment Diagnostic  (browser console, read-only)
 * ===========================================================================
 * Answers brief item A.1 by MEASURING, in the running scene, whether each
 * planet sits on its own drawn orbit line — and if not, whether the error is
 * radial (wrong ellipse shape/size) or along-track (wrong phase).
 *
 * It changes nothing. It only reads the scene graph three.js already exposes
 * through its built-in devtools hook.
 *
 * ── HOW TO RUN ────────────────────────────────────────────────────────────
 *
 *  STEP 1.  Open the app, open DevTools console, paste this ONE line, press
 *           Enter, then RELOAD the page (F5). The hook must exist before the
 *           scene is constructed.
 *
 *      window.__THREE_DEVTOOLS__=new EventTarget();window.__thsv=[];window.__THREE_DEVTOOLS__.addEventListener('observe',e=>window.__thsv.push(e.detail));
 *
 *  STEP 2.  After the reload, let the scene render for a second, then paste
 *           this whole file into the console and press Enter.
 *
 *  STEP 3.  Copy the printed table back to me.
 *
 *  Optional: scrub the timeline, then re-run STEP 2 to see how the numbers
 *  move with simulation time. Also worth running once zoomed in on Mars and
 *  once at the full-system framing — if the numbers change with the camera,
 *  the cause is render precision rather than geometry.
 * ========================================================================= */

(() => {
  const observed = window.__thsv;
  if (!observed || !observed.length) {
    console.error(
      '[THESEUS] No scene captured. Run STEP 1 (the one-liner), reload the page, then re-run this.'
    );
    return;
  }

  // The hook sees both the Scene and the WebGLRenderer; pick the Scene.
  const scene = observed.find(o => o && o.isScene);
  if (!scene) {
    console.error('[THESEUS] Devtools hook fired but no Scene was observed.', observed);
    return;
  }

  const AU_SCENE = 14959.78707;   // 1 AU in scene units (1 unit = 10,000 km)
  const byName = new Map();
  scene.traverse(o => { if (o.name) byName.set(o.name, o); });

  const ids = ['mercury','venus','earth','mars','jupiter','saturn',
               'uranus','neptune','pluto'];

  // ── minimum distance from a point to a polyline, plus the nearest point ──
  const nearestOnPolyline = (px, py, pz, verts, m) => {
    let best = Infinity, bx = 0, by = 0, bz = 0;
    const n = verts.length / 3;
    // transform helper (line matrixWorld, normally identity)
    const tx = (x, y, z) => m
      ? [ m[0]*x + m[4]*y + m[8]*z  + m[12],
          m[1]*x + m[5]*y + m[9]*z  + m[13],
          m[2]*x + m[6]*y + m[10]*z + m[14] ]
      : [x, y, z];

    let [ax, ay, az] = tx(verts[0], verts[1], verts[2]);
    for (let i = 1; i < n; i++) {
      const [cx, cy, cz] = tx(verts[i*3], verts[i*3+1], verts[i*3+2]);
      const ux = cx - ax, uy = cy - ay, uz = cz - az;
      const wx = px - ax, wy = py - ay, wz = pz - az;
      const uu = ux*ux + uy*uy + uz*uz;
      let t = uu > 0 ? (wx*ux + wy*uy + wz*uz) / uu : 0;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const qx = ax + ux*t, qy = ay + uy*t, qz = az + uz*t;
      const d = Math.hypot(px - qx, py - qy, pz - qz);
      if (d < best) { best = d; bx = qx; by = qy; bz = qz; }
      ax = cx; ay = cy; az = cz;
    }
    return { dist: best, x: bx, y: by, z: bz };
  };

  const rows = [];
  const notes = [];

  for (const id of ids) {
    const group = byName.get(`PlanetGroup_${id}`);
    const line  = byName.get(`PlanetOrbit_${id}`);
    if (!group || !line) { notes.push(`${id}: group=${!!group} line=${!!line} — not in scene`); continue; }

    scene.updateMatrixWorld(true);

    const p = { x: 0, y: 0, z: 0 };
    group.getWorldPosition(p);

    const attr = line.geometry.getAttribute('position');
    if (!attr) { notes.push(`${id}: orbit line has no position attribute`); continue; }
    const verts = attr.array;
    const m = line.matrixWorld ? line.matrixWorld.elements : null;

    const near = nearestOnPolyline(p.x, p.y, p.z, verts, m);

    // Radial split: how much of the miss is "wrong distance from the Sun"
    // versus "wrong place along the curve".
    const rPlanet = Math.hypot(p.x, p.y, p.z);
    const rPath   = Math.hypot(near.x, near.y, near.z);
    const radial  = Math.abs(rPlanet - rPath);
    const along   = Math.sqrt(Math.max(0, near.dist * near.dist - radial * radial));

    // Planet's own visual radius, for a meaningful "is this visible" scale.
    let radius = NaN;
    const mesh = byName.get(`PlanetMesh_${id}`);
    if (mesh && mesh.geometry) {
      mesh.geometry.computeBoundingSphere();
      radius = mesh.geometry.boundingSphere.radius;
    }

    // Semi-major axis implied by the drawn line, as a cross-check on shape.
    let rmin = Infinity, rmax = 0;
    for (let i = 0; i < verts.length; i += 3) {
      const r = Math.hypot(verts[i], verts[i+1], verts[i+2]);
      if (r < rmin) rmin = r;
      if (r > rmax) rmax = r;
    }
    const aLine = (rmin + rmax) / 2;
    const eLine = (rmax - rmin) / (rmax + rmin);

    const lineOffset = m ? Math.hypot(m[12], m[13], m[14]) : 0;

    rows.push({
      body: id,
      'miss (AU)':        +(near.dist / AU_SCENE).toFixed(6),
      'miss / a %':       +((near.dist / aLine) * 100).toFixed(4),
      'miss / radii':     +(near.dist / radius).toFixed(2),
      'radial part':      +(radial / near.dist || 0).toFixed(3),
      'along part':       +(along  / near.dist || 0).toFixed(3),
      'r body (AU)':      +(rPlanet / AU_SCENE).toFixed(5),
      'a from line (AU)': +(aLine / AU_SCENE).toFixed(5),
      'e from line':      +eLine.toFixed(5),
      'line origin off':  +lineOffset.toFixed(3),
      'grp vis':          group.visible,
      'line vis':         line.visible,
      pts:                verts.length / 3,
    });
  }

  console.log('%c THESEUS orbit alignment — live scene ', 'background:#9A6704;color:#000;font-weight:bold');
  console.table(rows);
  if (notes.length) console.warn('[THESEUS] ' + notes.join('\n[THESEUS] '));

  console.log(
    '%cHow to read this:',
    'font-weight:bold',
    '\n  miss / radii  — distance from the body to its own drawn line, in units of the body\'s own radius.' +
    '\n                  < 1  => the body IS on its line; the visible problem is something else.' +
    '\n                  >> 1 => genuinely off the line.' +
    '\n  radial part / along part — how the miss splits. ~1.0 radial means the ellipse is the wrong' +
    '\n                  size or shape; ~1.0 along-track means the phase (M0 / mean motion) is wrong.' +
    '\n  a from line / e from line — semi-major axis and eccentricity recovered from the drawn' +
    '\n                  vertices. Compare against the catalog to catch a bad element set.' +
    '\n  line origin off — translation on the orbit line. Should be 0 for a planet.'
  );

  // Copy-paste friendly output.
  window.__thsvReport = rows;
  console.log('Full rows also in window.__thsvReport — copy(__thsvReport) puts JSON on the clipboard.');
  return rows;
})();
