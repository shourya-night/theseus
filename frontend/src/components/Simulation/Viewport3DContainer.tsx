/**
 * THESEUS High-Fidelity 3D Viewport Container
 * ===========================================
 * Owns the Three.js scene for the simulator and drives every renderer from a
 * single render loop.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * LIVE STATE, NOT CAPTURED STATE
 * ─────────────────────────────────────────────────────────────────────────
 * The render callback is registered once, on mount. It therefore must not
 * read props directly: a callback registered in a `[]` effect closes over the
 * first render's values forever, which previously froze the entire scene at
 * t = 0 — planets never moved, the timeline drove nothing, and a mission built
 * after mount never reached the renderer.
 *
 * Props that change over time are mirrored into refs by their own effects and
 * read through `.current` inside the loop. Anything added here that varies at
 * runtime must follow the same pattern.
 */

import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { SceneManager } from '../../renderer/SceneManager';
import { CameraController } from '../../renderer/CameraController';
import { LODManager } from '../../renderer/LODManager';
import { SunRenderer } from '../../renderer/bodies/SunRenderer';
import { PlanetRenderer } from '../../renderer/bodies/PlanetRenderer';
import { MoonRenderer } from '../../renderer/bodies/MoonRenderer';
import { AsteroidBeltRenderer } from '../../renderer/populations/AsteroidBeltRenderer';
import { NEORenderer } from '../../renderer/populations/NEORenderer';
import { CometRenderer } from '../../renderer/populations/CometRenderer';
import { KuiperBeltRenderer } from '../../renderer/populations/KuiperBeltRenderer';
import { ScatteredDiskRenderer } from '../../renderer/populations/ScatteredDiskRenderer';
import { OortCloudRenderer } from '../../renderer/populations/OortCloudRenderer';
import { MeteorStreamRenderer } from '../../renderer/populations/MeteorStreamRenderer';
import { TrojanRenderer } from '../../renderer/populations/TrojanRenderer';
import { HildaRenderer } from '../../renderer/populations/HildaRenderer';
import { CybeleRenderer } from '../../renderer/populations/CybeleRenderer';
import { CentaurRenderer } from '../../renderer/populations/CentaurRenderer';
import { ZodiacalDustRenderer } from '../../renderer/populations/ZodiacalDustRenderer';
import { StarfieldRenderer } from '../../renderer/environment/StarfieldRenderer';
import { SpacecraftRenderer } from '../../renderer/spacecraft/SpacecraftRenderer';
import { OrbitalOverlayRenderer } from '../../renderer/overlays/OrbitalOverlayRenderer';
import { DynamicsOverlayRenderer } from '../../renderer/overlays/DynamicsOverlayRenderer';
import { MissionOverlayRenderer } from '../../renderer/overlays/MissionOverlayRenderer';
import { ReferenceOverlayRenderer } from '../../renderer/overlays/ReferenceOverlayRenderer';
import { LagrangePointRenderer } from '../../renderer/overlays/LagrangePointRenderer';
import { SOIRenderer } from '../../renderer/overlays/SOIRenderer';
import { ArtificialObjectRenderer } from '../../renderer/artificial/ArtificialObjectRenderer';
import { LayerState } from '../Visualization/VisualizationLayersPanel';
import { SOLAR_SYSTEM_OBJECTS, getMoons } from '../../data/astronomicalObjects';
// Side-effect import: triggers registerObjects(ALL_MOONS) so that getMoons()
// finds the 26 moons defined in moonSystems.ts.
import '../../data/moonSystems';
import { ActiveRocket } from '../../types/mission';
import { getRocketStateAtTime } from '../../lib/simulationClock';
import { ROCKET_PRESETS } from '../../data/rocketPresets';
import { ViewContext, spacecraftCharacteristicRadiusM } from '../../renderer/VisualScale';

/**
 * Resolve the scene position of the body an ORBIT-X state history is
 * referenced to, from `metadata.central_body`.
 *
 * This replaces a string comparison on the mission's DESTINATION
 * (`destination === 'orbit' | 'leo'`), which was wrong for every body-centred
 * mission that was not an Earth orbit insertion — the state was then treated
 * as heliocentric and the vehicle appeared a fraction of an AU from the Sun,
 * which is how a craft ended up rendering inside it.
 *
 * Returns null when the declared central body is not in the catalog, in which
 * case the caller must not guess a frame.
 */
function resolveFrameOrigin(
  centralBody: string | undefined,
  bodyPositions: Map<string, THREE.Vector3>
): THREE.Vector3 | null {
  const key = (centralBody ?? '').trim().toLowerCase();
  if (!key) return null;
  if (key === 'sun' || key === 'sol') return bodyPositions.get('sun') ?? null;
  return bodyPositions.get(key) ?? null;
}

interface Viewport3DContainerProps {
  layers: LayerState;
  activeRockets: ActiveRocket[];
  simTimeSec: number;
  selectedObjectId: string | null;
  focusedObjectId: string | null;
}

export const Viewport3DContainer: React.FC<Viewport3DContainerProps> = ({
  layers,
  activeRockets,
  simTimeSec,
  selectedObjectId,
  focusedObjectId,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  // ─── Live prop mirrors ────────────────────────────────────────────
  // Read inside the render loop instead of the props themselves.
  const simTimeRef = useRef(simTimeSec);
  const activeRocketsRef = useRef(activeRockets);
  const layersRef = useRef(layers);

  useEffect(() => { simTimeRef.current = simTimeSec; }, [simTimeSec]);
  useEffect(() => { activeRocketsRef.current = activeRockets; }, [activeRockets]);
  useEffect(() => { layersRef.current = layers; }, [layers]);

  // ─── Renderer handles ─────────────────────────────────────────────
  const sceneManagerRef = useRef<SceneManager | null>(null);
  const cameraControllerRef = useRef<CameraController | null>(null);
  const planetRenderersRef = useRef<Map<string, PlanetRenderer>>(new Map());
  const moonRenderersRef = useRef<Map<string, MoonRenderer>>(new Map());

  const sunRendererRef = useRef<SunRenderer | null>(null);
  const starfieldRef = useRef<StarfieldRenderer | null>(null);
  const asteroidBeltRef = useRef<AsteroidBeltRenderer | null>(null);
  const neoRendererRef = useRef<NEORenderer | null>(null);
  const cometRendererRef = useRef<CometRenderer | null>(null);
  const kuiperBeltRef = useRef<KuiperBeltRenderer | null>(null);
  const scatteredDiskRef = useRef<ScatteredDiskRenderer | null>(null);
  const oortCloudRef = useRef<OortCloudRenderer | null>(null);
  const meteorStreamRef = useRef<MeteorStreamRenderer | null>(null);
  const trojanRef = useRef<TrojanRenderer | null>(null);
  const hildaRef = useRef<HildaRenderer | null>(null);
  const cybeleRef = useRef<CybeleRenderer | null>(null);
  const centaurRef = useRef<CentaurRenderer | null>(null);
  const zodiacalDustRef = useRef<ZodiacalDustRenderer | null>(null);
  const artificialRendererRef = useRef<ArtificialObjectRenderer | null>(null);
  const spacecraftRendererRef = useRef<SpacecraftRenderer | null>(null);

  const orbitalOverlayRef = useRef<OrbitalOverlayRenderer | null>(null);
  const dynamicsOverlayRef = useRef<DynamicsOverlayRenderer | null>(null);
  const missionOverlayRef = useRef<MissionOverlayRenderer | null>(null);
  const referenceOverlayRef = useRef<ReferenceOverlayRenderer | null>(null);
  const lagrangePointRef = useRef<LagrangePointRenderer | null>(null);
  const soiRendererRef = useRef<SOIRenderer | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // ── 1. Scene & camera ──────────────────────────────────────────
    const sm = new SceneManager({ container: containerRef.current });
    sceneManagerRef.current = sm;

    const cc = new CameraController({
      camera: sm.camera,
      domElement: sm.renderer.domElement,
    });
    cameraControllerRef.current = cc;

    const lod = new LODManager(sm.camera);

    // ── 2. Environment ─────────────────────────────────────────────
    const starfield = new StarfieldRenderer();
    starfieldRef.current = starfield;
    sm.add(starfield.points);

    // ── 3. Sun ─────────────────────────────────────────────────────
    const sun = new SunRenderer();
    sunRendererRef.current = sun;
    sm.add(sun.group);

    // The Sun sits at the scene origin, so this is the illumination source
    // every body derives its own sun direction from.
    const sunWorldPos = new THREE.Vector3(0, 0, 0);

    // ── 4. Planets & moons ─────────────────────────────────────────
    SOLAR_SYSTEM_OBJECTS
      .filter(o => o.type === 'PLANET' || o.type === 'DWARF_PLANET')
      .forEach(obj => {
        const pRenderer = new PlanetRenderer(obj);
        planetRenderersRef.current.set(obj.id, pRenderer);
        sm.add(pRenderer.group);
        sm.add(pRenderer.orbitLine);

        getMoons(obj.id).forEach(m => {
          const mRenderer = new MoonRenderer(m);
          moonRenderersRef.current.set(m.id, mRenderer);
          sm.add(mRenderer.group);
          sm.add(mRenderer.orbitLine);
        });
      });

    // ── 5. Small bodies ────────────────────────────────────────────
    const asteroidBelt = new AsteroidBeltRenderer();
    asteroidBeltRef.current = asteroidBelt;
    sm.add(asteroidBelt.instancedMesh);

    const neoRenderer = new NEORenderer();
    neoRendererRef.current = neoRenderer;
    sm.add(neoRenderer.group);

    const cometRenderer = new CometRenderer();
    cometRendererRef.current = cometRenderer;
    sm.add(cometRenderer.group);

    const kuiperBelt = new KuiperBeltRenderer();
    kuiperBeltRef.current = kuiperBelt;
    sm.add(kuiperBelt.instancedMesh);

    const scatteredDisk = new ScatteredDiskRenderer();
    scatteredDiskRef.current = scatteredDisk;
    sm.add(scatteredDisk.instancedMesh);

    const oortCloud = new OortCloudRenderer();
    oortCloudRef.current = oortCloud;
    sm.add(oortCloud.points);

    const meteorStreams = new MeteorStreamRenderer();
    meteorStreamRef.current = meteorStreams;
    sm.add(meteorStreams.group);

    const trojans = new TrojanRenderer();
    trojanRef.current = trojans;
    sm.add(trojans.group);

    const hildas = new HildaRenderer();
    hildaRef.current = hildas;
    sm.add(hildas.instancedMesh);

    const cybeles = new CybeleRenderer();
    cybeleRef.current = cybeles;
    sm.add(cybeles.instancedMesh);

    const centaurs = new CentaurRenderer();
    centaurRef.current = centaurs;
    sm.add(centaurs.instancedMesh);

    const zodiacalDust = new ZodiacalDustRenderer();
    zodiacalDustRef.current = zodiacalDust;
    sm.add(zodiacalDust.points);

    // ── 6. Artificial objects ──────────────────────────────────────
    const artificialRenderer = new ArtificialObjectRenderer();
    artificialRendererRef.current = artificialRenderer;
    sm.add(artificialRenderer.group);

    // Objects the elliptical propagator cannot represent are reported once,
    // rather than being dropped on the scene origin with a fake position.
    if (artificialRenderer.unplaceable.length > 0) {
      console.warn(
        '[THESEUS] Artificial objects omitted — position DATA UNAVAILABLE:\n' +
        artificialRenderer.unplaceable.map(u => `  • ${u.name}: ${u.reason}`).join('\n')
      );
    }

    // ── 7. Mission spacecraft ──────────────────────────────────────
    const spacecraft = new SpacecraftRenderer('falcon9', '#c9a05a');
    spacecraftRendererRef.current = spacecraft;
    sm.add(spacecraft.group);
    sm.add(spacecraft.trajectoryLine);

    // ── 8. Scientific overlays ─────────────────────────────────────
    const orbOverlay = new OrbitalOverlayRenderer();
    orbitalOverlayRef.current = orbOverlay;
    sm.add(orbOverlay.group);

    const dynOverlay = new DynamicsOverlayRenderer();
    dynamicsOverlayRef.current = dynOverlay;
    sm.add(dynOverlay.group);

    const missOverlay = new MissionOverlayRenderer();
    missionOverlayRef.current = missOverlay;
    sm.add(missOverlay.group);

    const refOverlay = new ReferenceOverlayRenderer();
    referenceOverlayRef.current = refOverlay;
    sm.add(refOverlay.group);

    const lagrange = new LagrangePointRenderer();
    lagrangePointRef.current = lagrange;
    sm.add(lagrange.group);

    const soi = new SOIRenderer();
    soiRendererRef.current = soi;
    sm.add(soi.group);

    // ── 9. Render loop ─────────────────────────────────────────────
    const bodyPositions = new Map<string, THREE.Vector3>();
    const viewContext: ViewContext = { camera: sm.camera, viewportHeightPx: 800 };

    // Trail geometry and vehicle size are rebuilt only when the mission
    // changes, not every frame.
    let lastRocketId: string | null = null;
    const warnedUnknownFrames = new Set<string>();

    sm.onRender((dt, elapsed) => {
      // Live values — never the captured props.
      const simTime = simTimeRef.current;
      const rockets = activeRocketsRef.current;

      cc.update(dt);

      // Anchor the sky to the observer. Must run after the camera update and
      // before the draw, so the field is centred on this frame's camera
      // position rather than the previous one.
      starfield.update(sm.camera);

      sun.update(elapsed);

      const viewportHeight = containerRef.current?.clientHeight ?? 800;
      viewContext.viewportHeightPx = viewportHeight;

      bodyPositions.clear();
      bodyPositions.set('sun', sunWorldPos);

      // Planets: position on their own prepared orbit, then illuminate from
      // the actual Sun-to-body geometry.
      planetRenderersRef.current.forEach((pRenderer, id) => {
        const worldPos = pRenderer.positionAtTime(simTime);
        pRenderer.update(simTime, sunWorldPos);
        bodyPositions.set(id, worldPos);

        const lodState = lod.evaluateObject(pRenderer.objectData, worldPos, viewportHeight);
        pRenderer.setLOD(lodState.level);

        getMoons(id).forEach(m => {
          const mRenderer = moonRenderersRef.current.get(m.id);
          if (!mRenderer) return;
          mRenderer.updateOrbitPosition(simTime, worldPos, sunWorldPos);
          mRenderer.setLOD(lodState.level);
          bodyPositions.set(m.id, mRenderer.planetRenderer.group.position);
        });
      });

      asteroidBelt.update(simTime);
      neoRenderer.update(simTime, viewContext);
      cometRenderer.update(simTime, sunWorldPos, viewContext);
      kuiperBelt.update(simTime);
      scatteredDisk.update(simTime);
      meteorStreams.update(simTime);
      trojans.update(simTime);
      hildas.update(simTime);
      cybeles.update(simTime);
      centaurs.update(simTime);
      artificialRenderer.update(simTime, bodyPositions, viewContext);

      // Mission spacecraft.
      const activeRocket = rockets[0];
      const craft = spacecraftRendererRef.current;

      // MISSION-SCOPED OVERLAYS
      //
      // OrbitalOverlayRenderer and DynamicsOverlayRenderer build their
      // geometry in their constructors — apoapsis/periapsis markers, a
      // velocity arrow, a 100-unit orbital-plane disc, two force arrows — all
      // at the local origin, and they are only repositioned when a mission is
      // being propagated. With no mission loaded they therefore sat at the
      // scene origin, which is the centre of the Sun. Both layers default to
      // on, so toggling either one made an object appear inside the star.
      //
      // Visibility needs BOTH the layer toggle and an active mission. The loop
      // is authoritative; the layers effect below sets the same flags so the
      // state is right before the first frame runs.
      const hasMission = !!activeRocket;
      if (craft) craft.setVisible(hasMission && layersRef.current.missionSpacecraft);
      orbOverlay.setVisible(hasMission && layersRef.current.orbits);
      dynOverlay.setVisible(hasMission && layersRef.current.forceVectors);
      missOverlay.setVisible(hasMission && layersRef.current.trajectories);

      if (activeRocket && craft) {
        // Reference frame comes from the engine's own declaration, not from
        // guessing at the mission's destination string.
        const centralBody = activeRocket.result?.metadata?.central_body;
        const frameOrigin = resolveFrameOrigin(centralBody, bodyPositions);

        if (!frameOrigin && centralBody && !warnedUnknownFrames.has(centralBody)) {
          warnedUnknownFrames.add(centralBody);
          console.warn(
            `[THESEUS] Mission declares central_body "${centralBody}", which is not in the ` +
            `astronomical catalog. The trajectory cannot be placed in the scene and is being ` +
            `drawn heliocentrically; its position should not be trusted.`
          );
        }

        const origin = frameOrigin ?? sunWorldPos;
        craft.setFrameOrigin(origin);
        orbOverlay.group.position.copy(origin);
        dynOverlay.group.position.copy(origin);

        // Per-mission setup: trail geometry and catalogued vehicle size.
        if (activeRocket.id !== lastRocketId) {
          lastRocketId = activeRocket.id;

          const preset = ROCKET_PRESETS.find(r => r.id === activeRocket.presetId);
          craft.setPhysicalRadiusMeters(
            spacecraftCharacteristicRadiusM(preset?.cross_section_area_m2)
          );

          if (activeRocket.result?.state_history) {
            craft.updateTrajectoryHistory(activeRocket.result.state_history);
          }
        }

        const st = getRocketStateAtTime(activeRocket, simTime);
        if (st) {
          // Raw engine state, unmodified. The frame origin above places it.
          craft.updateFrame(st);
          craft.updateVisualScale(viewContext);

          orbOverlay.update(st, activeRocket.result?.state_history);
          dynOverlay.update(st);
        }
      }

      const earthRenderer = planetRenderersRef.current.get('earth');
      if (earthRenderer) {
        lagrange.updateSunEarth(earthRenderer.group.position);
      }
    });

    sm.start();

    return () => {
      planetRenderersRef.current.forEach(r => r.dispose());
      moonRenderersRef.current.forEach(r => r.dispose());
      planetRenderersRef.current.clear();
      moonRenderersRef.current.clear();
      starfield.dispose();
      sun.dispose();
      asteroidBelt.dispose();
      neoRenderer.dispose();
      cometRenderer.dispose();
      kuiperBelt.dispose();
      scatteredDisk.dispose();
      oortCloud.dispose();
      meteorStreams.dispose();
      trojans.dispose();
      hildas.dispose();
      cybeles.dispose();
      centaurs.dispose();
      zodiacalDust.dispose();
      artificialRenderer.dispose();
      spacecraft.dispose();
      sm.dispose();
      cc.dispose();
    };
  }, []);

  // ─── Layer visibility ─────────────────────────────────────────────
  // Every toggle here drives a renderer. Toggles with no renderer behind them
  // are intentionally absent rather than silently inert.
  useEffect(() => {
    planetRenderersRef.current.forEach(r => {
      const isDwarf = r.objectData.type === 'DWARF_PLANET';
      const visible = isDwarf ? layers.dwarfPlanets : layers.planets;
      r.group.visible = visible;
      r.orbitLine.visible = visible && layers.orbits;
    });

    moonRenderersRef.current.forEach(r => {
      r.group.visible = layers.moons;
      r.orbitLine.visible = layers.moons && layers.orbits;
    });

    if (starfieldRef.current) starfieldRef.current.setVisible(true);
    if (asteroidBeltRef.current) asteroidBeltRef.current.setVisible(layers.asteroidBelt);
    if (neoRendererRef.current) {
      // The renderer holds two independently toggleable groups.
      neoRendererRef.current.setVisible(layers.neos || layers.namedAsteroids);
      neoRendererRef.current.setNEOsVisible(layers.neos);
      neoRendererRef.current.setNamedAsteroidsVisible(layers.namedAsteroids);
      neoRendererRef.current.setOrbitsVisible(layers.orbits);
    }
    if (cometRendererRef.current) {
      cometRendererRef.current.setVisible(layers.comets);
      cometRendererRef.current.setOrbitsVisible(layers.orbits);
    }
    if (kuiperBeltRef.current) kuiperBeltRef.current.setVisible(layers.kuiperBelt);
    if (scatteredDiskRef.current) scatteredDiskRef.current.setVisible(layers.scatteredDisk);
    if (oortCloudRef.current) oortCloudRef.current.setVisible(layers.oortCloud);
    if (meteorStreamRef.current) meteorStreamRef.current.setVisible(layers.meteorStreams);
    if (trojanRef.current) {
      trojanRef.current.setVisible(layers.trojansL4 || layers.trojansL5);
      trojanRef.current.setL4Visible(layers.trojansL4);
      trojanRef.current.setL5Visible(layers.trojansL5);
    }
    if (hildaRef.current) hildaRef.current.setVisible(layers.hildas);
    if (cybeleRef.current) cybeleRef.current.setVisible(layers.cybeles);
    if (centaurRef.current) centaurRef.current.setVisible(layers.centaurs);
    if (zodiacalDustRef.current) zodiacalDustRef.current.setVisible(layers.zodiacalDust);
    if (artificialRendererRef.current) {
      artificialRendererRef.current.setVisible(
        layers.satellites || layers.stations || layers.telescopes || layers.probes
      );
    }
    if (spacecraftRendererRef.current) spacecraftRendererRef.current.setVisible(layers.missionSpacecraft);
    // Mission-scoped overlays: these have origin-anchored geometry and are
    // meaningless without a mission. See the render loop for the full note.
    const hasMission = activeRockets.length > 0;
    if (orbitalOverlayRef.current) orbitalOverlayRef.current.setVisible(hasMission && layers.orbits);
    if (dynamicsOverlayRef.current) dynamicsOverlayRef.current.setVisible(hasMission && layers.forceVectors);
    if (missionOverlayRef.current) missionOverlayRef.current.setVisible(hasMission && layers.trajectories);
    if (referenceOverlayRef.current) referenceOverlayRef.current.setVisible(layers.referencePlanes);
    if (lagrangePointRef.current) lagrangePointRef.current.setVisible(layers.lagrangePoints);
    if (soiRendererRef.current) soiRendererRef.current.setVisible(layers.soi);
  }, [layers, activeRockets]);

  // ─── Camera focus ─────────────────────────────────────────────────
  // Framing is derived from the target's own radius; no per-object constants.
  useEffect(() => {
    const cc = cameraControllerRef.current;
    if (!focusedObjectId || !cc) return;

    if (focusedObjectId === 'sun') {
      const sunRadius = sunRendererRef.current?.coronaRadiusScene ?? 100;
      cc.focusOnObject(new THREE.Vector3(0, 0, 0), sunRadius);
      return;
    }

    const pRenderer = planetRenderersRef.current.get(focusedObjectId);
    if (pRenderer) {
      cc.focusOnObject(pRenderer.group.position, pRenderer.visualRadiusScene);
      return;
    }

    const mRenderer = moonRenderersRef.current.get(focusedObjectId);
    if (mRenderer) {
      cc.focusOnObject(
        mRenderer.planetRenderer.group.position,
        mRenderer.planetRenderer.visualRadiusScene
      );
      return;
    }

    const neoPos = neoRendererRef.current?.getPosition(focusedObjectId);
    if (neoPos) { cc.focusOnPosition(neoPos, 0.5); return; }

    const cometPos = cometRendererRef.current?.getPosition(focusedObjectId);
    if (cometPos) { cc.focusOnPosition(cometPos, 1.5); return; }

    if (spacecraftRendererRef.current) {
      cc.focusOnPosition(spacecraftRendererRef.current.group.position, 0.5);
    }
  }, [focusedObjectId]);

  return (
    <div
      ref={containerRef}
      className="w-full h-full cursor-grab active:cursor-grabbing bg-[#000000]"
    />
  );
};
