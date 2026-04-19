"use client";

import { useEffect, useMemo, useRef } from "react";
import type { GlobeInstance } from "globe.gl";
import type { GeoJSONCollection } from "@/lib/api";
import {
  ESRI_WORLD_IMAGERY,
  type TileXYZ,
  tileCoverForView,
  tileKey,
  tileToBounds,
  tileUrl,
} from "@/lib/tiles";

export interface GlobeView {
  lat: number;
  lng: number;
  altitude?: number;
}

export interface GlobeProps {
  detections?: GeoJSONCollection | null;
  forecastPaths?: GeoJSONCollection | null;
  forecastFinal?: GeoJSONCollection | null;
  view?: GlobeView | null;
  /**
   * Normalized playback position in [0, 1]. When set, particles are rendered
   * at that fractional point along their trajectories (live drift animation).
   * When null, the static "final positions" layer is shown instead.
   */
  playbackProgress?: number | null;
  /** Auto-rotate the globe when no scene is selected. */
  autoRotate?: boolean;
}

interface PolygonDatum {
  geometry: { type: "Polygon"; coordinates: number[][][] };
  tooltip: string;
  maxProb: number;
}

interface PointDatum {
  lat: number;
  lng: number;
  color: string;
  radius: number;
  altitude: number;
  tooltip: string;
}

const DEFAULT_VIEW: Required<GlobeView> = { lat: 18, lng: -75, altitude: 2.2 };

// Texture sources, tried in order. The NASA Blue Marble at 5400x2700 is ~5x
// sharper than three-globe's bundled 1024x512 at the cost of a one-time ~3MB
// download. If it fails to load, we fall back to the always-available three.js
// 2048 texture, then to the unpkg one as a last resort.
const EARTH_TEXTURES = [
  "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x5400x2700.jpg",
  "https://cdn.jsdelivr.net/gh/mrdoob/three.js@master/examples/textures/planets/earth_atmos_2048.jpg",
  "//unpkg.com/three-globe/example/img/earth-blue-marble.jpg",
];
const EARTH_BUMP_FALLBACK =
  "//unpkg.com/three-globe/example/img/earth-topology.png";

/** Resolve to the first URL that loads, fall back through the list. */
function pickFirstReachableImage(urls: string[]): Promise<string> {
  return new Promise((resolve) => {
    const tryNext = (i: number) => {
      if (i >= urls.length) {
        resolve(urls[urls.length - 1]);
        return;
      }
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => resolve(urls[i]);
      img.onerror = () => tryNext(i + 1);
      img.src = urls[i];
    };
    tryNext(0);
  });
}

// ---------- winding helpers ----------------------------------------------- //
function ringSignedArea(ring: number[][]): number {
  let sum = 0;
  for (let i = 0, n = ring.length; i < n; i++) {
    const [x1, y1] = ring[i];
    const [x2, y2] = ring[(i + 1) % n];
    sum += (x2 - x1) * (y2 + y1);
  }
  return sum;
}

function ensureClockwise(ring: number[][]): number[][] {
  return ringSignedArea(ring) < 0 ? [...ring].reverse() : ring;
}

function ensureCounterClockwise(ring: number[][]): number[][] {
  return ringSignedArea(ring) > 0 ? [...ring].reverse() : ring;
}

function normalizePolygonRings(coords: number[][][]): number[][][] {
  if (coords.length === 0) return coords;
  const [outer, ...holes] = coords;
  return [ensureClockwise(outer), ...holes.map(ensureCounterClockwise)];
}

// ---------- conversions --------------------------------------------------- //
function detectionsToPolygons(
  fc: GeoJSONCollection | null | undefined,
): PolygonDatum[] {
  if (!fc?.features) return [];
  const out: PolygonDatum[] = [];
  for (const f of fc.features) {
    if (f.geometry?.type !== "Polygon") continue;
    const coords = normalizePolygonRings(f.geometry.coordinates as number[][][]);
    const props = (f.properties ?? {}) as Record<string, unknown>;
    const classNames = (props.class_names as string[]) ?? [];
    const maxProb = typeof props.max_prob === "number" ? props.max_prob : 0;
    const tileId = (props.tile_id as string) ?? "";
    out.push({
      geometry: { type: "Polygon", coordinates: coords },
      maxProb,
      tooltip: `<div style="font:12px/1.4 system-ui;color:#fff;background:rgba(6,26,45,0.95);padding:6px 8px;border-radius:4px;border:1px solid rgba(180,230,255,0.2)">
        <strong>${classNames[0] ?? "Marine Debris"}</strong><br/>
        <span style="color:#94a8ba">${tileId}</span><br/>
        <span style="color:#94a8ba">prob ${(maxProb * 100).toFixed(0)}%</span>
      </div>`,
    });
  }
  return out;
}

function finalsToPoints(fc: GeoJSONCollection | null | undefined): PointDatum[] {
  if (!fc?.features) return [];
  const out: PointDatum[] = [];
  for (const f of fc.features) {
    if (f.geometry?.type !== "Point") continue;
    const [lng, lat] = f.geometry.coordinates as [number, number];
    out.push({
      lat,
      lng,
      color: "rgba(255, 180, 84, 0.85)",
      radius: 0.012,
      altitude: 0.002,
      tooltip: "",
    });
  }
  return out;
}

interface ParticleTrack {
  coords: [number, number][]; // [lng, lat] per timestep
}

function pathsToTracks(
  fc: GeoJSONCollection | null | undefined,
): ParticleTrack[] {
  if (!fc?.features) return [];
  const out: ParticleTrack[] = [];
  for (const f of fc.features) {
    if (f.geometry?.type !== "LineString") continue;
    out.push({ coords: f.geometry.coordinates as [number, number][] });
  }
  return out;
}

/** Sample one (lng, lat) per track at fractional position p in [0, 1]. */
function sampleTracksAt(tracks: ParticleTrack[], p: number): PointDatum[] {
  const t = Math.max(0, Math.min(1, p));
  const out: PointDatum[] = [];
  for (const track of tracks) {
    const n = track.coords.length;
    if (n === 0) continue;
    const idxF = t * (n - 1);
    const i = Math.floor(idxF);
    const frac = idxF - i;
    const a = track.coords[i];
    const b = track.coords[Math.min(i + 1, n - 1)];
    const lng = a[0] + (b[0] - a[0]) * frac;
    const lat = a[1] + (b[1] - a[1]) * frac;
    out.push({
      lat,
      lng,
      color: "rgba(125, 211, 252, 0.95)",
      radius: 0.010,
      altitude: 0.003,
      tooltip: "",
    });
  }
  return out;
}

function pathsToData(tracks: ParticleTrack[]): [number, number][][] {
  // globe.gl pathsData expects [lat, lng] pairs.
  return tracks.map((t) => t.coords.map(([lng, lat]) => [lat, lng]));
}

// ---------- tile layer ---------------------------------------------------- //
interface TileDatum {
  key: string;
  centerLat: number;
  centerLng: number;
  width: number;
  height: number;
  url: string;
}

function tilesToData(tiles: TileXYZ[]): TileDatum[] {
  return tiles.map((t) => {
    const b = tileToBounds(t);
    return {
      key: tileKey(t),
      centerLat: b.centerLat,
      centerLng: b.centerLng,
      width: b.width,
      height: b.height,
      url: tileUrl(ESRI_WORLD_IMAGERY, t),
    };
  });
}

// ---------- component ----------------------------------------------------- //
export default function Globe({
  detections,
  forecastPaths,
  forecastFinal,
  view,
  playbackProgress = null,
  autoRotate = true,
}: GlobeProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const globeRef = useRef<GlobeInstance | null>(null);
  // Texture/material caches survive across tile-cover recomputations so we
  // don't re-download tiles that are still on screen.
  type ThreeTexture = { dispose: () => void };
  type ThreeMaterial = { dispose: () => void; map?: ThreeTexture | null };
  const tileTextureCache = useRef<Map<string, ThreeTexture>>(new Map());
  const tileMaterialCache = useRef<Map<string, ThreeMaterial>>(new Map());
  const threeRef = useRef<typeof import("three") | null>(null);

  const polygons = useMemo(() => detectionsToPolygons(detections), [detections]);
  const tracks = useMemo(() => pathsToTracks(forecastPaths), [forecastPaths]);
  const paths = useMemo(() => pathsToData(tracks), [tracks]);

  // Active particles to render: either samples along tracks (live playback)
  // or the static "final" positions when no playback is running.
  const activePoints = useMemo(() => {
    if (playbackProgress != null && tracks.length > 0) {
      return sampleTracksAt(tracks, playbackProgress);
    }
    return finalsToPoints(forecastFinal);
  }, [playbackProgress, tracks, forecastFinal]);

  useEffect(() => {
    let cancelled = false;
    let handleResize: (() => void) | undefined;

    const init = async () => {
      const [GlobeLib, earthUrl, three] = await Promise.all([
        import("globe.gl").then((m) => m.default),
        pickFirstReachableImage(EARTH_TEXTURES),
        import("three"),
      ]);
      if (!containerRef.current || cancelled) return;
      threeRef.current = three;

      const setSize = (target: GlobeInstance) => {
        if (!containerRef.current) return;
        target
          .width(containerRef.current.clientWidth)
          .height(containerRef.current.clientHeight);
      };

      const loader = new three.TextureLoader();
      loader.crossOrigin = "anonymous";

      // Open-ocean ESRI imagery is genuinely near-black in true color, which
      // looks broken on a dark UI. Multiplying the tile color by ~1.7 lifts
      // shadow detail without blowing out land. (MeshBasicMaterial multiplies
      // map * color in linear space; values >1 are valid.)
      const TILE_BRIGHTNESS = 1.7;
      const getOrLoadMaterial = (d: TileDatum): ThreeMaterial => {
        const cached = tileMaterialCache.current.get(d.key);
        if (cached) return cached;
        // Placeholder material; texture swaps in once the image arrives.
        const mat = new three.MeshBasicMaterial({
          color: new three.Color(TILE_BRIGHTNESS, TILE_BRIGHTNESS, TILE_BRIGHTNESS),
          transparent: true,
          opacity: 0.001,
          depthWrite: false,
        });
        loader.load(
          d.url,
          (tex) => {
            tex.colorSpace = three.SRGBColorSpace;
            mat.map = tex;
            mat.opacity = 1;
            mat.transparent = false;
            mat.needsUpdate = true;
            tileTextureCache.current.set(d.key, tex);
          },
          undefined,
          () => {
            // 404 / network error — leave the placeholder transparent so the
            // base globe texture shows through.
          },
        );
        tileMaterialCache.current.set(d.key, mat);
        return mat;
      };

      const g = new GlobeLib(containerRef.current, {
        animateIn: true,
        rendererConfig: { alpha: true, preserveDrawingBuffer: true, antialias: true },
      })
        .globeImageUrl(earthUrl)
        .bumpImageUrl(EARTH_BUMP_FALLBACK)
        .backgroundColor("rgba(0, 0, 0, 0)")
        .showAtmosphere(true)
        .atmosphereColor("#67e8f9")
        .atmosphereAltitude(0.18)
        // High-res slippy tiles overlay (ESRI World Imagery).
        .tileLat((d: TileDatum) => d.centerLat)
        .tileLng((d: TileDatum) => d.centerLng)
        .tileAltitude(0.0008)
        .tileWidth((d: TileDatum) => d.width)
        .tileHeight((d: TileDatum) => d.height)
        .tileUseGlobeProjection(true)
        .tileMaterial(getOrLoadMaterial)
        .tilesTransitionDuration(0)
        // Detection polygons.
        .polygonGeoJsonGeometry((d: PolygonDatum) => d.geometry)
        .polygonCapColor((d: PolygonDatum) => `rgba(255, 107, 95, ${0.45 + d.maxProb * 0.35})`)
        .polygonSideColor(() => "rgba(255, 107, 95, 0.22)")
        .polygonStrokeColor(() => "#ffb0a3")
        .polygonAltitude(0.0035)
        .polygonLabel((d: PolygonDatum) => d.tooltip)
        // Particle dots (live or final).
        .pointLat("lat")
        .pointLng("lng")
        .pointColor("color")
        .pointRadius("radius")
        .pointAltitude("altitude")
        .pointLabel("tooltip")
        .pointsTransitionDuration(120)
        // Trajectory polylines.
        .pathPointLat((p: [number, number]) => p[0])
        .pathPointLng((p: [number, number]) => p[1])
        .pathPointAlt(0.005)
        .pathStroke(0.4)
        .pathColor(() => "rgba(83, 234, 253, 0.45)")
        .pathTransitionDuration(0);

      globeRef.current = g;
      setSize(g);
      g.pointOfView(DEFAULT_VIEW, 1200);
      g.controls().autoRotate = autoRotate;
      g.controls().autoRotateSpeed = 0.25;

      // Compute and push tile cover from the current camera. Debounced via
      // a small timeout so panning doesn't thrash the request queue.
          // Above this camera altitude (in globe-radius units) we don't draw
          // slippy tiles — the base Blue Marble texture covers the planetary
          // view, and tiles would just produce a low-res ocean mosaic.
          const TILE_MAX_ALTITUDE = 1.4;
          let pendingTimer: ReturnType<typeof setTimeout> | null = null;
          const refreshTiles = () => {
            if (!globeRef.current) return;
            const pov = globeRef.current.pointOfView();
            if (pov.altitude > TILE_MAX_ALTITUDE) {
              // Drop everything we'd cached — keeps GPU memory low when the
              // user pulls back to the overview.
              for (const [k, mat] of tileMaterialCache.current) {
                mat.map?.dispose();
                mat.dispose();
                tileMaterialCache.current.delete(k);
                tileTextureCache.current.delete(k);
              }
              globeRef.current.tilesData([]);
              return;
            }
            const cover = tileCoverForView(pov.lat, pov.lng, pov.altitude);
            const data = tilesToData(cover);
        // Drop cached materials for tiles no longer in view to keep the
        // GPU memory footprint bounded.
        const keep = new Set(data.map((d) => d.key));
        for (const [k, mat] of tileMaterialCache.current) {
          if (keep.has(k)) continue;
          mat.map?.dispose();
          mat.dispose();
          tileMaterialCache.current.delete(k);
          tileTextureCache.current.delete(k);
        }
        globeRef.current.tilesData(data);
      };

      const scheduleRefresh = () => {
        if (pendingTimer) clearTimeout(pendingTimer);
        pendingTimer = setTimeout(refreshTiles, 180);
      };

      g.controls().addEventListener("change", scheduleRefresh);
      refreshTiles();

      handleResize = () => {
        setSize(g);
        scheduleRefresh();
      };
      window.addEventListener("resize", handleResize);

      cleanupExtra = () => {
        if (pendingTimer) clearTimeout(pendingTimer);
        g.controls().removeEventListener("change", scheduleRefresh);
      };
    };

    let cleanupExtra: (() => void) | null = null;
    init();

    return () => {
      cancelled = true;
      cleanupExtra?.();
      if (handleResize) window.removeEventListener("resize", handleResize);
      // Free GPU resources for cached tile textures.
      for (const mat of tileMaterialCache.current.values()) {
        mat.map?.dispose();
        mat.dispose();
      }
      tileMaterialCache.current.clear();
      tileTextureCache.current.clear();
      globeRef.current?._destructor();
      globeRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const g = globeRef.current;
    if (!g) return;
    g.polygonsData(polygons);
  }, [polygons]);

  useEffect(() => {
    const g = globeRef.current;
    if (!g) return;
    g.pointsData(activePoints);
  }, [activePoints]);

  useEffect(() => {
    const g = globeRef.current;
    if (!g) return;
    g.pathsData(paths);
  }, [paths]);

  useEffect(() => {
    const g = globeRef.current;
    if (!g || !view) return;
    g.controls().autoRotate = false;
    g.pointOfView(
      { lat: view.lat, lng: view.lng, altitude: view.altitude ?? 0.05 },
      1400,
    );
  }, [view]);

  useEffect(() => {
    const g = globeRef.current;
    if (!g) return;
    g.controls().autoRotate = autoRotate && !view;
  }, [autoRotate, view]);

  return (
    <div className="globe-canvas" ref={containerRef}>
      <div className="globe-shade" />
    </div>
  );
}
