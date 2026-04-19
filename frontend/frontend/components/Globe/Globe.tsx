"use client";

import { useEffect, useMemo, useRef } from "react";
import type { GlobeInstance } from "globe.gl";
import type { GeoJSONCollection } from "@/lib/api";

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
  /** When true, particle paths flow with an animated dash. */
  animatePaths?: boolean;
  /** Auto-rotate the globe when no scene is selected. */
  autoRotate?: boolean;
}

interface PointDatum {
  lat: number;
  lng: number;
  color: string;
  radius: number;
  altitude: number;
  tooltip: string;
  kind: "detection" | "final";
}

const DEFAULT_VIEW: Required<GlobeView> = { lat: 18, lng: -75, altitude: 2.2 };

function polygonCenter(coords: number[][][]): { lat: number; lng: number } | null {
  const ring = coords[0];
  if (!ring || ring.length === 0) return null;
  let sx = 0;
  let sy = 0;
  for (const [lng, lat] of ring) {
    sx += lng;
    sy += lat;
  }
  return { lat: sy / ring.length, lng: sx / ring.length };
}

/**
 * Snap nearby detection centroids to the same bucket so we don't render 12
 * stacked points + 12 stacked tooltips when MARIDA tiles are adjacent.
 */
function dedupeDetections(
  fc: GeoJSONCollection | null | undefined,
): PointDatum[] {
  if (!fc?.features) return [];

  // ~250 m at the equator. Adjacent 256x256 S2 tiles end up ~2.5 km apart, so
  // this snaps individual sub-tile clusters together but keeps neighboring
  // tiles distinct.
  const SNAP = 0.005;
  const buckets = new Map<
    string,
    { lat: number; lng: number; count: number; classes: Set<string>; maxProb: number }
  >();

  for (const f of fc.features) {
    if (f.geometry?.type !== "Polygon") continue;
    const c = polygonCenter(f.geometry.coordinates as number[][][]);
    if (!c) continue;
    const props = (f.properties ?? {}) as Record<string, unknown>;
    const classNames = (props.class_names as string[]) ?? [];
    const maxProb = typeof props.max_prob === "number" ? props.max_prob : 0;

    const bx = Math.round(c.lng / SNAP);
    const by = Math.round(c.lat / SNAP);
    const key = `${bx}|${by}`;
    const existing = buckets.get(key);
    if (existing) {
      existing.count += 1;
      existing.maxProb = Math.max(existing.maxProb, maxProb);
      classNames.forEach((cl) => existing.classes.add(cl));
    } else {
      buckets.set(key, {
        lat: c.lat,
        lng: c.lng,
        count: 1,
        classes: new Set(classNames),
        maxProb,
      });
    }
  }

  return Array.from(buckets.values()).map((b) => ({
    lat: b.lat,
    lng: b.lng,
    color: "#ff6b5f",
    radius: 0.18 + Math.min(0.35, b.maxProb * 0.4),
    altitude: 0.012,
    tooltip: `<div style="font:12px/1.4 system-ui;color:#fff;background:rgba(6,26,45,0.95);padding:6px 8px;border-radius:4px;border:1px solid rgba(180,230,255,0.2)">
      <strong>${b.count} detection${b.count > 1 ? "s" : ""}</strong><br/>
      ${Array.from(b.classes).join(", ") || "Marine debris"}<br/>
      <span style="color:#94a8ba">max prob ${(b.maxProb * 100).toFixed(0)}%</span>
    </div>`,
    kind: "detection",
  }));
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
      color: "#ffb454",
      radius: 0.06,
      altitude: 0.005,
      tooltip: "",
      kind: "final",
    });
  }
  return out;
}

function pathsToData(
  fc: GeoJSONCollection | null | undefined,
): [number, number][][] {
  if (!fc?.features) return [];
  const out: [number, number][][] = [];
  for (const f of fc.features) {
    if (f.geometry?.type !== "LineString") continue;
    const coords = f.geometry.coordinates as [number, number][];
    out.push(coords.map(([lng, lat]) => [lat, lng]));
  }
  return out;
}

export default function Globe({
  detections,
  forecastPaths,
  forecastFinal,
  view,
  animatePaths = true,
  autoRotate = true,
}: GlobeProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const globeRef = useRef<GlobeInstance | null>(null);

  const detectionPoints = useMemo(() => dedupeDetections(detections), [detections]);
  const finalPoints = useMemo(() => finalsToPoints(forecastFinal), [forecastFinal]);
  const allPoints = useMemo(
    () => [...detectionPoints, ...finalPoints],
    [detectionPoints, finalPoints],
  );
  const paths = useMemo(() => pathsToData(forecastPaths), [forecastPaths]);

  useEffect(() => {
    let cancelled = false;
    let handleResize: (() => void) | undefined;

    const init = async () => {
      const GlobeLib = (await import("globe.gl")).default;
      if (!containerRef.current || cancelled) return;

      const setSize = (target: GlobeInstance) => {
        if (!containerRef.current) return;
        target
          .width(containerRef.current.clientWidth)
          .height(containerRef.current.clientHeight);
      };

      const g = new GlobeLib(containerRef.current, {
        animateIn: true,
        rendererConfig: { alpha: true, preserveDrawingBuffer: true },
      })
        .globeImageUrl("//unpkg.com/three-globe/example/img/earth-blue-marble.jpg")
        .bumpImageUrl("//unpkg.com/three-globe/example/img/earth-topology.png")
        .backgroundColor("rgba(0, 0, 0, 0)")
        .showAtmosphere(true)
        .atmosphereColor("#67e8f9")
        .atmosphereAltitude(0.18)
        .pointLat("lat")
        .pointLng("lng")
        .pointColor("color")
        .pointRadius("radius")
        .pointAltitude("altitude")
        .pointLabel("tooltip")
        .pointsTransitionDuration(500)
        .pathPointLat((p: [number, number]) => p[0])
        .pathPointLng((p: [number, number]) => p[1])
        .pathPointAlt(0.008)
        .pathStroke(0.45)
        .pathColor(() => "rgba(83, 234, 253, 0.55)")
        .pathDashLength(0.35)
        .pathDashGap(0.18)
        .pathDashInitialGap(() => Math.random())
        .pathTransitionDuration(0);

      globeRef.current = g;
      setSize(g);
      g.pointOfView(DEFAULT_VIEW, 1200);
      g.controls().autoRotate = autoRotate;
      g.controls().autoRotateSpeed = 0.25;

      handleResize = () => setSize(g);
      window.addEventListener("resize", handleResize);
    };

    init();

    return () => {
      cancelled = true;
      if (handleResize) window.removeEventListener("resize", handleResize);
      globeRef.current?._destructor();
      globeRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Push data updates without re-creating the globe.
  useEffect(() => {
    const g = globeRef.current;
    if (!g) return;
    g.pointsData(allPoints);
  }, [allPoints]);

  useEffect(() => {
    const g = globeRef.current;
    if (!g) return;
    g.pathsData(paths);
  }, [paths]);

  useEffect(() => {
    const g = globeRef.current;
    if (!g) return;
    // 0 = static; >0 = milliseconds for one full dash cycle.
    g.pathDashAnimateTime(animatePaths ? 4500 : 0);
  }, [animatePaths]);

  useEffect(() => {
    const g = globeRef.current;
    if (!g || !view) return;
    g.controls().autoRotate = false;
    g.pointOfView(
      { lat: view.lat, lng: view.lng, altitude: view.altitude ?? 0.6 },
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
