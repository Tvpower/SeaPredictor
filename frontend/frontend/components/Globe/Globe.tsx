"use client";

import { useEffect, useRef } from "react";
import type { GlobeInstance } from "globe.gl";

const detectionSeeds = [
  { lat: 16.09, lng: -88.32, radius: 0.38, color: "#ff6b5f", label: "35 CNN debris seeds" },
  { lat: 16.18, lng: -88.58, radius: 0.24, color: "#ff8a72", label: "Sargassum / foam candidates" },
  { lat: 15.92, lng: -88.05, radius: 0.2, color: "#ffb454", label: "Final particle cluster" },
];

const forecastArcs = [
  {
    startLat: 16.09,
    startLng: -88.32,
    endLat: 16.55,
    endLng: -86.92,
    color: ["rgba(103, 232, 249, 0.25)", "rgba(34, 211, 238, 0.95)"],
  },
  {
    startLat: 16.18,
    startLng: -88.58,
    endLat: 16.86,
    endLng: -87.28,
    color: ["rgba(125, 211, 252, 0.25)", "rgba(56, 189, 248, 0.9)"],
  },
  {
    startLat: 15.92,
    startLng: -88.05,
    endLat: 15.74,
    endLng: -86.72,
    color: ["rgba(165, 243, 252, 0.18)", "rgba(14, 165, 233, 0.85)"],
  },
  {
    startLat: 16.09,
    startLng: -88.32,
    endLat: 15.62,
    endLng: -87.08,
    color: ["rgba(103, 232, 249, 0.16)", "rgba(45, 212, 191, 0.72)"],
  },
];

export default function Globe() {
  const globeRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | undefined;
    let globe: GlobeInstance | undefined;
    let handleResize: (() => void) | undefined;
    let cancelled = false;

    const init = async () => {
      const GlobeLib = (await import("globe.gl")).default;

      if (!globeRef.current || cancelled) return;

      const setSize = (target: GlobeInstance) => {
        if (!globeRef.current) return;
        target
          .width(globeRef.current.clientWidth)
          .height(globeRef.current.clientHeight);
      };

      const globeInstance = new GlobeLib(globeRef.current, {
        animateIn: true,
        rendererConfig: { alpha: true, preserveDrawingBuffer: true },
      })
        .globeImageUrl("//unpkg.com/three-globe/example/img/earth-blue-marble.jpg")
        .bumpImageUrl("//unpkg.com/three-globe/example/img/earth-topology.png")
        .backgroundColor("rgba(0, 0, 0, 0)")
        .showAtmosphere(true)
        .atmosphereColor("#67e8f9")
        .atmosphereAltitude(0.18)
        .pointsData(detectionSeeds)
        .pointLat("lat")
        .pointLng("lng")
        .pointColor("color")
        .pointRadius("radius")
        .pointAltitude(0.025)
        .pointsTransitionDuration(900)
        .ringsData(detectionSeeds)
        .ringLat("lat")
        .ringLng("lng")
        .ringColor(() => (t: number) => `rgba(103, 232, 249, ${1 - t})`)
        .ringMaxRadius(4)
        .ringPropagationSpeed(1.2)
        .ringRepeatPeriod(2200)
        .arcsData(forecastArcs)
        .arcStartLat("startLat")
        .arcStartLng("startLng")
        .arcEndLat("endLat")
        .arcEndLng("endLng")
        .arcColor("color")
        .arcStroke(0.65)
        .arcAltitude(0.18)
        .arcDashLength(0.42)
        .arcDashGap(0.18)
        .arcDashAnimateTime(2800)
        .labelsData(detectionSeeds)
        .labelLat("lat")
        .labelLng("lng")
        .labelText("label")
        .labelColor(() => "#dffbff")
        .labelSize(0.9)
        .labelAltitude(0.045)
        .labelDotRadius(0.22);

      globe = globeInstance;
      setSize(globeInstance);
      globeInstance.pointOfView({ lat: 17, lng: -86, altitude: 1.7 }, 1200);
      globeInstance.controls().autoRotate = true;
      globeInstance.controls().autoRotateSpeed = 0.28;

      interval = setInterval(() => {
        const pov = globeInstance.pointOfView();
        globeInstance.pointOfView({
          lat: pov.lat,
          lng: pov.lng + 0.035,
          altitude: pov.altitude,
        });
      }, 80);

      handleResize = () => setSize(globeInstance);
      window.addEventListener("resize", handleResize);
    };

    init();

    return () => {
      cancelled = true;
      if (interval) clearInterval(interval);
      if (handleResize) window.removeEventListener("resize", handleResize);
      globe?._destructor();
    };
  }, []);

  return (
    <div className="globe-canvas" ref={globeRef}>
      <div className="globe-shade" />
    </div>
  );
}
