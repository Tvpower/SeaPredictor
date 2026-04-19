"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Globe from "@/components/Globe/Globe";
import Sidebar from "@/components/Layout/Sidebar";
import Topbar from "@/components/Layout/Topbar";
import Card from "@/components/Layout/Card";
import ControlPanel, { type ForecastParams } from "@/components/Layout/ControlPanel";
import {
  type ForecastResponse,
  type GeoJSONCollection,
  type SceneIndexEntry,
  getForecastFinal,
  getForecastPaths,
  getSceneDetections,
  listScenes,
  runForecast,
} from "@/lib/api";

function bboxToAltitude(bbox?: [number, number, number, number]): number {
  if (!bbox) return 0.6;
  const [w, s, e, n] = bbox;
  const span = Math.max(Math.abs(e - w), Math.abs(n - s));
  // Empirical: small bbox (<0.5deg) -> tight zoom; larger -> pull back.
  return Math.min(1.4, Math.max(0.18, span * 1.2));
}

export default function Page() {
  const [scenes, setScenes] = useState<SceneIndexEntry[]>([]);
  const [scenesLoading, setScenesLoading] = useState(true);
  const [scenesError, setScenesError] = useState<string | null>(null);

  const [selectedSceneId, setSelectedSceneId] = useState<string | null>(null);
  const [detections, setDetections] = useState<GeoJSONCollection | null>(null);
  const [detectionsError, setDetectionsError] = useState<string | null>(null);

  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [forecastPaths, setForecastPaths] = useState<GeoJSONCollection | null>(null);
  const [forecastFinal, setForecastFinal] = useState<GeoJSONCollection | null>(null);
  const [forecastRunning, setForecastRunning] = useState(false);
  const [forecastStatus, setForecastStatus] = useState<string | null>(null);
  const [forecastError, setForecastError] = useState<string | null>(null);
  const [animate, setAnimate] = useState(true);

  // Load scene index on mount.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const idx = await listScenes();
        if (cancelled) return;
        setScenes(idx.scenes ?? []);
        if ((idx.scenes ?? []).length > 0 && !selectedSceneId) {
          setSelectedSceneId(idx.scenes[0].scene_id);
        }
      } catch (e) {
        if (!cancelled) setScenesError((e as Error).message);
      } finally {
        if (!cancelled) setScenesLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // When the selection changes, fetch detections and clear any prior forecast.
  useEffect(() => {
    if (!selectedSceneId) return;
    let cancelled = false;
    setDetections(null);
    setDetectionsError(null);
    setForecast(null);
    setForecastPaths(null);
    setForecastFinal(null);
    setForecastStatus(null);
    setForecastError(null);
    (async () => {
      try {
        const det = await getSceneDetections(selectedSceneId);
        if (!cancelled) setDetections(det);
      } catch (e) {
        if (!cancelled) setDetectionsError((e as Error).message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedSceneId]);

  const selectedScene = useMemo(
    () => scenes.find((s) => s.scene_id === selectedSceneId) ?? null,
    [scenes, selectedSceneId],
  );

  const view = useMemo(() => {
    if (!selectedScene) return null;
    return {
      lat: selectedScene.centroid_lat,
      lng: selectedScene.centroid_lon,
      altitude: bboxToAltitude(selectedScene.bbox_wgs84),
    };
  }, [selectedScene]);

  const handleRunForecast = useCallback(
    async (params: ForecastParams) => {
      if (!selectedSceneId) return;
      setForecastRunning(true);
      setForecastError(null);
      setForecastStatus("Submitting forecast…");
      try {
        const resp = await runForecast({
          scene_id: selectedSceneId,
          days: params.days,
          n_per_seed: params.n_per_seed,
          wind_speed_ms: params.wind_speed_ms,
          wind_dir_deg: params.wind_dir_deg,
          wind_drift_factor: params.wind_drift_factor,
        });
        setForecast(resp);
        setForecastStatus(
          resp.cached
            ? `Cached run (${resp.stats.n_particles} particles).`
            : `Computed in ${resp.stats.elapsed_s}s (${resp.stats.n_particles} particles).`,
        );
        const [paths, final] = await Promise.all([
          getForecastPaths(resp.cache_key),
          getForecastFinal(resp.cache_key),
        ]);
        setForecastPaths(paths);
        setForecastFinal(final);
      } catch (e) {
        setForecastError((e as Error).message);
        setForecastStatus(null);
      } finally {
        setForecastRunning(false);
      }
    },
    [selectedSceneId],
  );

  return (
    <main className="app-shell">
      <Sidebar
        scenes={scenes}
        selectedSceneId={selectedSceneId}
        onSelectScene={setSelectedSceneId}
        loading={scenesLoading}
        error={scenesError}
      />

      <section className="globe-stage">
        <Topbar scene={selectedScene} />

        <div className="globe-layer">
          <Globe
            detections={detections}
            forecastPaths={forecastPaths}
            forecastFinal={forecastFinal}
            view={view}
            animatePaths={animate}
            autoRotate={!selectedScene}
          />
        </div>

        <div className="stage-vignette" />

        <div className="scene-card-slot">
          <Card scene={selectedScene} forecast={forecast} />
          {detectionsError && (
            <p style={{ color: "#ff806f", fontSize: 12, marginTop: 8 }}>
              Failed to load detections: {detectionsError}
            </p>
          )}
        </div>

        <div className="pipeline-strip">
          {[
            ["Stage 1", "Detector", `${selectedScene?.n_detections ?? "—"} debris tiles`],
            ["Stage 2", "Forecast", forecast ? `${forecast.stats.n_particles} particles` : "OpenDrift + OSCAR"],
            ["Stage 3", "Explore", "FastAPI serves cached GeoJSON layers"],
          ].map(([stage, title, body]) => (
            <article className="pipeline-card" key={stage}>
              <p className="mini-label">{stage}</p>
              <h3>{title}</h3>
              <p>{body}</p>
            </article>
          ))}
        </div>

        <aside className="right-stack">
          <ControlPanel
            disabled={!selectedSceneId}
            running={forecastRunning}
            animate={animate}
            onAnimateChange={setAnimate}
            onRun={handleRunForecast}
            status={forecastStatus}
            error={forecastError}
          />

          <section className="legend-panel">
            <h3>Layer legend</h3>
            <div className="legend-list">
              <div>
                <span className="legend-dot seed" />
                Detector seed tiles
              </div>
              <div>
                <span className="legend-line" />
                OpenDrift particle paths
              </div>
              <div>
                <span className="legend-dot final" />
                Final particle positions
              </div>
            </div>
          </section>
        </aside>
      </section>
    </main>
  );
}
