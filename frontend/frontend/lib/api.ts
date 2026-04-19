/**
 * Typed client for the SeaPredictor FastAPI backend (`src/api/server.py`).
 *
 * In dev (`npm run dev` on :3000), point this at the FastAPI server on :8000
 * via `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`.
 *
 * In prod (Next.js static export served by FastAPI itself), leave the env
 * var unset and requests go to the same origin.
 */

const RAW_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";
export const API_BASE = RAW_BASE.replace(/\/+$/, "");

export interface SceneIndexEntry {
  scene_id: string;
  obs_date: string;
  has_oscar_coverage: boolean;
  n_tiles: number;
  n_detections: number;
  n_cloud_suppressed: number;
  per_class_detections: Record<string, number>;
  centroid_lat: number;
  centroid_lon: number;
  bbox_wgs84: [number, number, number, number]; // [west, south, east, north]
  debris_classes_considered: number[];
}

export interface SceneIndex {
  generated_from_ckpt: string;
  class_names: string[];
  default_debris_classes: number[];
  oscar_coverage: { start: string; end: string; n_dates: number };
  n_scenes: number;
  scenes: SceneIndexEntry[];
}

export type SceneMeta = SceneIndexEntry;

export interface ForecastRequest {
  scene_id: string;
  days?: number;
  n_per_seed?: number;
  seed_radius_m?: number;
  horizontal_diffusivity?: number;
  debris_classes?: number[];
  min_prob?: number;
  timestep_minutes?: number;
  wind_speed_ms?: number;
  wind_dir_deg?: number;
  wind_drift_factor?: number;
}

export interface ForecastStats {
  cache_key: string;
  cached: boolean;
  n_particles: number;
  n_features_paths: number;
  n_features_final: number;
  elapsed_s: number;
  has_czml?: boolean;
  time_start?: string | null;
  time_end?: string | null;
}

export interface ForecastResponse {
  cache_key: string;
  cached: boolean;
  stats: ForecastStats;
  params: Required<ForecastRequest>;
  paths_url: string;
  final_url: string;
  czml_url?: string | null;
}

// GeoJSON we only consume — keep typing loose.
export type GeoJSONCollection = {
  type: "FeatureCollection";
  features: Array<{
    type: "Feature";
    geometry: { type: string; coordinates: unknown };
    properties?: Record<string, unknown>;
  }>;
};

function url(path: string): string {
  if (!path.startsWith("/")) path = "/" + path;
  return `${API_BASE}${path}`;
}

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let body = "";
    try {
      body = await res.text();
    } catch {
      // ignore
    }
    throw new Error(`API ${res.status} ${res.statusText}: ${body || res.url}`);
  }
  return (await res.json()) as T;
}

export async function getHealth(): Promise<{ ok: boolean }> {
  return jsonOrThrow(await fetch(url("/api/health")));
}

export async function listScenes(): Promise<SceneIndex> {
  return jsonOrThrow(await fetch(url("/api/scenes")));
}

export async function getScene(sceneId: string): Promise<SceneMeta> {
  return jsonOrThrow(await fetch(url(`/api/scenes/${sceneId}`)));
}

export async function getSceneDetections(sceneId: string): Promise<GeoJSONCollection> {
  return jsonOrThrow(await fetch(url(`/api/scenes/${sceneId}/detections`)));
}

export async function runForecast(req: ForecastRequest): Promise<ForecastResponse> {
  return jsonOrThrow(
    await fetch(url("/api/forecast"), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req),
    }),
  );
}

export async function getForecastPaths(cacheKey: string): Promise<GeoJSONCollection> {
  return jsonOrThrow(await fetch(url(`/api/forecast/${cacheKey}/paths`)));
}

export async function getForecastFinal(cacheKey: string): Promise<GeoJSONCollection> {
  return jsonOrThrow(await fetch(url(`/api/forecast/${cacheKey}/final`)));
}

/** Absolute URL helpers — pass to viewers (Cesium, etc.) that fetch on their own. */
export const absUrl = (path: string): string => url(path);
