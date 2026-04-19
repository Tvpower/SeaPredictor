/**
 * Slippy-map tile helpers. Pure math — no DOM / Three.js deps.
 *
 * Coordinate system: standard XYZ tiles (Web Mercator, 256x256), like OSM
 * and ESRI World Imagery use:
 *   https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}
 *
 * IMPORTANT: ArcGIS uses /z/y/x order (y first). Don't swap them.
 */

export interface TileXYZ {
  z: number;
  x: number;
  y: number;
}

export interface TileBounds {
  /** West, south, east, north in degrees. */
  bbox: [number, number, number, number];
  centerLat: number;
  centerLng: number;
  /** Width and height in degrees of lat/lng. */
  width: number;
  height: number;
}

const MAX_LAT = 85.0511287798; // Web Mercator pole clamp.

export function clampLat(lat: number): number {
  return Math.max(-MAX_LAT, Math.min(MAX_LAT, lat));
}

export function lngToTileX(lng: number, z: number): number {
  return Math.floor(((lng + 180) / 360) * Math.pow(2, z));
}

export function latToTileY(lat: number, z: number): number {
  const rad = (clampLat(lat) * Math.PI) / 180;
  return Math.floor(
    ((1 - Math.log(Math.tan(rad) + 1 / Math.cos(rad)) / Math.PI) / 2) *
      Math.pow(2, z),
  );
}

export function tileXToLng(x: number, z: number): number {
  return (x / Math.pow(2, z)) * 360 - 180;
}

export function tileYToLat(y: number, z: number): number {
  const n = Math.PI - (2 * Math.PI * y) / Math.pow(2, z);
  return (180 / Math.PI) * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n)));
}

export function tileToBounds({ x, y, z }: TileXYZ): TileBounds {
  const w = tileXToLng(x, z);
  const e = tileXToLng(x + 1, z);
  const n = tileYToLat(y, z);
  const s = tileYToLat(y + 1, z);
  return {
    bbox: [w, s, e, n],
    centerLat: (n + s) / 2,
    centerLng: (w + e) / 2,
    width: e - w,
    height: n - s,
  };
}

export function tileKey(t: TileXYZ): string {
  return `${t.z}/${t.x}/${t.y}`;
}

/**
 * Pick a slippy zoom level appropriate for a globe.gl camera at altitude
 * `altGlobeRadii` (in units of Earth radii, like globe.gl's pointOfView).
 *
 * Heuristic: target ~3 tile widths visible across the viewport. globe.gl's
 * default FOV is ~60deg, so the visible angular half-width on the surface is
 * ~ tan(30deg) * alt (small-altitude approximation).
 */
export function pickZoomForAltitude(altGlobeRadii: number): number {
  const halfWidthDeg = Math.tan(Math.PI / 6) * altGlobeRadii * (180 / Math.PI);
  const visibleWidthDeg = Math.max(0.001, halfWidthDeg * 2);
  // 360deg / (2^z) is the width of one z-tile at the equator. We want
  // visibleWidthDeg ≈ 3 * (360 / 2^z)  →  z ≈ log2(3 * 360 / visibleWidthDeg).
  const z = Math.round(Math.log2((3 * 360) / visibleWidthDeg));
  return Math.max(2, Math.min(17, z));
}

/**
 * Enumerate every tile that intersects the camera's visible cap. Caller
 * should pass `lat`, `lng`, and `altGlobeRadii` from globe.gl's `pointOfView`.
 *
 * Adds a 1-tile buffer ring so panning by a small amount doesn't flash white
 * gaps before the new cover is computed.
 */
export function tileCoverForView(
  lat: number,
  lng: number,
  altGlobeRadii: number,
  opts?: { maxTiles?: number },
): TileXYZ[] {
  const z = pickZoomForAltitude(altGlobeRadii);
  const maxTiles = opts?.maxTiles ?? 36;

  // Visible angular half-width (approx). At small alt this is tan(fov/2)*alt;
  // at full-globe alt it saturates near 90deg, which is fine — we'll just
  // generate a lot of tiles (capped by maxTiles).
  const halfWidthDeg = Math.min(
    85,
    Math.tan(Math.PI / 6) * altGlobeRadii * (180 / Math.PI),
  );

  const south = clampLat(lat - halfWidthDeg);
  const north = clampLat(lat + halfWidthDeg);
  // Longitude span widens at high latitudes to cover the same arc length.
  const lngSpread = halfWidthDeg / Math.max(0.1, Math.cos((lat * Math.PI) / 180));
  const west = lng - lngSpread;
  const east = lng + lngSpread;

  // Slippy y is north-down; tile bounds use [yMin..yMax] inclusive.
  const xMin = lngToTileX(west, z) - 1;
  const xMax = lngToTileX(east, z) + 1;
  const yMin = latToTileY(north, z) - 1;
  const yMax = latToTileY(south, z) + 1;

  const max = Math.pow(2, z);
  const tiles: TileXYZ[] = [];
  for (let y = Math.max(0, yMin); y <= Math.min(max - 1, yMax); y++) {
    for (let xRaw = xMin; xRaw <= xMax; xRaw++) {
      // Wrap longitude so we still draw correctly across the antimeridian.
      const x = ((xRaw % max) + max) % max;
      tiles.push({ z, x, y });
    }
  }
  if (tiles.length <= maxTiles) return tiles;
  // Too many — center-bias and trim.
  const cx = lngToTileX(lng, z);
  const cy = latToTileY(lat, z);
  return tiles
    .map((t) => ({ t, d: Math.hypot(t.x - cx, t.y - cy) }))
    .sort((a, b) => a.d - b.d)
    .slice(0, maxTiles)
    .map(({ t }) => t);
}

/** Default ESRI World Imagery template (note y/x order). */
export const ESRI_WORLD_IMAGERY =
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";

export function tileUrl(template: string, t: TileXYZ): string {
  return template
    .replace("{z}", String(t.z))
    .replace("{y}", String(t.y))
    .replace("{x}", String(t.x));
}
