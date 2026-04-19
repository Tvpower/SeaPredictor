# DESIGN.md — Frontend Build Spec for Claude Code

You are building the frontend for **SeaPredictor** (branded as **OceanLens AI** in the UI),
a FullyHacks hackathon project that predicts ocean garbage patch locations from satellite
imagery + ocean current data. The model backend is being built in parallel — **build the
entire frontend against mock data** so it works standalone, then we swap in live API calls
later.

---

## 1. Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Framework | **React 18** (Vite) | Fast setup, team knows React |
| 3D Globe | **CesiumJS** via `resium` | Photorealistic Earth, native GeoJSON, Sentinel-2 overlay support |
| Charts | **Recharts** | Lightweight, React-native, good bar charts |
| Styling | **Tailwind CSS** | Utility-first, matches dark theme fast |
| Icons | **Lucide React** | Clean, consistent icon set |
| Routing | **React Router v6** | Sidebar nav between views |

### Install

```bash
npm create vite@latest frontend -- --template react
cd frontend
npm install resium cesium @cesium/engine recharts react-router-dom lucide-react tailwindcss @tailwindcss/vite
```

CesiumJS needs its static assets copied. Add to `vite.config.js`:

```js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  define: {
    CESIUM_BASE_URL: JSON.stringify('/cesium'),
  },
});
```

Copy Cesium's `Build/Cesium/Workers`, `Assets`, `Widgets`, and `ThirdParty` directories
into `public/cesium/`. See the `resium` docs for the exact setup.

Set the Cesium Ion access token in a `.env` file:
```
VITE_CESIUM_ION_TOKEN=<get a free token from https://ion.cesium.com>
```

---

## 2. Design Language

This is NOT a generic dashboard. The aesthetic is **deep-ocean command center** — dark,
immersive, data-dense, like a military situational awareness display crossed with a
nature documentary. Reference the two mockup images in this repo for exact visual targets.

### Color Palette (use as CSS variables / Tailwind theme extension)

```
--bg-primary:      #0a0f1a     (near-black with blue undertone — main background)
--bg-secondary:    #111827     (sidebar, card backgrounds)
--bg-card:         #0d1525     (glassmorphic panels, semi-transparent)
--bg-card-glass:   rgba(13, 21, 37, 0.85)  (backdrop-blur panels over the globe)
--border-subtle:   rgba(45, 212, 191, 0.15) (teal glow borders on cards)

--text-primary:    #e2e8f0     (main body text — light gray)
--text-secondary:  #94a3b8     (labels, metadata, timestamps)
--text-muted:      #64748b     (footer, inactive nav)

--accent-teal:     #2dd4bf     (primary accent — active nav, buttons, highlights)
--accent-teal-dim: rgba(45, 212, 191, 0.2) (teal background fills)
--accent-orange:   #f59e0b     (warning markers, "ACTION PRIORITY: HIGH" badge)
--accent-red:      #ef4444     (critical alerts)
--accent-green:    #10b981     (positive status, "Current" pill)

--chart-bar-low:   rgba(45, 212, 191, 0.3)  (spectral chart low bars)
--chart-bar-high:  rgba(45, 212, 191, 0.8)  (spectral chart high bars)
```

### Typography

```
--font-display:  'Orbitron', sans-serif     (section headers, zone IDs like "NP-742", brand name)
--font-heading:  'Rajdhani', sans-serif     (sub-headers, labels like "CURRENT ANALYSIS")
--font-body:     'IBM Plex Sans', sans-serif (body text, descriptions, insights)
--font-mono:     'IBM Plex Mono', monospace  (coordinates, data values, FDI scores)
```

Load from Google Fonts:
```
Orbitron:wght@700;900
Rajdhani:wght@500;600;700
IBM+Plex+Sans:wght@300;400;500;600
IBM+Plex+Mono:wght@400;500
```

### Key Visual Effects

- **Glassmorphism on cards**: `backdrop-filter: blur(16px)`, semi-transparent backgrounds
  with subtle teal-tinted borders (`1px solid var(--border-subtle)`)
- **Glow on accent elements**: `box-shadow: 0 0 20px rgba(45, 212, 191, 0.3)` on active
  states, teal buttons, alert badges
- **Uppercase letter-spacing on labels**: All section headers like "CURRENT ANALYSIS",
  "SPECTRAL SIGNATURE", "DRIFT FORECAST", "RECENT DISCOVERIES" use
  `text-transform: uppercase; letter-spacing: 0.15em; font-family: var(--font-heading)`
- **Pulsing markers on globe**: Debris markers gently pulse with a CSS animation
  (scale 1→1.3→1, opacity cycle) to indicate live monitoring

---

## 3. Layout Structure

```
┌─────────────────────────────────────────────────────────┐
│  Top Bar: Brand + Search + Notifications + Avatar       │
├────────┬────────────────────────────────────────────────┤
│        │                                                │
│  Side  │              Main Content Area                 │
│  bar   │         (Globe fills this entirely)            │
│        │                                                │
│  Nav   │    ┌──────────┐           ┌───────────────┐   │
│  items │    │ Analysis │           │ Alert Badge   │   │
│        │    │ Card     │           └───────────────┘   │
│        │    └──────────┘                                │
│        │                                                │
│  + New │    ┌──────────────┐                            │
│  Anlys │    │ Recent       │                            │
│        │    │ Discoveries  │                            │
│        │    └──────────────┘                            │
│        │                                                │
│ Support│    ┌─ Current ─ +7 ─ +14 ─ +30 ─┐            │
│Settings│    └─────── Time Slider ─────────┘            │
├────────┴────────────────────────────────────────────────┤
│  Footer: © OceanLens AI · System Status · API Docs     │
└─────────────────────────────────────────────────────────┘
```

### Detail Panel (slides in from right when a marker is clicked)

```
┌─────────────────────────────┐
│  TARGET ANALYSIS    [Badge] │
│  NP-742                     │
│─────────────────────────────│
│  🔶 HUMAN DELTA INSIGHTS   │
│  Description text...        │
│  ┌─ Blockquote ──────────┐ │
│  │  Citation from source │ │
│  └───────────────────────┘ │
│─────────────────────────────│
│  📊 SPECTRAL SIGNATURE     │
│       0.84 FDI SCORE       │
│  ┌─ Bar Chart ───────────┐ │
│  │  B4  B8  B11  FDI ... │ │
│  └───────────────────────┘ │
│─────────────────────────────│
│  ➤ DRIFT FORECAST          │
│  ┌─ Current Vector Map ──┐ │
│  │  (canvas or SVG)      │ │
│  │  24H OFFSET: 12.4 KM  │ │
│  └───────────────────────┘ │
│─────────────────────────────│
│  [Export Coords] [View Rpt] │
└─────────────────────────────┘
```

---

## 4. Component Breakdown

### `App.jsx`
- Top-level layout: sidebar + main area + footer
- React Router for Dashboard / Global Map / AI Insights / Reports
- Global Map is the default route

### `components/Sidebar.jsx`
- Fixed left, 200px wide, bg `--bg-secondary`
- Brand: "OceanLens AI" in Orbitron, "DEEP SEA MONITORING" subtitle in Rajdhani
- Nav items: Dashboard, Global Map, AI Insights, Reports — icons from Lucide
- Active state: teal left border + teal text + subtle teal bg fill
- Bottom: "+ New Analysis" button (rounded, teal bg), Support, Settings links

### `components/TopBar.jsx`
- Search input with magnifying glass icon, placeholder "Search coordinates..."
- Right side: notification bell icon, settings gear icon, avatar circle
- The search bar accepts "lat, lon" format and flies the globe camera to that location

### `components/GlobeView.jsx` (the main event)
- CesiumJS Viewer via `resium` — full width/height of the main content area
- **Imagery**: Use `IonImageryProvider` with Cesium's default Blue Marble or configure
  NASA Black Marble night imagery for the dark aesthetic
- **Atmosphere**: Enable `scene.globe.enableLighting = true` and atmosphere rendering
- **Debris Markers**: Render from mock data as `Entity` points with custom billboards.
  Teal color = monitored zones. Orange = anomalous / priority alerts.
  Each marker pulses via CSS animation on its billboard.
- **Heatmap Overlay**: For the detail view's concentric colored rings (red-orange-yellow-
  teal gradient), render as a `CustomDataSource` with concentric `EllipseGraphics` entities
  at different radii and decreasing opacity, OR as a single canvas-textured `RectangleGraphics`.
- **Camera**: Default position looking at the Pacific Ocean (lon: -150, lat: 20, height: 15000km)
  to feature the Great Pacific Garbage Patch.
- **Click handler**: Clicking a debris marker opens the `AnalysisDetailPanel`.
- Disable Cesium's default UI (timeline, animation widget, home button) — we have our own.

### `components/AnalysisCard.jsx`
- Floating glassmorphic card, top-left over the globe
- Header: "CURRENT ANALYSIS" with chart icon
- Content: patch name ("GREAT PACIFIC PATCH"), predicted growth stat ("+4.2%"),
  "predicted growth" label in teal, confidence progress bar, accuracy percentage
- Uses `--bg-card-glass` with `backdrop-blur`

### `components/AlertBadge.jsx`
- Floating card, top-right over the globe
- Orange warning icon + "Anomalous Concentration" title
- Subtitle: sector code + coordinates
- Orange left border or accent

### `components/RecentDiscoveries.jsx`
- Floating card, bottom-left over the globe (above footer)
- Header: "RECENT DISCOVERIES (SENTINEL-2)" in uppercase Rajdhani
- List of 2-3 discovery cards, each with:
  - Thumbnail image (placeholder satellite image, 64x64)
  - Title ("Microplastic Bloom", "Ghost Net Cluster")
  - Timestamp + location ("2h ago · North Atlantic")
- Subtle separator between items

### `components/TimeSlider.jsx`
- Centered at the bottom of the globe area
- Four pill buttons: "Current", "+7 Days", "+14 Days", "+30 Days"
- "Current" is active by default (teal bg, white text)
- Others are inactive (transparent bg, gray text, hover → subtle teal)
- Selecting a forecast day swaps the mock debris marker positions to simulate
  drift prediction (just shift coordinates slightly per time step)

### `components/AnalysisDetailPanel.jsx`
- **This is the second mockup** — a right-side slide-in panel, ~380px wide
- Triggered by clicking a debris marker on the globe
- Animates in from right (transform translateX, 300ms ease-out)
- Sections, top to bottom:

  **Header**: "TARGET ANALYSIS" label + zone ID in large Orbitron font ("NP-742")
  + "ACTION PRIORITY: HIGH" badge (orange bg, small rounded pill)

  **Human Delta Insights**: Orange diamond icon + section header. Body text describing
  the accumulation (IBM Plex Sans). Blockquote card with italic citation text and
  source attribution ("— Ocean Cleanup Data Cluster"). Teal-colored hyperlinks for
  ocean current names.

  **Spectral Signature**: Section header + "0.84 FDI SCORE" in large teal mono text,
  right-aligned. Bar chart (Recharts `BarChart`) showing reflectance values for bands
  B4, B8, B11, and derived indices. Use teal color scale — lower bars lighter, taller
  bars darker teal.

  **Drift Forecast**: Section header. Placeholder visualization — either a static SVG
  with wavy current lines, or a simple canvas animation with particle flow. Label at
  bottom: "24H PROJECTED OFFSET: 12.4 KM" in mono font.

  **Actions**: Two buttons at the bottom — "Export Coordinates" (outline style) and
  "View Full Report" (filled teal).

- Close button (X) in the top-right corner of the panel.

### `components/Footer.jsx`
- Full-width bottom bar
- Left: "© 2025 OceanLens AI. Powered by Human Delta."
- Right: "SYSTEM STATUS" · "API DOCUMENTATION" · "PRIVACY" links

### `components/MapControls.jsx`
- Zoom in/out buttons (+/-) and a layer toggle button (stacked diamond icon)
- Positioned bottom-left, just above the footer, to the right of the sidebar
- Styled as small glassmorphic square buttons

---

## 5. Mock Data

Create a `src/data/mockData.js` file with hardcoded data to drive the entire UI:

```js
export const debrisZones = [
  {
    id: "NP-742",
    name: "Great Pacific Patch — Sector 7G-Alpha",
    lat: 32.5,
    lon: -145.3,
    confidence: 0.87,
    fdiScore: 0.84,
    priority: "high",        // "high" | "medium" | "low"
    predictedGrowth: 4.2,    // percent
    classification: "priority_alert",
    spectralSignature: {
      B4: 0.12, B8: 0.45, B11: 0.38,
      NDVI: 0.08, FDI: 0.84, FAI: 0.31, NDWI: -0.22
    },
    driftForecast: {
      offset24h: 12.4,   // km
      bearing: 215,       // degrees
      projectedPositions: [
        { day: 7,  lat: 32.1, lon: -145.8 },
        { day: 14, lat: 31.6, lon: -146.5 },
        { day: 30, lat: 30.8, lon: -147.9 },
      ]
    },
    humanDeltaInsight: {
      summary: "Massive accumulation detected at the convergence of the Kuroshio Extension and the North Pacific Current. NOAA spectral analysis suggests a high concentration of high-density polyethylene.",
      citation: "Current flow patterns indicate a 88% probability of permanent entrapment within the sub-tropical gyre if intervention is not initiated within 14 days.",
      source: "Ocean Cleanup Data Cluster"
    }
  },
  {
    id: "NA-318",
    name: "North Atlantic Microplastic Bloom",
    lat: 28.3,
    lon: -42.7,
    confidence: 0.72,
    fdiScore: 0.61,
    priority: "medium",
    predictedGrowth: 1.8,
    classification: "monitored",
    spectralSignature: {
      B4: 0.09, B8: 0.31, B11: 0.28,
      NDVI: 0.05, FDI: 0.61, FAI: 0.22, NDWI: -0.15
    },
    driftForecast: {
      offset24h: 8.7,
      bearing: 185,
      projectedPositions: [
        { day: 7,  lat: 28.0, lon: -43.1 },
        { day: 14, lat: 27.5, lon: -43.6 },
        { day: 30, lat: 26.8, lon: -44.5 },
      ]
    },
    humanDeltaInsight: {
      summary: "Dispersed microplastic field detected in the Sargasso Sea convergence zone. Concentration patterns consistent with windrow formation from recent subtropical storm activity.",
      citation: "Satellite-derived surface roughness anomalies correlate with microplastic density at r=0.73 in this region.",
      source: "NOAA Marine Debris Program"
    }
  },
  {
    id: "TS-156",
    name: "Ghost Net Cluster — Tasman Sea",
    lat: -35.2,
    lon: 158.9,
    confidence: 0.65,
    fdiScore: 0.52,
    priority: "low",
    predictedGrowth: 0.4,
    classification: "monitored",
    spectralSignature: {
      B4: 0.07, B8: 0.25, B11: 0.21,
      NDVI: 0.03, FDI: 0.52, FAI: 0.18, NDWI: -0.11
    },
    driftForecast: {
      offset24h: 5.2,
      bearing: 240,
      projectedPositions: [
        { day: 7,  lat: -35.5, lon: 158.4 },
        { day: 14, lat: -35.9, lon: 157.8 },
        { day: 30, lat: -36.5, lon: 156.9 },
      ]
    },
    humanDeltaInsight: {
      summary: "Abandoned fishing net cluster identified via high NIR reflectance signature. Pattern consistent with derelict longline gear from commercial tuna operations.",
      citation: "Ghost gear drift models project landfall on northern New Zealand coast within 45 days at current velocity.",
      source: "Marine Debris Tracker"
    }
  }
];

export const recentDiscoveries = [
  {
    title: "Microplastic Bloom",
    time: "2h ago",
    location: "North Atlantic",
    zoneId: "NA-318"
  },
  {
    title: "Ghost Net Cluster",
    time: "5h ago",
    location: "Tasman Sea",
    zoneId: "TS-156"
  },
  {
    title: "HDPE Fragment Field",
    time: "12h ago",
    location: "South Pacific",
    zoneId: "SP-891"
  }
];
```

---

## 6. Globe Configuration Details

### Cesium Viewer Settings

```jsx
<Viewer
  full
  timeline={false}
  animation={false}
  homeButton={false}
  baseLayerPicker={false}
  navigationHelpButton={false}
  sceneModePicker={false}
  geocoder={false}
  selectionIndicator={false}
  infoBox={false}
  scene3DOnly={true}
>
```

### Camera Default

```js
viewer.camera.flyTo({
  destination: Cesium.Cartesian3.fromDegrees(-150, 20, 15000000),
  duration: 0
});
```

### Marker Rendering

For each debris zone, create an `Entity` with:
- `position`: `Cesium.Cartesian3.fromDegrees(lon, lat)`
- `point`: `{ pixelSize: 12, color: Cesium.Color.fromCssColorString('#2dd4bf') }` for
  monitored zones, orange `#f59e0b` for priority alerts
- On click: fly camera to `(lon, lat, 2000000)` altitude and open `AnalysisDetailPanel`

### Heatmap on Marker Focus

When a marker is clicked and the camera zooms in, overlay concentric ellipses at the
marker's position to simulate the heatmap visualization from the mockup:

```
Ring 1 (innermost): radius 20km, color red (#ef4444), opacity 0.6
Ring 2: radius 50km, color orange (#f59e0b), opacity 0.4
Ring 3: radius 100km, color yellow (#eab308), opacity 0.25
Ring 4 (outermost): radius 180km, color teal (#2dd4bf), opacity 0.1
```

---

## 7. Interaction Flow

1. **Page loads** → Globe renders with 3 debris markers visible. AnalysisCard shows
   the "GREAT PACIFIC PATCH" summary. AlertBadge shows the highest-priority zone.
   RecentDiscoveries shows the feed. TimeSlider defaults to "Current".

2. **User clicks a marker** → Camera flies to that zone (2M meter altitude, 1.5s
   duration). Heatmap rings appear around the marker. AnalysisDetailPanel slides in
   from the right with that zone's data.

3. **User switches time slider** → Marker positions shift to the `projectedPositions`
   for that day offset. AnalysisCard updates growth stat. This simulates the model's
   drift prediction.

4. **User clicks "Export Coordinates"** → Downloads a JSON file with the zone's
   coordinates and metadata.

5. **User clicks "X" on detail panel** → Panel slides out, camera zooms back to
   the global view, heatmap rings disappear.

6. **User types coordinates in search bar** → Camera flies to those coordinates.

---

## 8. File Structure

```
frontend/
├── public/
│   └── cesium/              # CesiumJS static assets (Workers, Assets, etc.)
├── src/
│   ├── App.jsx
│   ├── main.jsx
│   ├── index.css            # Tailwind directives + CSS variables + fonts
│   ├── data/
│   │   └── mockData.js      # All mock debris zones, discoveries
│   ├── components/
│   │   ├── Sidebar.jsx
│   │   ├── TopBar.jsx
│   │   ├── GlobeView.jsx    # CesiumJS globe + marker rendering
│   │   ├── AnalysisCard.jsx
│   │   ├── AlertBadge.jsx
│   │   ├── RecentDiscoveries.jsx
│   │   ├── TimeSlider.jsx
│   │   ├── AnalysisDetailPanel.jsx
│   │   ├── SpectralChart.jsx      # Recharts bar chart for spectral bands
│   │   ├── DriftForecastViz.jsx   # Canvas/SVG current flow visualization
│   │   ├── MapControls.jsx
│   │   └── Footer.jsx
│   └── hooks/
│       └── useGlobe.js      # Globe camera controls, marker click handlers
├── tailwind.config.js
├── vite.config.js
├── package.json
└── .env                     # VITE_CESIUM_ION_TOKEN
```

---

## 9. Constraints & Notes

- **This is a hackathon** — ship fast, polish later. Get the globe + sidebar + detail
  panel working first. Dashboard, AI Insights, and Reports pages can be placeholder
  "coming soon" screens.
- **No backend calls yet.** Everything reads from `mockData.js`. When the model API is
  ready, we'll add a `src/api/predictions.js` module that fetches from FastAPI and
  replaces the mock imports.
- **CesiumJS is heavy.** Lazy-load the Viewer component. Don't import Cesium at the
  top level of App.jsx.
- **Responsive is not a priority** for the hackathon demo — target 1920×1080 screens.
  The demo will be projected or screen-shared.
- **The globe IS the background.** There is no separate background color behind the
  globe. The sidebar, cards, and panels float over it with glassmorphic transparency.
- Do NOT use `localStorage` or `sessionStorage`. All state lives in React `useState`/
  `useReducer`.
- The mockup brand name is "OceanLens AI" with subtitle "DEEP SEA MONITORING", and
  the top bar says "ABYSSAL INTEL". Use "OceanLens AI" as the sidebar brand and
  "ABYSSAL INTEL" can be dropped or used as the top-bar brand — your call, just be
  consistent. The footer reads "Powered by Human Delta."