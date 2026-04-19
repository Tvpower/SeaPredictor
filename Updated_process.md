The full pipeline, start to finish
You've built a two-stage system. Think of it like this:

┌─────────────────────┐        ┌─────────────────────┐
│  STAGE 1: DETECTOR  │        │  STAGE 2: FORECAST  │
│  "Is debris here    │   ───► │  "Where will it go  │
│   NOW?"             │        │   over 7 days?"     │
│                     │        │                     │
│  CNN model          │        │  OpenDrift (physics │
│  (MARIDA-trained)   │        │   simulator)        │
└─────────────────────┘        └─────────────────────┘
         ↑                              ↑
  Sentinel-2 imagery            OSCAR ocean currents
  (11-band satellite)           (NOAA daily fields)
These are two completely different things:

Stage 1 is machine learning — you trained it on labeled images
Stage 2 is physics — it doesn't need training; it integrates a vector field
The only thing connecting them is a JSON file. Let me walk through it.

Stage 1: What your trained model does
Role: Look at a satellite image, decide what's in it.

Input: A 256×256 pixel tile of 11-band Sentinel-2 imagery (blue, green, red, near-IR, SWIR, etc.)

What the model is:

A ResNet-18 CNN that eats the 11-band image
Outputs 15 logits (one per MARIDA class: Marine Debris, Sargassum, Ship, Cloud, Water, etc.)
You apply sigmoid → each output is a probability 0..1 for that class
Trained on: 694 tiles from MARIDA where marine biologists had manually labeled what's in each tile.

How it's used at inference: You take a real Sentinel-2 scene, cut it into tiles, run each through the model, threshold the probabilities (using your thresholds.json) → get a list like:

{
  "tile_id": "S2_18-9-20_16PCC_0",
  "probs": [0.87, 0.02, ..., 0.91, ...],   // 15 values
  "preds": [1, 0, ..., 1, ...],             // class 0 (Marine Debris) = YES
  "predicted_classes": [0, 6],
  "geo": {
    "crs": "EPSG:16N",
    "bounds": [minx, miny, maxx, maxy]     // where on Earth this tile is
  }
}
So Stage 1's job is purely: "given pixels, say which of these tiles contain debris and where those tiles are located on Earth."

Key insight: Stage 1 does no prediction of the future. It's a pure "what does this image show right now" classifier. The model doesn't even know what a current is.

Stage 2: What OpenDrift does
Role: Given a list of locations where debris currently is, simulate where it will drift to over the next N days.

Input:

Seed points — list of (lat, lon, date) where debris exists today (this comes from Stage 1's output)
Ocean current field — OSCAR daily NetCDFs (just numbers: at this (lat, lon) on this day, the water is flowing (u, v) m/s)
What OpenDrift is:

A physics simulator, not an ML model
No learned parameters
Just numerical integration of dx/dt = u(x, y, t), dy/dt = v(x, y, t)
Adds random kicks for eddy diffusion (sub-grid turbulence)
Stops particles when they hit coastlines
The physics in plain English: It takes a tiny imaginary "debris particle," asks OSCAR "what's the water doing at this exact lat/lon on this date?", moves the particle a tiny step in that direction, advances the clock by 30 minutes, asks OSCAR again. Repeat for 7 days. Then do that 200 times per seed (Monte Carlo) so you get an uncertainty spread instead of a single line.

Key insight: OpenDrift doesn't care about images, pixels, CNNs, or debris at all. It just moves points around in a velocity field. You could seed it with duck decoys or oil droplets — it'd run the same way.

The handoff — how Stage 1 feeds Stage 2
This is where your src/forecast/seed.py does the magic. It reads Stage 1's output JSON and does three things:

1. Filter to debris-only tiles
# Keep only tiles where the model predicted class 0 (Marine Debris) = 1
if preds[0] != 1:
    skip
2. Convert pixel-space tile bounds → real-world lat/lon
Stage 1 outputs bounds in UTM (the satellite's native projection). OpenDrift wants WGS84 lat/lon. pyproj reprojects the centroid:

lon, lat = Transformer("EPSG:16N", "EPSG:4326").transform(cx, cy)
3. Look up when the image was taken
OpenDrift needs a start date for the simulation — it needs to know which day's OSCAR currents to use. The script either:

Looks up the date from MARIDA's tile_index.csv, or
Uses the --default-date CLI arg
Output of this handoff: a list of Seed objects:

Seed(tile_id="S2_18-9-20_16PCC_0", lat=16.09, lon=-88.32, obs_date=2020-09-18)
That's all OpenDrift needs. The image, the neural network, the tile pixels — all of that gets discarded. Only (lat, lon, date) survives into Stage 2.

How OSCAR fits in (important distinction)
OSCAR data is used in two completely different places in your system. People conflate these:

Use	When	Where
Context for the detector (unused)
Training
The LSTM branch of DebrisPredictor was supposed to use 30 days of OSCAR history as temporal context. You found this didn't help (coverage too sparse), so you're running --cnn-only. OSCAR is NOT used by the detector currently.
Velocity field for the simulator
Forecasting
This is what Stage 2 uses — the actual u/v current fields to move particles. Completely separate use.
In your current MVP: OSCAR only matters for Stage 2. The detector doesn't touch it.

End-to-end trace — what actually happened when you ran the Honduras demo
Step by step, in the order commands executed:

predict.py on S2_18-9-20_16PCC:
Load best.pt (the ResNet-18 you trained).
Load 79 tiles from data/data/raw/MARIDA/patches/S2_18-9-20_16PCC/.
For each tile:
Read 11 bands with rasterio.
Normalize using MARIDA's per-band mean/std.
Push through model → 15 sigmoid probabilities.
Apply per-class threshold from thresholds.json.
Record preds + geo bounds.
Write predictions/honduras_sep18.json with 79 records.
drift.py:
Read predictions/honduras_sep18.json.
For each of the 79 tiles, ask: "is preds[0] == 1?" → 35 tiles said yes.
Reproject those 35 tile centroids from EPSG:16N → EPSG:4326 → 35 (lat, lon) pairs.
Look up each tile's date from tile_index.csv → all say 2020-09-18.
Build/read OSCAR concat NetCDF covering 2020-09-17 to 2020-09-26.
Create OpenDrift OceanDrift object, attach the OSCAR reader.
Seed 200 particles around each of the 35 (lat, lon) locations → 7000 particles total.
For 7 days × 48 steps/day = 336 iterations:
Ask OSCAR what (u, v) is at each particle's current location
Move each particle by (u*dt, v*dt) + small random perturbation
If a particle crosses into land → freeze it
Save trajectories to forecast/honduras_sep18.nc.
Export 7000 particle paths as LineStrings → GeoJSON.
Export final-position points → GeoJSON.
validate.py:
Load the 7000-particle trajectory.
Tier 1: compute mean displacement, beaching, diffusion from just the trajectory.
Tier 2: load honduras_sep23.json (detector run on a scene 5 days later), find its 19 debris centroids, ask "how many of my 7000 forecast particles at Sep 23 are within 5 km of any of those 19 points?" → 72%.
The mental model I want you to internalize
Stage 1 = "see" Stage 2 = "extrapolate"

Your neural network is a seeing-machine. It looks at pixels and says "debris." It doesn't understand currents, wind, or time.

OpenDrift is a time-machine for particles. Given a starting point and a velocity field, it projects forward. It doesn't understand what debris is.

The whole trick of your system is: use the ML model to tell you where to start the physics simulation, then let physics do the time-travel.

That's why the combination is stronger than either alone:

Without the model: you need humans to find debris in satellite imagery (doesn't scale)
Without OpenDrift: you know where debris is but not where it's going
Together: autonomous detection + physics-grounded forecasting
Data flow summary
Sentinel-2 .tif tiles ──► [CNN detector] ──► predictions.json
                                                │
                                                ▼
                                     [filter: class 0 = 1]
                                                │
                                                ▼
                                  35 tile centroids (lat, lon, date)
                                                │
             OSCAR NetCDFs  ────────────────────┼──► [OpenDrift OceanDrift]
             (u, v m/s)                         │           │
                                                │           ▼
                                                │      7000 particles
                                                │      propagated 7 days
                                                │           │
                                                │           ▼
                                              trajectory.nc + GeoJSON
                                                            │
             predictions_sep23.json ─────────────────────┐  │
             (detector run 5 days later)                 ▼  ▼
                                               [validate.py comparison]
                                                            │
                                                            ▼
                                              72% hit-rate at 5 km
That's the current MVP. The CNN detects, OpenDrift forecasts, the validator proves it works.