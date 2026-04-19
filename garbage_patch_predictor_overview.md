# OceanWatch: AI-Powered Garbage Patch Predictor

## Project Overview

OceanWatch is a machine learning system designed to predict the location and movement of large marine debris accumulation zones — commonly known as garbage patches — across the world's oceans. The project combines satellite remote sensing data, ocean physics simulations, and a retrieval-augmented AI agent to produce time-forward predictions of where debris will concentrate, with the goal of supporting cleanup prioritization and environmental monitoring at a global scale.

The system is built for the FullyHacks hackathon and integrates with the [Human Delta API](https://humandelta.ai/) — a knowledge infrastructure layer for AI agents — to ground its predictions in peer-reviewed and institutional scientific sources rather than relying solely on model inference.

---

## The Problem

There are five major ocean garbage patches in the world, formed where rotating ocean current systems called gyres trap floating debris. The largest, the Great Pacific Garbage Patch, spans an estimated 1.6 million square kilometers between Hawaii and California. These patches are not static — they shift seasonally, grow with new debris input, and split or merge based on current dynamics. No existing tool provides a continuously updated, ML-driven prediction of where patches will be in 7, 14, or 30 days. Current monitoring relies on expensive oceanographic surveys, sparse citizen-reported sightings, and coarse gyre zone polygon estimates that are updated infrequently.

The consequence: cleanup organizations like The Ocean Cleanup must deploy vessels based on historical averages rather than real-time predictive intelligence. Better location forecasting directly translates to fewer vessel miles wasted and more debris intercepted per expedition.

---

## How It Works

### Data Sources

The model fuses two fundamentally different types of input data — what debris looks like from space, and how the ocean moves — to produce predictions that are both spatially precise and physically grounded.

**Satellite Imagery (Spatial Input)**

The primary imagery source is the European Space Agency's Sentinel-2 constellation, a pair of free-access satellites that photograph the entire Earth every 2–5 days at 10–20 meter resolution across 13 spectral bands. The system uses three specific bands — B4 (Red, 665 nm), B8 (Near-Infrared, 842 nm), and B11 (Short-Wave Infrared, 1610 nm) — to compute the Floating Debris Index (FDI):

\[ \text{FDI} = B8 - \left( B4 + (B11 - B4) \times \frac{\lambda_{NIR} - \lambda_{RED}}{\lambda_{SWIR} - \lambda_{RED}} \times 10 \right) \]

Clean seawater absorbs nearly all NIR and SWIR radiation, appearing dark in those bands. Floating debris — plastic, driftwood, foam — reflects anomalously high NIR and SWIR energy. The FDI isolates this anomaly as a per-pixel score, which is then thresholded to produce a binary debris presence mask. This mask serves as the ground truth label for model training.

Rather than generating these labels from scratch, the project uses two pre-annotated benchmark datasets:

- **MARIDA (Marine Debris Archive)** — 1,381 Sentinel-2 tiles with pixel-level annotations across 11 debris classes, spanning 2015–2021. This is the primary training dataset.
- **MADOS (Marine Debris and Oil Spill)** — 174 scenes across 47 tile regions with approximately 1.5 million annotated pixels and 15 classes, spanning 2015–2022. Used to augment MARIDA and increase geographic diversity.

A third dataset, **NASA-IMPACT Marine Debris** (1,370 bounding-box polygons from PlanetScope satellite imagery), is held out as a cross-sensor validation set. Because PlanetScope and Sentinel-2 have different sensor characteristics, strong performance on NASA-IMPACT validates that the model has learned actual debris features rather than instrument-specific artifacts.

**Ocean Physics (Temporal Input)**

Satellite imagery tells the model where debris is now. Ocean current data tells it where debris will go. Two NOAA datasets provide this temporal context:

- **NOAA OSCAR (Ocean Surface Current Analysis Real-time)** — surface current velocity vectors (U and V components) updated every 5 days, downloaded as NetCDF files. This is the primary temporal input and encodes the physical drift dynamics of the ocean surface.
- **NOAA HYCOM** — sea surface temperature (SST) and salinity fields. Debris tends to accumulate in warmer, calmer convergence zones; these features help the model distinguish transient debris from stable accumulation patches.

For each satellite tile in the training set, a 30-day window of OSCAR current sequences is extracted at the matching geographic coordinates and date. This pairs every spatial observation with its corresponding ocean physics context.

---

### Model Architecture

The model is a hybrid neural network with two parallel branches that fuse spatial and temporal features before producing a prediction.

```
Input A: Sentinel-2 tile (256×256, bands B4/B8/B11)
         → ResNet-18 CNN Encoder → spatial feature vector (512-dim)
                                                                    ↘
                                                              Concatenate (640-dim)
                                                                    → Linear layers
                                                                    → Debris probability
                                                                    ↗
Input B: OSCAR + HYCOM sequence (30 days × 6 features)
         → 2-layer LSTM Encoder → temporal feature vector (128-dim)
```

**CNN Branch (Spatial):** A pretrained ResNet-18 serves as the spatial encoder, with its input convolution modified to accept 3-band Sentinel-2 imagery instead of standard RGB. The final fully-connected layer is removed, leaving a 512-dimensional feature vector that encodes the spectral and textural signature of the tile. Pretraining on ImageNet provides learned edge and texture detectors that transfer effectively to satellite imagery feature extraction.

**LSTM Branch (Temporal):** A two-layer Long Short-Term Memory network processes the 30-day ocean physics sequence. Each timestep encodes 6 features: U current, V current, sea surface temperature, salinity, latitude, and longitude. The LSTM's final hidden state produces a 128-dimensional vector summarizing the drift trajectory leading up to the satellite observation date.

**Fusion Head:** The 512-dimensional spatial vector and 128-dimensional temporal vector are concatenated into a 640-dimensional representation, then passed through two linear layers with ReLU activation and dropout regularization, producing a final debris probability score.

**Training Details:**
- Loss function: `BCEWithLogitsLoss` with positive class weighting (ratio ≈ 1:10, debris to ocean pixels)
- Optimizer: AdamW with weight decay 1e-4
- Learning rate: 1e-4 with cosine annealing scheduler
- Batch size: 16, 30 training epochs
- Train/validation/test split: 70/15/15 by scene (not by tile, to prevent spatial leakage)
- Training data window: 2018–2020 (MARIDA + MADOS)
- Validation: 2021 scenes
- Generalization test: 2022–2023 fresh Sentinel-2 pulls

---

### Human Delta Integration

At inference time, the model produces a probability heatmap over the ocean grid. High-confidence zones (probability > threshold) are extracted as geographic polygons with associated confidence scores. The Human Delta API then enriches each prediction with grounded scientific context:

```
Model output: { lat: 32.1, lon: -145.3, confidence: 0.87 }
         ↓
Human Delta /v1/search:
  query = "marine debris accumulation 32.1 -145.3 ocean gyre"
         ↓
Returns: ranked passages from indexed NOAA, Ocean Cleanup,
         and Marine Debris Tracker corpus
         ↓
Agent classifies zone as:
  "monitored"       — cleanup activity documented nearby
  "priority_alert"  — high confidence, no existing cleanup coverage
  "novel_discovery" — no prior data exists for this zone
```

The Human Delta corpus is built by crawling NOAA Marine Debris, The Ocean Cleanup, and debristracker.org via `POST /v1/indexes`, and uploading dataset metadata and model documentation via `POST /v1/documents`. This transforms the tool from a black-box classifier into an agent that explains *why* a zone is flagged, citing specific institutional sources for each prediction.

---

### Inference & Deployment

After training in Python/PyTorch, the model is exported as a TorchScript file (`.pt`) for production deployment via a C++ LibTorch inference server. This design separates the training environment from the production runtime:

```python
# Python training → export
torch.jit.script(model).save("inference/debris_predictor.pt")
```

```cpp
// C++ production inference
torch::jit::script::Module model = torch::jit::load("debris_predictor.pt");
auto output = model.forward({image_tensor, current_tensor}).toTensor();
```

The C++ inference server receives Sentinel-2 tile and OSCAR current data, runs the model, and returns the enriched JSON prediction to the frontend. A Python FastAPI wrapper exposes the C++ binary as an HTTP endpoint for the Human Delta agent to call.

---

## Repository Structure

```
garbage-patch-predictor/
├── data/                    # Download scripts for MARIDA, MADOS, OSCAR, HYCOM
├── src/
│   ├── dataset/             # PyTorch Dataset class, per-source loaders
│   ├── models/              # CNN encoder, LSTM encoder, fusion model
│   ├── training/            # Training loop, evaluation, config
│   └── utils/               # FDI computation, geo utilities, export
├── notebooks/               # Exploratory notebooks for each data source
├── inference/               # C++ LibTorch inference server
└── agent/                   # Human Delta API integration
```

---

## Build Phases

| Phase | Deliverable | Key Files |
|-------|------------|-----------|
| 1 — Data Pipeline | Working `DebrisDataset` PyTorch class | `src/dataset/debris_dataset.py` |
| 2 — CNN Baseline | ResNet-18 trained on MARIDA (>75% accuracy) | `src/models/cnn_encoder.py` |
| 3 — Hybrid Model | CNN + LSTM fusion with OSCAR temporal input | `src/models/fusion_model.py` |
| 4 — Export | TorchScript `.pt` file for C++ server | `src/utils/export.py` |
| 5 — Agent | Human Delta enrichment at inference | `agent/enrich_prediction.py` |
| 6 — Frontend | Leaflet.js map with heatmap + citation sidebar | `frontend/` |

---

## References & Data Sources

- **MARIDA Dataset** — https://zenodo.org/record/5151941
- **MADOS Dataset** — https://zenodo.org/records/10664073
- **NASA-IMPACT Marine Debris ML** — https://github.com/NASA-IMPACT/marine_debris_ML
- **NOAA OSCAR Ocean Currents** — https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_third-deg
- **Sentinel-2 Copernicus Browser** — https://browser.dataspace.copernicus.eu
- **Human Delta API** — https://dev.humandelta.ai
- **Marine Debris Tracker** — https://debristracker.org
- **The Ocean Cleanup** — https://theoceancleanup.com/great-pacific-garbage-patch
