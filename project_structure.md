***

## Repo Structure

```
SeaPredictor/
│
├── README.md
├── requirements.txt
├── .env.example                  # HD_API_KEY, data paths
│
├── data/
│   ├── download_marida.sh        # pulls MARIDA from Zenodo
│   ├── download_mados.sh         # pulls MADOS from Zenodo
│   ├── download_oscar.py         # pulls NOAA OSCAR NetCDF by bbox/year
│   ├── download_hycom.py         # pulls NOAA HYCOM SST/salinity
│   └── raw/                      # gitignored, local data lives here
│       ├── marida/
│       ├── mados/
│       ├── oscar/
│       └── hycom/
│
├── src/
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── marida_loader.py      # reads MARIDA GeoTIFFs + masks
│   │   ├── mados_loader.py       # reads MADOS tiles + masks
│   │   ├── oscar_loader.py       # reads OSCAR NetCDF → 30-day sequences
│   │   ├── hycom_loader.py       # reads HYCOM NetCDF → SST/salinity
│   │   └── debris_dataset.py     # master Dataset class, merges all sources
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_encoder.py        # ResNet-18 spatial encoder
│   │   ├── lstm_encoder.py       # 2-layer LSTM temporal encoder
│   │   └── fusion_model.py       # combines both → heatmap output
│   │
│   ├── training/
│   │   ├── train.py              # main training loop
│   │   ├── evaluate.py           # validation + metrics
│   │   └── config.py             # hyperparams, paths, device setup
│   │
│   └── utils/
│       ├── fdi.py                # FDI computation from B4/B8/B11
│       ├── geo_utils.py          # lat/lon helpers, bbox cropping
│       └── export.py             # TorchScript export for C++ inference
│
├── notebooks/
│   ├── 01_explore_marida.ipynb   # sanity check MARIDA tiles
│   ├── 02_explore_oscar.ipynb    # visualize current vectors
│   └── 03_baseline_cnn.ipynb     # train CNN-only baseline interactively
│
├── inference/                    # C++ LibTorch server (later)
│   ├── CMakeLists.txt
│   └── inference_server.cpp
│
└── agent/                        # Human Delta integration (later)
    ├── index_sources.py          # one-time corpus indexing
    ├── enrich_prediction.py      # /v1/search at inference time
    └── output_schema.py          # JSON schema for frontend
```

***

## Phase 1 — Data Pipeline (Start Here)

### Step 1: Download Scripts

**`data/download_marida.sh`**
```bash
#!/bin/bash
# MARIDA from Zenodo — record 5151941
mkdir -p data/raw/marida
wget -O data/raw/marida/marida.zip \
  "https://zenodo.org/record/5151941/files/MARIDA.zip"
unzip data/raw/marida/marida.zip -d data/raw/marida/
```

**`data/download_oscar.py`**
```python
# NOAA OSCAR — download 2018-2020 for North Pacific gyre region
# bbox: lat 20-45N, lon 130-180W
import requests, os

BASE = "https://opendap.earthdata.nasa.gov/providers/POCLOUD/collections"
YEARS = [2018, 2019, 2020]
BBOX = {"lat_min": 20, "lat_max": 45, "lon_min": -180, "lon_max": -130}
# Use NASA Earthdata credentials (free account required)
```

### Step 2: Master Dataset Class

The core of everything — `src/dataset/debris_dataset.py` pairs MARIDA/MADOS tiles with their OSCAR temporal sequences by (lat, lon, date): [docs.pytorch](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)

```python
class DebrisDataset(Dataset):
    def __init__(self, split="train", use_hycom=False):
        # Load all tile metadata from MARIDA + MADOS
        marida_tiles = load_marida_index("data/raw/marida")
        mados_tiles  = load_mados_index("data/raw/mados")
        all_tiles = marida_tiles + mados_tiles

        # Split: 70% train, 15% val, 15% test (by scene, not tile)
        # Split by SCENE not by tile to avoid spatial leakage
        self.tiles = split_by_scene(all_tiles, split)
        self.oscar = OSCARLoader("data/raw/oscar")
        self.use_hycom = use_hycom
        if use_hycom:
            self.hycom = HYCOMLoader("data/raw/hycom")

    def __getitem__(self, idx):
        tile = self.tiles[idx]

        # CNN input: bands B4, B8, B11 → (3, 256, 256)
        image = load_sentinel_bands(tile.path, bands=[3, 7, 10])
        image = normalize_sentinel(image)

        # LSTM input: 30-day current sequence → (30, 4) or (30, 6)
        seq = self.oscar.get_sequence(tile.lat, tile.lon, tile.date, window=30)
        if self.use_hycom:
            sst_sal = self.hycom.get_sequence(tile.lat, tile.lon, tile.date)
            seq = np.concatenate([seq, sst_sal], axis=-1)  # (30, 6)

        # Label: binary debris mask → (1, 256, 256)
        mask = load_debris_mask(tile.mask_path)

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(seq,   dtype=torch.float32),
            torch.tensor(mask,  dtype=torch.float32)
        )
```

> **Critical**: split by **scene** not by individual tile — tiles from the same scene share spatial context and will leak into validation if you split randomly. [github](https://github.com/marine-debris/marine-debris.github.io)

***

## Phase 2 — Model

### CNN Encoder (`src/models/cnn_encoder.py`)
```python
import torchvision.models as models

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights="IMAGENET1K_V1")
        # Replace first conv: 3 bands input (B4, B8, B11)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove final FC layer, keep feature extractor
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # → (512,)

    def forward(self, x):
        return self.encoder(x).squeeze(-1).squeeze(-1)  # (B, 512)
```

### LSTM Encoder (`src/models/lstm_encoder.py`)
```python
class LSTMEncoder(nn.Module):
    def __init__(self, input_size=4, hidden=128, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=0.2)

    def forward(self, x):  # x: (B, 30, 4)
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]  # (B, 128) — last layer hidden state
```

### Fusion Model (`src/models/fusion_model.py`)
```python
class DebrisPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn  = CNNEncoder()   # → 512d
        self.lstm = LSTMEncoder()  # → 128d
        self.head = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()           # probability per zone
        )

    def forward(self, image, currents):
        spatial  = self.cnn(image)
        temporal = self.lstm(currents)
        fused    = torch.cat([spatial, temporal], dim=1)  # (B, 640)
        return self.head(fused)
```

***

## Phase 3 — Training Loop

**`src/training/train.py`** key elements:

```python
# config
BATCH_SIZE  = 16
EPOCHS      = 30
LR          = 1e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# loss — weighted BCE because debris pixels are rare (class imbalance)
pos_weight  = torch.tensor([10.0]).to(DEVICE)  # debris : non-debris ≈ 1:10
criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# metrics
from torchmetrics import F1Score, JaccardIndex  # IoU
```

> **Key detail**: use `BCEWithLogitsLoss` with `pos_weight` — debris pixels are heavily outnumbered by ocean pixels in each tile (~1:10 ratio), so without reweighting the model learns to predict "no debris everywhere" and still gets 90% accuracy. [geeksforgeeks](https://www.geeksforgeeks.org/deep-learning/resnet18-from-scratch-using-pytorch/)

***

## Phase 4 — Export for C++ Inference

After training, one script handles the handoff to your LibTorch server:

```python
# src/utils/export.py
model.eval()
scripted = torch.jit.script(model)
scripted.save("inference/debris_predictor.pt")
print("Exported to inference/debris_predictor.pt")
```

***

## `requirements.txt`

```
torch>=2.2.0
torchvision>=0.17.0
torchmetrics>=1.0.0
numpy>=1.24.0
pandas>=2.0.0
xarray>=2023.1.0
netCDF4>=1.6.0
rasterio>=1.3.0
shapely>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
folium>=0.14.0
requests>=2.31.0
python-dotenv>=1.0.0
tqdm>=4.65.0
```

***

## Git Init Checklist

```bash
git init garbage-patch-predictor
cd garbage-patch-predictor
touch README.md requirements.txt .env.example .gitignore

# .gitignore essentials
echo "data/raw/" >> .gitignore
echo ".env" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pt" >> .gitignore       # model weights stay local
echo "*.nc" >> .gitignore       # NetCDF files are large

mkdir -p data src/dataset src/models src/training src/utils notebooks inference agent
```

The build order is: **download scripts → `debris_dataset.py` → CNN baseline → add LSTM → export**. Every phase is independently testable — you can run the CNN alone before the LSTM exists, and the dataset class works before the model exists. [jordanbell](https://jordanbell.info/blog/2022/12/15/sea-surface-currents-OSCAR.html)