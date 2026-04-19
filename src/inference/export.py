"""Export a trained checkpoint to TorchScript for LibTorch / C++ deployment.

We use `torch.jit.trace` rather than `script` because:
  - The model has no data-dependent control flow (the `use_temporal` flag is a
    Python attribute fixed at __init__).
  - `Optional[Tensor]` arguments don't script cleanly, but trace is fine when
    we always pass a concrete tensor.

The exported module always takes (image, currents). For CNN-only checkpoints,
currents is ignored internally — pass a zero tensor of the right shape.

Usage:
    python -m src.inference.export \
        --ckpt checkpoints/cnn_only/best.pt \
        --out  checkpoints/cnn_only/model.ts.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.models import DebrisPredictor


def build_model_from_ckpt(ckpt_path: Path) -> tuple[torch.nn.Module, dict]:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state.get("cfg", {})

    model = DebrisPredictor(
        in_channels=cfg.get("in_channels", 11),
        seq_features=cfg.get("seq_features", 4),
        num_classes=cfg.get("num_classes", 15),
        cnn_pretrained=False,
        use_temporal=cfg.get("use_temporal", True),
        head_dropout=0.0,  # eval-time export, no dropout
    )
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--check-tolerance", type=float, default=1e-5,
                        help="Max abs diff between eager and traced output (sanity check).")
    args = parser.parse_args()

    model, cfg = build_model_from_ckpt(args.ckpt)
    in_channels = cfg.get("in_channels", 11)
    seq_features = cfg.get("seq_features", 4)
    seq_length = cfg.get("seq_length", 30)
    num_classes = cfg.get("num_classes", 15)

    # Trace on CPU — the saved TorchScript can be loaded on any device.
    example_image = torch.randn(1, in_channels, args.tile_size, args.tile_size)
    example_seq = torch.zeros(1, seq_length, seq_features)

    print(f"[export] tracing on CPU  image={tuple(example_image.shape)}  "
          f"seq={tuple(example_seq.shape)}")
    with torch.no_grad():
        traced = torch.jit.trace(model, (example_image, example_seq), check_trace=False)

    # Sanity check: eager vs. traced output should match.
    with torch.no_grad():
        out_eager = model(example_image, example_seq)
        out_traced = traced(example_image, example_seq)
    diff = (out_eager - out_traced).abs().max().item()
    print(f"[export] eager vs traced max abs diff: {diff:.2e}")
    if diff > args.check_tolerance:
        raise RuntimeError(f"Trace diverges from eager by {diff:.2e} > tol")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(args.out))
    print(f"[export] wrote TorchScript -> {args.out}")

    # Drop a sidecar with input shapes + class count so the C++ side knows
    # what to feed in.
    meta = {
        "input_image_shape": [1, in_channels, args.tile_size, args.tile_size],
        "input_seq_shape": [1, seq_length, seq_features],
        "num_classes": int(num_classes),
        "use_temporal": bool(cfg.get("use_temporal", True)),
        "source_ckpt": str(args.ckpt),
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[export] wrote metadata -> {meta_path}")


if __name__ == "__main__":
    main()
