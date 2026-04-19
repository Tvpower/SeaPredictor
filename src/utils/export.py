"""Export a trained checkpoint to TorchScript for the C++ LibTorch server."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.models import DebrisPredictor


def export(
    ckpt_path: Path,
    out_path: Path,
    in_channels: int = 3,
    seq_features: int = 4,
    use_temporal: bool = True,
) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    model = DebrisPredictor(
        in_channels=in_channels,
        seq_features=seq_features,
        cnn_pretrained=False,
        use_temporal=use_temporal,
    )
    model.load_state_dict(state["model"])
    model.eval()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # `torch.jit.trace` is friendlier than `script` for the optional-arg forward.
    example_image = torch.zeros(1, in_channels, 256, 256)
    example_seq = torch.zeros(1, 30, seq_features)
    traced = torch.jit.trace(model, (example_image, example_seq))
    traced.save(str(out_path))
    print(f"Exported TorchScript -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("inference/debris_predictor.pt"))
    p.add_argument("--in-channels", type=int, default=3)
    p.add_argument("--seq-features", type=int, default=4)
    p.add_argument("--cnn-only", action="store_true")
    args = p.parse_args()
    export(
        ckpt_path=args.ckpt,
        out_path=args.out,
        in_channels=args.in_channels,
        seq_features=args.seq_features,
        use_temporal=not args.cnn_only,
    )


if __name__ == "__main__":
    main()
