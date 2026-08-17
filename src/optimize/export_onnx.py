"""Export the V4 FCOS detector (and recognizer encoder) to ONNX + INT8 for CPU inference.

The model heads are per-FPN-level tensor lists; for a FIXED letterboxed input size the feature-map
sizes are deterministic, so we wrap the model to emit three flat tensors (cls/reg/ctr concatenated
over levels, in P3..P7 order) and precompute the matching per-location centers+strides at decode
time. Decode + NMS stay in numpy (see infer_obj_onnx.py).

  python -m src.optimize.export_onnx detector --ckpt data/checkpoints/detector_obj_v4/step_00007132/model.pt \
      --size 1024 --profile single --int8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.detect.model.fcos import build_obj_detector


class _DetExport(nn.Module):
    """Wrap FCOS to return (cls_sig [1,N,C], reg [1,N,4], ctr_sig [1,N]) concatenated P3..P7."""

    def __init__(self, model: nn.Module, num_classes: int):
        super().__init__()
        self.model = model
        self.nc = num_classes

    def forward(self, x):
        out = self.model(x)
        cls = torch.cat([o.permute(0, 2, 3, 1).reshape(1, -1, self.nc) for o in out["cls"]], 1)
        reg = torch.cat([o.permute(0, 2, 3, 1).reshape(1, -1, 4) for o in out["reg"]], 1)
        ctr = torch.cat([o.permute(0, 2, 3, 1).reshape(1, -1) for o in out["ctr"]], 1)
        return cls.sigmoid(), reg, ctr.sigmoid()


def export_detector(ckpt: str, size: int, profile: str, out_dir: str, int8: bool = True) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model = build_obj_detector(profile)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state.get("model", state))
    model.eval()
    wrap = _DetExport(model, model.num_classes).eval()

    dummy = torch.randn(1, 3, size, size)
    fp32 = out_dir / f"detector_{profile}_{size}.onnx"
    torch.onnx.export(
        wrap, dummy, str(fp32),
        input_names=["image"], output_names=["cls", "reg", "ctr"],
        opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"[export] fp32 -> {fp32}")

    paths = {"fp32": str(fp32)}
    if int8:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        q = out_dir / f"detector_{profile}_{size}.int8.onnx"
        quantize_dynamic(str(fp32), str(q), weight_type=QuantType.QInt8)
        print(f"[export] int8 -> {q}")
        paths["int8"] = str(q)
    # stash decode metadata next to the model
    meta = {"size": size, "profile": profile, "num_classes": model.num_classes}
    (out_dir / f"detector_{profile}_{size}.meta.json").write_text(__import__("json").dumps(meta))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["detector"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--profile", default="single")
    ap.add_argument("--out-dir", default="data/checkpoints/onnx")
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()
    if args.target == "detector":
        export_detector(args.ckpt, args.size, args.profile, args.out_dir, args.int8)


if __name__ == "__main__":
    main()
