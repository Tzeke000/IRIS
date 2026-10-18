"""Build the MiniFASNet anti-spoof ONNX models used by brain/face_liveness.py.

Source: minivision-ai/Silent-Face-Anti-Spoofing (Apache-2.0). The two .pth
weight files ship INSIDE that Apache-2.0 repo (resources/anti_spoof_models/),
so the weights carry the repo licence. Pinned by SHA-256 below.

Usage (with the venv python, -I so nothing in the download dir can shadow imports):
    python -I scripts/vision/build_antispoof_onnx.py <silentface_src_dir> [out_dir]

<silentface_src_dir> must contain src/model_lib/MiniFASNet.py and
resources/anti_spoof_models/*.pth (fetch with the curl lines in
profiles/vision/face_liveness.md). Default out_dir: models/antispoof/.

Conventions baked into the exported graph's metadata (and mirrored in
brain/face_liveness.py): input is the BGR uint8 crop as float32 WITHOUT /255
(their to_tensor drops the div), NCHW 1x3x80x80; output is 3 logits, class 1 = real.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

PINNED = {
    "2.7_80x80_MiniFASNetV2.pth": "a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0",
    "4_0_0_80x80_MiniFASNetV1SE.pth": "84ee1d37d96894d5e82de5a57df044ef80a58be2b218b5ed7cdfd875ec2f5990",
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src = Path(sys.argv[1]).resolve()
    repo = Path(__file__).resolve().parents[2]
    out = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else repo / "models" / "antispoof"
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch

    spec = importlib.util.spec_from_file_location("silentface_minifasnet", src / "src" / "model_lib" / "MiniFASNet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mapping = {"MiniFASNetV1": mod.MiniFASNetV1, "MiniFASNetV2": mod.MiniFASNetV2,
               "MiniFASNetV1SE": mod.MiniFASNetV1SE, "MiniFASNetV2SE": mod.MiniFASNetV2SE}

    manifest = {"source": "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing",
                "licence": "Apache-2.0", "models": []}
    for name, want in PINNED.items():
        pth = src / "resources" / "anti_spoof_models" / name
        got = _sha(pth)
        if got != want:
            print(f"SHA MISMATCH {name}: {got} != {want}")
            return 1
        info = name.split("_")[0:-1]
        h, w = (int(v) for v in info[-1].split("x"))
        mtype = name.split(".pth")[0].split("_")[-1]
        scale = float(info[0])
        kernel = ((h + 15) // 16, (w + 15) // 16)
        model = mapping[mtype](conv6_kernel=kernel)
        sd = torch.load(str(pth), map_location="cpu", weights_only=True)
        if next(iter(sd)).startswith("module."):
            sd = {k[7:]: v for k, v in sd.items()}
        model.load_state_dict(sd)
        model.eval()
        dummy = torch.from_numpy(np.random.RandomState(0).randint(0, 255, (1, 3, h, w)).astype("float32"))
        onnx_name = name.replace(".pth", ".onnx")
        onnx_path = out / onnx_name
        torch.onnx.export(model, dummy, str(onnx_path), input_names=["input"], output_names=["logits"],
                          opset_version=13, dynamo=False)
        # parity check vs torch
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            ref = model(dummy).numpy()
        got_o = sess.run(None, {"input": dummy.numpy()})[0]
        diff = float(np.abs(ref - got_o).max())
        print(f"{onnx_name}: scale={scale} input={h}x{w} max|torch-onnx|={diff:.2e}")
        if diff > 1e-3:
            print("PARITY FAIL")
            return 1
        manifest["models"].append({"file": onnx_name, "scale": scale, "h": h, "w": w,
                                   "type": mtype, "pth_sha256": want, "onnx_sha256": _sha(onnx_path),
                                   "real_class": 1, "input": "BGR float32 0..255 (no /255), NCHW"})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("wrote", out / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
