"""Build the in-browser version of the site and publish it as a free Hugging Face Static Space.

    python deploy/browser/export_onnx.py models models/onnx UTKFace   # once, after training (WSL/Linux)
    hf auth login                                                    # once
    python deploy/browser/deploy_static.py <username>/deep-age-estimator

Use --build-only to just produce frontend/dist (serve it with any static file server to test).
"""
import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
DIST = FRONTEND / "dist"
ONNX_DIR = ROOT / "models" / "onnx"
REQUIRED = ["face_detection_yunet.onnx", "age_efficientnet_diverse.onnx", "age_efficientnet.onnx", "age_cnn_gan.onnx", "age_cnn.onnx",
            "gan_generator.onnx", "metrics.json", "calibration.json"]

SPACE_README = """---
title: Deep Age Estimator
emoji: 🎞️
colorFrom: red
colorTo: gray
sdk: static
pinned: false
short_description: Estimate age from a face photo, in your browser
---

Upload a photo and three neural networks trained on UTKFace estimate the age of every face in it
(EfficientNetB0: 4.97 years mean error on the test set). A conditional GAN generates faces for any age.
Everything runs in the browser with ONNX Runtime Web: photos never leave the device.

Graduation project by Ayham Ayan · https://github.com/mayhamayan748-eng/deep-age-estimator
"""


def build() -> None:
    missing = [f for f in REQUIRED if not (ONNX_DIR / f).exists()]
    if missing:
        raise SystemExit(f"Missing in models/onnx/: {', '.join(missing)} — run export_onnx.py first")
    env = {**os.environ, "VITE_INFERENCE": "browser"}
    subprocess.run("npm run build", cwd=FRONTEND, env=env, shell=True, check=True)
    models = DIST / "models"
    shutil.rmtree(models, ignore_errors=True)
    models.mkdir()
    for f in REQUIRED:
        shutil.copy(ONNX_DIR / f, models / f)
    (DIST / "README.md").write_text(SPACE_README, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("space", nargs="?", help="Space id, e.g. username/deep-age-estimator")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    build()
    if args.build_only:
        print(f"Built {DIST}")
        return
    if not args.space:
        parser.error("space id is required unless --build-only is given")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(repo_id=args.space, repo_type="space", folder_path=str(DIST),
                      commit_message="Deploy Deep Age Estimator (in-browser)")
    host = api.space_info(args.space).host  # static Spaces are served from *.static.hf.space
    print(f"Live at {host}  (Space page: https://huggingface.co/spaces/{args.space})")


if __name__ == "__main__":
    main()
