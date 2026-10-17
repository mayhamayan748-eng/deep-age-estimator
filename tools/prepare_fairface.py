"""Crop FairFace faces exactly like the web app does, so they match UTKFace's framing.

    python tools/prepare_fairface.py <fairface_dir>/0.25 <out_dir>

Every image goes through backend/app/services/preprocessing.py (YuNet detection, UTKFace-style margins,
128x128 resize). Output per split: <split>_images.npy (uint8, N x 128 x 128 x 3, RGB) and <split>_labels.csv.
Images where no face is detected are skipped and counted.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.services import preprocessing as pp  # noqa: E402
from app.services.models import BUNDLED_FACE_DETECTOR  # noqa: E402

# Class-label order from the dataset card (HuggingFaceM4/FairFace)
AGE_BINS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
AGE_RANGES = [(0, 2), (3, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, pp.MAX_AGE)]
RACES = ["East Asian", "Indian", "Black", "White", "Middle Eastern", "Latino_Hispanic", "Southeast Asian"]
GENDERS = ["Male", "Female"]


def prepare(parquet_files: list[Path], out_dir: Path, split: str) -> None:
    rows = sum(pq.ParquetFile(f).metadata.num_rows for f in parquet_files)
    images = np.lib.format.open_memmap(out_dir / f"{split}_images.tmp.npy", mode="w+", dtype=np.uint8,
                                       shape=(rows, pp.IMG_SIZE, pp.IMG_SIZE, 3))
    labels, skipped, n = [], Counter(), 0
    for f in parquet_files:
        table = pq.read_table(f, columns=["image", "age", "gender", "race"])
        for img_cell, age, gender, race in zip(*(table.column(c).to_pylist() for c in ["image", "age", "gender", "race"])):
            img = pp.decode_image(img_cell["bytes"])
            boxes = pp.detect_faces(img, BUNDLED_FACE_DETECTOR) if img is not None else []
            if not boxes:
                skipped[RACES[race]] += 1
                continue
            face, _ = pp.crop_with_margin(img, boxes[0])
            images[n] = pp.to_model_input(face, channels=3)
            lo, hi = AGE_RANGES[age]
            labels.append({"age_bin": AGE_BINS[age], "age_lo": lo, "age_hi": hi,
                           "race": RACES[race], "gender": GENDERS[gender]})
            n += 1
            if n % 5000 == 0:
                print(f"  {split}: {n}/{rows}", flush=True)
    images.flush()
    # Keep only the rows that were filled
    final = np.lib.format.open_memmap(out_dir / f"{split}_images.npy", mode="w+", dtype=np.uint8, shape=(n, *images.shape[1:]))
    final[:] = images[:n]
    final.flush()
    del images
    (out_dir / f"{split}_images.tmp.npy").unlink()
    with open(out_dir / f"{split}_labels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(labels[0]))
        w.writeheader()
        w.writerows(labels)
    print(f"{split}: kept {n}/{rows} faces, no face detected in {sum(skipped.values())} ({dict(skipped)})")


def main() -> None:
    src, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    prepare(sorted(src.glob("train-*.parquet")), out_dir, "train")
    prepare(sorted(src.glob("validation-*.parquet")), out_dir, "val")


if __name__ == "__main__":
    main()
