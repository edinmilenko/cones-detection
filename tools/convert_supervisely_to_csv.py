#!/usr/bin/env python3
# File a parte, non collegato a CMakeLists: converte le annotazioni Supervisely (bitmap in
# base64+zlib, con offset "origin") del vero test set esterno (test_set/segmentation_test) nello
# stesso formato CSV usato da tutta la pipeline C++ per data/dataset.csv (image,x1,y1,x2,y2), cosi'
# funnel_analysis/eval_detector/eval_cascade possono girare sul test set reale senza modifiche
# strutturali, solo puntando a file diversi.
#
# Uso: python3 tools/convert_supervisely_to_csv.py
# Scrive: data/dataset_real.csv, data/test_real.txt

import base64
import io
import json
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ANN_DIR = ROOT / "test_set" / "segmentation_test" / "ann"
IMG_DIR = ROOT / "test_set" / "segmentation_test" / "img"
OUT_CSV = ROOT / "data" / "dataset_real.csv"
OUT_SPLIT = ROOT / "data" / "test_real.txt"

CONE_CLASSES = {"seg_blue_cone", "seg_yellow_cone", "seg_orange_cone",
                 "seg_large_orange_cone", "seg_unknown_cone"}


def decode_bitmap_bbox(obj):
    """Decodifica il campo bitmap.data (base64+zlib di un PNG) e ritorna la bbox
    (x1,y1,x2,y2) in coordinate immagine, usando origin + la bbox del contenuto non a zero
    della maschera (piu' robusto di fidarsi solo delle dimensioni del PNG, in caso di padding)."""
    raw = base64.b64decode(obj["bitmap"]["data"])
    png_bytes = zlib.decompress(raw)
    mask = np.array(Image.open(io.BytesIO(png_bytes)))
    if mask.ndim == 3:
        mask = mask[..., -1] if mask.shape[-1] == 4 else mask.any(axis=-1)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    ox, oy = obj["bitmap"]["origin"]
    x1, x2 = int(xs.min()) + ox, int(xs.max()) + ox + 1
    y1, y2 = int(ys.min()) + oy, int(ys.max()) + oy + 1
    return x1, y1, x2, y2


def main():
    ann_files = sorted(ANN_DIR.glob("*.json"))
    print(f"{len(ann_files)} file di annotazione trovati")

    rows = []
    img_names = []
    n_boxes = 0
    n_skipped_class = 0

    for ann_path in ann_files:
        img_name = ann_path.stem  # "amz_01260.png.json" -> "amz_01260.png"
        img_path = IMG_DIR / img_name
        if not img_path.exists():
            print(f"ATTENZIONE: immagine mancante per {ann_path.name}, salto")
            continue
        img_names.append(img_name)

        with open(ann_path) as f:
            data = json.load(f)

        for obj in data.get("objects", []):
            if obj.get("classTitle") not in CONE_CLASSES:
                n_skipped_class += 1
                continue
            if obj.get("geometryType") != "bitmap":
                continue
            bbox = decode_bitmap_bbox(obj)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            rows.append(f"{img_name},{x1},{y1},{x2},{y2}")
            n_boxes += 1

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.write_text("\n".join(rows) + "\n")
    OUT_SPLIT.write_text("\n".join(img_names) + "\n")

    print(f"immagini: {len(img_names)}, box totali: {n_boxes}, oggetti non-cono saltati: {n_skipped_class}")
    print(f"scritto: {OUT_CSV}")
    print(f"scritto: {OUT_SPLIT}")


if __name__ == "__main__":
    main()
