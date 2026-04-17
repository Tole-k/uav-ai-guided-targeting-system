"""
Quick bbox-format check for HIT-UAV rotate_json.

Draws the first image from train.json twice:
  * _center.jpg   — interpreting bbox as [cx, cy, w, h, theta]
  * _topleft.jpg  — interpreting bbox as [xmin, ymin, w, h, theta]  (COCO)

Open both. Whichever one has rectangles sitting on the actual people/cars
tells you which format HIT-UAV uses in your copy of the dataset.

Usage:
    python check_bbox_format.py \
        --json  HIT-UAV-Infrared-Thermal-Dataset/rotate_json/train.json \
        --images HIT-UAV-Infrared-Thermal-Dataset/images \
        --out   ./check
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def corners(cx, cy, w, h, theta):
    ca, sa = math.cos(theta), math.sin(theta)
    hw, hh = w / 2, h / 2
    return np.array(
        [
            (cx + dx * ca - dy * sa, cy + dx * sa + dy * ca)
            for dx, dy in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        ],
        dtype=np.int32,
    )


def draw(img, annotations, mode):
    out = img.copy()
    for ann in annotations:
        bbox = ann["bbox"]
        if len(bbox) == 5:
            b1, b2, w, h, theta = bbox
        else:
            b1, b2, w, h = bbox
            theta = 0.0
        if mode == "center":
            cx, cy = b1, b2
        else:  # "topleft"
            cx, cy = b1 + w / 2, b2 + h / 2
        pts = corners(cx, cy, w, h, theta)
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--json", required=True, help="path to train.json / val.json / test.json"
    )
    p.add_argument("--images", required=True, help="folder containing the .jpg images")
    p.add_argument("--out", default="./check", help="output folder")
    p.add_argument(
        "--n", type=int, default=3, help="how many images to render (default 3)"
    )
    args = p.parse_args()

    data = json.loads(Path(args.json).read_text())
    imgs_dir = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # group annotations per image
    anns_by_img = {}
    for ann in data["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    # take the first N images that have annotations
    picked = []
    for img_meta in data["images"]:
        if img_meta["id"] in anns_by_img:
            picked.append(img_meta)
        if len(picked) >= args.n:
            break

    for img_meta in picked:
        fname = (
            img_meta.get("file_name")
            or img_meta.get("filename")
            or f"{img_meta['id']}.jpg"
        )
        path = imgs_dir / fname
        if not path.exists():
            path = next(imgs_dir.rglob(Path(fname).name), None)  # try recursive
        if not path or not path.exists():
            print(f"  [skip] {fname} not found under {imgs_dir}")
            continue

        img = cv2.imread(str(path))
        anns = anns_by_img[img_meta["id"]]
        stem = Path(fname).stem

        cv2.imwrite(str(out_dir / f"{stem}_center.jpg"), draw(img, anns, "center"))
        cv2.imwrite(str(out_dir / f"{stem}_topleft.jpg"), draw(img, anns, "topleft"))
        print(
            f"  wrote  {stem}_center.jpg  +  {stem}_topleft.jpg   ({len(anns)} boxes)"
        )

    print(f"\nOpen both variants in {out_dir} and see which one aligns with objects.")


if __name__ == "__main__":
    main()
