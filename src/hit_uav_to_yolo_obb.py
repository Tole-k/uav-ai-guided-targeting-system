"""
HIT-UAV  →  YOLO OBB  converter  (v2, corrected)
==================================================

Converts HIT-UAV oriented-bbox annotations to Ultralytics YOLO OBB format:

    class_id  x1 y1  x2 y2  x3 y3  x4 y4   (normalised 0..1, clockwise from TL)


IMPORTANT NOTES about the real repo layout
------------------------------------------

1. The `rotate_json/` folder contains THREE files, one per split:
       train.json,  val.json,  test.json      (each COCO-style, all images)
   NOT one JSON per image.

2. The `bbox` field is stored COCO-style:  [x_min, y_min, w, h, theta]
   (top-left corner of the UN-ROTATED box + width + height + angle),
   NOT center-based.  This was confirmed by Issue #6 on the repo, which
   showed that visualising `bbox` as `[cx, cy, w, h]` produces a -w/2, -h/2
   offset error.  We therefore convert  (x_min, y_min) -> (cx, cy)  ourselves.

3. The `rotate_xml/` folder uses roLabelImg style `<robndbox>` with
   <cx>, <cy>, <w>, <h>, <angle> (radians) — the center-based convention.

4. `angle` (theta) units:
     * rotate_json  -> radians  (CCW from +x axis)
     * rotate_xml   -> radians  (roLabelImg convention, range [0, pi))
   Use `--angle-unit degrees` to override if your copy stores degrees.

5. The dataset images are 640 x 512 px.  Use --img-w / --img-h if yours differ.

6. Class IDs are zero-indexed in the output, matching the paper's ordering:
       0 Person, 1 Car, 2 Bicycle, 3 OtherVehicle, 4 DontCare
   (Adjust `CLASS_NAME_TO_ID` below if your JSON uses different `category_id`s.)


Usage
-----
    # JSON source (one COCO file per split - the common case)
    python hit_uav_to_yolo_obb.py \\
        --src  HIT-UAV-Infrared-Thermal-Dataset/rotate_json \\
        --out  HIT-UAV-Infrared-Thermal-Dataset/yolo_obb_labels

    # XML source (per-image roLabelImg files)
    python hit_uav_to_yolo_obb.py \\
        --src  HIT-UAV-Infrared-Thermal-Dataset/rotate_xml \\
        --out  HIT-UAV-Infrared-Thermal-Dataset/yolo_obb_labels

    # If your angles are stored as degrees instead of radians
    python hit_uav_to_yolo_obb.py --src ... --out ... --angle-unit degrees

    # If the bbox really is [cx,cy,w,h,theta] in your copy (override COCO default)
    python hit_uav_to_yolo_obb.py --src ... --out ... --bbox-is-center
"""

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


# ── class name -> YOLO id  (edit if your category ids differ) ────────────────
CLASS_NAME_TO_ID = {
    "Person": 0,
    "person": 0,
    "Car": 1,
    "car": 1,
    "Bicycle": 2,
    "bicycle": 2,
    "OtherVehicle": 3,
    "Other Vehicle": 3,
    "othervehicle": 3,
    "other_vehicle": 3,
    "DontCare": 4,
    "Dontcare": 4,
    "dontcare": 4,
    "Don'tCare": 4,
    "don't care": 4,
}


# ── geometry ─────────────────────────────────────────────────────────────────


def corners_from_cxcywh_theta(cx, cy, w, h, angle_rad):
    """
    Four corners of a rotated box defined by its centre + size + angle.
    Returns corners in clockwise order starting from the rotated TL.
    """
    ca = math.cos(angle_rad)
    sa = math.sin(angle_rad)
    hw, hh = w / 2.0, h / 2.0
    offsets = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]  # TL, TR, BR, BL
    return [(cx + dx * ca - dy * sa, cy + dx * sa + dy * ca) for dx, dy in offsets]


def normalise_corners(corners, img_w, img_h, clip=True):
    out = []
    for x, y in corners:
        nx, ny = x / img_w, y / img_h
        if clip:
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))
        out.append((nx, ny))
    return out


def yolo_obb_line(class_id, norm_corners):
    coords = " ".join(f"{v:.6f}" for pt in norm_corners for v in pt)
    return f"{class_id} {coords}"


# ── COCO (train.json / val.json / test.json) parser ──────────────────────────


def extract_image_stem(img_meta, img_id):
    """
    Extract the proper filename stem from an image record.

    HIT-UAV image names follow the pattern  T_HH(H)_AA_W_NNNNN  (e.g.
    `0_60_30_0_01609`).  Different mirrors of the dataset have been observed
    to store this under various keys, so we try a list of candidates before
    falling back to the numeric image id.
    """
    for key in ("file_name", "filename", "image_name", "img_name", "name", "path"):
        val = img_meta.get(key)
        if val:
            return Path(str(val)).stem
    # last-resort fallback
    return str(img_id)


def parse_coco_file(
    json_path,
    out_dir,
    img_w_default,
    img_h_default,
    angle_unit="radians",
    bbox_is_center=False,
):
    """
    Parse one COCO-style JSON (train.json / val.json / test.json).
    Writes one YOLO OBB .txt file per image (including empty files for
    images that contain no annotations).
    """
    with open(json_path) as f:
        data = json.load(f)

    # id -> image meta
    img_info = {img["id"]: img for img in data.get("images", [])}

    # Diagnostic: show the fields present in the first image record so the
    # user can spot issues (e.g. filename stored under an unexpected key).
    if img_info:
        sample_id, sample_meta = next(iter(img_info.items()))
        print(f"  sample image record: keys = {sorted(sample_meta.keys())}")
        print(f"  sample image record: first = {sample_meta}")

    # category_id -> YOLO class_id
    cat_map = {}
    skipped_cats = []
    for cat in data.get("categories", []):
        cid = CLASS_NAME_TO_ID.get(cat["name"])
        if cid is None:
            skipped_cats.append(cat["name"])
            continue
        cat_map[cat["id"]] = cid
    if skipped_cats:
        print(f"  [WARN] unknown category names (skipped): {skipped_cats}")

    # group annotations per image
    ann_by_img = defaultdict(list)
    for ann in data.get("annotations", []):
        ann_by_img[ann["image_id"]].append(ann)

    written = 0
    empty = 0
    bad_bbox = 0
    for img_id, img_meta in img_info.items():
        width = img_meta.get("width", img_w_default)
        height = img_meta.get("height", img_h_default)
        stem = extract_image_stem(img_meta, img_id)

        lines = []
        for ann in ann_by_img.get(img_id, []):
            class_id = cat_map.get(ann["category_id"])
            if class_id is None:
                continue
            bbox = ann.get("bbox", [])
            if len(bbox) == 5:
                b1, b2, w, h, theta = [float(v) for v in bbox]
            elif len(bbox) == 4:
                b1, b2, w, h = [float(v) for v in bbox]
                theta = 0.0
            else:
                bad_bbox += 1
                continue

            if angle_unit == "degrees":
                theta = math.radians(theta)

            # COCO stores top-left by default; convert to center
            if bbox_is_center:
                cx, cy = b1, b2
            else:
                cx = b1 + w / 2.0
                cy = b2 + h / 2.0

            corners = corners_from_cxcywh_theta(cx, cy, w, h, theta)
            norm_corners = normalise_corners(corners, width, height)
            lines.append(yolo_obb_line(class_id, norm_corners))

        out_file = out_dir / f"{stem}.txt"
        if lines:
            out_file.write_text("\n".join(lines) + "\n")
            written += 1
        else:
            out_file.write_text("")
            empty += 1

    tag = f" ({bad_bbox} malformed bboxes)" if bad_bbox else ""
    print(
        f"  {json_path.name}: wrote {written} labelled + {empty} empty .txt files{tag}"
    )


# ── roLabelImg XML parser (one file per image) ───────────────────────────────


def parse_xml_file(xml_path, img_w_default, img_h_default, angle_unit="radians"):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is not None:
        width = int(size.findtext("width", default=str(img_w_default)))
        height = int(size.findtext("height", default=str(img_h_default)))
    else:
        width, height = img_w_default, img_h_default

    lines = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        class_id = CLASS_NAME_TO_ID.get(name)
        if class_id is None:
            continue

        obj_type = (obj.findtext("type") or "bndbox").strip()
        if obj_type == "robndbox":
            rob = obj.find("robndbox")
            cx = float(rob.findtext("cx"))
            cy = float(rob.findtext("cy"))
            w = float(rob.findtext("w"))
            h = float(rob.findtext("h"))
            theta = float(rob.findtext("angle") or "0")
        else:
            bnd = obj.find("bndbox")
            xmin = float(bnd.findtext("xmin"))
            ymin = float(bnd.findtext("ymin"))
            xmax = float(bnd.findtext("xmax"))
            ymax = float(bnd.findtext("ymax"))
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            w = xmax - xmin
            h = ymax - ymin
            theta = 0.0

        if angle_unit == "degrees":
            theta = math.radians(theta)

        corners = corners_from_cxcywh_theta(cx, cy, w, h, theta)
        norm_corners = normalise_corners(corners, width, height)
        lines.append(yolo_obb_line(class_id, norm_corners))

    return lines


# ── main driver ──────────────────────────────────────────────────────────────


def convert(
    src,
    out,
    img_w,
    img_h,
    angle_unit="radians",
    bbox_is_center=False,
    split_subdirs=True,
):
    out.mkdir(parents=True, exist_ok=True)

    # Case-insensitive, recursive file discovery.
    # The HIT-UAV Zenodo zip sometimes unpacks with a nested folder, e.g.
    #   rotate_json/rotate_json/train.json   or
    #   rotate_json/HIT-UAV-.../rotate_json/train.json
    # so we look into subdirectories too.
    all_files = [p for p in src.rglob("*") if p.is_file()]
    json_files = sorted(p for p in all_files if p.suffix.lower() == ".json")
    xml_files = sorted(p for p in all_files if p.suffix.lower() == ".xml")

    print(f"  Scanned {src} recursively:")
    print(
        f"    {len(all_files)} files total, "
        f"{len(json_files)} .json, {len(xml_files)} .xml\n"
    )

    if not json_files and not xml_files:
        print("  [ERROR] No annotation files found.")
        print("  Directory listing of the --src path (up to 30 entries):")
        listing = sorted(src.iterdir()) if src.is_dir() else []
        for entry in listing[:30]:
            kind = "DIR " if entry.is_dir() else "FILE"
            print(f"    {kind}  {entry.name}")
        if len(listing) > 30:
            print(f"    ... ({len(listing) - 30} more)")
        if not listing:
            print("    (empty)")
            print("")
            print("  The GitHub repo lists these folders but annotation files")
            print("  are NOT committed to git — they are distributed via Zenodo.")
            print("  Download the 815 MB zip from:")
            print("    https://zenodo.org/records/7633134")
            print("  (or use the Kaggle mirror:")
            print(
                "   https://www.kaggle.com/datasets/pandrii000/hituav-a-highaltitude-infrared-thermal-dataset)"
            )
        return

    # ── JSON branch: one COCO file per split ──
    if json_files:
        for jf in json_files:
            split = jf.stem.lower()  # e.g. "train", "val", "test"
            if split_subdirs and split in ("train", "val", "valid", "test"):
                sub = out / ("val" if split == "valid" else split)
                sub.mkdir(parents=True, exist_ok=True)
                print(f"Converting {jf.name}  ->  {sub}/")
                parse_coco_file(
                    jf,
                    sub,
                    img_w,
                    img_h,
                    angle_unit=angle_unit,
                    bbox_is_center=bbox_is_center,
                )
            else:
                print(f"Converting {jf.name}  ->  {out}/")
                parse_coco_file(
                    jf,
                    out,
                    img_w,
                    img_h,
                    angle_unit=angle_unit,
                    bbox_is_center=bbox_is_center,
                )
        return

    # ── XML branch: one roLabelImg file per image ──
    written = empty = 0
    for xf in xml_files:
        lines = parse_xml_file(xf, img_w, img_h, angle_unit=angle_unit)
        out_file = out / f"{xf.stem}.txt"
        if lines:
            out_file.write_text("\n".join(lines) + "\n")
            written += 1
        else:
            out_file.write_text("")
            empty += 1
    print(f"  Converted XML: {written} labelled + {empty} empty .txt files")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description="HIT-UAV rotate_json / rotate_xml  ->  YOLO OBB labels",
    )
    p.add_argument(
        "--src", required=True, help="path to rotate_json/ or rotate_xml/ folder"
    )
    p.add_argument("--out", required=True, help="output folder for YOLO OBB .txt files")
    p.add_argument(
        "--img-w", type=int, default=640, help="image width in pixels (default 640)"
    )
    p.add_argument(
        "--img-h", type=int, default=512, help="image height in pixels (default 512)"
    )
    p.add_argument(
        "--angle-unit",
        choices=["radians", "degrees"],
        default="radians",
        help="unit of the theta/angle field (default: radians)",
    )
    p.add_argument(
        "--bbox-is-center",
        action="store_true",
        help="treat JSON bbox as [cx,cy,w,h,theta] instead of "
        "COCO default [xmin,ymin,w,h,theta]",
    )
    p.add_argument(
        "--flat-output",
        action="store_true",
        help="do not create train/val/test subfolders under --out",
    )
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.exists():
        print(f"[ERROR] source path does not exist: {src}")
        return

    bbox_desc = (
        "[cx,cy,w,h,theta]" if args.bbox_is_center else "[xmin,ymin,w,h,theta]  (COCO)"
    )
    print(f"Source      : {src}")
    print(f"Output      : {out}")
    print(f"Image size  : {args.img_w} x {args.img_h}")
    print(f"Angle unit  : {args.angle_unit}")
    print(f"bbox format : {bbox_desc}")
    print()

    convert(
        src,
        out,
        args.img_w,
        args.img_h,
        angle_unit=args.angle_unit,
        bbox_is_center=args.bbox_is_center,
        split_subdirs=not args.flat_output,
    )


if __name__ == "__main__":
    main()
