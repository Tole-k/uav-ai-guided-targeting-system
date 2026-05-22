"""
HIT-UAV rotate_json -> 4 YOLO OBB datasets split by scenario and weather.

This script reads the same COCO-style oriented annotations used by
`hit_uav_to_yolo_obb.py`, but writes four separate Ultralytics-style datasets:

    day_sunny
    day_rainy
    night_sunny
    night_rainy

Each dataset contains:

    images/train, images/val, images/test
    labels/train, labels/val, labels/test
    dataset.yaml

The script also copies the corresponding source image into the matching
condition dataset.

Example:
    python src/hit_uav_to_yolo_obb_by_condition.py \
        --src HIT-UAV-Infrared-Thermal-Dataset/rotate_json \
        --out HIT-UAV-Infrared-Thermal-Dataset/yolo_obb_by_condition
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

from hit_uav_to_yolo_obb import (
    CLASS_NAME_TO_ID,
    corners_from_cxcywh_theta,
    normalise_corners,
    yolo_obb_line,
)


EXPECTED_SPLITS = {"train", "val", "valid", "test"}
CLASS_ID_TO_NAME = {
    0: "Person",
    1: "Car",
    2: "Bicycle",
    3: "OtherVehicle",
    4: "DontCare",
}


def slugify(value: str) -> str:
    value = value.strip().lower()
    chars = []
    previous_was_sep = False
    for char in value:
        if char.isalnum():
            chars.append(char)
            previous_was_sep = False
            continue
        if previous_was_sep:
            continue
        chars.append("_")
        previous_was_sep = True
    return "".join(chars).strip("_") or "unknown"


def extract_image_filename(img_meta: dict, img_id: int) -> str:
    for key in ("file_name", "filename", "image_name", "img_name", "name", "path"):
        value = img_meta.get(key)
        if value:
            return Path(str(value)).name
    return f"{img_id}.jpg"


def build_id_to_name(entries: list[dict]) -> dict[str, str]:
    mapping = {}
    for entry in entries:
        entry_id = entry.get("id")
        name = entry.get("name")
        if entry_id is None or name is None:
            continue
        mapping[str(entry_id)] = slugify(str(name))
    return mapping


def resolve_condition_name(raw_value: object, id_to_name: dict[str, str]) -> str:
    if raw_value is None:
        return "unknown"
    raw_text = str(raw_value).strip()
    if raw_text in id_to_name:
        return id_to_name[raw_text]
    return slugify(raw_text)


def dataset_root_for_json(json_path: Path) -> Path:
    if json_path.parent.name.lower() == "annotations":
        return json_path.parent.parent
    return json_path.parent


def resolve_image_path(
    json_path: Path,
    split_name: str,
    image_filename: str,
    images_root: Path | None,
) -> Path | None:
    split_dir_name = "val" if split_name == "valid" else split_name
    dataset_root = dataset_root_for_json(json_path)
    candidates = []

    if images_root is not None:
        candidates.extend(
            [
                images_root / split_dir_name / image_filename,
                images_root / "JPEGImages" / image_filename,
                images_root / image_filename,
            ]
        )

    candidates.extend(
        [
            dataset_root / split_dir_name / image_filename,
            dataset_root / "JPEGImages" / image_filename,
            dataset_root / image_filename,
        ]
    )

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            return candidate

    return None


def yolo_lines_for_image(
    annotations: list[dict],
    cat_map: dict[int, int],
    width: int,
    height: int,
    angle_unit: str,
    bbox_is_center: bool,
) -> tuple[list[str], int]:
    lines = []
    malformed = 0
    for ann in annotations:
        class_id = cat_map.get(ann.get("category_id"))
        if class_id is None:
            continue

        bbox = ann.get("bbox", [])
        if len(bbox) == 5:
            b1, b2, box_w, box_h, theta = [float(value) for value in bbox]
        elif len(bbox) == 4:
            b1, b2, box_w, box_h = [float(value) for value in bbox]
            theta = 0.0
        else:
            malformed += 1
            continue

        if angle_unit == "degrees":
            theta = math.radians(theta)

        if bbox_is_center:
            cx, cy = b1, b2
        else:
            cx = b1 + box_w / 2.0
            cy = b2 + box_h / 2.0

        corners = corners_from_cxcywh_theta(cx, cy, box_w, box_h, theta)
        norm_corners = normalise_corners(corners, width, height)
        lines.append(yolo_obb_line(class_id, norm_corners))

    return lines, malformed


def ensure_dataset_dirs(out_root: Path, dataset_names: list[str]) -> dict[str, Path]:
    dataset_dirs = {}
    for dataset_name in dataset_names:
        dataset_dir = out_root / dataset_name
        dataset_dirs[dataset_name] = dataset_dir
        for split_name in ("train", "val", "test"):
            (dataset_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
            (dataset_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)
    return dataset_dirs


def write_dataset_yaml(dataset_dir: Path) -> None:
    yaml_lines = [
        f"path: {dataset_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for class_id, class_name in CLASS_ID_TO_NAME.items():
        yaml_lines.append(f"  {class_id}: {class_name}")
    yaml_lines.append(f"nc: {len(CLASS_ID_TO_NAME)}")
    (dataset_dir / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n")


def find_split_jsons(src: Path) -> list[Path]:
    split_files = []
    for json_path in src.rglob("*.json"):
        if json_path.stem.lower() in EXPECTED_SPLITS:
            split_files.append(json_path)
    return sorted(split_files)


def convert_split(
    json_path: Path,
    dataset_dirs: dict[str, Path],
    images_root: Path | None,
    img_w_default: int,
    img_h_default: int,
    angle_unit: str,
    bbox_is_center: bool,
    stats: dict[str, dict[str, int]],
) -> None:
    split_name = json_path.stem.lower()
    output_split = "val" if split_name == "valid" else split_name

    with open(json_path) as handle:
        data = json.load(handle)

    scenario_map = build_id_to_name(data.get("scenarios", []))
    weather_map = build_id_to_name(data.get("weather", []))

    img_info = {img["id"]: img for img in data.get("images", [])}

    cat_map = {}
    for category in data.get("categories", []):
        class_id = CLASS_NAME_TO_ID.get(category["name"])
        if class_id is not None:
            cat_map[category["id"]] = class_id

    ann_by_img = defaultdict(list)
    for ann in data.get("annotations", []):
        ann_by_img[ann["image_id"]].append(ann)

    missing_images = []
    malformed_total = 0
    skipped_conditions = defaultdict(int)

    for img_id, img_meta in img_info.items():
        scenario_name = resolve_condition_name(img_meta.get("scenario"), scenario_map)
        weather_name = resolve_condition_name(img_meta.get("weather"), weather_map)
        dataset_name = f"{scenario_name}_{weather_name}"
        dataset_dir = dataset_dirs.get(dataset_name)
        if dataset_dir is None:
            skipped_conditions[dataset_name] += 1
            continue

        width = int(img_meta.get("width", img_w_default))
        height = int(img_meta.get("height", img_h_default))
        image_filename = extract_image_filename(img_meta, img_id)
        image_stem = Path(image_filename).stem

        lines, malformed = yolo_lines_for_image(
            ann_by_img.get(img_id, []),
            cat_map,
            width,
            height,
            angle_unit,
            bbox_is_center,
        )
        malformed_total += malformed

        label_path = dataset_dir / "labels" / output_split / f"{image_stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        image_path = resolve_image_path(json_path, split_name, image_filename, images_root)
        if image_path is None:
            missing_images.append(image_filename)
            continue

        target_image_path = dataset_dir / "images" / output_split / image_filename
        shutil.copy2(image_path, target_image_path)

        stats[dataset_name][f"{output_split}_images"] += 1
        if lines:
            stats[dataset_name][f"{output_split}_labelled"] += 1
        else:
            stats[dataset_name][f"{output_split}_empty"] += 1

    if skipped_conditions:
        print(f"[WARN] {json_path.name}: skipped unknown conditions {dict(skipped_conditions)}")
    if missing_images:
        preview = ", ".join(missing_images[:8])
        extra = "" if len(missing_images) <= 8 else f" ... (+{len(missing_images) - 8} more)"
        print(
            f"[WARN] {json_path.name}: missing {len(missing_images)} source images: {preview}{extra}"
        )
    if malformed_total:
        print(f"[WARN] {json_path.name}: skipped {malformed_total} malformed bboxes")


def convert(
    src: Path,
    out: Path,
    images_root: Path | None,
    img_w: int,
    img_h: int,
    angle_unit: str,
    bbox_is_center: bool,
) -> None:
    json_files = find_split_jsons(src)
    if not json_files:
        print(f"[ERROR] No train/val/test JSON files found under: {src}")
        return

    dataset_names = ["day_sunny", "day_rainy", "night_sunny", "night_rainy"]
    dataset_dirs = ensure_dataset_dirs(out, dataset_names)
    for dataset_dir in dataset_dirs.values():
        write_dataset_yaml(dataset_dir)

    stats = defaultdict(lambda: defaultdict(int))

    print(f"Found split JSON files: {[path.name for path in json_files]}")
    for json_path in json_files:
        print(f"Converting {json_path}")
        convert_split(
            json_path=json_path,
            dataset_dirs=dataset_dirs,
            images_root=images_root,
            img_w_default=img_w,
            img_h_default=img_h,
            angle_unit=angle_unit,
            bbox_is_center=bbox_is_center,
            stats=stats,
        )

    print("\nDone. Dataset summary:")
    for dataset_name in dataset_names:
        dataset_stats = stats[dataset_name]
        print(
            f"  {dataset_name}: "
            f"train={dataset_stats['train_images']}, "
            f"val={dataset_stats['val_images']}, "
            f"test={dataset_stats['test_images']} images | "
            f"labelled={dataset_stats['train_labelled'] + dataset_stats['val_labelled'] + dataset_stats['test_labelled']}, "
            f"empty={dataset_stats['train_empty'] + dataset_stats['val_empty'] + dataset_stats['test_empty']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create 4 HIT-UAV YOLO OBB datasets split by scenario and weather",
    )
    parser.add_argument(
        "--src",
        required=True,
        help="path to the HIT-UAV rotate_json folder (or its annotations subfolder)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="output root for the four generated datasets",
    )
    parser.add_argument(
        "--images-root",
        help="optional override for the folder that contains source images",
    )
    parser.add_argument(
        "--img-w",
        type=int,
        default=640,
        help="default image width if missing in JSON metadata",
    )
    parser.add_argument(
        "--img-h",
        type=int,
        default=512,
        help="default image height if missing in JSON metadata",
    )
    parser.add_argument(
        "--angle-unit",
        choices=["radians", "degrees"],
        default="radians",
        help="unit of the bbox angle field",
    )
    parser.add_argument(
        "--bbox-is-center",
        action="store_true",
        help="treat JSON bbox as [cx,cy,w,h,theta] instead of [xmin,ymin,w,h,theta]",
    )
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    images_root = Path(args.images_root) if args.images_root else None

    if not src.exists():
        print(f"[ERROR] source path does not exist: {src}")
        return
    if images_root is not None and not images_root.exists():
        print(f"[ERROR] images root does not exist: {images_root}")
        return

    out.mkdir(parents=True, exist_ok=True)
    convert(
        src=src,
        out=out,
        images_root=images_root,
        img_w=args.img_w,
        img_h=args.img_h,
        angle_unit=args.angle_unit,
        bbox_is_center=args.bbox_is_center,
    )


if __name__ == "__main__":
    main()
