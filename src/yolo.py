import torch
from ultralytics import YOLO

DATA = "dataset.yaml"
BACKBONE_FREEZE = 10

augmentation_params = {
    "fliplr": 0.5,  # horizontal flip — people/vehicles are symmetric
    "degrees": 10.0,  # small rotation — IR cameras are sometimes tilted
    "scale": 0.6,  # aggressive scale variation — helps with small people
    "translate": 0.2,
    # Color — be careful, IR is single-channel
    "hsv_h": 0.0,  # NO hue shift — meaningless for grayscale
    "hsv_s": 0.0,  # NO saturation — you have none
    "hsv_v": 0.4,  # YES brightness — simulates different thermal conditions
    # Mixing — very useful for small datasets
    "mosaic": 1.0,  # always on — critical for 2k images
    "copy_paste": 0.3,  # paste instances across images — great for rare poses
    "mixup": 0.05,  # light mixup only
}


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )

    model = YOLO("yolo26x-obb.pt")
    model.train(
        data=DATA,
        epochs=60,
        device=device,
        freeze=BACKBONE_FREEZE,
        batch=-1,
        compile=True,
        lr0=0.01,
        lrf=0.01,
        name="phase1",
        **augmentation_params,
    )

    model.train(
        data=DATA,
        epochs=140,
        device=device,
        batch=-1,
        compile=True,
        lr0=0.001,
        lrf=0.01,
        name="phase2",
        **augmentation_params,
    )

    model.val(classes=[0, 1, 2, 3])


if __name__ == "__main__":
    main()
