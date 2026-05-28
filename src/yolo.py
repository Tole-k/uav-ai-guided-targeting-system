import torch
from ultralytics import YOLO
from ultralytics.utils.loss import E2ELoss

from custom_loss import v8OBBLoss

DATA = "dataset.yaml"
BACKBONE_FREEZE = 10

# Provide one proportion per class in dataset.yaml order:
# Person, Car, Bicycle, OtherVehicle, DontCare

# MACRO
CLASS_PROPORTIONS = [12312, 7311, 4980, 148, 148]


# SOTA CLASS WEIGHTING
BETA = 0.999

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


def use_varifocal_obb_loss(model):
    obb_model = model.model
    obb_model.criterion = (
        E2ELoss(obb_model, v8OBBLoss)
        if getattr(obb_model, "end2end", False)
        else v8OBBLoss(obb_model)
    )
    return model


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )
    print(f"{device=}")

    model = YOLO("yolo26x-obb.pt")
    use_varifocal_obb_loss(model)
    model.train(
        data=DATA,
        imgsz=640,
        rect=True,
        epochs=60,
        device=device,
        freeze=BACKBONE_FREEZE,
        batch=-1,
        compile=True,
        lr0=0.01,
        lrf=0.01,
        exist_ok=True,
        name="phase1-verifocal",
        **augmentation_params,
    )

    model = YOLO("runs/obb/phase1-verifocal/weights/best.pt")
    use_varifocal_obb_loss(model)
    model.train(
        data=DATA,
        imgsz=640,
        rect=True,
        epochs=140,
        device=device,
        batch=-1,
        lr0=0.001,
        lrf=0.01,
        name="phase2-verifocal",
        exist_ok=True,
        **augmentation_params,
    )
    model = YOLO("runs/obb/phase2-verifocal/weights/best.pt")
    model.val(classes=[0, 1, 2, 3], name="val-verifocal")


if __name__ == "__main__":
    main()
