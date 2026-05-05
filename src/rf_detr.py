import torch
from ultralytics import RTDETR

DATA = "dataset.yaml"

augmentation_params = {
    "fliplr": 0.5,
    "degrees": 10.0,
    "scale": 0.6,
    "translate": 0.2,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.4,
    "mosaic": 1.0,
    "copy_paste": 0.3,
    "mixup": 0.05,
}


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )

    model = RTDETR("rtdetr-x.pt")
    model.train(
        data=DATA,
        imgsz=640,
        rect=True,
        epochs=100,
        device=device,
        batch=-1,
        lr0=0.001,
        lrf=0.01,
        exist_ok=True,
        name="rf_detr",
        **augmentation_params,
    )

    model = RTDETR("runs/detect/rf_detr/weights/best.pt")
    model.val(classes=[0, 1, 2, 3])


if __name__ == "__main__":
    main()
