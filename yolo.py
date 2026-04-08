import torch
from ultralytics import YOLO

DATA = "dataset.yaml"
BACKBONE_FREEZE = 10


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

    model = YOLO("yolo26x.pt")
    model.train(
        data=DATA,
        epochs=60,
        device=device,
        freeze=BACKBONE_FREEZE,
        lr0=0.01,
        lrf=0.01,
        project="runs/train",
        name="phase1",
    )

    model = YOLO("runs/train/phase1/weights/last.pt")
    model.train(
        data=DATA,
        epochs=140,
        device=device,
        lr0=0.001,
        lrf=0.01,
        project="runs/train",
        name="phase2",
    )


if __name__ == "__main__":
    main()