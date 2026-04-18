import os

import numpy as np
import torch
from ultralytics import YOLO


def main():
    test_path = "data/hit-uav/images/test"
    test_files = os.listdir(test_path)
    test = [os.path.join(test_path, name) for name in test_files]
    model = YOLO("runs/obb/phase2/weights/best.pt")
    for file, path in zip(test_files, test):
        result = model.predict(path, classes=[0, 1, 2, 3], half=True, imgsz=640)[0]
        np_obb = (
            torch.cat([result.obb.cls.unsqueeze(1), result.obb.xywhr], dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        with open(os.path.join("Inference", file[:-4] + ".npy"), mode="wb") as f:
            np.save(f, np_obb)
        result.save(os.path.join("Inference", file))


if __name__ == "__main__":
    main()
