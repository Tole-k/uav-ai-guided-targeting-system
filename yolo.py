import torch
from ultralytics import YOLO


def main():
    model = YOLO("yolo26x.pt")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    model.train(data="data/hit-uav/dataset.yaml", epochs=200, device=device)

if __name__=="__main__":
    main()