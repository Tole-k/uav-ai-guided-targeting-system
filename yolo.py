from ultralytics import YOLO

model = YOLO("yolo26x-obb.pt")

model.train(data="data/hit-uav/dataset.yaml")