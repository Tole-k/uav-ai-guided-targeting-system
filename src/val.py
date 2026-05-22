from ultralytics import YOLO


def main():
    model = YOLO("runs/obb/phase2/weights/best.pt")
    # model.val(data="day_rainy/dataset.yaml",classes=[0, 1, 2, 3])
    model.val(data="day_sunny/dataset.yaml",classes=[0, 1, 2, 3])
    model.val(data="night_sunny/dataset.yaml",classes=[0, 1, 2, 3])
    # model.val(data="night_rainy/dataset.yaml",classes=[0, 1, 2, 3])

if __name__ == "__main__":
    main()