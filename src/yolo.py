import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import E2ELoss, v8OBBLoss, FocalLoss

DATA = "dataset.yaml"
BACKBONE_FREEZE = 10

# Provide one proportion per class in dataset.yaml order:
# Person, Car, Bicycle, OtherVehicle, DontCare

# MACRO
CLASS_PROPORTIONS = [12312,7311, 4980, 148, 148]


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


def proportions_to_pos_weight(proportions):
    probs = torch.tensor(proportions, dtype=torch.float32)
    if probs.ndim != 1:
        raise ValueError("CLASS_PROPORTIONS must be a 1D list of class frequencies.")
    if torch.any(probs <= 0):
        raise ValueError("CLASS_PROPORTIONS entries must be strictly positive.")

    # Inverse-frequency weights normalized to mean 1 to avoid changing
    # the overall classification-loss scale too aggressively.
    return (1-BETA)/(1-BETA**probs)


class WeightedOBBLoss(v8OBBLoss):
    def __init__(self, model, class_pos_weight=None, tal_topk=10, tal_topk2=None):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        if class_pos_weight is not None:
            # The weighting happens here:
            # v8OBBLoss uses self.bce for the classification term, so by replacing
            # it with BCEWithLogitsLoss(pos_weight=...), positive examples for rare
            # classes contribute more to the class-loss part.
            self.bce = nn.BCEWithLogitsLoss(
                pos_weight=class_pos_weight.to(self.device),
                reduction="none",
            )

class Focalloss(v8OBBLoss):
    def __init__(self, model):
        super().__init__(model)
        
        self.bce = FocalLoss(2)


class WeightedOBBE2ELoss(E2ELoss):
    def __init__(self, model, class_pos_weight=None):
        def weighted_loss_fn(model, tal_topk=10, tal_topk2=None):
            return WeightedOBBLoss(
                model,
                class_pos_weight=class_pos_weight,
                tal_topk=tal_topk,
                tal_topk2=tal_topk2,
            )
        def focal_loss(model):
            return Focalloss(model)

        super().__init__(model, loss_fn=focal_loss)


class WeightedOBBModel(OBBModel):
    #def __init__(self, cfg="yolo26n-obb.yaml", ch=3, nc=None, verbose=True, class_pos_weight=None):
        
        #super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
       
        #self._class_pos_weight = class_pos_weight

    def init_criterion(self):
        weights = proportions_to_pos_weight(CLASS_PROPORTIONS)
        if getattr(self, "end2end", False):
            return WeightedOBBE2ELoss(self, class_pos_weight=weights)
        return WeightedOBBLoss(self, class_pos_weight=weights)


class WeightedOBBTrainer(OBBTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        #class_pos_weight = proportions_to_pos_weight(self.args.class_proportions)
        model = WeightedOBBModel(
            cfg,
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )
        # if weights:
        #     model.load(weights)
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
    model.train(
        data=DATA,
        trainer=WeightedOBBTrainer,
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
        name="phase1",
        **augmentation_params,
    )

    model = YOLO("runs/obb/phase1/weights/best.pt")
    # model = YOLO("runs/obb/phase2/weights/last.pt")
    model.train(
        data=DATA,
        trainer=WeightedOBBTrainer,
        imgsz=640,
        rect=True,
        epochs=140,
        device=device,
        batch=-1,
        lr0=0.001,
        lrf=0.01,
        name="final",
        exist_ok=True,
        **augmentation_params,
    )
    model = YOLO("runs/obb/phase2/weights/best.pt")
    model.val(classes=[0, 1, 2, 3])


if __name__ == "__main__":
    main()
