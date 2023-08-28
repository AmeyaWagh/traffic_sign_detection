from typing import Any, Optional
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn.functional as F

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class TrafficLightDetector(pl.LightningModule):
    def __init__(self, n_classes: int=6) -> None:
        super().__init__()
        self._num_classes = n_classes
        self._model = self._build_model()
        
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        img, label = batch
        pred_cls, pred_box = self._model(img)
        loss = F.mse_loss(pred_cls, label)        
        self.log("train_loss",loss.item(), prog_bar=False)
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        img, label = batch
        pred_cls, pred_box = self._model(img)
        loss = F.mse_loss(pred_cls, label)
        self.log("val_loss",loss.item(), prog_bar=True)
        return loss
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        img, label = batch
        pred = self._model(img)
        label = torch.argmax(pred)
        return {'img':img, "label":label}
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def _build_model(self):
        _model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = _model.roi_heads.box_predictor.cls_score.in_features
        _model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self._num_classes)
        return _model