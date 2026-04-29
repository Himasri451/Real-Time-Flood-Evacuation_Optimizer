"""
training.py - Model Training & Architecture
Flood Evacuation Route Optimizer
 
Contains:
  1. YOLOv8 custom training pipeline
  2. CNN (EfficientNet-B0) architecture + training
  3. Dataset preparation helpers
"""
 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from config import (
    YOLOV8_MODEL_PATH, CNN_MODEL_PATH, BASE_MODEL,
    DATASET_YAML, TRAIN_EPOCHS, TRAIN_BATCH_SIZE,
    TRAIN_IMAGE_SIZE, CNN_TRAIN_EPOCHS, TRAIN_LR,
    CNN_NUM_CLASSES, CNN_IMAGE_SIZE, CNN_RISK_LABELS
)
from logger import logger
 
 
# ════════════════════════════════════════
# PART 1: YOLOV8 TRAINING
# ════════════════════════════════════════
 
class YOLOv8Trainer:
    """
    Trains a custom YOLOv8 model on flood detection dataset.
 
    Dataset must be prepared in Roboflow and exported in YOLOv8 format.
    Expected data.yaml:
        train: flood_dataset/images/train
        val:   flood_dataset/images/val
        nc: 7
        names: [flood_water, submerged_vehicle, stranded_person,
                road_debris, water_marker_safe, water_marker_danger, sos_person]
    """
 
    def __init__(self, base_model: str = BASE_MODEL):
        from ultralytics import YOLO
        self.model = YOLO(base_model)
        logger.info(f"YOLOv8 trainer initialized with base: {base_model}")
 
    def train(self, data_yaml: str = DATASET_YAML):
        """Train YOLOv8 on flood detection dataset."""
        logger.info("🚀 Starting YOLOv8 training...")
 
        results = self.model.train(
            data     = data_yaml,
            epochs   = TRAIN_EPOCHS,
            imgsz    = TRAIN_IMAGE_SIZE,
            batch    = TRAIN_BATCH_SIZE,
            name     = "flood_yolov8",
            patience = 20,
            device   = 0 if torch.cuda.is_available() else "cpu",
            workers  = 4,
            augment  = True,
            # Augmentation params
            hsv_h    = 0.015,
            hsv_s    = 0.7,
            hsv_v    = 0.4,
            fliplr   = 0.5,
            mosaic   = 1.0,
            mixup    = 0.1,
        )
 
        best_path = "runs/detect/flood_yolov8/weights/best.pt"
        logger.info(f"✅ YOLOv8 training complete. Best: {best_path}")
        return results
 
    def validate(self):
        """Validate trained model and print metrics."""
        metrics = self.model.val()
        logger.info(f"mAP50: {metrics.box.map50:.4f} | mAP50-95: {metrics.box.map:.4f}")
        return metrics
 
    def export_model(self, format: str = "onnx"):
        """Export model for deployment (onnx, torchscript, tflite)."""
        self.model.export(format=format)
        logger.info(f"✅ Model exported as {format}")
 
 
# ════════════════════════════════════════
# PART 2: CNN ARCHITECTURE
# ════════════════════════════════════════
 
class FloodRiskCNN(nn.Module):
    """
    CNN Risk Classifier Architecture.
 
    Backbone : EfficientNet-B0 (pretrained on ImageNet)
    Head     : Custom 4-class classifier
    Classes  : SAFE | LOW | MEDIUM | HIGH
 
    Input  : (batch, 3, 224, 224) normalized image tensor
    Output : (batch, 4) logits
    """
 
    def __init__(self, num_classes: int = CNN_NUM_CLASSES, pretrained: bool = True):
        super(FloodRiskCNN, self).__init__()
 
        # ── Backbone ──────────────────────
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
 
        # Freeze early layers (transfer learning)
        for i, (name, param) in enumerate(self.backbone.named_parameters()):
            if i < 100:
                param.requires_grad = False
 
        # ── Custom Classifier Head ─────────
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
 
        self.class_names = CNN_RISK_LABELS
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
 
    def predict_risk(self, tensor: torch.Tensor) -> dict:
        """Run inference on a preprocessed tensor. Returns risk dict."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(tensor)
            probs  = F.softmax(logits, dim=1)[0]
            idx    = probs.argmax().item()
        return {
            "risk_level":  self.class_names[idx],
            "confidence":  round(probs[idx].item(), 4),
            "all_probs": {
                name: round(probs[i].item(), 4)
                for i, name in enumerate(self.class_names)
            }
        }
 
 
# ════════════════════════════════════════
# PART 3: CNN TRAINING PIPELINE
# ════════════════════════════════════════
 
class CNNTrainer:
    """Handles CNN training, validation, and saving."""
 
    def __init__(self, model: FloodRiskCNN):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = model.to(self.device)
        logger.info(f"CNN trainer initialized. Device: {self.device}")
 
    def train(self, data_dir: str, epochs: int = CNN_TRAIN_EPOCHS, lr: float = TRAIN_LR):
        """
        Train CNN on flood zone risk dataset.
 
        Dataset structure:
            data_dir/
            ├── train/
            │   ├── SAFE/
            │   ├── LOW/
            │   ├── MEDIUM/
            │   └── HIGH/
            └── val/
                ├── SAFE/  ...
        """
        train_loader, val_loader = self._get_dataloaders(data_dir)
 
        optimizer  = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr
        )
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion  = nn.CrossEntropyLoss()
        best_val   = 0.0
 
        logger.info(f"🚀 CNN training started | Epochs: {epochs} | LR: {lr}")
 
        for epoch in range(epochs):
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, criterion)
            val_acc               = self._validate(val_loader)
 
            logger.info(
                f"Epoch [{epoch+1:02d}/{epochs}] "
                f"Loss: {train_loss:.4f} | "
                f"Train: {train_acc:.4f} | "
                f"Val: {val_acc:.4f}"
            )
 
            if val_acc > best_val:
                best_val = val_acc
                torch.save(self.model.state_dict(), CNN_MODEL_PATH)
                logger.info(f"   ✅ Best CNN saved (Val: {best_val:.4f})")
 
            scheduler.step()
 
        logger.info(f"🏆 CNN training complete. Best Val: {best_val:.4f}")
 
    def _train_epoch(self, loader, optimizer, criterion):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)
        return total_loss / len(loader), correct / total
 
    def _validate(self, loader) -> float:
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                preds   = self.model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        return correct / total if total > 0 else 0.0
 
    def _get_dataloaders(self, data_dir: str):
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(CNN_IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Resize((CNN_IMAGE_SIZE, CNN_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        train_ds = ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
        val_ds   = ImageFolder(os.path.join(data_dir, "val"),   transform=val_transform)
        return (
            DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True,  num_workers=2),
            DataLoader(val_ds,   batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=2)
        )
 
    def load_best(self):
        self.model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=self.device))
        logger.info("✅ Best CNN model loaded.")
