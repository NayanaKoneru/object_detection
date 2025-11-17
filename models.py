import torch.nn as nn
from dataclasses import dataclass
from transformers import ViTModel
from typing import Optional

@dataclass
class ObjectDetectionConfig:
    """Configuration for object detection training"""
    model_name: str = "google/vit-base-patch16-224"
    num_labels: int = 91  # COCO has 91 categories (including background)
    learning_rate: float = 2e-5
    batch_size: int = 8
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    output_dir: str = "./checkpoints"
    logging_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    gradient_accumulation_steps: int = 4
    fp16: bool = True  # Use mixed precision training if available
    max_train_samples: Optional[int] = None  # Limit training samples for testing
    max_eval_samples: Optional[int] = None


class ViTObjectDetectionModel(nn.Module):
    """
    Vision Transformer adapted for object detection.
    This is a simplified version - for production, consider using DETR or similar.
    """
    def __init__(self, config: ObjectDetectionConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained ViT model
        self.vit = ViTModel.from_pretrained(config.model_name)
        
        # Detection head for bounding box regression and classification
        hidden_size = self.vit.config.hidden_size
        
        # Classification head (predict object categories)
        self.classifier = nn.Linear(hidden_size, config.num_labels)
        
        # Bounding box regression head (predict x, y, w, h)
        self.bbox_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 4)  # 4 coordinates: x, y, width, height
        )
        
    def forward(self, pixel_values, labels=None, bboxes=None):
        # Get ViT features
        outputs = self.vit(pixel_values=pixel_values)
        
        # Use [CLS] token for predictions
        cls_output = outputs.last_hidden_state[:, 0]
        
        # Predict class labels
        logits = self.classifier(cls_output)
        
        # Predict bounding boxes
        bbox_preds = self.bbox_predictor(cls_output)
        
        loss = None
        if labels is not None and bboxes is not None:
            # Classification loss
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(logits, labels)
            
            # Bounding box regression loss (L1 loss)
            bbox_loss_fct = nn.SmoothL1Loss()
            bbox_loss = bbox_loss_fct(bbox_preds, bboxes)
            
            # Combined loss
            loss = classification_loss + bbox_loss
        
        return {
            'loss': loss,
            'logits': logits,
            'bbox_predictions': bbox_preds,
            'hidden_states': outputs.hidden_states if outputs.hidden_states is not None else None
        }