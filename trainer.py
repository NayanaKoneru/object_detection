import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    ViTModel,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import evaluate
from models import ViTObjectDetectionModel, ObjectDetectionConfig

class COCODataset(torch.utils.data.Dataset):
    """Custom Dataset for COCO object detection"""
    
    def __init__(self, split='train', processor=None, max_samples=None):
        print(f"Loading COCO dataset ({split} split)...")
        self.dataset = load_dataset("detection-datasets/coco", split=split)
        
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
        self.processor = processor
        print(f"Dataset loaded: {len(self.dataset)} samples \n")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process image
        if self.processor:
            encoding = self.processor(images=image, return_tensors="pt")
            pixel_values = encoding['pixel_values'].squeeze(0)
        else:
            pixel_values = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Get first object's annotation (simplified - in production, handle multiple objects)
        objects = item['objects']
        if isinstance(objects, dict) and 'bbox' in objects and len(objects['bbox']) > 0:
            # Get first object
            label = objects['category'][0] if len(objects['category']) > 0 else 0
            bbox = objects['bbox'][0] if len(objects['bbox']) > 0 else [0, 0, 0, 0]
            
            # Normalize bbox to [0, 1] range
            img_width, img_height = image.size
            bbox_normalized = [
                bbox[0] / img_width,
                bbox[1] / img_height,
                bbox[2] / img_width,
                bbox[3] / img_height
            ]
        else:
            label = 0
            bbox_normalized = [0, 0, 0, 0]
        
        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(label, dtype=torch.long),
            'bboxes': torch.tensor(bbox_normalized, dtype=torch.float)
        }


def collate_fn(batch):
    """Custom collate function for batching"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    bboxes = torch.stack([item['bboxes'] for item in batch])
    
    return {
        'pixel_values': pixel_values,
        'labels': labels,
        'bboxes': bboxes
    }


class ObjectDetectionTrainer:
    """Trainer class for object detection with ViT"""
    
    def __init__(self, config: ObjectDetectionConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load processor
        print(f"Loading ViT processor from {config.model_name}...")
        self.processor = ViTImageProcessor.from_pretrained(config.model_name)
        
        # Initialize model
        print("Initializing model...")
        self.model = ViTObjectDetectionModel(config).to(self.device)
        
        # Load datasets
        self.train_dataset = COCODataset(
            split='train',
            processor=self.processor,
            max_samples=config.max_train_samples
        )
        self.val_dataset = COCODataset(
            split='val',
            processor=self.processor,
            max_samples=config.max_eval_samples
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        total_steps = len(self.train_loader) * config.num_epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=0.1
        )
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            bboxes = batch['bboxes'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                pixel_values=pixel_values,
                labels=labels,
                bboxes=bboxes
            )
            
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.6f}"
            })
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                print(f"\nStep {self.global_step}: Loss = {loss.item():.4f}")
            
            # Save checkpoint
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint(f"checkpoint-{self.global_step}")
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                bboxes = batch['bboxes'].to(self.device)
                
                outputs = self.model(
                    pixel_values=pixel_values,
                    labels=labels,
                    bboxes=bboxes
                )
                
                total_loss += outputs['loss'].item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self):
        """Full training loop"""
        print(f"\n{'='*50}")
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"{'='*50}\n")
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"\nEpoch {epoch + 1} - Average train loss: {train_loss:.4f}")
            
            # Evaluate
            val_loss = self.evaluate()
            print(f"Epoch {epoch + 1} - Validation loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model")
                print(f"New best model saved! Validation loss: {val_loss:.4f}")
            
            print(f"{'-'*50}\n")
        
        print("Training completed!")
    
    def save_checkpoint(self, name):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.config.output_dir, name)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        torch.save(self.model.state_dict(), os.path.join(checkpoint_path, "model.pth"))
        
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(os.path.join(checkpoint_path, 'pytorch_model.bin'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Checkpoint loaded from {checkpoint_path}")


def main():
    """Main training function"""
    # Configuration
    config = ObjectDetectionConfig(
        model_name="google/vit-base-patch16-224",
        num_labels=91,
        learning_rate=2e-5,
        batch_size=8,
        num_epochs=1,
        output_dir="./checkpoints",
        max_train_samples=10,  # Limit for testing - remove for full training
        max_eval_samples=2,
        gradient_accumulation_steps=4,
        fp16=torch.cuda.is_available()
    )
    
    # Initialize trainer
    trainer = ObjectDetectionTrainer(config)
    
    # Start training
    trainer.train()
    
    print("\nTraining pipeline completed successfully!")
    print(f"Best model saved at: {os.path.join(config.output_dir, 'best_model')}")


if __name__ == "__main__":
    main()
