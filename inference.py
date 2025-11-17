import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
from typing import Optional, Union, List
import os
from pathlib import Path
from models import ViTObjectDetectionModel, ObjectDetectionConfig


# COCO category names (91 categories including background)
COCO_CATEGORIES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
    'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush'
]

class ObjectDetectionInference:
    """Inference class for object detection with trained ViT model"""
    
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        """
        Initialize inference model
        
        Args:
            checkpoint_path: Path to the saved checkpoint directory
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
        """
        self.checkpoint_path = checkpoint_path
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading model on device: {self.device}")
        
        # Load checkpoint
        self._load_model()
        
        print("Model loaded successfully!")
    
    def _load_model(self):
        """Load model from checkpoint"""
        checkpoint_file = os.path.join(self.checkpoint_path, 'model.pth')
        
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=True)
        
        # Get config
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        else:
            # Use default config if not saved
            self.config = ObjectDetectionConfig()
        
        # Load processor
        print(f"Loading image processor from {self.config.model_name}...")
        self.processor = ViTImageProcessor.from_pretrained(self.config.model_name)
        
        # Initialize model
        print("Initializing model architecture...")
        self.model = ViTObjectDetectionModel(self.config)
        
        # Load weights
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        # Get additional info
        if 'best_val_loss' in checkpoint:
            print(f"Model validation loss: {checkpoint['best_val_loss']:.4f}")
        if 'global_step' in checkpoint:
            print(f"Training steps: {checkpoint['global_step']}")
    
    def preprocess_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for inference
        
        Args:
            image: Path to image file or PIL Image object
            
        Returns:
            Preprocessed image tensor
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process with ViT processor
        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding['pixel_values'].to(self.device)
        
        return pixel_values, image
    
    def predict(self, image: Union[str, Path, Image.Image], confidence_threshold: float = 0.5):
        """
        Run inference on a single image
        
        Args:
            image: Path to image file or PIL Image object
            confidence_threshold: Minimum confidence score for predictions
            
        Returns:
            Dictionary containing predictions
        """
        # Preprocess image
        pixel_values, original_image = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(pixel_values)
        
        # Get predictions
        logits = outputs['logits']
        bbox_preds = outputs['bbox_predictions']
        
        # Get predicted class and confidence
        probs = torch.softmax(logits, dim=-1)
        confidence, predicted_class = torch.max(probs, dim=-1)
        
        # Convert to numpy
        predicted_class = predicted_class.cpu().item()
        confidence = confidence.cpu().item()
        bbox = bbox_preds.cpu().numpy()[0]  # [x, y, w, h] in normalized coords
        
        # Get image dimensions
        img_width, img_height = original_image.size
        
        # Denormalize bbox coordinates
        bbox_denorm = [
            max(0, bbox[0] * img_width),
            max(0, bbox[1] * img_height),
            min(bbox[2] * img_width, img_width),
            min(bbox[3] * img_height, img_height)
        ]
        
        # Get category name
        category_name = COCO_CATEGORIES[predicted_class] if predicted_class < len(COCO_CATEGORIES) else f"class_{predicted_class}"
        
        result = {
            'category_id': predicted_class,
            'category_name': category_name,
            'confidence': confidence,
            'bbox': bbox_denorm,  # [x, y, width, height] in pixel coordinates
            'bbox_normalized': bbox.tolist(),  # [x, y, width, height] in [0, 1] range
            'image_size': (img_width, img_height)
        }
        
        return result, original_image
    
    def predict_batch(self, images: List[Union[str, Path, Image.Image]], confidence_threshold: float = 0.5):
        """
        Run inference on multiple images
        
        Args:
            images: List of image paths or PIL Image objects
            confidence_threshold: Minimum confidence score for predictions
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in images:
            result, _ = self.predict(image, confidence_threshold)
            results.append(result)
        return results
    
    def visualize_prediction(self, image: Union[str, Path, Image.Image], 
                            save_path: Optional[str] = None,
                            figsize: tuple = (12, 8),
                            show: bool = True):
        """
        Visualize prediction on image
        
        Args:
            image: Path to image file or PIL Image object
            save_path: Optional path to save visualization
            figsize: Figure size
            show: Whether to display the plot
        """
        # Get prediction
        result, original_image = self.predict(image)
        
        # Create figure
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(original_image)
        
        # Draw bounding box
        bbox = result['bbox']
        x, y, w, h = bbox
        
        # Create rectangle
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=3,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        label_text = f"{result['category_name']}: {result['confidence']:.2f}"
        ax.text(
            x, y - 10,
            label_text,
            color='white',
            fontsize=12,
            weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.7)
        )
        
        # Set title
        ax.set_title(f"Object Detection - {result['category_name']}", fontsize=14, weight='bold')
        ax.axis('off')
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig, result
    
    def print_prediction(self, result: dict):
        """Print prediction in a formatted way"""
        print("\n" + "="*60)
        print("OBJECT DETECTION PREDICTION")
        print("="*60)
        print(f"Category: {result['category_name']}")
        print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print(f"Bounding Box (pixels): x={result['bbox'][0]:.1f}, y={result['bbox'][1]:.1f}, "
              f"w={result['bbox'][2]:.1f}, h={result['bbox'][3]:.1f}")
        print(f"Image Size: {result['image_size'][0]} x {result['image_size'][1]}")
        print("="*60 + "\n")


def main():
    """Main inference function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Object Detection Inference')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model',
                        help='Path to checkpoint directory')
    parser.add_argument('--image', type=str, required=False,
                        help='Path to input image')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output visualization')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='Device to run inference on')
    parser.add_argument('--use-validation', action='store_true',
                        help='Use a sample from validation dataset')
    
    args = parser.parse_args()
    
    # Initialize inference model
    print("Initializing inference model...")
    inferencer = ObjectDetectionInference(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    if args.use_validation or args.image is None:
        # Use validation dataset sample
        print("\nLoading sample from validation dataset...")
        from datasets import load_dataset
        dataset = load_dataset("detection-datasets/coco", split='validation')
        sample_image = dataset[0]['image']
        
        print("Running inference on validation sample...")
        result, original_image = inferencer.predict(sample_image)
        
    else:
        # Use provided image
        print(f"\nRunning inference on: {args.image}")
        result, original_image = inferencer.predict(args.image)
    
    # Print results
    inferencer.print_prediction(result)
    
    # Visualize
    output_path = args.output if args.output else 'inference_result.png'
    inferencer.visualize_prediction(
        original_image if args.use_validation or args.image is None else args.image,
        save_path=output_path,
        show=True
    )
    
    print(f"\n✓ Inference completed successfully!")


if __name__ == "__main__":
    main()
