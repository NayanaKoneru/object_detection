from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

def load_coco_dataset(split='train', streaming=False):
    """
    Load COCO dataset from Hugging Face detection-datasets
    
    Args:
        split: 'train' or 'validation'
        streaming: If True, load dataset in streaming mode (useful for large datasets)
    
    Returns:
        dataset: Hugging Face dataset object
    """
    print(f"Loading COCO dataset ({split} split)...")
    dataset = load_dataset(
        "detection-datasets/coco",
        split=split,
        streaming=streaming
    )
    print(f"Dataset loaded successfully!")
    return dataset

def visualize_detection(image, annotations, idx=0, figsize=(12, 8)):
    """
    Visualize an image with bounding boxes and labels
    
    Args:
        image: PIL Image or numpy array
        annotations: Dictionary containing 'objects' with bbox, category, etc.
        idx: Index of the image (for title)
        figsize: Figure size tuple
    """
    fig, ax = plt.subplots(1, figsize=figsize)
    
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    ax.imshow(image_np)
    
    # Get objects from annotations
    objects = annotations.get('objects', annotations)
    
    # Handle different annotation formats
    if isinstance(objects, dict):
        bboxes = objects.get('bbox', [])
        categories = objects.get('category', [])
        areas = objects.get('area', [])
    else:
        bboxes = []
        categories = []
        areas = []
    
    # Draw bounding boxes
    colors = plt.cm.rainbow(np.linspace(0, 1, len(bboxes)))
    
    for i, (bbox, category, color) in enumerate(zip(bboxes, categories, colors)):
        # COCO format: [x_min, y_min, width, height]
        x, y, w, h = bbox
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        label = f"Class {category}"
        ax.text(
            x, y - 5,
            label,
            color='white',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7)
        )
    
    ax.set_title(f"Image {idx} - {len(bboxes)} objects detected")
    ax.axis('off')
    plt.tight_layout()
    return fig

def visualize_samples(dataset, num_samples=3, start_idx=0):
    """
    Visualize multiple samples from the dataset
    
    Args:
        dataset: Hugging Face dataset
        num_samples: Number of samples to visualize
        start_idx: Starting index
    """
    # Handle streaming datasets
    if hasattr(dataset, '__iter__') and not hasattr(dataset, '__getitem__'):
        # Streaming dataset
        samples = []
        for i, sample in enumerate(dataset):
            if i >= start_idx + num_samples:
                break
            if i >= start_idx:
                samples.append(sample)
    else:
        # Regular dataset
        samples = [dataset[i] for i in range(start_idx, min(start_idx + num_samples, len(dataset)))]
    
    for i, sample in enumerate(samples):
        image = sample['image']
        annotations = sample
        
        print(f"\nSample {start_idx + i}:")
        print(f"  Image size: {image.size if isinstance(image, Image.Image) else image.shape}")
        
        if 'objects' in sample:
            objects = sample['objects']
            if isinstance(objects, dict) and 'bbox' in objects:
                print(f"  Number of objects: {len(objects['bbox'])}")
                print(f"  Categories: {objects.get('category', [])[:10]}")  # Show first 10
        
        visualize_detection(image, annotations, idx=start_idx + i)
    
    plt.show()

if __name__ == "__main__":
    # Load a small portion of the dataset for visualization
    # Use streaming=True for large datasets to avoid downloading everything
    print("Loading COCO dataset from Hugging Face...")
    print("Note: First time may take a while to download.\n")
    
    # Load validation split (smaller than train)
    dataset = load_coco_dataset(split='validation', streaming=False)
    
    # Visualize first 3 samples
    print("\nVisualizing samples...")
    visualize_samples(dataset, num_samples=3, start_idx=0)
