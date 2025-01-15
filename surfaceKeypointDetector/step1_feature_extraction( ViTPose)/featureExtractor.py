import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from transformers import AutoProcessor, VitPoseForPoseEstimation

def visualize_bbox(image, bbox, output_path):
    """
    Visualize bounding box on image and save the result.
    Args:
        image: PIL Image
        bbox: Array of [x, y, width, height] in COCO format
        output_path: Path to save visualization
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    # Add bounding box
    x, y, w, h = bbox[0]
    rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def get_bbox_from_segmentation(segm_path):
    """
    Extract bounding box from segmentation mask.
    Args:
        segm_path: Path to segmentation mask PNG file
    Returns:
        bbox: Array of [x, y, width, height] in COCO format
    """
    try:
        # Check if segmentation file exists
        if not os.path.exists(segm_path):
            raise FileNotFoundError(f"Segmentation file not found: {segm_path}")
            
        # Read segmentation mask (all non-zero values are part of the person)
        segm = np.array(Image.open(segm_path))
        person_mask = segm > 0
        
        # Find bounding box coordinates
        y_indices, x_indices = np.where(person_mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            raise ValueError(f"No person found in segmentation mask: {segm_path}")
            
        x1, x2 = np.min(x_indices), np.max(x_indices)
        y1, y2 = np.min(y_indices), np.max(y_indices)
        
        # Convert to COCO format: [x, y, width, height]
        return np.array([[x1, y1, x2 - x1, y2 - y1]], dtype=np.float32)
        
    except Exception as e:
        print(f"Error processing segmentation file {segm_path}: {str(e)}")
        return None

def extract_features(img_dir, segm_dir, output_dir, vis_dir=None, batch_size=32, vis_samples=2):
    """
    Extract ViTPose features for all images in the dataset.
    Args:
        img_dir: Directory containing RGB images
        segm_dir: Directory containing segmentation masks
        output_dir: Directory to save feature numpy files
        vis_dir: Directory to save visualizations (optional)
        batch_size: Number of images to process at once
        vis_samples: Number of samples to visualize
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize ViTPose model and processor
        processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
        model = VitPoseForPoseEstimation.from_pretrained(
            "usyd-community/vitpose-base-simple",
            device_map=device,
            output_hidden_states=True
        )
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        if vis_dir:
            os.makedirs(vis_dir, exist_ok=True)
        
        # Get list of all image files
        image_files = sorted([f for f in os.listdir(img_dir) if f.startswith('img_') and f.endswith('.jpg')])
        if not image_files:
            raise FileNotFoundError(f"No image files found in {img_dir}")
        
        vis_count = 0
        
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            batch_images = []
            batch_boxes = []
            valid_indices = []
            
            # Prepare batch
            for idx, img_file in enumerate(batch_files):
                try:
                    # Get corresponding segmentation file
                    segm_file = f"segm_parts_{img_file[4:-4]}.png"
                    segm_path = os.path.join(segm_dir, segm_file)
                    img_path = os.path.join(img_dir, img_file)
                    
                    # Check if image file exists
                    if not os.path.exists(img_path):
                        raise FileNotFoundError(f"Image file not found: {img_path}")
                    
                    # Get bounding box from segmentation
                    bbox = get_bbox_from_segmentation(segm_path)
                    if bbox is None:
                        print("bbox is None")
                        continue
                    
                    # Load image
                    image = Image.open(img_path)
                    
                    # Save visualization if requested
                    if vis_dir and vis_count < vis_samples:
                        vis_path = os.path.join(vis_dir, f"bbox_vis_{img_file}")
                        visualize_bbox(image, bbox, vis_path)
                        vis_count += 1
                    
                    batch_images.append(image)
                    batch_boxes.append(bbox)
                    valid_indices.append(idx)
                    
                except Exception as e:
                    print(f"Error processing image {img_file}: {str(e)}")
                    continue
            
            if not batch_images:
                continue
            
            try:
                # Process batch
                inputs = processor(batch_images, boxes=batch_boxes, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Extract features from the last hidden state
                features = outputs.hidden_states[-1][:, 0].cpu().numpy()
                
                # Save features for each valid image in batch
                for idx, feat in zip(valid_indices, features):
                    img_file = batch_files[idx]
                    identity = img_file[4:-4]
                    output_path = os.path.join(output_dir, f"features_{identity}.npy")
                    np.save(output_path, feat)
                
            except Exception as e:
                print(f"Error processing batch starting with {batch_files[0]}: {str(e)}")
                continue
            
            print(f"Processed {i + len(batch_files)}/{len(image_files)} images")
            
    except Exception as e:
        print(f"Fatal error in feature extraction: {str(e)}")
        raise

if __name__ == "__main__":
    # Update these paths according to your setup
    img_dir = "./synth_body/img"
    segm_dir = "./synth_body/segm_parts"
    output_dir = "./features"
    vis_dir = "./visualizations"  # New directory for visualizations
    
    extract_features(img_dir, segm_dir, output_dir, vis_dir=vis_dir, vis_samples=2)