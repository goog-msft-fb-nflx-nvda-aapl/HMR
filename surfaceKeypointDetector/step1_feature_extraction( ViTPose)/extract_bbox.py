import numpy as np
import os
from PIL import Image

# Load the original data
data = np.load('/home/iismtl519-2/Desktop/SynthMoCap/synthmocap.npz')
imgnames = data['imgname']  # shape = (95575,)

def get_bbox_from_segm(segm_path):
    """
    Extract bounding box from segmentation mask in both (x1,y1,x2,y2) and (center,scale) formats.
    
    Args:
        segm_path (str): Path to the segmentation mask PNG file
        
    Returns:
        tuple: ((x1,y1,x2,y2), (center_x,center_y), scale)
    """
    # Read segmentation mask
    segm = np.array(Image.open(segm_path))
    
    # Find non-background pixels (any value > 0)
    y_indices, x_indices = np.nonzero(segm > 0)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        raise ValueError(f"No foreground pixels found in {segm_path}")
    
    # Get bbox in (x1,y1,x2,y2) format
    x1, y1 = x_indices.min(), y_indices.min()
    x2, y2 = x_indices.max(), y_indices.max()
    
    # Calculate center point
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Calculate box size (maximum of width and height)
    width = x2 - x1
    height = y2 - y1
    box_size = max(width, height)
    
    # Calculate scale (box_size/200 as used in the dataset class)
    scale = box_size / 200
    
    return (x1, y1, x2, y2), (center_x, center_y), scale

def process_all_images(imgnames, base_dir):
    """
    Process all images and store their bounding boxes.
    
    Args:
        imgnames (np.ndarray): Array of image names
        base_dir (str): Base directory containing segmentation masks
        
    Returns:
        tuple: Two arrays for both bbox formats
            - bbox_xyxy: Array of shape (N, 4) for (x1,y1,x2,y2) format
            - bbox_cs: Array of shape (N, 3) for (center_x,center_y,scale) format
    """
    N = len(imgnames)
    bbox_xyxy = np.zeros((N, 4), dtype=np.float32)
    bbox_cs = np.zeros((N, 3), dtype=np.float32)  # [center_x, center_y, scale]
    
    for i, imgname in enumerate(imgnames):
        # Convert image name to segmentation mask path
        seq_num = imgname[4:11]
        frame_num = imgname[12:15]
        segm_path = os.path.join(base_dir, 'synth_body/segm_parts', 
                                f'segm_parts_{seq_num}_{frame_num}.png')
        
        try:
            (x1, y1, x2, y2), (center_x, center_y), scale = get_bbox_from_segm(segm_path)
            bbox_xyxy[i] = [x1, y1, x2, y2]
            bbox_cs[i] = [center_x, center_y, scale]
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i+1}/{N} images")
                
        except Exception as e:
            print(f"Error processing {imgname}: {str(e)}")
            continue
    
    return bbox_xyxy, bbox_cs

# Process all images
base_dir = '/home/iismtl519-2/Desktop/SynthMoCap'
# bbox_xyxy, bbox_cs = process_all_images(imgnames, base_dir)

# Load the existing npz file
data_dict = dict(data)  # Convert the npz object to a dictionary

# Add new keys for bounding boxes
# data_dict['bbox_xyxy'] = bbox_xyxy
# data_dict['bbox_cs'] = bbox_cs

# Save the updated dictionary to a new npz file
# np.savez('/home/iismtl519-2/Desktop/SynthMoCap/synthmocap_with_bbox.npz', **data_dict)

# Verify the save
# verification = np.load('/home/iismtl519-2/Desktop/SynthMoCap/synthmocap_with_bbox.npz')
# print("New keys added:", 'bbox_xyxy' in verification, 'bbox_cs' in verification)
# print("Shapes:", verification['bbox_xyxy'].shape, verification['bbox_cs'].shape)

import numpy as np
import cv2
import os
from PIL import Image
import random

def visualize_bboxes(img_path, bbox_xyxy, center, scale, output_path=None):
    """
    Visualize both types of bounding boxes on an image.
    
    Args:
        img_path (str): Path to the original image
        bbox_xyxy (np.ndarray): Bounding box in (x1,y1,x2,y2) format
        center (np.ndarray): Center point (x,y)
        scale (float): Scale value
        output_path (str, optional): Path to save the visualization
    """
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    # Draw bbox_xyxy in red
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Draw center,scale bbox in green
    center_x, center_y = map(int, center)
    box_size = int(scale * 200)  # Convert scale back to pixels
    half_size = box_size // 2
    cs_x1 = center_x - half_size
    cs_y1 = center_y - half_size
    cs_x2 = center_x + half_size
    cs_y2 = center_y + half_size
    cv2.rectangle(img, (cs_x1, cs_y1), (cs_x2, cs_y2), (0, 255, 0), 2)
    
    # Draw center point
    cv2.circle(img, (center_x, center_y), 5, (255, 0, 0), -1)
    
    # Add legend
    cv2.putText(img, "XYXY format", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, "Center-Scale format", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, "Center point", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, img)
    else:
        cv2.imshow('Bounding Box Visualization', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Load the data with bounding boxes
data = np.load('/home/iismtl519-2/Desktop/SynthMoCap/synthmocap_with_bbox.npz')
imgnames = data['imgname']
bbox_xyxy = data['bbox_xyxy']
bbox_cs = data['bbox_cs']  # [center_x, center_y, scale]

# Create output directory for visualizations
output_dir = '/home/iismtl519-2/Desktop/SynthMoCap/bbox_visualizations'
os.makedirs(output_dir, exist_ok=True)

# Randomly select 5 images to visualize
num_samples = 5
total_images = len(imgnames)
sample_indices = random.sample(range(total_images), num_samples)

for idx in sample_indices:
    imgname = imgnames[idx]
    img_path = os.path.join('/home/iismtl519-2/Desktop/SynthMoCap', 'synth_body/img', imgname)
    
    # Get bounding box data
    xyxy = bbox_xyxy[idx]
    center = bbox_cs[idx][:2]  # first two values are center_x, center_y
    scale = bbox_cs[idx][2]    # third value is scale
    
    # Create output path
    output_path = os.path.join(output_dir, f'bbox_vis_{imgname}')
    
    try:
        visualize_bboxes(img_path, xyxy, center, scale, output_path)
        print(f"Saved visualization for {imgname}")
    except Exception as e:
        print(f"Error processing {imgname}: {str(e)}")

print(f"\nVisualizations saved to: {output_dir}")

import numpy as np

# Load the data
data = np.load('/home/iismtl519-2/Desktop/SynthMoCap/synthmocap_with_bbox.npz')

# Create a new dictionary with all existing data
data_dict = dict(data)

# Split bbox_cs into center and scale
center = data['bbox_cs'][:, :2]  # First two columns (center_x, center_y)
scale = data['bbox_cs'][:, 2]    # Third column (scale)

# Add the new keys
data_dict['center'] = center
data_dict['scale'] = scale

# Remove the old bbox_cs key if you don't want to keep it
if 'bbox_cs' in data_dict:
    del data_dict['bbox_cs']

# Save the updated dictionary to the npz file
np.savez('/home/iismtl519-2/Desktop/SynthMoCap/synthmocap_with_bbox.npz', **data_dict)

# Verify the save
verification = np.load('/home/iismtl519-2/Desktop/SynthMoCap/synthmocap_with_bbox.npz')
print("\nVerification:")
print("New keys added:", 'center' in verification, 'scale' in verification)
print("Old key removed:", 'bbox_cs' not in verification)
print("Shapes:")
print("- center:", verification['center'].shape)  # Should be (N, 2)
print("- scale:", verification['scale'].shape)    # Should be (N,)