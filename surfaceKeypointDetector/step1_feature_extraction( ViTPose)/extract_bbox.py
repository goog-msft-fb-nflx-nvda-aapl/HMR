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
bbox_xyxy, bbox_cs = process_all_images(imgnames, base_dir)

# Load the existing npz file
data_dict = dict(data)  # Convert the npz object to a dictionary

# Add new keys for bounding boxes
data_dict['bbox_xyxy'] = bbox_xyxy
data_dict['bbox_cs'] = bbox_cs

# Save the updated dictionary to a new npz file
np.savez('/home/iismtl519-2/Desktop/SynthMoCap/synthmocap_with_bbox.npz', **data_dict)

# Verify the save
verification = np.load('/home/iismtl519-2/Desktop/SynthMoCap/synthmocap_with_bbox.npz')
print("New keys added:", 'bbox_xyxy' in verification, 'bbox_cs' in verification)
print("Shapes:", verification['bbox_xyxy'].shape, verification['bbox_cs'].shape)