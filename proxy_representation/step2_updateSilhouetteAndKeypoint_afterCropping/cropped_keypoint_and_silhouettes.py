import os
import numpy as np
import cv2
import argparse
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process silhouettes, update keypoints, and visualize them.")
parser.add_argument('--silh_dir', type=str, required=True, help='Directory containing silhouettes.')
parser.add_argument('--crop_silh_dir', type=str, required=True, help='Directory to save cropped silhouettes.')
parser.add_argument('--img_dir', type=str, required=True, help='Directory containing original images.')
parser.add_argument('--annotation_file', type=str, required=True, help='Path to the annotation file.')
parser.add_argument('--visualization_dir', type=str, required=True, help='Directory to save visualized keypoints.')
args = parser.parse_args()

# Constant
CONFIDENCE_THRESHOLD = 0.0

# Directories
SILH_DIR = args.silh_dir
CROP_SILH_DIR = args.crop_silh_dir
IMG_DIR = args.img_dir
ANNOTATION_FILE_PATH = args.annotation_file
VISUALIZATION_DIR = args.visualization_dir

# Create output directories if they don't exist
os.makedirs(CROP_SILH_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Keypoint names
KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", "Left Shoulder", "Right Shoulder",
    "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", "Left Knee",
    "Right Knee", "Left Ankle", "Right Ankle"
]

# Load annotations
with np.load(ANNOTATION_FILE_PATH) as data:
    fnames = data['fnames']
    joints2D = data['joints2D']
    bbox_centres = data['bbox_centres']
    bbox_whs = data['bbox_whs']
    cam_trans = data['cam_trans']
    genders = data['genders']
    betas = data['betas']
    poses = data['poses']

# Updated keypoints storage
joints2D_cropped = []

# Process each image
for i, fname in enumerate(tqdm(fnames)):
    # Load silhouette
    silh_path = os.path.join(SILH_DIR, fname)
    silhouette = cv2.imread(silh_path, cv2.IMREAD_GRAYSCALE)

    # Load original image
    img_path = os.path.join(IMG_DIR, fname)
    original_img = cv2.imread(img_path)

    # Bounding box information
    centre_x, centre_y = bbox_centres[i]
    wh = bbox_whs[i]
    half_wh = wh // 2

    # Define bounding box
    x1 = max(0, int(centre_x - half_wh))
    y1 = max(0, int(centre_y - half_wh))
    x2 = min(silhouette.shape[1], int(centre_x + half_wh))
    y2 = min(silhouette.shape[0], int(centre_y + half_wh))

    # Crop silhouette
    cropped_silhouette = silhouette[y1:y2, x1:x2]

    # Crop original image with the same bounding box
    cropped_original_img = original_img[y1:y2, x1:x2]

    # Save cropped silhouette
    cropped_silh_path = os.path.join(CROP_SILH_DIR, fname)
    cv2.imwrite(cropped_silh_path, cropped_silhouette)

    # Update keypoints for cropped silhouette
    keypoints = joints2D[i]
    updated_keypoints = []
    for keypoint in keypoints:
        x, y, conf = keypoint
        updated_x = x - x1
        updated_y = y - y1
        updated_keypoints.append([updated_x, updated_y, conf])
    joints2D_cropped.append(updated_keypoints)

# Convert updated keypoints to numpy array
joints2D_cropped = np.array(joints2D_cropped)

# Save updated annotations
np.savez(ANNOTATION_FILE_PATH, 
         fnames=fnames, 
         poses=poses, 
         joints2D=joints2D, 
         cam_trans=cam_trans, 
         genders=genders, 
         bbox_centres=bbox_centres, 
         bbox_whs=bbox_whs, 
         betas=betas, 
         joints2D_cropped=joints2D_cropped)

# Visualize updated keypoints on cropped silhouettes
for i, fname in enumerate(tqdm(fnames)):
    # Load cropped silhouette
    cropped_silh_path = os.path.join(CROP_SILH_DIR, fname)
    cropped_silhouette = cv2.imread(cropped_silh_path, cv2.IMREAD_GRAYSCALE)

    # Load cropped original image 
    img_path = os.path.join(IMG_DIR, fname)
    original_img = cv2.imread(img_path)
    
    # Crop original image with the same bounding box
    centre_x, centre_y = bbox_centres[i]
    wh = bbox_whs[i]
    half_wh = wh // 2
    x1 = max(0, int(centre_x - half_wh))
    y1 = max(0, int(centre_y - half_wh))
    x2 = min(original_img.shape[1], int(centre_x + half_wh))
    y2 = min(original_img.shape[0], int(centre_y + half_wh))
    cropped_original_img = original_img[y1:y2, x1:x2]

    # Convert cropped silhouette to color for side-by-side visualization
    cropped_silhouette_color = cv2.cvtColor(cropped_silhouette, cv2.COLOR_GRAY2BGR)

    # Create side-by-side visualization
    visual_image = np.hstack((cropped_original_img, cropped_silhouette_color))

    # Overlay keypoints on cropped silhouette side
    for j, keypoint in enumerate(joints2D_cropped[i]):
        x, y, conf = keypoint
        offset = cropped_original_img.shape[1]  # Offset x coordinate to right side of image
        if conf > CONFIDENCE_THRESHOLD:  # Only visualize keypoints with confidence > confidence threshold
            cv2.circle(visual_image, (int(x) + offset, int(y)), 3, (0, 255, 0), -1)
            cv2.putText(visual_image, KEYPOINT_NAMES[j], (int(x) + offset + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

    # Save visualized image
    vis_path = os.path.join(VISUALIZATION_DIR, fname)
    cv2.imwrite(vis_path, visual_image)

print("Processing complete.")