import numpy as np

# Paths
cropped_keypoints_path = "/home/iismtl519-2/Desktop/SynthMoCap/surfaceKP_v2_cropped.npy"
bbox_path = "/home/iismtl519-2/Desktop/SynthMoCap/features/bbox.npy"
output_path = "/home/iismtl519-2/Desktop/SynthMoCap/surfaceKP_v3_resized.npy"

# Load Data
cropped_keypoints = np.load(cropped_keypoints_path)  # Shape: (95575, 100, 2)
bboxes = np.load(bbox_path)  # Shape: (95575, 4)

# Initialize the resized keypoints array
resized_keypoints = cropped_keypoints.copy()

for i in range(bboxes.shape[0]):
    _, _, width, height = bboxes[i]  # Original bbox dimensions
    
    # Step 1: Padding
    if width > height:
        pad_top = (width - height) // 2
        pad_bottom = width - height - pad_top
        pad_left, pad_right = 0, 0
    else:
        pad_left = (height - width) // 2
        pad_right = height - width - pad_left
        pad_top, pad_bottom = 0, 0

    # Adjust keypoints for padding
    resized_keypoints[i, :, 0] += pad_left
    resized_keypoints[i, :, 1] += pad_top
    
    # Step 2: Resize to (256, 256)
    scale_x = 256 / max(width, height)
    scale_y = 256 / max(width, height)
    resized_keypoints[i, :, 0] *= scale_x
    resized_keypoints[i, :, 1] *= scale_y

# Save the resized keypoints
np.save(output_path, resized_keypoints)
print(f"Saved resized keypoints to {output_path}")