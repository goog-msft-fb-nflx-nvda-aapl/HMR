import numpy as np

# Paths
bbox_path = "/home/iismtl519-2/Desktop/SynthMoCap/features/bbox.npy"
surfaceKP_path = "/home/iismtl519-2/Desktop/SynthMoCap/surfaceKP.npy"
output_path = "/home/iismtl519-2/Desktop/SynthMoCap/surfaceKP_v2_cropped.npy"

# Load data
bboxes = np.load(bbox_path)  # Shape: (95575, 4)
keypoints = np.load(surfaceKP_path)  # Shape: (95575, 100, 2)

# Check shapes
if bboxes.shape[0] != keypoints.shape[0]:
    raise ValueError("Mismatch between number of bounding boxes and keypoint sets!")

# Adjust keypoints to cropped image coordinates
cropped_keypoints = keypoints.copy()
for i in range(bboxes.shape[0]):
    x, y, _, _ = bboxes[i]  # Extract bbox [x, y, width, height]
    cropped_keypoints[i, :, 0] -= x  # Adjust x-coordinates
    cropped_keypoints[i, :, 1] -= y  # Adjust y-coordinates

# Save the adjusted keypoints
np.save(output_path, cropped_keypoints)
print(f"Saved cropped keypoints to {output_path}")