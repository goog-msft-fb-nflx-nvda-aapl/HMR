import os
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor
import argparse

class SilhouetteGenerator:
    def __init__(self, sam_checkpoint, model_type='vit_h', device='cuda'):
        self.device = torch.device(device)
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

    def get_bbox_from_keypoints(self, keypoints, padding=1.5):
        """Calculate bounding box from keypoints"""
        valid_keypoints = keypoints[keypoints[:, 2] > 0][:, :2]
        if len(valid_keypoints) == 0:
            raise ValueError("No valid keypoints found")
            
        x_min, y_min = valid_keypoints.min(axis=0)
        x_max, y_max = valid_keypoints.max(axis=0)
        
        # Calculate center and size
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = (x_max - x_min) * padding
        height = (y_max - y_min) * padding
        
        # Calculate padded bbox
        x1 = max(0, int(center_x - width/2))
        y1 = max(0, int(center_y - height/2))
        x2 = int(center_x + width/2)
        y2 = int(center_y + height/2)
        
        return np.array([x1, y1, x2, y2])

    def get_bbox_from_silhouette(self, silhouette, padding=1.0):
        """Calculate bounding box from binary silhouette"""
        # Find non-zero points in the silhouette
        y_coords, x_coords = np.nonzero(silhouette)
        
        if len(x_coords) == 0 or len(y_coords) == 0:
            raise ValueError("No silhouette points found")
            
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Add padding
        width = x_max - x_min
        height = y_max - y_min
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Calculate padded bbox
        x1 = max(0, int(center_x - (width * padding)/2))
        y1 = max(0, int(center_y - (height * padding)/2))
        x2 = int(center_x + (width * padding)/2)
        y2 = int(center_y + (height * padding)/2)
        
        return np.array([x1, y1, x2, y2])

    def refine_silhouette_with_bbox(self, silhouette, keypoints_bbox, padding=1.2):
        """
        Refine the SAM-generated silhouette using keypoints bbox as a constraint
        
        Args:
            silhouette: Binary mask from SAM
            keypoints_bbox: [x1, y1, x2, y2] from keypoints
            padding: Factor to expand keypoints bbox (default 1.2 to avoid cutting person)
        
        Returns:
            Refined binary silhouette
        """
        # Expand keypoints bbox by padding factor
        x1, y1, x2, y2 = keypoints_bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = (x2 - x1) * padding
        height = (y2 - y1) * padding
        
        # Calculate padded bbox
        padded_x1 = max(0, int(center_x - width/2))
        padded_y1 = max(0, int(center_y - height/2))
        padded_x2 = min(silhouette.shape[1], int(center_x + width/2))
        padded_y2 = min(silhouette.shape[0], int(center_y + height/2))
        
        # Create mask from padded bbox
        bbox_mask = np.zeros_like(silhouette)
        bbox_mask[padded_y1:padded_y2, padded_x1:padded_x2] = 1
        
        # Combine SAM silhouette with bbox mask
        refined_silhouette = silhouette * bbox_mask
        
        return refined_silhouette

    def generate_silhouette(self, image_path, keypoints, confidence_threshold=0.0, bbox_padding=1.5):
        """Generate refined binary silhouette using SAM and keypoints bbox"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        valid_keypoints = keypoints[keypoints[:, 2] >= confidence_threshold][:, :2]
        print(valid_keypoints.shape)
        input_box = self.get_bbox_from_keypoints(keypoints)

        self.predictor.set_image(image)
        # masks, _, _ = self.predictor.predict(
        #     point_coords=valid_keypoints,
        #     point_labels=np.ones(len(valid_keypoints)),
        #     box=input_box[None, :],
        #     multimask_output=False
        # )
        masks, scores, logits = self.predictor.predict(
            point_coords=valid_keypoints,
            point_labels=np.ones(len(valid_keypoints)),
            multimask_output=True,
        )
        mask_input = logits[np.argmax(scores), :, :]
        
        masks, _, _ = self.predictor.predict(
            point_coords=valid_keypoints,
            point_labels=np.ones(len(valid_keypoints)),
            mask_input=mask_input[None, :, :],
            multimask_output=False,
        )
        initial_silhouette = masks[0].astype(np.uint8) * 255
        
        # Refine silhouette using keypoints bbox
        """
        refined_silhouette = self.refine_silhouette_with_bbox(
            initial_silhouette, 
            input_box,
            padding=bbox_padding
        )
        """
        return initial_silhouette

def draw_keypoints(image, keypoints, color=(0, 0, 255), radius=20, thickness=20):
    """
    Draw keypoints on the image.
    
    Args:
        image: Original image array (H, W, 3).
        keypoints: Array of shape (N, 3) with (x, y, confidence).
        color: Color of the keypoints (default blue).
        radius: Radius of keypoint circles.
        thickness: Thickness of circles (-1 for filled).
    
    Returns:
        Image with keypoints drawn.
    """
    image_with_keypoints = image.copy()
    for x, y, confidence in keypoints:
        if confidence > 0:  # Only draw keypoints with confidence > 0
            cv2.circle(image_with_keypoints, (int(x), int(y)), radius, color, thickness)
    return image_with_keypoints

def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    """Draw bounding box on image"""
    x1, y1, x2, y2 = bbox
    return cv2.rectangle(image.copy(), (x1, y1), (x2, y2), color, thickness)

def create_directory_if_not_exists(directory_path):
    """Create a directory if it does not exist."""
    os.makedirs(directory_path, exist_ok=True)

# def visualize_results(image_path, sam_silhouette, kpts_bbox, silh_bbox, visualization_dir, img_dir):
#     """
#     Save visualization with bounding boxes
#     """
#     # Read original image
#     original_image = cv2.imread(image_path)
#     if original_image is None:
#         raise ValueError(f"Could not read image: {image_path}")
    
#     # Convert silhouette to 3-channel
#     sam_silhouette_colored = cv2.cvtColor(sam_silhouette, cv2.COLOR_GRAY2BGR)
    
#     # Draw bounding boxes
#     original_with_bbox = draw_bbox(original_image, kpts_bbox, (0, 255, 0))  # Green for keypoints
#     original_with_bbox = draw_bbox(original_with_bbox, silh_bbox, (0, 0, 255))  # Blue for silhouette
    
#     silhouette_with_bbox = draw_bbox(sam_silhouette_colored, kpts_bbox, (0, 255, 0))
#     silhouette_with_bbox = draw_bbox(silhouette_with_bbox, silh_bbox, (0, 0, 255))
    
#     # Concatenate images
#     visualization = np.hstack((original_with_bbox, silhouette_with_bbox))

#     # Add labels
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(visualization, "Original + BBs", (10, 30), font, 1, (255, 255, 255), 2)
#     cv2.putText(visualization, "Silhouette + BBs", (original_image.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
    
#     # Add legend
#     cv2.putText(visualization, "Green: Keypoints BB", (10, 60), font, 0.7, (0, 255, 0), 2)
#     cv2.putText(visualization, "Red: Silhouette BB", (10, 90), font, 0.7, (0, 0, 255), 2)

#     # Create output path
#     rel_path = os.path.relpath(image_path, img_dir)
#     base_path = os.path.splitext(rel_path)[0]
#     output_path = os.path.join(visualization_dir, f"{base_path}_visualization.png")
    
#     # Create necessary subdirectories
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
#     cv2.imwrite(output_path, visualization)

def visualize_results(image_path, sam_silhouette, kpts_bbox, silh_bbox, visualization_dir, img_dir, keypoints):
    """
    Save visualization with bounding boxes and keypoints.
    """
    # Read original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Draw keypoints on the original image
    original_with_keypoints = draw_keypoints(original_image, keypoints)
    
    # Convert silhouette to 3-channel
    sam_silhouette_colored = cv2.cvtColor(sam_silhouette, cv2.COLOR_GRAY2BGR)
    
    # Draw bounding boxes
    original_with_bbox = draw_bbox(original_with_keypoints, kpts_bbox, (0, 255, 0))  # Green for keypoints
    original_with_bbox = draw_bbox(original_with_bbox, silh_bbox, (0, 0, 255))  # Blue for silhouette
    
    silhouette_with_bbox = draw_bbox(sam_silhouette_colored, kpts_bbox, (0, 255, 0))
    silhouette_with_bbox = draw_bbox(silhouette_with_bbox, silh_bbox, (0, 0, 255))
    
    # Concatenate images
    visualization = np.hstack((original_with_bbox, silhouette_with_bbox))

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(visualization, "Original + BBs + Keypoints", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(visualization, "Silhouette + BBs", (original_image.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
    
    # Add legend
    cv2.putText(visualization, "Green: Keypoints BB", (10, 60), font, 0.7, (0, 255, 0), 2)
    cv2.putText(visualization, "Red: Silhouette BB", (10, 90), font, 0.7, (0, 0, 255), 2)
    cv2.putText(visualization, "Blue: Keypoints", (10, 120), font, 0.7, (255, 0, 0), 2)

    # Create output path
    rel_path = os.path.relpath(image_path, img_dir)
    base_path = os.path.splitext(rel_path)[0]
    output_path = os.path.join(visualization_dir, f"{base_path}_visualization.png")
    
    # Create necessary subdirectories
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cv2.imwrite(output_path, visualization)

def main():
    parser = argparse.ArgumentParser(description='Generate silhouettes and bounding boxes from HBW dataset')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the HBW dataset .npz file')
    parser.add_argument('--img_dir', type=str, required=True, help='Root directory containing the images')
    parser.add_argument('--out_dir', type=str, default='./silhouettes_SAM', help='Output directory for silhouettes')
    parser.add_argument('--visualization_dir', type=str, default='./visualization', help='Output directory for visualizations')
    parser.add_argument('--sam_checkpoint', type=str, required=True, help='Path to the SAM checkpoint')
    parser.add_argument('--confidence_threshold', type=float, default=0.3, help='Minimum confidence for keypoint inclusion')

    args = parser.parse_args()

    # Create output directories
    create_directory_if_not_exists(args.out_dir)
    create_directory_if_not_exists(args.visualization_dir)

    # Load HBW dataset
    data = np.load(args.dataset_path)
    keypoints = data['body_keypoints_2d']  # Shape: (N, 25, 3)
    image_names = data['imgname']

    # Initialize arrays for bounding boxes
    keypoint_bboxes = np.zeros((len(image_names), 4))
    silhouette_bboxes = np.zeros((len(image_names), 4))

    # Initialize SAM
    generator = SilhouetteGenerator(args.sam_checkpoint)
    
    # Process each image
    for i, (fname, kpts) in enumerate(zip(image_names, keypoints)):
        try:
            image_path = fname  # Using full path directly from dataset
            
            # Generate silhouette
            silhouette = generator.generate_silhouette(
                image_path, 
                kpts,
                confidence_threshold=args.confidence_threshold
            )

            # Get bounding boxes
            kpts_bbox = generator.get_bbox_from_keypoints(kpts)
            silh_bbox = generator.get_bbox_from_silhouette(silhouette)
            
            # Store bounding boxes
            keypoint_bboxes[i] = kpts_bbox
            silhouette_bboxes[i] = silh_bbox

            # Create output path for silhouette
            rel_path = os.path.relpath(image_path, args.img_dir)
            base_path = os.path.splitext(rel_path)[0]
            output_path = os.path.join(args.out_dir, f"{base_path}_silhouette.png")
            
            # Create necessary subdirectories
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save silhouette
            cv2.imwrite(output_path, silhouette)
            
            # Generate and save visualization
            visualize_results(image_path, silhouette, kpts_bbox, silh_bbox, 
                            args.visualization_dir, args.img_dir, keypoints[i])
    
            print(f"Processed {i+1}/{len(image_names)}: {fname}")
            
        except Exception as e:
            print(f"Error processing {fname}: {str(e)}")
    
    # Save bounding boxes
    bbox_save_path = os.path.join(args.out_dir, 'bounding_boxes.npz')
    np.savez(bbox_save_path, 
             keypoint_bboxes=keypoint_bboxes,
             silhouette_bboxes=silhouette_bboxes)
    print(f"Saved bounding boxes to {bbox_save_path}")

if __name__ == "__main__":
    main()