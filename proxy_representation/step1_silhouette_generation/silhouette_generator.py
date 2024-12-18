import os
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor
import argparse

class SilhouetteGenerator:
    def __init__(self, sam_checkpoint, model_type='vit_h', device='cuda'):
        """
        Initialize SAM model and predictor
        
        :param sam_checkpoint: Path to SAM model checkpoint
        :param model_type: Type of SAM model (vit_h, vit_l, vit_b)
        :param device: Device to run the model on (cuda/cpu)
        """
        self.device = torch.device(device)
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

    def transform_bbox_to_xyxy(self, bbox_centre, bbox_wh):
        """
        Transform bbox from centre+width/height to xyxy format
        
        :param bbox_centre: [x_centre, y_centre]
        :param bbox_wh: [width, height]
        :return: [x1, y1, x2, y2]
        """
        x_centre, y_centre = bbox_centre
        width = bbox_wh
        height = bbox_wh
        x1 = int(x_centre - width / 2)
        y1 = int(y_centre - height / 2)
        x2 = int(x_centre + width / 2)
        y2 = int(y_centre + height / 2)
        return np.array([x1, y1, x2, y2])

    def generate_silhouette(self, image_path, keypoints, bbox_centre, bbox_wh, confidence_threshold=0.0):
        """
        Generate binary silhouette using SAM
        
        :param image_path: Path to input image
        :param keypoints: Keypoints from joints2D 
        :param bbox_centre: Bounding box centre
        :param bbox_wh: Bounding box width and height
        :param confidence_threshold: Minimum confidence for keypoint inclusion
        :return: Binary silhouette mask
        """
        # Read image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Filter keypoints based on confidence
        valid_keypoints = keypoints[keypoints[:, 2] >= confidence_threshold][:, :2]
        
        # Transform bbox to xyxy format
        input_box = self.transform_bbox_to_xyxy(bbox_centre, bbox_wh)

        # Prepare predictor
        self.predictor.set_image(image)

        # Prepare point inputs
        input_point = valid_keypoints.astype(int)
        input_label = np.ones(len(input_point), dtype=int)  # All points are foreground

        # Generate masks
        masks, _, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        
        # Return the first (and only) mask
        return masks[0].astype(np.uint8) * 255

def create_directory_if_not_exists(directory_path):
    """
    Create a directory if it does not exist.
    """
    os.makedirs(directory_path, exist_ok=True)

def visualize_results(image_path, ssp_silhouette_path, sam_silhouette, visualization_dir, fname):
    """
    Save a visualization of the original image, SSP-3D silhouette, and SAM-predicted silhouette.
    
    :param image_path: Path to the original image
    :param ssp_silhouette_path: Path to SSP-3D silhouette image
    :param sam_silhouette: SAM-predicted silhouette (binary mask)
    :param visualization_dir: Directory to save visualization
    :param fname: Original filename
    """
    # Read images
    original_image = cv2.imread(image_path)
    ssp_silhouette = cv2.imread(ssp_silhouette_path, cv2.IMREAD_GRAYSCALE) * 255
    
    # Convert SAM silhouette to 3-channel for visualization
    sam_silhouette_colored = cv2.cvtColor(sam_silhouette, cv2.COLOR_GRAY2BGR)

    # Concatenate images for side-by-side comparison
    visualization = np.hstack((
        original_image,
        cv2.cvtColor(ssp_silhouette, cv2.COLOR_GRAY2BGR),
        sam_silhouette_colored
    ))

        # Annotation settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White text
    thickness = 2
    line_type = cv2.LINE_AA

    # Add annotations to each section
    text_positions = [
        (10, 50),  # Position for "Original Image"
        (original_image.shape[1] + 10, 50),  # Position for "Silhouette by SSP-3D"
        (original_image.shape[1] * 2 + 10, 50)  # Position for "Silhouette by SAM"
    ]
    labels = ["Original Image", "Silhouette by SSP-3D", "Silhouette by SAM"]

    for position, label in zip(text_positions, labels):
        cv2.putText(visualization, label, position, font, font_scale, font_color, thickness, line_type)

    # Save visualization
    visualization_path = os.path.join(visualization_dir, fname.replace('.png', '_visualization.png'))
    cv2.imwrite(visualization_path, visualization)
    print(f"Visualization saved to {visualization_path}")

def main():
    parser = argparse.ArgumentParser(description='Process labels and generate silhouettes')
    parser.add_argument('--labels_path', type=str, required=True, help='Path to the labels .npz file')
    parser.add_argument('--img_dir', type=str, default='./images', help='Input directory for input images')
    parser.add_argument('--silh_dir', type=str, default='./images', help='Input directory for silhouettes provided by SSP-3D')
    parser.add_argument('--out_dir', type=str, default='./silhouettes_SAM', help='Output directory for silhouettes predicted by SAM')
    parser.add_argument('--visualization_dir', type=str, default='./visualization', help='Output directory for visualization')
    parser.add_argument('--sam_checkpoint', type=str, default='./sam_vit_h_4b8939.pth', help='Path to the SAM checkpoint')

    args = parser.parse_args()

    # Create required directories
    create_directory_if_not_exists(args.out_dir)
    create_directory_if_not_exists(args.visualization_dir)

    # Load data
    with np.load(args.labels_path) as data:
        fnames = data['fnames']
        joints2D = data['joints2D']
        bbox_centres = data['bbox_centres']
        bbox_whs = data['bbox_whs']
    
    # Initialize SAM
    generator = SilhouetteGenerator(args.sam_checkpoint)
    
    # Process each image
    for i, (fname, keypoints, bbox_centre, bbox_wh) in enumerate(
        zip(fnames, joints2D, bbox_centres, bbox_whs)
    ):
        try:
            image_path = os.path.join(args.img_dir, fname)
            ssp_silhouette_path = os.path.join(args.silh_dir, fname)

            # Generate silhouette
            sam_silhouette = generator.generate_silhouette(image_path, keypoints, bbox_centre, bbox_wh)

            # Save SAM silhouette
            output_path = os.path.join(args.out_dir, fname.replace('.png', '_silhouette.png'))
            cv2.imwrite(output_path, sam_silhouette)
            
            # Visualize and save the results
            visualize_results(image_path, ssp_silhouette_path, sam_silhouette, args.visualization_dir, fname)
            
            print(f"Processed {i+1}/{len(fnames)}: {output_path}")
        
        except Exception as e:
            print(f"Error processing {fname}: {e}")

if __name__ == "__main__":
    main()