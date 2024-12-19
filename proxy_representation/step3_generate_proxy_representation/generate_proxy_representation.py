import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

def convert_2Djoints_to_gaussian_heatmaps(joints, heatmap_size):
    """
    Convert 2D joints to Gaussian heatmaps.

    Args:
        joints (np.ndarray): Shape (num_joints, 2)
        heatmap_size (int): Size of output heatmap

    Returns:
        np.ndarray: Heatmaps of shape (num_joints, heatmap_size, heatmap_size)
    """
    heatmaps = np.zeros((joints.shape[0], heatmap_size, heatmap_size), dtype=np.float32)

    for i, (x, y) in enumerate(joints):
        if 0 <= x < heatmap_size and 0 <= y < heatmap_size:
            x_grid, y_grid = np.meshgrid(np.arange(heatmap_size), np.arange(heatmap_size))
            heatmap = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * 4**2))
            heatmaps[i] = heatmap / heatmap.max()

    return heatmaps


def process_data(silh_dir, label_path, output_path, visualization_dir):
    """
    Process the data to generate proxy representations and save them as a .npy file.
    """
    proxy_representations = []

    with np.load(label_path) as data:
        filenames = data["fnames"]
        joints2D_cropped = data["joints2D_cropped"]  # Shape: (311, 17, 3)

        for idx, fname in enumerate(filenames):
            # Load silhouette image
            silhouette_path = os.path.join(silh_dir, fname)
            silhouette = cv2.imread(silhouette_path, cv2.IMREAD_GRAYSCALE)

            if silhouette is None:
                raise FileNotFoundError(f"Silhouette file not found: {silhouette_path}")

            # Resize silhouette to (256, 256)
            silhouette_resized = cv2.resize(silhouette, (256, 256), interpolation=cv2.INTER_NEAREST)

            # Extract and scale keypoints
            keypoints = joints2D_cropped[idx, :, :2]  # Extract (x, y)
            confidence = joints2D_cropped[idx, :, 2]  # Confidence

            original_size = silhouette.shape[0]  # Width and height are the same (square images)
            scale_factor = 256 / original_size

            keypoints_scaled = keypoints * scale_factor

            # Create Gaussian heatmaps for keypoints
            keypoint_heatmaps = convert_2Djoints_to_gaussian_heatmaps(keypoints_scaled, 256)

            # Create proxy representation
            silhouette_resized = silhouette_resized.astype(np.float32) / 255.0  # Normalize to [0, 1]
            silhouette_resized = np.squeeze(silhouette_resized)  # Remove the singleton dimension

            # Add singleton dimension to keypoint heatmaps for concatenation
            silhouette_resized = silhouette_resized[np.newaxis, :, :]

            proxy_rep = np.concatenate([silhouette_resized, keypoint_heatmaps], axis=0)

            # Store proxy representation
            proxy_representations.append(proxy_rep)

    # Save proxy representations as a .npy file
    proxy_representations = np.array(proxy_representations, dtype=np.float32)
    np.save(output_path, proxy_representations)
    print(f"Proxy representations saved to {output_path}")


def visualize_proxy_representation(index):
    """
    Visualize each layer of the saved proxy representation and save the visualization.

    Args:
        index (int): Index of the proxy representation to visualize.
    """
    # Load proxy representations
    proxy_representations = np.load(OUTPUT_PATH)

    if index >= len(proxy_representations):
        raise IndexError(f"Index {index} is out of range for the proxy representations.")

    proxy_rep = proxy_representations[index]  # Shape: (18, 256, 256)

    fig, axes = plt.subplots(6, 3, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(proxy_rep.shape[0]):
        axes[i].imshow(proxy_rep[i], cmap="gray")
        axes[i].set_title(f"Layer {i}")
        axes[i].axis("off")

    plt.tight_layout()

    # Ensure visualization directory exists
    os.makedirs(visualization_dir, exist_ok=True)
    save_path = os.path.join(visualization_dir, f"proxy_representation_{index}.png")
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data to generate proxy representations')
    parser.add_argument('--silh_dir', type=str, required=True, help='Path to silhouette image directory')
    parser.add_argument('--label_path', type=str, required=True, help='Path to labels .npz file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save generated proxy representations')
    parser.add_argument('--visualization_dir', type=str, default=None, help='Path to save visualizations (optional)')

    args = parser.parse_args()

    silh_dir = args.silh_dir
    label_path = args.label_path
    output_path = args.output_path
    visualization_dir = args.visualization_dir

    process_data(silh_dir, label_path, output_path, visualization_dir)

    # Example: Visualize the first proxy representation (optional)
    index = 0
    if visualization_dir is not None:
        visualize_proxy_representation(index, visualization_dir)