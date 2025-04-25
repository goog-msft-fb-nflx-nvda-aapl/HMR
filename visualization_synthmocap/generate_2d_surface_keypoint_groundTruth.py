import argparse
import json
import numpy as np
from pathlib import Path
import tqdm
import matplotlib.pyplot as plt
import lzma
import subprocess
from getpass import getpass
from tarfile import TarFile
import cv2
import pyrender
import trimesh
from transformations import rotation_matrix
from smpl_numpy import SMPL

SMPLH_MODEL = None

def _download_smplh() -> None:
    print("Downloading SMPL-H...")
    username = input("Username for https://mano.is.tue.mpg.de/: ")
    password = getpass("Password for https://mano.is.tue.mpg.de/: ")
    out_path = Path(__file__).parent / "smplh" / "smplh.tar.xz"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    url = "https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=smplh.tar.xz"
    try:
        subprocess.check_call(
            [
                "wget",
                "--post-data",
                f"username={username}&password={password}",
                url,
                "-O",
                out_path.as_posix(),
                "--no-check-certificate",
                "--continue",
            ]
        )
    except FileNotFoundError as exc:
        raise RuntimeError("wget not found, please install it") from exc
    except subprocess.CalledProcessError as exc:
        if out_path.exists():
            out_path.unlink()
        raise RuntimeError("Download failed, check your login details") from exc
    with lzma.open(out_path) as fd:
        with TarFile(fileobj=fd) as f:
            f.extractall(out_path.parent)
    out_path.unlink()


def _get_smplh() -> SMPL:
    global SMPLH_MODEL

    if SMPLH_MODEL is None:
        model_path = Path(__file__).parent / "smplh" / "neutral" / "model.npz"
        if not model_path.exists():
            _download_smplh()

        SMPLH_MODEL = SMPL(model_path)

    return SMPLH_MODEL

def project_vertices(vertices, world_to_cam, cam_to_img):
    """Project 3D vertices to 2D using camera parameters."""
    # Apply world-to-camera transformation
    vertices_cam = (world_to_cam @ np.hstack((vertices, np.ones((vertices.shape[0], 1)))).T).T[:, :3]

    # Apply camera intrinsics
    vertices_img = cam_to_img @ vertices_cam.T
    vertices_img = vertices_img[:2] / vertices_img[2]  # Normalize by depth
    return vertices_img.T

def build_index_mapping(data_dir):
    """Build a mapping from sequential index to (sidx, fidx) and vice versa."""
    data_dir = Path(data_dir)
    seq_to_sf = {}  # Maps sequential index to (sidx, fidx)
    sf_to_seq = {}  # Maps (sidx, fidx) to sequential index
    
    seq_idx = 0
    for sidx in tqdm.tqdm(range(20000), desc="Building index mapping"):
        for fidx in range(5):
            meta_file = data_dir / "metadata" / f"metadata_{sidx:07d}_{fidx:03d}.json"
            img_file = data_dir / "img" / f"img_{sidx:07d}_{fidx:03d}.jpg"
            
            if meta_file.exists() and img_file.exists():
                seq_to_sf[seq_idx] = (sidx, fidx)
                sf_to_seq[(sidx, fidx)] = seq_idx
                seq_idx += 1
    
    return seq_to_sf, sf_to_seq

def generate_projections(data_dir, output_file_2d, output_file_3d, n_vertices):
    """Generate and save the 2D and 3D vertex projections with correct indexing."""
    data_dir = Path(data_dir)
    
    # Initialize SMPL model
    smplh = _get_smplh()
    
    # Fix the sampled vertex indices for all images
    np.random.seed(42)  # Ensure reproducibility
    sampled_indices = np.random.choice(smplh.vertices.shape[0], n_vertices, replace=False)
    
    # First, build the index mapping
    seq_to_sf, sf_to_seq = build_index_mapping(data_dir)
    num_images = len(seq_to_sf)
    print(f"Found {num_images} valid images.")
    
    # Prepare output arrays
    projected_positions = np.zeros((num_images, n_vertices, 2), dtype=np.float32)
    vertices_output = np.zeros((num_images, n_vertices, 3), dtype=np.float32)
    
    # Process each valid image
    for seq_idx, (sidx, fidx) in tqdm.tqdm(seq_to_sf.items(), desc="Processing images"):
        meta_file = data_dir / "metadata" / f"metadata_{sidx:07d}_{fidx:03d}.json"
        
        with open(meta_file, "r") as f:
            metadata = json.load(f)
        
        # Set SMPL parameters
        smplh.beta = np.asarray(metadata["body_identity"][:smplh.shape_dim])
        smplh.theta = np.asarray(metadata["pose"])
        smplh.translation = np.asarray(metadata["translation"])
        
        # Get vertices
        vertices = smplh.vertices
        
        # Project to 2D
        world_to_cam = np.asarray(metadata["camera"]["world_to_camera"])
        cam_to_img = np.asarray(metadata["camera"]["camera_to_image"])
        projected_positions[seq_idx] = project_vertices(vertices, world_to_cam, cam_to_img)
        
        vertices_output[seq_idx] = vertices
    
    # Save the outputs
    np.save(output_file_2d, projected_positions)
    print(f"Saved projected positions to {output_file_2d}")
    
    np.save(output_file_3d, vertices_output)
    print(f"Saved 3D vertices to {output_file_3d}")
    
    return sampled_indices, seq_to_sf
def visualize_projected_vertices_on_image(data_dir, image_indices, projected_positions, sampled_indices, output_dir, seq_to_sf):
    """Visualize original image, projected 2D vertices, and both overlaid in a single figure."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for seq_idx in image_indices:
        if seq_idx not in seq_to_sf:
            print(f"Sequential index {seq_idx} not found in mapping. Skipping.")
            continue
        
        sidx, fidx = seq_to_sf[seq_idx]
        img_file = data_dir / "img" / f"img_{sidx:07d}_{fidx:03d}.jpg"
        
        if not img_file.exists():
            print(f"Image file {img_file} does not exist. Skipping.")
            continue
        
        # Load the original image
        image = cv2.imread(str(img_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create a blank image (white background) for the projected vertices visualization
        vertices_viz = np.ones_like(image) * 255  # White background
        
        # Create a copy of the original image for the overlay
        overlay_viz = image.copy()
        
        # Plot vertices on both the blank image and the overlay
        for v_idx, (x, y) in enumerate(projected_positions[seq_idx]):
            color = (0, 255, 0) if v_idx >= 3 else (255, 0, 0)  # First 3 vertices in red
            
            # Draw on the vertices-only visualization
            cv2.circle(vertices_viz, (int(x), int(y)), radius=3, color=color, thickness=-1)
            
            # Draw on the overlay visualization
            cv2.circle(overlay_viz, (int(x), int(y)), radius=3, color=color, thickness=-1)
            
        
        # Create a figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Plot vertices-only visualization
        axes[1].imshow(vertices_viz)
        axes[1].set_title("Projected Vertices")
        axes[1].axis("off")
        
        # Plot overlay visualization
        axes[2].imshow(overlay_viz)
        axes[2].set_title("Overlay")
        axes[2].axis("off")
        
        # Add a main title
        fig.suptitle(f"Image {sidx:07d}_{fidx:03d}", fontsize=16)
        plt.tight_layout()
        
        # Save the visualization
        output_path = output_dir / f"visualization_{sidx:07d}_{fidx:03d}.png"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to {output_path}")

def main():
    data_dir = "/home/iismtl519-2/Desktop/SynthMoCap/synth_body"
    output_file_3d = "/home/iismtl519-2/Desktop/smplvertices3d_synthmocap_v2.npy"
    output_file_2d = "/home/iismtl519-2/Desktop/smplvertices2d_synthmocap_v2.npy"
    n_vertices = 6890
    
    # Check if projections already exist
    if Path(output_file_2d).exists() and Path(output_file_3d).exists():
        print(f"Loading existing projection files...")
        projected_positions = np.load(output_file_2d)
        
        # Build the index mapping
        seq_to_sf, _ = build_index_mapping(data_dir)
        
        # Initialize SMPL model to get the sampled indices
        smplh = _get_smplh()
        np.random.seed(42)  # Ensure reproducibility
        sampled_indices = np.random.choice(smplh.vertices.shape[0], n_vertices, replace=False)
    else:
        # Generate the projections and get the mapping
        sampled_indices, seq_to_sf = generate_projections(
            data_dir, output_file_2d, output_file_3d, n_vertices
        )
        projected_positions = np.load(output_file_2d)
    
    # Set up visualization
    output_dir = Path(data_dir) / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Randomly select images to visualize
    import random
    available_indices = list(seq_to_sf.keys())
    print(f"Total available images: {len(available_indices)}")
    
    num_samples = min(100, len(available_indices))
    index_list = random.sample(available_indices, num_samples)
    
    # Visualize the selected images
    visualize_projected_vertices_on_image(
        data_dir, index_list, projected_positions, sampled_indices, output_dir, seq_to_sf
    )

if __name__ == "__main__":
    main()
