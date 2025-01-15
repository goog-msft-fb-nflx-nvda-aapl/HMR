import argparse
import json
import numpy as np
from pathlib import Path
import tqdm
import matplotlib.pyplot as plt
import argparse
import json
import lzma
import subprocess
from getpass import getpass
from pathlib import Path
from tarfile import TarFile
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrender
import trimesh
from transformations import rotation_matrix

from smpl_numpy import SMPL
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

def visualize_projected_vertices_on_image(
    data_dir, image_idx, projected_positions, sampled_indices, output_dir
):
    """Visualize projected 2D vertices on the original image."""
    for idx in image_idx:
        # Locate the corresponding image file
        sidx = idx // 5
        fidx = idx % 5
        img_file = Path(data_dir) / "img" / f"img_{sidx:07d}_{fidx:03d}.jpg"

        if not img_file.exists():
            print(f"Image file {img_file} does not exist. Skipping.")
            continue

        # Load the original image
        image = cv2.imread(str(img_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Overlay projected vertices
        for v_idx, (x, y) in enumerate(projected_positions[idx]):
            color = (0, 255, 0) if v_idx >= 3 else (255, 0, 0)  # Annotate first 3 vertices in red
            cv2.circle(image, (int(x), int(y)), radius=3, color=color, thickness=-1)

            if v_idx < 3:  # Annotate the first three vertices
                cv2.putText(
                    image,
                    f"V{sampled_indices[v_idx]}",
                    (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1,
                )

        # Save the visualization
        output_path = output_dir / f"visualization_{idx:07d}.png"
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Projected Vertices on Image {idx}")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved visualization to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, help="Path to the dataset directory.")
    parser.add_argument("output_file", type=Path, help="Path to save the npy file.")
    parser.add_argument("--n_vertices", type=int, default=100, help="Number of vertices to sample.")
    args = parser.parse_args()
    
    data_dir = args.data_dir

    print("data_dir",data_dir)
    output_file = args.output_file
    n_vertices = args.n_vertices
    
    # Initialize SMPL model
    smplh = _get_smplh()

    # Fix the sampled vertex indices for all images
    np.random.seed(42)  # Ensure reproducibility
    sampled_indices = np.random.choice(smplh.vertices.shape[0], n_vertices, replace=False)

    # Prepare output array
    num_images = 95575
    projected_positions = np.zeros((num_images, n_vertices, 2), dtype=np.float32)

    image_idx = 0

    for sidx in tqdm.tqdm(range(20000), desc="Processing identities"):
        for fidx in range(5):
            meta_file = data_dir / f"metadata/metadata_{sidx:07d}_{fidx:03d}.json"

            if not meta_file.exists():
                continue

            with open(meta_file, "r") as f:
                metadata = json.load(f)

            # Set SMPL parameters
            smplh.beta = np.asarray(metadata["body_identity"][:smplh.shape_dim])
            smplh.theta = np.asarray(metadata["pose"])
            smplh.translation = np.asarray(metadata["translation"])

            # Get vertices and sample the required indices
            vertices = smplh.vertices[sampled_indices]

            # Project to 2D
            world_to_cam = np.asarray(metadata["camera"]["world_to_camera"])
            cam_to_img = np.asarray(metadata["camera"]["camera_to_image"])
            projected_positions[image_idx] = project_vertices(vertices, world_to_cam, cam_to_img)

            image_idx += 1

    # Save the output
    np.save(output_file, projected_positions)
    print(f"Saved projected positions to {output_file}")

    # Visualize a few examples
    output_dir = data_dir / "visualizations"
    projected_positions = np.load("/home/iismtl519-2/Desktop/SynthMoCap/surfaceKP.npy")
    output_dir.mkdir(parents=True, exist_ok=True)
    visualize_projected_vertices_on_image(
        data_dir, [0, 1, 2], projected_positions, sampled_indices, output_dir
    )

if __name__ == "__main__":
    main()