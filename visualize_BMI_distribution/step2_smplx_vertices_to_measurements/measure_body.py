import os
import trimesh
import torch
import smplx
from typing import Tuple
from body_measurements import BodyMeasurements
import numpy as np

def calculate_measurement(
    vertices: torch.Tensor,
    model_path: str = "output_fileshapy/data/body_models/",
    meas_definition_path: str = "output_fileshapy/data/utility_files/measurements/measurement_defitions.yaml",
    meas_vertices_path: str = "output_fileshapy/data/utility_files/measurements/smplx_measurements.yaml",
    key_order: Tuple[str, str, str, str, str] = ('height', 'chest', 'waist', 'hips', 'mass')
) -> torch.Tensor:
    # Check if CUDA is available and set device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move input vertices to the appropriate device
    vertices = vertices.to(device)
    dtype = vertices.dtype
    batch_size = vertices.shape[0]

    # Load SMPL model
    smpl = smplx.create(
        model_path=model_path,
        gender="neutral",
        num_betas=10,
        model_type='smplx',
        batch_size=batch_size,
    ).to(device)

    # Initialize BodyMeasurements
    body_measurements = BodyMeasurements(
        {'meas_definition_path': meas_definition_path,
         'meas_vertices_path': meas_vertices_path}
    ).to(device)

    shaped_vertices = vertices  # torch.Size([4, 10475, 3])

    # Get shaped triangles for the entire batch
    # Make sure faces tensor is on the same device
    faces_tensor = smpl.faces_tensor.to(device)
    shaped_triangles = shaped_vertices[:, faces_tensor]

    print(f"Shaped triangles device: {shaped_triangles.device}")
    print(f"Shaped triangles shape: {shaped_triangles.shape}")

    # Compute measurements for the entire batch
    measurements = body_measurements(shaped_triangles)['measurements']
    
    # Initialize tensor to store measurements
    body_meas = torch.zeros((batch_size, 5), device=device, dtype=dtype)

    # Extract measurements for each key in the specified order
    for i, key in enumerate(key_order):
        body_meas[:, i] = measurements[key]['tensor']

    return body_meas

def process_and_measure(input_dir: str, output_file: str):
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. This code requires GPU support.")
        return
        
    # Collect all .obj files
    obj_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".obj")
    ]

    if not obj_files:
        print("No .obj files found in the directory.")
        return

    # List to store all vertices
    all_vertices = []

    for obj_file in obj_files:
        mesh = trimesh.load(obj_file, process=False)
        if not hasattr(mesh, 'vertices'):
            print(f"Warning: No vertices found in {obj_file}. Skipping.")
            continue

        vertices = np.array(mesh.vertices)  # Shape: (10475, 3)
        if vertices.shape[0] != 10475:
            print(f"Warning: {obj_file} does not have 10475 vertices. Skipping.")
            continue

        all_vertices.append(vertices)

    if not all_vertices:
        print("No valid meshes were found.")
        return

    try:
        # Stack all vertices into a tensor and move to CUDA
        integrated_meshes = torch.tensor(np.stack(all_vertices), 
                                       dtype=torch.float32)

        # Call calculate_measurement
        measurements = calculate_measurement(
            vertices=integrated_meshes,
            model_path="output_fileshapy/data/body_models/",
            meas_definition_path="output_fileshapy/data/utility_files/measurements/measurement_defitions.yaml",
            meas_vertices_path="output_fileshapy/data/utility_files/measurements/smplx_measurements.yaml",
            key_order=('height', 'chest', 'waist', 'hips', 'mass')
        )

        # Move measurements back to CPU for numpy conversion
        measurements = measurements.cpu()

        # Save the measurements
        np.save(output_file, measurements.numpy())

        print(f"Measurements saved to {output_file}")
        
    except RuntimeError as e:
        print(f"CUDA Error: {str(e)}")
        print("Please make sure you have enough GPU memory and CUDA is properly set up.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage
if __name__ == "__main__":
    input_dir = PATH_TO_YOUR_INPUT_SMPL_VERTICES
    output_file = PATH_TO_YOUR_OUTPUT_NPY
    process_and_measure(input_dir, output_file)