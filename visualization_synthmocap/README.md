# SMPL Vertex Visualization Tool

## Overview
This Python script is designed to process SMPL (Skinned Multi-Person Linear model) human body model data and generate visualizations of both 2D and 3D vertices. The tool takes a dataset of synthetic human images with corresponding SMPL parameters and creates three visualization views for each image:

1. The original image
2. Projected SMPL vertices on a white background
3. Projected vertices overlaid on the original image

## Features
- Automatically downloads the SMPL-H model if not present
- Processes large batches of synthetic human data
- Projects 3D vertices to 2D using camera parameters
- Saves both 2D and 3D vertex positions as NumPy arrays
- Creates visual representations of projected vertices
- Maintains consistent vertex sampling across all images for comparability

## Requirements
- Python 3.7+
- NumPy
- OpenCV (cv2)
- Matplotlib
- PyRender
- Trimesh
- tqdm
- SMPL-H model credentials (will be requested if model needs downloading)

## Installation
1. Clone the repository
2. Install the required dependencies:
   ```
   pip install numpy opencv-python matplotlib pyrender trimesh tqdm
   ```
3. Run the script. If the SMPL-H model is not found, you'll be prompted to enter your credentials for https://mano.is.tue.mpg.de/

## Usage
```
python smpl_vertex_visualization.py
```

The script uses hardcoded paths by default:
- Input data directory: `/home/iismtl519-2/Desktop/SynthMoCap/synth_body`
- Output 3D vertices: `/home/iismtl519-2/Desktop/smplvertices3d_synthmocap_v2.npy`
- Output 2D vertices: `/home/iismtl519-2/Desktop/smplvertices2d_synthmocap_v2.npy`

To change these paths, modify the variables at the beginning of the `main()` function.

## Data Directory Structure
The script expects the following structure in the data directory:
```
data_dir/
├── img/
│   ├── img_0000000_000.jpg
│   ├── img_0000000_001.jpg
│   └── ...
├── metadata/
│   ├── metadata_0000000_000.json
│   ├── metadata_0000000_001.json
│   └── ...
└── visualizations/  (created by the script)
    ├── visualization_0000000_000.png
    ├── visualization_0000000_001.png
    └── ...
```

## Output
The script generates three types of output:
1. A 2D NumPy array containing the 2D projected vertex positions for all images
2. A 3D NumPy array containing the 3D vertex positions for all images
3. Visualization images with three panels showing the original image, projected vertices, and overlay

## Functions
- `_download_smplh()`: Downloads the SMPL-H model if not present
- `_get_smplh()`: Initializes and returns the SMPL model
- `project_vertices()`: Projects 3D vertices to 2D using camera parameters
- `build_index_mapping()`: Creates mappings between sequential and (sidx, fidx) indices
- `generate_projections()`: Creates and saves 2D and 3D vertex projections
- `visualize_projected_vertices_on_image()`: Creates the three-panel visualizations
- `main()`: Orchestrates the overall processing flow

## Notes
- The script randomly selects up to 100 images for visualization
- The first three vertices are highlighted in red for reference
- If the projection files already exist, they will be loaded instead of regenerated