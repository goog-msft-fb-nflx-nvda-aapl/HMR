# Body Measurement Calculator

This tool calculates body measurements (height, chest, waist, hips, and mass) from SMPL 3D mesh vertices. The measurement calculations are adapted from [SHAPY](https://github.com/muelea/shapy/blob/master/measurements/virtual_measurements.py). The SMPL-X meshes used as input are generated from SMPL parameters using [BOMOTO](https://github.com/PerceivingSystems/bomoto).

## Features

- Batch processing of multiple .obj files
- CUDA-accelerated computation
- Supports SMPL-X model
- Outputs five key measurements: height, chest, waist, hips, and mass
- Handles multiple meshes in parallel

## Prerequisites

- Python 3.x
- CUDA-capable GPU
- PyTorch with CUDA support

## Dependencies

```bash
pip install torch
pip install trimesh
pip install smplx
pip install numpy
```

Additionally, you'll need:
- `body_measurements` package from the SHAPY repository
- [BOMOTO](https://github.com/PerceivingSystems/bomoto) for generating SMPL-X meshes from SMPL parameters

## Directory Structure

The code expects the following directory structure:

```
output_fileshapy/
├── data/
│   ├── body_models/              # SMPL model files
│   └── utility_files/
│       └── measurements/
│           ├── measurement_defitions.yaml
│           └── smplx_measurements.yaml
```

## Usage

1. First, generate SMPL-X meshes from your SMPL parameters using BOMOTO
2. Update the paths in the script to match your directory structure:

```python
input_dir = "PATH_TO_YOUR_INPUT_SMPL_VERTICES"  # Directory containing .obj files
output_file = "PATH_TO_YOUR_OUTPUT_NPY"         # Path for output .npy file
```

3. Run the script:

```python
python measure_body.py
```

## Input Format

- Input directory should contain .obj files generated using BOMOTO
- Each .obj file should have exactly 10,475 vertices
- Vertices should be in SMPL-X format

## Output Format

The script generates a NumPy array file (.npy) containing measurements for each input mesh:
- Shape: (N, 5) where N is the number of processed meshes
- Measurements order: [height, chest, waist, hips, mass]

## Pipeline

1. SMPL parameters → BOMOTO → SMPL-X meshes (.obj files)
2. SMPL-X meshes → This tool → Body measurements

## Credits

- Measurement definitions and calculations are adopted from the [SHAPY project](https://github.com/muelea/shapy/blob/master/measurements/virtual_measurements.py)
- SMPL-X mesh generation is performed using [BOMOTO](https://github.com/PerceivingSystems/bomoto)

## Troubleshooting

If you encounter CUDA errors:
1. Verify CUDA is properly installed
2. Check GPU memory availability
3. Ensure PyTorch is built with CUDA support
4. Try processing fewer meshes at once if running out of memory

## License

Please refer to the SHAPY project's license for terms regarding the measurement calculations and BOMOTO's license for mesh generation terms.