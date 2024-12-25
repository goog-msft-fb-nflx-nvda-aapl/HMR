import os
import glob
import numpy as np
import subprocess
import argparse
import csv
from pathlib import Path

def npy_to_npz(input_dir, output_base_dir):
    """Convert .npy files to .npz format with required structure."""
    # Create output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Get list of all .npy files
    npy_files = glob.glob(os.path.join(input_dir, "*.npy"))
    
    processed_dirs = []
    for npy_file in npy_files:
        base_name = os.path.splitext(os.path.basename(npy_file))[0]
        output_dir = os.path.join(output_base_dir, base_name[19:].lower())
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and process data
        betas = np.load(npy_file)
        if betas.ndim != 2 or betas.shape[1] != 10:
            raise ValueError(f"File {npy_file} has incorrect shape. Expected (number_of_data, 10), got {betas.shape}")
        
        number_of_data = betas.shape[0]
        poses = np.zeros((number_of_data, 72), dtype=np.float64)
        trans = np.zeros((number_of_data, 3), dtype=np.float32)
        
        # Save as .npz
        npz_path = os.path.join(output_dir, f"{base_name}.npz")

        np.savez(npz_path, betas=betas, poses=poses, trans=trans)
        processed_dirs.append(output_dir)
        
    return processed_dirs

def run_bomoto(input_dirs, output_base_dir, cfg_path, batch_size=705):
    """Run bomoto conversion for each directory."""
    output_dirs = []
    for input_dir in input_dirs:
        dir_name = os.path.basename(input_dir)
        output_dir = os.path.join(output_base_dir, dir_name)
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = [
            "python", "/home/iismtl519-2/Desktop/bomoto/run.py",
            "--cfg", cfg_path,
            "--npz_files_dir", input_dir,
            "--save_dir", output_dir,
            "--batch_size", str(batch_size)
        ]
        
        subprocess.run(cmd, check=True)
        output_dirs.append(output_dir)
    
    return output_dirs

def load_obj_vertices(file_path):
    """Load vertices from an .obj file."""
    vertices = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append([float(v) for v in line.strip().split()[1:]])
    return np.array(vertices, dtype=np.float32)

def process_obj_files(obj_directory, image_names_npz):
    """Process .obj files into evaluation-compatible .npz format."""
    output_dir = os.path.join(obj_directory, "meshes")
    os.makedirs(output_dir, exist_ok=True)
    output_npz_path = os.path.join(output_dir, "smplxMeshes.npz")
    
    # Load image names
    input_data = np.load(image_names_npz)
    image_names = input_data["imgname"]
    
    # Process obj files
    num_objs = 705
    v_shaped = np.zeros((num_objs, 10475, 3), dtype=np.float32)
    
    for i in range(num_objs):
        obj_file = os.path.join(obj_directory,"meshes/batch_0",  f"{i:06d}.obj")
        vertices = load_obj_vertices(obj_file)
        if vertices.shape != (10475, 3):
            raise ValueError(f"File {obj_file} has incorrect vertex shape {vertices.shape}")
        v_shaped[i] = vertices
    
    np.savez(output_npz_path, image_name=image_names, v_shaped=v_shaped)
    return output_npz_path

def run_evaluation(npz_files, hbw_folder, evaluation_script):
    """Run HBW evaluation for each .npz file."""
    for npz_file in npz_files:
        dir_name = os.path.basename(os.path.dirname(os.path.dirname(npz_file)))
        output_file = f"evaluation_summary_{dir_name}.txt"
        
        cmd = [
            "python", evaluation_script,
            "--input-npz-file", npz_file,
            "--model-type", "smplx",
            "--hbw-folder", hbw_folder
        ]
        
        with open(output_file, 'w') as f:
            result = subprocess.run(cmd, capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')[-7:]
            f.write('\n'.join(lines))

def create_summary_csv(suffix_):
    """Create final CSV summary from evaluation files with custom row ordering."""
    files = glob.glob("evaluation_summary_*.txt")
    headers = [
        'Suffix', 'V2V Error (mm)', 'P2P-20k Error (mm)',
        'height Error (mm)', 'chest Error (mm)', 'waist Error (mm)',
        'hips Error (mm)', 'mass Error (kg)'
    ]
    
    # Define custom order
    custom_order = [
        'attr_meas_kp',
        'attr_meas',
        'attr_kp',
        'meas_kp',
        'kp',
        'meas',
        'attr'
    ]
    
    data = {}
    for file_path in files:
        suffix = file_path.split('evaluation_summary_')[1].replace('.txt', '')
        measurements = {}
        with open(file_path, 'r') as f:
            for line in f.readlines()[:7]:
                name, value = line.strip().split(': ')
                numeric_value = value.split()[0]
                measurements[name] = numeric_value
        data[suffix] = measurements
    
    with open("evaluation_summary"+suffix_+".csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        measurement_map = {
            'V2V Error': 'V2V Error (mm)',
            'P2P-20k Error': 'P2P-20k Error (mm)',
            'height Error': 'height Error (mm)',
            'chest Error': 'chest Error (mm)',
            'waist Error': 'waist Error (mm)',
            'hips Error': 'hips Error (mm)',
            'mass Error': 'mass Error (kg)'
        }
        
        # Use custom order instead of sorted()
        for suffix in custom_order:
            if suffix in data:  # Only write row if data exists for that suffix
                row = [suffix]
                for old_header, new_header in measurement_map.items():
                    row.append(data[suffix].get(old_header, ''))
                writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description="End-to-end SMPL to SMPLX evaluation pipeline")
    parser.add_argument("--input-dir", required=True, help="Directory containing the original SMPL .npy files")
    parser.add_argument("--bomoto-cfg", default="examples/smpl2smplx/cfg.yaml", 
                      help="Path to bomoto config file")
    parser.add_argument("--hbw-folder", default="/home/iismtl519-2/Desktop/shapy/datasets/HBW",
                      help="Path to HBW dataset folder")
    parser.add_argument("--batch-size", type=int, default=705,
                      help="Batch size for bomoto processing")
    parser.add_argument("--image-names-npz", 
                      default="/home/iismtl519-2/Desktop/ScoreHMR/data/datasets/hbw_val_v4.npz",
                      help="Path to NPZ file containing image names")
    parser.add_argument("--evaluation-script",
                      default="/home/iismtl519-2/Desktop/shapy/regressor/hbw_evaluation/evaluate_hbw.py",
                      help="Path to HBW evaluation script")
    
    args = parser.parse_args()
    
    # Create necessary directories
    work_dir = os.path.dirname(args.input_dir)
    smpl_npz_dir = os.path.join(work_dir, "smpl")
    smplx_dir = os.path.join(work_dir, "smplx")

    print("Step 1: Converting NPY to NPZ...")
    processed_dirs = npy_to_npz(args.input_dir, smpl_npz_dir)
    
    print("Step 2: Running bomoto conversion...")
    smplx_dirs = run_bomoto(processed_dirs, smplx_dir, args.bomoto_cfg, args.batch_size)


    print("Step 3: Processing OBJ files...")
    npz_files = []
    for smplx_dir in smplx_dirs:
        npz_path = process_obj_files(smplx_dir, args.image_names_npz)
        npz_files.append(npz_path)

    print("Step 4: Running HBW evaluation...")
    run_evaluation(npz_files, args.hbw_folder, args.evaluation_script)
    
    print("Step 5: Creating summary CSV...")
    create_summary_csv(args.input_dir.split("/")[-2])
    
    print("Pipeline complete! Results saved in evaluation_summary.csv")

if __name__ == "__main__":
    main()