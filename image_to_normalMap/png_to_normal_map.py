import os
import argparse
import subprocess
import numpy as np
from PIL import Image


def create_video_from_image(image_path, output_path, duration=1, fps=30):
    """
    Create a video from a single image using ffmpeg
    """
    # Check if the image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Create video using ffmpeg
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-loop', '1',  # Loop the input
        '-i', image_path,  # Input file
        '-c:v', 'libx264',  # Codec
        '-t', str(duration),  # Duration in seconds
        '-pix_fmt', 'yuv420p',  # Pixel format
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Ensure dimensions are even (required by some codecs)
        '-r', str(fps),  # Frame rate
        output_path  # Output file
    ]
    
    print(f"Creating video from image: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return output_path


def extract_normal_map_from_npz(npz_file, output_dir, frame_idx=0, save_npy=True):
    """
    Extract a single frame from the normal map NPZ file and save it as PNG and NPY
    """
    # Load the NPZ file
    data = np.load(npz_file)
    
    # Get the depth/normal data
    normal_data = data['depth']
    
    print(f"Normal map data shape: {normal_data.shape}")
    
    # Check if the frame index is valid
    if frame_idx >= normal_data.shape[0]:
        raise ValueError(f"Frame index {frame_idx} out of range (0-{normal_data.shape[0]-1})")
    
    # Extract the frame
    normal_frame = normal_data[frame_idx]
    
    # Convert from [-1, 1] to [0, 255] for visualization
    normal_vis = ((normal_frame + 1) / 2 * 255).astype(np.uint8)
    
    # Base filename without extension
    base_filename = os.path.join(output_dir, f"normal_map_frame_{frame_idx}")
    
    # Save as PNG for visualization
    png_path = base_filename + ".png"
    Image.fromarray(normal_vis).save(png_path)
    print(f"Saved normal map visualization to: {png_path}")
    
    # Save as NPY (raw data)
    if save_npy:
        npy_path = base_filename + ".npy"
        np.save(npy_path, normal_frame)
        print(f"Saved raw normal map data to: {npy_path}")
    
    return png_path, normal_frame


def main():
    parser = argparse.ArgumentParser(description='Convert a PNG to a normal map using NormalCrafter')
    parser.add_argument('--image', required=True, help='Path to input PNG image')
    parser.add_argument('--output-dir', default='./output', help='Directory to save output files')
    parser.add_argument('--video-duration', type=float, default=1.0, help='Duration of temporary video in seconds')
    parser.add_argument('--video-fps', type=int, default=30, help='Frame rate of temporary video')
    parser.add_argument('--frame-idx', type=int, default=0, help='Frame index to extract from the normal map sequence')
    parser.add_argument('--normalcrafter-args', default='', help='Additional arguments to pass to NormalCrafter')
    parser.add_argument('--save-full-sequence', action='store_true', help='Save the full normal map sequence as a single NPY file')
    parser.add_argument('--save-all-frames', action='store_true', help='Save all frames as individual PNG and NPY files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a temporary video file path
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    temp_video_path = os.path.join(args.output_dir, f'{base_name}_temp.mp4')
    
    # Convert PNG to MP4
    print(f"Converting {args.image} to temporary video...")
    create_video_from_image(
        args.image, 
        temp_video_path, 
        duration=args.video_duration, 
        fps=args.video_fps
    )
    
    # Run NormalCrafter on the video
    print(f"Running NormalCrafter on the temporary video...")
    normalcrafter_cmd = f"python run.py --video-path {temp_video_path} --save-folder {args.output_dir} --save-npz True {args.normalcrafter_args}"
    print(f"Executing: {normalcrafter_cmd}")
    subprocess.run(normalcrafter_cmd, shell=True, check=True)
    
    # Find the NPZ file
    npz_file = os.path.join(args.output_dir, f'{base_name}_temp.npz')
    if not os.path.exists(npz_file):
        print(f"Warning: NPZ file not found at {npz_file}")
        # Try to find it with another name
        npz_files = [f for f in os.listdir(args.output_dir) if f.endswith('.npz')]
        if npz_files:
            npz_file = os.path.join(args.output_dir, npz_files[0])
            print(f"Found NPZ file: {npz_file}")
        else:
            raise FileNotFoundError("No NPZ files found in output directory")
    
    # Extract the normal map from the NPZ file
    png_path, normal_frame = extract_normal_map_from_npz(npz_file, args.output_dir, args.frame_idx)
    
    # Save the full sequence as NPY if requested
    if args.save_full_sequence:
        # Load the full data
        data = np.load(npz_file)
        normal_data = data['depth']
        
        # Save as NPY
        full_seq_path = os.path.join(args.output_dir, f"{base_name}_full_sequence.npy")
        np.save(full_seq_path, normal_data)
        print(f"Saved full normal map sequence to: {full_seq_path}")
    
    # Save all frames if requested
    if args.save_all_frames:
        data = np.load(npz_file)
        normal_data = data['depth']
        
        print(f"Saving all {normal_data.shape[0]} frames...")
        for i in range(normal_data.shape[0]):
            frame = normal_data[i]
            
            # Convert from [-1, 1] to [0, 255] for visualization
            normal_vis = ((frame + 1) / 2 * 255).astype(np.uint8)
            
            # Save PNG
            png_path = os.path.join(args.output_dir, f"{base_name}_frame_{i:03d}.png")
            Image.fromarray(normal_vis).save(png_path)
            
            # Save NPY
            npy_path = os.path.join(args.output_dir, f"{base_name}_frame_{i:03d}.npy")
            np.save(npy_path, frame)
            
        print(f"Saved all frames to {args.output_dir}")
    
    print(f"\nProcess completed:")
    print(f"- Original image: {args.image}")
    print(f"- Normal map visualization: {png_path}")
    print(f"- Normal map shape: {normal_frame.shape}")
    
    # Cleanup temporary files (optional)
    # os.remove(temp_video_path)
    # os.remove(npz_file)


if __name__ == "__main__":
    main()