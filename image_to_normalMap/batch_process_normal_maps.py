#!/usr/bin/env python3
import os
import argparse
import subprocess
import time
import glob
from tqdm import tqdm
import concurrent.futures

def process_image(args):
    """Process a single image using the png_to_normal_map.py script"""
    image_path, output_dir, normalcrafter_args, frame_idx = args
    
    # Create the command
    cmd = [
        "python", "png_to_normal_map.py",
        "--image", image_path,
        "--output-dir", output_dir,
        "--frame-idx", str(frame_idx)
    ]
    
    if normalcrafter_args:
        cmd.extend(["--normalcrafter-args", normalcrafter_args])
    
    # Get base filename for logging
    base_name = os.path.basename(image_path)
    
    try:
        # Run the command
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return f"✓ Successfully processed {base_name}"
    except subprocess.CalledProcessError as e:
        return f"✗ Error processing {base_name}: {e}"


def main():
    parser = argparse.ArgumentParser(description='Batch process multiple PNG images to normal maps')
    parser.add_argument('--input-dir', required=True, help='Directory containing PNG images')
    parser.add_argument('--output-dir', default='./normal_maps', help='Directory to save output files')
    parser.add_argument('--pattern', default='*.png', help='Glob pattern to match image files')
    parser.add_argument('--normalcrafter-args', default='', help='Additional arguments for NormalCrafter')
    parser.add_argument('--frame-idx', type=int, default=15, help='Frame index to extract (middle frame recommended)')
    parser.add_argument('--parallel', type=int, default=1, help='Number of images to process in parallel (use with caution)')
    parser.add_argument('--dry-run', action='store_true', help='List files to be processed without processing them')
    
    args = parser.parse_args()
    
    # Find all PNG files in the input directory
    image_paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    
    if not image_paths:
        print(f"No files matching '{args.pattern}' found in {args.input_dir}")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    if args.dry_run:
        print("\nDry run mode - would process these files:")
        for path in image_paths:
            print(f"  {path}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process files
    start_time = time.time()
    
    # Create arguments for each file
    process_args = [(path, args.output_dir, args.normalcrafter_args, args.frame_idx) for path in image_paths]
    
    if args.parallel > 1:
        print(f"Processing {len(image_paths)} images in parallel with {args.parallel} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel) as executor:
            results = list(tqdm(executor.map(process_image, process_args), total=len(image_paths)))
        
        # Count successes and failures
        successes = sum(1 for result in results if result.startswith("✓"))
        failures = sum(1 for result in results if result.startswith("✗"))
        
        print(f"\nProcessed {len(image_paths)} images: {successes} succeeded, {failures} failed")
        
        # Print failures if any
        if failures > 0:
            print("\nFailed images:")
            for result in results:
                if result.startswith("✗"):
                    print(f"  {result}")
    else:
        print(f"Processing {len(image_paths)} images sequentially...")
        for i, args_tuple in enumerate(process_args):
            print(f"[{i+1}/{len(image_paths)}] Processing {os.path.basename(args_tuple[0])}...")
            result = process_image(args_tuple)
            print(f"  {result}")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds ({elapsed_time/len(image_paths):.2f} seconds per image)")


if __name__ == "__main__":
    main()
