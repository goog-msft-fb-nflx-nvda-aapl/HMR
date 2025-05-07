#!/bin/bash

# Configuration
INPUT_DIR="/home/iismtl519-2/Desktop/shapy/datasets/ssp_3d/images/"
OUTPUT_DIR="./normal_maps"
MAX_RES=512
FRAME_IDX=15
PARALLEL=1  # Set to higher value if you have multiple GPUs

# Create output directory
mkdir -p $OUTPUT_DIR

# First run a dry-run to check all files
echo "Dry run to check files..."
python batch_process_normal_maps.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --dry-run

# Ask user if they want to continue
read -p "Found files to process. Continue? (y/n): " choice
if [[ "$choice" != "y" && "$choice" != "Y" ]]; then
  echo "Aborted."
  exit 0
fi

# Run the batch processing
echo "Starting batch processing of normal maps..."
python batch_process_normal_maps.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --normalcrafter-args "--max-res $MAX_RES" \
  --frame-idx $FRAME_IDX \
  --parallel $PARALLEL

echo "Processing complete!"

# Show a summary of the results
echo "Results summary:"
echo "================"
total=$(ls "$INPUT_DIR" | wc -l)
processed=$(ls "$OUTPUT_DIR"/*_normal_map_frame_*.png 2>/dev/null | wc -l)
echo "Total images: $total"
echo "Processed images: $processed"
echo "Success rate: $(( processed * 100 / total ))%"

# Check if any files failed
if [ $processed -lt $total ]; then
  echo "Some files may have failed. Check the logs above for details."
else
  echo "All files were processed successfully!"
fi
