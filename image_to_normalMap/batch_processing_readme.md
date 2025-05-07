# Batch Processing Normal Maps

This guide explains how to process multiple PNG images to generate normal maps using the NormalCrafter model.

## Setup

1. Make sure you have the following files in your working directory:
   - `run.py` (original NormalCrafter script)
   - `png_to_normal_map.py` (single image processor)
   - `batch_process_normal_maps.py` (batch processor)

2. Install required dependencies:
   ```bash
   pip install tqdm
   ```

## Basic Usage

To process all PNG images in a directory:

```bash
python batch_process_normal_maps.py --input-dir /home/iismtl519-2/Desktop/shapy/datasets/ssp_3d/images/ --output-dir ./normal_maps
```

This will:
1. Find all PNG images in the input directory
2. Process each image to generate a normal map
3. Save the normal maps and visualizations in the output directory

## Command-Line Options

```bash
python batch_process_normal_maps.py \
  --input-dir /path/to/images \
  --output-dir ./normal_maps \
  --pattern "*.png" \
  --normalcrafter-args "--max-res 512" \
  --frame-idx 15 \
  --parallel 1
```

### Parameters

| Option | Description | Default |
|--------|-------------|---------|
| `--input-dir` | Directory containing PNG images | (Required) |
| `--output-dir` | Directory to save output files | `./normal_maps` |
| `--pattern` | Glob pattern to match image files | `*.png` |
| `--normalcrafter-args` | Additional arguments for NormalCrafter | `""` |
| `--frame-idx` | Frame index to extract from each sequence | `15` |
| `--parallel` | Number of images to process in parallel | `1` |
| `--dry-run` | List files without processing them | False |

## Processing Options

### Sequential Processing (Default)

By default, the script processes one image at a time. This is memory-efficient but slower:

```bash
python batch_process_normal_maps.py --input-dir /path/to/images
```

### Parallel Processing

For faster processing on powerful systems with multiple GPUs or large RAM:

```bash
python batch_process_normal_maps.py --input-dir /path/to/images --parallel 4
```

**Warning:** Parallel processing requires more memory. Start with a low number and increase if your system can handle it.

### Memory Efficiency

To reduce memory usage (for processing many large images):

```bash
python batch_process_normal_maps.py --input-dir /path/to/images --normalcrafter-args "--max-res 512 --cpu-offload sequential"
```

## Output Files

For each input image `example.png`, the script creates:
- `normal_map_frame_15.png`: Visualization of the normal map
- `normal_map_frame_15.npy`: Raw normal map data (in [-1,1] range)

## Common Use Cases

### Process All PNGs in a Directory

```bash
python batch_process_normal_maps.py --input-dir /path/to/images
```

### Reduce Resolution for Memory Constraints

```bash
python batch_process_normal_maps.py --input-dir /path/to/images --normalcrafter-args "--max-res 512"
```

### Process Only a Subset of Images

```bash
python batch_process_normal_maps.py --input-dir /path/to/images --pattern "person_*.png"
```

### Dry Run (Just List Files)

```bash
python batch_process_normal_maps.py --input-dir /path/to/images --dry-run
```

## Troubleshooting

1. **Out of memory errors**: Reduce `--max-res` or use `--cpu-offload sequential` in the normalcrafter args
2. **Slow processing**: Try parallel processing if your system has multiple GPUs
3. **Failed conversions**: Check individual errors in the output log

## Example For Your Specific Case (311 images)

```bash
# First test with a dry run
python batch_process_normal_maps.py \
  --input-dir /home/iismtl519-2/Desktop/shapy/datasets/ssp_3d/images/ \
  --output-dir ./normal_maps \
  --dry-run

# Then process sequentially (safe option)
python batch_process_normal_maps.py \
  --input-dir /home/iismtl519-2/Desktop/shapy/datasets/ssp_3d/images/ \
  --output-dir ./normal_maps \
  --normalcrafter-args "--max-res 512"

# OR process with parallelism if you have a powerful system
python batch_process_normal_maps.py \
  --input-dir /home/iismtl519-2/Desktop/shapy/datasets/ssp_3d/images/ \
  --output-dir ./normal_maps \
  --normalcrafter-args "--max-res 512" \
  --parallel 2
```
