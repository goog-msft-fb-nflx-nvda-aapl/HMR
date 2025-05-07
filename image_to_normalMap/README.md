# PNG to Normal Map Converter

This tool converts a single PNG image to a normal map using the NormalCrafter model. It works by first converting your image to a short video and then using the original NormalCrafter pipeline to process it.

## Requirements

- Python 3.7+
- ffmpeg
- NormalCrafter dependencies (PyTorch, diffusers, etc.)
- Original NormalCrafter `run.py` script in the same directory

## Installation

1. Make sure you have the original NormalCrafter repository set up with all its dependencies
2. Place the `png_to_normal_map.py` script in the same directory as the original `run.py`
3. Install ffmpeg if not already installed:
   - On Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - On macOS with Homebrew: `brew install ffmpeg`
   - On Windows: Download from https://ffmpeg.org/download.html

## Basic Usage

```bash
python png_to_normal_map.py --image your_image.png
```

This will:
1. Convert your PNG image to a short video
2. Process it with NormalCrafter
3. Extract and save the normal map as both PNG and NPY files

## Output Files

For an input image `example.png`, the default output will be:
- `normal_map_frame_0.png`: Visualization of the normal map
- `normal_map_frame_0.npy`: Raw normal map data (in [-1,1] range)

## Normal Map Format

- **Shape**: (H, W, 3) where H and W are the height and width of the image
- **Value Range**: [-1, 1]
- **Channels**: RGB corresponds to XYZ normal vector components
  - Red channel: X component (left to right)
  - Green channel: Y component (bottom to top)
  - Blue channel: Z component (depth)

## Advanced Options

```bash
python png_to_normal_map.py \
  --image your_image.png \
  --output-dir ./output \
  --video-duration 1.0 \
  --video-fps 30 \
  --frame-idx 15 \
  --save-full-sequence \
  --normalcrafter-args "--max-res 512"
```

### Parameters

| Option | Description | Default |
|--------|-------------|---------|
| `--image` | Path to input PNG image | (Required) |
| `--output-dir` | Directory to save output files | `./output` |
| `--video-duration` | Duration of temporary video in seconds | 1.0 |
| `--video-fps` | Frame rate of temporary video | 30 |
| `--frame-idx` | Frame index to extract from the sequence | 0 |
| `--save-full-sequence` | Save the entire normal map sequence as NPY | False |
| `--save-all-frames` | Save all frames as individual PNGs and NPYs | False |
| `--normalcrafter-args` | Additional arguments for NormalCrafter | `""` |

## Understanding the Output

### NPY vs NPZ Files

- **NPY**: A single NumPy array saved to disk
- **NPZ**: A zipped archive containing multiple NumPy arrays with keys

The original NormalCrafter outputs an NPZ file with a key called 'depth' that contains the normal map data. This script extracts that data and saves it as a simpler NPY file for easier loading.

### Working with the Normal Map

To load and use the normal map in your code:

```python
import numpy as np
from PIL import Image

# Load the raw normal map data (in [-1,1] range)
normal_map = np.load('normal_map_frame_0.npy')

# For visualization, convert to [0,255] range
normal_vis = ((normal_map + 1) / 2 * 255).astype(np.uint8)

# Save or display
Image.fromarray(normal_vis).show()
```

### Frame Selection

NormalCrafter processes the video and generates a sequence of normal maps. Since we're using a static image, all frames should be similar, but there may be subtle variations. By default, we extract frame 0, but you can specify any frame with `--frame-idx`.

## Troubleshooting

1. **CUDA out of memory**: Try reducing resolution with `--normalcrafter-args "--max-res 512"`
2. **ffmpeg not found**: Make sure ffmpeg is installed and in your PATH
3. **Import errors**: Ensure you're in the correct directory with NormalCrafter installed

## How It Works

1. The script uses ffmpeg to create a short video from your image (repeating the same frame)
2. It runs NormalCrafter's original `run.py` script on this video
3. NormalCrafter processes the video and produces a sequence of normal maps
4. The script extracts a single frame or the full sequence, as requested
5. The normal maps are saved in both visual (PNG) and data (NPY) formats

## Notes

- For best results with single images, the middle frame (frame_idx = total_frames/2) often works well
- The original normal map data is in [-1,1] range, while the PNG visualization is scaled to [0,255]
- Processing takes time and requires a GPU, similar to the original NormalCrafter