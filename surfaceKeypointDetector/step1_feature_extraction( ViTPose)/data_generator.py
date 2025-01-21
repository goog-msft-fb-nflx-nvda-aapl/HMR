DATASET_FILE = "/home/iismtl519-2/Desktop/SynthMoCap/synthmocap.npz"
PARE_PREDS_FILE = "/home/iismtl519-2/Desktop/ScoreHMR/cache/pare/synthmocap.npz"
import numpy as np

import numpy as np

def transform_vertices(vertices, bbox_xyxy):
    """
    Transform vertex positions through cropping, squaring, and resizing steps.
    Returns transformed vertices with confidence values.
    
    Args:
        vertices: Original vertex positions (N, 6890, 2)
        bbox_xyxy: Bounding boxes in (x1,y1,x2,y2) format (N, 4)
    
    Returns:
        transformed_vertices: (N, 6890, 3) where last dim is (x, y, confidence)
    """
    N = vertices.shape[0]
    num_vertices = vertices.shape[1]
    
    # Initialize output array with transformed vertices and confidence
    transformed_vertices = np.zeros((N, num_vertices, 3))
    
    # Track statistics
    total_inside_vertices = 0
    total_vertices = N * num_vertices
    
    for i in range(N):
        # Get current bounding box
        x1, y1, x2, y2 = bbox_xyxy[i]
        
        # Step 1: Transform to cropped coordinates
        # Subtract top-left corner coordinates
        vertices_cropped = vertices[i].copy()
        vertices_cropped[:, 0] = vertices_cropped[:, 0] - x1
        vertices_cropped[:, 1] = vertices_cropped[:, 1] - y1
        
        # Calculate crop dimensions
        crop_width = x2 - x1
        crop_height = y2 - y1
        
        # Step 2: Transform to squared coordinates
        # Calculate padding needed to make it square
        max_dim = max(crop_width, crop_height)
        pad_w = (max_dim - crop_width) / 2
        pad_h = (max_dim - crop_height) / 2
        
        # Add padding to coordinates
        vertices_squared = vertices_cropped.copy()
        vertices_squared[:, 0] = vertices_squared[:, 0] + pad_w
        vertices_squared[:, 1] = vertices_squared[:, 1] + pad_h
        
        # Step 3: Transform to 256x256
        scale = 256.0 / max_dim
        vertices_256 = vertices_squared * scale
        
        # Calculate confidence values
        # A vertex is inside if its original coordinates are within the bounding box
        confidence = np.logical_and.reduce([
            vertices[i, :, 0] >= x1,
            vertices[i, :, 0] <= x2,
            vertices[i, :, 1] >= y1,
            vertices[i, :, 1] <= y2
        ]).astype(float)
        
        # Update statistics
        total_inside_vertices += np.sum(confidence)
        
        # Store transformed coordinates and confidence
        transformed_vertices[i, :, 0:2] = vertices_256
        transformed_vertices[i, :, 2] = confidence
        
        # Print progress every 10% of the data
        if (i + 1) % (N // 10) == 0:
            print(f"Processed {(i + 1) / N * 100:.1f}% of the data")
    
    # Calculate and print statistics
    percentage_inside = (total_inside_vertices / total_vertices) * 100
    print(f"\nValidation Statistics:")
    print(f"Total vertices processed: {total_vertices:,}")
    print(f"Vertices inside bounding boxes: {total_inside_vertices:,}")
    print(f"Percentage inside: {percentage_inside:.2f}%")
    
    # Warning if percentage seems low
    if percentage_inside < 90:
        print("\nWARNING: Less than 90% of vertices are within bounding boxes.")
        print("This might indicate an issue with the bounding box calculations.")
    """
    Validation Statistics:
    Total vertices processed: 658,511,750
    Vertices inside bounding boxes: 657,332,993.0
    Percentage inside: 99.82%
    """
    return transformed_vertices

# Load the original data
with np.load(DATASET_FILE) as data:
    vertices = data['vertices']
    bbox_xyxy = data['bbox_xyxy']
    
    print(f"Processing dataset with {len(vertices)} images...")
    
    # Transform vertices
    vertices_256 = transform_vertices(vertices, bbox_xyxy)
    
    # Save the transformed vertices to a new NPZ file
    output_file = DATASET_FILE.replace('.npz', '_with_vertices256.npz')
    np.savez(
        output_file,
        vertices_256=vertices_256,
        **{key: data[key] for key in data.files}  # Include original data
    )

print(f"\nTransformed vertices saved with shape: {vertices_256.shape}")
print(f"Output saved to: {output_file}")