import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def validate_features(feature_dir):
    """
    Validate the extracted features for correctness and quality.
    
    Checks:
    1. Dimension consistency
    2. Value distribution
    3. Duplicate detection
    4. Basic statistical analysis
    """
    print("Starting feature validation...")
    
    # Initialize statistics
    dimensions = set()
    value_ranges = []
    means = []
    stds = []
    zero_counts = []
    duplicate_check = defaultdict(list)  # Store feature hashes for duplicate detection
    
    # Get all feature files
    feature_files = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]
    total_files = len(feature_files)
    print(f"Found {total_files} feature files")
    
    # Process each file
    for file in tqdm(feature_files, desc="Validating features"):
        try:
            feature_path = os.path.join(feature_dir, file)
            feature = np.load(feature_path)
            
            # 1. Check dimensions
            dimensions.add(feature.shape)
            
            # 2. Check value distribution
            value_ranges.append((np.min(feature), np.max(feature)))
            means.append(np.mean(feature))
            stds.append(np.std(feature))
            zero_counts.append(np.sum(feature == 0))
            
            # 3. Check for exact duplicates using feature hash
            feature_hash = hash(feature.tobytes())
            duplicate_check[feature_hash].append(file)
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    # Report findings
    print("\n=== Validation Results ===")
    
    # 1. Dimension Analysis
    print("\nDimension Analysis:")
    for dim in dimensions:
        print(f"Found dimension: {dim}")
    if len(dimensions) > 1:
        print("WARNING: Inconsistent dimensions detected!")
    
    # 2. Value Distribution Analysis
    ranges = np.array(value_ranges)
    means = np.array(means)
    stds = np.array(stds)
    
    print("\nValue Distribution Analysis:")
    print(f"Global min: {np.min(ranges[:, 0]):.4f}")
    print(f"Global max: {np.max(ranges[:, 1]):.4f}")
    print(f"Mean of means: {np.mean(means):.4f} ± {np.std(means):.4f}")
    print(f"Mean of stds: {np.mean(stds):.4f} ± {np.std(stds):.4f}")
    
    # 3. Zero Value Analysis
    zero_counts = np.array(zero_counts)
    print("\nZero Value Analysis:")
    print(f"Average zero values per feature: {np.mean(zero_counts):.2f} ± {np.std(zero_counts):.2f}")
    
    # 4. Duplicate Analysis
    print("\nDuplicate Analysis:")
    duplicates = {k: v for k, v in duplicate_check.items() if len(v) > 1}
    if duplicates:
        print(f"Found {len(duplicates)} sets of identical features:")
        for hash_val, files in duplicates.items():
            if len(files) > 5:
                print(f"- {len(files)} files share identical features (showing first 5): {files[:5]}")
            else:
                print(f"- Files with identical features: {files}")
    else:
        print("No exact duplicates found")
    
    # 5. Feature Correlation Analysis (on a subset if there are many features)
    print("\nFeature Correlation Analysis:")
    sample_size = min(1000, total_files)
    sample_indices = np.random.choice(total_files, sample_size, replace=False)
    
    sample_features = []
    for idx in sample_indices:
        feature_path = os.path.join(feature_dir, feature_files[idx])
        sample_features.append(np.load(feature_path))
    
    sample_features = np.stack(sample_features)
    correlation_matrix = np.corrcoef(sample_features)
    
    print(f"Average correlation between features: {np.mean(np.abs(correlation_matrix)):.4f}")
    print(f"Max correlation between different features: {np.max(np.abs(correlation_matrix - np.eye(sample_size))):.4f}")

if __name__ == "__main__":
    feature_dir = "./features"  # Update this path
    validate_features(feature_dir)